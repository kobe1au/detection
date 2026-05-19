#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate prediction complementarity among trained fusion-mode checkpoints.

This script answers a practical question:
if fusion does not outperform the best single modality, is it because the
modalities/models have little complementary signal, or because the fusion
module failed to exploit it?

Example:
    python -m fusion.complementarity \
      --base config/base.yaml \
      --split test \
      --model api=experiments/b_00_api_only/42/best_b_00_api_only.pt \
      --model graph=experiments/b_01_graph_only/42/best_b_01_graph_only.pt \
      --model concat=experiments/b_02_concat_erm/42/best_b_02_concat_erm.pt
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion.constants import TrainingConstants
from fusion.mm_dataset import MultiModalMalwareDataset, hierarchical_collate_fn
from fusion.model import MalwareModelWithXAttn
from fusion.train import (
    build_global_domain_years,
    deep_update,
    resolve_path,
    select_device,
    validate_full_config,
)
from fusion.utils import get_amp_context, prepare_batch


DEFAULT_MODELS = {
    "api": "experiments/b_00_api_only/42/best_b_00_api_only.pt",
    "graph": "experiments/b_01_graph_only/42/best_b_01_graph_only.pt",
    "concat": "experiments/b_02_concat_erm/42/best_b_02_concat_erm.pt",
    "cross_attention": "experiments/b_03_cross_attention/42/best_b_03_cross_attention.pt",
    "ours": "experiments/final_ours_2026/42/best_final_ours_2026.pt",
}


@dataclass
class ModelSpec:
    name: str
    ckpt_path: str
    cfg: dict[str, Any]


@dataclass
class PredictionPack:
    name: str
    fusion_mode: str
    rows: list[dict[str, Any]]

    @property
    def by_sid(self) -> dict[str, dict[str, Any]]:
        return {str(r["sid"]): r for r in self.rows}


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_model_arg(raw: str) -> tuple[str, str]:
    if "=" in raw:
        name, path = raw.split("=", 1)
    elif ":" in raw:
        name, path = raw.split(":", 1)
    else:
        p = Path(raw)
        name = p.stem
        path = raw
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid --model value: {raw!r}")
    return name, path


def resolve_existing_path(data_root: str, path: str) -> str:
    resolved = resolve_path(data_root, path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {path} -> {resolved}")
    return resolved


def load_checkpoint_config(base_cfg: dict[str, Any], ckpt_path: str) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        return validate_full_config(deep_update(base_cfg, ckpt_cfg))
    return validate_full_config(base_cfg)


def build_model_from_cfg(cfg: dict[str, Any], device: torch.device) -> MalwareModelWithXAttn:
    c_model = cfg["model"]
    c_loss = cfg["loss"]
    c_api = c_model["api_encoder"]
    c_graph = c_model["graph_encoder"]
    c_alignment = c_model["alignment"]
    c_gate = c_model["gate"]
    c_temporal = c_model["temporal"]
    fusion_mode = str(c_model["fusion_mode"])
    need_temporal_features = (
        float(c_loss["temporal_proto_current_weight"]) > 0.0
        or float(c_loss["temporal_proto_future_weight"]) > 0.0
        or float(c_loss["temporal_risk_calibration_weight"]) > 0.0
    )
    model = MalwareModelWithXAttn(
        num_classes=int(c_model["num_classes"]),
        api_emb_dim=TrainingConstants.API_EMB_DIM,
        graph_emb_dim=TrainingConstants.GRAPH_EMB_DIM,
        align_dim=TrainingConstants.ALIGN_DIM,
        max_nodes_gnn=int(c_model["max_nodes_gnn"]),
        max_xattn_nodes=int(c_model["max_xattn_nodes"]),
        in_feat_dim=int(c_model.get("in_feat_dim", TrainingConstants.IN_FEAT_DIM)),
        use_temporal_regularization=need_temporal_features,
        xattn_heads=TrainingConstants.XATTN_HEADS,
        fusion_mode=fusion_mode,
        graph_encoder_type=str(c_graph["type"]),
        graph_hidden=int(c_graph["hidden"]),
        graph_heads=int(c_graph["heads"]),
        graph_layers=int(c_graph["layers"]),
        graph_use_behavior_hint=bool(c_graph["use_behavior_hint"]),
        api_encoder_type=str(c_api["type"]),
        api_num_hash_buckets=int(c_api["num_hash_buckets"]),
        api_type_vocab_size=int(c_api["type_vocab_size"]),
        api_max_seq_len=int(c_api["max_seq_len"]),
        api_heads=int(c_api["heads"]),
        api_layers=int(c_api["layers"]),
        alignment_penalty_scale=float(c_alignment["penalty_scale"]),
        alignment_bonus_scale=float(c_alignment["bonus_scale"]),
        alignment_context_scale=float(c_alignment["context_scale"]),
        use_alignment_bias=bool(c_alignment["enabled"]),
        use_adaptive_alignment_bias=bool(c_alignment["adaptive_bias"]),
        use_alignment_drift_guidance=bool(c_alignment.get("drift_guided", True)),
        use_quality_gate_inputs=bool(c_gate["quality_inputs"]),
        use_drift_gate=bool(c_gate["drift_inputs"]),
        gate_mode=str(c_gate.get("mode", "learned")),
        gate_detach=bool(c_gate["detach"]),
        late_fusion_api_weight=0.5,
        temporal_num_domains=int(c_temporal.get("num_domains", 1)),
        temporal_prototype_momentum=float(c_temporal["prototype_momentum"]),
        temporal_prototype_clusters=int(c_temporal["prototype_clusters"]),
        temporal_drift_velocity_scale=float(c_loss["temporal_proto_velocity_scale"]),
        temporal_drift_min_history=int(c_loss["temporal_proto_min_history"]),
        use_future_temporal_drift=(
            bool(c_temporal.get("use_future_drift", True))
            and (
                float(c_loss.get("temporal_proto_future_weight", 0.0)) > 0.0
                or float(c_loss.get("temporal_risk_calibration_weight", 0.0)) > 0.0
            )
        ),
        use_temporal_risk_calibration=(float(c_loss.get("temporal_risk_calibration_weight", 0.0)) > 0.0),
    ).to(device)
    return model


def load_model(spec: ModelSpec, device: torch.device) -> MalwareModelWithXAttn:
    ckpt = torch.load(spec.ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    in_proj_key = next((k for k in state.keys() if k.endswith("graph_encoder.in_proj.weight")), None)
    if in_proj_key is not None and getattr(state[in_proj_key], "ndim", 0) == 2:
        spec.cfg.setdefault("model", {})["in_feat_dim"] = int(state[in_proj_key].shape[1])
    proto_key = next((k for k in state.keys() if k.endswith("temporal_prototype_memory.prototypes")), None)
    if proto_key is not None and getattr(state[proto_key], "ndim", 0) == 4:
        spec.cfg.setdefault("model", {}).setdefault("temporal", {})["num_domains"] = int(state[proto_key].shape[0])
        spec.cfg.setdefault("model", {}).setdefault("temporal", {})["prototype_clusters"] = int(state[proto_key].shape[2])
    elif proto_key is not None and getattr(state[proto_key], "ndim", 0) == 3:
        raise ValueError(
            f"{spec.name}: legacy 3D temporal prototype checkpoint is incompatible "
            "with the current year-label-cluster prototype model. Retrain this "
            "checkpoint from the current codebase."
        )
    model = build_model_from_cfg(spec.cfg, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {spec.name}: missing keys={len(missing)}")
    if unexpected:
        print(f"[warn] {spec.name}: unexpected keys={len(unexpected)}")
    model.eval()
    return model


def make_loader(
    cfg: dict[str, Any],
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    c_data = cfg["data"]
    c_train = cfg["train"]
    c_model = cfg["model"]
    c_alignment = c_model["alignment"]
    fusion_mode = str(c_model["fusion_mode"])
    need_alignment_mask = fusion_mode == "ours" and bool(c_alignment["enabled"])

    pt_key = f"{split}_pt_dir"
    csv_key = f"{split}_csv"
    if pt_key not in c_data or csv_key not in c_data:
        raise KeyError(f"Config has no data.{pt_key} / data.{csv_key}")

    domain_years = build_global_domain_years(
        resolve_path(data_root, c_data["train_csv"]),
        resolve_path(data_root, c_data["val_csv"]),
        resolve_path(data_root, c_data["test_csv"]),
    )

    ds = MultiModalMalwareDataset(
        pt_dir=resolve_path(data_root, c_data[pt_key]),
        csv_path=resolve_path(data_root, c_data[csv_key]),
        is_train=False,
        robust_aug=False,
        max_api_events_per_sample=c_data["max_api_events_per_sample"],
        fusion_mode=fusion_mode,
        need_alignment_mask=need_alignment_mask,
        domain_years=domain_years,
        drop_graph_behavior_hints=bool(c_model["graph_encoder"]["drop_extracted_behavior_hints"]),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=hierarchical_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def collect_predictions(
    spec: ModelSpec,
    data_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> PredictionPack:
    model = load_model(spec, device)
    fusion_mode = model.fusion_mode
    loader = make_loader(spec.cfg, data_root, split, batch_size, num_workers)

    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"Predict {spec.name} [{fusion_mode}]", dynamic_ncols=True):
        if batch is None:
            continue
        r = prepare_batch(
            batch,
            device,
            skip_graph=False,
            skip_masks=(fusion_mode != "ours" or not getattr(model, "use_alignment_bias", False)),
        )
        if r[2] is None:
            continue

        graph, masks, y, sids, explicit_info, _ = r
        qapi, qg, qa, papi, pg, tids = explicit_info
        with get_amp_context(device, enabled=use_amp):
            logits, extra = model(
                graph_data=graph,
                explicit_qs=(qapi, qg, qa, papi, pg),
                time_ids=tids,
                masks=masks,
            )
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

        probs_cpu = probs.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()
        conf_cpu = conf.detach().cpu().numpy()
        labels_cpu = y.detach().cpu().numpy()
        tids_cpu = tids.detach().cpu().numpy()
        for i, sid in enumerate(sids):
            rows.append({
                "sid": str(sid),
                "label": int(labels_cpu[i]),
                "pred": int(preds_cpu[i]),
                "correct": int(preds_cpu[i] == labels_cpu[i]),
                "conf": float(conf_cpu[i]),
                "prob_0": float(probs_cpu[i, 0]),
                "prob_1": float(probs_cpu[i, 1]) if probs_cpu.shape[1] > 1 else 0.0,
                "time_id": int(tids_cpu[i]),
            })
    return PredictionPack(name=spec.name, fusion_mode=fusion_mode, rows=rows)


def metric_summary(pack: PredictionPack) -> dict[str, Any]:
    y = np.array([r["label"] for r in pack.rows], dtype=np.int64)
    p = np.array([r["pred"] for r in pack.rows], dtype=np.int64)
    return {
        "name": pack.name,
        "fusion_mode": pack.fusion_mode,
        "n": int(len(pack.rows)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, p)) if len(y) else 0.0,
    }


def aligned_arrays(pack_a: PredictionPack, pack_b: PredictionPack):
    a_map = pack_a.by_sid
    b_map = pack_b.by_sid
    sids = sorted(set(a_map) & set(b_map))
    y = np.array([a_map[s]["label"] for s in sids], dtype=np.int64)
    pa = np.array([a_map[s]["pred"] for s in sids], dtype=np.int64)
    pb = np.array([b_map[s]["pred"] for s in sids], dtype=np.int64)
    proba = np.array([[a_map[s]["prob_0"], a_map[s]["prob_1"]] for s in sids], dtype=np.float64)
    probb = np.array([[b_map[s]["prob_0"], b_map[s]["prob_1"]] for s in sids], dtype=np.float64)
    return sids, y, pa, pb, proba, probb


def pairwise_complementarity(pack_a: PredictionPack, pack_b: PredictionPack) -> dict[str, Any]:
    sids, y, pa, pb, proba, probb = aligned_arrays(pack_a, pack_b)
    n = len(sids)
    if n == 0:
        return {"model_a": pack_a.name, "model_b": pack_b.name, "n": 0}

    ca = pa == y
    cb = pb == y
    both_correct = ca & cb
    only_a = ca & ~cb
    only_b = ~ca & cb
    both_wrong = ~ca & ~cb
    union_wrong = (~ca) | (~cb)
    inter_wrong = (~ca) & (~cb)

    oracle_pred = pa.copy()
    oracle_pred[only_b] = pb[only_b]
    oracle_acc = float((ca | cb).mean())
    oracle_f1 = float(f1_score(y, oracle_pred, average="macro", zero_division=0))

    best_w, best_f1, best_acc = 0.0, -1.0, 0.0
    for w in np.linspace(0.0, 1.0, 11):
        prob = w * proba + (1.0 - w) * probb
        pred = prob.argmax(axis=1)
        cur_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
        cur_acc = float(accuracy_score(y, pred))
        if cur_f1 > best_f1:
            best_w, best_f1, best_acc = float(w), cur_f1, cur_acc

    a_wrong = max(int((~ca).sum()), 1)
    b_wrong = max(int((~cb).sum()), 1)
    return {
        "model_a": pack_a.name,
        "mode_a": pack_a.fusion_mode,
        "model_b": pack_b.name,
        "mode_b": pack_b.fusion_mode,
        "n": int(n),
        "a_f1": float(f1_score(y, pa, average="macro", zero_division=0)),
        "b_f1": float(f1_score(y, pb, average="macro", zero_division=0)),
        "both_correct": int(both_correct.sum()),
        "only_a_correct": int(only_a.sum()),
        "only_b_correct": int(only_b.sum()),
        "both_wrong": int(both_wrong.sum()),
        "only_a_rate": float(only_a.mean()),
        "only_b_rate": float(only_b.mean()),
        "disagreement_rate": float((pa != pb).mean()),
        "error_jaccard": float(inter_wrong.sum() / max(union_wrong.sum(), 1)),
        "a_rescued_by_b_rate": float(only_b.sum() / a_wrong),
        "b_rescued_by_a_rate": float(only_a.sum() / b_wrong),
        "oracle_acc": oracle_acc,
        "oracle_f1": oracle_f1,
        "oracle_gain_over_best_f1": float(oracle_f1 - max(
            f1_score(y, pa, average="macro", zero_division=0),
            f1_score(y, pb, average="macro", zero_division=0),
        )),
        "best_prob_ensemble_f1": best_f1,
        "best_prob_ensemble_acc": best_acc,
        "best_weight_for_a": best_w,
    }


def multi_model_oracle(packs: list[PredictionPack]) -> dict[str, Any]:
    if not packs:
        return {}
    sid_sets = [set(p.by_sid) for p in packs]
    sids = sorted(set.intersection(*sid_sets))
    if not sids:
        return {"n": 0}

    maps = [p.by_sid for p in packs]
    y = np.array([maps[0][s]["label"] for s in sids], dtype=np.int64)
    pred_first = np.array([maps[0][s]["pred"] for s in sids], dtype=np.int64)
    oracle_pred = pred_first.copy()
    any_correct = np.zeros(len(sids), dtype=bool)
    for m in maps:
        cur = np.array([m[s]["pred"] for s in sids], dtype=np.int64)
        correct = cur == y
        replace = correct & ~any_correct
        oracle_pred[replace] = cur[replace]
        any_correct |= correct

    return {
        "models": [p.name for p in packs],
        "n": int(len(sids)),
        "any_model_correct_acc": float(any_correct.mean()),
        "oracle_f1": float(f1_score(y, oracle_pred, average="macro", zero_division=0)),
        "all_wrong": int((~any_correct).sum()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate complementarity among best.pt checkpoints")
    parser.add_argument("--base", default="config/base.yaml", help="Base YAML used as fallback config")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--model", action="append", default=[], help="name=checkpoint.pt; can be repeated")
    parser.add_argument("--use-default-models", action="store_true", help="Use built-in common checkpoint paths")
    parser.add_argument("--out-dir", default="test/results/complementarity", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    data_root = os.getenv("DATA_ROOT", ".")
    base_cfg = validate_full_config(load_yaml(args.base))

    requested: dict[str, str] = {}
    if args.use_default_models or not args.model:
        requested.update(DEFAULT_MODELS)
    for item in args.model:
        name, path = parse_model_arg(item)
        requested[name] = path

    specs: list[ModelSpec] = []
    for name, raw_path in requested.items():
        try:
            ckpt_path = resolve_existing_path(data_root, raw_path)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        cfg = load_checkpoint_config(base_cfg, ckpt_path)
        specs.append(ModelSpec(name=name, ckpt_path=ckpt_path, cfg=cfg))

    if len(specs) < 2:
        raise RuntimeError("Need at least two valid checkpoints. Pass --model name=path for each best.pt.")

    device = select_device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.split}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] DATA_ROOT={data_root}")
    print(f"[info] device={device} amp={use_amp} split={args.split}")
    print(f"[info] output={out_dir}")

    packs: list[PredictionPack] = []
    for spec in specs:
        print(f"[info] loading {spec.name}: {spec.ckpt_path}")
        pack = collect_predictions(
            spec,
            data_root=data_root,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
        )
        packs.append(pack)
        write_csv(out_dir / f"predictions_{pack.name}.csv", pack.rows)

    model_rows = [metric_summary(p) for p in packs]
    pair_rows = [pairwise_complementarity(a, b) for a, b in itertools.combinations(packs, 2)]
    oracle = multi_model_oracle(packs)

    write_csv(out_dir / "model_metrics.csv", model_rows)
    write_csv(out_dir / "pairwise_complementarity.csv", pair_rows)
    with (out_dir / "multi_model_oracle.json").open("w", encoding="utf-8") as f:
        json.dump(oracle, f, ensure_ascii=False, indent=2)

    print("\n=== Single-model metrics ===")
    for row in model_rows:
        print(f"{row['name']:24s} mode={row['fusion_mode']:15s} n={row['n']:5d} "
              f"F1={row['macro_f1']:.4f} Acc={row['accuracy']:.4f}")

    print("\n=== Pairwise complementarity: oracle gain over stronger model ===")
    for row in sorted(pair_rows, key=lambda x: x.get("oracle_gain_over_best_f1", 0.0), reverse=True):
        print(f"{row['model_a']} vs {row['model_b']}: "
              f"oracle_F1={row['oracle_f1']:.4f}, "
              f"gain={row['oracle_gain_over_best_f1']:+.4f}, "
              f"only_a={row['only_a_rate']:.3f}, only_b={row['only_b_rate']:.3f}, "
              f"err_jaccard={row['error_jaccard']:.3f}, "
              f"best_ens_F1={row['best_prob_ensemble_f1']:.4f}@wA={row['best_weight_for_a']:.1f}")

    print("\n=== Multi-model oracle ===")
    print(json.dumps(oracle, ensure_ascii=False, indent=2))
    print(f"\n[done] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
