from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion.robust.losses import compute_robust_loss
from fusion.robust.dataset import (
    RobustTriModalDataset,
    prepare_robust_batch,
    robust_collate_fn,
)
from fusion.robust.model import TriModalRobustModel
from fusion.robust.utils import build_grad_scaler, get_amp_context


logger = logging.getLogger("tri_modal_robust")


GATE_DIAGNOSTIC_KEYS = (
    "q_api",
    "q_graph",
    "q_manifest",
    "q_align",
    "pert_api",
    "pert_graph",
    "pert_manifest",
    "r_api",
    "r_graph",
    "r_manifest",
    "api_graph_consistency",
    "api_manifest_consistency",
    "graph_manifest_consistency",
    "api_graph_disagreement",
    "api_confidence",
    "graph_confidence",
    "manifest_confidence",
    "joint_confidence",
    "api_alive",
    "graph_alive",
    "manifest_alive",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def select_device(value: str) -> torch.device:
    value = str(value or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device=cuda requested but CUDA is unavailable")
    return torch.device(value)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_path(path: str | Path, seen: set[Path] | None = None) -> dict:
    path = Path(path)
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Recursive config defaults detected: {path}")
    seen.add(resolved)
    raw = load_yaml(path)
    defaults = raw.pop("defaults", []) or []
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    cfg: dict[str, Any] = {}
    for item in defaults:
        item_path = Path(item)
        if not item_path.is_absolute():
            item_path = path.parent / item_path
        cfg = deep_update(cfg, load_config_path(item_path, seen))
    return deep_update(cfg, raw)


def load_config(paths: list[str]) -> dict:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_update(cfg, load_config_path(path))
    return cfg


def resolve(root: str | Path, path: str | Path) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    return str(Path(root) / path)


def build_dataset(cfg: dict, split: str, is_train: bool, perturb_type: str | None = None, perturb_strength: float = 0.0):
    data_cfg = cfg["data"]
    robust_cfg = cfg.get("robust", {})
    model_cfg = cfg.get("model", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    data_root = data_cfg.get("root", "")
    pt_dir = resolve(data_root, data_cfg[f"{split}_pt_dir"])
    csv_path = resolve(data_root, data_cfg[f"{split}_csv"])
    return RobustTriModalDataset(
        pt_dir=pt_dir,
        csv_path=csv_path,
        is_train=is_train,
        robust_aug=bool(robust_cfg.get("train_aug", False)) if is_train else False,
        perturb_prob=float(robust_cfg.get("perturb_prob", 0.5)),
        perturb_strengths=list(robust_cfg.get("perturb_strengths", [0.1, 0.3, 0.5])),
        eval_perturb_type=perturb_type,
        eval_perturb_strength=perturb_strength,
        manifest_dim=int(manifest_cfg.get("in_dim", 256)),
        manifest_category_dim=int(manifest_cfg.get("category_dim", 12)),
        manifest_stats_dim=int(manifest_cfg.get("stats_dim", 11)),
        manifest_permission_dim=int(manifest_cfg.get("permission_dim", 128)),
        manifest_intent_dim=int(manifest_cfg.get("intent_dim", 64)),
        manifest_feature_dim=int(manifest_cfg.get("feature_dim", 32)),
        max_api_events_per_sample=data_cfg.get("max_api_events_per_sample"),
        drop_graph_behavior_hints=bool(model_cfg.get("graph_encoder", {}).get("drop_extracted_behavior_hints", False)),
        degrade_category_counts=bool(robust_cfg.get("degrade_category_counts", True)),
        graph_semantic_source=str(data_cfg.get("graph_semantic_source", "alignment")),
    )


def build_loader(cfg: dict, dataset, is_train: bool):
    train_cfg = cfg["train"]
    workers = int(train_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size" if is_train else "eval_batch_size", train_cfg.get("batch_size", 32))),
        shuffle=is_train,
        num_workers=workers,
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        persistent_workers=bool(train_cfg.get("persistent_workers", False)) and workers > 0,
        collate_fn=robust_collate_fn,
    )


def enforce_failed_ratio(metrics: dict[str, Any], cfg: dict, split_name: str) -> None:
    total = int(metrics.get("num_eval", 0)) + int(metrics.get("num_failed", 0))
    if total <= 0:
        raise RuntimeError(f"{split_name}: no valid or failed samples were seen")
    failed_ratio = float(metrics.get("num_failed", 0)) / float(total)
    max_failed_ratio = float(cfg.get("data", {}).get("max_failed_ratio", 0.0))
    if failed_ratio > max_failed_ratio:
        raise RuntimeError(
            f"{split_name}: failed sample ratio {failed_ratio:.4f} exceeds "
            f"data.max_failed_ratio={max_failed_ratio:.4f}"
        )


def build_model(cfg: dict, feature_dim: int) -> TriModalRobustModel:
    model_cfg = cfg.get("model", {})
    api_cfg = model_cfg.get("api_encoder", {})
    graph_cfg = model_cfg.get("graph_encoder", {})
    manifest_cfg = model_cfg.get("manifest_encoder", {})
    gate_cfg = model_cfg.get("gate", {})
    return TriModalRobustModel(
        in_feat_dim=feature_dim,
        num_classes=int(model_cfg.get("num_classes", 2)),
        fusion_mode=str(model_cfg.get("fusion_mode", "tri_modal_ours")),
        api_num_hash_buckets=int(api_cfg.get("num_hash_buckets", 8192)),
        api_type_vocab_size=int(api_cfg.get("type_vocab_size", 16)),
        api_emb_dim=int(api_cfg.get("emb_dim", 128)),
        api_hidden_dim=int(api_cfg.get("hidden_dim", 256)),
        api_dropout=float(api_cfg.get("dropout", 0.15)),
        api_encoder_type=str(api_cfg.get("type", "transformer")),
        api_layers=int(api_cfg.get("layers", 2)),
        api_heads=int(api_cfg.get("heads", 4)),
        api_max_seq_len=int(api_cfg.get("max_seq_len", 1024)),
        graph_emb_dim=int(graph_cfg.get("emb_dim", 128)),
        graph_hidden=int(graph_cfg.get("hidden", 128)),
        graph_heads=int(graph_cfg.get("heads", 4)),
        graph_layers=int(graph_cfg.get("layers", 2)),
        graph_encoder_type=str(graph_cfg.get("type", "gatv2")),
        max_nodes_gnn=int(model_cfg.get("max_nodes_gnn", graph_cfg.get("max_nodes", 12288))),
        use_graph_behavior_hint=bool(graph_cfg.get("use_behavior_hint", True)),
        manifest_in_dim=int(manifest_cfg.get("in_dim", 256)),
        manifest_emb_dim=int(manifest_cfg.get("emb_dim", 128)),
        manifest_hidden_dim=int(manifest_cfg.get("hidden_dim", 256)),
        manifest_dropout=float(manifest_cfg.get("dropout", 0.1)),
        joint_emb_dim=int(model_cfg.get("joint_emb_dim", 128)),
        gate_hidden_dim=int(gate_cfg.get("hidden_dim", 128)),
        gate_detach=bool(gate_cfg.get("detach", True)),
    )


def _metrics(labels: list[int], probs: list[float], preds: list[int]) -> dict[str, float]:
    if not labels:
        return {"acc": 0.0, "f1": 0.0, "recall": 0.0, "auc": 0.0, "ap": 0.0}
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    if len(set(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
    else:
        out["auc"] = 0.0
        out["ap"] = 0.0
    return out


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, split_name: str, dump_rows: bool = False):
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    preds_all: list[int] = []
    rows: list[dict[str, Any]] = []
    num_failed = 0

    for batch in tqdm(loader, desc=split_name, leave=False):
        graph, labels, sids, _quality, failed = prepare_robust_batch(batch, device)
        num_failed += failed
        if graph is None:
            continue
        with get_amp_context(device, use_amp):
            logits, extra = model(graph, return_features=False)
        prob = torch.softmax(logits.float(), dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        labels_all.extend(labels.detach().cpu().long().tolist())
        probs_all.extend(prob.detach().cpu().tolist())
        preds_all.extend(pred.detach().cpu().long().tolist())

        if dump_rows:
            gate = extra.get("gate_weights")
            for i, sid in enumerate(sids or []):
                row = {
                    "split": split_name,
                    "sid": sid,
                    "label": int(labels[i].detach().cpu().item()),
                    "prob_malware": float(prob[i].detach().cpu().item()),
                    "pred": int(pred[i].detach().cpu().item()),
                    "year": int(batch.get("years")[i].detach().cpu().item()) if batch.get("years") is not None else 0,
                }
                if isinstance(gate, torch.Tensor) and gate.size(0) > i:
                    gate_i = gate[i].detach().cpu()
                    row.update({
                        "w_api": float(gate_i[0].item()),
                        "w_graph": float(gate_i[1].item()),
                        "w_manifest": float(gate_i[2].item()),
                        "w_joint": float(gate_i[3].item()),
                    })
                for key in GATE_DIAGNOSTIC_KEYS:
                    value = extra.get(key)
                    if isinstance(value, torch.Tensor) and value.numel() > i:
                        row[key] = float(value.view(-1)[i].detach().cpu().item())
                rows.append(row)

    metrics = _metrics(labels_all, probs_all, preds_all)
    metrics["num_failed"] = int(num_failed)
    metrics["num_eval"] = int(len(labels_all))
    return metrics, rows


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int):
    model.train()
    use_amp = bool(cfg["train"].get("use_amp", True))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    loss_cfg = dict(cfg.get("loss", {}))
    loss_cfg["label_smoothing"] = float(cfg["train"].get("label_smoothing", loss_cfg.get("label_smoothing", 0.0)))
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    steps = 0
    failed_seen = 0
    valid_seen = 0

    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        graph, labels, _, _quality, failed = prepare_robust_batch(batch, device)
        failed_seen += int(failed)
        if graph is None:
            continue
        valid_seen += int(labels.size(0))
        with get_amp_context(device, use_amp):
            logits, extra = model(graph, return_features=False)
            loss, _ = compute_robust_loss(logits, labels, extra, loss_cfg)
            loss = loss / max(grad_accum, 1)
        steps += 1
        scaler.scale(loss).backward()
        if steps % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().item()) * max(grad_accum, 1)

    if steps > 0 and steps % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    enforce_failed_ratio({"num_eval": valid_seen, "num_failed": failed_seen}, cfg, f"train_epoch_{epoch}")
    return total_loss / max(steps, 1)


def write_gate_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: dict) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO))
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = select_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("use_amp", True))

    train_ds = build_dataset(cfg, "train", is_train=True)
    val_ds = build_dataset(cfg, "val", is_train=False)
    test_ds = build_dataset(cfg, "test", is_train=False)
    train_loader = build_loader(cfg, train_ds, is_train=True)
    val_loader = build_loader(cfg, val_ds, is_train=False)
    test_loader = build_loader(cfg, test_ds, is_train=False)

    model = build_model(cfg, train_ds.feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg.get("epochs", 1))),
        eta_min=float(train_cfg.get("eta_min", 1e-6)),
    )
    scaler = build_grad_scaler(device, use_amp)

    out_dir = Path(data_cfg.get("out_dir", "experiments")) / str(train_cfg.get("exp_name", "tri_modal_robust")) / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = out_dir / "best_tri_modal_robust.pt"
    patience = int(train_cfg.get("patience", 10))
    stale = 0

    for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val_metrics, _ = evaluate(model, val_loader, device, use_amp, "val", dump_rows=False)
        enforce_failed_ratio(val_metrics, cfg, "val")
        scheduler.step()
        logger.info(
            "epoch=%s train_loss=%.4f val_f1=%.4f val_auc=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            val_metrics["f1"],
            val_metrics["auc"],
            val_metrics["acc"],
        )
        if val_metrics["f1"] > best_f1 + float(train_cfg.get("min_delta", 1e-4)):
            best_f1 = val_metrics["f1"]
            stale = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "val": val_metrics, "epoch": epoch}, best_path)
        else:
            stale += 1
            if stale >= patience:
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    val_metrics, val_rows = evaluate(model, val_loader, device, use_amp, "val_clean", dump_rows=True)
    test_metrics, test_rows = evaluate(model, test_loader, device, use_amp, "test_clean", dump_rows=True)
    enforce_failed_ratio(val_metrics, cfg, "val_clean")
    enforce_failed_ratio(test_metrics, cfg, "test_clean")

    all_rows = val_rows + test_rows
    robust_results = {}
    eval_cfg = cfg.get("eval", {})
    perturb_tests = list(eval_cfg.get("perturb_tests", ["clean"]))
    if eval_cfg.get("perturb_strengths") is not None:
        perturb_strengths = [float(v) for v in eval_cfg.get("perturb_strengths") or []]
    else:
        perturb_strengths = [float(eval_cfg.get("perturb_strength", 0.5))]
    perturb_strengths = perturb_strengths or [0.5]
    for perturb in perturb_tests:
        if perturb == "clean":
            robust_results[perturb] = test_metrics
            continue
        for strength in perturb_strengths:
            result_key = perturb if len(perturb_strengths) == 1 else f"{perturb}_s{strength:g}"
            robust_ds = build_dataset(cfg, "test", is_train=False, perturb_type=perturb, perturb_strength=strength)
            robust_loader = build_loader(cfg, robust_ds, is_train=False)
            metrics, rows = evaluate(model, robust_loader, device, use_amp, f"test_{result_key}", dump_rows=True)
            enforce_failed_ratio(metrics, cfg, f"test_{result_key}")
            robust_results[result_key] = metrics
            all_rows.extend(rows)

    write_gate_dump(out_dir / "gate_diagnostics.csv", all_rows)
    summary = {"best_val_f1": best_f1, "val": val_metrics, "test": test_metrics, "robust": robust_results}
    with open(out_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    logger.info("finished: %s", out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train API+Graph+Manifest robust tri-modal fusion.")
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML configs, applied left to right.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
