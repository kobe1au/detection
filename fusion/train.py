from __future__ import annotations

import argparse
import copy
import csv
import gc
import logging
import os
import random
import time as _time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from fusion.mm_dataset import MultiModalMalwareDataset, hierarchical_collate_fn
from fusion.model import MalwareModelWithXAttn
from fusion.losses import compute_total_loss
from fusion.constants import TrainingConstants
from fusion.calibration import TemperatureScaling
from fusion.utils import get_amp_context, build_grad_scaler, prepare_batch
from fusion.selective_metrics import aurc, eaurc, risk_at_coverage, coverage_at_risk
from fusion.temporal_metrics import compute_aut, compute_aut_suite

torch.multiprocessing.set_sharing_strategy('file_system')


# ═══════════════════════════════════════════════════════════════════════
# Seed / helpers
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def select_device(preferred: str = "auto") -> torch.device:
    preferred = str(preferred or "auto").lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("train.device=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if preferred == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("train.device=mps was requested, but MPS is not available")
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported train.device: {preferred}")


def _worker_init_fn(worker_id):
    s = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(s); random.seed(s)


def resolve_path(root, p):
    return p if os.path.isabs(p) else (os.path.join(root, p) if root else p)


def deep_update(base, ov):
    r = copy.deepcopy(base)
    for k, v in ov.items():
        if k in r and isinstance(r[k], dict) and isinstance(v, dict):
            r[k] = deep_update(r[k], v)
        else:
            r[k] = copy.deepcopy(v)
    return r


def load_yaml_file(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_split_years(csv_path: str) -> list[int]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        year_col = next((c for c in ["year", "Year", "年份", "vt_year", "dex_year"] if c in reader.fieldnames), None)
        if year_col is None:
            raise ValueError(f"Temporal setting requires a year column in CSV: {csv_path}")
        years = set()
        for row in reader:
            raw = row.get(year_col, "")
            try:
                years.add(int(float(raw)))
            except (TypeError, ValueError):
                continue
    if not years:
        raise ValueError(f"No valid years found in CSV: {csv_path}")
    return sorted(years)


def build_global_domain_years(*csv_paths: str) -> list[int]:
    years = set()
    for csv_path in csv_paths:
        years.update(read_split_years(csv_path))
    if not years:
        raise ValueError("No years found for global temporal domain mapping")
    return list(range(min(years), max(years) + 1))


def validate_full_config(cfg):
    """Validate the cleaned 2026 config schema.

    This project now assumes fresh training from the new chronological split.
    Legacy knobs such as resume, warm-start checkpoints, graph pretraining,
    TTA/conformal switches, and robustness-suite switches are intentionally
    rejected so old YAMLs fail loudly instead of silently changing behavior.
    """
    for k in ("data", "model", "train", "loss"):
        if k not in cfg:
            raise ValueError(f"missing config key: {k}")

    allowed_top = {"data", "model", "train", "loss"}
    allowed_data = {
        "out_dir", "train_pt_dir", "val_pt_dir", "test_pt_dir",
        "train_csv", "val_csv", "test_csv", "extra_tests",
        "adapt_pt_dir", "adapt_csv",
        "max_api_events_per_sample",
    }
    allowed_extra_test = {"name", "test_pt_dir", "test_csv"}
    allowed_model = {
        "num_classes", "fusion_mode", "max_nodes_gnn", "max_xattn_nodes",
        "api_encoder", "graph_encoder", "temporal", "alignment", "gate",
        "in_feat_dim",
    }
    allowed_api = {"type", "num_hash_buckets", "type_vocab_size", "max_seq_len", "layers", "heads"}
    allowed_graph = {"type", "hidden", "heads", "layers", "use_behavior_hint", "drop_extracted_behavior_hints"}
    allowed_temporal = {"prototype_momentum", "prototype_clusters", "use_future_drift"}
    allowed_alignment = {"enabled", "adaptive_bias", "drift_guided", "penalty_scale", "bonus_scale", "context_scale"}
    allowed_gate = {"mode", "quality_inputs", "drift_inputs", "detach"}
    allowed_train = {
        "exp_name", "seed", "device", "use_amp", "epochs", "batch_size",
        "eval_batch_size", "grad_accum_steps", "num_workers", "pin_memory",
        "persistent_workers", "prefetch_factor", "lr", "weight_decay",
        "warmup_epochs", "eta_min", "label_smoothing", "patience",
        "min_delta", "warmup_stage_epochs",
        "historical_epochs", "adaptation_epochs", "adaptation_ratio", "replay_ratio",
    }
    allowed_loss = {
        "temporal_proto_current_weight", "temporal_proto_future_weight",
        "temporal_risk_calibration_weight",
        "temporal_proto_temperature", "temporal_proto_velocity_scale",
        "temporal_proto_min_history", "semantic_alignment_weight",
        "branch_aux_weight", "stage1_branch_aux_weight",
        "class_aware_alignment_same_class_weight", "class_aware_alignment_temperature",
        "gate_oracle_weight", "gate_oracle_temperature",
    }

    def _reject_unknown(path, value, allowed):
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown config key(s) under {path}: {sorted(extra)}")

    def _require_keys(path, value, required, optional=()):
        missing = set(required) - set(optional) - set(value)
        if missing:
            raise ValueError(f"missing config key(s) under {path}: {sorted(missing)}")

    _reject_unknown("root", cfg, allowed_top)
    _reject_unknown("data", cfg["data"], allowed_data)
    for idx, extra_test in enumerate(cfg["data"].get("extra_tests", []) or []):
        if not isinstance(extra_test, dict):
            raise ValueError(f"data.extra_tests[{idx}] must be a mapping")
        _reject_unknown(f"data.extra_tests[{idx}]", extra_test, allowed_extra_test)
        _require_keys(
            f"data.extra_tests[{idx}]",
            extra_test,
            allowed_extra_test,
        )
    _reject_unknown("model", cfg["model"], allowed_model)
    _reject_unknown("model.api_encoder", cfg["model"].get("api_encoder", {}), allowed_api)
    _reject_unknown("model.graph_encoder", cfg["model"].get("graph_encoder", {}), allowed_graph)
    _reject_unknown("model.temporal", cfg["model"].get("temporal", {}), allowed_temporal)
    _reject_unknown("model.alignment", cfg["model"].get("alignment", {}), allowed_alignment)
    _reject_unknown("model.gate", cfg["model"].get("gate", {}), allowed_gate)
    _reject_unknown("train", cfg["train"], allowed_train)
    _reject_unknown("loss", cfg["loss"], allowed_loss)

    _require_keys("data", cfg["data"], allowed_data, optional={"extra_tests", "adapt_pt_dir", "adapt_csv"})
    _require_keys("model", cfg["model"], allowed_model, optional={"in_feat_dim"})
    _require_keys("model.api_encoder", cfg["model"]["api_encoder"], allowed_api)
    _require_keys("model.graph_encoder", cfg["model"]["graph_encoder"], allowed_graph)
    _require_keys("model.temporal", cfg["model"]["temporal"], allowed_temporal)
    _require_keys("model.alignment", cfg["model"]["alignment"], allowed_alignment)
    _require_keys("model.gate", cfg["model"]["gate"], allowed_gate)
    _require_keys(
        "train",
        cfg["train"],
        allowed_train,
        optional={"historical_epochs", "adaptation_epochs", "adaptation_ratio", "replay_ratio"},
    )
    _require_keys(
        "loss",
        cfg["loss"],
        allowed_loss,
        optional={
            "class_aware_alignment_same_class_weight",
            "class_aware_alignment_temperature",
            "gate_oracle_weight",
            "gate_oracle_temperature",
        },
    )

    nc = int(cfg["model"].get("num_classes", 2))
    if nc < 2:
        raise ValueError(f"num_classes must be >= 2, got {nc}")

    forbidden_fragments = ("pretrained", "resume", "finetune", "tta", "conformal", "robustness", "pretrain")
    dumped = yaml.safe_dump(cfg, allow_unicode=True)
    for frag in forbidden_fragments:
        if frag in dumped:
            raise ValueError(f"legacy config fragment is not supported in clean training schema: {frag}")
    return cfg


def setup_logger(log_path):
    lg = logging.getLogger("train_logger")
    lg.setLevel(logging.INFO); lg.handlers.clear(); lg.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg


def save_config_snapshot(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, allow_unicode=True, sort_keys=False)


# ═══════════════════════════════════════════════════════════════════════
# CSV metrics (simplified — no more reject_rate/keep_f1/corrected_acc)
# ═══════════════════════════════════════════════════════════════════════

_CSV_HEADER = [
    "epoch", "stage", "train_loss", "train_f1", "train_acc",
    "train_cls", "train_temporal", "train_proto_current", "train_proto_future",
    "train_alignment", "train_branch_aux", "train_risk_calib", "train_drift",
    "train_gate_oracle",
    "train_temporal_drift", "train_temporal_risk", "train_alignment_temporal_drift",
    "train_alignment_drift_gate", "train_semantic_drift_gate",
    "train_gate_temporal_drift", "train_gate_disagreement", "train_gate_entropy",
    "train_gate_api", "train_gate_graph", "train_gate_joint",
    "val_loss", "val_cls_loss", "val_f1", "val_acc", "val_softmax_aurc", "val_hybrid_aurc",
    "val_temporal_drift", "val_temporal_risk", "val_temporal_risk_error_auc",
    "val_temporal_risk_aurc", "val_temporal_risk_gap", "val_alignment_temporal_drift",
    "val_alignment_drift_gate", "val_semantic_drift_gate",
    "val_gate_temporal_drift", "val_gate_disagreement", "val_gate_entropy",
    "val_gate_api", "val_gate_graph", "val_gate_joint",
    "lr", "best_score", "is_best", "no_improve",
    "selection_score", "latest_f1", "worst_f1", "aut_f1",
]


def init_metrics_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_CSV_HEADER)


def append_metrics_csv(path, row_dict):
    row = [row_dict.get(k, "") for k in _CSV_HEADER]
    fmt_row = []
    for v in row:
        if isinstance(v, float):
            fmt_row.append(f"{v:.6f}")
        elif isinstance(v, int):
            fmt_row.append(str(v))
        else:
            fmt_row.append(str(v) if v != "" else "")
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fmt_row)


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_score, no_improve, cfg):
    torch.save({
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "best_score": float(best_score),
        "no_improve": int(no_improve),
        "config": cfg,
        "exp_name": cfg.get("train", {}).get("exp_name", "unknown"),
    }, path)


# ═══════════════════════════════════════════════════════════════════════
# Gate weight logging helper
# ═══════════════════════════════════════════════════════════════════════

def _extract_gate_weights(extra):
    gw = extra.get("gate_weights")
    if gw is not None and gw.numel() > 0:
        return float(gw[:, 0].mean()), float(gw[:, 1].mean()), float(gw[:, 2].mean()), True
    return 0.0, 0.0, 0.0, False


def _extract_extra_mean(extra, key):
    value = extra.get(key)
    if value is not None and value.numel() > 0:
        return float(value.float().mean()), True
    return 0.0, False


def _extract_loss_component(extra, key):
    comps = extra.get("loss_components") or {}
    value = comps.get(key)
    if value is not None and value.numel() > 0:
        return float(value.float().mean()), True
    return 0.0, False


def _binary_rank_auc(scores, positives):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = np.asarray(positives, dtype=np.bool_).reshape(-1)
    if scores.shape[0] != positives.shape[0] or scores.shape[0] == 0:
        return 0.0
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    i = 0
    n = scores.shape[0]
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = ranks[positives].sum()
    auc_v = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc_v, 0.0, 1.0))


def _temporal_risk_diagnostics(risk_scores, correct_arr):
    risk_scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    correct_arr = np.asarray(correct_arr, dtype=np.float64).reshape(-1)
    if risk_scores.shape[0] != correct_arr.shape[0] or risk_scores.shape[0] == 0:
        return {
            "error_auc": 0.0,
            "risk_aurc": 0.0,
            "risk_gap": 0.0,
            "risk_correct": 0.0,
            "risk_wrong": 0.0,
        }

    risk_scores = np.clip(risk_scores, 0.0, 1.0)
    correct_bool = correct_arr >= 0.5
    wrong_bool = ~correct_bool
    risk_correct = float(risk_scores[correct_bool].mean()) if correct_bool.any() else 0.0
    risk_wrong = float(risk_scores[wrong_bool].mean()) if wrong_bool.any() else 0.0
    return {
        "error_auc": _binary_rank_auc(risk_scores, wrong_bool),
        "risk_aurc": float(aurc(1.0 - risk_scores, correct_arr)),
        "risk_gap": risk_wrong - risk_correct,
        "risk_correct": risk_correct,
        "risk_wrong": risk_wrong,
    }


def _hybrid_confidence(softmax_conf, risk_scores):
    softmax_conf = np.asarray(softmax_conf, dtype=np.float64).reshape(-1)
    risk_scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    if softmax_conf.shape[0] == 0 or risk_scores.shape[0] != softmax_conf.shape[0]:
        return softmax_conf
    risk_scores = np.clip(risk_scores, 0.0, 1.0)
    return np.clip(softmax_conf * (1.0 - risk_scores), 0.0, 1.0)


def _gate_diagnostics(gate_weights, correct=None, risk=None, q_api=None, q_graph=None):
    if not gate_weights:
        return {}
    gw = np.asarray(gate_weights, dtype=np.float64)
    if gw.ndim != 2 or gw.shape[1] != 3 or gw.shape[0] == 0:
        return {}
    gw = np.clip(gw, 1e-8, 1.0)
    entropy = float((-(gw * np.log(gw)).sum(axis=1) / np.log(3.0)).mean())
    out = {
        "entropy": entropy,
        "std": gw.std(axis=0).tolist(),
    }

    def _mean_by_mask(name, values, high_name=None):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != gw.shape[0]:
            return
        if high_name is None:
            mask = arr >= 0.5
            out[f"{name}_ge_0.5"] = gw[mask].mean(axis=0).tolist() if mask.any() else None
            out[f"{name}_lt_0.5"] = gw[~mask].mean(axis=0).tolist() if (~mask).any() else None
        else:
            q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
            low = arr <= q1
            high = arr >= q2
            out[f"{name}_low"] = gw[low].mean(axis=0).tolist() if low.any() else None
            out[f"{high_name}_high"] = gw[high].mean(axis=0).tolist() if high.any() else None

    if correct is not None:
        _mean_by_mask("correct", correct)
    if risk is not None:
        _mean_by_mask("risk", risk, high_name="risk")
    if q_api is not None:
        _mean_by_mask("qapi", q_api, high_name="qapi")
    if q_graph is not None:
        _mean_by_mask("qgraph", q_graph, high_name="qgraph")
    return out


def _extract_gate_weight_stds(extra):
    gw = extra.get("gate_weights")
    if gw is not None and gw.numel() > 0:
        std = gw.float().std(dim=0, unbiased=False)
        return float(std[0]), float(std[1]), float(std[2]), True
    return 0.0, 0.0, 0.0, False


def _maybe_update_temporal_prototype_memory(extra, y, loss_cfg):
    if (
        float(loss_cfg["temporal_proto_current_weight"]) == 0.0
        and float(loss_cfg["temporal_proto_future_weight"]) == 0.0
        and float(loss_cfg["temporal_risk_calibration_weight"]) == 0.0
    ):
        return
    memory = extra.get("temporal_prototype_memory")
    features = extra.get("temporal_features")
    time_ids = extra.get("time_ids")
    if memory is None or features is None or time_ids is None:
        return
    memory.update_weighted(features, y, time_ids, extra.get("temporal_quality"))


def _is_multimodal_mode(fusion_mode: str) -> bool:
    return str(fusion_mode) in {"concat", "cross_attention", "late_fusion", "ours"}


def _stage_loss_cfg(loss_cfg: dict, stage: str) -> dict:
    cfg = copy.deepcopy(loss_cfg)
    if stage == "warmup":
        cfg["temporal_proto_future_weight"] = 0.0
        cfg["temporal_risk_calibration_weight"] = 0.0
        cfg["semantic_alignment_weight"] = 0.0
        cfg["branch_aux_weight"] = float(cfg["stage1_branch_aux_weight"])
    return cfg


def _balanced_sample_indices(dataset, ratio: float, seed: int, min_per_class: int = 1):
    ratio = float(ratio)
    if ratio >= 1.0:
        return list(range(len(dataset)))
    if ratio <= 0.0:
        return []
    by_label: dict[int, list[int]] = {}
    labels = getattr(dataset, "labels", None)
    sample_sids = getattr(dataset, "sample_sids", None)
    for idx in range(len(dataset)):
        label = None
        if labels is not None and sample_sids is not None and idx < len(sample_sids):
            label = int(labels[sample_sids[idx]])
        else:
            sample = getattr(dataset, "samples", [])[idx]
            if len(sample) >= 2:
                label = int(sample[1])
        if label is None:
            continue
        by_label.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    selected = []
    for _, indices in sorted(by_label.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        keep = int(round(len(shuffled) * ratio))
        keep = max(min_per_class, keep) if shuffled else 0
        selected.extend(shuffled[: min(keep, len(shuffled))])
    selected.sort()
    return selected


def _build_continual_adaptation_loader(
    historical_dataset,
    adapt_dataset,
    cfg,
    batch_size,
    loader_kwargs,
):
    train_cfg = cfg["train"]
    adapt_ratio = float(train_cfg.get("adaptation_ratio", 1.0))
    replay_ratio = float(train_cfg.get("replay_ratio", 0.25))
    seed = int(train_cfg.get("seed", 42))

    adapt_indices = _balanced_sample_indices(adapt_dataset, adapt_ratio, seed)
    replay_indices = _balanced_sample_indices(historical_dataset, replay_ratio, seed + 1009, min_per_class=1)
    if not adapt_indices:
        raise ValueError("continual adaptation requested, but adaptation_ratio selected no samples")
    datasets = [Subset(adapt_dataset, adapt_indices)]
    if replay_indices:
        datasets.append(Subset(historical_dataset, replay_indices))
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    return loader, len(adapt_indices), len(replay_indices)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch, num_epochs, loss_cfg=None, logger=None, use_amp=False,
                    grad_accum_steps=1):
    model.train()
    loss_cfg = loss_cfg or {}
    grad_accum_steps = max(1, int(grad_accum_steps))

    total_loss = total_steps = total_samples = 0
    sum_cls = sum_temp = sum_proto_current = sum_proto_future = sum_align = 0.0
    sum_branch_aux = 0.0
    sum_gate_oracle = 0.0
    sum_risk_calib = 0.0
    sum_drift = 0.0; num_drift = 0
    sum_gate_temporal_drift = 0.0; num_gate_temporal_drift = 0
    sum_gate_disagreement = 0.0; num_gate_disagreement = 0
    sum_gate_entropy = 0.0; num_gate_entropy = 0
    sum_temporal_drift = 0.0; num_temporal_drift = 0
    sum_temporal_risk = 0.0; num_temporal_risk = 0
    sum_proto_pred_dist = 0.0; num_proto_pred_dist = 0
    sum_proto_margin_risk = 0.0; num_proto_margin_risk = 0
    sum_proto_label_mismatch = 0.0; num_proto_label_mismatch = 0
    sum_proto_reliability_risk = 0.0; num_proto_reliability_risk = 0
    sum_align_temporal_drift = 0.0; num_align_temporal_drift = 0
    sum_align_drift_gate = 0.0; num_align_drift_gate = 0
    sum_semantic_drift_gate = 0.0; num_semantic_drift_gate = 0
    sum_wi = sum_wg = sum_wj = 0.0; num_w = 0
    sum_wi_std = sum_wg_std = sum_wj_std = 0.0; num_w_std = 0
    sum_align_cov = 0.0; num_align_cov = 0
    sum_align_density = 0.0; num_align_density = 0
    skipped = failed = total_valid = 0
    accum_steps = optimizer_steps = 0
    all_preds, all_labels = [], []

    optimizer.zero_grad(set_to_none=True)

    def _optimizer_step():
        nonlocal accum_steps, optimizer_steps
        if accum_steps <= 0:
            return
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=TrainingConstants.GRAD_CLIP_MAX_NORM,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        accum_steps = 0
        optimizer_steps += 1

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{num_epochs}", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        if batch is None:
            skipped += 1; continue
        try:
            r = prepare_batch(batch, device,
                              skip_graph=False,
                              skip_masks=(model.fusion_mode != "ours" or not getattr(model, "use_alignment_bias", False)))
            if r[2] is None:
                skipped += 1; failed += r[-1]; continue

            graph, masks, y, _, ex, nf = r
            failed += nf
            qapi, qg, qa, papi, pg, tids = ex

            bs = y.size(0)
            total_valid += bs

            with get_amp_context(device, enabled=use_amp):
                logits, extra = model(graph_data=graph, y=y,
                                      explicit_qs=(qapi, qg, qa, papi, pg),
                                      time_ids=tids, masks=masks)
                loss, l_cls, l_temp, l_align = compute_total_loss(
                    logits, extra, y, criterion, loss_cfg,
                    epoch=epoch, total_epochs=num_epochs)
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            _maybe_update_temporal_prototype_memory(extra, y, loss_cfg)
            accum_steps += 1
            if accum_steps >= grad_accum_steps:
                _optimizer_step()

            lv = float(loss.item())
            sum_cls += float(l_cls.item()) * bs
            sum_temp += float(l_temp.item()) * bs
            sum_align += float(l_align.item()) * bs
            proto_current, ok_proto_current = _extract_loss_component(extra, "proto_current")
            if ok_proto_current:
                sum_proto_current += proto_current * bs
            proto_future, ok_proto_future = _extract_loss_component(extra, "proto_future")
            if ok_proto_future:
                sum_proto_future += proto_future * bs
            branch_aux, ok_branch_aux = _extract_loss_component(extra, "branch_aux")
            if ok_branch_aux:
                sum_branch_aux += branch_aux * bs
            gate_oracle, ok_gate_oracle = _extract_loss_component(extra, "gate_oracle")
            if ok_gate_oracle:
                sum_gate_oracle += gate_oracle * bs
            risk_calib, ok_risk_calib = _extract_loss_component(extra, "temporal_risk_calibration")
            if ok_risk_calib:
                sum_risk_calib += risk_calib * bs
            drift_mean, ok_drift = _extract_extra_mean(extra, "drift_score")
            if ok_drift:
                sum_drift += drift_mean * bs; num_drift += bs
            gate_temporal_drift, ok_gate_temporal_drift = _extract_extra_mean(extra, "gate_temporal_drift")
            if ok_gate_temporal_drift:
                sum_gate_temporal_drift += gate_temporal_drift * bs; num_gate_temporal_drift += bs
            gate_disagreement, ok_gate_disagreement = _extract_extra_mean(extra, "gate_disagreement")
            if ok_gate_disagreement:
                sum_gate_disagreement += gate_disagreement * bs; num_gate_disagreement += bs
            gate_entropy, ok_gate_entropy = _extract_extra_mean(extra, "gate_entropy")
            if ok_gate_entropy:
                sum_gate_entropy += gate_entropy * bs; num_gate_entropy += bs
            temporal_drift, ok_temporal_drift = _extract_extra_mean(extra, "temporal_drift_score")
            if ok_temporal_drift:
                sum_temporal_drift += temporal_drift * bs; num_temporal_drift += bs
            temporal_risk, ok_temporal_risk = _extract_extra_mean(extra, "temporal_risk_score")
            if ok_temporal_risk:
                sum_temporal_risk += temporal_risk * bs; num_temporal_risk += bs
            proto_pred_dist, ok_proto_pred_dist = _extract_extra_mean(extra, "temporal_proto_pred_dist")
            if ok_proto_pred_dist:
                sum_proto_pred_dist += proto_pred_dist * bs; num_proto_pred_dist += bs
            proto_margin_risk, ok_proto_margin_risk = _extract_extra_mean(extra, "temporal_proto_margin_risk")
            if ok_proto_margin_risk:
                sum_proto_margin_risk += proto_margin_risk * bs; num_proto_margin_risk += bs
            proto_label_mismatch, ok_proto_label_mismatch = _extract_extra_mean(extra, "temporal_proto_label_mismatch")
            if ok_proto_label_mismatch:
                sum_proto_label_mismatch += proto_label_mismatch * bs; num_proto_label_mismatch += bs
            proto_reliability_risk, ok_proto_reliability_risk = _extract_extra_mean(extra, "temporal_proto_reliability_risk")
            if ok_proto_reliability_risk:
                sum_proto_reliability_risk += proto_reliability_risk * bs; num_proto_reliability_risk += bs
            align_temporal_drift, ok_align_temporal_drift = _extract_extra_mean(extra, "alignment_temporal_drift")
            if ok_align_temporal_drift:
                sum_align_temporal_drift += align_temporal_drift * bs; num_align_temporal_drift += bs
            align_drift_gate, ok_align_drift_gate = _extract_extra_mean(extra, "alignment_drift_gate")
            if ok_align_drift_gate:
                sum_align_drift_gate += align_drift_gate * bs; num_align_drift_gate += bs
            semantic_drift_gate, ok_semantic_drift_gate = _extract_extra_mean(extra, "semantic_alignment_drift_gate")
            if ok_semantic_drift_gate:
                sum_semantic_drift_gate += semantic_drift_gate * bs; num_semantic_drift_gate += bs
            total_loss += lv * bs
            total_samples += bs
            total_steps += 1

            wi, wg, wj, ok = _extract_gate_weights(extra)
            if ok:
                sum_wi += wi; sum_wg += wg; sum_wj += wj; num_w += 1
            wi_std, wg_std, wj_std, ok_std = _extract_gate_weight_stds(extra)
            if ok_std:
                sum_wi_std += wi_std; sum_wg_std += wg_std; sum_wj_std += wj_std; num_w_std += 1
            align_cov, ok_align = _extract_extra_mean(extra, "alignment_coverage")
            if ok_align:
                sum_align_cov += align_cov; num_align_cov += 1
            align_density, ok_density = _extract_extra_mean(extra, "alignment_density")
            if ok_density:
                sum_align_density += align_density; num_align_density += 1

            all_preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

            pbar.set_postfix(
                loss=f"{lv:.4f}",
                cls=f"{l_cls.item():.4f}",
                temp=f"{l_temp.item():.4f}",
                pcur=f"{proto_current:.4f}",
                pfut=f"{proto_future:.4f}",
                aux=f"{branch_aux:.4f}",
                gorl=f"{gate_oracle:.4f}",
                rcal=f"{risk_calib:.4f}",
                align=f"{l_align.item():.4f}",
                accum=f"{accum_steps}/{grad_accum_steps}",
            )

            del graph, masks, y, ex, logits, extra, loss, loss_for_backward

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if logger:
                    if device.type == "cuda":
                        alloc = torch.cuda.memory_allocated(device) / (1024**3)
                        reserved = torch.cuda.memory_reserved(device) / (1024**3)
                        logger.error(
                            f"OOM step {step}, skipping | "
                            f"allocated={alloc:.2f}GB reserved={reserved:.2f}GB"
                        )
                    else:
                        logger.error(f"OOM step {step}, skipping")
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                skipped += 1; continue
            raise

    _optimizer_step()

    n = max(total_samples, 1)
    avg = total_loss / n

    if not all_labels:
        if logger:
            logger.warning(f"[train][epoch {epoch}] No valid predictions collected!")
        return (0.0,) * 23
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    if logger:
        logger.info(f"[train][epoch {epoch}] valid={total_valid} failed={failed} "
                    f"skipped={skipped} optim_steps={optimizer_steps} "
                    f"grad_accum={grad_accum_steps} avg_w=({sum_wi/max(num_w,1):.3f},"
                    f"{sum_wg/max(num_w,1):.3f},{sum_wj/max(num_w,1):.3f}) "
                    f"std_w=({sum_wi_std/max(num_w_std,1):.3f},"
                    f"{sum_wg_std/max(num_w_std,1):.3f},{sum_wj_std/max(num_w_std,1):.3f}) "
                    f"drift={sum_drift/max(num_drift,1):.3f} "
                    f"gate_t={sum_gate_temporal_drift/max(num_gate_temporal_drift,1):.3f} "
                    f"gate_dis={sum_gate_disagreement/max(num_gate_disagreement,1):.3f} "
                    f"gate_ent={sum_gate_entropy/max(num_gate_entropy,1):.3f} "
                    f"tdrift={sum_temporal_drift/max(num_temporal_drift,1):.3f} "
                    f"trisk={sum_temporal_risk/max(num_temporal_risk,1):.3f} "
                    f"pdist={sum_proto_pred_dist/max(num_proto_pred_dist,1):.3f} "
                    f"pmargin={sum_proto_margin_risk/max(num_proto_margin_risk,1):.3f} "
                    f"pmis={sum_proto_label_mismatch/max(num_proto_label_mismatch,1):.3f} "
                    f"prel={sum_proto_reliability_risk/max(num_proto_reliability_risk,1):.3f} "
                    f"align_tdrift={sum_align_temporal_drift/max(num_align_temporal_drift,1):.3f} "
                    f"align_dgate={sum_align_drift_gate/max(num_align_drift_gate,1):.3f} "
                    f"sem_dgate={sum_semantic_drift_gate/max(num_semantic_drift_gate,1):.3f} "
                    f"align_cov={sum_align_cov/max(num_align_cov,1):.3f} "
                    f"align_density={sum_align_density/max(num_align_density,1):.4f}")

    return (
        avg, f1, acc, sum_cls/n, sum_temp/n,
        sum_align/n, sum_branch_aux/n, sum_risk_calib/n, sum_proto_current/n, sum_proto_future/n,
        sum_gate_oracle/n,
        sum_drift/max(num_drift, 1),
        sum_temporal_drift/max(num_temporal_drift, 1),
        sum_temporal_risk/max(num_temporal_risk, 1),
        sum_align_temporal_drift/max(num_align_temporal_drift, 1),
        sum_align_drift_gate/max(num_align_drift_gate, 1),
        sum_semantic_drift_gate/max(num_semantic_drift_gate, 1),
        sum_gate_temporal_drift/max(num_gate_temporal_drift, 1),
        sum_gate_disagreement/max(num_gate_disagreement, 1),
        sum_gate_entropy/max(num_gate_entropy, 1),
        sum_wi/max(num_w, 1), sum_wg/max(num_w, 1), sum_wj/max(num_w, 1),
    )


# ═══════════════════════════════════════════════════════════════════════
# eval_one_epoch (returns AURC instead of keep_f1/corrected_acc)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch, num_epochs=1,
                   use_amp=False, logger=None):
    model.eval()

    total_loss = total_steps = total_samples = 0
    sum_cls = 0.0
    sum_drift = 0.0; num_drift = 0
    sum_gate_temporal_drift = 0.0; num_gate_temporal_drift = 0
    sum_gate_disagreement = 0.0; num_gate_disagreement = 0
    sum_gate_entropy = 0.0; num_gate_entropy = 0
    sum_temporal_drift = 0.0; num_temporal_drift = 0
    sum_temporal_risk = 0.0; num_temporal_risk = 0
    sum_proto_pred_dist = 0.0; num_proto_pred_dist = 0
    sum_proto_margin_risk = 0.0; num_proto_margin_risk = 0
    sum_proto_label_mismatch = 0.0; num_proto_label_mismatch = 0
    sum_proto_reliability_risk = 0.0; num_proto_reliability_risk = 0
    sum_align_temporal_drift = 0.0; num_align_temporal_drift = 0
    sum_align_drift_gate = 0.0; num_align_drift_gate = 0
    sum_semantic_drift_gate = 0.0; num_semantic_drift_gate = 0
    sum_wi = sum_wg = sum_wj = 0.0; num_w = 0
    sum_wi_std = sum_wg_std = sum_wj_std = 0.0; num_w_std = 0
    sum_align_cov = 0.0; num_align_cov = 0
    sum_align_density = 0.0; num_align_density = 0
    skipped = failed = total_valid = 0
    all_preds, all_labels, all_confs = [], [], []
    all_temporal_risks = []
    all_gate_weights = []
    all_gate_correct = []
    all_gate_risk = []
    all_gate_qapi = []
    all_gate_qgraph = []
    times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in tqdm(loader, desc=f"Eval Epoch {epoch}", dynamic_ncols=True):
        if batch is None:
            skipped += 1; continue
        r = prepare_batch(batch, device,
                          skip_graph=False,
                          skip_masks=(model.fusion_mode != "ours" or not getattr(model, "use_alignment_bias", False)))
        if r[2] is None:
            skipped += 1; failed += r[-1]; continue

        graph, masks, y, _, ex, nf = r
        failed += nf
        qapi, qg, qa, papi, pg, tids = ex
        bs = y.size(0)
        total_valid += bs
        batch_gate_ok = False

        with get_amp_context(device, enabled=use_amp):
            t0 = _time.perf_counter()
            logits, extra = model(graph_data=graph,
                                  explicit_qs=(qapi, qg, qa, papi, pg),
                                  time_ids=tids, masks=masks)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(_time.perf_counter() - t0)

            wi, wg, wj, ok = _extract_gate_weights(extra)
            if ok:
                sum_wi += wi; sum_wg += wg; sum_wj += wj; num_w += 1
                gate_values = extra.get("gate_weights")
                if isinstance(gate_values, torch.Tensor) and gate_values.numel() > 0:
                    gv = gate_values.detach().float().cpu()
                    if gv.ndim == 2 and gv.size(0) == bs and gv.size(1) == 3:
                        all_gate_weights.extend(gv.tolist())
                        batch_gate_ok = True
            wi_std, wg_std, wj_std, ok_std = _extract_gate_weight_stds(extra)
            if ok_std:
                sum_wi_std += wi_std; sum_wg_std += wg_std; sum_wj_std += wj_std; num_w_std += 1
            drift_mean, ok_drift = _extract_extra_mean(extra, "drift_score")
            if ok_drift:
                sum_drift += drift_mean * bs; num_drift += bs
            gate_temporal_drift, ok_gate_temporal_drift = _extract_extra_mean(extra, "gate_temporal_drift")
            if ok_gate_temporal_drift:
                sum_gate_temporal_drift += gate_temporal_drift * bs; num_gate_temporal_drift += bs
            gate_disagreement, ok_gate_disagreement = _extract_extra_mean(extra, "gate_disagreement")
            if ok_gate_disagreement:
                sum_gate_disagreement += gate_disagreement * bs; num_gate_disagreement += bs
            gate_entropy, ok_gate_entropy = _extract_extra_mean(extra, "gate_entropy")
            if ok_gate_entropy:
                sum_gate_entropy += gate_entropy * bs; num_gate_entropy += bs
            temporal_drift, ok_temporal_drift = _extract_extra_mean(extra, "temporal_drift_score")
            if ok_temporal_drift:
                sum_temporal_drift += temporal_drift * bs; num_temporal_drift += bs
            temporal_risk, ok_temporal_risk = _extract_extra_mean(extra, "temporal_risk_score")
            if ok_temporal_risk:
                sum_temporal_risk += temporal_risk * bs; num_temporal_risk += bs
                risk_values = extra.get("temporal_risk_score")
                if isinstance(risk_values, torch.Tensor):
                    risk_values = risk_values.detach().float().view(-1)
                    if risk_values.numel() == bs:
                        all_temporal_risks.extend(risk_values.cpu().tolist())
            proto_pred_dist, ok_proto_pred_dist = _extract_extra_mean(extra, "temporal_proto_pred_dist")
            if ok_proto_pred_dist:
                sum_proto_pred_dist += proto_pred_dist * bs; num_proto_pred_dist += bs
            proto_margin_risk, ok_proto_margin_risk = _extract_extra_mean(extra, "temporal_proto_margin_risk")
            if ok_proto_margin_risk:
                sum_proto_margin_risk += proto_margin_risk * bs; num_proto_margin_risk += bs
            proto_label_mismatch, ok_proto_label_mismatch = _extract_extra_mean(extra, "temporal_proto_label_mismatch")
            if ok_proto_label_mismatch:
                sum_proto_label_mismatch += proto_label_mismatch * bs; num_proto_label_mismatch += bs
            proto_reliability_risk, ok_proto_reliability_risk = _extract_extra_mean(extra, "temporal_proto_reliability_risk")
            if ok_proto_reliability_risk:
                sum_proto_reliability_risk += proto_reliability_risk * bs; num_proto_reliability_risk += bs
            align_temporal_drift, ok_align_temporal_drift = _extract_extra_mean(extra, "alignment_temporal_drift")
            if ok_align_temporal_drift:
                sum_align_temporal_drift += align_temporal_drift * bs; num_align_temporal_drift += bs
            align_drift_gate, ok_align_drift_gate = _extract_extra_mean(extra, "alignment_drift_gate")
            if ok_align_drift_gate:
                sum_align_drift_gate += align_drift_gate * bs; num_align_drift_gate += bs
            semantic_drift_gate, ok_semantic_drift_gate = _extract_extra_mean(extra, "semantic_alignment_drift_gate")
            if ok_semantic_drift_gate:
                sum_semantic_drift_gate += semantic_drift_gate * bs; num_semantic_drift_gate += bs
            align_cov, ok_align = _extract_extra_mean(extra, "alignment_coverage")
            if ok_align:
                sum_align_cov += align_cov; num_align_cov += 1
            align_density, ok_density = _extract_extra_mean(extra, "alignment_density")
            if ok_density:
                sum_align_density += align_density; num_align_density += 1

            loss_cls = criterion(logits, y)
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

        sum_cls += float(loss_cls.item()) * bs
        total_loss += float(loss_cls.item()) * bs
        total_samples += bs
        total_steps += 1

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_confs.extend(conf.cpu().tolist())
        if batch_gate_ok:
            batch_correct = (preds == y).detach().float().cpu().tolist()
            all_gate_correct.extend(batch_correct)
            all_gate_qapi.extend(qapi.detach().float().view(-1).cpu().tolist())
            all_gate_qgraph.extend(qg.detach().float().view(-1).cpu().tolist())
            risk_values = extra.get("temporal_risk_score")
            if isinstance(risk_values, torch.Tensor) and risk_values.numel() == bs:
                all_gate_risk.extend(risk_values.detach().float().view(-1).cpu().tolist())

    if not all_labels:
        return (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0)

    n = max(total_samples, 1)
    avg_loss = total_loss / n
    avg_cls = sum_cls / n
    pred_arr = np.array(all_preds)
    label_arr = np.array(all_labels)
    conf_arr = np.array(all_confs)
    correct_arr = (pred_arr == label_arr).astype(np.float64)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    softmax_aurc_v = float(aurc(conf_arr, correct_arr))
    risk_diag = _temporal_risk_diagnostics(all_temporal_risks, correct_arr)
    hybrid_conf = _hybrid_confidence(conf_arr, all_temporal_risks)
    hybrid_aurc_v = float(aurc(hybrid_conf, correct_arr))
    gate_diag = _gate_diagnostics(
        all_gate_weights,
        correct=all_gate_correct,
        risk=(all_gate_risk if len(all_gate_risk) == len(all_gate_weights) else None),
        q_api=(all_gate_qapi if len(all_gate_qapi) == len(all_gate_weights) else None),
        q_graph=(all_gate_qgraph if len(all_gate_qgraph) == len(all_gate_weights) else None),
    )

    aw = (sum_wi/max(num_w,1), sum_wg/max(num_w,1), sum_wj/max(num_w,1))

    if logger:
        logger.info(f"[eval][epoch {epoch}] loss={avg_loss:.4f} F1={f1:.4f} "
                    f"Acc={acc:.4f} softmax_AURC={softmax_aurc_v:.4f} "
                    f"hybrid_AURC={hybrid_aurc_v:.4f} "
                    f"avg_w=({aw[0]:.3f},{aw[1]:.3f},{aw[2]:.3f}) "
                    f"std_w=({sum_wi_std/max(num_w_std,1):.3f},"
                    f"{sum_wg_std/max(num_w_std,1):.3f},{sum_wj_std/max(num_w_std,1):.3f}) "
                    f"drift={sum_drift/max(num_drift,1):.3f} "
                    f"gate_t={sum_gate_temporal_drift/max(num_gate_temporal_drift,1):.3f} "
                    f"gate_dis={sum_gate_disagreement/max(num_gate_disagreement,1):.3f} "
                    f"gate_ent={sum_gate_entropy/max(num_gate_entropy,1):.3f} "
                    f"tdrift={sum_temporal_drift/max(num_temporal_drift,1):.3f} "
                    f"trisk={sum_temporal_risk/max(num_temporal_risk,1):.3f} "
                    f"pdist={sum_proto_pred_dist/max(num_proto_pred_dist,1):.3f} "
                    f"pmargin={sum_proto_margin_risk/max(num_proto_margin_risk,1):.3f} "
                    f"pmis={sum_proto_label_mismatch/max(num_proto_label_mismatch,1):.3f} "
                    f"prel={sum_proto_reliability_risk/max(num_proto_reliability_risk,1):.3f} "
                    f"align_tdrift={sum_align_temporal_drift/max(num_align_temporal_drift,1):.3f} "
                    f"align_dgate={sum_align_drift_gate/max(num_align_drift_gate,1):.3f} "
                    f"sem_dgate={sum_semantic_drift_gate/max(num_semantic_drift_gate,1):.3f} "
                    f"align_cov={sum_align_cov/max(num_align_cov,1):.3f} "
                    f"align_density={sum_align_density/max(num_align_density,1):.4f} "
                    f"valid={total_valid} failed={failed} skipped={skipped}")
        if all_temporal_risks:
            logger.info(
                f"[eval][epoch {epoch}] temporal-risk: "
                f"error_auc={risk_diag['error_auc']:.4f} "
                f"risk_aurc={risk_diag['risk_aurc']:.4f} "
                f"gap_wrong_minus_correct={risk_diag['risk_gap']:.4f} "
                f"mean_correct={risk_diag['risk_correct']:.4f} "
                f"mean_wrong={risk_diag['risk_wrong']:.4f}"
            )
        if gate_diag:
            std = gate_diag.get("std", [0.0, 0.0, 0.0])
            logger.info(
                f"[eval][epoch {epoch}] gate-diagnostics: "
                f"weight_entropy={gate_diag.get('entropy', 0.0):.4f} "
                f"std=({std[0]:.3f},{std[1]:.3f},{std[2]:.3f}) "
                f"correct={gate_diag.get('correct_ge_0.5')} "
                f"wrong={gate_diag.get('correct_lt_0.5')} "
                f"risk_low={gate_diag.get('risk_low')} "
                f"risk_high={gate_diag.get('risk_high')}"
            )
        if times and total_valid > 0:
            ts = sum(times)
            logger.info(f"[eval][epoch {epoch}] ⏱ per_sample={ts/total_valid*1000:.2f}ms "
                        f"throughput={total_valid/max(ts,1e-6):.1f}/s")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            logger.info(f"[eval][epoch {epoch}] ⏱ peak_gpu={peak:.1f}MB")

    return (
        avg_loss, avg_cls, f1, acc, softmax_aurc_v, hybrid_aurc_v, aw[0], aw[1], aw[2],
        sum_temporal_drift/max(num_temporal_drift, 1),
        sum_temporal_risk/max(num_temporal_risk, 1),
        risk_diag["error_auc"],
        risk_diag["risk_aurc"],
        risk_diag["risk_gap"],
        sum_align_temporal_drift/max(num_align_temporal_drift, 1),
        sum_align_drift_gate/max(num_align_drift_gate, 1),
        sum_semantic_drift_gate/max(num_semantic_drift_gate, 1),
        sum_gate_temporal_drift/max(num_gate_temporal_drift, 1),
        sum_gate_disagreement/max(num_gate_disagreement, 1),
        sum_gate_entropy/max(num_gate_entropy, 1),
    )

# ═══════════════════════════════════════════════════════════════════════
# Temporal eval + selection
# ═══════════════════════════════════════════════════════════════════════

def build_year_subset_loaders(dataset, batch_size, loader_kwargs):
    y2i = getattr(dataset, "year_to_indices", None)
    if not y2i or len(y2i) <= 1:
        return {}
    loaders = {}
    kw = dict(loader_kwargs); kw.pop("generator", None)
    for y in sorted(y2i.keys()):
        loaders[int(y)] = DataLoader(
            Subset(dataset, y2i[y]), batch_size=batch_size, shuffle=False, **kw)
    return loaders


def evaluate_temporal_windows(model, year_loaders, criterion, device, epoch,
                              num_epochs, use_amp=False, logger=None, tag="val"):
    metrics = {}
    for y, dl in sorted(year_loaders.items()):
        (loss, cls, f1, acc, softmax_aurc_v, hybrid_aurc_v, aw_api, aw_graph, aw_joint,
         temporal_drift, temporal_risk, temporal_risk_error_auc,
         temporal_risk_aurc, temporal_risk_gap,
         align_temporal_drift, align_drift_gate, semantic_drift_gate,
         gate_temporal_drift, gate_disagreement, gate_entropy) = eval_one_epoch(
            model, dl, criterion, device, epoch, num_epochs=num_epochs,
            use_amp=use_amp, logger=None)
        metrics[int(y)] = {
            "loss": loss,
            "f1": f1,
            "acc": acc,
            "softmax_aurc": softmax_aurc_v,
            "hybrid_aurc": hybrid_aurc_v,
            "gate_api": aw_api,
            "gate_graph": aw_graph,
            "gate_joint": aw_joint,
            "temporal_drift": temporal_drift,
            "temporal_risk": temporal_risk,
            "temporal_risk_error_auc": temporal_risk_error_auc,
            "temporal_risk_aurc": temporal_risk_aurc,
            "temporal_risk_gap": temporal_risk_gap,
            "alignment_temporal_drift": align_temporal_drift,
            "alignment_drift_gate": align_drift_gate,
            "semantic_drift_gate": semantic_drift_gate,
            "gate_temporal_drift": gate_temporal_drift,
            "gate_disagreement": gate_disagreement,
            "gate_entropy": gate_entropy,
        }
    if logger and metrics:
        lines = [f"{y}:F1={m['f1']:.3f}/softmax_AURC={m['softmax_aurc']:.3f}"
                 for y, m in sorted(metrics.items())]
        logger.info(f"[{tag}][epoch {epoch}] yearly: " + " | ".join(lines))
    return metrics


def compute_temporal_selection_score(year_metrics):
    """Selection: 0.5*AUT(F1) + 0.3*latest_F1 + 0.2*worst_F1."""
    if not year_metrics:
        return 0.0, 0.0, 0.0, 0.0
    years = sorted(year_metrics.keys())
    per_year_f1 = {y: year_metrics[y]["f1"] for y in years}
    aut_f1 = compute_aut(per_year_f1)
    latest = per_year_f1[years[-1]]
    worst = min(per_year_f1.values())
    score = 0.5 * aut_f1 + 0.3 * latest + 0.2 * worst
    return score, latest, worst, aut_f1


# ═══════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, after):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after
        self._step = 0
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        init = TrainingConstants.WARMUP_INIT_SCALE if self.warmup_epochs > 0 else 1.0
        for i, lr in enumerate(self.base_lrs):
            optimizer.param_groups[i]["lr"] = lr * init

    def step(self):
        self._step += 1
        if self.warmup_epochs > 0 and self._step <= self.warmup_epochs:
            s = self._step / self.warmup_epochs
            for i in range(len(self.base_lrs)):
                self.optimizer.param_groups[i]["lr"] = self.base_lrs[i] * s
            return
        self.after_scheduler.step()

    def state_dict(self):
        return {"step": self._step, "base_lrs": self.base_lrs,
                "warmup_epochs": self.warmup_epochs,
                "after": self.after_scheduler.state_dict()}

    def load_state_dict(self, d):
        self._step = int(d["step"])
        self.base_lrs = list(d["base_lrs"])
        self.warmup_epochs = int(d["warmup_epochs"])
        self.after_scheduler.load_state_dict(d["after"])
        for i, lr in enumerate(self.get_last_lr()):
            self.optimizer.param_groups[i]["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ═══════════════════════════════════════════════════════════════════════
# Collect calibrated probs (used by selective metrics)
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_calibrated_probs(model, calibrator, loader, device, use_amp=False):
    model.eval(); 
    calibrator.eval()
    all_p, all_y = [], []
    for batch in loader:
        if batch is None: continue
        r = prepare_batch(batch, device,
                          skip_graph=False,
                          skip_masks=(model.fusion_mode != "ours" or not getattr(model, "use_alignment_bias", False)))
        if r[2] is None: continue
        graph, masks, y, _, ex, _ = r
        qapi, qg, qa, papi, pg, tids = ex
        with get_amp_context(device, enabled=use_amp):
            logits, _ = model(graph_data=graph,
                              explicit_qs=(qapi, qg, qa, papi, pg),
                              time_ids=tids, masks=masks)
        p = calibrator(logits.float()).detach().cpu().numpy()
        all_p.append(p)
        all_y.append(y.detach().cpu().numpy())
    
    # ★ 空数据保护
    if not all_p or not all_y:
        raise RuntimeError("collect_calibrated_probs: no valid batches collected")
    
    return np.concatenate(all_p), np.concatenate(all_y)


@torch.no_grad()
def collect_calibrated_outputs(model, calibrator, loader, device, use_amp=False):
    model.eval()
    calibrator.eval()
    all_p, all_y, all_risk = [], [], []
    for batch in loader:
        if batch is None:
            continue
        r = prepare_batch(
            batch,
            device,
            skip_graph=False,
            skip_masks=(model.fusion_mode != "ours" or not getattr(model, "use_alignment_bias", False)),
        )
        if r[2] is None:
            continue
        graph, masks, y, _, ex, _ = r
        qapi, qg, qa, papi, pg, tids = ex
        with get_amp_context(device, enabled=use_amp):
            logits, extra = model(
                graph_data=graph,
                explicit_qs=(qapi, qg, qa, papi, pg),
                time_ids=tids,
                masks=masks,
            )
        all_p.append(calibrator(logits.float()).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        risk_values = extra.get("temporal_risk_score") if isinstance(extra, dict) else None
        if isinstance(risk_values, torch.Tensor) and risk_values.numel() == y.size(0):
            all_risk.append(risk_values.detach().float().view(-1).cpu().numpy())

    if not all_p or not all_y:
        raise RuntimeError("collect_calibrated_outputs: no valid batches collected")

    probs = np.concatenate(all_p)
    labels = np.concatenate(all_y)
    risk_count = sum(int(x.shape[0]) for x in all_risk)
    risks = np.concatenate(all_risk) if risk_count == labels.shape[0] else None
    return probs, labels, risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--override", type=str, default="")
    args = ap.parse_args()
    load_dotenv()

    cfg = (deep_update(load_yaml_file(args.base), load_yaml_file(args.override))
           if args.override else load_yaml_file(args.base))
    cfg = validate_full_config(cfg)

    c_data, c_model, c_train, c_loss = cfg["data"], cfg["model"], cfg["train"], cfg["loss"]
    data_root = os.getenv("DATA_ROOT", "")
    set_seed(int(c_train["seed"]))

    out_dir = resolve_path(data_root, c_data["out_dir"])
    exp_name = str(c_train["exp_name"])
    ckpt_dir = os.path.join(out_dir, exp_name, str(c_train["seed"]))
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(ckpt_dir, "train.log")
    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    snap_path = os.path.join(ckpt_dir, "config_snapshot.yaml")

    logger = setup_logger(log_path)
    save_config_snapshot(cfg, snap_path)

    init_metrics_csv(csv_path)

    device = select_device(c_train["device"])
    use_amp = bool(c_train["use_amp"]) and (device.type == "cuda")
    if bool(c_train["use_amp"]) and device.type != "cuda":
        logger.warning(f"AMP requested but unsupported on device={device.type}; disabling AMP.")
    if device.type == "cpu":
        logger.warning(
            "Running on CPU. Graph/API training can be slow. "
            "Check CUDA/MPS availability if this is unexpected."
        )

    # ── Datasets ──
    _fm = str(c_model["fusion_mode"])
    c_api = c_model["api_encoder"]
    c_graph = c_model["graph_encoder"]
    c_temporal = c_model["temporal"]
    c_alignment = c_model["alignment"]
    c_gate = c_model["gate"]

    def _loss_weight(name):
        return float(c_loss[name])

    need_temporal_features = (
        _loss_weight("temporal_proto_current_weight") > 0.0
        or _loss_weight("temporal_proto_future_weight") > 0.0
        or _loss_weight("temporal_risk_calibration_weight") > 0.0
    )

    need_alignment_mask = _fm == "ours" and bool(c_alignment["enabled"])

    drop_graph_behavior_hints = bool(c_graph["drop_extracted_behavior_hints"])
    train_csv_path = resolve_path(data_root, c_data["train_csv"])
    adapt_csv_path = (
        resolve_path(data_root, c_data["adapt_csv"])
        if c_data.get("adapt_csv")
        else None
    )
    val_csv_path = resolve_path(data_root, c_data["val_csv"])
    test_csv_path = resolve_path(data_root, c_data["test_csv"])
    extra_test_specs = c_data.get("extra_tests", []) or []
    extra_test_csv_paths = [
        resolve_path(data_root, spec["test_csv"])
        for spec in extra_test_specs
    ]
    domain_years = build_global_domain_years(
        train_csv_path,
        *( [adapt_csv_path] if adapt_csv_path else [] ),
        val_csv_path,
        test_csv_path,
        *extra_test_csv_paths,
    )

    ds_tr = MultiModalMalwareDataset(
        pt_dir=resolve_path(data_root, c_data["train_pt_dir"]),
        csv_path=train_csv_path,
        is_train=True, robust_aug=False,
        max_api_events_per_sample=c_data["max_api_events_per_sample"],
        fusion_mode=_fm,
        need_alignment_mask=need_alignment_mask,
        domain_years=domain_years,
        drop_graph_behavior_hints=drop_graph_behavior_hints)

    ds_val = MultiModalMalwareDataset(
        pt_dir=resolve_path(data_root, c_data["val_pt_dir"]),
        csv_path=val_csv_path,
        is_train=False, robust_aug=False,
        max_api_events_per_sample=c_data["max_api_events_per_sample"],
        fusion_mode=_fm,
        need_alignment_mask=need_alignment_mask,
        domain_years=domain_years,
        drop_graph_behavior_hints=drop_graph_behavior_hints)

    ds_adapt = None
    if adapt_csv_path is not None:
        ds_adapt = MultiModalMalwareDataset(
            pt_dir=resolve_path(data_root, c_data.get("adapt_pt_dir") or c_data["train_pt_dir"]),
            csv_path=adapt_csv_path,
            is_train=True, robust_aug=False,
            max_api_events_per_sample=c_data["max_api_events_per_sample"],
            fusion_mode=_fm,
            need_alignment_mask=need_alignment_mask,
            domain_years=domain_years,
            drop_graph_behavior_hints=drop_graph_behavior_hints)

    if getattr(ds_tr, "feature_dim", None) != getattr(ds_val, "feature_dim", None):
        raise ValueError(
            "Train/val graph feature dimensions do not match: "
            f"train={getattr(ds_tr, 'feature_dim', None)} "
            f"val={getattr(ds_val, 'feature_dim', None)}. "
            "This usually means old 515-dim .pt files are mixed with new "
            "519-dim Graph-Lite .pt files. Regenerate all splits into a clean directory."
        )
    if ds_adapt is not None and getattr(ds_tr, "feature_dim", None) != getattr(ds_adapt, "feature_dim", None):
        raise ValueError(
            "Historical/adaptation graph feature dimensions do not match: "
            f"historical={getattr(ds_tr, 'feature_dim', None)} "
            f"adapt={getattr(ds_adapt, 'feature_dim', None)}."
        )

    num_classes = int(c_model["num_classes"])
    num_workers = int(c_train["num_workers"])
    batch_size = int(c_train["batch_size"])
    eval_batch_size = int(c_train["eval_batch_size"])
    epochs = int(c_train["epochs"])
    warmup_stage_epochs = (
        int(c_train["warmup_stage_epochs"])
        if _is_multimodal_mode(_fm)
        else 0
    )
    warmup_stage_epochs = max(0, min(warmup_stage_epochs, max(epochs - 1, 0)))

    pin_memory_requested = bool(c_train["pin_memory"])
    pin_memory_enabled = pin_memory_requested and not need_alignment_mask
    if pin_memory_requested and not pin_memory_enabled:
        logger.warning("Disabling pin_memory because alignment-mask batches can fail in PyTorch/PyG pin memory threads.")

    loader_base = dict(num_workers=num_workers, collate_fn=hierarchical_collate_fn,
                       pin_memory=pin_memory_enabled,
                       persistent_workers=bool(c_train["persistent_workers"]))
    if num_workers > 0:
        loader_base["prefetch_factor"] = int(c_train["prefetch_factor"])

    g = torch.Generator(); g.manual_seed(int(c_train["seed"]))
    train_lk = {**loader_base, "worker_init_fn": _worker_init_fn, "generator": g}
    val_lk = {**loader_base, "worker_init_fn": _worker_init_fn, "generator": g}

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, **train_lk)
    dl_adapt = None
    if ds_adapt is not None:
        dl_adapt, n_adapt, n_replay = _build_continual_adaptation_loader(
            ds_tr,
            ds_adapt,
            cfg,
            batch_size,
            train_lk,
        )
        logger.info(
            "Continual adaptation loader | "
            f"adapt_samples={n_adapt} replay_samples={n_replay} "
            f"adapt_ratio={float(c_train.get('adaptation_ratio', 1.0)):.3f} "
            f"replay_ratio={float(c_train.get('replay_ratio', 0.25)):.3f}"
        )

    dl_val = DataLoader(ds_val, batch_size=eval_batch_size, shuffle=False, **val_lk)
    val_year_loaders = build_year_subset_loaders(ds_val, eval_batch_size, val_lk)

    # ── Model ──
    inferred_graph_dim = int(getattr(ds_tr, "feature_dim", TrainingConstants.IN_FEAT_DIM))
    configured_graph_dim = c_model.get("in_feat_dim", None)
    if configured_graph_dim is not None and int(configured_graph_dim) != inferred_graph_dim:
        raise ValueError(
            "Configured model.in_feat_dim does not match extracted graph features: "
            f"configured={configured_graph_dim} extracted={inferred_graph_dim}. "
            "Remove model.in_feat_dim from YAML or regenerate compatible .pt files."
        )
    graph_in_feat_dim = inferred_graph_dim
    c_model["in_feat_dim"] = graph_in_feat_dim
    model = MalwareModelWithXAttn(
        num_classes=num_classes,
        api_emb_dim=TrainingConstants.API_EMB_DIM,
        graph_emb_dim=TrainingConstants.GRAPH_EMB_DIM,
        align_dim=TrainingConstants.ALIGN_DIM,
        max_nodes_gnn=int(c_model["max_nodes_gnn"]),
        max_xattn_nodes=int(c_model["max_xattn_nodes"]),
        in_feat_dim=graph_in_feat_dim,
        use_temporal_regularization=need_temporal_features,
        xattn_heads=TrainingConstants.XATTN_HEADS,
        fusion_mode=_fm,
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
        use_alignment_drift_guidance=bool(c_alignment["drift_guided"]),
        use_quality_gate_inputs=bool(c_gate["quality_inputs"]),
        use_drift_gate=bool(c_gate["drift_inputs"]),
        gate_mode=str(c_gate["mode"]),
        gate_detach=bool(c_gate["detach"]),
        late_fusion_api_weight=0.5,
        temporal_num_domains=len(domain_years),
        temporal_prototype_momentum=float(c_temporal["prototype_momentum"]),
        temporal_prototype_clusters=int(c_temporal["prototype_clusters"]),
        temporal_drift_velocity_scale=float(c_loss["temporal_proto_velocity_scale"]),
        temporal_drift_min_history=int(c_loss["temporal_proto_min_history"]),
        use_future_temporal_drift=(
            bool(c_temporal["use_future_drift"])
            and (
                _loss_weight("temporal_proto_future_weight") > 0.0
                or _loss_weight("temporal_risk_calibration_weight") > 0.0
            )
        ),
        use_temporal_risk_calibration=(_loss_weight("temporal_risk_calibration_weight") > 0.0),
    ).to(device)

    start_epoch, best_score, no_improve = 1, float("-inf"), 0

    # ── Optimizer / Scheduler ──
    base_lr = float(c_train["lr"])
    wd = float(c_train["weight_decay"])
    warmup = int(c_train["warmup_epochs"])
    eta_min = float(c_train["eta_min"])

    def _build_optimizer_scheduler(remaining_epochs: int):
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
        after = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(int(remaining_epochs) - warmup, 1),
            eta_min=eta_min,
        )
        return opt, WarmupCosineScheduler(opt, warmup, after)

    optimizer, scheduler = _build_optimizer_scheduler(epochs)

    # ── Loss ──
    criterion = nn.CrossEntropyLoss(label_smoothing=float(c_train["label_smoothing"]))
    scaler = build_grad_scaler(device=device, enabled=use_amp)

    patience = int(c_train["patience"])
    min_delta = float(c_train["min_delta"])

    logger.info(
        f"Start | device={device} amp={use_amp} train={len(ds_tr)} val={len(ds_val)}"
    )
    logger.info(
        f"Loader | train_batches={len(dl_tr)} val_batches={len(dl_val)} "
        f"batch_size={batch_size} "
        f"eval_batch_size={eval_batch_size} grad_accum={int(c_train['grad_accum_steps'])} "
        f"alignment_mask={need_alignment_mask}"
    )
    logger.info(
        f"Training protocol | warmup_stage_epochs={warmup_stage_epochs} "
        f"historical_epochs={int(c_train.get('historical_epochs', epochs))} "
        f"adaptation_epochs={int(c_train.get('adaptation_epochs', 0))} "
        f"stage1_aux={float(c_loss['stage1_branch_aux_weight']):.3f} "
        f"main_aux={float(c_loss['branch_aux_weight']):.3f}"
    )
    logger.info(f"Temporal domains | years={domain_years}")

    # ═══════════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════════
    last_stage = None
    last_continual_phase = None
    historical_epochs = min(max(int(c_train.get("historical_epochs", epochs)), 0), epochs)
    for epoch in range(start_epoch, epochs + 1):
        stage_name = "warmup" if epoch <= warmup_stage_epochs else "main"
        continual_phase = "adaptation" if (dl_adapt is not None and epoch > historical_epochs) else "historical"
        if stage_name != last_stage:
            if hasattr(model, "set_training_stage"):
                model.set_training_stage(stage_name)
            if last_stage == "warmup" and stage_name == "main":
                remaining_epochs = epochs - epoch + 1
                optimizer, scheduler = _build_optimizer_scheduler(remaining_epochs)
                scaler = build_grad_scaler(device=device, enabled=use_amp)
                best_score = float("-inf")
                no_improve = 0
                logger.info(
                    "Stage transition: warmup -> main | optimizer/scheduler reset; "
                    "best checkpoint selection restarts for the full method."
                )
            logger.info(f"Training stage: {stage_name}")
            last_stage = stage_name
        if continual_phase != last_continual_phase:
            if last_continual_phase == "historical" and continual_phase == "adaptation":
                remaining_epochs = epochs - epoch + 1
                optimizer, scheduler = _build_optimizer_scheduler(remaining_epochs)
                scaler = build_grad_scaler(device=device, enabled=use_amp)
                best_score = float("-inf")
                no_improve = 0
                logger.info(
                    "Continual phase transition: historical -> adaptation | "
                    "training on recent-year samples plus historical replay; "
                    "optimizer/scheduler and best checkpoint selection reset."
                )
            logger.info(f"Continual phase: {continual_phase}")
            last_continual_phase = continual_phase

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_loss_cfg = _stage_loss_cfg(c_loss, stage_name)
        epoch_loader = dl_adapt if continual_phase == "adaptation" else dl_tr

        (tr_loss, tr_f1, tr_acc, tr_cls, tr_temp, tr_align, tr_branch_aux, tr_risk_calib,
         tr_proto_current, tr_proto_future, tr_gate_oracle, tr_drift,
         tr_temporal_drift, tr_temporal_risk, tr_align_temporal_drift,
         tr_align_drift_gate, tr_semantic_drift_gate,
         tr_gate_temporal_drift, tr_gate_disagreement, tr_gate_entropy,
         tr_gate_api, tr_gate_graph, tr_gate_joint) = train_one_epoch(
            model, epoch_loader, optimizer, scaler, criterion, device,
            epoch, epochs, loss_cfg=epoch_loss_cfg, logger=logger, use_amp=use_amp,
            grad_accum_steps=int(c_train["grad_accum_steps"]))

        (val_loss, val_cls, val_f1, val_acc, val_softmax_aurc, val_hybrid_aurc,
         val_gate_api, val_gate_graph, val_gate_joint,
         val_temporal_drift, val_temporal_risk, val_temporal_risk_error_auc,
         val_temporal_risk_aurc, val_temporal_risk_gap, val_align_temporal_drift,
         val_align_drift_gate, val_semantic_drift_gate,
         val_gate_temporal_drift, val_gate_disagreement, val_gate_entropy) = eval_one_epoch(
            model, dl_val, criterion, device, epoch, num_epochs=epochs,
            use_amp=use_amp, logger=logger)

        # Temporal selection
        selection_score, latest_f1, worst_f1, aut_f1 = val_f1, val_f1, val_f1, val_f1
        if val_year_loaders:
            ym = evaluate_temporal_windows(
                model, val_year_loaders, criterion, device, epoch, epochs,
                use_amp=use_amp, logger=logger, tag="val")
            selection_score, latest_f1, worst_f1, aut_f1 = compute_temporal_selection_score(ym)
            logger.info(f"[Epoch {epoch:03d}] sel={selection_score:.4f} "
                        f"(AUT={aut_f1:.4f} latest={latest_f1:.4f} worst={worst_f1:.4f})")

        if stage_name == "warmup":
            improved = False
        else:
            improved = selection_score > (best_score + min_delta)
        if improved:
            best_score = selection_score
            no_improve = 0
        else:
            no_improve += 0 if stage_name == "warmup" else 1

        last_p = os.path.join(ckpt_dir, f"last_{exp_name}.pt")
        best_p = os.path.join(ckpt_dir, f"best_{exp_name}.pt")
        save_checkpoint(last_p, epoch, model, optimizer, scheduler, scaler,
                        best_score, no_improve, cfg)
        if improved:
            save_checkpoint(best_p, epoch, model, optimizer, scheduler, scaler,
                            best_score, no_improve, cfg)

        append_metrics_csv(csv_path, dict(
            epoch=epoch, stage=stage_name,
            train_loss=tr_loss, train_f1=tr_f1, train_acc=tr_acc,
            train_cls=tr_cls, train_temporal=tr_temp,
            train_proto_current=tr_proto_current,
            train_proto_future=tr_proto_future,
            train_alignment=tr_align,
            train_branch_aux=tr_branch_aux,
            train_risk_calib=tr_risk_calib,
            train_gate_oracle=tr_gate_oracle,
            train_drift=tr_drift,
            train_temporal_drift=tr_temporal_drift,
            train_temporal_risk=tr_temporal_risk,
            train_alignment_temporal_drift=tr_align_temporal_drift,
            train_alignment_drift_gate=tr_align_drift_gate,
            train_semantic_drift_gate=tr_semantic_drift_gate,
            train_gate_temporal_drift=tr_gate_temporal_drift,
            train_gate_disagreement=tr_gate_disagreement,
            train_gate_entropy=tr_gate_entropy,
            train_gate_api=tr_gate_api,
            train_gate_graph=tr_gate_graph,
            train_gate_joint=tr_gate_joint,
            val_loss=val_loss, val_cls_loss=val_cls, val_f1=val_f1, val_acc=val_acc,
            val_softmax_aurc=val_softmax_aurc,
            val_hybrid_aurc=val_hybrid_aurc,
            val_temporal_drift=val_temporal_drift,
            val_temporal_risk=val_temporal_risk,
            val_temporal_risk_error_auc=val_temporal_risk_error_auc,
            val_temporal_risk_aurc=val_temporal_risk_aurc,
            val_temporal_risk_gap=val_temporal_risk_gap,
            val_alignment_temporal_drift=val_align_temporal_drift,
            val_alignment_drift_gate=val_align_drift_gate,
            val_semantic_drift_gate=val_semantic_drift_gate,
            val_gate_temporal_drift=val_gate_temporal_drift,
            val_gate_disagreement=val_gate_disagreement,
            val_gate_entropy=val_gate_entropy,
            val_gate_api=val_gate_api,
            val_gate_graph=val_gate_graph,
            val_gate_joint=val_gate_joint,
            lr=current_lr,
            best_score=(best_score if best_score != float("-inf") else 0.0),
            is_best=int(improved), no_improve=no_improve,
            selection_score=selection_score, latest_f1=latest_f1,
            worst_f1=worst_f1, aut_f1=aut_f1))

        scheduler.step()

        logger.info(
            f"[Epoch {epoch:03d}] train: loss={tr_loss:.4f} f1={tr_f1:.4f} | "
            f"val: loss={val_loss:.4f} f1={val_f1:.4f} "
            f"softmax_aurc={val_softmax_aurc:.4f} hybrid_aurc={val_hybrid_aurc:.4f} | "
            f"sel={selection_score:.4f}")
        temporal_memory = getattr(model, "temporal_prototype_memory", None)
        if temporal_memory is not None and hasattr(temporal_memory, "occupancy_stats"):
            proto_stats = temporal_memory.occupancy_stats()
            logger.info(
                f"[Epoch {epoch:03d}] proto_clusters: "
                f"cells={int(proto_stats['occupied_cells'])}/{int(proto_stats['total_cells'])} "
                f"clusters={int(proto_stats['occupied_clusters'])} "
                f"initialized={int(proto_stats.get('initialized_clusters', 0))} "
                f"inherited={int(proto_stats.get('inherited_clusters', 0))} "
                f"mean_per_cell={proto_stats['mean_clusters_per_seen_cell']:.2f}"
            )

        if stage_name != "warmup" and no_improve >= patience:
            logger.info(f"Early stop: no_improve={no_improve}")
            break

    logger.info(f"Training finished | best_score={best_score:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Post-training evaluation
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("🎯 Temperature scaling calibration")

    calibrator = TemperatureScaling().to(device)
    best_p = os.path.join(ckpt_dir, f"best_{exp_name}.pt")
    if os.path.exists(best_p):
        model.load_state_dict(
            torch.load(best_p, map_location=device, weights_only=False)["state_dict"])

    cal_loader = dl_val
    if val_year_loaders:
        ly = sorted(val_year_loaders.keys())[-1]
        cal_loader = val_year_loaders[ly]
        logger.info(f"Calibrating on latest year: {ly}")
    # train.py 中 calibrator.fit 调用，增加 use_amp 参数
    calibrator.fit(model, cal_loader, device, max_iter=50, lr=0.01, use_amp=use_amp)
    torch.save(calibrator.state_dict(), os.path.join(ckpt_dir, f"calibrator_{exp_name}.pt"))

    # ── Test ──
    test_pt = resolve_path(data_root, c_data["test_pt_dir"])
    test_csv = resolve_path(data_root, c_data["test_csv"])
    if not (os.path.exists(test_pt) and os.path.exists(test_csv)):
        raise FileNotFoundError(f"Test set not found: pt_dir={test_pt}, csv={test_csv}")

    ds_test = MultiModalMalwareDataset(
        pt_dir=test_pt, csv_path=test_csv, is_train=False, robust_aug=False,
        max_api_events_per_sample=c_data["max_api_events_per_sample"],
        fusion_mode=_fm,
        need_alignment_mask=need_alignment_mask,
        domain_years=domain_years,
        drop_graph_behavior_hints=drop_graph_behavior_hints)
    if getattr(ds_test, "feature_dim", graph_in_feat_dim) != graph_in_feat_dim:
        raise ValueError(
            "Train/test graph feature dimensions do not match: "
            f"train={graph_in_feat_dim} test={getattr(ds_test, 'feature_dim', None)}. "
            "Regenerate train/val/test with the same extract configuration."
        )
    # Test loader: disable pin_memory to avoid CUDA state corruption
    # after calibration step (calibration leaves GPU memory in a fragmented state
    # that can cause "invalid argument" errors in pin_memory).
    test_lk = {k: v for k, v in val_lk.items()}
    test_lk["pin_memory"] = False
    test_lk["persistent_workers"] = False
    dl_test = DataLoader(ds_test, batch_size=eval_batch_size, shuffle=False, **test_lk)
    test_year_loaders = build_year_subset_loaders(ds_test, eval_batch_size, test_lk)

    test_loss, test_cls, test_f1, test_acc, test_softmax_aurc, test_hybrid_aurc, *_ = eval_one_epoch(
        model, dl_test, criterion, device, epoch=0, num_epochs=1,
        use_amp=use_amp, logger=logger)
    logger.info(f"🏆 TEST: loss={test_loss:.4f} F1={test_f1:.4f} "
                f"Acc={test_acc:.4f} softmax_AURC={test_softmax_aurc:.4f} "
                f"hybrid_AURC={test_hybrid_aurc:.4f}")

    # ── Selective classification on calibrated probs ──
    test_probs, test_labels, test_risks = collect_calibrated_outputs(model, calibrator, dl_test, device, use_amp)
    test_preds = test_probs.argmax(axis=1)
    test_conf = test_probs.max(axis=1)
    test_correct = (test_preds == test_labels).astype(np.float64)
    test_hybrid_conf = _hybrid_confidence(test_conf, test_risks) if test_risks is not None else test_conf
    logger.info(
        f"Selective calibrated summary: softmax_AURC={aurc(test_conf,test_correct):.4f} "
        f"hybrid_AURC={aurc(test_hybrid_conf,test_correct):.4f}"
    )
    logger.info(
        f"🎯 Selective (calibrated): AURC={aurc(test_conf,test_correct):.4f} "
        f"softmax_E-AURC={eaurc(test_conf,test_correct):.4f} "
        f"hybrid_AURC={aurc(test_hybrid_conf,test_correct):.4f} | "
        f"risk@cov0.8={risk_at_coverage(test_conf,test_correct,0.8):.4f} "
        f"risk@cov0.9={risk_at_coverage(test_conf,test_correct,0.9):.4f} | "
        f"cov@risk≤1%={coverage_at_risk(test_conf,test_correct,0.01):.4f} "
        f"cov@risk≤5%={coverage_at_risk(test_conf,test_correct,0.05):.4f}")

    # ── Per-year AUT ──
    if test_year_loaders:
        yearly = evaluate_temporal_windows(
            model, test_year_loaders, criterion, device, epoch=0, num_epochs=1,
            use_amp=use_amp, logger=logger, tag="test")
        aut_suite = compute_aut_suite(yearly)
        logger.info("🕐 Temporal AUT: " + " | ".join(
            f"{k}={v:.4f}" for k, v in aut_suite.items()))
        aut_path = os.path.join(ckpt_dir, f"aut_{exp_name}.csv")
        with open(aut_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            keys = list(next(iter(yearly.values())).keys())
            w.writerow(["year"] + keys)
            for y in sorted(yearly.keys()):
                w.writerow([y] + [f"{yearly[y][k]:.6f}" for k in keys])
            w.writerow(["AUT"] + [f"{aut_suite.get(f'AUT_{k}', 0):.6f}" for k in keys])

    def _safe_eval_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))

    for spec in extra_test_specs:
        extra_name = str(spec["name"])
        extra_pt = resolve_path(data_root, spec["test_pt_dir"])
        extra_csv = resolve_path(data_root, spec["test_csv"])
        if not (os.path.exists(extra_pt) and os.path.exists(extra_csv)):
            raise FileNotFoundError(
                f"Extra test set not found for {extra_name}: pt_dir={extra_pt}, csv={extra_csv}"
            )

        logger.info("=" * 60)
        logger.info(f"Extra test evaluation: {extra_name}")
        ds_extra = MultiModalMalwareDataset(
            pt_dir=extra_pt, csv_path=extra_csv, is_train=False, robust_aug=False,
            max_api_events_per_sample=c_data["max_api_events_per_sample"],
            fusion_mode=_fm,
            need_alignment_mask=need_alignment_mask,
            domain_years=domain_years,
            drop_graph_behavior_hints=drop_graph_behavior_hints)
        if getattr(ds_extra, "feature_dim", graph_in_feat_dim) != graph_in_feat_dim:
            raise ValueError(
                "Train/extra-test graph feature dimensions do not match: "
                f"train={graph_in_feat_dim} extra={getattr(ds_extra, 'feature_dim', None)}. "
                "Regenerate all splits with the same extract configuration."
            )

        extra_lk = {k: v for k, v in val_lk.items()}
        extra_lk["pin_memory"] = False
        extra_lk["persistent_workers"] = False
        dl_extra = DataLoader(ds_extra, batch_size=eval_batch_size, shuffle=False, **extra_lk)
        extra_year_loaders = build_year_subset_loaders(ds_extra, eval_batch_size, extra_lk)

        extra_loss, extra_cls, extra_f1, extra_acc, extra_softmax_aurc, extra_hybrid_aurc, *_ = eval_one_epoch(
            model, dl_extra, criterion, device, epoch=0, num_epochs=1,
            use_amp=use_amp, logger=logger)
        logger.info(
            f"EXTRA_TEST[{extra_name}]: loss={extra_loss:.4f} F1={extra_f1:.4f} "
            f"Acc={extra_acc:.4f} softmax_AURC={extra_softmax_aurc:.4f} "
            f"hybrid_AURC={extra_hybrid_aurc:.4f}"
        )

        extra_probs, extra_labels, extra_risks = collect_calibrated_outputs(model, calibrator, dl_extra, device, use_amp)
        extra_preds = extra_probs.argmax(axis=1)
        extra_conf = extra_probs.max(axis=1)
        extra_correct = (extra_preds == extra_labels).astype(np.float64)
        extra_hybrid_conf = _hybrid_confidence(extra_conf, extra_risks) if extra_risks is not None else extra_conf
        logger.info(
            f"Selective[{extra_name}] calibrated: softmax_AURC={aurc(extra_conf,extra_correct):.4f} "
            f"softmax_E-AURC={eaurc(extra_conf,extra_correct):.4f} "
            f"hybrid_AURC={aurc(extra_hybrid_conf,extra_correct):.4f} | "
            f"risk@cov0.8={risk_at_coverage(extra_conf,extra_correct,0.8):.4f} "
            f"risk@cov0.9={risk_at_coverage(extra_conf,extra_correct,0.9):.4f} | "
            f"cov@risk<=1%={coverage_at_risk(extra_conf,extra_correct,0.01):.4f} "
            f"cov@risk<=5%={coverage_at_risk(extra_conf,extra_correct,0.05):.4f}"
        )

        if extra_year_loaders:
            extra_yearly = evaluate_temporal_windows(
                model, extra_year_loaders, criterion, device, epoch=0, num_epochs=1,
                use_amp=use_amp, logger=logger, tag=f"extra_{_safe_eval_name(extra_name)}")
            extra_aut_suite = compute_aut_suite(extra_yearly)
            logger.info(f"Extra Temporal AUT[{extra_name}]: " + " | ".join(
                f"{k}={v:.4f}" for k, v in extra_aut_suite.items()))
            extra_aut_path = os.path.join(
                ckpt_dir,
                f"aut_{exp_name}_{_safe_eval_name(extra_name)}.csv",
            )
            with open(extra_aut_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                keys = list(next(iter(extra_yearly.values())).keys())
                w.writerow(["year"] + keys)
                for y in sorted(extra_yearly.keys()):
                    w.writerow([y] + [f"{extra_yearly[y][k]:.6f}" for k in keys])
                w.writerow(["AUT"] + [f"{extra_aut_suite.get(f'AUT_{k}', 0):.6f}" for k in keys])


if __name__ == "__main__":
    main()
