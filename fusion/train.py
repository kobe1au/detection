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
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
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
        "api_encoder", "graph_encoder", "alignment", "gate",
        "in_feat_dim",
    }
    allowed_api = {"type", "num_hash_buckets", "type_vocab_size", "max_seq_len", "layers", "heads"}
    allowed_graph = {"type", "hidden", "heads", "layers", "use_behavior_hint", "drop_extracted_behavior_hints"}
    allowed_alignment = {"enabled", "adaptive_bias", "penalty_scale", "bonus_scale", "context_scale"}
    allowed_gate = {"mode", "quality_inputs", "uncertainty_inputs", "detach"}
    allowed_train = {
        "exp_name", "seed", "device", "use_amp", "epochs", "batch_size",
        "eval_batch_size", "grad_accum_steps", "num_workers", "pin_memory",
        "persistent_workers", "prefetch_factor", "lr", "weight_decay",
        "warmup_epochs", "eta_min", "label_smoothing", "patience",
        "min_delta", "warmup_stage_epochs",
        "historical_epochs", "adaptation_epochs", "adaptation_ratio", "replay_ratio",
    }
    allowed_loss = {
        "semantic_alignment_weight",
        "branch_aux_weight",
        "stage1_branch_aux_weight",
        "class_aware_alignment_same_class_weight",
        "class_aware_alignment_temperature",
        "gate_oracle_weight",
        "gate_oracle_temperature",
        "gate_oracle_smoothing",
        "gate_oracle_start_epoch",
        "gate_oracle_start_phase",
        "gate_oracle_adaptation_only",
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
    _reject_unknown("model.alignment", cfg["model"].get("alignment", {}), allowed_alignment)
    _reject_unknown("model.gate", cfg["model"].get("gate", {}), allowed_gate)
    _reject_unknown("train", cfg["train"], allowed_train)
    _reject_unknown("loss", cfg["loss"], allowed_loss)

    _require_keys("data", cfg["data"], allowed_data, optional={"extra_tests", "adapt_pt_dir", "adapt_csv"})
    _require_keys("model", cfg["model"], allowed_model, optional={"in_feat_dim"})
    _require_keys("model.api_encoder", cfg["model"]["api_encoder"], allowed_api)
    _require_keys("model.graph_encoder", cfg["model"]["graph_encoder"], allowed_graph)
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
            "gate_oracle_smoothing",
            "gate_oracle_start_epoch",
            "gate_oracle_start_phase",
            "gate_oracle_adaptation_only",
        },
    )

    nc = int(cfg["model"].get("num_classes", 2))
    if nc < 2:
        raise ValueError(f"num_classes must be >= 2, got {nc}")

    forbidden_fragments = ("resume", "finetune", "tta", "conformal", "robustness")
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
    "epoch", "stage", "continual_phase",
    "train_loss", "train_f1", "train_acc",
    "train_cls", "train_alignment", "train_branch_aux", "train_gate_oracle",
    "train_uncertainty", "train_gate_disagreement", "train_gate_entropy",
    "train_gate_api", "train_gate_graph", "train_gate_joint",
    "val_loss", "val_cls_loss", "val_f1", "val_acc", "val_softmax_aurc",
    "val_uncertainty", "val_gate_disagreement", "val_gate_entropy",
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

def _binary_detection_metrics(labels, probs, preds=None):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        malware_score = probs[:, 1]
        pred_labels = probs.argmax(axis=1) if preds is None else np.asarray(preds, dtype=np.int64).reshape(-1)
    else:
        malware_score = probs.reshape(-1)
        pred_labels = (malware_score >= 0.5).astype(np.int64) if preds is None else np.asarray(preds, dtype=np.int64).reshape(-1)

    if labels.shape[0] == 0 or pred_labels.shape[0] != labels.shape[0]:
        return {
            "malware_recall": 0.0,
            "fnr": 0.0,
            "auroc": 0.0,
            "auprc": 0.0,
        }

    malware_recall = float(recall_score(labels, pred_labels, pos_label=1, zero_division=0))
    has_both_classes = np.unique(labels).size == 2
    auroc_v = float(roc_auc_score(labels, malware_score)) if has_both_classes else 0.0
    auprc_v = float(average_precision_score(labels, malware_score)) if has_both_classes else 0.0
    return {
        "malware_recall": malware_recall,
        "fnr": 1.0 - malware_recall,
        "auroc": auroc_v,
        "auprc": auprc_v,
    }


def _gate_diagnostics(gate_weights, correct=None, q_api=None, q_graph=None):
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

def _is_multimodal_mode(fusion_mode: str) -> bool:
    return str(fusion_mode) in {"concat", "cross_attention", "late_fusion", "ours"}


def _stage_loss_cfg(loss_cfg: dict, stage: str) -> dict:
    cfg = copy.deepcopy(loss_cfg)
    if stage == "warmup":
        cfg["semantic_alignment_weight"] = 0.0
        cfg["branch_aux_weight"] = float(cfg["stage1_branch_aux_weight"])
    return cfg


def _balanced_sample_indices(
    dataset,
    ratio: float,
    seed: int,
    min_per_group: int = 1,
    group_by_year: bool = False,
):
    ratio = float(ratio)
    if ratio >= 1.0:
        return list(range(len(dataset)))
    if ratio <= 0.0:
        return []
    by_group: dict[tuple[int, int] | int, list[int]] = {}
    labels = getattr(dataset, "labels", None)
    sample_sids = getattr(dataset, "sample_sids", None)
    for idx in range(len(dataset)):
        label = None
        year = None
        if labels is not None and sample_sids is not None and idx < len(sample_sids):
            label = int(labels[sample_sids[idx]])
            sid_to_year = getattr(dataset, "sid_to_year", None)
            if sid_to_year is not None:
                year = int(sid_to_year[sample_sids[idx]])
        else:
            sample = getattr(dataset, "samples", [])[idx]
            if len(sample) >= 2:
                label = int(sample[1])
            if len(sample) >= 4:
                year = int(sample[3])
        if label is None:
            continue
        group = (int(year), int(label)) if group_by_year and year is not None else int(label)
        by_group.setdefault(group, []).append(idx)

    rng = random.Random(seed)
    selected = []
    for _, indices in sorted(by_group.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        keep = int(round(len(shuffled) * ratio))
        keep = max(min_per_group, keep) if shuffled else 0
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
    replay_indices = _balanced_sample_indices(
        historical_dataset,
        replay_ratio,
        seed + 1009,
        min_per_group=1,
        group_by_year=True,
    )
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    epoch,
    num_epochs,
    loss_cfg=None,
    logger=None,
    use_amp=False,
    grad_accum_steps=1,
):
    model.train()
    loss_cfg = loss_cfg or {}
    grad_accum_steps = max(1, int(grad_accum_steps))

    total_loss = 0.0
    total_samples = 0
    total_valid = 0
    skipped = 0
    failed = 0
    accum_steps = 0
    optimizer_steps = 0

    sum_cls = 0.0
    sum_align = 0.0
    sum_branch_aux = 0.0
    sum_gate_oracle = 0.0

    sum_uncertainty = 0.0
    num_uncertainty = 0

    sum_gate_disagreement = 0.0
    num_gate_disagreement = 0

    sum_gate_entropy = 0.0
    num_gate_entropy = 0

    sum_wi = 0.0
    sum_wg = 0.0
    sum_wj = 0.0
    num_w = 0

    sum_wi_std = 0.0
    sum_wg_std = 0.0
    sum_wj_std = 0.0
    num_w_std = 0

    sum_align_cov = 0.0
    num_align_cov = 0

    sum_align_density = 0.0
    num_align_density = 0

    all_preds = []
    all_labels = []

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
            skipped += 1
            continue

        try:
            r = prepare_batch(
                batch,
                device,
                skip_graph=False,
                skip_masks=(
                    model.fusion_mode != "ours"
                    or not getattr(model, "use_alignment_bias", False)
                ),
            )

            if r[2] is None:
                skipped += 1
                failed += r[-1]
                continue

            graph, masks, y, _, ex, nf = r
            failed += nf
            qapi, qg, qa, papi, pg, tids = ex

            bs = y.size(0)
            total_valid += bs

            with get_amp_context(device, enabled=use_amp):
                logits, extra = model(
                    graph_data=graph,
                    y=y,
                    explicit_qs=(qapi, qg, qa, papi, pg),
                    time_ids=tids,
                    masks=masks,
                )

                loss_out = compute_total_loss(
                    logits,
                    extra,
                    y,
                    criterion,
                    loss_cfg,
                    epoch=epoch,
                    total_epochs=num_epochs,
                )

                # Compatible with both:
                #   old: return total, cls, temporal_zero, align
                #   new: return total, cls, align
                if len(loss_out) == 4:
                    loss, l_cls, _, l_align = loss_out
                elif len(loss_out) == 3:
                    loss, l_cls, l_align = loss_out
                else:
                    raise RuntimeError(
                        f"compute_total_loss returned unexpected tuple length: {len(loss_out)}"
                    )

                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            accum_steps += 1

            if accum_steps >= grad_accum_steps:
                _optimizer_step()

            loss_v = float(loss.item())
            total_loss += loss_v * bs
            total_samples += bs

            sum_cls += float(l_cls.item()) * bs
            sum_align += float(l_align.item()) * bs

            branch_aux, ok_branch_aux = _extract_loss_component(extra, "branch_aux")
            if ok_branch_aux:
                sum_branch_aux += branch_aux * bs

            gate_oracle, ok_gate_oracle = _extract_loss_component(extra, "gate_oracle")
            if ok_gate_oracle:
                sum_gate_oracle += gate_oracle * bs

            uncertainty, ok_uncertainty = _extract_extra_mean(extra, "uncertainty_score")
            if ok_uncertainty:
                sum_uncertainty += uncertainty * bs
                num_uncertainty += bs

            gate_disagreement, ok_gate_disagreement = _extract_extra_mean(extra, "gate_disagreement")
            if ok_gate_disagreement:
                sum_gate_disagreement += gate_disagreement * bs
                num_gate_disagreement += bs

            gate_entropy, ok_gate_entropy = _extract_extra_mean(extra, "gate_entropy")
            if ok_gate_entropy:
                sum_gate_entropy += gate_entropy * bs
                num_gate_entropy += bs

            wi, wg, wj, ok_w = _extract_gate_weights(extra)
            if ok_w:
                sum_wi += wi
                sum_wg += wg
                sum_wj += wj
                num_w += 1

            wi_std, wg_std, wj_std, ok_std = _extract_gate_weight_stds(extra)
            if ok_std:
                sum_wi_std += wi_std
                sum_wg_std += wg_std
                sum_wj_std += wj_std
                num_w_std += 1

            align_cov, ok_align_cov = _extract_extra_mean(extra, "alignment_coverage")
            if ok_align_cov:
                sum_align_cov += align_cov
                num_align_cov += 1

            align_density, ok_align_density = _extract_extra_mean(extra, "alignment_density")
            if ok_align_density:
                sum_align_density += align_density
                num_align_density += 1

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

            pbar.set_postfix(
                loss=f"{loss_v:.4f}",
                cls=f"{float(l_cls.item()):.4f}",
                align=f"{float(l_align.item()):.4f}",
                aux=f"{branch_aux:.4f}",
                gorl=f"{gate_oracle:.4f}",
                unc=f"{uncertainty:.4f}",
                accum=f"{accum_steps}/{grad_accum_steps}",
            )

            del graph, masks, y, ex, logits, extra, loss, loss_for_backward

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if logger:
                    if device.type == "cuda":
                        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
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

                skipped += 1
                continue

            raise

    _optimizer_step()

    if not all_labels:
        if logger:
            logger.warning(f"[train][epoch {epoch}] No valid predictions collected!")

        return (
            0.0,  # avg loss
            0.0,  # f1
            0.0,  # acc
            0.0,  # cls
            0.0,  # align
            0.0,  # branch aux
            0.0,  # gate oracle
            0.0,  # uncertainty
            0.0,  # gate disagreement
            0.0,  # gate entropy
            0.0,  # gate api
            0.0,  # gate graph
            0.0,  # gate joint
        )

    n = max(total_samples, 1)
    avg_loss = total_loss / n
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    avg_uncertainty = sum_uncertainty / max(num_uncertainty, 1)
    avg_gate_disagreement = sum_gate_disagreement / max(num_gate_disagreement, 1)
    avg_gate_entropy = sum_gate_entropy / max(num_gate_entropy, 1)

    avg_gate_api = sum_wi / max(num_w, 1)
    avg_gate_graph = sum_wg / max(num_w, 1)
    avg_gate_joint = sum_wj / max(num_w, 1)

    if logger:
        logger.info(
            f"[train][epoch {epoch}] "
            f"valid={total_valid} failed={failed} skipped={skipped} "
            f"optim_steps={optimizer_steps} grad_accum={grad_accum_steps} "
            f"avg_w=({avg_gate_api:.3f},{avg_gate_graph:.3f},{avg_gate_joint:.3f}) "
            f"std_w=({sum_wi_std / max(num_w_std, 1):.3f},"
            f"{sum_wg_std / max(num_w_std, 1):.3f},"
            f"{sum_wj_std / max(num_w_std, 1):.3f}) "
            f"uncertainty={avg_uncertainty:.3f} "
            f"gate_dis={avg_gate_disagreement:.3f} "
            f"gate_ent={avg_gate_entropy:.3f} "
            f"align_cov={sum_align_cov / max(num_align_cov, 1):.3f} "
            f"align_density={sum_align_density / max(num_align_density, 1):.4f}"
        )

    return (
        avg_loss,
        f1,
        acc,
        sum_cls / n,
        sum_align / n,
        sum_branch_aux / n,
        sum_gate_oracle / n,
        avg_uncertainty,
        avg_gate_disagreement,
        avg_gate_entropy,
        avg_gate_api,
        avg_gate_graph,
        avg_gate_joint,
    )

# ═══════════════════════════════════════════════════════════════════════
# eval_one_epoch (returns AURC instead of keep_f1/corrected_acc)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    criterion,
    device,
    epoch,
    num_epochs=1,
    use_amp=False,
    logger=None,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_valid = 0
    skipped = 0
    failed = 0

    sum_cls = 0.0

    sum_uncertainty = 0.0
    num_uncertainty = 0

    sum_gate_disagreement = 0.0
    num_gate_disagreement = 0

    sum_gate_entropy = 0.0
    num_gate_entropy = 0

    sum_wi = 0.0
    sum_wg = 0.0
    sum_wj = 0.0
    num_w = 0

    sum_wi_std = 0.0
    sum_wg_std = 0.0
    sum_wj_std = 0.0
    num_w_std = 0

    sum_align_cov = 0.0
    num_align_cov = 0

    sum_align_density = 0.0
    num_align_density = 0

    all_preds = []
    all_labels = []
    all_confs = []

    all_gate_weights = []
    all_gate_correct = []
    all_gate_qapi = []
    all_gate_qgraph = []

    times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in tqdm(loader, desc=f"Eval Epoch {epoch}", dynamic_ncols=True):
        if batch is None:
            skipped += 1
            continue

        r = prepare_batch(
            batch,
            device,
            skip_graph=False,
            skip_masks=(
                model.fusion_mode != "ours"
                or not getattr(model, "use_alignment_bias", False)
            ),
        )

        if r[2] is None:
            skipped += 1
            failed += r[-1]
            continue

        graph, masks, y, _, ex, nf = r
        failed += nf
        qapi, qg, qa, papi, pg, tids = ex

        bs = y.size(0)
        total_valid += bs
        batch_gate_ok = False

        with get_amp_context(device, enabled=use_amp):
            t0 = _time.perf_counter()

            logits, extra = model(
                graph_data=graph,
                explicit_qs=(qapi, qg, qa, papi, pg),
                time_ids=tids,
                masks=masks,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append(_time.perf_counter() - t0)

            loss_cls = criterion(logits, y)
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

        sum_cls += float(loss_cls.item()) * bs
        total_loss += float(loss_cls.item()) * bs
        total_samples += bs

        uncertainty, ok_uncertainty = _extract_extra_mean(extra, "uncertainty_score")
        if ok_uncertainty:
            sum_uncertainty += uncertainty * bs
            num_uncertainty += bs

        gate_disagreement, ok_gate_disagreement = _extract_extra_mean(extra, "gate_disagreement")
        if ok_gate_disagreement:
            sum_gate_disagreement += gate_disagreement * bs
            num_gate_disagreement += bs

        gate_entropy, ok_gate_entropy = _extract_extra_mean(extra, "gate_entropy")
        if ok_gate_entropy:
            sum_gate_entropy += gate_entropy * bs
            num_gate_entropy += bs

        wi, wg, wj, ok_w = _extract_gate_weights(extra)
        if ok_w:
            sum_wi += wi
            sum_wg += wg
            sum_wj += wj
            num_w += 1

            gate_values = extra.get("gate_weights")
            if isinstance(gate_values, torch.Tensor) and gate_values.numel() > 0:
                gv = gate_values.detach().float().cpu()
                if gv.ndim == 2 and gv.size(0) == bs and gv.size(1) == 3:
                    all_gate_weights.extend(gv.tolist())
                    batch_gate_ok = True

        wi_std, wg_std, wj_std, ok_std = _extract_gate_weight_stds(extra)
        if ok_std:
            sum_wi_std += wi_std
            sum_wg_std += wg_std
            sum_wj_std += wj_std
            num_w_std += 1

        align_cov, ok_align_cov = _extract_extra_mean(extra, "alignment_coverage")
        if ok_align_cov:
            sum_align_cov += align_cov
            num_align_cov += 1

        align_density, ok_align_density = _extract_extra_mean(extra, "alignment_density")
        if ok_align_density:
            sum_align_density += align_density
            num_align_density += 1

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())
        all_confs.extend(conf.detach().cpu().tolist())

        if batch_gate_ok:
            all_gate_correct.extend((preds == y).detach().float().cpu().tolist())
            all_gate_qapi.extend(qapi.detach().float().view(-1).cpu().tolist())
            all_gate_qgraph.extend(qg.detach().float().view(-1).cpu().tolist())

    if not all_labels:
        return (
            0.0,  # avg loss
            0.0,  # avg cls
            0.0,  # f1
            0.0,  # acc
            1.0,  # softmax aurc
            0.0,  # gate api
            0.0,  # gate graph
            0.0,  # gate joint
            0.0,  # uncertainty
            0.0,  # gate disagreement
            0.0,  # gate entropy
        )

    n = max(total_samples, 1)
    avg_loss = total_loss / n
    avg_cls = sum_cls / n

    pred_arr = np.asarray(all_preds, dtype=np.int64)
    label_arr = np.asarray(all_labels, dtype=np.int64)
    conf_arr = np.asarray(all_confs, dtype=np.float64)
    correct_arr = (pred_arr == label_arr).astype(np.float64)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    softmax_aurc_v = float(aurc(conf_arr, correct_arr))

    avg_gate_api = sum_wi / max(num_w, 1)
    avg_gate_graph = sum_wg / max(num_w, 1)
    avg_gate_joint = sum_wj / max(num_w, 1)

    avg_uncertainty = sum_uncertainty / max(num_uncertainty, 1)
    avg_gate_disagreement = sum_gate_disagreement / max(num_gate_disagreement, 1)
    avg_gate_entropy = sum_gate_entropy / max(num_gate_entropy, 1)

    gate_diag = _gate_diagnostics(
        all_gate_weights,
        correct=all_gate_correct,
        q_api=(all_gate_qapi if len(all_gate_qapi) == len(all_gate_weights) else None),
        q_graph=(all_gate_qgraph if len(all_gate_qgraph) == len(all_gate_weights) else None),
    )

    if logger:
        logger.info(
            f"[eval][epoch {epoch}] "
            f"loss={avg_loss:.4f} F1={f1:.4f} Acc={acc:.4f} "
            f"softmax_AURC={softmax_aurc_v:.4f} "
            f"avg_w=({avg_gate_api:.3f},{avg_gate_graph:.3f},{avg_gate_joint:.3f}) "
            f"std_w=({sum_wi_std / max(num_w_std, 1):.3f},"
            f"{sum_wg_std / max(num_w_std, 1):.3f},"
            f"{sum_wj_std / max(num_w_std, 1):.3f}) "
            f"uncertainty={avg_uncertainty:.3f} "
            f"gate_dis={avg_gate_disagreement:.3f} "
            f"gate_ent={avg_gate_entropy:.3f} "
            f"align_cov={sum_align_cov / max(num_align_cov, 1):.3f} "
            f"align_density={sum_align_density / max(num_align_density, 1):.4f} "
            f"valid={total_valid} failed={failed} skipped={skipped}"
        )

        if gate_diag:
            std = gate_diag.get("std", [0.0, 0.0, 0.0])
            logger.info(
                f"[eval][epoch {epoch}] gate-diagnostics: "
                f"weight_entropy={gate_diag.get('entropy', 0.0):.4f} "
                f"std=({std[0]:.3f},{std[1]:.3f},{std[2]:.3f}) "
                f"correct={gate_diag.get('correct_ge_0.5')} "
                f"wrong={gate_diag.get('correct_lt_0.5')} "
                f"qapi_low={gate_diag.get('qapi_low')} "
                f"qapi_high={gate_diag.get('qapi_high')} "
                f"qgraph_low={gate_diag.get('qgraph_low')} "
                f"qgraph_high={gate_diag.get('qgraph_high')}"
            )

        if times and total_valid > 0:
            total_time = sum(times)
            logger.info(
                f"[eval][epoch {epoch}] "
                f"per_sample={total_time / total_valid * 1000:.2f}ms "
                f"throughput={total_valid / max(total_time, 1e-6):.1f}/s"
            )

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info(f"[eval][epoch {epoch}] peak_gpu={peak:.1f}MB")

    return (
        avg_loss,
        avg_cls,
        f1,
        acc,
        softmax_aurc_v,
        avg_gate_api,
        avg_gate_graph,
        avg_gate_joint,
        avg_uncertainty,
        avg_gate_disagreement,
        avg_gate_entropy,
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
        (
            loss,
            cls,
            f1,
            acc,
            softmax_aurc_v,
            aw_api,
            aw_graph,
            aw_joint,
            uncertainty,
            gate_disagreement,
            gate_entropy,
        ) = eval_one_epoch(
            model, dl, criterion, device, epoch, num_epochs=num_epochs,
            use_amp=use_amp, logger=None
        )

        metrics[int(y)] = {
            "loss": loss,
            "cls_loss": cls,
            "f1": f1,
            "acc": acc,
            "softmax_aurc": softmax_aurc_v,
            "gate_api": aw_api,
            "gate_graph": aw_graph,
            "gate_joint": aw_joint,
            "uncertainty": uncertainty,
            "gate_disagreement": gate_disagreement,
            "gate_entropy": gate_entropy,
        }

    if logger and metrics:
        lines = [
            f"{y}:F1={m['f1']:.3f}/softmax_AURC={m['softmax_aurc']:.3f}"
            for y, m in sorted(metrics.items())
        ]
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

@torch.no_grad()
def collect_calibrated_outputs(model, calibrator, loader, device, use_amp=False):
    model.eval()
    calibrator.eval()
    all_p, all_y = [], []

    for batch in loader:
        if batch is None:
            continue

        r = prepare_batch(
            batch,
            device,
            skip_graph=False,
            skip_masks=(
                model.fusion_mode != "ours"
                or not getattr(model, "use_alignment_bias", False)
            ),
        )
        if r[2] is None:
            continue

        graph, masks, y, _, ex, _ = r
        qapi, qg, qa, papi, pg, tids = ex

        with get_amp_context(device, enabled=use_amp):
            logits, _ = model(
                graph_data=graph,
                explicit_qs=(qapi, qg, qa, papi, pg),
                time_ids=tids,
                masks=masks,
            )

        all_p.append(calibrator(logits.float()).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    if not all_p or not all_y:
        raise RuntimeError("collect_calibrated_outputs: no valid batches collected")

    return np.concatenate(all_p), np.concatenate(all_y)


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
    c_alignment = c_model["alignment"]
    c_gate = c_model["gate"]

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
    configured_historical_epochs = int(c_train.get("historical_epochs", epochs))
    configured_adaptation_epochs = int(c_train.get("adaptation_epochs", 0))
    if c_data.get("adapt_csv"):
        if configured_historical_epochs <= 0 or configured_adaptation_epochs <= 0:
            raise ValueError(
                "continual adaptation requires positive train.historical_epochs "
                "and train.adaptation_epochs when data.adapt_csv is set"
            )
        derived_epochs = configured_historical_epochs + configured_adaptation_epochs
        if epochs != derived_epochs:
            logger.warning(
                "Overriding train.epochs to historical_epochs + adaptation_epochs: "
                f"{epochs} -> {derived_epochs}"
            )
            epochs = derived_epochs
            c_train["epochs"] = derived_epochs
    else:
        configured_historical_epochs = epochs
        configured_adaptation_epochs = 0
        c_train["historical_epochs"] = epochs
        c_train["adaptation_epochs"] = 0
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
        use_quality_gate_inputs=bool(c_gate["quality_inputs"]),
        use_uncertainty_gate=bool(c_gate["uncertainty_inputs"]),
        gate_mode=str(c_gate["mode"]),
        gate_detach=bool(c_gate["detach"]),
        late_fusion_api_weight=0.5,
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
    historical_best_p = os.path.join(ckpt_dir, f"best_historical_{exp_name}.pt")
    adapted_best_p = os.path.join(ckpt_dir, f"best_{exp_name}.pt")
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
                if os.path.exists(historical_best_p):
                    state = torch.load(historical_best_p, map_location=device, weights_only=False)
                    model.load_state_dict(state["state_dict"])
                    logger.info(
                        "Loaded historical best checkpoint before adaptation: "
                        f"{historical_best_p}"
                    )
                else:
                    logger.warning(
                        "Historical best checkpoint was not found; adaptation continues "
                        "from the last historical epoch."
                    )
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
        epoch_loss_cfg["_continual_phase"] = continual_phase
        epoch_loader = dl_adapt if continual_phase == "adaptation" else dl_tr

        (
            tr_loss,
            tr_f1,
            tr_acc,
            tr_cls,
            tr_align,
            tr_branch_aux,
            tr_gate_oracle,
            tr_uncertainty,
            tr_gate_disagreement,
            tr_gate_entropy,
            tr_gate_api,
            tr_gate_graph,
            tr_gate_joint,
        ) = train_one_epoch(
            model, epoch_loader, optimizer, scaler, criterion, device,
            epoch, epochs, loss_cfg=epoch_loss_cfg, logger=logger, use_amp=use_amp,
            grad_accum_steps=int(c_train["grad_accum_steps"]))

        (
            val_loss,
            val_cls,
            val_f1,
            val_acc,
            val_softmax_aurc,
            val_gate_api,
            val_gate_graph,
            val_gate_joint,
            val_uncertainty,
            val_gate_disagreement,
            val_gate_entropy,
        ) = eval_one_epoch(
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
        best_p = historical_best_p if continual_phase == "historical" and dl_adapt is not None else adapted_best_p
        save_checkpoint(last_p, epoch, model, optimizer, scheduler, scaler,
                        best_score, no_improve, cfg)
        if improved:
            save_checkpoint(best_p, epoch, model, optimizer, scheduler, scaler,
                            best_score, no_improve, cfg)

        append_metrics_csv(csv_path, dict(
            epoch=epoch, stage=stage_name, continual_phase=continual_phase,
            train_loss=tr_loss, train_f1=tr_f1, train_acc=tr_acc,
            train_cls=tr_cls,
            train_alignment=tr_align,
            train_branch_aux=tr_branch_aux,
            train_gate_oracle=tr_gate_oracle,
            train_uncertainty=tr_uncertainty,
            train_gate_disagreement=tr_gate_disagreement,
            train_gate_entropy=tr_gate_entropy,
            train_gate_api=tr_gate_api,
            train_gate_graph=tr_gate_graph,
            train_gate_joint=tr_gate_joint,
            val_loss=val_loss, val_cls_loss=val_cls, val_f1=val_f1, val_acc=val_acc,
            val_softmax_aurc=val_softmax_aurc,
            val_uncertainty=val_uncertainty,
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
            f"softmax_aurc={val_softmax_aurc:.4f}| "
            f"sel={selection_score:.4f}")
        early_stop_allowed = (
            stage_name != "warmup"
            and (dl_adapt is None or continual_phase == "adaptation")
        )
        if early_stop_allowed and no_improve >= patience:
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

    (
        test_loss,
        test_cls,
        test_f1,
        test_acc,
        test_softmax_aurc,
        test_gate_api,
        test_gate_graph,
        test_gate_joint,
        test_uncertainty,
        test_gate_disagreement,
        test_gate_entropy,
    ) = eval_one_epoch(
        model, dl_test, criterion, device, epoch=0, num_epochs=1,
        use_amp=use_amp, logger=logger
    )

    logger.info(
        f"🏆 TEST: loss={test_loss:.4f} F1={test_f1:.4f} "
        f"Acc={test_acc:.4f} softmax_AURC={test_softmax_aurc:.4f} "
        f"uncertainty={test_uncertainty:.4f}"
    )

    # ── Selective classification on calibrated probs ──
    test_probs, test_labels = collect_calibrated_outputs(
        model, calibrator, dl_test, device, use_amp
    )
    test_preds = test_probs.argmax(axis=1)
    det_metrics = _binary_detection_metrics(test_labels, test_probs, test_preds)
    test_conf = test_probs.max(axis=1)
    test_correct = (test_preds == test_labels).astype(np.float64)
    logger.info(
        "Final detection metrics: "
        f"F1={test_f1:.4f} MalwareRecall={det_metrics['malware_recall']:.4f} "
        f"FNR={det_metrics['fnr']:.4f} AUROC={det_metrics['auroc']:.4f} "
        f"AUPRC={det_metrics['auprc']:.4f}"
    )
    final_metrics_path = os.path.join(ckpt_dir, f"final_metrics_{exp_name}.csv")
    test_year = (
        str(ds_test.unique_years[0])
        if len(getattr(ds_test, "unique_years", [])) == 1
        else "combined"
    )
    with open(final_metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "exp_name", "adaptation_ratio", "phase", "test_year",
                "macro_f1", "malware_recall", "fnr", "auroc", "auprc",
                "softmax_aurc",
                "gate_api", "gate_graph", "gate_joint",
                "uncertainty", "gate_disagreement", "gate_entropy",
            ]
        )
        w.writeheader()
        w.writerow({
            "exp_name": exp_name,
            "adaptation_ratio": float(c_train.get("adaptation_ratio", 0.0)),
            "phase": "final_test",
            "test_year": test_year,
            "macro_f1": f"{test_f1:.6f}",
            "malware_recall": f"{det_metrics['malware_recall']:.6f}",
            "fnr": f"{det_metrics['fnr']:.6f}",
            "auroc": f"{det_metrics['auroc']:.6f}",
            "auprc": f"{det_metrics['auprc']:.6f}",
            "softmax_aurc": f"{test_softmax_aurc:.6f}",
            "gate_api": f"{test_gate_api:.6f}",
            "gate_graph": f"{test_gate_graph:.6f}",
            "gate_joint": f"{test_gate_joint:.6f}",
            "uncertainty": f"{test_uncertainty:.6f}",
            "gate_disagreement": f"{test_gate_disagreement:.6f}",
            "gate_entropy": f"{test_gate_entropy:.6f}",
        })
    logger.info(f"Final metrics CSV: {final_metrics_path}")
    logger.info(
        f"Selective calibrated summary: softmax_AURC={aurc(test_conf,test_correct):.4f} "
    )
    logger.info(
        f"🎯 Selective (calibrated): AURC={aurc(test_conf,test_correct):.4f} "
        f"softmax_E-AURC={eaurc(test_conf,test_correct):.4f} | "
        f"error@cov0.8={risk_at_coverage(test_conf,test_correct,0.8):.4f} "
        f"error@cov0.9={risk_at_coverage(test_conf,test_correct,0.9):.4f} | "
        f"coverage@error≤1%={coverage_at_risk(test_conf,test_correct,0.01):.4f} "
        f"coverage@error≤5%={coverage_at_risk(test_conf,test_correct,0.05):.4f}"
    )

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

        (
            extra_loss,
            extra_cls,
            extra_f1,
            extra_acc,
            extra_softmax_aurc,
            extra_gate_api,
            extra_gate_graph,
            extra_gate_joint,
            extra_uncertainty,
            extra_gate_disagreement,
            extra_gate_entropy,
        ) = eval_one_epoch(
            model, dl_extra, criterion, device, epoch=0, num_epochs=1,
            use_amp=use_amp, logger=logger
        )

        logger.info(
            f"EXTRA_TEST[{extra_name}]: loss={extra_loss:.4f} F1={extra_f1:.4f} "
            f"Acc={extra_acc:.4f} softmax_AURC={extra_softmax_aurc:.4f} "
            f"uncertainty={extra_uncertainty:.4f}"
        )

        extra_probs, extra_labels = collect_calibrated_outputs(
            model, calibrator, dl_extra, device, use_amp
        )
        extra_preds = extra_probs.argmax(axis=1)
        extra_conf = extra_probs.max(axis=1)
        extra_correct = (extra_preds == extra_labels).astype(np.float64)
        logger.info(
            f"Selective[{extra_name}] calibrated: "
            f"softmax_AURC={aurc(extra_conf, extra_correct):.4f} "
            f"softmax_E-AURC={eaurc(extra_conf, extra_correct):.4f} | "
            f"error@cov0.8={risk_at_coverage(extra_conf, extra_correct, 0.8):.4f} "
            f"error@cov0.9={risk_at_coverage(extra_conf, extra_correct, 0.9):.4f} | "
            f"coverage@error<=1%={coverage_at_risk(extra_conf, extra_correct, 0.01):.4f} "
            f"coverage@error<=5%={coverage_at_risk(extra_conf, extra_correct, 0.05):.4f}"
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
