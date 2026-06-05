from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fusion.dataset import AEGDataset, aeg_collate_fn, split_label_stats
from fusion.losses import compute_aeg_loss
from fusion.model import build_model


LOGGER = logging.getLogger(__name__)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    bases = cfg.pop("base", None) or cfg.pop("bases", None) or []
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_update(merged, load_config(base_path))
    return deep_update(merged, cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def _binary_metrics(labels: list[int], probs: list[float], preds: list[int]) -> dict[str, float]:
    if not labels:
        return {}
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    acc = (tp + tn) / max(1, len(labels))

    def _safe(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    precision_pos = _safe(tp, tp + fp)
    recall_pos = _safe(tp, tp + fn)
    f1_pos = _safe(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = _safe(tn, tn + fn)
    recall_neg = _safe(tn, tn + fp)
    f1_neg = _safe(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    brier = sum((p - y) ** 2 for y, p in zip(labels, probs)) / max(1, len(labels))
    ece = _ece(labels, probs, preds, bins=10)
    out = {
        "acc": acc,
        "macro_f1": macro_f1,
        "f1": macro_f1,
        "f1_pos": f1_pos,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "macro_recall": 0.5 * (recall_pos + recall_neg),
        "brier": brier,
        "ece_10": ece,
        "mean_confidence": sum(max(p, 1.0 - p) for p in probs) / max(1, len(probs)),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
    except Exception:
        out["auc"] = math.nan
        out["ap"] = math.nan
    return out


def _ece(labels: list[int], probs: list[float], preds: list[int], *, bins: int = 10) -> float:
    total = len(labels)
    if total <= 0:
        return 0.0
    confidences = [max(p, 1.0 - p) for p in probs]
    correct = [1.0 if y == pred else 0.0 for y, pred in zip(labels, preds)]
    ece = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = [lo <= c < hi if idx < bins - 1 else lo <= c <= hi for c in confidences]
        n = sum(mask)
        if n == 0:
            continue
        conf = sum(c for c, m in zip(confidences, mask) if m) / n
        acc = sum(c for c, m in zip(correct, mask) if m) / n
        ece += (n / total) * abs(acc - conf)
    return float(ece)


def _device(cfg: dict[str, Any]) -> torch.device:
    requested = str((cfg.get("train", {}) or {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _make_dataset(cfg: dict[str, Any], split: str, *, aug: bool = False, view: str | None = None, strength: float | None = None) -> AEGDataset:
    data_cfg = cfg.get("data", {}) or {}
    split_cfg = data_cfg.get(split, {}) or {}
    pt_dir = split_cfg.get("pt_dir") or data_cfg.get(f"{split}_pt_dir")
    csv_path = split_cfg.get("csv") or split_cfg.get("label_csv") or data_cfg.get(f"{split}_csv")
    if not pt_dir:
        raise ValueError(f"data.{split}.pt_dir is required")
    if not csv_path:
        raise ValueError(f"data.{split}.csv is required")
    robust_cfg = cfg.get("robust", {}) or {}
    return AEGDataset(
        pt_dir,
        csv_path,
        split=split,
        train_aug=aug,
        aug_prob=1.0 if view else float(robust_cfg.get("perturb_prob", 0.5)),
        aug_views=[view] if view else list(robust_cfg.get("train_views", ["api_degraded", "graph_degraded", "api_graph_degraded", "manifest_degraded", "all_degraded"])),
        aug_strengths=[float(strength)] if strength is not None else list(robust_cfg.get("perturb_strengths", [0.1, 0.3, 0.5])),
        seed=int((cfg.get("train", {}) or {}).get("seed", 42)),
        strict_integrity=bool(data_cfg.get("strict_integrity", True)),
    )


def _loader(cfg: dict[str, Any], dataset: AEGDataset, *, train: bool) -> DataLoader:
    train_cfg = cfg.get("train", {}) or {}
    batch_size = int(train_cfg.get("batch_size" if train else "eval_batch_size", train_cfg.get("batch_size", 24)))
    workers = int(train_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(train_cfg.get("seed", 42)) + (0 if train else 100_000))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        persistent_workers=workers > 0,
        generator=generator,
        worker_init_fn=_seed_worker if workers > 0 else None,
        collate_fn=aeg_collate_fn,
    )


def _validate_split_isolation(*datasets: AEGDataset) -> None:
    seen: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    for dataset in datasets:
        for path, _label in dataset.samples:
            sid = path.stem.lower()
            prev = seen.get(sid)
            if prev is not None and prev != dataset.split:
                conflicts.append((sid, prev, dataset.split))
            seen[sid] = dataset.split
    if conflicts:
        raise ValueError(f"Sample id overlap across splits: count={len(conflicts)} examples={conflicts[:5]}")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    out["clean"] = batch["clean"].to(device)
    out["aug"] = batch["aug"].to(device)
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    use_aug = bool((cfg.get("robust", {}) or {}).get("train_aug", True))
    grad_clip = float((cfg.get("train", {}) or {}).get("grad_clip", 1.0))
    totals: dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        clean_logits, clean_extra = model(batch["clean"])
        aug_logits = aug_extra = None
        if use_aug:
            aug_logits, aug_extra = model(batch["aug"])
        loss, parts = compute_aeg_loss(
            clean_logits,
            batch["clean"].y.view(-1),
            clean_extra,
            aug_logits=aug_logits,
            aug_extra=aug_extra,
            loss_cfg=cfg.get("loss", {}) or {},
        )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
    return {key: value / max(1, steps) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    split_name: str,
    batch_key: str = "clean",
    dump_rows: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    probs: list[float] = []
    preds: list[int] = []
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=split_name, leave=False):
        data = batch[batch_key].to(device)
        logits, extra = model(data)
        prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
        pred = logits.argmax(dim=-1).detach().cpu()
        y = data.y.view(-1).detach().cpu()
        labels.extend([int(v) for v in y.tolist()])
        probs.extend([float(v) for v in prob.tolist()])
        preds.extend([int(v) for v in pred.tolist()])
        if dump_rows:
            attn = extra.get("attention_mass")
            for idx in range(y.numel()):
                row = {
                    "sid": batch["sid"][idx] if idx < len(batch["sid"]) else "",
                    "label": int(y[idx].item()),
                    "pred": int(pred[idx].item()),
                    "prob_malware": float(prob[idx].item()),
                    "q_api": float(extra["q_api"][idx].item()),
                    "q_graph": float(extra["q_graph"][idx].item()),
                    "q_manifest": float(extra["q_manifest"][idx].item()),
                    "q_align": float(extra["q_align"][idx].item()),
                    "code_reliability": float(extra["code_reliability"][idx].item()),
                    "manifest_reliability": float(extra["manifest_reliability"][idx].item()),
                    "code_manifest_similarity": float(extra["code_manifest_similarity"][idx].item()),
                    "code_manifest_conflict": float(extra["code_manifest_conflict"][idx].item()),
                }
                if isinstance(attn, torch.Tensor) and attn.ndim == 2 and idx < attn.size(0):
                    attn_names = [
                        "attn_method",
                        "attn_api_family",
                        "attn_permission",
                        "attn_component",
                        "attn_risk",
                        "attn_string_hint",
                        "attn_global",
                    ]
                    row.update(
                        {
                            name: float(attn[idx, col].item())
                            for col, name in enumerate(attn_names)
                            if col < attn.size(1)
                        }
                    )
                rows.append(row)
    return _binary_metrics(labels, probs, preds), rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _robust_eval_loaders(cfg: dict[str, Any], split: str) -> list[tuple[str, DataLoader]]:
    eval_cfg = cfg.get("eval", {}) or {}
    views = list(eval_cfg.get("robust_views", ["api_degraded", "graph_degraded", "api_graph_degraded", "manifest_degraded", "all_degraded", "api_missing", "graph_missing", "manifest_missing"]))
    strengths = list(eval_cfg.get("perturb_strengths", [0.5]))
    loaders: list[tuple[str, DataLoader]] = []
    for view in views:
        if view.endswith("_missing") or view in {"manifest_zeroed"}:
            ds = _make_dataset(cfg, split, aug=True, view=view, strength=1.0)
            loaders.append((view, _loader(cfg, ds, train=False)))
        else:
            for strength in strengths:
                ds = _make_dataset(cfg, split, aug=True, view=view, strength=float(strength))
                loaders.append((f"{view}@{float(strength):.2f}", _loader(cfg, ds, train=False)))
    return loaders


def _robust_val_loaders(cfg: dict[str, Any]) -> list[tuple[str, float, DataLoader]]:
    val_cfg = ((cfg.get("eval", {}) or {}).get("robust_val", {}) or {})
    if not bool(val_cfg.get("enabled", False)):
        return []
    scenarios = val_cfg.get("scenarios") or [
        {"name": "api_graph_degraded", "view": "api_graph_degraded", "strength": 0.5, "weight": 0.4},
        {"name": "manifest_noisy", "view": "manifest_noisy", "strength": 0.5, "weight": 0.3},
        {"name": "all_degraded", "view": "all_degraded", "strength": 0.5, "weight": 0.3},
    ]
    loaders: list[tuple[str, float, DataLoader]] = []
    seen: set[str] = set()
    for item in scenarios:
        view = str(item.get("view") or item.get("name") or "")
        if not view:
            raise ValueError("eval.robust_val.scenarios entries require view")
        strength = float(item.get("strength", 1.0 if view.endswith("_missing") else 0.5))
        name = str(item.get("name") or f"{view}@{strength:.2f}")
        if name in seen:
            raise ValueError(f"Duplicate robust validation scenario name: {name}")
        seen.add(name)
        weight = float(item.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"Robust validation scenario {name} has negative weight")
        ds = _make_dataset(cfg, "val", aug=True, view=view, strength=strength)
        loaders.append((name, weight, _loader(cfg, ds, train=False)))
    return loaders


def _checkpoint_score(
    cfg: dict[str, Any],
    clean_metrics: dict[str, float],
    robust_metrics: dict[str, dict[str, float]],
    robust_loaders: list[tuple[str, float, DataLoader]],
) -> float:
    train_cfg = cfg.get("train", {}) or {}
    metric = str(train_cfg.get("checkpoint_metric", "macro_f1"))
    if metric != "robust_composite":
        return float(clean_metrics.get(metric, clean_metrics.get("macro_f1", 0.0)))
    if not robust_loaders:
        raise ValueError("train.checkpoint_metric=robust_composite requires eval.robust_val.enabled=true")
    val_cfg = ((cfg.get("eval", {}) or {}).get("robust_val", {}) or {})
    clean_weight = float(val_cfg.get("clean_weight", 0.5))
    if clean_weight < 0:
        raise ValueError("eval.robust_val.clean_weight must be non-negative")
    total_weight = clean_weight
    score = clean_weight * float(clean_metrics.get("macro_f1", 0.0))
    for name, weight, _loader_obj in robust_loaders:
        total_weight += weight
        score += weight * float((robust_metrics.get(name) or {}).get("macro_f1", 0.0))
    return score / max(total_weight, 1e-8)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    train_cfg = cfg.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = _device(cfg)
    out_dir = Path(train_cfg.get("output_dir", "results/aeg_robust/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _make_dataset(cfg, "train", aug=bool((cfg.get("robust", {}) or {}).get("train_aug", True)))
    val_ds = _make_dataset(cfg, "val", aug=False)
    test_ds = _make_dataset(cfg, "test", aug=False)
    _validate_split_isolation(train_ds, val_ds, test_ds)
    LOGGER.info("Train stats: %s", split_label_stats(train_ds))
    LOGGER.info("Val stats: %s", split_label_stats(val_ds))
    LOGGER.info("Test stats: %s", split_label_stats(test_ds))

    train_loader = _loader(cfg, train_ds, train=True)
    val_loader = _loader(cfg, val_ds, train=False)
    test_loader = _loader(cfg, test_ds, train=False)
    robust_val_loaders = _robust_val_loaders(cfg)
    first_payload = torch.load(train_ds.samples[0][0], map_location="cpu")
    node_input_dim = int(first_payload["node_x"].size(1))
    model = build_model(cfg, node_input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )

    best_score = -1.0
    best_epoch = 0
    patience = int(train_cfg.get("patience", 8))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_cfg.get("epochs", 60)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        val_metrics, _ = evaluate(model, val_loader, device, split_name="val")
        val_robust_metrics: dict[str, dict[str, float]] = {}
        for name, _weight, loader_obj in robust_val_loaders:
            metrics, _ = evaluate(model, loader_obj, device, split_name=f"val_{name}", batch_key="aug")
            val_robust_metrics[name] = metrics
        score = _checkpoint_score(cfg, val_metrics, val_robust_metrics, robust_val_loaders)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_loss.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        for name, metrics in val_robust_metrics.items():
            for key, value in metrics.items():
                row[f"val_{name}_{key}"] = value
        row["checkpoint_score"] = score
        history.append(row)
        LOGGER.info(
            "epoch=%s val_macro_f1=%.4f checkpoint_score=%.4f train_loss=%.4f",
            epoch,
            val_metrics.get("macro_f1", 0.0),
            score,
            train_loss.get("loss", 0.0),
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "score": best_score}, out_dir / "best.pt")
        elif epoch - best_epoch >= patience:
            LOGGER.info("Early stopping at epoch %s; best epoch %s", epoch, best_epoch)
            break

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    val_metrics, val_rows = evaluate(model, val_loader, device, split_name="val_best", dump_rows=True)
    test_metrics, test_rows = evaluate(model, test_loader, device, split_name="test", dump_rows=True)
    summary: dict[str, Any] = {"best_epoch": best_epoch, "best_score": best_score, "val": val_metrics, "test": test_metrics}

    robust_results: dict[str, dict[str, float]] = {}
    if bool((cfg.get("eval", {}) or {}).get("robust_eval", True)):
        for name, loader in _robust_eval_loaders(cfg, "test"):
            metrics, rows = evaluate(model, loader, device, split_name=f"test_{name}", batch_key="aug", dump_rows=True)
            robust_results[name] = metrics
            _write_rows(out_dir / f"diagnostics_test_{name.replace('@', '_')}.csv", rows)
    summary["robust_test"] = robust_results

    _write_rows(out_dir / "diagnostics_val.csv", val_rows)
    _write_rows(out_dir / "diagnostics_test_clean.csv", test_rows)
    with (out_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-aware AEG robust Android malware detector.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
