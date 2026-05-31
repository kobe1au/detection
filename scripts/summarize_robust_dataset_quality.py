#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.dataset import RobustTriModalDataset, _normalize_loaded_pt
from fusion.train import build_dataset, load_config, resolve


QUALITY_FIELDS = ("q_api", "q_graph", "q_align", "q_manifest")


def _scalar(value: Any, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().view(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _tensor_sum(value: Any) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).sum().item())


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: 0.0 for k in ("mean", "std", "p10", "p50", "p90", "min", "max")}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "p10": float(torch.quantile(t, 0.10).item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p90": float(torch.quantile(t, 0.90).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def _analyze_pt(dataset: RobustTriModalDataset, pt_path: Path) -> dict[str, Any]:
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    dex_list, sources = _normalize_loaded_pt(raw)
    data = dataset._aggregate_api_graph(dex_list)
    if data is None:
        raise ValueError("empty valid sample")
    graph_counts_from_source = data.get("graph_semantic_category_counts")
    manifest_payload = dataset._manifest_payload(sources)
    data.update(manifest_payload)
    if isinstance(graph_counts_from_source, torch.Tensor):
        data["graph_semantic_category_counts"] = graph_counts_from_source
        data["graph_category_counts"] = graph_counts_from_source

    meta = raw.get("direct_build_meta", {}) if isinstance(raw, dict) else {}
    dex_failures = meta.get("dex_failures") if isinstance(meta, dict) else None
    manifest_meta = data.get("manifest_meta") if isinstance(data.get("manifest_meta"), dict) else {}
    return {
        "q_api": _scalar(data.get("q_api")),
        "q_graph": _scalar(data.get("q_graph")),
        "q_align": _scalar(data.get("q_align")),
        "q_manifest": _scalar(data.get("q_manifest")),
        "api_semantic_sum": _tensor_sum(data.get("api_semantic_category_counts")),
        "graph_semantic_sum": _tensor_sum(data.get("graph_semantic_category_counts")),
        "manifest_semantic_sum": _tensor_sum(data.get("manifest_category_counts")),
        "manifest_parse_error": bool(manifest_meta.get("parse_error")),
        "multi_dex_partial_failed": bool(isinstance(dex_failures, list) and len(dex_failures) > 0),
    }


def _label_ratio(labels: list[int]) -> dict[str, float | int]:
    n = len(labels)
    pos = sum(1 for v in labels if int(v) == 1)
    neg = sum(1 for v in labels if int(v) == 0)
    return {
        "num_samples": n,
        "num_malware": pos,
        "num_benign": neg,
        "malware_ratio": float(pos / n) if n else 0.0,
        "benign_ratio": float(neg / n) if n else 0.0,
    }


def summarize_split(cfg: dict[str, Any], split: str, write_rows_dir: Path | None = None) -> dict[str, Any]:
    dataset = build_dataset(cfg, split, is_train=False)
    labels: list[int] = []
    quality_values = {field: [] for field in QUALITY_FIELDS}
    failed = 0
    rows: list[dict[str, Any]] = []
    counters = {
        "api_semantic_nonzero": 0,
        "graph_semantic_nonzero": 0,
        "manifest_semantic_nonzero": 0,
        "manifest_parse_error": 0,
        "multi_dex_partial_failed": 0,
    }

    for pt_path, label, sid, year in tqdm(
        dataset.samples,
        desc=f"summarize {split}",
        unit="pt",
    ):
        labels.append(int(label))
        try:
            info = _analyze_pt(dataset, pt_path)
        except Exception as exc:
            failed += 1
            info = {
                "q_api": math.nan,
                "q_graph": math.nan,
                "q_align": math.nan,
                "q_manifest": math.nan,
                "api_semantic_sum": 0.0,
                "graph_semantic_sum": 0.0,
                "manifest_semantic_sum": 0.0,
                "manifest_parse_error": False,
                "multi_dex_partial_failed": False,
                "fail_reason": f"{type(exc).__name__}: {exc}",
            }

        for field in QUALITY_FIELDS:
            value = float(info.get(field, math.nan))
            if math.isfinite(value):
                quality_values[field].append(value)
        if float(info.get("api_semantic_sum", 0.0)) > 0.0:
            counters["api_semantic_nonzero"] += 1
        if float(info.get("graph_semantic_sum", 0.0)) > 0.0:
            counters["graph_semantic_nonzero"] += 1
        if float(info.get("manifest_semantic_sum", 0.0)) > 0.0:
            counters["manifest_semantic_nonzero"] += 1
        if bool(info.get("manifest_parse_error")):
            counters["manifest_parse_error"] += 1
        if bool(info.get("multi_dex_partial_failed")):
            counters["multi_dex_partial_failed"] += 1
        rows.append(
            {
                "split": split,
                "sid": sid,
                "label": int(label),
                "year": int(year),
                "pt_path": str(pt_path),
                **info,
            }
        )

    n = len(dataset.samples)
    summary: dict[str, Any] = {
        "split": split,
        **_label_ratio(labels),
        "num_failed_to_analyze": failed,
        "analysis_failed_ratio": float(failed / n) if n else 0.0,
        "api_semantic_nonzero_ratio": float(counters["api_semantic_nonzero"] / n) if n else 0.0,
        "graph_semantic_nonzero_ratio": float(counters["graph_semantic_nonzero"] / n) if n else 0.0,
        "manifest_semantic_nonzero_ratio": float(counters["manifest_semantic_nonzero"] / n) if n else 0.0,
        "manifest_parse_error_ratio": float(counters["manifest_parse_error"] / n) if n else 0.0,
        "multi_dex_partial_failed_ratio": float(counters["multi_dex_partial_failed"] / n) if n else 0.0,
    }
    for field in QUALITY_FIELDS:
        for stat_name, stat_value in _stats(quality_values[field]).items():
            summary[f"{field}_{stat_name}"] = stat_value

    if write_rows_dir is not None:
        write_rows_dir.mkdir(parents=True, exist_ok=True)
        path = write_rows_dir / f"{split}_quality_rows.csv"
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    rows_dir = out_dir / "rows" if args.write_rows else None
    summaries = [summarize_split(cfg, split, rows_dir) for split in args.splits]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(out_dir / "dataset_quality_summary.csv", summaries)
    payload = {"splits": summaries}
    (out_dir / "dataset_quality_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tri-modal robust dataset quality before training.")
    parser.add_argument("--config", nargs="+", default=["config/experiments/tri_modal_robust/base_tri_modal_robust.yaml"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out-dir", default="results/dataset_quality")
    parser.add_argument("--write-rows", action="store_true", help="Write per-sample quality rows for debugging.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
