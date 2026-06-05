#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.dataset import RobustTriModalDataset, _normalize_loaded_pt, apply_dex_success_ratio
from fusion.train import build_dataset, load_config, resolve


ID_CANDIDATES = ("id", "sha256", "ID", "Id")
SLICE_FIELDNAMES = [
    "id",
    "sha256",
    "label",
    "split",
    "sid",
    "slice",
    "year",
    "pt_path",
    "dataset_failed",
    "fail_reason",
    "q_api",
    "q_graph",
    "q_manifest",
    "q_align",
    "pert_api",
    "pert_graph",
    "pert_manifest",
    "api_semantic_sum",
    "graph_semantic_sum",
    "manifest_semantic_sum",
    "manifest_parse_error",
    "dex_failure_count",
    "num_dex",
]


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


def _find_id_col(df: pd.DataFrame) -> str:
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"CSV must contain one of id columns: {', '.join(ID_CANDIDATES)}")


def _read_label_rows(csv_path: Path) -> tuple[str, dict[str, dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        id_col = next((col for col in ID_CANDIDATES if col in fieldnames), "")
        if not id_col:
            raise ValueError(f"{csv_path} must contain one of id columns: {', '.join(ID_CANDIDATES)}")
        rows = {str(row.get(id_col, "")).strip().lower(): row for row in reader}
    return id_col, rows, fieldnames


def _safe_load_raw(pt_path: Path) -> tuple[Any | None, str]:
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=False), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _analyze_pt(dataset: RobustTriModalDataset, pt_path: Path) -> dict[str, Any]:
    raw, load_error = _safe_load_raw(pt_path)
    if load_error:
        return {
            "dataset_failed": True,
            "fail_reason": load_error,
            "q_api": 0.0,
            "q_graph": 0.0,
            "q_manifest": 0.0,
            "q_align": 0.0,
            "api_semantic_sum": 0.0,
            "graph_semantic_sum": 0.0,
            "manifest_semantic_sum": 0.0,
            "manifest_parse_error": "",
            "dex_failure_count": 0,
            "num_dex": 0,
        }

    dex_list, sources = _normalize_loaded_pt(raw)
    try:
        data = dataset._aggregate_api_graph(dex_list)
        if data is None:
            raise ValueError("empty valid sample")
        apply_dex_success_ratio(data, sources)
        graph_counts_from_source = data.get("graph_semantic_category_counts")
        manifest_payload = dataset._manifest_payload(sources)
        data.update(manifest_payload)
        if isinstance(graph_counts_from_source, torch.Tensor):
            data["graph_semantic_category_counts"] = graph_counts_from_source
            data["graph_category_counts"] = graph_counts_from_source
    except Exception as exc:
        return {
            "dataset_failed": True,
            "fail_reason": f"{type(exc).__name__}: {exc}",
            "q_api": 0.0,
            "q_graph": 0.0,
            "q_manifest": 0.0,
            "q_align": 0.0,
            "api_semantic_sum": 0.0,
            "graph_semantic_sum": 0.0,
            "manifest_semantic_sum": 0.0,
            "manifest_parse_error": "",
            "dex_failure_count": 0,
            "num_dex": len(dex_list),
        }

    meta = raw.get("direct_build_meta", {}) if isinstance(raw, dict) else {}
    dex_failures = meta.get("dex_failures") if isinstance(meta, dict) else None
    manifest_meta = data.get("manifest_meta") if isinstance(data.get("manifest_meta"), dict) else {}
    return {
        "dataset_failed": False,
        "fail_reason": "",
        "q_api": _scalar(data.get("q_api")),
        "q_graph": _scalar(data.get("q_graph")),
        "q_manifest": _scalar(data.get("q_manifest")),
        "q_align": _scalar(data.get("q_align")),
        "pert_api": _scalar(data.get("pert_api")),
        "pert_graph": _scalar(data.get("pert_graph")),
        "pert_manifest": _scalar(data.get("pert_manifest")),
        "api_semantic_sum": _tensor_sum(data.get("api_semantic_category_counts")),
        "graph_semantic_sum": _tensor_sum(data.get("graph_semantic_category_counts")),
        "manifest_semantic_sum": _tensor_sum(data.get("manifest_category_counts")),
        "manifest_parse_error": str(manifest_meta.get("parse_error") or ""),
        "dex_failure_count": len(dex_failures or []) if isinstance(dex_failures, list) else 0,
        "num_dex": int(meta.get("num_dex_total", meta.get("num_dex", len(dex_list)))) if isinstance(meta, dict) else len(dex_list),
    }


def _slice_flags(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    q_api = float(metrics["q_api"])
    q_graph = float(metrics["q_graph"])
    q_manifest = float(metrics["q_manifest"])
    q_align = float(metrics["q_align"])
    return {
        "dataset_failed": bool(metrics.get("dataset_failed")),
        "api_low_quality": q_api < args.api_threshold,
        "graph_low_quality": q_graph < args.graph_threshold,
        "align_low_quality": q_align < args.align_threshold,
        "manifest_low_quality": q_manifest < args.manifest_threshold,
        "code_common_failure": q_api < args.common_api_threshold and q_graph < args.common_graph_threshold,
        "graph_semantic_missing": float(metrics.get("graph_semantic_sum", 0.0)) <= 0.0,
        "manifest_parse_failed": bool(metrics.get("manifest_parse_error")),
        "multi_dex_partial_failed": int(metrics.get("dex_failure_count", 0)) > 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = list(SLICE_FIELDNAMES)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_slices(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    splits = args.splits or ["train", "val", "test"]
    out_dir = Path(args.out_dir)
    summary: dict[str, Any] = {}

    for split in splits:
        dataset = build_dataset(cfg, split, is_train=False)
        data_cfg = cfg["data"]
        csv_path = Path(resolve(data_cfg.get("root", ""), data_cfg[f"{split}_csv"]))
        _id_col, row_map, _fieldnames = _read_label_rows(csv_path)

        slice_rows: dict[str, list[dict[str, Any]]] = {}
        all_rows: list[dict[str, Any]] = []
        for pt_path, label, sid, year in dataset.samples:
            metrics = _analyze_pt(dataset, pt_path)
            flags = _slice_flags(metrics, args)
            base = dict(row_map.get(sid, {}))
            base.update(
                {
                    "split": split,
                    "sid": sid,
                    "label": int(label),
                    "year": int(year),
                    "pt_path": str(pt_path),
                    **metrics,
                }
            )
            all_rows.append(base)
            for slice_name, enabled in flags.items():
                if enabled:
                    row = dict(base)
                    row["slice"] = slice_name
                    slice_rows.setdefault(slice_name, []).append(row)

        split_summary = {
            "total": len(dataset.samples),
            "slices": {name: len(rows) for name, rows in sorted(slice_rows.items())},
        }
        summary[split] = split_summary
        _write_csv(out_dir / f"{split}_all_quality.csv", all_rows)
        for slice_name, rows in sorted(slice_rows.items()):
            _write_csv(out_dir / f"{split}_{slice_name}.csv", rows)
        if args.write_empty:
            for slice_name in (
                "dataset_failed",
                "api_low_quality",
                "graph_low_quality",
                "align_low_quality",
                "manifest_low_quality",
                "code_common_failure",
                "graph_semantic_missing",
                "manifest_parse_failed",
                "multi_dex_partial_failed",
            ):
                path = out_dir / f"{split}_{slice_name}.csv"
                if not path.exists():
                    _write_csv(path, [])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real extractor-failure slice CSVs from tri-modal .pt files.")
    parser.add_argument("--config", nargs="+", default=["config/experiments/tri_modal_robust/base_tri_modal_robust.yaml"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out-dir", default="results/robust_slices")
    parser.add_argument("--api-threshold", type=float, default=0.3)
    parser.add_argument("--graph-threshold", type=float, default=0.3)
    parser.add_argument("--align-threshold", type=float, default=0.2)
    parser.add_argument("--manifest-threshold", type=float, default=0.5)
    parser.add_argument("--common-api-threshold", type=float, default=0.4)
    parser.add_argument("--common-graph-threshold", type=float, default=0.4)
    parser.add_argument("--write-empty", action="store_true", help="Write header-only CSVs for empty slices.")
    return parser.parse_args()


if __name__ == "__main__":
    build_slices(parse_args())
