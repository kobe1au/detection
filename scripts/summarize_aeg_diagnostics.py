#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.train import _binary_metrics


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _slice_specs(rows: list[dict[str, str]]) -> list[tuple[str, Callable[[dict[str, str]], bool]]]:
    specs: list[tuple[str, Callable[[dict[str, str]], bool]]] = [
        ("all", lambda _row: True),
        ("manifest_parse_failed", lambda row: _number(row, "manifest_parse_ok", 1.0) < 0.5),
        ("dex_partial_failed", lambda row: _number(row, "dex_success_ratio", 1.0) < 1.0),
        ("multi_dex", lambda row: _number(row, "multi_dex_total", 0.0) > 1.0),
        ("reflection_hint", lambda row: _number(row, "has_reflection") > 0.5),
        ("dynamic_loading_hint", lambda row: _number(row, "has_dynamic_loading") > 0.5),
        ("native_hint", lambda row: _number(row, "has_native") > 0.5),
        ("string_encryption_hint", lambda row: _number(row, "has_string_encryption_hint") > 0.5),
        ("low_code_reliability", lambda row: _number(row, "code_reliability", 1.0) < 0.5),
        ("low_manifest_reliability", lambda row: _number(row, "manifest_reliability", 1.0) < 0.5),
        ("high_code_manifest_conflict", lambda row: _number(row, "code_manifest_conflict") >= 0.5),
    ]
    years = sorted({int(_number(row, "year")) for row in rows if int(_number(row, "year")) > 0})
    specs.extend(
        (f"year_{year}", lambda row, target=year: int(_number(row, "year")) == target)
        for year in years
    )
    return specs


def _summarize(scenario: str, slice_name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = [int(_number(row, "label")) for row in rows]
    probs = [_number(row, "prob_malware") for row in rows]
    preds = [int(_number(row, "pred")) for row in rows]
    metrics = _binary_metrics(labels, probs, preds)
    return {
        "scenario": scenario,
        "slice": slice_name,
        "num_samples": len(rows),
        "positive_ratio": sum(labels) / max(1, len(labels)),
        **metrics,
    }


def run(input_dir: Path, output: Path, min_count: int) -> None:
    paths = sorted(input_dir.glob("diagnostics_test*.csv"))
    if not paths:
        raise FileNotFoundError(f"No diagnostics_test*.csv files found under {input_dir}")
    summaries: list[dict[str, Any]] = []
    for path in paths:
        rows = _read_rows(path)
        scenario = path.stem.removeprefix("diagnostics_test_")
        for slice_name, predicate in _slice_specs(rows):
            selected = [row for row in rows if predicate(row)]
            if len(selected) >= min_count:
                summaries.append(_summarize(scenario, slice_name, selected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} scenario/slice summaries to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clean, degraded, and real-failure AEG diagnostic slices.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or args.input_dir / "slice_metrics.csv"
    run(args.input_dir, output, max(1, args.min_count))
