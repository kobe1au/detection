#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_aeg_pts_direct import _load_config, _read_label_records, _resolve_path, _sha256_file  # noqa: E402


def _original_id(path: Path, labels: dict[str, dict[str, str]]) -> str:
    stem = path.stem.lower()
    if stem in labels:
        return stem
    for candidate in re.findall(r"[0-9a-f]{64}", stem):
        if candidate in labels:
            return candidate
    return ""


def run(config: Path, clean_labels: Path, output_dir: Path) -> None:
    cfg = _load_config(config)
    data = cfg.get("data", {}) or {}
    splits = [str(value) for value in data.get("splits", [])]
    split_dirs = {str(key): _resolve_path(value) for key, value in (data.get("split_dirs") or {}).items()}
    labels = _read_label_records(clean_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_rows: list[dict[str, str]] = []
    for split in splits:
        rows: list[dict[str, str]] = []
        for apk_path in sorted(split_dirs[split].glob("*.apk")):
            source_id = _original_id(apk_path, labels)
            if not source_id:
                missing_rows.append({"scenario": split, "apk_path": str(apk_path), "reason": "original_id_not_found"})
                continue
            source = labels[source_id]
            sha256 = _sha256_file(apk_path)
            rows.append(
                {
                    "id": sha256,
                    "sha256": sha256,
                    "label": source.get("label", ""),
                    "year": source.get("year", ""),
                    "split": split,
                    "source_id": source_id,
                    "apk_name": apk_path.name,
                }
            )
        with (output_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "sha256", "label", "year", "split", "source_id", "apk_name"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: labels={len(rows)} missing={sum(row['scenario'] == split for row in missing_rows)}")
    with (output_dir / "missing_original_ids.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "apk_path", "reason"])
        writer.writeheader()
        writer.writerows(missing_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map obfuscated APK hashes back to clean test labels.")
    parser.add_argument("--config", type=Path, default=Path("config/extract_obfuscapk.yaml"))
    parser.add_argument("--clean-labels", type=Path, default=Path("results/labels/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/labels_obfuscapk"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.clean_labels, args.output_dir)
