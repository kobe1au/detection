#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.manifest_features import extract_manifest_record


def _find_col(fieldnames: list[str] | None, candidates: list[str]) -> str | None:
    if not fieldnames:
        return None
    lower = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _iter_csv_apks(csv_path: Path, apk_dir: Path | None, id_col: str | None, apk_col: str | None):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        resolved_id_col = id_col or _find_col(reader.fieldnames, ["id", "sha256", "apk_name", "sample"])
        resolved_apk_col = apk_col or _find_col(reader.fieldnames, ["apk_path", "path", "file", "apk"])
        if resolved_id_col is None:
            raise ValueError(f"No id column found in {csv_path}")
        for row in reader:
            sid = str(row.get(resolved_id_col, "")).strip().lower()
            if not sid:
                continue
            apk_path = None
            if resolved_apk_col and row.get(resolved_apk_col):
                apk_path = Path(row[resolved_apk_col])
                if not apk_path.is_absolute() and apk_dir is not None:
                    apk_path = apk_dir / apk_path
            elif apk_dir is not None:
                candidates = [apk_dir / f"{sid}.apk", apk_dir / sid]
                apk_path = next((p for p in candidates if p.exists()), candidates[0])
            if apk_path is None:
                continue
            yield sid, apk_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AndroidManifest-derived raw features from APK files.")
    parser.add_argument("--csv", required=True, help="CSV containing sample ids and optionally apk_path.")
    parser.add_argument("--apk-dir", default="", help="Directory containing APKs when CSV paths are relative or absent.")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL manifest records.")
    parser.add_argument("--id-col", default="", help="Optional sample id column.")
    parser.add_argument("--apk-col", default="", help="Optional APK path column.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    apk_dir = Path(args.apk_dir) if args.apk_dir else None
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_csv_apks(csv_path, apk_dir, args.id_col or None, args.apk_col or None))
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, apk_path in tqdm(rows, desc="extract-manifest"):
            rec = extract_manifest_record(apk_path, sid=sid).to_json()
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
