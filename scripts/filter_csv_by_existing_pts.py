#!/usr/bin/env python3
"""Filter split CSV files by existing .pt feature files.

For each split, this script compares the CSV id column, usually sha256, with
the file stems in the corresponding .pt directory. Rows whose sha256 does not
have a matching .pt file are removed from the CSV.

Default layout:
  resource/dataset_split_2018_2024/{train,val,test}.csv
  pts/{train,val,test}/*.pt
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_SPLITS = ("train", "val", "test")
ID_CANDIDATES = ("sha256", "id", "ID", "Id")


@dataclass
class SplitResult:
    split: str
    csv_path: Path
    pt_dir: Path
    id_col: str
    total_rows: int
    kept_rows: int
    removed_rows: int
    pt_count: int
    backup_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        default="resource/dataset_split_2018_2024",
        help="Directory containing train.csv, val.csv and test.csv.",
    )
    parser.add_argument(
        "--pt-root",
        default="pts",
        help="Directory containing train/val/test .pt subdirectories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to process.",
    )
    parser.add_argument(
        "--id-col",
        default="auto",
        help="CSV id column. Use 'auto' to search sha256/id/ID/Id.",
    )
    parser.add_argument(
        "--csv-template",
        default="{split}.csv",
        help="CSV filename template relative to --csv-dir.",
    )
    parser.add_argument(
        "--pt-template",
        default="{split}",
        help="PT directory template relative to --pt-root.",
    )
    parser.add_argument(
        "--missing-report",
        default="pt_missing_report.csv",
        help="Missing-row report path. Relative paths are written under --csv-dir. Use empty string to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print/report changes without modifying CSV files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create timestamped .bak files before overwriting CSVs.",
    )
    return parser.parse_args()


def normalize_sid(value: object) -> str:
    return str(value or "").strip().lower()


def find_id_col(fieldnames: list[str] | None, requested: str) -> str:
    if not fieldnames:
        raise ValueError("CSV has no header")
    if requested != "auto":
        if requested not in fieldnames:
            raise ValueError(f"CSV does not contain requested id column: {requested}")
        return requested
    for col in ID_CANDIDATES:
        if col in fieldnames:
            return col
    raise ValueError(f"CSV must contain one of id columns: {', '.join(ID_CANDIDATES)}")


def collect_pt_stems(pt_dir: Path) -> set[str]:
    if not pt_dir.exists():
        raise FileNotFoundError(f"PT directory does not exist: {pt_dir}")
    if not pt_dir.is_dir():
        raise NotADirectoryError(f"PT path is not a directory: {pt_dir}")
    return {p.stem.lower() for p in pt_dir.rglob("*.pt") if p.is_file()}


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv_atomic(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def process_split(
    split: str,
    csv_dir: Path,
    pt_root: Path,
    csv_template: str,
    pt_template: str,
    requested_id_col: str,
    dry_run: bool,
    make_backup: bool,
    stamp: str,
) -> tuple[SplitResult, list[dict[str, str]]]:
    csv_path = csv_dir / csv_template.format(split=split)
    pt_dir = pt_root / pt_template.format(split=split)

    fieldnames, rows = read_csv_rows(csv_path)
    id_col = find_id_col(fieldnames, requested_id_col)
    pt_stems = collect_pt_stems(pt_dir)

    kept_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    for row in rows:
        sid = normalize_sid(row.get(id_col, ""))
        if sid and sid in pt_stems:
            kept_rows.append(row)
        else:
            missing_rows.append({
                "split": split,
                "sid": sid,
                "reason": "missing_pt" if sid else "empty_id",
                **row,
            })

    backup_path: Path | None = None
    if not dry_run and len(kept_rows) != len(rows):
        if make_backup:
            backup_path = csv_path.with_suffix(csv_path.suffix + f".{stamp}.bak")
            shutil.copy2(csv_path, backup_path)
        write_csv_atomic(csv_path, fieldnames, kept_rows)

    result = SplitResult(
        split=split,
        csv_path=csv_path,
        pt_dir=pt_dir,
        id_col=id_col,
        total_rows=len(rows),
        kept_rows=len(kept_rows),
        removed_rows=len(missing_rows),
        pt_count=len(pt_stems),
        backup_path=backup_path,
    )
    return result, missing_rows


def write_missing_report(path: Path, missing_rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if missing_rows:
        base_fields = ["split", "sid", "reason"]
        extra_fields: list[str] = []
        for row in missing_rows:
            for key in row.keys():
                if key not in base_fields and key not in extra_fields:
                    extra_fields.append(key)
        fieldnames = base_fields + extra_fields
    else:
        fieldnames = ["split", "sid", "reason"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    pt_root = Path(args.pt_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_missing: list[dict[str, str]] = []
    results: list[SplitResult] = []

    for split in args.splits:
        result, missing_rows = process_split(
            split=split,
            csv_dir=csv_dir,
            pt_root=pt_root,
            csv_template=args.csv_template,
            pt_template=args.pt_template,
            requested_id_col=args.id_col,
            dry_run=args.dry_run,
            make_backup=not args.no_backup,
            stamp=stamp,
        )
        results.append(result)
        all_missing.extend(missing_rows)

    if args.missing_report:
        report_path = Path(args.missing_report)
        if not report_path.is_absolute():
            report_path = csv_dir / report_path
        write_missing_report(report_path, all_missing)
    else:
        report_path = None

    print("========== CSV/PT FILTER SUMMARY ==========")
    print(f"mode={'DRY-RUN' if args.dry_run else 'WRITE'}")
    for r in results:
        print(
            f"{r.split:5s} | csv={r.csv_path} | pt_dir={r.pt_dir} | "
            f"pt={r.pt_count} rows={r.total_rows} kept={r.kept_rows} "
            f"removed={r.removed_rows} id_col={r.id_col}"
        )
        if r.backup_path is not None:
            print(f"      backup={r.backup_path}")
    if report_path is not None:
        print(f"missing_report={report_path}")


if __name__ == "__main__":
    main()
