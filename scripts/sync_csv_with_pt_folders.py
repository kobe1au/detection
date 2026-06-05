#!/usr/bin/env python
"""Synchronize train/val/test CSV files with split-specific .pt folders.

Rules:
  1. If a CSV row id has no matching <id>.pt in the corresponding split folder,
     remove that CSV row.
  2. If a .pt file stem exists in the split folder but is not present in the
     corresponding CSV, delete that .pt file.

The script is dry-run by default. Pass --execute to actually update CSV files
and delete .pt files. Original CSVs are backed up before modification unless
--no-backup is provided.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - fallback parser handles current config.
    yaml = None


DEFAULT_SPLITS = ("train", "val", "test")
ID_CANDIDATES = ("id", "sha256", "ID", "Id")


@dataclass
class SplitSyncResult:
    split: str
    csv_path: Path
    pt_dir: Path
    id_col: str
    csv_rows_before: int
    csv_rows_after: int
    csv_only: int
    pt_files_before: int
    pt_only: int
    deleted_pt: int
    missing_pt_dir: bool = False
    backup_path: Path | None = None


def _load_minimal_extract_config(path: Path) -> dict[str, Any]:
    """Parse only data.out_dirs without PyYAML."""

    cfg: dict[str, Any] = {"data": {"out_dirs": {}}}
    in_data = False
    current_mapping: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_data = stripped == "data:"
            current_mapping = None
            continue
        if not in_data:
            continue
        if indent == 2 and stripped == "out_dirs:":
            current_mapping = "out_dirs"
            continue
        if indent >= 4 and current_mapping and ":" in stripped:
            key, value = stripped.split(":", 1)
            cfg["data"][current_mapping][key.strip()] = value.strip().strip("'\"")
    return cfg


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        return _load_minimal_extract_config(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def roots_from_config(path: Path) -> dict[str, Path]:
    cfg = load_config(path)
    out_dirs = ((cfg.get("data") or {}).get("out_dirs") or {})
    if not isinstance(out_dirs, dict) or not out_dirs:
        raise ValueError(f"No data.out_dirs mapping found in {path}.")
    return {str(split): Path(str(root)).expanduser() for split, root in out_dirs.items()}


def parse_root_assignments(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --pt-dir value {value!r}; expected split=path.")
        split, raw_path = value.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Invalid --pt-dir value {value!r}; split is empty.")
        roots[split] = Path(raw_path).expanduser()
    return roots


def normalize_id(value: object) -> str:
    return str(value or "").strip().lower()


def find_id_col(fieldnames: list[str], requested: str) -> str:
    if requested != "auto":
        if requested not in fieldnames:
            raise ValueError(f"CSV does not contain requested id column {requested!r}.")
        return requested
    for col in ID_CANDIDATES:
        if col in fieldnames:
            return col
    raise ValueError(f"CSV must contain one of id columns: {', '.join(ID_CANDIDATES)}")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return fieldnames, list(reader)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def collect_pt_files(pt_dir: Path, recursive: bool) -> dict[str, Path]:
    iterator = pt_dir.rglob("*.pt") if recursive else pt_dir.glob("*.pt")
    files: dict[str, Path] = {}
    duplicates: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        stem = path.stem.lower()
        if stem in files:
            duplicates.append(path)
            continue
        files[stem] = path
    if duplicates:
        raise ValueError(
            f"Duplicate .pt stems under {pt_dir}; first duplicate examples: "
            + ", ".join(str(p) for p in duplicates[:5])
        )
    return files


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["action", "status", "split", "id", "path", "reason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sync_split(
    split: str,
    csv_path: Path,
    pt_dir: Path,
    requested_id_col: str,
    recursive: bool,
    execute: bool,
    make_backup: bool,
    allow_missing_pt_dir: bool,
    stamp: str,
) -> tuple[SplitSyncResult, list[dict[str, str]]]:
    fieldnames, rows = read_csv_rows(csv_path)
    id_col = find_id_col(fieldnames, requested_id_col)

    if not pt_dir.exists():
        if not allow_missing_pt_dir:
            raise FileNotFoundError(f"PT directory does not exist for split {split}: {pt_dir}")
        result = SplitSyncResult(
            split=split,
            csv_path=csv_path,
            pt_dir=pt_dir,
            id_col=id_col,
            csv_rows_before=len(rows),
            csv_rows_after=len(rows),
            csv_only=0,
            pt_files_before=0,
            pt_only=0,
            deleted_pt=0,
            missing_pt_dir=True,
        )
        return result, [
            {
                "action": "skip_split",
                "status": "missing_pt_dir",
                "split": split,
                "id": "",
                "path": str(pt_dir),
                "reason": "pt directory does not exist",
            }
        ]
    if not pt_dir.is_dir():
        raise NotADirectoryError(f"PT path is not a directory for split {split}: {pt_dir}")

    pt_files = collect_pt_files(pt_dir, recursive=recursive)
    pt_ids = set(pt_files)

    kept_rows: list[dict[str, str]] = []
    csv_ids: set[str] = set()
    log_rows: list[dict[str, str]] = []
    for row in rows:
        sid = normalize_id(row.get(id_col, ""))
        if not sid:
            log_rows.append(
                {"action": "remove_csv_row", "status": "would_remove" if not execute else "removed", "split": split, "id": "", "path": str(csv_path), "reason": "empty id"}
            )
            continue
        if sid in pt_ids:
            kept_rows.append(row)
            csv_ids.add(sid)
        else:
            log_rows.append(
                {
                    "action": "remove_csv_row",
                    "status": "would_remove" if not execute else "removed",
                    "split": split,
                    "id": sid,
                    "path": str(csv_path),
                    "reason": "id exists in csv but matching .pt is missing",
                }
            )

    pt_only_ids = sorted(pt_ids - csv_ids)
    deleted_pt = 0
    for sid in pt_only_ids:
        path = pt_files[sid]
        if execute:
            try:
                path.unlink()
                deleted_pt += 1
                status = "deleted"
                reason = ""
            except OSError as exc:
                status = "error"
                reason = str(exc)
        else:
            status = "would_delete"
            reason = "id exists as .pt but not in csv"
        log_rows.append(
            {
                "action": "delete_pt_file",
                "status": status,
                "split": split,
                "id": sid,
                "path": str(path),
                "reason": reason,
            }
        )

    backup_path: Path | None = None
    csv_changed = len(kept_rows) != len(rows)
    if execute and csv_changed:
        if make_backup:
            backup_path = csv_path.with_suffix(csv_path.suffix + f".bak_sync_pt_{stamp}")
            shutil.copy2(csv_path, backup_path)
        write_csv_atomic(csv_path, fieldnames, kept_rows)

    result = SplitSyncResult(
        split=split,
        csv_path=csv_path,
        pt_dir=pt_dir,
        id_col=id_col,
        csv_rows_before=len(rows),
        csv_rows_after=len(kept_rows),
        csv_only=len(rows) - len(kept_rows),
        pt_files_before=len(pt_files),
        pt_only=len(pt_only_ids),
        deleted_pt=deleted_pt,
        backup_path=backup_path,
    )
    return result, log_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="results/labels", help="Directory containing split CSV files.")
    parser.add_argument("--csv-template", default="{split}.csv", help="CSV filename template.")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Splits to process.")
    parser.add_argument("--id-col", default="auto", help="CSV id column. Default: auto.")
    parser.add_argument("--config", default=None, help="Config YAML with data.out_dirs.")
    parser.add_argument("--pt-root", default=None, help="Root containing train/val/test .pt folders.")
    parser.add_argument("--pt-template", default="{split}", help="PT dir template under --pt-root.")
    parser.add_argument("--pt-dir", action="append", default=[], help="Manual mapping split=path. Can be repeated.")
    parser.add_argument("--recursive", action="store_true", help="Search .pt files recursively in each split dir.")
    parser.add_argument("--allow-missing-pt-dir", action="store_true", help="Skip splits whose PT dir is missing.")
    parser.add_argument("--execute", action="store_true", help="Actually update CSVs and delete extra .pt files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up CSVs before overwriting.")
    parser.add_argument("--log", default=None, help="Detailed action log CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    roots: dict[str, Path] = {}
    if args.config:
        roots.update(roots_from_config(Path(args.config)))
    if args.pt_root:
        pt_root = Path(args.pt_root)
        roots.update({split: pt_root / args.pt_template.format(split=split) for split in args.splits})
    roots.update(parse_root_assignments(args.pt_dir))
    if not roots:
        raise ValueError("No PT folders provided. Use --config, --pt-root, or --pt-dir split=path.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "executed" if args.execute else "dry_run"
    log_path = Path(args.log) if args.log else csv_dir / f"sync_csv_pt_{stamp}_{mode}.csv"

    all_logs: list[dict[str, str]] = []
    results: list[SplitSyncResult] = []
    for split in args.splits:
        if split not in roots:
            raise ValueError(f"No PT root configured for split {split!r}.")
        csv_path = csv_dir / args.csv_template.format(split=split)
        result, logs = sync_split(
            split=split,
            csv_path=csv_path,
            pt_dir=roots[split],
            requested_id_col=args.id_col,
            recursive=args.recursive,
            execute=args.execute,
            make_backup=not args.no_backup,
            allow_missing_pt_dir=args.allow_missing_pt_dir,
            stamp=stamp,
        )
        results.append(result)
        all_logs.extend(logs)

    write_log(log_path, all_logs)

    print("========== CSV/PT SYNC SUMMARY ==========")
    print(f"mode={mode}")
    for result in results:
        missing_note = " MISSING_PT_DIR" if result.missing_pt_dir else ""
        print(
            f"{result.split:5s}{missing_note} | csv={result.csv_path} | pt_dir={result.pt_dir} | "
            f"csv_rows={result.csv_rows_before}->{result.csv_rows_after} "
            f"csv_only_removed={result.csv_only} pt_files={result.pt_files_before} "
            f"pt_only={result.pt_only} pt_deleted={result.deleted_pt} id_col={result.id_col}"
        )
        if result.backup_path:
            print(f"      backup={result.backup_path}")
    print(f"log={log_path}")
    if not args.execute:
        print("Dry-run only. Re-run with --execute to update CSVs and delete extra .pt files.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
