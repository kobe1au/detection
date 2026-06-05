#!/usr/bin/env python
"""Delete files listed by a CSV id column from split-specific folders.

The script is dry-run by default. Pass --execute to actually delete files.
It is intended for cleanup CSVs such as pkg_conflict_removed_*.csv, where each
row has an id/sha256 and a split-like column indicating where the file lives.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - kept for minimal environments.
    yaml = None


DEFAULT_EXTENSIONS = (".apk", ".pt")


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        return _load_minimal_extract_config(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_minimal_extract_config(path: Path) -> dict[str, Any]:
    """Parse only data.split_dirs and data.out_dirs without PyYAML."""

    cfg: dict[str, Any] = {"data": {"split_dirs": {}, "out_dirs": {}}}
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
        if indent == 2 and stripped in ("split_dirs:", "out_dirs:"):
            current_mapping = stripped[:-1]
            continue
        if indent >= 4 and current_mapping and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip().strip("'\"")
            cfg["data"][current_mapping][key.strip()] = value
    return cfg


def _parse_root_assignments(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --root value {value!r}; expected split=path.")
        split, raw_path = value.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Invalid --root value {value!r}; split is empty.")
        roots[split] = Path(raw_path).expanduser()
    return roots


def _roots_from_config(config_path: Path, root_kind: str) -> dict[str, Path]:
    cfg = _load_yaml(config_path)
    data = cfg.get("data") or {}
    key = "split_dirs" if root_kind == "apk" else "out_dirs"
    raw_roots = data.get(key) or {}
    if not isinstance(raw_roots, dict) or not raw_roots:
        raise ValueError(f"No data.{key} mapping found in {config_path}.")
    return {str(split): Path(str(path)).expanduser() for split, path in raw_roots.items()}


def _choose_split(row: dict[str, str], split_col: str | None) -> str:
    if split_col:
        return (row.get(split_col) or "").strip()
    for col in ("removed_from", "split"):
        value = (row.get(col) or "").strip()
        if value:
            return value
    return ""


def _candidate_paths(root: Path, sample_id: str, extensions: tuple[str, ...], recursive: bool) -> list[Path]:
    candidates: list[Path] = []
    for ext in extensions:
        name = sample_id if sample_id.lower().endswith(ext.lower()) else f"{sample_id}{ext}"
        direct = root / name
        if direct.exists():
            candidates.append(direct)
        if recursive:
            pattern = name
            candidates.extend(p for p in root.rglob(pattern) if p.is_file() and p not in candidates)
    return candidates


def _write_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "id",
        "split",
        "path",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete APK/PT files by reading an id column and split column from a CSV. "
            "Dry-run by default; pass --execute to delete."
        )
    )
    parser.add_argument("--csv", required=True, help="CSV containing ids to delete.")
    parser.add_argument("--id-col", default="id", help="ID column used as file stem. Default: id.")
    parser.add_argument(
        "--split-col",
        default=None,
        help="Split column. Default: use removed_from if present, otherwise split.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config YAML. Reads data.split_dirs for --root-kind apk or data.out_dirs for pt.",
    )
    parser.add_argument(
        "--root-kind",
        choices=("apk", "pt"),
        default="apk",
        help="Which config root mapping to use. apk=data.split_dirs, pt=data.out_dirs.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Manual split root mapping, e.g. --root train=E:/resource/train. Can be repeated.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="File extensions to delete. Defaults to .apk for apk roots, .pt for pt roots.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search under each split root recursively.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Without this, dry-run only.")
    parser.add_argument(
        "--log",
        default=None,
        help="CSV log path. Default: <input_csv_stem>_delete_<dry_run|executed>.csv beside input CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    roots: dict[str, Path] = {}
    if args.config:
        roots.update(_roots_from_config(Path(args.config), args.root_kind))
    roots.update(_parse_root_assignments(args.root))
    if not roots:
        raise ValueError("No split roots provided. Use --config or --root split=path.")

    extensions = tuple(args.extensions or ([".apk"] if args.root_kind == "apk" else [".pt"]))
    extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)

    mode = "executed" if args.execute else "dry_run"
    log_path = (
        Path(args.log)
        if args.log
        else csv_path.with_name(f"{csv_path.stem}_delete_{args.root_kind}_{mode}.csv")
    )

    log_rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header.")
        if args.id_col not in reader.fieldnames:
            raise ValueError(f"Missing id column {args.id_col!r}; header={reader.fieldnames}")

        for row in reader:
            sample_id = (row.get(args.id_col) or "").strip()
            split = _choose_split(row, args.split_col)
            if not sample_id:
                log_rows.append({"status": "skipped", "id": "", "split": split, "path": "", "reason": "empty id"})
                continue
            if not split:
                log_rows.append({"status": "skipped", "id": sample_id, "split": "", "path": "", "reason": "empty split"})
                continue
            root = roots.get(split)
            if root is None:
                log_rows.append(
                    {"status": "skipped", "id": sample_id, "split": split, "path": "", "reason": "unknown split root"}
                )
                continue

            paths = _candidate_paths(root, sample_id, extensions, args.recursive)
            if not paths:
                log_rows.append({"status": "missing", "id": sample_id, "split": split, "path": "", "reason": "not found"})
                continue

            for path in paths:
                status = "would_delete"
                reason = "dry run"
                if args.execute:
                    try:
                        path.unlink()
                        status = "deleted"
                        reason = ""
                    except OSError as exc:
                        status = "error"
                        reason = str(exc)
                log_rows.append(
                    {
                        "status": status,
                        "id": sample_id,
                        "split": split,
                        "path": str(path),
                        "reason": reason,
                    }
                )

    _write_log(log_path, log_rows)
    counts: dict[str, int] = {}
    for row in log_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"mode={mode}")
    print(f"roots={{{', '.join(f'{k}: {v}' for k, v in roots.items())}}}")
    print(f"extensions={extensions}")
    print(f"log={log_path}")
    print(f"counts={counts}")
    if not args.execute:
        print("Dry-run only. Re-run with --execute to delete files.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
