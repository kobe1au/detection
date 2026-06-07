#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _read_labels(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        lowered = {name.lower(): name for name in fieldnames}
        id_field = next(
            (lowered[key] for key in ("id", "sha256", "sid", "sample_id") if key in lowered),
            None,
        )
        if id_field is None:
            raise ValueError(f"Label CSV has no id/sha256/sid column: {path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            sid = str(row.get(id_field) or "").strip().lower()
            if not sid:
                continue
            if sid in rows:
                raise ValueError(f"Duplicate id in label CSV: {sid}")
            rows[sid] = {key: str(value or "") for key, value in row.items() if key is not None}
    return fieldnames, rows


def _invalid_reason(path: Path) -> str:
    try:
        if path.stat().st_size <= 0:
            return "zero_byte"
        if not zipfile.is_zipfile(path):
            return "non_zip_content"
        with zipfile.ZipFile(path) as archive:
            if "AndroidManifest.xml" not in archive.namelist():
                return "zip_without_manifest"
    except (OSError, zipfile.BadZipFile):
        return "read_error"
    return ""


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    apk_root = args.apk_root.resolve()
    labels_path = args.labels.resolve()
    output_path = args.output.resolve()
    if not apk_root.is_dir():
        raise NotADirectoryError(f"APK root does not exist: {apk_root}")
    label_fields, labels = _read_labels(labels_path)

    apk_paths = sorted(path for path in apk_root.iterdir() if path.is_file() and path.suffix.lower() == ".apk")
    invalid: list[tuple[Path, str]] = []
    for path in tqdm(apk_paths, desc="validate APK containers", unit="apk"):
        reason = _invalid_reason(path)
        if reason:
            invalid.append((path, reason))

    if args.expect_invalid is not None and len(invalid) != args.expect_invalid:
        raise RuntimeError(
            f"Invalid APK count changed: expected={args.expect_invalid} actual={len(invalid)}. "
            "No files were deleted."
        )

    unlabeled = [path for path, _ in invalid if path.stem.lower() not in labels]
    if unlabeled and args.require_labeled:
        raise RuntimeError(
            f"{len(unlabeled)} invalid APK filenames are absent from label CSV; "
            f"examples={[path.name for path in unlabeled[:5]]}. No files were deleted."
        )

    extra_fields = ["invalid_reason", "apk_name", "apk_path", "size_bytes", "deletion_status"]
    fieldnames = [*label_fields, *[field for field in extra_fields if field not in label_fields]]
    rows: list[dict[str, Any]] = []
    for path, reason in invalid:
        sid = path.stem.lower()
        row: dict[str, Any] = dict(labels.get(sid) or {})
        row.update(
            {
                "invalid_reason": reason,
                "apk_name": path.name,
                "apk_path": str(path),
                "size_bytes": path.stat().st_size,
                "deletion_status": "pending" if args.delete else "dry_run",
            }
        )
        rows.append(row)

    # Persist the complete recovery list before deleting any input.
    _atomic_write_csv(output_path, fieldnames, rows)

    if args.delete:
        for row, (path, _) in tqdm(
            zip(rows, invalid),
            total=len(invalid),
            desc="delete invalid APKs",
            unit="apk",
        ):
            resolved = path.resolve()
            if resolved.parent != apk_root or resolved.suffix.lower() != ".apk":
                raise RuntimeError(f"Refusing to delete path outside APK root: {resolved}")
            try:
                resolved.unlink()
                row["deletion_status"] = "deleted"
            except OSError as exc:
                row["deletion_status"] = f"failed:{type(exc).__name__}:{exc}"
        _atomic_write_csv(output_path, fieldnames, rows)

    remaining = sum(1 for path in apk_root.iterdir() if path.is_file() and path.suffix.lower() == ".apk")
    reason_counts = dict(Counter(reason for _, reason in invalid))
    deletion_counts = dict(Counter(str(row["deletion_status"]).split(":", 1)[0] for row in rows))
    print(
        f"APK cleanup complete: scanned={len(apk_paths)} invalid={len(invalid)} "
        f"reasons={reason_counts} deletion={deletion_counts} remaining={remaining} report={output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report and optionally delete invalid APK containers.")
    parser.add_argument("--apk-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-invalid", type=int, default=None)
    parser.add_argument("--require-labeled", action="store_true")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
