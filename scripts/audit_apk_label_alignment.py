#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _read_labels(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        lowered = {field.lower(): field for field in fields}
        id_field = next((lowered[key] for key in ("id", "sha256", "sid", "sample_id") if key in lowered), None)
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
    return fields, rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_one(path: Path, expected_ids: set[str]) -> dict[str, Any]:
    filename_id = path.stem.lower()
    size_bytes = path.stat().st_size
    actual_sha256 = ""
    if size_bytes <= 0:
        status = "zero_byte"
    elif not zipfile.is_zipfile(path):
        status = "non_zip_content"
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                if "AndroidManifest.xml" not in archive.namelist():
                    status = "zip_without_manifest"
                else:
                    actual_sha256 = _sha256_file(path)
                    status = "ok" if actual_sha256 == filename_id else "sha256_mismatch"
        except (OSError, zipfile.BadZipFile) as exc:
            status = f"read_error:{type(exc).__name__}"
    return {
        "filename_id": filename_id,
        "apk_name": path.name,
        "apk_path": str(path),
        "size_bytes": size_bytes,
        "actual_sha256": actual_sha256,
        "filename_in_label_csv": int(filename_id in expected_ids),
        "actual_sha_in_label_csv": int(bool(actual_sha256) and actual_sha256 in expected_ids),
        "status": status,
    }


def run(args: argparse.Namespace) -> None:
    apk_root = args.apk_root.resolve()
    label_fields, labels = _read_labels(args.labels.resolve())
    expected_ids = set(labels)
    paths = sorted(path for path in apk_root.iterdir() if path.is_file() and path.suffix.lower() == ".apk")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(
            tqdm(
                executor.map(lambda path: _audit_one(path, expected_ids), paths),
                total=len(paths),
                desc="audit APK/label alignment",
                unit="apk",
            )
        )

    found_filename_ids = {str(row["filename_id"]) for row in rows}
    missing_label_ids = sorted(expected_ids - found_filename_ids)
    extra_filename_ids = sorted(found_filename_ids - expected_ids)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename_id",
        "apk_name",
        "apk_path",
        "size_bytes",
        "actual_sha256",
        "filename_in_label_csv",
        "actual_sha_in_label_csv",
        "status",
        *[field for field in label_fields if field not in {"id", "sha256"}],
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**labels.get(str(row["filename_id"]), {}), **row})

    summary = output.with_name(f"{output.stem}_summary.csv")
    status_counts = Counter(str(row["status"]) for row in rows)
    summary_rows = [
        {"metric": "label_rows", "value": len(labels)},
        {"metric": "apk_files", "value": len(paths)},
        {"metric": "missing_label_ids", "value": len(missing_label_ids)},
        {"metric": "extra_filename_ids", "value": len(extra_filename_ids)},
        *[{"metric": f"status:{key}", "value": value} for key, value in sorted(status_counts.items())],
    ]
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"APK/label audit complete: labels={len(labels)} apks={len(paths)} "
        f"missing={len(missing_label_ids)} extra={len(extra_filename_ids)} "
        f"statuses={dict(status_counts)} report={output}"
    )
    if args.strict and (
        missing_label_ids
        or extra_filename_ids
        or status_counts != Counter({"ok": len(paths)})
        or len(paths) != len(labels)
    ):
        raise RuntimeError(f"APK directory is not one-to-one aligned with labels; inspect {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit APK filename/content SHA256 alignment with a label CSV.")
    parser.add_argument("--apk-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
