#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.config_utils import load_config as _load_config  # noqa: E402
from fusion.constants import (  # noqa: E402
    AEG_PAYLOAD_CONTRACT_VERSION,
    AEG_SCHEMA_VERSION,
    stable_table_hash,
)
from fusion.io_utils import load_aeg_payload  # noqa: E402
from fusion.payload_contract import validate_aeg_payload  # noqa: E402


_WORKER_CFG: dict[str, Any] | None = None
_WORKER_VOCAB: dict[str, Any] | None = None


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _silence_third_party_logs() -> None:
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


def _resolve_split_dirs(data: dict[str, Any], splits: list[str]) -> dict[str, Path]:
    split_dirs_raw = data.get("split_dirs") or data.get("apk_dirs") or {}
    if split_dirs_raw:
        split_dirs_raw = {str(k): v for k, v in split_dirs_raw.items()}
        missing = [split for split in splits if not split_dirs_raw.get(split)]
        if missing:
            raise ValueError(f"data.split_dirs is missing configured splits: {missing}")
        return {split: _resolve_path(split_dirs_raw[split]) for split in splits}
    apk_root_raw = data.get("apk_root", "")
    if not apk_root_raw:
        raise ValueError("Either data.split_dirs or data.apk_root is required")
    apk_root = _resolve_path(apk_root_raw)
    return {split: apk_root / split for split in splits}


def _resolve_out_dirs(data: dict[str, Any], out_root: Path, splits: list[str]) -> dict[str, Path]:
    out_dirs_raw = data.get("out_dirs") or {}
    if out_dirs_raw:
        out_dirs_raw = {str(k): v for k, v in out_dirs_raw.items()}
        missing = [split for split in splits if not out_dirs_raw.get(split)]
        if missing:
            raise ValueError(f"data.out_dirs is missing configured splits: {missing}")
        return {split: _resolve_path(out_dirs_raw[split]) for split in splits}
    return {split: out_root / split for split in splits}


def _resolve_label_csvs(data: dict[str, Any], splits: list[str]) -> dict[str, Path]:
    raw = data.get("label_csvs") or {}
    if not raw:
        return {}
    raw = {str(k): v for k, v in raw.items()}
    missing = [split for split in splits if not raw.get(split)]
    if missing:
        raise ValueError(f"data.label_csvs is missing configured splits: {missing}")
    return {split: _resolve_path(raw[split]) for split in splits}


def _resolve_optional_train_label_csv(data: dict[str, Any]) -> Path | None:
    raw = data.get("label_csvs") or {}
    if not isinstance(raw, dict):
        return None
    value = {str(k): v for k, v in raw.items()}.get("train")
    return _resolve_path(value) if value else None


def _read_label_records(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Label CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {str(name).lower(): name for name in (reader.fieldnames or [])}
        id_field = next((fields[key] for key in ("sha256", "apk_sha256", "id", "sid", "sample_id") if key in fields), None)
        if id_field is None:
            raise ValueError(f"Label CSV must contain an id/sha256/sid column: {path}")
        records: dict[str, dict[str, str]] = {}
        for row in reader:
            sid = str(row.get(id_field) or "").strip().lower()
            if not sid:
                continue
            if sid in records:
                raise ValueError(f"Duplicate sample id in label CSV {path}: {sid}")
            records[sid] = {
                str(key): str(value or "").strip()
                for key, value in row.items()
                if key is not None
            }
    return records


def _read_label_ids(path: Path) -> set[str]:
    return set(_read_label_records(path))


def _filter_jobs_to_labels(
    jobs: list[dict[str, Any]],
    label_csvs: dict[str, Path],
    splits: list[str],
    *,
    require_all_label_ids: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not label_csvs:
        return jobs, []
    allowed = {split: _read_label_records(label_csvs[split]) for split in splits}
    for split, records in allowed.items():
        wrong_split = [
            sid
            for sid, row in records.items()
            if str(row.get("split") or "").strip() and str(row.get("split") or "").strip().lower() != split.lower()
        ]
        if wrong_split:
            raise ValueError(
                f"Label CSV mapped to split {split!r} contains rows declaring another split: "
                f"count={len(wrong_split)} examples={wrong_split[:5]}"
            )
    filtered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    present = {split: set() for split in splits}
    for job in jobs:
        split = job["split"]
        if job["sha256"] in allowed[split]:
            filtered_job = dict(job)
            row = allowed[split][job["sha256"]]
            filtered_job["sample_meta"] = {
                "year": row.get("year", ""),
                "pkg_name": row.get("pkg_name", ""),
                "market": row.get("market", ""),
                "source_split": row.get("source_split", ""),
            }
            filtered.append(filtered_job)
            present[split].add(job["sha256"])
        else:
            ignored.append(job)
    if require_all_label_ids:
        missing = {
            split: sorted(set(allowed[split]) - present[split])
            for split in splits
            if set(allowed[split]) - present[split]
        }
        if missing:
            examples = {split: ids[:5] for split, ids in missing.items()}
            counts = {split: len(ids) for split, ids in missing.items()}
            raise RuntimeError(f"Configured APK inputs do not cover all label CSV ids: counts={counts} examples={examples}")
    return filtered, ignored


def _read_hash_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {
            str(row.get("apk_path") or ""): dict(row)
            for row in csv.DictReader(f)
            if str(row.get("apk_path") or "") and str(row.get("sha256") or "")
        }


def _write_hash_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["apk_path", "size_bytes", "mtime_ns", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cache[key] for key in sorted(cache))


def _apk_container_status(path: Path) -> str:
    try:
        if path.stat().st_size <= 0:
            return "zero_byte"
        if not zipfile.is_zipfile(path):
            return "non_zip_content"
        with zipfile.ZipFile(path) as archive:
            return "valid_apk_container" if "AndroidManifest.xml" in archive.namelist() else "zip_without_manifest"
    except (OSError, zipfile.BadZipFile):
        return "read_error"


def _scan_invalid_apk_containers(
    split_dirs: dict[str, Path],
    splits: list[str],
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for split in splits:
        split_dir = split_dirs[split]
        if not split_dir.exists():
            continue
        paths = sorted(p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() == ".apk")
        if workers > 1 and paths:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                statuses = list(
                    tqdm(
                        ex.map(_apk_container_status, paths),
                        total=len(paths),
                        desc=f"validate APK {split}",
                        unit="apk",
                    )
                )
        else:
            statuses = [
                _apk_container_status(path)
                for path in tqdm(paths, desc=f"validate APK {split}", unit="apk")
            ]
        for path, status in zip(paths, statuses):
            if status != "valid_apk_container":
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    size_bytes = -1
                invalid.append(
                    {
                        "split": split,
                        "apk_name": path.name,
                        "apk_path": str(path.resolve()),
                        "filename_id": path.stem.lower(),
                        "size_bytes": size_bytes,
                        "status": status,
                    }
                )
    return invalid


def _write_invalid_apk_report(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "apk_name", "apk_path", "filename_id", "size_bytes", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _invalid_apks_required_by_labels(
    rows: list[dict[str, Any]],
    label_csvs: dict[str, Path],
) -> list[dict[str, Any]]:
    if not label_csvs:
        return rows
    allowed = {
        split: _read_label_ids(label_path)
        for split, label_path in label_csvs.items()
    }
    return [
        row
        for row in rows
        if str(row.get("filename_id") or "").lower() in allowed.get(str(row.get("split") or ""), set())
    ]


def _cached_sha256(
    path: Path,
    stat: Any,
    cache: dict[str, dict[str, str]],
) -> tuple[str | None, bool]:
    key = str(path.resolve())
    row = cache.get(key)
    if (
        row
        and str(row.get("size_bytes") or "") == str(stat.st_size)
        and str(row.get("mtime_ns") or "") == str(stat.st_mtime_ns)
        and len(str(row.get("sha256") or "")) == 64
    ):
        return str(row["sha256"]).lower(), True
    return None, False


def _collect_apks(
    split_dirs: dict[str, Path],
    splits: list[str],
    *,
    hash_files: bool = True,
    hash_workers: int = 1,
    hash_cache_path: Path | None = None,
    excluded_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    cache = _read_hash_cache(hash_cache_path) if hash_files and hash_cache_path else {}
    excluded = excluded_paths or set()
    jobs: list[dict[str, Any]] = []
    for split in splits:
        split_dir = split_dirs[split]
        if not split_dir.exists():
            continue
        paths = sorted(
            p
            for p in split_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".apk"
            and str(p.resolve()) not in excluded
        )
        stats = [path.stat() for path in paths]
        hashes: list[str | None] = [None] * len(paths)
        cache_hits = [False] * len(paths)
        pending: list[tuple[int, Path]] = []
        if hash_files:
            for index, (path, stat) in enumerate(zip(paths, stats)):
                sha256, hit = _cached_sha256(path, stat, cache)
                hashes[index] = sha256
                cache_hits[index] = hit
                if sha256 is None:
                    pending.append((index, path))
        if hash_files and hash_workers > 1 and pending:
            with ThreadPoolExecutor(max_workers=hash_workers) as ex:
                computed = list(
                    tqdm(
                        ex.map(_sha256_file, (path for _, path in pending)),
                        total=len(pending),
                        desc=f"scan/hash {split}",
                        unit="apk",
                    )
                )
            for (index, _), sha256 in zip(pending, computed):
                hashes[index] = sha256
        elif hash_files and pending:
            computed = [
                _sha256_file(path)
                for _, path in tqdm(pending, desc=f"scan/hash {split}", unit="apk")
            ]
            for (index, _), sha256 in zip(pending, computed):
                hashes[index] = sha256
        elif not hash_files:
            hashes = [path.stem.lower() for path in paths]
        for apk_path, stat, sha256, cache_hit in zip(paths, stats, hashes, cache_hits):
            if sha256 is None:
                raise RuntimeError(f"Failed to compute SHA256 for {apk_path}")
            resolved_path = str(apk_path.resolve())
            cache[resolved_path] = {
                "apk_path": resolved_path,
                "size_bytes": str(stat.st_size),
                "mtime_ns": str(stat.st_mtime_ns),
                "sha256": sha256,
            }
            jobs.append(
                {
                    "split": split,
                    "apk_path": resolved_path,
                    "apk_name": apk_path.name,
                    "sha256": sha256,
                    "filename_id": apk_path.stem.lower(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "hash_cache_hit": cache_hit,
                }
            )
    if hash_files and hash_cache_path:
        _write_hash_cache(hash_cache_path, cache)
    return jobs


def _write_apk_scan_index(
    jobs: list[dict[str, Any]],
    label_csvs: dict[str, Path],
    path: Path,
) -> None:
    allowed = {
        split: _read_label_ids(label_path)
        for split, label_path in label_csvs.items()
    }
    rows: list[dict[str, Any]] = []
    for job in jobs:
        split = str(job["split"])
        filename_id = str(job.get("filename_id") or "").lower()
        actual_sha256 = str(job["sha256"]).lower()
        split_allowed = allowed.get(split, set())
        size_bytes = int(job.get("size_bytes") or 0)
        filename_in_labels = filename_id in split_allowed
        actual_in_labels = actual_sha256 in split_allowed
        if size_bytes == 0:
            status = "zero_byte"
        elif actual_sha256 == filename_id and actual_in_labels:
            status = "ok"
        elif actual_in_labels:
            status = "actual_hash_in_labels_wrong_filename"
        elif filename_in_labels:
            status = "content_hash_mismatch"
        else:
            status = "not_in_label_csv"
        rows.append(
            {
                "split": split,
                "apk_name": job["apk_name"],
                "apk_path": job["apk_path"],
                "filename_id": filename_id,
                "actual_sha256": actual_sha256,
                "size_bytes": size_bytes,
                "filename_in_label_csv": int(filename_in_labels),
                "actual_sha_in_label_csv": int(actual_in_labels),
                "hash_cache_hit": int(bool(job.get("hash_cache_hit", False))),
                "status": status,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "apk_name",
        "apk_path",
        "filename_id",
        "actual_sha256",
        "size_bytes",
        "filename_in_label_csv",
        "actual_sha_in_label_csv",
        "hash_cache_hit",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_split_counts(
    jobs: list[dict[str, Any]],
    splits: list[str],
    *,
    allow_empty_splits: bool = False,
) -> None:
    counts = {split: 0 for split in splits}
    for job in jobs:
        counts[job["split"]] += 1
    empty = [split for split, count in counts.items() if count <= 0]
    if empty and not allow_empty_splits:
        raise RuntimeError(f"No APK files found for configured splits: {empty}")
    if not jobs:
        raise RuntimeError("No APK files found for any configured split")


def _validate_unique_hashes(jobs: list[dict[str, Any]]) -> None:
    seen: dict[str, list[str]] = {}
    for job in jobs:
        seen.setdefault(job["sha256"], []).append(f"{job['split']}:{job['apk_path']}")
    duplicates = {sha: paths for sha, paths in seen.items() if len(paths) > 1}
    if duplicates:
        examples = [{"sha256": sha, "locations": paths[:5]} for sha, paths in list(duplicates.items())[:5]]
        raise RuntimeError(f"Duplicate APK hashes detected across configured inputs; count={len(duplicates)} examples={examples}")


def _parse_config(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data = raw.get("data", {}) or {}
    graph = raw.get("graph", {}) or {}
    api = raw.get("api", {}) or {}
    manifest = raw.get("manifest", {}) or {}
    aeg = raw.get("aeg", {}) or {}
    execution = raw.get("execution", {}) or {}
    out_root_raw = data.get("out_root", "")
    if not out_root_raw:
        raise ValueError("data.out_root is required")
    out_root = _resolve_path(out_root_raw)
    vocab_only = bool(getattr(args, "vocab_only", False))
    configured_splits = [str(v) for v in data.get("splits", ["train", "val", "test"])]
    if vocab_only and "train" not in configured_splits:
        raise ValueError("--vocab-only requires train in data.splits")
    splits = ["train"] if vocab_only else configured_splits
    split_dirs = _resolve_split_dirs(data, splits)
    out_dirs = _resolve_out_dirs(data, out_root, splits)
    label_csvs = _resolve_label_csvs(data, splits)
    train_label_csv_for_vocab = _resolve_optional_train_label_csv(data)
    vocab_path = _resolve_path(manifest.get("vocab_path", "config/extract/manifest_vocab_aeg.yaml"))
    rebuild_vocab = bool(args.rebuild_vocab if args.rebuild_vocab is not None else manifest.get("rebuild_vocab", False))
    resume = bool(args.resume if args.resume is not None else execution.get("resume", True))
    vocab_will_be_built = rebuild_vocab or not vocab_path.exists()
    if vocab_will_be_built and resume and not vocab_only:
        raise ValueError(
            "Manifest vocab will be built/rebuilt, but resume=true may skip existing .pt files. "
            "Use --no-resume when building/rebuilding vocab."
        )
    node_feature_dim = int(aeg.get("node_feature_dim", 128))
    storage_dtype = str(aeg.get("storage_dtype", "float16")).lower()
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"aeg.storage_dtype must be float16 or float32; got {storage_dtype!r}")
    vocab_size = int(graph.get("vocab_size", 256))
    use_graph_behavior_hints = bool(graph.get("use_behavior_hints", False))
    graph_behavior_hint_start = vocab_size * 2 + 3
    graph_behavior_hint_dim = 4 if use_graph_behavior_hints else 0
    if use_graph_behavior_hints:
        required_dim = graph_behavior_hint_start + graph_behavior_hint_dim
        if node_feature_dim < required_dim:
            raise ValueError(
                "graph.use_behavior_hints=true requires aeg.node_feature_dim >= "
                f"{required_dim} (2 * graph.vocab_size + 3 + 4); got {node_feature_dim}. "
                "Use config/extract/extract_aeg_behavior_hints.yaml or disable behavior hints."
            )
    workers = int(args.workers or execution.get("workers", 1))
    hash_workers = int(execution.get("hash_workers", min(workers, 4)))
    max_methods_per_dex = int(graph.get("max_methods_per_dex", 4096))
    max_methods_per_apk = int(graph.get("max_methods_per_apk", max_methods_per_dex))
    max_api_events_per_dex = int(api.get("max_events_per_dex", 1024))
    max_api_events_per_apk = int(api.get("max_events_per_apk", max_api_events_per_dex * 4))
    positive_values = {
        "execution.workers": workers,
        "execution.hash_workers": hash_workers,
        "aeg.node_feature_dim": node_feature_dim,
        "manifest.manifest_dim": int(manifest.get("manifest_dim", 256)),
        "graph.vocab_size": vocab_size,
        "graph.max_methods_per_dex": max_methods_per_dex,
        "graph.max_methods_per_apk": max_methods_per_apk,
        "api.num_hash_buckets": int(api.get("num_hash_buckets", 8192)),
        "api.max_events_per_dex": max_api_events_per_dex,
        "api.max_events_per_apk": max_api_events_per_apk,
    }
    invalid = {key: value for key, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Extraction config values must be positive: {invalid}")
    return {
        "splits": splits,
        "split_dirs": split_dirs,
        "out_root": out_root,
        "out_dirs": out_dirs,
        "label_csvs": label_csvs,
        "train_label_csv_for_vocab": train_label_csv_for_vocab,
        "filter_to_label_csv": bool(data.get("filter_to_label_csv", bool(label_csvs))),
        "require_all_label_ids": bool(data.get("require_all_label_ids", False)),
        "allow_empty_splits": bool(execution.get("allow_empty_splits", False)),
        "hash_files": bool(data.get("hash_files", True)),
        "hash_cache_path": out_root / "aeg_apk_hash_cache.csv",
        "scan_index_path": out_root / "aeg_apk_scan_index.csv",
        "invalid_apk_report_path": out_root / "aeg_invalid_apks.csv",
        "vocab_path": vocab_path,
        "rebuild_vocab": rebuild_vocab,
        "manifest_dim": int(manifest.get("manifest_dim", 256)),
        "max_permissions": int(manifest.get("max_permissions", 128)),
        "max_intents": int(manifest.get("max_intents", 64)),
        "max_features": int(manifest.get("max_features", 32)),
        "allow_empty_vocab": bool(manifest.get("allow_empty_vocab", False)),
        "node_feature_dim": node_feature_dim,
        "vocab_size": vocab_size,
        "sensitive_hops": int(graph.get("sensitive_hops", 1)),
        "max_methods_per_dex": max_methods_per_dex,
        "max_methods_per_apk": max_methods_per_apk,
        "fallback_max_methods": int(graph.get("fallback_max_methods", 512)),
        "fallback_policy": str(graph.get("fallback_policy", "api_rich")),
        "use_graph_behavior_hints": use_graph_behavior_hints,
        "graph_behavior_hint_start": graph_behavior_hint_start,
        "graph_behavior_hint_dim": graph_behavior_hint_dim,
        "num_api_buckets": int(api.get("num_hash_buckets", 8192)),
        "max_api_events_per_dex": max_api_events_per_dex,
        "max_api_events_per_apk": max_api_events_per_apk,
        "max_api_events_per_method": int(api.get("max_events_per_method", 32)),
        "api_event_scope": str(api.get("event_scope", "all_methods")),
        "framework_only": bool(api.get("framework_only", True)),
        "include_descriptor": bool(api.get("include_descriptor", False)),
        "keep_method_names": bool(aeg.get("keep_method_names", True)),
        "keep_api_tokens": bool(aeg.get("keep_api_tokens", True)),
        "retain_intermediate_features": bool(aeg.get("retain_intermediate_features", False)),
        "storage_dtype": storage_dtype,
        "workers": workers,
        "hash_workers": hash_workers,
        "validate_apk_containers": bool(execution.get("validate_apk_containers", True)),
        "resume": resume,
        "vocab_only": vocab_only,
        "fail_on_error": bool(execution.get("fail_on_error", False)),
    }


def _extract_manifest_one(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _silence_third_party_logs()
    from fusion.manifest_features import extract_manifest_record

    try:
        rec = extract_manifest_record(job["apk_path"], sid=job["sha256"]).to_json()
        rec["sid"] = job["sha256"]
        rec["sha256"] = job["sha256"]
        sample_meta = dict(job.get("sample_meta") or {})
        rec["sample_meta"] = sample_meta
        try:
            rec["year"] = int(sample_meta.get("year") or 0)
        except (TypeError, ValueError):
            rec["year"] = 0
        if not rec.get("package_name"):
            rec["package_name"] = str(sample_meta.get("pkg_name") or "").strip().lower()
        return job["sha256"], rec
    except Exception as exc:
        sample_meta = dict(job.get("sample_meta") or {})
        try:
            year = int(sample_meta.get("year") or 0)
        except (TypeError, ValueError):
            year = 0
        return job["sha256"], {
            "sid": job["sha256"],
            "sha256": job["sha256"],
            "apk_name": job["apk_name"],
            "package_name": str(sample_meta.get("pkg_name") or "").strip().lower(),
            "year": year,
            "sample_meta": sample_meta,
            "parse_error": f"{type(exc).__name__}: {exc}",
        }


def _extract_manifest_records(jobs: list[dict[str, Any]], *, workers: int = 1) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if workers <= 1:
        iterator = (_extract_manifest_one(job) for job in jobs)
        for sid, rec in tqdm(iterator, total=len(jobs), desc="extract manifests", unit="apk"):
            records[sid] = rec
        return records

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_manifest_one, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract manifests", unit="apk"):
            job = futures[future]
            try:
                sid, rec = future.result()
            except Exception as exc:
                sid = job["sha256"]
                sample_meta = dict(job.get("sample_meta") or {})
                try:
                    year = int(sample_meta.get("year") or 0)
                except (TypeError, ValueError):
                    year = 0
                rec = {
                    "sid": sid,
                    "sha256": sid,
                    "apk_name": job["apk_name"],
                    "package_name": str(sample_meta.get("pkg_name") or "").strip().lower(),
                    "year": year,
                    "sample_meta": sample_meta,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            records[sid] = rec
    return records


def _build_or_load_vocab(jobs: list[dict[str, Any]], records: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    from fusion.manifest_features import build_manifest_vocab, load_manifest_vocab, save_manifest_vocab, validate_manifest_vocab

    if cfg["rebuild_vocab"] or not cfg["vocab_path"].exists():
        train_records = [records[job["sha256"]] for job in jobs if job["split"] == "train"]
        if not train_records:
            raise RuntimeError("Cannot build Manifest vocab: no train manifest records found")
        vocab = build_manifest_vocab(
            train_records,
            max_permissions=cfg["max_permissions"],
            max_intents=cfg["max_intents"],
            max_features=cfg["max_features"],
        )
        vocab["metadata"] = {
            "source_split": "train",
            "source": "scripts/build_aeg_pts_direct.py",
            "leakage_guard": "train_only",
            "schema_version": AEG_SCHEMA_VERSION,
            "source_sample_count": len(train_records),
            "source_id_fingerprint": stable_table_hash(
                sorted(str(job["sha256"]).lower() for job in jobs if job["split"] == "train")
            ),
            "manifest_parse_success_count": sum(1 for record in train_records if not record.get("parse_error")),
        }
        validate_manifest_vocab(vocab, require_train_metadata=True, allow_empty=cfg["allow_empty_vocab"])
        _validate_vocab_matches_train_csv(vocab, cfg)
        save_manifest_vocab(vocab, cfg["vocab_path"])
        return vocab
    vocab = load_manifest_vocab(cfg["vocab_path"], require_train_metadata=True, allow_empty=cfg["allow_empty_vocab"])
    _validate_vocab_matches_train_csv(vocab, cfg)
    return vocab


def _validate_vocab_matches_train_csv(vocab: dict[str, Any], cfg: dict[str, Any]) -> None:
    train_label_csv = cfg.get("train_label_csv_for_vocab")
    if not train_label_csv:
        return
    train_label_csv = Path(train_label_csv)
    if not train_label_csv.exists():
        return
    train_ids = sorted(_read_label_ids(train_label_csv))
    expected_count = len(train_ids)
    metadata = vocab.get("metadata") or {}
    try:
        observed_count = int(metadata.get("source_sample_count"))
    except (TypeError, ValueError):
        observed_count = -1
    if observed_count != expected_count:
        raise ValueError(
            "Manifest vocab metadata does not match the configured train CSV. "
            f"expected_count={expected_count} observed_count={observed_count} "
            "Rebuild the Manifest vocab from the current train split."
        )


def _index_row(job: dict[str, Any], out_path: Path, status: str, reason: str = "") -> dict[str, str]:
    return {
        "split": job["split"],
        "sha256": job["sha256"],
        "apk_name": job["apk_name"],
        "apk_path": job["apk_path"],
        "pt_path": str(out_path),
        "status": status,
        "reason": reason,
    }


def _out_path(job: dict[str, Any], cfg: dict[str, Any]) -> Path:
    return cfg["out_dirs"][job["split"]] / f"{job['sha256']}.pt"


def _resume_existing(job: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str] | None:
    """Check if an existing PT matches required versions and tensor structure."""
    if not cfg["resume"]:
        return None
    out_path = _out_path(job, cfg)
    if not out_path.exists():
        return None
    try:
        existing = load_aeg_payload(out_path, validate=False)

        # ✅ Check schema version (affects PT structure)
        if existing.get("schema_version") != AEG_SCHEMA_VERSION:
            return None
        # ✅ Check contract version (required fields list)
        if existing.get("aeg_payload_contract_version") != AEG_PAYLOAD_CONTRACT_VERSION:
            return None

        # ✅ Check node feature dimension (affects model input)
        node_x = existing.get("node_x")
        if node_x is None or node_x.shape[1] != cfg["node_feature_dim"]:
            return None

        # ✅ Check sample ID matches
        if str(existing.get("sid") or "").lower() != job["sha256"].lower():
            return None

        # ✅ Check required fields exist
        required = ["node_x", "edge_index", "node_type", "edge_type", "node_source",
                    "edge_source", "node_quality", "edge_quality", "node_semantic"]
        if not all(field in existing for field in required):
            return None

        return _index_row(job, out_path, "ok", "resume")
    except Exception:
        return None
    return None


def _process_one(
    job: dict[str, Any],
    cfg: dict[str, Any],
    vocab: dict[str, Any],
    record: dict[str, Any] | None,
) -> tuple[bool, dict[str, str]]:
    _silence_third_party_logs()
    from androguard.core.dex import DEX
    from extract.extract_graph_api import atomic_torch_save, build_graph_api_for_dex, list_dex_entries
    from fusion.aeg_builder import build_aeg_payload
    from fusion.manifest_features import vectorize_manifest_record

    apk_path = Path(job["apk_path"])
    out_path = _out_path(job, cfg)
    resume_row = _resume_existing(job, cfg)
    if resume_row is not None:
        return True, resume_row

    try:
        if record is None:
            _sid, record = _extract_manifest_one(job)
        dex_entries = list_dex_entries(apk_path)
        dex_count = max(1, len(dex_entries))
        method_budget_per_dex = max(
            1,
            min(cfg["max_methods_per_dex"], cfg["max_methods_per_apk"] // dex_count),
        )
        api_budget_per_dex = max(
            1,
            min(cfg["max_api_events_per_dex"], cfg["max_api_events_per_apk"] // dex_count),
        )
        dex_results: list[dict[str, Any]] = []
        failed = 0
        with zipfile.ZipFile(apk_path, "r") as zf:
            for dex_name in dex_entries:
                try:
                    raw = zf.read(dex_name)
                    dex = DEX(raw)
                    dex_results.append(
                        build_graph_api_for_dex(
                            dex,
                            raw,
                            vocab_size=cfg["vocab_size"],
                            keep_method_names=cfg["keep_method_names"],
                            keep_api_tokens=cfg["keep_api_tokens"],
                            sensitive_hops=cfg["sensitive_hops"],
                            max_methods_per_dex=method_budget_per_dex,
                            fallback_max_methods=cfg["fallback_max_methods"],
                            num_api_buckets=cfg["num_api_buckets"],
                            max_api_events_per_dex=api_budget_per_dex,
                            max_api_events_per_method=cfg["max_api_events_per_method"],
                            api_event_scope=cfg["api_event_scope"],
                            framework_only=cfg["framework_only"],
                            include_descriptor=cfg["include_descriptor"],
                            fallback_policy=cfg["fallback_policy"],
                            use_graph_behavior_hints=cfg["use_graph_behavior_hints"],
                        )
                    )
                except Exception:
                    failed += 1
        manifest_payload = vectorize_manifest_record(record, vocab, manifest_dim=cfg["manifest_dim"])
        direct_meta = {
            "num_dex_total": len(dex_entries),
            "num_dex_success": len(dex_results),
            "dex_success_ratio": len(dex_results) / max(1, len(dex_entries)),
            "num_dex_failed": failed,
            "method_budget_per_dex": method_budget_per_dex,
            "api_event_budget_per_dex": api_budget_per_dex,
            "use_graph_behavior_hints": bool(cfg.get("use_graph_behavior_hints", False)),
            "graph_behavior_hint_start": int(cfg["graph_behavior_hint_start"]),
            "graph_behavior_hint_dim": int(cfg["graph_behavior_hint_dim"]),
        }
        payload = build_aeg_payload(
            sid=job["sha256"],
            apk_name=apk_path.name,
            split=job["split"],
            dex_list=dex_results,
            manifest_payload=manifest_payload,
            manifest_record=record,
            direct_meta=direct_meta,
            node_feature_dim=cfg["node_feature_dim"],
            retain_intermediate_features=cfg["retain_intermediate_features"],
            storage_dtype=cfg["storage_dtype"],
        )
        _validate_payload_for_save(payload, cfg, job)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(payload, out_path)
        warnings: list[str] = []
        if record.get("parse_error"):
            warnings.append(f"manifest_parse_error={record['parse_error']}")
        if failed:
            warnings.append(f"dex_failed={failed}/{len(dex_entries)}")
        if not dex_entries:
            warnings.append("no_dex_entries")
        return True, _index_row(job, out_path, "ok", "; ".join(warnings))
    except Exception as exc:
        cleanup_error = ""
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception as cleanup_exc:
                cleanup_error = f"; stale_pt_cleanup_error={type(cleanup_exc).__name__}: {cleanup_exc}"
        return False, _index_row(job, out_path, "failed", f"{type(exc).__name__}: {exc}{cleanup_error}")


def _init_build_worker(cfg: dict[str, Any], vocab: dict[str, Any]) -> None:
    global _WORKER_CFG, _WORKER_VOCAB
    _WORKER_CFG = cfg
    _WORKER_VOCAB = vocab
    _silence_third_party_logs()


def _process_one_worker(job: dict[str, Any], record: dict[str, Any] | None) -> tuple[bool, dict[str, str]]:
    if _WORKER_CFG is None or _WORKER_VOCAB is None:
        raise RuntimeError("AEG build worker was not initialized")
    return _process_one(job, _WORKER_CFG, _WORKER_VOCAB, record)


def _write_index(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["split", "sha256", "apk_name", "apk_path", "pt_path", "status", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _merge_index_rows(path: Path, rows: list[dict[str, str]], current_splits: list[str]) -> list[dict[str, str]]:
    current = set(current_splits)
    preserved = [row for row in _read_index(path) if str(row.get("split") or "") not in current]
    merged = [*preserved, *rows]
    merged.sort(key=lambda r: (str(r.get("split") or ""), str(r.get("apk_name") or "")))
    return merged


def _validate_payload_for_save(payload: dict[str, Any], cfg: dict[str, Any], job: dict[str, Any]) -> None:
    validate_aeg_payload(
        payload,
        expected_node_feature_dim=cfg["node_feature_dim"],
    )
    if str(payload["sid"]).lower() != job["sha256"].lower():
        raise ValueError("Generated AEG sid does not match APK SHA256")


def run(args: argparse.Namespace) -> None:
    cfg = _parse_config(_load_config(Path(args.config)), args)
    invalid_apks: list[dict[str, Any]] = []
    if cfg["validate_apk_containers"]:
        invalid_apks = _scan_invalid_apk_containers(
            cfg["split_dirs"],
            cfg["splits"],
            workers=cfg["hash_workers"],
        )
        _write_invalid_apk_report(invalid_apks, cfg["invalid_apk_report_path"])
        if invalid_apks:
            blocking_invalid = (
                _invalid_apks_required_by_labels(invalid_apks, cfg["label_csvs"])
                if cfg["filter_to_label_csv"]
                else invalid_apks
            )
            reported_invalid = blocking_invalid or invalid_apks
            counts: dict[str, int] = {}
            for row in reported_invalid:
                key = f"{row['split']}:{row['status']}"
                counts[key] = counts.get(key, 0) + 1
            message = (
                "Invalid APK containers detected before hashing; "
                f"counts={counts}; inspect report: {cfg['invalid_apk_report_path']}"
            )
            if blocking_invalid and cfg["require_all_label_ids"]:
                raise RuntimeError(message)
            print(f"WARNING: {message}")
    jobs = _collect_apks(
        cfg["split_dirs"],
        cfg["splits"],
        hash_files=cfg["hash_files"],
        hash_workers=cfg["hash_workers"],
        hash_cache_path=cfg["hash_cache_path"],
        excluded_paths={str(row["apk_path"]) for row in invalid_apks},
    )
    _write_apk_scan_index(jobs, cfg["label_csvs"], cfg["scan_index_path"])
    cache_hits = sum(int(bool(job.get("hash_cache_hit", False))) for job in jobs)
    print(
        f"APK scan index ready: {cfg['scan_index_path']} "
        f"(files={len(jobs)} hash_cache_hits={cache_hits})"
    )
    ignored_jobs: list[dict[str, Any]] = []
    if cfg["filter_to_label_csv"]:
        if not cfg["label_csvs"]:
            raise ValueError("data.filter_to_label_csv=true requires data.label_csvs")
        try:
            jobs, ignored_jobs = _filter_jobs_to_labels(
                jobs,
                cfg["label_csvs"],
                cfg["splits"],
                require_all_label_ids=cfg["require_all_label_ids"],
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; inspect APK scan report: {cfg['scan_index_path']}") from exc
        if ignored_jobs:
            print(f"Ignoring {len(ignored_jobs)} APK files not present in configured label CSVs.")
    _validate_split_counts(
        jobs,
        cfg["splits"],
        allow_empty_splits=cfg["allow_empty_splits"],
    )
    _validate_unique_hashes(jobs)
    vocab_needs_build = cfg["rebuild_vocab"] or not cfg["vocab_path"].exists()
    records: dict[str, dict[str, Any]] = {}
    if vocab_needs_build:
        train_jobs = [job for job in jobs if job["split"] == "train"]
        if not train_jobs:
            raise RuntimeError(
                "Cannot build Manifest vocab without train APKs. "
                "Use a train-only vocab config first, then generate val/test with --no-rebuild-vocab."
            )
        records = _extract_manifest_records(train_jobs, workers=cfg["workers"])
    vocab = _build_or_load_vocab(jobs, records, cfg)
    if cfg["vocab_only"]:
        print(f"Manifest vocab ready: {cfg['vocab_path']}")
        return
    resume_rows: list[dict[str, str]] = []
    pending_jobs: list[dict[str, Any]] = []
    for job in jobs:
        row = _resume_existing(job, cfg)
        if row is None:
            pending_jobs.append(job)
        else:
            resume_rows.append(row)
    print(
    "AEG PT resume precheck: "
    f"total_jobs={len(jobs)} "
    f"resumable={len(resume_rows)} "
    f"pending={len(pending_jobs)}"
)

    rows: list[dict[str, str]] = list(resume_rows)
    ok = 0
    fail = 0
    if cfg["workers"] == 1:
        iterator = (_process_one(job, cfg, vocab, records.get(job["sha256"])) for job in pending_jobs)
        for success, row in tqdm(iterator, total=len(pending_jobs), desc="build AEG PT", unit="apk"):
            rows.append(row)
            ok += int(success)
            fail += int(not success)
            if not success and cfg["fail_on_error"]:
                raise RuntimeError(row["reason"])
    else:
        with ProcessPoolExecutor(
            max_workers=cfg["workers"],
            initializer=_init_build_worker,
            initargs=(cfg, vocab),
        ) as ex:
            futures = {
                ex.submit(_process_one_worker, job, records.get(job["sha256"])): job
                for job in pending_jobs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="build AEG PT", unit="apk"):
                job = futures[fut]
                try:
                    success, row = fut.result()
                except Exception as exc:
                    success = False
                    row = _index_row(job, _out_path(job, cfg), "failed", f"{type(exc).__name__}: {exc}")
                rows.append(row)
                ok += int(success)
                fail += int(not success)
                if not success and cfg["fail_on_error"]:
                    raise RuntimeError(row["reason"])
    index_path = cfg["out_root"] / "aeg_pt_index.csv"
    rows.sort(key=lambda r: (r["split"], r["apk_name"]))
    index_update_splits = (
        sorted({job["split"] for job in jobs})
        if cfg["allow_empty_splits"]
        else cfg["splits"]
    )
    merged_rows = _merge_index_rows(index_path, rows, index_update_splits)
    _write_index(merged_rows, index_path)
    ignored_path = cfg["out_root"] / "aeg_ignored_apks.csv"
    ignored_rows = [_index_row(job, _out_path(job, cfg), "ignored", "not_in_label_csv") for job in ignored_jobs]
    _write_index(_merge_index_rows(ignored_path, ignored_rows, index_update_splits), ignored_path)
    print(
        "AEG PT build complete: "
        f"resume={len(resume_rows)} built_ok={ok} failed={fail} "
        f"index={cfg['out_root'] / 'aeg_pt_index.csv'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-aware APK evidence graph PT files directly from APKs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--rebuild-vocab", dest="rebuild_vocab", action="store_true", default=None)
    parser.add_argument("--no-rebuild-vocab", dest="rebuild_vocab", action="store_false")
    parser.add_argument("--vocab-only", action="store_true", help="Build/load the train-only Manifest vocab and exit without writing PT files.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
