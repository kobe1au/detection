#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.constants import (  # noqa: E402
    AEG_SCHEMA_TABLE_FINGERPRINT,
    AEG_SCHEMA_TABLES,
    AEG_SCHEMA_VERSION,
    stable_table_hash,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = _load_yaml(path)
    bases = cfg.pop("base", None) or cfg.pop("bases", None) or []
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _deep_update(merged, _load_config(base_path))
    return _deep_update(merged, cfg)


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fingerprint(cfg: dict[str, Any], vocab: dict[str, Any]) -> str:
    source_paths = [
        "fusion/constants.py",
        "fusion/aeg_builder.py",
        "fusion/manifest_features.py",
        "fusion/semantic_categories.py",
        "fusion/quality.py",
        "extract/extract_graph_api.py",
        "scripts/build_aeg_pts_direct.py",
    ]
    source_hashes = {path: _sha256_text_file(PROJECT_ROOT / path) for path in source_paths}
    config_keys = [
        "manifest_dim",
        "node_feature_dim",
        "vocab_size",
        "sensitive_hops",
        "max_methods_per_dex",
        "fallback_max_methods",
        "fallback_policy",
        "use_graph_behavior_hints",
        "graph_behavior_hint_start",
        "graph_behavior_hint_dim",
        "num_api_buckets",
        "max_api_events_per_dex",
        "max_api_events_per_method",
        "api_event_scope",
        "framework_only",
        "include_descriptor",
        "keep_method_names",
        "keep_api_tokens",
    ]
    payload = {
        "schema_version": AEG_SCHEMA_VERSION,
        "schema_table_fingerprint": AEG_SCHEMA_TABLE_FINGERPRINT,
        "schema_tables": AEG_SCHEMA_TABLES,
        "config": {key: cfg.get(key) for key in config_keys},
        "manifest_vocab": {
            "categories": vocab.get("categories") or [],
            "permission_vocab": vocab.get("permission_vocab") or [],
            "intent_vocab": vocab.get("intent_vocab") or [],
            "feature_vocab": vocab.get("feature_vocab") or [],
            "metadata": vocab.get("metadata") or {},
        },
        "source_hashes": source_hashes,
    }
    return stable_table_hash(payload)


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


def _collect_apks(split_dirs: dict[str, Path], splits: list[str], *, hash_files: bool = True) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for split in splits:
        split_dir = split_dirs[split]
        if not split_dir.exists():
            continue
        paths = sorted(p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() == ".apk")
        for apk_path in tqdm(paths, desc=f"scan/hash {split}", unit="apk"):
            jobs.append(
                {
                    "split": split,
                    "apk_path": str(apk_path),
                    "apk_name": apk_path.name,
                    "sha256": _sha256_file(apk_path) if hash_files else apk_path.stem.lower(),
                }
            )
    return jobs


def _validate_split_counts(jobs: list[dict[str, str]], splits: list[str]) -> None:
    counts = {split: 0 for split in splits}
    for job in jobs:
        counts[job["split"]] += 1
    empty = [split for split, count in counts.items() if count <= 0]
    if empty:
        raise RuntimeError(f"No APK files found for configured splits: {empty}")


def _validate_unique_hashes(jobs: list[dict[str, str]]) -> None:
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
    splits = [str(v) for v in data.get("splits", ["train", "val", "test"])]
    split_dirs = _resolve_split_dirs(data, splits)
    out_dirs = _resolve_out_dirs(data, out_root, splits)
    vocab_path = _resolve_path(manifest.get("vocab_path", "config/manifest_vocab_aeg.yaml"))
    rebuild_vocab = bool(args.rebuild_vocab if args.rebuild_vocab is not None else manifest.get("rebuild_vocab", False))
    resume = bool(args.resume if args.resume is not None else execution.get("resume", True))
    vocab_will_be_built = rebuild_vocab or not vocab_path.exists()
    if vocab_will_be_built and resume:
        raise ValueError(
            "Manifest vocab will be built/rebuilt, but resume=true may skip existing .pt files. "
            "Use --no-resume when building/rebuilding vocab."
        )
    node_feature_dim = int(aeg.get("node_feature_dim", 128))
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
                "Use config/extract_aeg_behavior_hints.yaml or disable behavior hints."
            )
    return {
        "splits": splits,
        "split_dirs": split_dirs,
        "out_root": out_root,
        "out_dirs": out_dirs,
        "hash_files": bool(data.get("hash_files", True)),
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
        "max_methods_per_dex": int(graph.get("max_methods_per_dex", 4096)),
        "fallback_max_methods": int(graph.get("fallback_max_methods", 512)),
        "fallback_policy": str(graph.get("fallback_policy", "api_rich")),
        "use_graph_behavior_hints": use_graph_behavior_hints,
        "graph_behavior_hint_start": graph_behavior_hint_start,
        "graph_behavior_hint_dim": graph_behavior_hint_dim,
        "num_api_buckets": int(api.get("num_hash_buckets", 8192)),
        "max_api_events_per_dex": int(api.get("max_events_per_dex", 1024)),
        "max_api_events_per_method": int(api.get("max_events_per_method", 32)),
        "api_event_scope": str(api.get("event_scope", "all_methods")),
        "framework_only": bool(api.get("framework_only", True)),
        "include_descriptor": bool(api.get("include_descriptor", False)),
        "keep_method_names": bool(aeg.get("keep_method_names", True)),
        "keep_api_tokens": bool(aeg.get("keep_api_tokens", True)),
        "workers": int(args.workers or execution.get("workers", 1)),
        "resume": resume,
        "fail_on_error": bool(execution.get("fail_on_error", False)),
    }


def _extract_manifest_records(jobs: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    from fusion.manifest_features import extract_manifest_record

    records: dict[str, dict[str, Any]] = {}
    for job in tqdm(jobs, desc="extract manifests", unit="apk"):
        rec = extract_manifest_record(job["apk_path"], sid=job["sha256"]).to_json()
        rec["sid"] = job["sha256"]
        rec["sha256"] = job["sha256"]
        records[job["sha256"]] = rec
    return records


def _build_or_load_vocab(jobs: list[dict[str, str]], records: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
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
        }
        validate_manifest_vocab(vocab, require_train_metadata=True, allow_empty=cfg["allow_empty_vocab"])
        save_manifest_vocab(vocab, cfg["vocab_path"])
        return vocab
    return load_manifest_vocab(cfg["vocab_path"], require_train_metadata=True, allow_empty=cfg["allow_empty_vocab"])


def _index_row(job: dict[str, str], out_path: Path, status: str, reason: str = "") -> dict[str, str]:
    return {
        "split": job["split"],
        "sha256": job["sha256"],
        "apk_name": job["apk_name"],
        "apk_path": job["apk_path"],
        "pt_path": str(out_path),
        "status": status,
        "reason": reason,
    }


def _out_path(job: dict[str, str], cfg: dict[str, Any]) -> Path:
    return cfg["out_dirs"][job["split"]] / f"{job['sha256']}.pt"


def _resume_existing(job: dict[str, str], cfg: dict[str, Any]) -> dict[str, str] | None:
    if not cfg["resume"]:
        return None
    out_path = _out_path(job, cfg)
    if not out_path.exists():
        return None
    try:
        existing = torch.load(out_path, map_location="cpu")
        if int(existing.get("schema_version", 0)) != AEG_SCHEMA_VERSION:
            return None
        if existing.get("aeg_schema_fingerprint") != AEG_SCHEMA_TABLE_FINGERPRINT:
            return None
        expected_fingerprint = str(cfg.get("build_fingerprint") or "")
        if not expected_fingerprint or existing.get("aeg_build_fingerprint") != expected_fingerprint:
            return None
        meta = existing.get("aeg_meta") or {}
        if meta.get("schema_fingerprint") != AEG_SCHEMA_TABLE_FINGERPRINT:
            return None
        for key, expected in AEG_SCHEMA_TABLES.items():
            if dict(meta.get(key) or {}) != expected:
                return None
        if int(existing.get("schema_version", 0)) == AEG_SCHEMA_VERSION:
            return _index_row(job, out_path, "ok", "resume")
    except Exception:
        return None
    return None


def _process_one(job: dict[str, str], cfg: dict[str, Any], vocab: dict[str, Any], record: dict[str, Any]) -> tuple[bool, dict[str, str]]:
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
        dex_entries = list_dex_entries(apk_path)
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
                            max_methods_per_dex=cfg["max_methods_per_dex"],
                            fallback_max_methods=cfg["fallback_max_methods"],
                            num_api_buckets=cfg["num_api_buckets"],
                            max_api_events_per_dex=cfg["max_api_events_per_dex"],
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
            "aeg_build_fingerprint": cfg.get("build_fingerprint", ""),
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
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(payload, out_path)
        return True, _index_row(job, out_path, "ok", "")
    except Exception as exc:
        return False, _index_row(job, out_path, "failed", f"{type(exc).__name__}: {exc}")


def _write_index(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["split", "sha256", "apk_name", "apk_path", "pt_path", "status", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    cfg = _parse_config(_load_config(Path(args.config)), args)
    jobs = _collect_apks(cfg["split_dirs"], cfg["splits"], hash_files=cfg["hash_files"])
    _validate_split_counts(jobs, cfg["splits"])
    _validate_unique_hashes(jobs)
    vocab_needs_build = cfg["rebuild_vocab"] or not cfg["vocab_path"].exists()
    records: dict[str, dict[str, Any]] = {}
    if vocab_needs_build:
        records = _extract_manifest_records(jobs)
    vocab = _build_or_load_vocab(jobs, records, cfg)
    cfg["build_fingerprint"] = _build_fingerprint(cfg, vocab)

    resume_rows: list[dict[str, str]] = []
    pending_jobs: list[dict[str, str]] = []
    for job in jobs:
        row = _resume_existing(job, cfg)
        if row is None:
            pending_jobs.append(job)
        else:
            resume_rows.append(row)

    if pending_jobs:
        pending_records = _extract_manifest_records(pending_jobs)
        records.update(pending_records)
    rows: list[dict[str, str]] = list(resume_rows)
    ok = 0
    fail = 0
    if cfg["workers"] == 1:
        iterator = (_process_one(job, cfg, vocab, records[job["sha256"]]) for job in pending_jobs)
        for success, row in tqdm(iterator, total=len(pending_jobs), desc="build AEG PT", unit="apk"):
            rows.append(row)
            ok += int(success)
            fail += int(not success)
            if not success and cfg["fail_on_error"]:
                break
    else:
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
            futures = [ex.submit(_process_one, job, cfg, vocab, records[job["sha256"]]) for job in pending_jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="build AEG PT", unit="apk"):
                success, row = fut.result()
                rows.append(row)
                ok += int(success)
                fail += int(not success)
                if not success and cfg["fail_on_error"]:
                    raise RuntimeError(row["reason"])
    rows.sort(key=lambda r: (r["split"], r["apk_name"]))
    _write_index(rows, cfg["out_root"] / "aeg_pt_index.csv")
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
