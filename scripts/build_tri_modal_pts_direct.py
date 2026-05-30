#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_runtime_dependencies() -> None:
    try:
        import extract.extract_graph_api  # noqa: F401
        import fusion.robust.manifest_features  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing runtime dependency for tri-modal PT build: {exc.name}. "
            "Run this script with the same Python environment used for training, including PyTorch."
        ) from exc


def _resolve_split_dirs(data: dict[str, Any], splits: list[str]) -> tuple[dict[str, Path], Path | None]:
    split_dirs_raw = data.get("split_dirs") or data.get("apk_dirs") or {}
    if split_dirs_raw:
        split_dirs_raw = {str(k): v for k, v in split_dirs_raw.items()}
        missing = [split for split in splits if not split_dirs_raw.get(split)]
        if missing:
            raise ValueError(f"data.split_dirs is missing configured splits: {missing}")
        return {
            split: _resolve_path(split_dirs_raw[split])
            for split in splits
        }, None

    apk_root_raw = data.get("apk_root", "")
    if not apk_root_raw:
        raise ValueError("Either data.split_dirs or data.apk_root is required")
    apk_root = _resolve_path(apk_root_raw)
    return {split: apk_root / split for split in splits}, apk_root


def _resolve_out_dirs(data: dict[str, Any], out_root: Path, splits: list[str]) -> dict[str, Path]:
    out_dirs_raw = data.get("out_dirs") or {}
    if out_dirs_raw:
        out_dirs_raw = {str(k): v for k, v in out_dirs_raw.items()}
        missing = [split for split in splits if not out_dirs_raw.get(split)]
        if missing:
            raise ValueError(f"data.out_dirs is missing configured splits: {missing}")
        return {
            split: _resolve_path(out_dirs_raw[split])
            for split in splits
        }
    return {split: out_root / split for split in splits}


def _collect_apks(split_dirs: dict[str, Path], splits: list[str], hash_files: bool = True) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for split in splits:
        split_dir = split_dirs[split]
        if not split_dir.exists():
            continue

        apk_paths = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".apk"
        )
        for apk_path in tqdm(apk_paths, desc=f"scan/hash {split}", unit="apk"):
            jobs.append(
                {
                    "split": split,
                    "apk_path": str(apk_path),
                    "sha256": _sha256_file(apk_path) if hash_files else "",
                }
            )
    return jobs


def _split_counts(jobs: list[dict[str, str]], splits: list[str]) -> dict[str, int]:
    counts = {split: 0 for split in splits}
    for job in jobs:
        counts[job["split"]] = counts.get(job["split"], 0) + 1
    return counts


def _validate_split_counts(split_counts: dict[str, int]) -> None:
    empty = [split for split, count in split_counts.items() if count <= 0]
    if empty:
        raise RuntimeError(f"No APK files found for configured splits: {empty}")


def _index_row(
    job: dict[str, str],
    out_path: Path,
    status: str,
    reason: str = "",
) -> dict[str, str]:
    apk_path = Path(job["apk_path"])
    return {
        "split": job["split"],
        "sha256": job["sha256"],
        "apk_name": apk_path.name,
        "apk_path": str(apk_path),
        "pt_path": str(out_path),
        "status": status,
        "reason": reason,
    }


def _write_index_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "sha256", "apk_name", "apk_path", "pt_path", "status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest_jsonl(records: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in sorted(records):
            f.write(json.dumps(records[sid], ensure_ascii=True, sort_keys=True) + "\n")


def _parse_config(raw: dict[str, Any]) -> dict[str, Any]:
    data = raw.get("data", {})
    hp = raw.get("hyperparameters", {})
    graph = raw.get("graph", {})
    api = raw.get("api", {})
    manifest = raw.get("manifest", {})
    storage = raw.get("storage", {})
    execution = raw.get("execution", {})

    out_root_raw = data.get("out_root", "")
    if not out_root_raw:
        raise ValueError("data.out_root is required")
    out_root = _resolve_path(out_root_raw)
    splits = [str(split) for split in data.get("splits", ["train", "val", "test"])]
    if not splits:
        raise ValueError("data.splits must not be empty")
    split_dirs, apk_root = _resolve_split_dirs(data, splits)
    out_dirs = _resolve_out_dirs(data, out_root, splits)
    vocab_path = _resolve_path(manifest.get("vocab_path", "config/manifest_vocab.yaml"))
    manifest_jsonl_dir_raw = manifest.get("manifest_jsonl_dir", "")
    manifest_jsonl_dir = (
        _resolve_path(manifest_jsonl_dir_raw)
        if manifest_jsonl_dir_raw
        else out_root / "_manifest_jsonl"
    )

    cfg = {
        "apk_root": apk_root,
        "split_dirs": split_dirs,
        "out_root": out_root,
        "out_dirs": out_dirs,
        "splits": splits,
        "vocab_size": int(hp.get("vocab_size", 256)),
        "sensitive_hops": int(graph.get("sensitive_hops", 1)),
        "max_methods_per_dex": int(graph.get("max_methods_per_dex", 4096)),
        "fallback_max_methods": int(graph.get("fallback_max_methods", 512)),
        "fallback_policy": str(graph.get("fallback_policy", "api_rich")),
        "use_graph_behavior_hints": bool(graph.get("use_behavior_hints", True)),
        "num_api_buckets": int(api.get("num_hash_buckets", 8192)),
        "max_api_events_per_dex": int(api.get("max_events_per_dex", 1024)),
        "max_api_events_per_method": int(api.get("max_events_per_method", 32)),
        "api_event_scope": str(api.get("event_scope", "all_methods")),
        "framework_only": bool(api.get("framework_only", True)),
        "include_descriptor": bool(api.get("include_descriptor", False)),
        "manifest_dim": int(manifest.get("manifest_dim", 256)),
        "vocab_path": vocab_path,
        "rebuild_vocab": bool(manifest.get("rebuild_vocab", False)),
        "max_permissions": int(manifest.get("max_permissions", 128)),
        "max_intents": int(manifest.get("max_intents", 64)),
        "max_features": int(manifest.get("max_features", 32)),
        "allow_empty_vocab": bool(manifest.get("allow_empty_vocab", False)),
        "save_manifest_jsonl": bool(manifest.get("save_manifest_jsonl", True)),
        "manifest_jsonl_dir": manifest_jsonl_dir,
        "keep_method_names": bool(storage.get("keep_method_names", False)),
        "keep_api_tokens": bool(storage.get("keep_api_tokens", False)),
        "workers": int(execution.get("workers", 1)),
        "resume": bool(execution.get("resume", True)),
        "fail_on_error": bool(execution.get("fail_on_error", False)),
        "failed_json": str(execution.get("failed_json", "")),
    }

    vocab_will_be_built = cfg["rebuild_vocab"] or not cfg["vocab_path"].exists()
    if "train" not in cfg["splits"] and vocab_will_be_built:
        raise ValueError("train split is required when building Manifest vocab")
    if cfg["fallback_policy"] not in {"api_rich", "all_capped", "empty_dex"}:
        raise ValueError("graph.fallback_policy must be api_rich, all_capped, or empty_dex")
    if cfg["api_event_scope"] not in {"all_methods", "graph_methods"}:
        raise ValueError("api.event_scope must be all_methods or graph_methods")
    if cfg["workers"] <= 0:
        raise ValueError("execution.workers must be >= 1")
    if vocab_will_be_built and cfg["resume"]:
        raise ValueError(
            "Manifest vocab will be built/rebuilt, but resume=true may skip existing .pt files. "
            "Use --no-resume when building/rebuilding vocab."
        )
    return cfg


def _extract_manifest_records(jobs: list[dict[str, str]], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from fusion.robust.manifest_features import extract_manifest_record

    records: dict[str, dict[str, Any]] = {}
    for job in tqdm(jobs, desc="extract manifests", unit="apk"):
        sid = job["sha256"]
        rec = extract_manifest_record(job["apk_path"], sid=sid).to_json()
        rec["sid"] = sid
        rec["sha256"] = sid
        records[sid] = rec

    if cfg["save_manifest_jsonl"]:
        by_split: dict[str, dict[str, dict[str, Any]]] = {}
        for job in jobs:
            by_split.setdefault(job["split"], {})[job["sha256"]] = records[job["sha256"]]
        for split, split_records in by_split.items():
            _write_manifest_jsonl(split_records, cfg["manifest_jsonl_dir"] / f"{split}.jsonl")
    return records


def _build_or_load_vocab(
    jobs: list[dict[str, str]],
    manifest_records: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    from fusion.robust.manifest_features import (
        build_manifest_vocab,
        load_manifest_vocab,
        save_manifest_vocab,
        validate_manifest_vocab,
    )

    vocab_path: Path = cfg["vocab_path"]
    if cfg["rebuild_vocab"] or not vocab_path.exists():
        train_records = [
            manifest_records[job["sha256"]]
            for job in jobs
            if job["split"] == "train" and job["sha256"] in manifest_records
        ]
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
            "source": "scripts/build_tri_modal_pts_direct.py",
            "leakage_guard": "train_only",
        }
        validate_manifest_vocab(
            vocab,
            require_train_metadata=True,
            allow_empty=cfg["allow_empty_vocab"],
        )
        save_manifest_vocab(vocab, vocab_path)
        return vocab
    return load_manifest_vocab(
        vocab_path,
        require_train_metadata=True,
        allow_empty=cfg["allow_empty_vocab"],
    )


def _manifest_payload(record: dict[str, Any], vocab: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    from fusion.robust.manifest_features import vectorize_manifest_record

    payload = vectorize_manifest_record(record, vocab, manifest_dim=cfg["manifest_dim"])
    meta = dict(payload.get("manifest_meta") or {})
    meta.setdefault("sha256", record.get("sha256", ""))
    meta.setdefault("apk_name", record.get("apk_name", ""))
    payload["manifest_meta"] = meta
    return payload


def _process_one(job: dict[str, str], cfg: dict[str, Any], vocab: dict[str, Any], record: dict[str, Any]):
    from extract.extract_graph_api import (
        DEX,
        atomic_torch_save,
        build_graph_api_for_dex,
        list_dex_entries,
    )

    apk_path = Path(job["apk_path"])
    split = job["split"]
    sha = job["sha256"]
    out_dir = Path(cfg["out_dirs"][split])
    out_path = out_dir / f"{sha}.pt"
    if cfg["resume"] and out_path.exists():
        return True, _index_row(job, out_path, "ok", "resumed_existing_pt")

    try:
        dex_entries = list_dex_entries(apk_path)
        if not dex_entries:
            return False, _index_row(job, out_path, "failed", "no classes*.dex")

        dex_list: list[dict[str, Any]] = []
        dex_failures: list[dict[str, str]] = []
        with zipfile.ZipFile(apk_path, "r") as zf:
            for entry in dex_entries:
                dex_name = Path(entry).name
                try:
                    dex_bytes = zf.read(entry)
                    item = build_graph_api_for_dex(
                        dvm=DEX(dex_bytes),
                        raw_bytes=dex_bytes,
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
                    item["dex_name"] = dex_name
                    item["meta"].update(
                        {
                            "sha256_apk": sha,
                            "apk_name": apk_path.name,
                            "dex_name": dex_name,
                            "split": split,
                            "representation": {
                                "graph_branch": "sensitive_method_call_graph",
                                "api_branch": "framework_api_behavior_sequence",
                                "manifest_branch": "soft_declaration_prior",
                                "alignment_anchor": "method_api_edge_index",
                            },
                        }
                    )
                    dex_list.append(item)
                except Exception as exc:
                    dex_failures.append({"dex": dex_name, "reason": f"{type(exc).__name__}: {exc}"})

        if not dex_list:
            return False, _index_row(job, out_path, "failed", f"all dex entries failed: {dex_failures[:3]}")

        payload = _manifest_payload(record, vocab, cfg)
        output = {
            "sid": sha,
            "sha256": sha,
            "apk_name": apk_path.name,
            "split": split,
            "dex_list": dex_list,
            "direct_build_meta": {
                "builder": "scripts/build_tri_modal_pts_direct.py",
                "num_dex": len(dex_list),
                "dex_failures": dex_failures,
            },
            **payload,
        }
        atomic_torch_save(output, out_path)
        return True, _index_row(job, out_path, "ok")
    except Exception as exc:
        return False, _index_row(job, out_path, "failed", f"{type(exc).__name__}: {exc}")


def _worker(args):
    job, cfg, vocab, record = args
    return _process_one(job, cfg, vocab, record)


def run(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    if not dry_run:
        _validate_runtime_dependencies()
        from fusion.robust.semantic_categories import validate_api_type_mapping

        validate_api_type_mapping()

    jobs = _collect_apks(cfg["split_dirs"], cfg["splits"], hash_files=not dry_run)
    split_counts = _split_counts(jobs, cfg["splits"])

    print(
        json.dumps(
            {
                "apk_root": str(cfg["apk_root"]) if cfg["apk_root"] is not None else "",
                "split_dirs": {split: str(path) for split, path in cfg["split_dirs"].items()},
                "out_root": str(cfg["out_root"]),
                "out_dirs": {split: str(path) for split, path in cfg["out_dirs"].items()},
                "splits": cfg["splits"],
                "split_counts": split_counts,
                "vocab_path": str(cfg["vocab_path"]),
                "workers": cfg["workers"],
                "resume": cfg["resume"],
                "fail_on_error": cfg["fail_on_error"],
                "direct": True,
            },
            indent=2,
        ),
        flush=True,
    )
    print("------------------------------------------")
    _validate_split_counts(split_counts)
    if dry_run:
        return {"ok": 0, "fail": 0, "dry_run": True, "split_counts": split_counts}

    manifest_records = _extract_manifest_records(jobs, cfg)
    vocab = _build_or_load_vocab(jobs, manifest_records, cfg)

    tasks = [
        (job, cfg, vocab, manifest_records.get(job["sha256"], {}))
        for job in jobs
    ]
    ok = 0
    fail = 0
    index_rows: list[dict[str, str]] = []

    if cfg["workers"] == 1:
        iterator = tqdm(tasks, desc="build tri-modal .pt", unit="apk")
        for task in iterator:
            succ, row = _worker(task)
            index_rows.append(row)
            if succ:
                ok += 1
            else:
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
            future_map = {ex.submit(_worker, task): task[0] for task in tasks}
            for fut in tqdm(as_completed(future_map), total=len(future_map), desc="build tri-modal .pt", unit="apk"):
                job = future_map[fut]
                try:
                    succ, row = fut.result()
                except Exception as exc:
                    out_path = Path(cfg["out_dirs"][job["split"]]) / f"{job['sha256']}.pt"
                    succ = False
                    row = _index_row(job, out_path, "failed", f"{type(exc).__name__}: {exc}")
                index_rows.append(row)
                if succ:
                    ok += 1
                else:
                    fail += 1

    index_rows.sort(key=lambda row: (row["split"], row["sha256"], row["apk_name"]))
    index_path = Path(cfg["out_root"]) / "tri_modal_pt_index.csv"
    _write_index_csv(index_rows, index_path)
    print(f"index -> {index_path}", flush=True)

    failed = [row for row in index_rows if row["status"] != "ok"]
    if failed:
        failed_path = Path(cfg["failed_json"]) if cfg["failed_json"] else Path(cfg["out_root"]) / "failed_tri_modal_direct.json"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"failed list -> {failed_path}", flush=True)

    summary = {"ok": ok, "fail": fail, "failed": failed, "split_counts": split_counts}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if fail > 0 and cfg["fail_on_error"]:
        raise RuntimeError(
            f"{fail} APK(s) failed to build tri-modal .pt files. "
            f"See {failed_path if failed else 'tri_modal_pt_index.csv'} for details."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly build API+Graph+Manifest tri-modal .pt files from APKs.")
    parser.add_argument("--config", default="config/extract_tri_model.yaml")
    parser.add_argument("--apk-root", default="", help="Override data.apk_root.")
    parser.add_argument("--train-dir", default="", help="Override data.split_dirs.train.")
    parser.add_argument("--val-dir", default="", help="Override data.split_dirs.val.")
    parser.add_argument("--test-dir", default="", help="Override data.split_dirs.test.")
    parser.add_argument("--out-root", default="", help="Override data.out_root.")
    parser.add_argument("--train-out-dir", default="", help="Override data.out_dirs.train.")
    parser.add_argument("--val-out-dir", default="", help="Override data.out_dirs.val.")
    parser.add_argument("--test-out-dir", default="", help="Override data.out_dirs.test.")
    parser.add_argument("--splits", nargs="+", default=None, help="Override data.splits.")
    parser.add_argument("--workers", type=int, default=None, help="Override execution.workers.")
    parser.add_argument("--resume", action="store_true", help="Force resume=true.")
    parser.add_argument("--no-resume", action="store_true", help="Force resume=false.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any APK fails.")
    parser.add_argument("--allow-failures", action="store_true", help="Set execution.fail_on_error=false.")
    parser.add_argument("--rebuild-vocab", action="store_true", help="Force rebuild Manifest vocab from train split.")
    parser.add_argument("--no-rebuild-vocab", action="store_true", help="Force rebuild_vocab=false.")
    parser.add_argument("--allow-empty-vocab", action="store_true", help="Allow an empty Manifest vocab for debug runs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = _resolve_path(args.config)
    raw = _load_yaml(cfg_path)
    if args.apk_root:
        data = raw.setdefault("data", {})
        data["apk_root"] = args.apk_root
        data.pop("split_dirs", None)
        data.pop("apk_dirs", None)
    split_dir_overrides = {
        "train": args.train_dir,
        "val": args.val_dir,
        "test": args.test_dir,
    }
    if any(split_dir_overrides.values()):
        data = raw.setdefault("data", {})
        split_dirs = data.setdefault("split_dirs", {})
        for split, path in split_dir_overrides.items():
            if path:
                split_dirs[split] = path
    if args.out_root:
        data = raw.setdefault("data", {})
        data["out_root"] = args.out_root
        data.pop("out_dirs", None)
    out_dir_overrides = {
        "train": args.train_out_dir,
        "val": args.val_out_dir,
        "test": args.test_out_dir,
    }
    if any(out_dir_overrides.values()):
        data = raw.setdefault("data", {})
        out_dirs = data.setdefault("out_dirs", {})
        for split, path in out_dir_overrides.items():
            if path:
                out_dirs[split] = path
    if args.splits is not None:
        raw.setdefault("data", {})["splits"] = args.splits
    if args.workers is not None:
        raw.setdefault("execution", {})["workers"] = args.workers
    if args.resume and args.no_resume:
        raise SystemExit("Use only one of --resume or --no-resume.")
    if args.rebuild_vocab and args.no_rebuild_vocab:
        raise SystemExit("Use only one of --rebuild-vocab or --no-rebuild-vocab.")
    if args.resume:
        raw.setdefault("execution", {})["resume"] = True
    if args.no_resume:
        raw.setdefault("execution", {})["resume"] = False
    if args.fail_on_error:
        raw.setdefault("execution", {})["fail_on_error"] = True
    if args.allow_failures:
        raw.setdefault("execution", {})["fail_on_error"] = False
    if args.rebuild_vocab:
        raw.setdefault("manifest", {})["rebuild_vocab"] = True
    if args.no_rebuild_vocab:
        raw.setdefault("manifest", {})["rebuild_vocab"] = False
    if args.allow_empty_vocab:
        raw.setdefault("manifest", {})["allow_empty_vocab"] = True

    try:
        cfg = _parse_config(raw)
        run(cfg, dry_run=args.dry_run)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
