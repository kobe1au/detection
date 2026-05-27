#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(f'"{p}"' if " " in p else p for p in cmd)
    print(f"[tri-modal] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _apk_rows(apk_root: Path, split: str) -> list[dict[str, str]]:
    split_dir = apk_root / split
    if not split_dir.exists():
        return []
    rows = []
    for apk_path in sorted(split_dir.glob("*.apk")):
        rows.append({"id": _sha256_file(apk_path), "apk_path": str(apk_path.resolve())})
    return rows


def _write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "apk_path"])
        writer.writeheader()
        writer.writerows(rows)


def _write_effective_graph_config(
    base_config: Path,
    out_path: Path,
    apk_root: Path,
    graph_out_root: Path,
    splits: list[str],
) -> Path:
    cfg = _load_yaml(base_config)
    cfg.setdefault("data", {})
    cfg["data"]["apk_root"] = str(apk_root)
    cfg["data"]["out_root"] = str(graph_out_root)
    cfg["data"]["splits"] = splits
    _save_yaml(cfg, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build API+Graph+Manifest tri-modal .pt files from split APK directories."
    )
    parser.add_argument("--graph-config", default="config/extract_graph_api.yaml", help="Base graph/API extraction YAML.")
    parser.add_argument("--apk-root", default="", help="APK root containing train/val/test split folders. Overrides graph config.")
    parser.add_argument("--graph-out-root", default="", help="Intermediate API+Graph .pt root. Overrides graph config.")
    parser.add_argument("--tri-out-root", default="", help="Final API+Graph+Manifest .pt root. Default: <graph-out-root>_tri.")
    parser.add_argument("--work-dir", default="", help="Temporary manifest CSV/JSONL/config directory.")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to build. Default comes from graph config.")
    parser.add_argument("--vocab", default="config/manifest_vocab.yaml", help="Train-built Manifest vocab YAML.")
    parser.add_argument("--rebuild-vocab", action="store_true", help="Rebuild Manifest vocab from train Manifest JSONL.")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph/API extraction and reuse --graph-out-root.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--manifest-dim", type=int, default=256)
    parser.add_argument("--max-permissions", type=int, default=128)
    parser.add_argument("--max-intents", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=32)
    args = parser.parse_args()

    graph_config = Path(args.graph_config)
    if not graph_config.is_absolute():
        graph_config = PROJECT_ROOT / graph_config
    cfg = _load_yaml(graph_config)
    data_cfg = cfg.get("data", {})

    apk_root_raw = args.apk_root or data_cfg.get("apk_root", "")
    graph_out_root_raw = args.graph_out_root or data_cfg.get("out_root", "")
    if not apk_root_raw:
        raise SystemExit("--apk-root is required or must be set in graph config data.apk_root")
    if not graph_out_root_raw:
        raise SystemExit("--graph-out-root is required or must be set in graph config data.out_root")
    apk_root = Path(apk_root_raw)
    graph_out_root = Path(graph_out_root_raw)
    splits = list(args.splits or data_cfg.get("splits", ["train", "val", "test"]))

    tri_out_root = Path(args.tri_out_root) if args.tri_out_root else graph_out_root.with_name(f"{graph_out_root.name}_tri")
    work_dir = Path(args.work_dir) if args.work_dir else tri_out_root / "_build_manifest"
    vocab_path = Path(args.vocab)
    if not vocab_path.is_absolute():
        vocab_path = PROJECT_ROOT / vocab_path

    graph_config_effective = work_dir / "extract_graph_api.effective.yaml"
    _write_effective_graph_config(graph_config, graph_config_effective, apk_root, graph_out_root, splits)

    print(json.dumps({
        "apk_root": str(apk_root),
        "graph_out_root": str(graph_out_root),
        "tri_out_root": str(tri_out_root),
        "splits": splits,
        "vocab": str(vocab_path),
    }, indent=2), flush=True)

    if not args.skip_graph:
        _run([sys.executable, "extract/extract_graph_api.py", "--config", str(graph_config_effective)], args.dry_run)

    manifest_jsonl_by_split: dict[str, Path] = {}
    for split in splits:
        rows = _apk_rows(apk_root, split)
        if not rows:
            print(f"[tri-modal] warning: no APKs found for split={split}", flush=True)
        csv_path = work_dir / "manifest_csv" / f"{split}.csv"
        jsonl_path = work_dir / "manifest_jsonl" / f"{split}.jsonl"
        _write_manifest_csv(csv_path, rows)
        manifest_jsonl_by_split[split] = jsonl_path
        _run(
            [
                sys.executable,
                "scripts/extract_manifest_features.py",
                "--csv",
                str(csv_path),
                "--out-jsonl",
                str(jsonl_path),
                "--id-col",
                "id",
                "--apk-col",
                "apk_path",
            ],
            args.dry_run,
        )

    train_jsonl = manifest_jsonl_by_split.get("train")
    should_build_vocab = args.rebuild_vocab or not vocab_path.exists()
    if should_build_vocab and train_jsonl is None:
        raise SystemExit("A train split is required to build the Manifest vocab.")
    if should_build_vocab:
        _run(
            [
                sys.executable,
                "scripts/build_manifest_vocab_from_train.py",
                "--train-manifest-jsonl",
                str(train_jsonl),
                "--vocab",
                str(vocab_path),
                "--max-permissions",
                str(args.max_permissions),
                "--max-intents",
                str(args.max_intents),
                "--max-features",
                str(args.max_features),
            ],
            args.dry_run,
        )

    for split in splits:
        cmd = [
            sys.executable,
            "scripts/augment_pts_with_manifest.py",
            "--pt-dir",
            str(graph_out_root / split),
            "--manifest-jsonl",
            str(manifest_jsonl_by_split[split]),
            "--out-dir",
            str(tri_out_root / split),
            "--vocab",
            str(vocab_path),
            "--split",
            split,
            "--manifest-dim",
            str(args.manifest_dim),
        ]
        _run(cmd, args.dry_run)

    print(f"[tri-modal] done -> {tri_out_root}", flush=True)


if __name__ == "__main__":
    main()
