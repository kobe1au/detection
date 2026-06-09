from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from scripts.build_aeg_pts_direct import _load_config, _parse_config  # noqa: E402
from fusion.io_utils import load_aeg_payload  # noqa: E402


def _norm_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _find_id_column(fieldnames: list[str]) -> str:
    candidates = [
        "sha256",
        "sha256_hash",
        "sid",
        "sample_id",
        "id",
        "hash",
    ]
    lower = {name.lower(): name for name in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise RuntimeError(f"Cannot find SHA256/id column. CSV columns={fieldnames}")


def read_csv_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        id_col = _find_id_column(reader.fieldnames)
        ids = {_norm_id(row.get(id_col)) for row in reader}
    ids.discard("")
    return ids


def get_payload_sid(payload: dict[str, Any]) -> str:
    for key in ["sid", "sha256", "sha256_hash", "sample_id", "id"]:
        if key in payload:
            return _norm_id(payload[key])
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for key in ["sid", "sha256", "sha256_hash", "sample_id", "id"]:
            if key in meta:
                return _norm_id(meta[key])
    return ""


def get_node_dim(payload: dict[str, Any]) -> int | None:
    for key in ["node_x", "x"]:
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.size(1))
    return None


def choose_files(files: list[Path], sample: int, seed: int) -> list[Path]:
    if sample <= 0 or sample >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, sample))


def validate_one_pt(path: Path, *, expected_dim: int) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    info: dict[str, Any] = {
        "sid": path.stem.lower(),
        "node_dim": None,
        "schema_version": "",
    }

    try:
        payload = load_aeg_payload(path, validate=True, expected_node_feature_dim=expected_dim)
    except Exception as exc:
        return False, [f"payload load failed: {type(exc).__name__}: {exc}"], info

    if not isinstance(payload, dict):
        return False, [f"payload is not dict: {type(payload)}"], info

    file_sid = path.stem.lower()
    payload_sid = get_payload_sid(payload)
    if payload_sid and payload_sid != file_sid:
        errors.append(f"sid mismatch: file={file_sid}, payload={payload_sid}")

    node_dim = get_node_dim(payload)
    info["node_dim"] = node_dim
    if node_dim != expected_dim:
        errors.append(f"node_feature_dim mismatch: expected={expected_dim}, got={node_dim}")

    schema_version = payload.get("schema_version") or payload.get("aeg_schema_version") or ""
    info["schema_version"] = str(schema_version)

    return len(errors) == 0, errors, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/extract_aeg.yaml"))
    parser.add_argument("--sample-per-split", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()

    # 关键：这里模拟 build_aeg_pts_direct.py 需要的 argparse 参数。
    # 不生成、不重建 vocab、不改任何 PT。
    build_args = SimpleNamespace(
        config=config_path,
        vocab_only=False,
        rebuild_vocab=False,
        resume=True,
        workers=None,
    )

    raw_cfg = _load_config(config_path)
    cfg = _parse_config(raw_cfg, build_args)

    splits: list[str] = list(cfg["splits"])
    out_dirs: dict[str, Path] = dict(cfg["out_dirs"])
    label_csvs: dict[str, Path] = dict(cfg["label_csvs"])
    expected_dim = int(cfg["node_feature_dim"])

    sample = 0 if args.all else int(args.sample_per_split)

    print("=" * 80)
    print("Validate generated AEG PT files")
    print("=" * 80)
    print(f"config:             {config_path}")
    print(f"splits:             {splits}")
    print(f"expected_node_dim:  {expected_dim}")
    print(f"sample_per_split:   {'ALL' if sample <= 0 else sample}")
    print()

    errors: list[str] = []
    node_dim_counter: Counter[int] = Counter()
    schema_counter: Counter[str] = Counter()

    csv_ids_by_split: dict[str, set[str]] = {}
    pt_ids_by_split: dict[str, set[str]] = {}

    for split in splits:
        csv_path = label_csvs.get(split)
        pt_dir = out_dirs.get(split)

        if csv_path is None:
            errors.append(f"[{split}] missing label_csvs entry")
            continue
        if pt_dir is None:
            errors.append(f"[{split}] missing out_dirs entry")
            continue

        if not csv_path.exists():
            errors.append(f"[{split}] label CSV not found: {csv_path}")
            continue
        if not pt_dir.exists():
            errors.append(f"[{split}] PT dir not found: {pt_dir}")
            continue

        csv_ids = read_csv_ids(csv_path)
        pt_files = sorted(pt_dir.glob("*.pt"))
        pt_ids = {p.stem.lower() for p in pt_files}

        csv_ids_by_split[split] = csv_ids
        pt_ids_by_split[split] = pt_ids

        csv_only = sorted(csv_ids - pt_ids)
        pt_only = sorted(pt_ids - csv_ids)

        print(f"[{split}]")
        print(f"  label_csv: {csv_path}")
        print(f"  pt_dir:    {pt_dir}")
        print(f"  csv_ids:   {len(csv_ids)}")
        print(f"  pt_files:  {len(pt_files)}")
        print(f"  csv_only:  {len(csv_only)}")
        print(f"  pt_only:   {len(pt_only)}")

        if csv_only:
            errors.append(f"[{split}] CSV ids missing PT files: count={len(csv_only)} examples={csv_only[:5]}")
        if pt_only:
            errors.append(f"[{split}] PT files not in CSV: count={len(pt_only)} examples={pt_only[:5]}")

        chosen = choose_files(pt_files, sample, args.seed)
        print(f"  loading/checking PT files: {len(chosen)}")

        split_load_errors = 0
        for path in chosen:
            ok, item_errors, info = validate_one_pt(path, expected_dim=expected_dim)

            if info.get("node_dim") is not None:
                node_dim_counter[int(info["node_dim"])] += 1
            if info.get("schema_version"):
                schema_counter[str(info["schema_version"])] += 1

            if not ok:
                split_load_errors += len(item_errors)
                for e in item_errors:
                    msg = f"[{split}] {path.name}: {e}"
                    errors.append(msg)
                    if args.fail_fast:
                        print(f"[ERROR] {msg}")
                        return 2

        print(f"  loaded_file_errors: {split_load_errors}")
        print()

    # 跨 split 重复检查
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            if a not in pt_ids_by_split or b not in pt_ids_by_split:
                continue
            overlap = sorted(pt_ids_by_split[a] & pt_ids_by_split[b])
            if overlap:
                errors.append(f"[split-overlap] {a} vs {b}: count={len(overlap)} examples={overlap[:5]}")

    print("=" * 80)
    print("Summary")
    print("=" * 80)

    print("Node feature dims observed:")
    if node_dim_counter:
        for dim, count in node_dim_counter.most_common():
            print(f"  dim={dim}: {count}")
    else:
        print("  <none>")

    print("Schema versions observed:")
    if schema_counter:
        for schema, count in schema_counter.most_common():
            print(f"  {schema}: {count}")
    else:
        print("  <none>")

    print(f"errors={len(errors)}")
    for e in errors[:50]:
        print(f"[ERROR] {e}")
    if len(errors) > 50:
        print(f"... {len(errors) - 50} more errors omitted")

    print()
    if errors:
        print("RESULT: FAIL")
        return 2

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
