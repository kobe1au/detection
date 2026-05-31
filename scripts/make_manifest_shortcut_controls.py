#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.robust.semantic_categories import SEMANTIC_CATEGORY_DIM
from fusion.robust.train import load_config, resolve


ID_CANDIDATES = ("id", "sha256", "ID", "Id")
MANIFEST_KEYS = (
    "manifest_x",
    "manifest_permission_ids",
    "manifest_intent_ids",
    "manifest_category_counts",
    "manifest_stats",
    "q_manifest",
    "pert_manifest",
    "manifest_meta",
    "manifest_permission_dim",
    "manifest_intent_dim",
    "manifest_feature_dim",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    id_col = next((col for col in ID_CANDIDATES if col in fieldnames), "")
    if not id_col:
        raise ValueError(f"{path} must contain one of id columns: {', '.join(ID_CANDIDATES)}")
    return fieldnames, rows, id_col


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _manifest_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw.get(key) for key in MANIFEST_KEYS if key in raw}


def _set_manifest_payload(raw: dict[str, Any], payload: dict[str, Any]) -> None:
    for key in MANIFEST_KEYS:
        raw.pop(key, None)
    for key, value in payload.items():
        raw[key] = value


def _tensor_like_or_default(raw: dict[str, Any], key: str, shape: tuple[int, ...], fill: float = 0.0) -> torch.Tensor:
    value = raw.get(key)
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return torch.full_like(value.detach().float(), float(fill))
    return torch.full(shape, float(fill), dtype=torch.float32)


def _zero_payload(raw: dict[str, Any], manifest_dim: int, stats_dim: int) -> dict[str, Any]:
    return {
        "manifest_x": _tensor_like_or_default(raw, "manifest_x", (manifest_dim,), 0.0),
        "manifest_permission_ids": torch.empty((0,), dtype=torch.long),
        "manifest_intent_ids": torch.empty((0,), dtype=torch.long),
        "manifest_category_counts": torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32),
        "manifest_stats": torch.zeros((stats_dim,), dtype=torch.float32),
        "q_manifest": torch.tensor([0.0], dtype=torch.float32),
        "pert_manifest": torch.tensor([1.0], dtype=torch.float32),
        "manifest_meta": {"shortcut_control": "manifest_zeroed"},
        "manifest_permission_dim": int(raw.get("manifest_permission_dim", 0) or 0),
        "manifest_intent_dim": int(raw.get("manifest_intent_dim", 0) or 0),
        "manifest_feature_dim": int(raw.get("manifest_feature_dim", 0) or 0),
    }


def _noisy_payload(raw: dict[str, Any], manifest_dim: int, stats_dim: int, seed: int) -> dict[str, Any]:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    manifest_x = torch.rand((manifest_dim,), generator=gen)
    counts = torch.poisson(torch.full((SEMANTIC_CATEGORY_DIM,), 1.0, dtype=torch.float32), generator=gen)
    stats = torch.rand((stats_dim,), generator=gen)
    perm_dim = int(raw.get("manifest_permission_dim", 128) or 128)
    intent_dim = int(raw.get("manifest_intent_dim", 64) or 64)
    n_perm = min(max(1, perm_dim // 16), perm_dim)
    n_intent = min(max(1, intent_dim // 16), intent_dim)
    perm_ids = torch.randperm(max(perm_dim, 1), generator=gen)[:n_perm] + 1
    intent_ids = torch.randperm(max(intent_dim, 1), generator=gen)[:n_intent] + 1
    return {
        "manifest_x": manifest_x,
        "manifest_permission_ids": perm_ids.long(),
        "manifest_intent_ids": intent_ids.long(),
        "manifest_category_counts": counts.float(),
        "manifest_stats": stats.float(),
        "q_manifest": torch.tensor([1.0], dtype=torch.float32),
        "pert_manifest": torch.tensor([0.5], dtype=torch.float32),
        "manifest_meta": {"shortcut_control": "manifest_noisy"},
        "manifest_permission_dim": perm_dim,
        "manifest_intent_dim": intent_dim,
        "manifest_feature_dim": int(raw.get("manifest_feature_dim", 32) or 32),
    }


def _load_raw(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict .pt payload: {path}")
    return raw


def _copy_with_control(
    *,
    src_path: Path,
    dst_path: Path,
    control: str,
    donor_payload: dict[str, Any] | None,
    manifest_dim: int,
    stats_dim: int,
    seed: int,
    resume: bool,
) -> str:
    if resume and dst_path.exists():
        return "resumed_existing_pt"
    raw = _load_raw(src_path)
    if control == "shuffled":
        if donor_payload is None:
            raise ValueError("shuffled control requires donor manifest payload")
        payload = dict(donor_payload)
        meta = dict(payload.get("manifest_meta") or {})
        meta["shortcut_control"] = "manifest_shuffled"
        payload["manifest_meta"] = meta
    elif control == "zeroed":
        payload = _zero_payload(raw, manifest_dim, stats_dim)
    elif control == "noisy":
        payload = _noisy_payload(raw, manifest_dim, stats_dim, seed)
    else:
        raise ValueError(f"Unsupported control: {control}")
    _set_manifest_payload(raw, payload)
    meta = dict(raw.get("direct_build_meta") or {})
    meta["manifest_shortcut_control"] = control
    raw["direct_build_meta"] = meta
    _atomic_torch_save(raw, dst_path)
    return "ok"


def _split_paths(cfg: dict[str, Any], split: str) -> tuple[Path, Path]:
    data_cfg = cfg["data"]
    root = data_cfg.get("root", "")
    return (
        Path(resolve(root, data_cfg[f"{split}_pt_dir"])),
        Path(resolve(root, data_cfg[f"{split}_csv"])),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    cfg = load_config(args.config)
    controls = args.controls or ["shuffled", "zeroed", "noisy"]
    out_root = Path(args.out_root)
    csv_out_dir = Path(args.csv_out_dir)
    manifest_cfg = cfg.get("model", {}).get("manifest_encoder", {})
    manifest_dim = int(manifest_cfg.get("in_dim", 256))
    stats_dim = int(manifest_cfg.get("stats_dim", 11))
    summary: dict[str, Any] = {}
    index_rows: list[dict[str, Any]] = []

    for split in args.splits:
        src_pt_dir, csv_path = _split_paths(cfg, split)
        fieldnames, rows, id_col = _read_csv(csv_path)
        sid_rows = {str(row.get(id_col, "")).strip().lower(): row for row in rows}
        pt_paths = [p for p in sorted(src_pt_dir.rglob("*.pt")) if p.stem.lower() in sid_rows]
        if args.limit and args.limit > 0:
            pt_paths = random.sample(pt_paths, min(int(args.limit), len(pt_paths)))

        donor_payloads: list[tuple[str, dict[str, Any]]] = []
        for pt_path in pt_paths:
            raw = _load_raw(pt_path)
            donor_payloads.append((pt_path.stem.lower(), _manifest_payload(raw)))
        if len(donor_payloads) > 1:
            donor_payloads = donor_payloads[1:] + donor_payloads[:1]

        split_summary: dict[str, int] = {}
        for control in controls:
            ok = 0
            failed = 0
            generated_sids: set[str] = set()
            for idx, pt_path in enumerate(pt_paths):
                sid = pt_path.stem.lower()
                donor_sid = ""
                donor_payload = None
                if control == "shuffled" and donor_payloads:
                    donor_sid, donor_payload = donor_payloads[idx % len(donor_payloads)]
                dst_path = out_root / control / split / pt_path.name
                try:
                    reason = _copy_with_control(
                        src_path=pt_path,
                        dst_path=dst_path,
                        control=control,
                        donor_payload=donor_payload,
                        manifest_dim=manifest_dim,
                        stats_dim=stats_dim,
                        seed=int(args.seed) + idx,
                        resume=bool(args.resume),
                    )
                    status = "ok"
                    ok += 1
                    generated_sids.add(sid)
                except Exception as exc:
                    status = "failed"
                    reason = f"{type(exc).__name__}: {exc}"
                    failed += 1
                index_rows.append(
                    {
                        "split": split,
                        "control": control,
                        "sid": sid,
                        "donor_sid": donor_sid,
                        "src_pt_path": str(pt_path),
                        "dst_pt_path": str(dst_path),
                        "status": status,
                        "reason": reason,
                    }
                )

            filtered_rows = [row for row in rows if str(row.get(id_col, "")).strip().lower() in generated_sids]
            _write_csv(csv_out_dir / f"{control}_{split}.csv", fieldnames, filtered_rows)
            split_summary[control] = ok
            if failed:
                split_summary[f"{control}_failed"] = failed
        summary[split] = split_summary

    _write_index(out_root / "manifest_shortcut_index.csv", index_rows)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "control", "sid", "donor_sid", "src_pt_path", "dst_pt_path", "status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Manifest shortcut-control .pt sets.")
    parser.add_argument("--config", nargs="+", default=["config/experiments/tri_modal_robust/base_tri_modal_robust.yaml"])
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--controls", nargs="+", choices=["shuffled", "zeroed", "noisy"], default=["shuffled", "zeroed", "noisy"])
    parser.add_argument("--out-root", default="D:/pts_manifest_controls")
    parser.add_argument("--csv-out-dir", default="results/manifest_shortcut_controls")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
