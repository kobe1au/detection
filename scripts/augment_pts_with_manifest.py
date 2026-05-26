#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fusion.robust.manifest_features import (
    build_manifest_vocab,
    load_manifest_vocab,
    read_manifest_jsonl,
    save_manifest_vocab,
    vectorize_manifest_record,
)


def _attach_manifest(raw, payload: dict):
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, dict):
                merged = dict(item)
                merged.update(payload)
                out.append(merged)
            else:
                out.append(item)
        return out
    if isinstance(raw, dict):
        out = dict(raw)
        out.update(payload)
        return out
    return raw


def _payload_for_missing(manifest_dim: int, category_dim: int, stats_dim: int):
    return {
        "manifest_x": torch.zeros((manifest_dim,), dtype=torch.float32),
        "manifest_permission_ids": torch.empty((0,), dtype=torch.long),
        "manifest_intent_ids": torch.empty((0,), dtype=torch.long),
        "manifest_category_counts": torch.zeros((category_dim,), dtype=torch.float32),
        "manifest_stats": torch.zeros((stats_dim,), dtype=torch.float32),
        "q_manifest": torch.tensor([0.0], dtype=torch.float32),
        "pert_manifest": torch.tensor([1.0], dtype=torch.float32),
        "manifest_meta": {"parse_error": "missing manifest record"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach vectorized Manifest features to existing API+Graph .pt files.")
    parser.add_argument("--pt-dir", required=True, help="Input .pt directory.")
    parser.add_argument("--manifest-jsonl", required=True, help="Manifest raw JSONL produced by extract_manifest_features.py.")
    parser.add_argument("--out-dir", required=True, help="Output .pt directory.")
    parser.add_argument("--vocab", required=True, help="Manifest vocab YAML. Build it only on train set.")
    parser.add_argument("--build-vocab", action="store_true", help="Build vocab from this JSONL before vectorizing.")
    parser.add_argument("--max-permissions", type=int, default=128)
    parser.add_argument("--max-intents", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--manifest-dim", type=int, default=256)
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest_jsonl(args.manifest_jsonl)

    if args.build_vocab:
        vocab = build_manifest_vocab(
            records.values(),
            max_permissions=args.max_permissions,
            max_intents=args.max_intents,
            max_features=args.max_features,
        )
        save_manifest_vocab(vocab, args.vocab)
    else:
        vocab = load_manifest_vocab(args.vocab)

    category_dim = len(vocab.get("categories") or [])
    stats_dim = 11
    missing_payload = _payload_for_missing(args.manifest_dim, category_dim, stats_dim)

    for pt_path in sorted(pt_dir.rglob("*.pt")):
        rel = pt_path.relative_to(pt_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sid = pt_path.stem.lower()
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
        rec = records.get(sid)
        payload = vectorize_manifest_record(rec, vocab, manifest_dim=args.manifest_dim) if rec else missing_payload
        torch.save(_attach_manifest(raw, payload), out_path)


if __name__ == "__main__":
    main()
