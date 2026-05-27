#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.robust.manifest_features import (
    build_manifest_vocab,
    load_manifest_vocab,
    read_manifest_jsonl,
    save_manifest_vocab,
    validate_manifest_vocab,
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
        "manifest_permission_dim": 0,
        "manifest_intent_dim": 0,
        "manifest_feature_dim": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach vectorized Manifest features to existing API+Graph .pt files.")
    parser.add_argument("--pt-dir", required=True, help="Input .pt directory.")
    parser.add_argument("--manifest-jsonl", required=True, help="Manifest raw JSONL produced by extract_manifest_features.py.")
    parser.add_argument("--out-dir", required=True, help="Output .pt directory.")
    parser.add_argument("--vocab", required=True, help="Manifest vocab YAML. Build it only on train set.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "extra"], help="Dataset split being augmented.")
    parser.add_argument("--build-vocab", action="store_true", help="Build vocab from --train-jsonl-for-vocab before vectorizing.")
    parser.add_argument("--train-jsonl-for-vocab", default="", help="Train Manifest JSONL used only when --build-vocab is set.")
    parser.add_argument("--max-permissions", type=int, default=128)
    parser.add_argument("--max-intents", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--manifest-dim", type=int, default=256)
    parser.add_argument("--allow-empty-vocab", action="store_true", help="Allow an empty Manifest vocab for debugging only.")
    args = parser.parse_args()

    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest_jsonl(args.manifest_jsonl)

    if args.build_vocab:
        if args.split != "train":
            raise SystemExit("--build-vocab is only allowed with --split train to prevent val/test vocabulary leakage.")
        if not args.train_jsonl_for_vocab:
            raise SystemExit("--build-vocab requires --train-jsonl-for-vocab; do not build vocab from --manifest-jsonl implicitly.")
        vocab_records = read_manifest_jsonl(args.train_jsonl_for_vocab)
        vocab = build_manifest_vocab(
            vocab_records.values(),
            max_permissions=args.max_permissions,
            max_intents=args.max_intents,
            max_features=args.max_features,
        )
        vocab["metadata"] = {
            "source_split": "train",
            "source_manifest_jsonl": str(Path(args.train_jsonl_for_vocab)),
            "leakage_guard": "train_only",
        }
        validate_manifest_vocab(
            vocab,
            require_train_metadata=True,
            allow_empty=args.allow_empty_vocab,
        )
        save_manifest_vocab(vocab, args.vocab)
    else:
        if not Path(args.vocab).exists():
            raise SystemExit("Manifest vocab does not exist. Build it once from --split train before augmenting val/test.")
        vocab = load_manifest_vocab(
            args.vocab,
            require_train_metadata=True,
            allow_empty=args.allow_empty_vocab,
        )

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
