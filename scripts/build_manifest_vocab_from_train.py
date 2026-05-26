#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.robust.manifest_features import build_manifest_vocab, read_manifest_jsonl, save_manifest_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Manifest vocabulary from train split JSONL only.")
    parser.add_argument("--train-manifest-jsonl", required=True, help="Train split Manifest JSONL.")
    parser.add_argument("--vocab", required=True, help="Output Manifest vocab YAML.")
    parser.add_argument("--max-permissions", type=int, default=128)
    parser.add_argument("--max-intents", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=32)
    args = parser.parse_args()

    records = read_manifest_jsonl(args.train_manifest_jsonl)
    vocab = build_manifest_vocab(
        records.values(),
        max_permissions=args.max_permissions,
        max_intents=args.max_intents,
        max_features=args.max_features,
    )
    vocab["metadata"] = {
        "source_split": "train",
        "source_manifest_jsonl": str(Path(args.train_manifest_jsonl)),
        "leakage_guard": "train_only",
    }
    save_manifest_vocab(vocab, args.vocab)


if __name__ == "__main__":
    main()
