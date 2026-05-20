#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml


def deep_update(base, patch):
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def write_yaml(path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base YAML config")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    out_dir = Path(args.out_dir)

    ablations = {
        "B0_api": {
            "model": {"fusion_mode": "api"},
            "loss": {
                "semantic_alignment_weight": 0.0,
                "gate_oracle_weight": 0.0,
                "branch_aux_weight": 0.0,
                "stage1_branch_aux_weight": 0.0,
            },
        },
        "B1_graph": {
            "model": {"fusion_mode": "graph"},
            "loss": {
                "semantic_alignment_weight": 0.0,
                "gate_oracle_weight": 0.0,
                "branch_aux_weight": 0.0,
                "stage1_branch_aux_weight": 0.0,
            },
        },
        "B2_concat": {
            "model": {"fusion_mode": "concat"},
            "loss": {
                "semantic_alignment_weight": 0.0,
                "gate_oracle_weight": 0.0,
            },
        },
        "B3_cross_attention": {
            "model": {"fusion_mode": "cross_attention"},
            "loss": {
                "semantic_alignment_weight": 0.0,
                "gate_oracle_weight": 0.0,
            },
        },
        "Ours_no_alignment": {
            "model": {
                "fusion_mode": "ours",
                "alignment": {"enabled": False},
            },
            "loss": {
                "semantic_alignment_weight": 0.0,
            },
        },
        "Ours_no_gate_oracle": {
            "model": {"fusion_mode": "ours"},
            "loss": {
                "gate_oracle_weight": 0.0,
            },
        },
        "Ours_no_uncertainty_gate": {
            "model": {
                "fusion_mode": "ours",
                "gate": {"uncertainty_inputs": False},
            },
        },
        "Ours_no_adaptation": {
            "data": {
                "adapt_csv": None,
                "adapt_pt_dir": None,
            },
            "train": {
                "adaptation_epochs": 0,
                "adaptation_ratio": 0.0,
                "replay_ratio": 0.0,
            },
        },
        "Ours_full": {
            "model": {"fusion_mode": "ours"},
        },
    }

    for name, patch in ablations.items():
        cfg = deep_update(base, patch)
        cfg.setdefault("train", {})
        cfg["train"]["exp_name"] = name
        write_yaml(out_dir / f"{name}.yaml", cfg)

    print(f"Wrote {len(ablations)} ablation configs to {out_dir}")


if __name__ == "__main__":
    main()