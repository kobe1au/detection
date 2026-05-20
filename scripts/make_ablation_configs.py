#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


CONTINUAL_DATA_PATCH = {
    "data": {
        "adapt_pt_dir": "pts/adapt",
        "adapt_csv": "resource/dataset_split_2018_2024/adapt_2023.csv",
    },
    "train": {
        "epochs": 70,
        "historical_epochs": 60,
        "adaptation_epochs": 10,
        "adaptation_ratio": 0.20,
        "replay_ratio": 0.25,
        "replay_strategy": "dynamic_year_class",
    },
}


OURS_FULL_PATCH = {
    "model": {
        "fusion_mode": "ours",
        "alignment": {
            "enabled": True,
            "adaptive_bias": True,
            "penalty_scale": 0.5,
            "bonus_scale": 1.0,
            "context_scale": 0.35,
        },
        "gate": {
            "mode": "learned",
            "quality_inputs": True,
            "uncertainty_inputs": True,
            "detach": True,
        },
    },
    "loss": {
        "semantic_alignment_weight": 0.03,
        "class_aware_alignment_same_class_weight": 0.25,
        "class_aware_alignment_temperature": 0.2,
        "branch_aux_weight": 0.10,
        "stage1_branch_aux_weight": 0.30,
        "gate_oracle_weight": 0.05,
        "gate_oracle_temperature": 0.5,
        "gate_oracle_smoothing": 0.10,
        "gate_oracle_start_phase": "adaptation",
        "gate_oracle_adaptation_only": True,
    },
}


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

    ours_full = deep_update(CONTINUAL_DATA_PATCH, OURS_FULL_PATCH)

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
            **ours_full,
            "model": deep_update(ours_full["model"], {"alignment": {"enabled": False}}),
            "loss": deep_update(
                ours_full["loss"],
                {
                    "semantic_alignment_weight": 0.0,
                    "class_aware_alignment_same_class_weight": 0.0,
                },
            ),
        },
        "Ours_no_gate_oracle": {
            **ours_full,
            "loss": deep_update(ours_full["loss"], {"gate_oracle_weight": 0.0}),
        },
        "Ours_no_uncertainty_gate": {
            **ours_full,
            "model": deep_update(ours_full["model"], {"gate": {"uncertainty_inputs": False}}),
        },
        "Ours_no_adaptation": {
            **ours_full,
            "data": deep_update(
                ours_full["data"],
                {
                    "adapt_csv": None,
                    "adapt_pt_dir": None,
                },
            ),
            "train": deep_update(
                ours_full["train"],
                {
                    "epochs": 60,
                    "historical_epochs": 60,
                    "adaptation_epochs": 0,
                    "adaptation_ratio": 0.0,
                    "replay_ratio": 0.0,
                    "replay_strategy": "static",
                },
            ),
        },
        "Ours_full": {
            **ours_full,
        },
        "Ours_static_replay": {
            **ours_full,
            "train": deep_update(
                ours_full["train"],
                {"replay_strategy": "static"},
            ),
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
