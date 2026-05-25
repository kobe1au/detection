#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate the full 2026 experiment YAML matrix.

The generated files are intentionally minimal override configs. They are
merged with config/base.yaml at runtime, so each YAML states only the
experimental claim it is testing.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


DEFAULT_OUT_DIR = Path("config/experiments")
ADAPT_PT_DIR = "/pts/adapt"
ADAPT_CSV = "results/labels/test_2023.csv"
DROP = object()

BASE_DEFAULTS = {
    "data": {
        "out_dir": "experiments",
        "adapt_pt_dir": ADAPT_PT_DIR,
        "adapt_csv": ADAPT_CSV,
    },
    "model": {
        "fusion_mode": "ours",
        "alignment": {
            "enabled": True,
            "adaptive_bias": True,
            "penalty_scale": 0.30,
            "bonus_scale": 0.80,
            "context_scale": 0.25,
        },
        "gate": {
            "mode": "learned",
            "quality_inputs": True,
            "temporal_reliability_inputs": True,
            "uncertainty_inputs": True,
            "time_inputs": True,
            "confidence_inputs": True,
            "confidence_source": "raw",
            "time_feature_set": "basic",
            "detach": True,
        },
    },
    "train": {
        "epochs": 80,
        "historical_epochs": 60,
        "adaptation_epochs": 20,
        "adaptation_ratio": 0.20,
        "replay_budget_mode": "selected_adapt_relative",
        "replay_budget_ratio": 0.50,
        "replay_strategy": "drift_matched",
        "adaptation_selection": "dbta",
        "dbta_balance": "predicted_label",
        "dbta_uncertainty_weight": 1.0,
        "dbta_disagreement_weight": 1.0,
        "dbta_prototype_weight": 1.0,
        "dbta_candidate_top_p": 0.5,
        "dbta_representative_k": 10,
        "dbta_representative_weight": 0.7,
        "dbta_selection_mode": "diversity_aware",
        "dbta_diversity_weight": 0.3,
        "dbta_diversity_metric": "cosine",
        "dbta_diversity_within_balance": True,
        "dbta_drift_replay_fraction": 0.5,
        "warmup_stage_epochs": 3,
    },
    "loss": {
        "semantic_alignment_weight": 0.05,
        "class_aware_alignment_same_class_weight": 0.20,
        "class_aware_alignment_temperature": 0.2,
        "local_alignment_weight": 0.02,
        "max_local_align_nodes": 128,
        "max_local_align_tokens": 256,
        "alignment_use_temporal_soft_weight": False,
        "alignment_use_temporal_reliability": True,
        "alignment_use_drift_reliability": True,
        "stage1_branch_aux_weight": 0.10,
        "branch_aux_weight": 0.10,
    },
}


def deep_update(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def merge(*patches: dict) -> dict:
    out: dict = {}
    for patch in patches:
        out = deep_update(out, patch)
    return out


def ratio_tag(ratio: float) -> str:
    return f"{int(round(float(ratio) * 100)):03d}"


def prune_defaults(value, default):
    if isinstance(value, dict) and isinstance(default, dict):
        pruned = {}
        for key, item in value.items():
            child = prune_defaults(item, default.get(key, DROP))
            if child is not DROP:
                pruned[key] = child
        return DROP if not pruned else pruned
    if default is not DROP and value == default:
        return DROP
    return value


def minimal_override(cfg: dict) -> dict:
    pruned = prune_defaults(cfg, BASE_DEFAULTS)
    return {} if pruned is DROP else pruned


def meta(exp_name: str, group: str) -> dict:
    return {
        "data": {
            "out_dir": f"experiments/{group}",
        },
        "train": {
            "exp_name": exp_name,
        },
    }

I1_HIST_CKPT = (
    "experiments/i1_dbta/I1_00_historical_concat/42/"
    "best_I1_00_historical_concat.pt"
)

def i1_warm_start() -> dict:
    return {
        "train": {
            "warm_start_historical_ckpt": I1_HIST_CKPT,
        }
    }


def historical_train() -> dict:
    return {
        "data": {
            "adapt_pt_dir": None,
            "adapt_csv": None,
        },
        "train": {
            "epochs": 60,
            "historical_epochs": 60,
            "adaptation_epochs": 0,
            "adaptation_ratio": 0.0,
            "replay_budget_ratio": 0.0,
            "replay_strategy": "static",
            "adaptation_selection": "random_pure",
            "warmup_stage_epochs": 0,
        },
    }


def continual_train(
    ratio: float = 0.20,
    replay_budget_ratio: float = 0.50,
    replay_strategy: str = "drift_matched",
    adaptation_selection: str = "dbta",
    adaptation_epochs: int = 20,
    dbta_drift_only: bool = False,
) -> dict:
    if replay_strategy == "drift_matched" and adaptation_selection != "dbta":
        raise ValueError("drift_matched replay requires dbta adaptation_selection")
    if adaptation_selection == "dbta" and ratio <= 0.0:
        raise ValueError("DBTA needs a positive adaptation ratio")

    historical_epochs = 60
    return {
        "data": {
            "adapt_pt_dir": ADAPT_PT_DIR,
            "adapt_csv": ADAPT_CSV,
        },
        "train": {
            "epochs": historical_epochs + adaptation_epochs,
            "historical_epochs": historical_epochs,
            "adaptation_epochs": adaptation_epochs,
            "adaptation_ratio": round(float(ratio), 4),
            "replay_budget_mode": "selected_adapt_relative",
            "replay_budget_ratio": round(float(replay_budget_ratio), 4),
            "replay_strategy": replay_strategy,
            "adaptation_selection": adaptation_selection,
            "dbta_balance": "predicted_label",
            "dbta_uncertainty_weight": 1.0,
            "dbta_disagreement_weight": 1.0,
            "dbta_prototype_weight": 1.0,
            "dbta_candidate_top_p": 1.0 if dbta_drift_only else 0.5,
            "dbta_representative_k": 10,
            "dbta_representative_weight": 0.0 if dbta_drift_only else 0.7,
            "dbta_selection_mode": "topk" if dbta_drift_only else "diversity_aware",
            "dbta_diversity_weight": 0.0 if dbta_drift_only else 0.3,
            "dbta_diversity_metric": "cosine",
            "dbta_diversity_within_balance": True,
            "dbta_drift_replay_fraction": 0.5,
        },
    }


def alignment_off_loss() -> dict:
    return {
        "loss": {
            "semantic_alignment_weight": 0.0,
            "local_alignment_weight": 0.0,
            "stage1_branch_aux_weight": 0.0,
            "branch_aux_weight": 0.0,
        },
    }


def alignment_off_model(fusion_mode: str) -> dict:
    return {
        "model": {
            "fusion_mode": fusion_mode,
            "alignment": {
                "enabled": False,
            },
            "gate": {
                "mode": "fixed",
            },
        },
    }


def fixed_gate() -> dict:
    return {
        "model": {
            "gate": {
                "mode": "fixed",
            },
        },
    }


def learned_gate(
    *,
    quality: bool,
    temporal_reliability: bool,
    uncertainty: bool,
    time: bool,
    confidence: bool,
) -> dict:
    return {
        "model": {
            "gate": {
                "mode": "learned",
                "quality_inputs": quality,
                "temporal_reliability_inputs": temporal_reliability,
                "uncertainty_inputs": uncertainty,
                "time_inputs": time,
                "confidence_inputs": confidence,
                "confidence_source": "raw",
                "time_feature_set": "basic",
                "detach": True,
            },
        },
    }


def method_alignment(enabled: bool = True) -> dict:
    return {
        "model": {
            "alignment": {
                "enabled": enabled,
            },
        },
    }


def ours_base() -> dict:
    return merge(
        alignment_off_model("ours"),
        {
            "train": {
                "warmup_stage_epochs": 0,
            },
        },
        alignment_off_loss(),
    )


def semantic_alignment_loss(
    *,
    semantic: float,
    class_aware: float = 0.20,
    local: float = 0.0,
    temporal_reliability: bool = False,
    drift_reliability: bool = False,
    branch_aux: float = 0.10,
    stage1_aux: float = 0.10,
) -> dict:
    return {
        "loss": {
            "semantic_alignment_weight": float(semantic),
            "class_aware_alignment_same_class_weight": float(class_aware),
            "class_aware_alignment_temperature": 0.2,
            "local_alignment_weight": float(local),
            "max_local_align_nodes": 128,
            "max_local_align_tokens": 256,
            "alignment_use_temporal_soft_weight": False,
            "alignment_use_temporal_reliability": temporal_reliability,
            "alignment_use_drift_reliability": drift_reliability,
            "stage1_branch_aux_weight": float(stage1_aux),
            "branch_aux_weight": float(branch_aux),
        },
    }


def full_alignment(fixed: bool = True) -> dict:
    gate = fixed_gate() if fixed else learned_gate(
        quality=True,
        temporal_reliability=True,
        uncertainty=True,
        time=True,
        confidence=True,
    )
    return merge(
        {
            "model": {
                "fusion_mode": "ours",
            },
            "train": {
                "warmup_stage_epochs": 3,
            },
        },
        method_alignment(True),
        gate,
        semantic_alignment_loss(
            semantic=0.05,
            class_aware=0.20,
            local=0.02,
            temporal_reliability=True,
            drift_reliability=True,
        ),
    )


def full_model() -> dict:
    return full_alignment(fixed=False)


def add(configs: dict, group: str, filename: str, exp_name: str, *patches: dict) -> None:
    configs[f"{group}/{filename}"] = merge(meta(exp_name, group), *patches)


def build_configs() -> tuple[dict[str, dict], dict[str, list[str]]]:
    configs: dict[str, dict] = {}

    # Baselines: modality and fusion controls without 2023 adaptation.
    for idx, (mode, label) in enumerate(
        [
            ("api", "api"),
            ("graph", "graph"),
            ("concat", "concat"),
            ("late_fusion", "late_fusion"),
            ("cross_attention", "cross_attention"),
        ]
    ):
        add(
            configs,
            "baselines",
            f"B{idx}_{label}_erm.yaml",
            f"B{idx}_{label}_erm",
            historical_train(),
            alignment_off_model(mode),
            alignment_off_loss(),
        )

    # I1: adaptation selection and replay are isolated on concat, so architecture
    # changes cannot explain budgeted adaptation/replay gains.
    i1_group = "i1_dbta"
    add(
        configs,
        i1_group,
        "I1_00_historical_concat.yaml",
        "I1_00_historical_concat",
        historical_train(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_01_random_pure020_dynamic_replay.yaml",
        "I1_01_random_pure020_dynamic_replay",
        continual_train(0.20, 0.50, "dynamic_year_class", "random_pure"),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_02_random_class_balanced020_dynamic_replay.yaml",
        "I1_02_random_class_balanced020_dynamic_replay",
        continual_train(0.20, 0.50, "dynamic_year_class", "random_class_balanced"),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_03_dbta_drift_only_020_dynamic_replay.yaml",
        "I1_03_dbta_drift_only_020_dynamic_replay",
        continual_train(0.20, 0.50, "dynamic_year_class", "dbta", dbta_drift_only=True),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_04_dbta_v2_020_no_replay.yaml",
        "I1_04_dbta_v2_020_no_replay",
        continual_train(0.20, 0.0, "static", "dbta"),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_05_dbta_v2_020_static_replay.yaml",
        "I1_05_dbta_v2_020_static_replay",
        continual_train(0.20, 0.50, "static", "dbta"),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_06_dbta_v2_020_dynamic_replay.yaml",
        "I1_06_dbta_v2_020_dynamic_replay",
        continual_train(0.20, 0.50, "dynamic_year_class", "dbta"),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        i1_group,
        "I1_07_dbta_v2_020_drift_matched.yaml",
        "I1_07_dbta_v2_020_drift_matched",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        i1_warm_start(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    for idx, ratio in enumerate([0.05, 0.10, 0.50, 1.00], start=8):
        tag = ratio_tag(ratio)
        add(
            configs,
            i1_group,
            f"I1_{idx:02d}_dbta_v2_{tag}_drift_matched.yaml",
            f"I1_{idx:02d}_dbta_v2_{tag}_drift_matched",
            continual_train(ratio, 0.50, "drift_matched", "dbta"),
            i1_warm_start(),
            alignment_off_model("concat"),
            alignment_off_loss(),
        )

    # Replay ablation: same full model, same selected 20% recent pool.
    replay_group = "replay_ablation"
    for idx, (strategy, replay_budget_ratio, label) in enumerate(
        [
            ("static", 0.0, "no_replay"),
            ("static", 0.50, "static_replay"),
            ("dynamic_year_class", 0.50, "dynamic_year_class"),
            ("drift_matched", 0.50, "drift_matched"),
        ]
    ):
        add(
            configs,
            replay_group,
            f"RE{idx}_full_dbta_v2_020_{label}.yaml",
            f"RE{idx}_full_dbta_v2_020_{label}",
            continual_train(0.20, replay_budget_ratio, strategy, "dbta"),
            full_model(),
        )

    # I2: alignment hierarchy, with I1 fixed to DBTA v2 20% + budget-normalized drift-matched replay.
    i2_group = "i2_alignment"
    add(
        configs,
        i2_group,
        "I2_00_no_alignment_fixed_gate.yaml",
        "I2_00_no_alignment_fixed_gate",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        method_alignment(False),
        fixed_gate(),
        {"loss": {"semantic_alignment_weight": 0.0, "local_alignment_weight": 0.0}},
    )
    add(
        configs,
        i2_group,
        "I2_01_semantic_only_fixed_gate.yaml",
        "I2_01_semantic_only_fixed_gate",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        method_alignment(False),
        fixed_gate(),
        semantic_alignment_loss(semantic=0.05, class_aware=0.00, local=0.0),
    )
    add(
        configs,
        i2_group,
        "I2_02_class_aware_semantic_fixed_gate.yaml",
        "I2_02_class_aware_semantic_fixed_gate",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        method_alignment(False),
        fixed_gate(),
        semantic_alignment_loss(semantic=0.05, class_aware=0.20, local=0.0),
    )
    add(
        configs,
        i2_group,
        "I2_03_method_bias_only_fixed_gate.yaml",
        "I2_03_method_bias_only_fixed_gate",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        fixed_gate(),
        {"loss": {"semantic_alignment_weight": 0.0, "local_alignment_weight": 0.0}},
        {"train": {"warmup_stage_epochs": 3}},
    )
    add(
        configs,
        i2_group,
        "I2_04_hierarchical_full_fixed_gate.yaml",
        "I2_04_hierarchical_full_fixed_gate",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        full_alignment(fixed=True),
    )

    # I3: gate evidence ablation, holding I1 and I2 fixed.
    i3_group = "i3_gate"
    gate_variants = [
        ("I3_00_fixed_gate.yaml", "I3_00_fixed_gate", fixed_gate()),
        ("I3_01_learned_emb_only.yaml", "I3_01_learned_emb_only", learned_gate(quality=False, temporal_reliability=False, uncertainty=False, time=False, confidence=False)),
        ("I3_02_quality_only.yaml", "I3_02_quality_only", learned_gate(quality=True, temporal_reliability=False, uncertainty=False, time=False, confidence=False)),
        ("I3_03_temporal_reliability_only.yaml", "I3_03_temporal_reliability_only", learned_gate(quality=False, temporal_reliability=True, uncertainty=False, time=False, confidence=False)),
        ("I3_04_time_features_only.yaml", "I3_04_time_features_only", learned_gate(quality=False, temporal_reliability=False, uncertainty=False, time=True, confidence=False)),
        ("I3_05_uncertainty_only.yaml", "I3_05_uncertainty_only", learned_gate(quality=False, temporal_reliability=False, uncertainty=True, time=False, confidence=False)),
        ("I3_06_confidence_only.yaml", "I3_06_confidence_only", learned_gate(quality=False, temporal_reliability=False, uncertainty=False, time=False, confidence=True)),
        ("I3_07_full_gate.yaml", "I3_07_full_gate", learned_gate(quality=True, temporal_reliability=True, uncertainty=True, time=True, confidence=True)),
    ]
    for filename, exp_name, gate_patch in gate_variants:
        add(
            configs,
            i3_group,
            filename,
            exp_name,
            continual_train(0.20, 0.50, "drift_matched", "dbta"),
            full_alignment(fixed=True),
            gate_patch,
        )

    # Ratio sweep: final model under increasing recent-year budget.
    ratio_group = "ratio_sweep"
    add(
        configs,
        ratio_group,
        "R0_full_no_adapt.yaml",
        "R0_full_no_adapt",
        historical_train(),
        full_model(),
    )
    for idx, ratio in enumerate([0.05, 0.10, 0.20, 0.50, 1.00], start=1):
        tag = ratio_tag(ratio)
        add(
            configs,
            ratio_group,
            f"R{idx}_full_dbta_v2_{tag}.yaml",
            f"R{idx}_full_dbta_v2_{tag}",
            continual_train(ratio, 0.50, "drift_matched", "dbta"),
            full_model(),
        )

    # Main chain: the shortest story from baseline to final full system.
    main_group = "main_chain"
    add(
        configs,
        main_group,
        "M0_concat_erm.yaml",
        "M0_concat_erm",
        historical_train(),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        main_group,
        "M1_i1_dbta_v2_concat020.yaml",
        "M1_i1_dbta_v2_concat020",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        alignment_off_model("concat"),
        alignment_off_loss(),
    )
    add(
        configs,
        main_group,
        "M2_i1_i2_alignment_fixed_gate020.yaml",
        "M2_i1_i2_alignment_fixed_gate020",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        full_alignment(fixed=True),
    )
    add(
        configs,
        main_group,
        "M3_full_dbta_v2_020.yaml",
        "M3_full_dbta_v2_020",
        continual_train(0.20, 0.50, "drift_matched", "dbta"),
        full_model(),
    )
    add(
        configs,
        main_group,
        "M4_full_random_class_balanced100_static.yaml",
        "M4_full_random_class_balanced100_static",
        continual_train(1.00, 0.50, "static", "random_class_balanced"),
        full_model(),
    )

    groups = {
        "baselines": sorted(k for k in configs if k.startswith("baselines/")),
        "i1": sorted(k for k in configs if k.startswith("i1_dbta/")),
        "replay": sorted(k for k in configs if k.startswith("replay_ablation/")),
        "i2": sorted(k for k in configs if k.startswith("i2_alignment/")),
        "i3": sorted(k for k in configs if k.startswith("i3_gate/")),
        "ratio": sorted(k for k in configs if k.startswith("ratio_sweep/")),
        "main": sorted(k for k in configs if k.startswith("main_chain/")),
    }
    groups["full"] = groups["ratio"]
    groups["final"] = ["main_chain/M3_full_dbta_v2_020.yaml"]
    groups["all"] = (
        groups["baselines"]
        + groups["i1"]
        + groups["replay"]
        + groups["i2"]
        + groups["i3"]
        + groups["ratio"]
    )
    return configs, groups


def build_i3_gate_signal_manifest(configs: dict[str, dict], groups: dict[str, list[str]]) -> dict:
    signals = {}
    for rel_path in groups.get("i3", []):
        effective = merge(BASE_DEFAULTS, configs[rel_path])
        exp_name = str(effective.get("train", {}).get("exp_name", Path(rel_path).stem))
        gate = effective.get("model", {}).get("gate", {})
        mode = str(gate.get("mode", "learned"))
        gate_uses_inputs = mode != "fixed"
        signals[exp_name] = {
            "mode": mode,
            "quality_inputs": gate_uses_inputs and bool(gate.get("quality_inputs", False)),
            "temporal_reliability_inputs": gate_uses_inputs and bool(gate.get("temporal_reliability_inputs", False)),
            "time_inputs": gate_uses_inputs and bool(gate.get("time_inputs", False)),
            "uncertainty_inputs": gate_uses_inputs and bool(gate.get("uncertainty_inputs", False)),
            "confidence_inputs": gate_uses_inputs and bool(gate.get("confidence_inputs", False)),
        }
    return signals


def write_yaml(path: Path, cfg: dict, generated: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if generated:
            f.write("# Auto-generated by scripts/make_ablation_configs.py. Do not edit by hand.\n")
        f.write(dump_yaml(minimal_override(cfg)))


def yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def dump_yaml(value, indent: int = 0) -> str:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(dump_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(dump_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{pad}{yaml_scalar(value)}\n"


def clean_generated_yaml(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    marker = "# Auto-generated by scripts/make_ablation_configs.py."
    for path in out_dir.rglob("*.yaml"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        except OSError:
            continue
        if first_line.startswith(marker):
            path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated experiment YAMLs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    configs, groups = build_configs()
    clean_generated_yaml(out_dir)

    for rel_path, cfg in configs.items():
        write_yaml(out_dir / rel_path, cfg)

    manifest = {
        "generated_by": "scripts/make_ablation_configs.py",
        "base_config": "config/base.yaml",
        "groups": {
            key: [str(out_dir / path).replace("\\", "/") for path in paths]
            for key, paths in groups.items()
        },
        "i3_gate_signals": build_i3_gate_signal_manifest(configs, groups),
        "recommended_order": ["baselines", "i1", "replay", "i2", "i3", "ratio", "main"],
    }
    write_yaml(out_dir / "_manifest.yaml", manifest)

    print(f"Wrote {len(configs)} experiment configs to {out_dir}")
    print(f"Wrote manifest to {out_dir / '_manifest.yaml'}")


if __name__ == "__main__":
    main()
