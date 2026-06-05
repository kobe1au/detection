from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.train import deep_update, load_config, run


DEFAULT_CONFIG = "config/experiments/tri_modal_robust/tune/optuna_base.yaml"


def _suggest_i2(trial) -> dict[str, Any]:
    weight = trial.suggest_categorical(
        "cross_source_consistency_weight",
        [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03],
    )
    if weight > 0.0:
        min_reliability = trial.suggest_categorical(
            "cross_source_min_reliability",
            [0.2, 0.3, 0.4, 0.5],
        )
        min_consistency = trial.suggest_categorical(
            "cross_source_min_consistency",
            [0.0, 0.1, 0.2, 0.3],
        )
        active_only = trial.suggest_categorical(
            "semantic_active_only",
            [True, False],
        )
    else:
        min_reliability = 0.0
        min_consistency = 0.0
        active_only = False
    return {
        "model": {
            "fusion_mode": "tri_modal_ours",
            "gate": {
                "use_consistency_evidence": True,
                "use_conflict_evidence": True,
            },
        },
        "loss": {
            "branch_aux_weight": 0.05,
            "semantic_reconstruction_weight": 0.0,
            "cross_source_consistency_weight": weight,
            "gate_prior_weight": 0.0,
            "cross_source_min_reliability": min_reliability,
            "cross_source_min_consistency": min_consistency,
            "semantic_active_only": active_only,
        },
    }


def _suggest_i3(trial) -> dict[str, Any]:
    return {
        "model": {
            "fusion_mode": "tri_modal_ours",
            "gate": {
                "hidden_dim": trial.suggest_categorical("gate_hidden_dim", [64, 128, 256]),
                "apply_alive_mask": True,
            },
        },
        "loss": {
            "branch_aux_weight": trial.suggest_categorical("branch_aux_weight", [0.02, 0.05, 0.1]),
            "gate_prior_weight": trial.suggest_categorical(
                "gate_prior_weight",
                [0.0, 0.001, 0.003, 0.005, 0.01],
            ),
        },
    }


def _suggest_aug(trial) -> dict[str, Any]:
    profile = trial.suggest_categorical("perturb_strength_profile", ["low", "mid", "high"])
    strengths = {
        "low": [0.1, 0.3],
        "mid": [0.1, 0.3, 0.5],
        "high": [0.3, 0.5, 0.7],
    }[profile]
    return {
        "robust": {
            "train_aug": True,
            "perturb_prob": trial.suggest_categorical("perturb_prob", [0.3, 0.5, 0.7]),
            "perturb_strengths": strengths,
        }
    }


def _stage_override(stage: str, trial) -> dict[str, Any]:
    if stage == "i2":
        return _suggest_i2(trial)
    if stage == "i3":
        return _suggest_i3(trial)
    if stage == "aug":
        return _suggest_aug(trial)
    raise ValueError(f"Unsupported tuning stage: {stage}")


def _objective_base_override(args, trial_number: int, seed: int) -> dict[str, Any]:
    return {
        "data": {"out_dir": str(args.output_dir / "trials" / args.study_name)},
        "train": {
            "exp_name": f"optuna_{args.stage}_trial_{trial_number:04d}",
            "seed": int(seed),
            "tuning_mode": True,
            "checkpoint_metric": "robust_composite",
        },
        "eval": {
            "run_test": False,
            "run_robust_test": False,
            "extra_sets": [],
            "robust_val": {"enabled": True},
        },
    }


def _cleanup_trial_artifacts(cfg: dict, keep_checkpoints: bool) -> None:
    if keep_checkpoints:
        return
    trial_dir = (
        Path(cfg["data"]["out_dir"])
        / str(cfg["train"]["exp_name"])
        / str(cfg["train"]["seed"])
    )
    for name in ("best_tri_modal_robust.pt", "gate_diagnostics.csv", "gate_diagnostics_extra_eval.csv"):
        path = trial_dir / name
        if path.exists():
            path.unlink()


def _best_override(stage: str, params: dict[str, Any]) -> dict[str, Any]:
    if stage == "i2":
        return {
            "model": {
                "fusion_mode": "tri_modal_ours",
                "gate": {
                    "use_consistency_evidence": True,
                    "use_conflict_evidence": True,
                },
            },
            "loss": {
                "branch_aux_weight": 0.05,
                "semantic_reconstruction_weight": 0.0,
                "cross_source_consistency_weight": params["cross_source_consistency_weight"],
                "gate_prior_weight": 0.0,
                "cross_source_min_reliability": params.get("cross_source_min_reliability", 0.0),
                "cross_source_min_consistency": params.get("cross_source_min_consistency", 0.0),
                "semantic_active_only": params.get("semantic_active_only", False),
            },
        }
    if stage == "i3":
        return {
            "model": {
                "fusion_mode": "tri_modal_ours",
                "gate": {
                    "hidden_dim": params["gate_hidden_dim"],
                    "apply_alive_mask": True,
                },
            },
            "loss": {
                "branch_aux_weight": params["branch_aux_weight"],
                "gate_prior_weight": params["gate_prior_weight"],
            },
        }
    profile = params["perturb_strength_profile"]
    return {
        "robust": {
            "train_aug": True,
            "perturb_prob": params["perturb_prob"],
            "perturb_strengths": {
                "low": [0.1, 0.3],
                "mid": [0.1, 0.3, 0.5],
                "high": [0.3, 0.5, 0.7],
            }[profile],
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-wise Optuna tuning with robust validation only.")
    parser.add_argument("--stage", choices=("i2", "i3", "aug"), required=True)
    parser.add_argument("--config", nargs="+", default=[DEFAULT_CONFIG], help="Base YAML configs, applied left to right.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Optional study timeout in seconds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Average objective across these seeds.")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to a local SQLite study.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/optuna"))
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.stage == "i3" and len(args.config) < 2:
        raise ValueError("i3 tuning requires a fixed i2 override after the base config")
    if args.stage == "aug" and len(args.config) < 3:
        raise ValueError("augmentation tuning requires fixed i2 and i3 overrides after the base config")
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required. Install it with: pip install optuna") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"tri_modal_robust_{args.stage}"
    args.study_name = study_name
    storage = args.storage or f"sqlite:///{(args.output_dir / 'study.db').resolve().as_posix()}"
    base_cfg = load_config(args.config)
    fingerprint_payload = {
        "version": 2,
        "stage": args.stage,
        "base_configs": list(args.config),
        "base_cfg": base_cfg,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    def objective(trial) -> float:
        stage_cfg = _stage_override(args.stage, trial)
        scores: list[float] = []
        per_seed: dict[str, float] = {}
        for seed in args.seeds:
            cfg = deep_update(copy.deepcopy(base_cfg), stage_cfg)
            cfg = deep_update(cfg, _objective_base_override(args, trial.number, seed))
            summary = run(cfg)
            score = float(summary["best_checkpoint_score"])
            scores.append(score)
            per_seed[str(seed)] = score
            _cleanup_trial_artifacts(cfg, args.keep_checkpoints)
        mean_score = sum(scores) / len(scores)
        trial.set_user_attr("per_seed_scores", per_seed)
        trial.set_user_attr("mean_score", mean_score)
        return mean_score

    sampler = (
        optuna.samplers.GridSampler(
            {
                "perturb_strength_profile": ["low", "mid", "high"],
                "perturb_prob": [0.3, 0.5, 0.7],
            },
            seed=args.sampler_seed,
        )
        if args.stage == "aug"
        else optuna.samplers.TPESampler(seed=args.sampler_seed)
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    previous_fingerprint = study.user_attrs.get("config_fingerprint")
    if previous_fingerprint and previous_fingerprint != fingerprint:
        raise ValueError(
            "Existing Optuna study was created from a different base configuration/search protocol. "
            "Use a new --study-name or --storage."
        )
    if not previous_fingerprint and len(study.trials) > 0:
        raise ValueError(
            "Existing Optuna study has trials but no configuration fingerprint. "
            "Use a new --study-name or --storage to avoid mixing incompatible trials."
        )
    study.set_user_attr("config_fingerprint", fingerprint)
    n_trials = min(args.trials, 9) if args.stage == "aug" else args.trials
    study.optimize(objective, n_trials=n_trials, timeout=args.timeout, n_jobs=1)

    best_payload = {
        "stage": args.stage,
        "study_name": study.study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "base_configs": list(args.config),
        "seeds": list(args.seeds),
        "config_fingerprint": fingerprint,
    }
    with open(args.output_dir / f"best_{args.stage}.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)
    with open(args.output_dir / f"best_{args.stage}_override.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_best_override(args.stage, study.best_params), f, sort_keys=False)
    print(json.dumps(best_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
