#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.train import deep_update, load_config


CONFIG_ROOT = Path("config/experiments/tri_modal_robust")


def _posix_rel(base: Path, target: Path) -> str:
    import os

    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _name_payload(exp_name: str) -> dict[str, Any]:
    return {"train": {"exp_name": exp_name}}


def _seed_payload(seed: int, exp_name: str) -> dict[str, Any]:
    return {"train": {"exp_name": exp_name, "seed": int(seed), "deterministic": True}}


def _strip_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    payload.pop("defaults", None)
    return payload


def _i2_only_override(path: str | Path | None) -> dict[str, Any]:
    """Keep only I2-related fields from an Optuna best-i2 override.

    The saved best-i2 override intentionally contains neutral branch/gate-prior
    values from the tuning stage.  Post-Optuna I2 ablations must not let those
    values overwrite the fixed best-I3 gate settings.
    """
    if not path:
        return {
            "model": {
                "fusion_mode": "tri_modal_ours",
                "gate": {
                    "use_consistency_evidence": True,
                    "use_conflict_evidence": True,
                },
            },
            "loss": {
                "semantic_reconstruction_weight": 0.0,
                "cross_source_consistency_weight": 0.02,
                "cross_source_min_reliability": 0.3,
                "cross_source_min_consistency": 0.1,
                "semantic_active_only": False,
            },
        }
    raw = _strip_defaults(_load_yaml(Path(path)))
    model_gate = ((raw.get("model") or {}).get("gate") or {})
    loss = raw.get("loss") or {}
    out: dict[str, Any] = {
        "model": {
            "fusion_mode": "tri_modal_ours",
            "gate": {
                "use_consistency_evidence": bool(model_gate.get("use_consistency_evidence", True)),
                "use_conflict_evidence": bool(model_gate.get("use_conflict_evidence", True)),
            },
        },
        "loss": {},
    }
    for key in (
        "semantic_reconstruction_weight",
        "cross_source_consistency_weight",
        "cross_source_min_reliability",
        "cross_source_min_consistency",
        "semantic_active_only",
    ):
        if key in loss:
            out["loss"][key] = loss[key]
    out["loss"].setdefault("semantic_reconstruction_weight", 0.0)
    out["loss"].setdefault("cross_source_consistency_weight", 0.02)
    out["loss"].setdefault("cross_source_min_reliability", 0.3)
    out["loss"].setdefault("cross_source_min_consistency", 0.1)
    out["loss"].setdefault("semantic_active_only", False)
    return out


def _i3_branch_aux_override(path: str | Path | None) -> dict[str, Any]:
    """Carry the tuned branch-auxiliary weight into all i3 gate ablations.

    The i3 stage tunes gate-side training choices.  Fixed, confidence, and
    heuristic reliability gates should still share the same branch auxiliary
    supervision strength; otherwise the ablation changes two factors at once.
    """
    if not path:
        return {"loss": {"branch_aux_weight": 0.05}}
    raw = _strip_defaults(_load_yaml(Path(path)))
    loss = raw.get("loss") or {}
    if "branch_aux_weight" not in loss:
        return {"loss": {"branch_aux_weight": 0.05}}
    return {"loss": {"branch_aux_weight": loss["branch_aux_weight"]}}


def _generated_config(
    out_file: Path,
    defaults: list[Path],
    leaf_override: dict[str, Any],
) -> dict[str, Any]:
    return {
        "defaults": [_posix_rel(out_file.parent, item) for item in defaults],
        **leaf_override,
    }


def _final_defaults(args: argparse.Namespace) -> list[Path]:
    defaults = [CONFIG_ROOT / "full" / "ours.yaml"]
    if args.best_i2:
        defaults.append(Path(args.best_i2))
    if args.best_i3:
        defaults.append(Path(args.best_i3))
    if args.best_aug:
        defaults.append(Path(args.best_aug))
    return defaults


def _i1_configs(args: argparse.Namespace) -> list[tuple[str, list[Path], dict[str, Any]]]:
    # I1 tests modality/reliability modeling. It intentionally does not inherit
    # i2/i3 best overrides because those would alter the mechanism under test.
    paths = [
        "api_only.yaml",
        "graph_only.yaml",
        "manifest_only.yaml",
        "api_graph_concat.yaml",
        "tri_modal_concat.yaml",
        "reliability_gate.yaml",
    ]
    out = []
    for name in paths:
        source = CONFIG_ROOT / "i1" / name
        stem = Path(name).stem
        out.append((f"i1_{stem}.yaml", [source], _name_payload(f"post_{args.tag}_i1_{stem}")))
    return out


def _i2_configs(args: argparse.Namespace) -> list[tuple[str, list[Path], dict[str, Any]]]:
    # Fix i3/augmentation, then vary only i2-related consistency settings.
    fixed = [CONFIG_ROOT / "full" / "ours.yaml"]
    if args.best_i3:
        fixed.append(Path(args.best_i3))
    if args.best_aug:
        fixed.append(Path(args.best_aug))
    variants = {
        "i2_no_consistency.yaml": {
            "model": {"fusion_mode": "tri_modal_ours", "gate": {"use_consistency_evidence": False, "use_conflict_evidence": False}},
            "loss": {
                "semantic_reconstruction_weight": 0.0,
                "cross_source_consistency_weight": 0.0,
                "cross_source_min_reliability": 0.0,
                "cross_source_min_consistency": 0.0,
                "semantic_active_only": False,
            },
        },
        "i2_consistency_evidence_only.yaml": {
            "model": {"fusion_mode": "tri_modal_ours", "gate": {"use_consistency_evidence": True, "use_conflict_evidence": False}},
            "loss": {"semantic_reconstruction_weight": 0.0, "cross_source_consistency_weight": 0.0},
        },
        "i2_conflict_evidence_only.yaml": {
            "model": {"fusion_mode": "tri_modal_ours", "gate": {"use_consistency_evidence": False, "use_conflict_evidence": True}},
            "loss": {"semantic_reconstruction_weight": 0.0, "cross_source_consistency_weight": 0.0},
        },
        "i2_evidence_only.yaml": {
            "model": {"fusion_mode": "tri_modal_ours", "gate": {"use_consistency_evidence": True, "use_conflict_evidence": True}},
            "loss": {"semantic_reconstruction_weight": 0.0, "cross_source_consistency_weight": 0.0},
        },
        "i2_loss_only.yaml": {
            "model": {"fusion_mode": "tri_modal_ours", "gate": {"use_consistency_evidence": False, "use_conflict_evidence": False}},
            "loss": _i2_only_override(args.best_i2)["loss"],
        },
        "i2_semantic_reconstruction_only.yaml": {
            "model": {"fusion_mode": "tri_modal_ours", "gate": {"use_consistency_evidence": False, "use_conflict_evidence": False}},
            "loss": {
                "semantic_reconstruction_weight": 0.02,
                "cross_source_consistency_weight": 0.0,
                "cross_source_min_reliability": 0.3,
                "cross_source_min_consistency": 0.0,
                "semantic_active_only": False,
            },
        },
        "i2_evidence_plus_loss.yaml": _i2_only_override(args.best_i2),
    }
    out = []
    for name, override in variants.items():
        stem = Path(name).stem
        leaf = deep_update(_name_payload(f"post_{args.tag}_{stem}"), override)
        out.append((name, fixed, leaf))
    return out


def _i3_configs(args: argparse.Namespace) -> list[tuple[str, list[Path], dict[str, Any]]]:
    # Fix i2/augmentation, then vary only the gate mechanism.  best_i3 is not a
    # default here because it would overwrite fixed/reliability/confidence gates.
    fixed = [CONFIG_ROOT / "full" / "ours.yaml"]
    if args.best_i2:
        fixed.append(Path(args.best_i2))
    if args.best_aug:
        fixed.append(Path(args.best_aug))
    tuned_i3 = _strip_defaults(_load_yaml(Path(args.best_i3))) if args.best_i3 else {
        "model": {"fusion_mode": "tri_modal_ours", "gate": {"apply_alive_mask": True}},
        "loss": {"gate_prior_weight": 0.01},
    }
    branch_aux = _i3_branch_aux_override(args.best_i3)
    variants = {
        "i3_fixed_gate.yaml": deep_update(branch_aux, {
            "model": {"fusion_mode": "tri_modal_fixed_gate"},
            "loss": {"gate_prior_weight": 0.0},
        }),
        "i3_confidence_gate.yaml": deep_update(branch_aux, {
            "model": {"fusion_mode": "tri_modal_confidence_gate"},
            "loss": {"gate_prior_weight": 0.0},
        }),
        "i3_reliability_gate.yaml": deep_update(branch_aux, {
            "model": {"fusion_mode": "tri_modal_reliability_gate"},
            "loss": {"gate_prior_weight": 0.0},
        }),
        "i3_learned_gate_no_prior.yaml": deep_update(tuned_i3, {"loss": {"gate_prior_weight": 0.0}}),
        "i3_learned_gate_with_prior.yaml": tuned_i3,
        "i3_learned_gate_no_alive_mask.yaml": deep_update(tuned_i3, {"model": {"gate": {"apply_alive_mask": False}}}),
    }
    out = []
    for name, override in variants.items():
        stem = Path(name).stem
        leaf = deep_update(_name_payload(f"post_{args.tag}_{stem}"), override)
        out.append((name, fixed, leaf))
    return out


def _full_configs(args: argparse.Namespace) -> list[tuple[str, list[Path], dict[str, Any]]]:
    defaults = _final_defaults(args)
    variants = {
        "full_ours.yaml": {},
        "full_no_aug.yaml": {"robust": {"train_aug": False}},
        "full_no_cross_source_consistency.yaml": {
            "loss": {
                "semantic_reconstruction_weight": 0.0,
                "cross_source_consistency_weight": 0.0,
                "cross_source_min_reliability": 0.0,
                "cross_source_min_consistency": 0.0,
                "semantic_active_only": False,
            },
        },
        "full_no_gate_prior.yaml": {"loss": {"gate_prior_weight": 0.0}},
        "full_no_branch_aux.yaml": {"loss": {"branch_aux_weight": 0.0}},
        "full_graph_zero.yaml": {"data": {"graph_semantic_source": "zero"}},
        "full_graph_full_api.yaml": {"data": {"graph_semantic_source": "full_api"}},
    }
    out = []
    for name, override in variants.items():
        stem = Path(name).stem
        leaf = deep_update(_name_payload(f"post_{args.tag}_{stem}"), override)
        out.append((name, defaults, leaf))
    return out


def _seed_configs(args: argparse.Namespace) -> list[tuple[str, list[Path], dict[str, Any]]]:
    defaults = _final_defaults(args)
    out = []
    for seed in args.seeds:
        name = f"seed_{int(seed)}.yaml"
        out.append((name, defaults, _seed_payload(int(seed), f"post_{args.tag}_seed_{int(seed)}")))
    return out


def _audit_config(path: Path, cfg: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    rel = path.as_posix()
    model = cfg.get("model", {}) or {}
    gate = model.get("gate", {}) or {}
    loss = cfg.get("loss", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    if bool(train_cfg.get("tuning_mode", False)):
        issues.append(f"{rel}: generated post-optuna config must not have tuning_mode=true")
    if not bool(eval_cfg.get("run_test", True)):
        issues.append(f"{rel}: generated final config has run_test=false")
    if bool(gate.get("use_perturbation_evidence", False)):
        issues.append(f"{rel}: generated final config exposes oracle perturbation evidence")
    if path.name.startswith("i3_fixed") and model.get("fusion_mode") != "tri_modal_fixed_gate":
        issues.append(f"{rel}: fixed gate ablation was overwritten")
    if path.name.startswith("i3_confidence") and model.get("fusion_mode") != "tri_modal_confidence_gate":
        issues.append(f"{rel}: confidence gate ablation was overwritten")
    if path.name.startswith("i3_reliability") and model.get("fusion_mode") != "tri_modal_reliability_gate":
        issues.append(f"{rel}: reliability gate ablation was overwritten")
    if "no_gate_prior" in path.stem and float(loss.get("gate_prior_weight", 0.0)) != 0.0:
        issues.append(f"{rel}: no_gate_prior has nonzero gate_prior_weight")
    if "no_cross_source" in path.stem and float(loss.get("cross_source_consistency_weight", 0.0)) != 0.0:
        issues.append(f"{rel}: no_cross_source has nonzero cross_source_consistency_weight")
    if "no_aug" in path.stem and bool(cfg.get("robust", {}).get("train_aug", True)):
        issues.append(f"{rel}: no_aug still has train_aug=true")
    return issues


def _validate_inputs(args: argparse.Namespace) -> None:
    missing = []
    for label, value in (("best-i2", args.best_i2), ("best-i3", args.best_i3), ("best-aug", args.best_aug)):
        if value and not Path(value).exists():
            missing.append(f"--{label} {value}")
    if missing:
        raise FileNotFoundError(
            "Post-Optuna generation requires existing best override files. Missing: "
            + ", ".join(missing)
        )


def generate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_inputs(args)
    out_root = Path(args.out_dir) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    specs = {
        "i1": _i1_configs(args),
        "i2": _i2_configs(args),
        "i3": _i3_configs(args),
        "full": _full_configs(args),
        "seed": _seed_configs(args),
    }
    groups: dict[str, list[str]] = {}
    audit: dict[str, Any] = {"tag": args.tag, "issues": [], "configs": {}}

    for group, items in specs.items():
        for filename, defaults, leaf in items:
            out_file = out_root / group / filename
            payload = _generated_config(out_file, defaults, leaf)
            _write_yaml(out_file, payload)
            groups.setdefault(group, []).append(out_file.relative_to(CONFIG_ROOT).as_posix())
            cfg = load_config([str(out_file)])
            issues = _audit_config(out_file, cfg)
            audit["issues"].extend(issues)
            audit["configs"][out_file.relative_to(CONFIG_ROOT).as_posix()] = {
                "exp_name": cfg.get("train", {}).get("exp_name"),
                "fusion_mode": cfg.get("model", {}).get("fusion_mode"),
                "train_aug": cfg.get("robust", {}).get("train_aug"),
                "cross_source_consistency_weight": cfg.get("loss", {}).get("cross_source_consistency_weight"),
                "gate_prior_weight": cfg.get("loss", {}).get("gate_prior_weight"),
                "branch_aux_weight": cfg.get("loss", {}).get("branch_aux_weight"),
            }

    _write_yaml(out_root / "groups.yaml", {"groups": groups})
    (out_root / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["issues"]:
        raise RuntimeError("Generated post-optuna config audit failed:\n" + "\n".join(audit["issues"]))
    print(json.dumps({"out_dir": str(out_root), "groups": groups}, indent=2, ensure_ascii=False))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final post-Optuna experiment YAMLs with safe override order.")
    parser.add_argument("--tag", default="robust_v3", help="Subdirectory name under config/experiments/tri_modal_robust/post_optuna.")
    parser.add_argument("--best-i2", default="results/optuna/robust_v3/best_i2_override.yaml")
    parser.add_argument("--best-i3", default="results/optuna/robust_v3/best_i3_override.yaml")
    parser.add_argument("--best-aug", default="results/optuna/robust_v3/best_aug_override.yaml")
    parser.add_argument("--out-dir", default=str(CONFIG_ROOT / "post_optuna"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407])
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
