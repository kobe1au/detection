from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
CONFIG_DIR = Path("config/experiments/tri_modal_robust")

ALIASES = {
    "api": "i1/api_only.yaml",
    "graph": "i1/graph_only.yaml",
    "manifest": "i1/manifest_only.yaml",
    "api_graph": "i1/api_graph_concat.yaml",
    "concat": "i1/tri_modal_concat.yaml",
    "reliability": "i1/reliability_gate.yaml",
    "fixed": "i3/fixed_gate.yaml",
    "confidence": "i3/confidence_gate.yaml",
    "learned_no_prior": "i3/learned_gate_no_prior.yaml",
    "learned_with_prior": "i3/learned_gate_with_prior.yaml",
    "learned_no_alive": "i3/learned_gate_no_alive_mask.yaml",
    "consistency_evidence": "i2/consistency_evidence_only.yaml",
    "conflict_evidence": "i2/conflict_evidence_only.yaml",
    "full": "full/ours.yaml",
    "ours": "full/ours.yaml",
    "final": "full/ours.yaml",
}

GROUPS = {
    "main": [
        "full/ours.yaml",
        "i1/api_only.yaml",
        "i1/graph_only.yaml",
        "i1/manifest_only.yaml",
        "i1/api_graph_concat.yaml",
        "i1/tri_modal_concat.yaml",
        "i1/reliability_gate.yaml",
        "i2/no_consistency.yaml",
        "i2/consistency_evidence_only.yaml",
        "i2/conflict_evidence_only.yaml",
        "i2/evidence_only.yaml",
        "i2/loss_only.yaml",
        "i2/semantic_reconstruction_only.yaml",
        "i2/evidence_plus_loss.yaml",
        "i3/fixed_gate.yaml",
        "i3/confidence_gate.yaml",
        "i3/reliability_gate.yaml",
        "i3/learned_gate_no_alive_mask.yaml",
        "i3/learned_gate_no_prior.yaml",
        "i3/learned_gate_with_prior.yaml",
    ],
    "i1": [
        "i1/api_only.yaml",
        "i1/graph_only.yaml",
        "i1/manifest_only.yaml",
        "i1/api_graph_concat.yaml",
        "i1/tri_modal_concat.yaml",
        "i1/reliability_gate.yaml",
    ],
    "i2": [
        "i2/no_consistency.yaml",
        "i2/consistency_evidence_only.yaml",
        "i2/conflict_evidence_only.yaml",
        "i2/evidence_only.yaml",
        "i2/loss_only.yaml",
        "i2/semantic_reconstruction_only.yaml",
        "i2/evidence_plus_loss.yaml",
    ],
    "i3": [
        "i3/fixed_gate.yaml",
        "i3/confidence_gate.yaml",
        "i3/reliability_gate.yaml",
        "i3/learned_gate_no_alive_mask.yaml",
        "i3/learned_gate_no_prior.yaml",
        "i3/learned_gate_with_prior.yaml",
    ],
    "full_ablation": [
        "full/ours.yaml",
        "full/ours_no_aug.yaml",
        "full/ours_no_cross_source_consistency.yaml",
        "full/ours_no_gate_prior.yaml",
        "full/ours_no_branch_aux.yaml",
        "full/ours_graph_zero.yaml",
        "full/ours_graph_full_api.yaml",
        "full/ours_oracle_perturbation_evidence.yaml",
    ],
    "seed": [
        "seed/seed_42.yaml",
        "seed/seed_2024.yaml",
        "seed/seed_3407.yaml",
    ],
}


def available_configs() -> dict[str, Path]:
    configs: dict[str, Path] = {}
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        if path.name in {"base_tri_modal_robust.yaml", "optuna_base.yaml"}:
            continue
        if "tune" in path.relative_to(CONFIG_DIR).parts:
            continue
        if "oracle" in path.stem:
            continue
        if path.name.upper() == "README.MD":
            continue
        rel = path.relative_to(CONFIG_DIR).with_suffix("")
        configs[str(rel).replace("\\", "/")] = path
        configs.setdefault(path.stem, path)
    return configs


def resolve_targets(target: str) -> list[Path]:
    if target == "tune":
        raise ValueError(
            "Manual tune configs are excluded because they can mix tuning and final evaluation. "
            "Use scripts/tune_robust_optuna.py."
        )
    if target in GROUPS:
        return [CONFIG_DIR / item for item in GROUPS[target]]
    if target == "all":
        seen: set[Path] = set()
        out: list[Path] = []
        for path in available_configs().values():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
        return out
    if target.endswith((".yaml", ".yml")):
        return [Path(target)]
    target = ALIASES.get(target, target)
    path = CONFIG_DIR / target
    if path.exists():
        return [path]
    configs = available_configs()
    if target in configs:
        return [configs[target]]
    known = ", ".join(["all", *sorted(GROUPS), *sorted(ALIASES), *sorted(configs)])
    raise ValueError(f"Unknown robust experiment target '{target}'. Known: {known}")


def resolve_target_specs(targets: list[str]) -> list[Path]:
    parts: list[str] = []
    for target in targets:
        for part in str(target).split(","):
            part = part.strip()
            if part:
                parts.append(part)
    if not parts:
        parts = ["final"]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for part in parts:
        for path in resolve_targets(part):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
    return resolved


def run_config(config_path: Path) -> None:
    print(f"==> Running {config_path}", flush=True)
    subprocess.run(
        [PYTHON_BIN, "-m", "fusion.train", "--config", str(config_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust tri-modal fusion experiments.")
    parser.add_argument(
        "target",
        nargs="*",
        default=["final"],
        help=(
            "Targets to run. Supports groups/aliases/YAML paths and comma lists, "
            "e.g. 'i2,i3' or 'i2 i3'."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List robust experiment configs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected configs without launching training.")
    args = parser.parse_args()

    if args.list:
        seen: set[Path] = set()
        for path in sorted(available_configs().values()):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            print(path.relative_to(CONFIG_DIR).with_suffix("").as_posix() + f": {path}")
        return

    targets = resolve_target_specs(args.target)
    if args.dry_run:
        for path in targets:
            print(path)
        return
    for path in targets:
        run_config(path)


if __name__ == "__main__":
    main()
