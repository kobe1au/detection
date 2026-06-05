from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
CONFIG_DIR = Path("config/experiments/aeg_robust")

ALIASES = {
    "ours": "full/ours.yaml",
    "full": "full/ours.yaml",
    "final": "full/ours.yaml",
    "no_aug": "ablation/no_aug.yaml",
    "no_robust_losses": "ablation/no_robust_losses.yaml",
    "no_clean_degraded_contrast": "ablation/no_clean_degraded_contrast.yaml",
    "no_cross_source_contrast": "ablation/no_cross_source_contrast.yaml",
    "no_counterfactual": "ablation/no_counterfactual.yaml",
    "no_reliability_bias": "ablation/no_reliability_bias.yaml",
    "no_conflict_bias": "ablation/no_conflict_bias.yaml",
}

GROUPS = {
    "main": ["full/ours.yaml"],
    "ablation": [
        "ablation/no_clean_degraded_contrast.yaml",
        "ablation/no_cross_source_contrast.yaml",
        "ablation/no_counterfactual.yaml",
        "ablation/no_reliability_bias.yaml",
        "ablation/no_conflict_bias.yaml",
        "ablation/no_aug.yaml",
        "ablation/no_robust_losses.yaml",
    ],
    "all": [
        "full/ours.yaml",
        "ablation/no_clean_degraded_contrast.yaml",
        "ablation/no_cross_source_contrast.yaml",
        "ablation/no_counterfactual.yaml",
        "ablation/no_reliability_bias.yaml",
        "ablation/no_conflict_bias.yaml",
        "ablation/no_aug.yaml",
        "ablation/no_robust_losses.yaml",
    ],
}


def resolve_targets(target: str) -> list[Path]:
    target = ALIASES.get(target, target)
    if target in GROUPS:
        return [CONFIG_DIR / item for item in GROUPS[target]]
    if target.endswith((".yaml", ".yml")):
        path = Path(target)
        return [path if path.exists() else CONFIG_DIR / target]
    path = CONFIG_DIR / target
    if path.exists():
        return [path]
    known = ", ".join(sorted([*ALIASES.keys(), *GROUPS.keys()]))
    raise ValueError(f"Unknown AEG experiment target '{target}'. Known: {known}")


def resolve_target_specs(targets: list[str]) -> list[Path]:
    parts: list[str] = []
    for target in targets:
        parts.extend([p.strip() for p in str(target).split(",") if p.strip()])
    if not parts:
        parts = ["final"]
    out: list[Path] = []
    seen: set[Path] = set()
    for part in parts:
        for path in resolve_targets(part):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AEG robust malware detection experiments.")
    parser.add_argument("target", nargs="*", default=["final"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = resolve_target_specs(args.target)
    if args.dry_run:
        for cfg in configs:
            print(cfg)
        return
    for cfg in configs:
        print(f"==> Running {cfg}", flush=True)
        subprocess.run([PYTHON_BIN, "-m", "fusion.train", "--config", str(cfg)], check=True)


if __name__ == "__main__":
    main()
