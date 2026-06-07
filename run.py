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
    "i1_full": "i1/typed_source_quality.yaml",
    "i2_full": "i2/multiview_contrast.yaml",
    "i3_full": "full/ours.yaml",
}

GROUPS = {
    "main": ["full/ours.yaml"],
    "i1": [
        "i1/typed_source_quality.yaml",
        "i1/homogeneous_graph.yaml",
        "i1/no_relation_types.yaml",
        "i1/no_source_encoding.yaml",
        "i1/no_quality_encoding.yaml",
    ],
    "i2": [
        "i2/multiview_contrast.yaml",
        "i2/no_clean_degraded_contrast.yaml",
        "i2/no_source_degraded_contrast.yaml",
        "i2/no_cross_source_contrast.yaml",
        "i2/no_contrast.yaml",
    ],
    "i3": [
        "full/ours.yaml",
        "i3/mean_pool_fusion.yaml",
        "i3/no_source_bias.yaml",
        "i3/no_reliability_bias.yaml",
        "i3/no_conflict_bias.yaml",
        "i3/no_counterfactual.yaml",
    ],
    "full_seeds": [
        "full/ours.yaml",
        "full/ours_seed52.yaml",
        "full/ours_seed62.yaml",
    ],
    "all": [
        "i1/typed_source_quality.yaml",
        "i1/homogeneous_graph.yaml",
        "i1/no_relation_types.yaml",
        "i1/no_source_encoding.yaml",
        "i1/no_quality_encoding.yaml",
        "i2/multiview_contrast.yaml",
        "i2/no_clean_degraded_contrast.yaml",
        "i2/no_source_degraded_contrast.yaml",
        "i2/no_cross_source_contrast.yaml",
        "i2/no_contrast.yaml",
        "full/ours.yaml",
        "i3/mean_pool_fusion.yaml",
        "i3/no_source_bias.yaml",
        "i3/no_reliability_bias.yaml",
        "i3/no_conflict_bias.yaml",
        "i3/no_counterfactual.yaml",
        "full/ours_seed52.yaml",
        "full/ours_seed62.yaml",
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
