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
    "ours": "main/full_compact_kl_seed42.yaml",
    "full": "main/full_compact_kl_seed42.yaml",
    "final": "main/full_compact_kl_seed42.yaml",
    "compact": "main/full_compact_kl_seed42.yaml",
    "compact_kl": "main/full_compact_kl_seed42.yaml",
    "plain_kl": "stage3/full_plain_kl.yaml",
    "ce_only": "stage2/full_fusion_ce.yaml",
    "aeg_only": "stage1/aeg_only_ce.yaml",
    "fusion": "stage2/full_fusion_ce.yaml",
}

STAGE1 = [
    "stage1/api_only_ce.yaml",
    "stage1/graph_only_ce.yaml",
    "stage1/manifest_only_ce.yaml",
    "stage1/aeg_only_ce.yaml",
    "stage1/aeg_no_source_metadata_ce.yaml",
    "stage1/aeg_no_relation_types_ce.yaml",
    "stage1/aeg_no_quality_ce.yaml",
    "stage1/aeg_no_alignment_ce.yaml",
    "stage1/aeg_no_risk_ce.yaml",
]
STAGE2 = [
    "stage2/latent_content_ce.yaml",
    "stage2/latent_reliability_ce.yaml",
    "stage2/latent_conflict_ce.yaml",
    "stage2/latent_source_bias_ce.yaml",
    "stage2/latent_no_reliability_ce.yaml",
    "stage2/latent_no_conflict_ce.yaml",
    "stage2/latent_no_source_bias_ce.yaml",
    "stage2/full_fusion_ce.yaml",
]
STAGE3 = [
    "stage2/full_fusion_ce.yaml",
    "stage3/full_plain_kl.yaml",
    "main/full_compact_kl_seed42.yaml",
    "stage3/full_compact_kl_w002.yaml",
    "stage3/full_compact_kl_w010.yaml",
]
FULL_SEEDS = [
    "main/full_compact_kl_seed42.yaml",
    "main/full_compact_kl_seed43.yaml",
    "main/full_compact_kl_seed44.yaml",
]

GROUPS = {
    "stage1": STAGE1,
    "stage2": STAGE2,
    "stage3": STAGE3,
    "main": ["main/full_compact_kl_seed42.yaml"],
    "full_seeds": FULL_SEEDS,
    # Backward-compatible group names now follow the staged experiment plan.
    "r1_graph": STAGE1,
    "r3_fusion": STAGE2,
    "loss": STAGE3,
    "all": [*STAGE1, *STAGE2, *STAGE3, *FULL_SEEDS],
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
