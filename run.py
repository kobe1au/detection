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
    "compact": "loss/compact_kl.yaml",
    "compact_kl": "loss/compact_kl.yaml",
    "plain_kl": "loss/plain_kl.yaml",
    "ce_only": "loss/ce_only.yaml",
}

GROUPS = {
    "main": [
        "main/full_compact_kl_seed42.yaml",
    ],
    "loss": [
        "loss/ce_only.yaml",
        "loss/plain_kl.yaml",
        "loss/compact_kl.yaml",
        "loss/compact_kl_w002.yaml",
        "loss/compact_kl_w010.yaml",
    ],
    "r1_graph": [
        "r1_graph/code_only.yaml",
        "r1_graph/manifest_only.yaml",
        "r1_graph/no_edge_source.yaml",
        "r1_graph/no_node_quality.yaml",
        "r1_graph/no_edge_quality.yaml",
        "r1_graph/no_alignment.yaml",
        "r1_graph/no_risk_nodes.yaml",
    ],
    "r3_fusion": [
        "r3_fusion/no_reliability_bias.yaml",
        "r3_fusion/no_conflict_bias.yaml",
        "r3_fusion/mean_fusion.yaml",
    ],
    "full_seeds": [
        "main/full_compact_kl_seed42.yaml",
        "main/full_compact_kl_seed43.yaml",
        "main/full_compact_kl_seed44.yaml",
    ],
    "all": [
        "main/full_compact_kl_seed42.yaml",
        "main/full_compact_kl_seed43.yaml",
        "main/full_compact_kl_seed44.yaml",
        "loss/ce_only.yaml",
        "loss/plain_kl.yaml",
        "loss/compact_kl.yaml",
        "r1_graph/code_only.yaml",
        "r1_graph/manifest_only.yaml",
        "r1_graph/no_edge_source.yaml",
        "r1_graph/no_node_quality.yaml",
        "r1_graph/no_edge_quality.yaml",
        "r1_graph/no_alignment.yaml",
        "r1_graph/no_risk_nodes.yaml",
        "r3_fusion/no_reliability_bias.yaml",
        "r3_fusion/no_conflict_bias.yaml",
        "r3_fusion/mean_fusion.yaml",
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
