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
    "api": "T0_api_only.yaml",
    "graph": "T1_graph_only.yaml",
    "manifest": "T2_manifest_only.yaml",
    "api_graph": "T3_api_graph_concat.yaml",
    "concat": "T4_api_graph_manifest_concat.yaml",
    "fixed": "T5_tri_modal_fixed_gate.yaml",
    "reliability": "T6_tri_modal_reliability_gate.yaml",
    "full": "T7_tri_modal_full_soft_consistency.yaml",
    "ours": "T7_tri_modal_full_soft_consistency.yaml",
    "final": "T7_tri_modal_full_soft_consistency.yaml",
}


def available_configs() -> dict[str, Path]:
    return {path.stem: path for path in sorted(CONFIG_DIR.glob("T*.yaml"))}


def resolve_targets(target: str) -> list[Path]:
    if target == "all":
        return list(available_configs().values())
    if target.endswith((".yaml", ".yml")):
        return [Path(target)]
    target = ALIASES.get(target, target)
    path = CONFIG_DIR / target
    if path.exists():
        return [path]
    configs = available_configs()
    if target in configs:
        return [configs[target]]
    known = ", ".join(["all", *sorted(ALIASES), *sorted(configs)])
    raise ValueError(f"Unknown robust experiment target '{target}'. Known: {known}")


def run_config(config_path: Path) -> None:
    print(f"==> Running {config_path}", flush=True)
    subprocess.run(
        [PYTHON_BIN, "-m", "fusion.robust.train", "--config", str(config_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust tri-modal fusion experiments.")
    parser.add_argument("target", nargs="?", default="final", help="all, final, api, graph, manifest, concat, fixed, reliability, or YAML path")
    parser.add_argument("--list", action="store_true", help="List robust experiment configs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected configs without launching training.")
    args = parser.parse_args()

    if args.list:
        for name, path in available_configs().items():
            print(f"{name}: {path}")
        return

    targets = resolve_targets(args.target)
    if args.dry_run:
        for path in targets:
            print(path)
        return
    for path in targets:
        run_config(path)


if __name__ == "__main__":
    main()
