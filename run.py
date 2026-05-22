import argparse
import os
import subprocess
from pathlib import Path

from scripts.make_ablation_configs import DEFAULT_OUT_DIR, build_configs

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
load_dotenv()

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
BASE_CONFIG = os.getenv("BASE_CONFIG", "config/base.yaml")
EXPERIMENT_CONFIG_DIR = Path(os.getenv("EXPERIMENT_CONFIG_DIR", str(DEFAULT_OUT_DIR)))

ALIASES = {
    "baseline": "baselines",
    "base": "baselines",
    "cmp": "baselines",
    "adaptation": "i1",
    "dbta": "i1",
    "memory": "replay",
    "align": "i2",
    "alignment": "i2",
    "gate": "i3",
    "fusion": "i3",
    "sweep": "ratio",
    "ratios": "ratio",
    "final": "final",
    "chain": "main",
}


def load_groups() -> dict[str, list[str]]:
    _, groups = build_configs()
    return {
        name: [str(EXPERIMENT_CONFIG_DIR / rel_path).replace("\\", "/") for rel_path in rel_paths]
        for name, rel_paths in groups.items()
    }


def resolve_overrides(target: str, groups: dict[str, list[str]]) -> list[str]:
    target = ALIASES.get(target, target)
    if target.endswith((".yaml", ".yml")):
        return [target]

    if target not in groups:
        known = ", ".join(sorted(groups))
        raise ValueError(f"Unknown experiment group '{target}'. Known groups: {known}")
    return list(groups[target])


def require_generated(overrides: list[str]) -> None:
    missing = [path for path in overrides if not Path(path).exists()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Missing generated experiment YAMLs: {preview}. "
            "Run `python scripts/make_ablation_configs.py` first."
        )


def run_override(override: str) -> None:
    print(f"==> Running {override}", flush=True)
    subprocess.run(
        [
            PYTHON_BIN,
            "-m",
            "fusion.train",
            f"--base={BASE_CONFIG}",
            f"--override={override}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Experiment group, or a single YAML path.",
    )
    parser.add_argument("--list", action="store_true", help="List available experiment groups and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected configs without launching training.")
    args = parser.parse_args()

    groups = load_groups()
    if args.list:
        for name in sorted(groups):
            print(f"{name}: {len(groups[name])}")
        return

    overrides = resolve_overrides(args.target, groups)
    require_generated(overrides)
    print(f"Base config: {BASE_CONFIG}", flush=True)
    print(f"Target: {args.target} ({len(overrides)} configs)", flush=True)
    if args.dry_run:
        for override in overrides:
            print(override)
        return
    for override in overrides:
        run_override(override)


if __name__ == "__main__":
    main()
