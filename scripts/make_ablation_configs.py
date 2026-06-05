from __future__ import annotations

from pathlib import Path


DEFAULT_OUT_DIR = Path("config/experiments/tri_modal_robust")


def build_configs() -> tuple[list[str], dict[str, list[str]]]:
    configs = [
        str(p)
        for p in sorted(DEFAULT_OUT_DIR.rglob("*.yaml"))
        if p.name not in {"base_tri_modal_robust.yaml", "optuna_base.yaml"}
        and "tune" not in p.relative_to(DEFAULT_OUT_DIR).parts
        and "oracle" not in p.stem
    ]
    groups = {
        "all": configs,
        "final": [str(DEFAULT_OUT_DIR / "full" / "ours.yaml")],
        "baselines": [
            str(DEFAULT_OUT_DIR / "i1" / "api_only.yaml"),
            str(DEFAULT_OUT_DIR / "i1" / "graph_only.yaml"),
            str(DEFAULT_OUT_DIR / "i1" / "manifest_only.yaml"),
            str(DEFAULT_OUT_DIR / "i1" / "api_graph_concat.yaml"),
            str(DEFAULT_OUT_DIR / "i1" / "tri_modal_concat.yaml"),
        ],
        "i1": [str(p) for p in sorted((DEFAULT_OUT_DIR / "i1").glob("*.yaml"))],
        "i2": [str(p) for p in sorted((DEFAULT_OUT_DIR / "i2").glob("*.yaml"))],
        "i3": [str(p) for p in sorted((DEFAULT_OUT_DIR / "i3").glob("*.yaml"))],
        "full": [str(p) for p in sorted((DEFAULT_OUT_DIR / "full").glob("*.yaml"))],
        "seed": [str(p) for p in sorted((DEFAULT_OUT_DIR / "seed").glob("*.yaml"))],
        "gates": [
            str(DEFAULT_OUT_DIR / "i3" / "fixed_gate.yaml"),
            str(DEFAULT_OUT_DIR / "i3" / "confidence_gate.yaml"),
            str(DEFAULT_OUT_DIR / "i3" / "reliability_gate.yaml"),
            str(DEFAULT_OUT_DIR / "i3" / "learned_gate_no_alive_mask.yaml"),
            str(DEFAULT_OUT_DIR / "i3" / "learned_gate_no_prior.yaml"),
            str(DEFAULT_OUT_DIR / "i3" / "learned_gate_with_prior.yaml"),
        ],
    }
    return configs, groups


def main() -> None:
    configs, groups = build_configs()
    print(f"Robust configs: {len(configs)}")
    for name, paths in groups.items():
        print(f"{name}: {len(paths)}")


if __name__ == "__main__":
    main()
