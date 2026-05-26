from __future__ import annotations

from pathlib import Path


DEFAULT_OUT_DIR = Path("config/experiments/tri_modal_robust")


def build_configs() -> tuple[list[str], dict[str, list[str]]]:
    configs = [str(p) for p in sorted(DEFAULT_OUT_DIR.glob("T*.yaml"))]
    groups = {
        "all": configs,
        "final": [str(DEFAULT_OUT_DIR / "T7_tri_modal_full_soft_consistency.yaml")],
        "baselines": [
            str(DEFAULT_OUT_DIR / "T0_api_only.yaml"),
            str(DEFAULT_OUT_DIR / "T1_graph_only.yaml"),
            str(DEFAULT_OUT_DIR / "T2_manifest_only.yaml"),
            str(DEFAULT_OUT_DIR / "T3_api_graph_concat.yaml"),
            str(DEFAULT_OUT_DIR / "T4_api_graph_manifest_concat.yaml"),
        ],
        "gates": [
            str(DEFAULT_OUT_DIR / "T5_tri_modal_fixed_gate.yaml"),
            str(DEFAULT_OUT_DIR / "T6_tri_modal_reliability_gate.yaml"),
            str(DEFAULT_OUT_DIR / "T7_tri_modal_full_soft_consistency.yaml"),
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
