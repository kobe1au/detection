#!/usr/bin/env python3
"""Generate non-redundant I1 temporal prototype grid YAML files."""

from __future__ import annotations

import csv
from pathlib import Path


OUT_DIR = Path("config/train_2026/i1_grid")
MANIFEST = OUT_DIR / "manifest.csv"

CURRENT_WEIGHTS = [0.01, 0.03, 0.05]
FUTURE_WEIGHTS = [0.0, 0.005, 0.01]
VELOCITY_SCALES = [0.25, 0.5, 1.0]
CLUSTERS = [2, 4]


def weight_code(value: float, scale: int = 1000) -> str:
    return f"{int(round(value * scale)):04d}"


def velocity_code(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []

    anchor_name = "i1g_000_erm_concat"
    write_yaml(
        OUT_DIR / f"{anchor_name}.yaml",
        "\n".join([
            "train:",
            f"  exp_name: {anchor_name}",
            "",
            "model:",
            "  fusion_mode: concat",
            "",
            "loss:",
            "  temporal_proto_current_weight: 0.0",
            "  temporal_proto_future_weight: 0.0",
            "",
        ]),
    )
    rows.append({
        "yaml": f"{anchor_name}.yaml",
        "exp_name": anchor_name,
        "fusion_mode": "concat",
        "current_weight": 0.0,
        "future_weight": 0.0,
        "velocity_scale": 0.5,
        "prototype_clusters": 4,
        "note": "ERM anchor",
    })

    for current in CURRENT_WEIGHTS:
        for future in FUTURE_WEIGHTS:
            velocities = [0.5] if future == 0.0 else VELOCITY_SCALES
            for velocity in velocities:
                for clusters in CLUSTERS:
                    exp_name = (
                        "i1g_"
                        f"c{weight_code(current)}_"
                        f"f{weight_code(future)}_"
                        f"v{velocity_code(velocity)}_"
                        f"k{clusters}"
                    )
                    filename = f"{exp_name}.yaml"
                    write_yaml(
                        OUT_DIR / filename,
                        "\n".join([
                            "train:",
                            f"  exp_name: {exp_name}",
                            "",
                            "model:",
                            "  fusion_mode: concat",
                            "  temporal:",
                            f"    prototype_clusters: {clusters}",
                            "",
                            "loss:",
                            f"  temporal_proto_current_weight: {current}",
                            f"  temporal_proto_future_weight: {future}",
                            f"  temporal_proto_velocity_scale: {velocity}",
                            "",
                        ]),
                    )
                    rows.append({
                        "yaml": filename,
                        "exp_name": exp_name,
                        "fusion_mode": "concat",
                        "current_weight": current,
                        "future_weight": future,
                        "velocity_scale": velocity,
                        "prototype_clusters": clusters,
                        "note": "",
                    })

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "yaml",
                "exp_name",
                "fusion_mode",
                "current_weight",
                "future_weight",
                "velocity_scale",
                "prototype_clusters",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} configs under {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
