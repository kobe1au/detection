import os
import subprocess

from dotenv import load_dotenv


os.chdir(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
BASE_CONFIG = os.getenv("BASE_CONFIG", "./config/base.yaml")


# Clean 2026 experiment order: all experiments are trained from scratch.
EXPERIMENTS = [
    "./config/train_2026/baselines/00_api_only.yaml",
    "./config/train_2026/baselines/01_graph_only.yaml",
    "./config/train_2026/baselines/02_concat_erm.yaml",
    "./config/train_2026/baselines/03_cross_attention.yaml",
    # "./config/train_2026/baselines/04_temporal_supcon.yaml",
    # "./config/train_2026/baselines/05_groupdro.yaml",
    # "./config/train_2026/baselines/06_vrex.yaml",
    # "./config/train_2026/baselines/07_irm.yaml",
    # "./config/train_2026/baselines/08_coral.yaml",
    # "./config/train_2026/i1_temporal/00_erm_concat.yaml",
    # "./config/train_2026/i1_temporal/01_proto_current.yaml",
    # "./config/train_2026/i1_temporal/02_proto_current_010.yaml",
    # "./config/train_2026/i1_temporal/03_proto_future_weak.yaml",
    # "./config/train_2026/i1_temporal/04_ours_fixed_scaffold_erm.yaml",
    # "./config/train_2026/i1_temporal/05_ours_fixed_proto_trajectory.yaml",
    # "./config/train_2026/i2_alignment/00_temporal_concat.yaml",
    # "./config/train_2026/i2_alignment/01_cross_attention.yaml",
    # "./config/train_2026/i2_alignment/02_ours_fixed_no_alignment.yaml",
    # "./config/train_2026/i2_alignment/03_semantic_alignment.yaml",
    # "./config/train_2026/i2_alignment/04_method_aware_context.yaml",
    # "./config/train_2026/i2_alignment/05_temporal_guided_alignment.yaml",
    # "./config/train_2026/i3_fusion/00_fixed_no_gate.yaml",
    # "./config/train_2026/i3_fusion/01_learned_gate_no_reliability.yaml",
    # "./config/train_2026/i3_fusion/02_quality_gate.yaml",
    # "./config/train_2026/i3_fusion/03_drift_gate.yaml",
    # "./config/train_2026/i3_fusion/04_quality_drift_gate.yaml",
    # "./config/train_2026/final/ours_2026_no_future.yaml",
    # "./config/train_2026/final/ours_2026_no_semantic_align.yaml",
    # "./config/train_2026/final/ours_2026_no_reliability_gate.yaml",
    # "./config/train_2026/final/ours_2026.yaml",
]


for override in EXPERIMENTS:
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
