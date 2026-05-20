import os
import subprocess

from dotenv import load_dotenv


os.chdir(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
BASE_CONFIG = os.getenv("BASE_CONFIG", "./config/base.yaml")


# Clean 2026 experiment order: all experiments are trained from scratch.
EXPERIMENTS = [
    # "./config/train_2026/baselines/00_api_only.yaml",
    # "./config/train_2026/baselines/01_graph_only.yaml",
    # "./config/train_2026/baselines/02_concat_erm.yaml",
    # "./config/train_2026/baselines/03_cross_attention.yaml",
    "./config/train_2026/i1_temporal/00_erm_concat.yaml",
    "./config/train_2026/i1_temporal/01_proto_current.yaml",
    "./config/train_2026/i1_temporal/02_proto_current_010.yaml",
    "./config/train_2026/i1_temporal/03_proto_future_weak.yaml",
    "./config/train_2026/i1_temporal/03b_proto_future_strong.yaml",
    "./config/train_2026/i1_temporal/04_proto_current_risk.yaml",
    "./config/train_2026/i1_temporal/05_proto_current_future_risk.yaml",
    # "./config/train_2026/i1_temporal/06_ours_fixed_scaffold_erm.yaml",
    # "./config/train_2026/i1_temporal/07_ours_fixed_current_risk_scaffold.yaml",
    # "./config/train_2026/i2_alignment/00_temporal_concat.yaml",
    # "./config/train_2026/i3_fusion/00_fixed_no_gate.yaml",
    # "./config/train_2026/final/ours_2026_no_future.yaml",
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
