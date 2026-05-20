import os
import subprocess

from dotenv import load_dotenv


os.chdir(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
BASE_CONFIG = os.getenv("BASE_CONFIG", "./config/base.yaml")


# Main 2026 continual-adaptation chain.
EXPERIMENTS = [
    "./config/train_2026/baselines/00_api_only.yaml",
    "./config/train_2026/baselines/01_graph_only.yaml",
    "./config/train_2026/baselines/02_concat_erm.yaml",
    "./config/train_2026/baselines/03_cross_attention.yaml",
    "./config/train_2026/continual/00_zero_adapt_concat.yaml",
    "./config/train_2026/continual/01_i1_adapt_010.yaml",
    "./config/train_2026/continual/02_i1_i2_adapt_010.yaml",
    "./config/train_2026/continual/03_i1_i2_i3_adapt_010.yaml",
    "./config/train_2026/continual/04_i1_i2_i3_static_replay_adapt_010.yaml",
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
