import os
import subprocess

from dotenv import load_dotenv


os.chdir(os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
BASE_CONFIG = os.getenv("BASE_CONFIG", "./config/base.yaml")


# Main 2026 experiment chain mapped to actual config/exp_*.yaml files.
EXPERIMENTS = [
    "./config/exp_erm_concat_baseline.yaml",
    "./config/exp_ours_align_gate.yaml",
    "./config/exp_ours_time_gate.yaml",
    "./config/exp_ours_time_reliability.yaml",
    "./config/exp_ours_hierarchical_alignment.yaml",
    "./config/exp_ours_continual_2023adapt.yaml",
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
