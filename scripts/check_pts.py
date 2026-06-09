from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.io_utils import load_aeg_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AEG PT node feature dimensions.")
    parser.add_argument("--pt-dir", type=Path, default=Path("D:/pts_aeg/train"))
    parser.add_argument("--sample", type=int, default=0, help="0 means scan all PT files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pt_dir = args.pt_dir
    if not pt_dir.exists():
        raise FileNotFoundError(f"PT directory not found: {pt_dir}")

    files = sorted(pt_dir.rglob("*.pt"))
    if args.sample > 0:
        files = files[: args.sample]
    if not files:
        raise RuntimeError(f"No PT files found under {pt_dir}")

    dims: Counter[int] = Counter()
    failures: list[tuple[str, str]] = []
    for path in files:
        try:
            payload = load_aeg_payload(path, validate=True)
            dims[int(payload["node_x"].size(1))] += 1
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))

    print("Node_x dimensions:")
    for dim, count in sorted(dims.items()):
        print(f"  {dim}: {count} files")
    if failures:
        print(f"Failed files: {len(failures)}")
        for path, reason in failures[:10]:
            print(f"  {path}: {reason}")
    if len(dims) > 1:
        print("Inconsistent dimensions found.")
        return 2
    if failures:
        return 1
    print("All checked PT files have consistent dimensions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
