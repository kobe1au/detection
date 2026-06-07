#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.dataset import AEGDataset  # noqa: E402
from fusion.constants import AEG_PAYLOAD_CONTRACT_FINGERPRINT  # noqa: E402
from fusion.model import build_model  # noqa: E402
from fusion.payload_contract import validate_aeg_payload  # noqa: E402
from fusion.train import _device, _loader, _write_rows, evaluate  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _first_pt(path: Path) -> Path:
    try:
        return next(path.glob("*.pt"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No PT files found under {path}") from exc


def _expected_training_contract(ckpt: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, int]:
    fingerprint = str(ckpt.get("aeg_build_fingerprint") or "")
    node_input_dim = int(ckpt.get("node_input_dim") or 0)
    if fingerprint and node_input_dim > 0:
        return fingerprint, node_input_dim
    train_data = ((cfg.get("data", {}) or {}).get("train", {}) or {})
    pt_dir = _resolve(train_data.get("pt_dir", ""))
    payload = torch.load(_first_pt(pt_dir), map_location="cpu")
    validate_aeg_payload(payload)
    return str(payload["aeg_build_fingerprint"]), int(payload["node_x"].size(1))


def _validate_scenario(dataset: AEGDataset, build_fingerprint: str, node_input_dim: int) -> None:
    for path, _label in dataset.samples:
        payload = torch.load(path, map_location="cpu")
        validate_aeg_payload(
            payload,
            expected_build_fingerprint=build_fingerprint,
            expected_node_feature_dim=node_input_dim,
        )


def run(checkpoint: Path, scenario_config: Path, output_dir: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = dict(ckpt.get("cfg") or {})
    if not cfg:
        raise ValueError("Checkpoint does not contain its training config")
    checkpoint_contract = str(ckpt.get("aeg_payload_contract_fingerprint") or "")
    if checkpoint_contract and checkpoint_contract != AEG_PAYLOAD_CONTRACT_FINGERPRINT:
        raise ValueError("Checkpoint AEG payload contract does not match the current code")
    scenarios = (_load_yaml(scenario_config).get("scenarios") or {})
    if not scenarios:
        raise ValueError("Evaluation config requires a non-empty scenarios mapping")

    build_fingerprint, node_input_dim = _expected_training_contract(ckpt, cfg)
    device = _device(cfg)
    model = build_model(cfg, node_input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for name, item in scenarios.items():
        pt_dir = _resolve(item["pt_dir"])
        csv_path = _resolve(item["csv"])
        dataset = AEGDataset(
            pt_dir,
            csv_path,
            split=str(name),
            strict_integrity=True,
            validate_payload_on_load=False,
        )
        _validate_scenario(dataset, build_fingerprint, node_input_dim)
        metrics, rows = evaluate(
            model,
            _loader(cfg, dataset, train=False),
            device,
            split_name=f"external_{name}",
            dump_rows=True,
        )
        summary[str(name)] = metrics
        _write_rows(output_dir / f"diagnostics_test_external_{name}.csv", rows)
    with (output_dir / "summary_external.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AEG checkpoint on real obfuscation/failure scenarios.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.checkpoint, args.config, args.output_dir)
