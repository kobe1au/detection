#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.dataset import AEGDataset  # noqa: E402
from fusion.constants import AEG_PAYLOAD_CONTRACT_FINGERPRINT  # noqa: E402
from fusion.io_utils import load_aeg_payload, load_checkpoint  # noqa: E402
from fusion.model import build_model  # noqa: E402
from fusion.train import _device, _loader, _write_rows, evaluate  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_external_metadata(csv_path: Path) -> dict[str, dict[str, str]]:
    """Read optional external-evaluation metadata keyed by obfuscated sample id.

    The Obfuscapk label builder writes:
      id, sha256, label, year, split, source_id, apk_name

    Here:
      id / sha256   = obfuscated APK hash / PT sid
      source_id     = original clean APK hash
    """
    mapping: dict[str, dict[str, str]] = {}
    if not csv_path.exists():
        return mapping

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("id") or row.get("sha256") or "").strip().lower()
            if not sample_id:
                continue

            source_id = str(row.get("source_id") or "").strip().lower()
            apk_name = str(row.get("apk_name") or "").strip()
            scenario = str(row.get("split") or "").strip()

            mapping[sample_id] = {
                "source_id": source_id,
                "apk_name": apk_name,
                "scenario": scenario,
            }

            sha256 = str(row.get("sha256") or "").strip().lower()
            if sha256 and sha256 not in mapping:
                mapping[sha256] = mapping[sample_id]

    return mapping


def _attach_external_metadata(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, str]],
    scenario_name: str,
) -> list[dict[str, Any]]:
    """Attach source_id and scenario metadata to diagnostics rows."""
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        sid = str(item.get("sid") or "").strip().lower()
        meta = metadata.get(sid, {})

        item["source_id"] = meta.get("source_id", "")
        item["apk_name"] = meta.get("apk_name", "")
        item["scenario"] = meta.get("scenario") or str(scenario_name)

        out.append(item)
    return out


def _first_pt(path: Path) -> Path:
    try:
        return next(path.glob("*.pt"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No PT files found under {path}") from exc


def _expected_training_contract(ckpt: dict[str, Any], cfg: dict[str, Any]) -> int:
    node_input_dim = int(ckpt.get("node_input_dim") or 0)
    if node_input_dim > 0:
        return node_input_dim
    train_data = ((cfg.get("data", {}) or {}).get("train", {}) or {})
    pt_dir = _resolve(train_data.get("pt_dir", ""))
    payload = load_aeg_payload(_first_pt(pt_dir), validate=True)
    return int(payload["node_x"].size(1))


def _validate_scenario(dataset: AEGDataset, node_input_dim: int) -> None:
    for path, _label in dataset.samples:
        load_aeg_payload(path, validate=True, expected_node_feature_dim=node_input_dim)


def run(checkpoint: Path, scenario_config: Path, output_dir: Path) -> None:
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    cfg = dict(ckpt.get("cfg") or {})
    if not cfg:
        raise ValueError("Checkpoint does not contain its training config")
    checkpoint_contract = str(ckpt.get("aeg_payload_contract_fingerprint") or "")
    if checkpoint_contract and checkpoint_contract != AEG_PAYLOAD_CONTRACT_FINGERPRINT:
        raise ValueError("Checkpoint AEG payload contract does not match the current code")
    scenarios = (_load_yaml(scenario_config).get("scenarios") or {})
    if not scenarios:
        raise ValueError("Evaluation config requires a non-empty scenarios mapping")

    node_input_dim = _expected_training_contract(ckpt, cfg)
    device = _device(cfg)
    model = build_model(cfg, node_input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    output_dir.mkdir(parents=True, exist_ok=True)


    summary: dict[str, Any] = {}
    for name, item in scenarios.items():
        pt_dir = _resolve(item["pt_dir"])
        csv_path = _resolve(item["csv"])
        strict_integrity = bool(item.get("strict_integrity", False))
        dataset = AEGDataset(
            pt_dir,
            csv_path,
            split=str(name),
            strict_integrity=strict_integrity,
            validate_payload_on_load=False,
        )
        _validate_scenario(dataset, node_input_dim)
        metrics, rows = evaluate(
            model,
            _loader(cfg, dataset, train=False),
            device,
            split_name=f"external_{name}",
            dump_rows=True,
        )

        external_meta = _read_external_metadata(csv_path)
        rows = _attach_external_metadata(rows, external_meta, str(name))

        num_rows = len(rows)
        num_with_source_id = sum(1 for row in rows if str(row.get("source_id") or "").strip())
        metrics["num_rows"] = num_rows
        metrics["num_with_source_id"] = num_with_source_id
        metrics["source_id_match_rate"] = num_with_source_id / num_rows if num_rows else 0.0

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
