#!/usr/bin/env python3
from __future__ import annotations

import argparse
from cProfile import label
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _safe_rate(values: list[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else float("nan")


def _index_clean_rows(clean_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index clean diagnostics by clean sample id.

    Expected clean diagnostics columns from fusion.train.evaluate():
      sid, label, pred, prob_malware, ...
    """
    out: dict[str, dict[str, str]] = {}
    for row in clean_rows:
        sid = str(row.get("sid") or "").strip().lower()
        if sid:
            out[sid] = row
    return out


def _pair_external_rows(
    clean_by_sid: dict[str, dict[str, str]],
    external_rows: list[dict[str, str]],
    scenario_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    total_external = len(external_rows)
    with_source_id = 0
    missing_clean = 0

    for obf in external_rows:
        obf_sid = str(obf.get("sid") or "").strip().lower()
        source_id = str(obf.get("source_id") or "").strip().lower()

        if source_id:
            with_source_id += 1

        clean = clean_by_sid.get(source_id)
        if clean is None:
            missing_clean += 1
            continue

        label = _to_int(obf.get("label"))
        clean_label = _to_int(clean.get("label"))
        pred_clean = _to_int(clean.get("pred"))
        pred_obf = _to_int(obf.get("pred"))

        prob_clean = _to_float(clean.get("prob_malware"))
        prob_obf = _to_float(obf.get("prob_malware"))

        flip = pred_clean != pred_obf
        clean_correct = pred_clean == clean_label
        obf_correct = pred_obf == label

        # Confidence of the true class, useful for label-aware confidence drop.
        if label == 1:
            true_prob_clean = prob_clean
            true_prob_obf = prob_obf
        elif label == 0:
            true_prob_clean = 1.0 - prob_clean if not math.isnan(prob_clean) else float("nan")
            true_prob_obf = 1.0 - prob_obf if not math.isnan(prob_obf) else float("nan")
        else:
            true_prob_clean = float("nan")
            true_prob_obf = float("nan")

        row = {
            "scenario": str(obf.get("scenario") or scenario_name),
            "source_id": source_id,
            "obf_sid": obf_sid,
            "apk_name": obf.get("apk_name", ""),
            "label": label,
            "clean_label": clean_label,
            "label_mismatch": int(clean_label != label),
            "pred_clean": pred_clean,
            "pred_obf": pred_obf,
            "flip": int(flip),
            "clean_correct": int(clean_correct),
            "obf_correct": int(obf_correct),
            "prob_clean": prob_clean,
            "prob_obf": prob_obf,
            "prob_delta": prob_obf - prob_clean,
            "prob_abs_delta": abs(prob_obf - prob_clean)
            if not math.isnan(prob_clean) and not math.isnan(prob_obf)
            else float("nan"),
            "true_prob_clean": true_prob_clean,
            "true_prob_obf": true_prob_obf,
            "true_confidence_drop": true_prob_clean - true_prob_obf
            if not math.isnan(true_prob_clean) and not math.isnan(true_prob_obf)
            else float("nan"),
        }
        paired.append(row)

    summary = _summarize_pairs(
        scenario_name=scenario_name,
        paired=paired,
        total_external=total_external,
        with_source_id=with_source_id,
        missing_clean=missing_clean,
    )
    return paired, summary


def _summarize_pairs(
    scenario_name: str,
    paired: list[dict[str, Any]],
    total_external: int,
    with_source_id: int,
    missing_clean: int,
) -> dict[str, Any]:
    paired_count = len(paired)

    flips = [bool(row["flip"]) for row in paired]
    clean_correct = [bool(row["clean_correct"]) for row in paired]
    obf_correct = [bool(row["obf_correct"]) for row in paired]

    prob_abs_delta = [_to_float(row.get("prob_abs_delta")) for row in paired]
    prob_delta = [_to_float(row.get("prob_delta")) for row in paired]
    true_confidence_drop = [_to_float(row.get("true_confidence_drop")) for row in paired]

    clean_acc = _safe_rate(clean_correct)
    obf_acc = _safe_rate(obf_correct)
    label_mismatches = [bool(row.get("label_mismatch", 0)) for row in paired]

    return {
        "scenario": scenario_name,
        "external_rows": total_external,
        "rows_with_source_id": with_source_id,
        "paired_count": paired_count,
        "missing_clean_pairs": missing_clean,
        "source_id_rate": with_source_id / total_external if total_external else float("nan"),
        "pair_rate": paired_count / total_external if total_external else float("nan"),
        "flip_rate": _safe_rate(flips),
        "clean_acc_on_paired": clean_acc,
        "obf_acc_on_paired": obf_acc,
        "acc_drop_on_paired": clean_acc - obf_acc
        if not math.isnan(clean_acc) and not math.isnan(obf_acc)
        else float("nan"),
        "mean_prob_delta_obf_minus_clean": _safe_mean(prob_delta),
        "mean_prob_abs_delta": _safe_mean(prob_abs_delta),
        "mean_true_confidence_drop": _safe_mean(true_confidence_drop),
        "label_mismatch_count": sum(1 for v in label_mismatches if v),
        "label_mismatch_rate": _safe_rate(label_mismatches),
    }


def _find_external_files(external_dir: Path) -> list[Path]:
    files = sorted(external_dir.glob("diagnostics_test_external_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No diagnostics_test_external_*.csv found under {external_dir}"
        )
    return files


def _scenario_name_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "diagnostics_test_external_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def run(clean_csv: Path, external_dir: Path, output_dir: Path) -> None:
    clean_rows = _read_csv(clean_csv)
    clean_by_sid = _index_clean_rows(clean_rows)

    if not clean_by_sid:
        raise ValueError(f"No clean rows with sid found in {clean_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for external_csv in _find_external_files(external_dir):
        scenario = _scenario_name_from_path(external_csv)
        external_rows = _read_csv(external_csv)

        paired, summary = _pair_external_rows(
            clean_by_sid=clean_by_sid,
            external_rows=external_rows,
            scenario_name=scenario,
        )

        all_pair_rows.extend(paired)
        summary_rows.append(summary)

        _write_csv(output_dir / f"paired_{scenario}.csv", paired)

    if all_pair_rows:
        overall = _summarize_pairs(
            scenario_name="overall",
            paired=all_pair_rows,
            total_external=sum(int(row["external_rows"]) for row in summary_rows),
            with_source_id=sum(int(row["rows_with_source_id"]) for row in summary_rows),
            missing_clean=sum(int(row["missing_clean_pairs"]) for row in summary_rows),
        )
        summary_rows.append(overall)

    _write_csv(output_dir / "paired_all.csv", all_pair_rows)
    _write_csv(output_dir / "summary_pairs.csv", summary_rows)

    with (output_dir / "summary_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote paired summaries to {output_dir}")
    for row in summary_rows:
        print(
            f"{row['scenario']}: "
            f"paired={row['paired_count']}, "
            f"flip_rate={row['flip_rate']:.4f}, "
            f"acc_drop={row['acc_drop_on_paired']:.4f}, "
            f"true_conf_drop={row['mean_true_confidence_drop']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired clean-vs-Obfuscapk diagnostics using "
            "clean.sid == external.source_id."
        )
    )
    parser.add_argument(
        "--clean",
        required=True,
        type=Path,
        help="Path to diagnostics_test_clean.csv from the clean test run.",
    )
    parser.add_argument(
        "--external-dir",
        required=True,
        type=Path,
        help="Directory containing diagnostics_test_external_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for paired CSV/JSON summaries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.clean, args.external_dir, args.output_dir)