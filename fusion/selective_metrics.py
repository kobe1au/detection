"""
Selective classification metrics.

- AURC (Area Under Risk-Coverage curve):  Geifman & El-Yaniv 2017, re-popularized
  in 2024-2025 selective classification work as the core selective metric.
- E-AURC (Excess AURC over optimal):  stronger model-agnostic comparison.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def risk_coverage_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)

    if confidences.shape[0] != correct.shape[0]:
        raise ValueError(
            f"confidences and correct must have same length, "
            f"got {confidences.shape[0]} and {correct.shape[0]}"
        )

    n = confidences.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    correct = np.clip(correct, 0.0, 1.0)

    order = np.argsort(-confidences)
    correct_sorted = correct[order]

    cum_correct = np.cumsum(correct_sorted)
    k = np.arange(1, n + 1, dtype=np.float64)

    coverage = k / float(n)
    risk = 1.0 - cum_correct / k
    return coverage, risk


def aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    coverage, risk = risk_coverage_curve(confidences, correct)
    if coverage.size == 0:
        return 0.0
    return float(np.trapz(risk, coverage))


def eaurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)

    if confidences.shape[0] != correct.shape[0]:
        raise ValueError(
            f"confidences and correct must have same length, "
            f"got {confidences.shape[0]} and {correct.shape[0]}"
        )

    n = correct.shape[0]
    if n == 0:
        return 0.0

    correct = np.clip(correct, 0.0, 1.0)
    n_correct = int(np.round(correct.sum()))

    opt_correct = np.concatenate([
        np.ones(n_correct, dtype=np.float64),
        np.zeros(n - n_correct, dtype=np.float64),
    ])
    opt_conf = np.arange(n, 0, -1, dtype=np.float64)

    aurc_opt = aurc(opt_conf, opt_correct)
    return float(aurc(confidences, correct) - aurc_opt)


def risk_at_coverage(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_coverage: float,
) -> float:
    coverage, risk = risk_coverage_curve(confidences, correct)
    if coverage.size == 0:
        return 1.0

    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    idx = int(np.searchsorted(coverage, target_coverage, side="left"))
    idx = min(idx, len(risk) - 1)
    return float(risk[idx])


def coverage_at_risk(
    confidences: np.ndarray,
    correct: np.ndarray,
    max_risk: float,
) -> float:
    coverage, risk = risk_coverage_curve(confidences, correct)
    if coverage.size == 0:
        return 0.0

    max_risk = float(np.clip(max_risk, 0.0, 1.0))
    valid = risk <= max_risk
    if not valid.any():
        return 0.0
    return float(coverage[valid].max())