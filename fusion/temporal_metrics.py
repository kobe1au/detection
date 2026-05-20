from __future__ import annotations

from typing import Dict
import numpy as np


def compute_aut(year_metric: Dict[int, float], clip=True) -> float:
    if not year_metric:
        return 0.0

    years_sorted = sorted(year_metric.keys())
    values = np.array([year_metric[y] for y in years_sorted], dtype=np.float64)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if clip:
        values = np.clip(values, 0.0, 1.0)

    if len(values) == 1:
        return float(values[0])

    years_arr = np.asarray(years_sorted, dtype=np.float64)
    horizon = years_arr[-1] - years_arr[0]
    if horizon <= 0:
        return float(values.mean())

    return float(np.trapz(values, x=years_arr) / horizon)

def compute_aut_suite(year_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute AUT for all metric names that appear in any year.
    """
    if not year_metrics:
        return {}

    metric_names = sorted({k for metrics in year_metrics.values() for k in metrics.keys()})

    out = {}
    for m in metric_names:
        per_year = {y: v[m] for y, v in year_metrics.items() if m in v}
        out[f"AUT_{m}"] = compute_aut(per_year)
    return out