"""Patient-level aggregation, bootstrap CIs, Holm adjustment, and MDE."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any

import numpy as np

from sailor.constants import (
    MDE_ALPHA,
    MDE_ILLUSTRATIVE_DICE_SD,
    MDE_POWER,
    PATIENT_BOOTSTRAP_REPLICATES,
)


def aggregate_patient_scores(
    window_rows: list[dict[str, Any]],
    *,
    metric: str = "dice",
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in window_rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        grouped[row["subject"]].append(float(value))
    return {
        subject: float(np.mean(values))
        for subject, values in sorted(grouped.items())
    }


def patient_mean(scores: dict[str, float]) -> float:
    if not scores:
        raise ValueError("No patient scores were provided.")
    return float(np.mean(list(scores.values())))


def percentile_ci(
    samples: np.ndarray,
    *,
    alpha: float = MDE_ALPHA,
) -> tuple[float, float]:
    lower = 100.0 * (alpha / 2.0)
    upper = 100.0 * (1.0 - alpha / 2.0)
    return (
        float(np.percentile(samples, lower)),
        float(np.percentile(samples, upper)),
    )


def bootstrap_patient_mean(
    scores: dict[str, float],
    *,
    replicates: int = PATIENT_BOOTSTRAP_REPLICATES,
    seed: int,
) -> dict[str, Any]:
    patients = sorted(scores)
    values = np.asarray([scores[patient] for patient in patients], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(values), size=len(values))
        draws[index] = float(values[chosen].mean())
    point = float(values.mean())
    low, high = percentile_ci(draws)
    return {
        "n_patients": len(values),
        "mean": point,
        "ci95": [low, high],
        "replicates": replicates,
        "unit": "patient",
    }


def paired_patient_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    replicates: int = PATIENT_BOOTSTRAP_REPLICATES,
    seed: int,
) -> dict[str, Any]:
    patients = sorted(set(left) & set(right))
    if not patients:
        raise ValueError("Paired comparison has an empty patient intersection.")
    delta = np.asarray(
        [left[patient] - right[patient] for patient in patients],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(delta), size=len(delta))
        draws[index] = float(delta[chosen].mean())
    point = float(delta.mean())
    low, high = percentile_ci(draws)
    proportion_nonpositive = float(np.mean(draws <= 0.0))
    proportion_nonnegative = float(np.mean(draws >= 0.0))
    p_value = min(1.0, 2.0 * min(proportion_nonpositive, proportion_nonnegative))
    beats = bool(low > 0.0)
    return {
        "n_patients": len(patients),
        "patients": patients,
        "mean_difference": point,
        "ci95": [low, high],
        "p_bootstrap": p_value,
        "beats": beats,
        "replicates": replicates,
        "unit": "patient",
        "interpretation": (
            "left beats right outside the paired 95% CI"
            if beats
            else "not distinguishable from zero at the paired 95% CI"
        ),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - index)))
        adjusted[name] = running
    return adjusted


def minimum_detectable_effect(
    n_patients: int,
    sd_difference: float,
    *,
    alpha: float = MDE_ALPHA,
    power: float = MDE_POWER,
) -> float:
    if n_patients <= 1:
        raise ValueError("MDE requires at least two patients.")
    z_alpha = 1.959963984540054
    z_power = 0.841621233572914
    if abs(alpha - 0.05) > 1e-12 or abs(power - 0.80) > 1e-12:
        raise ValueError("Phase 3 MDE is locked to alpha=0.05 and power=0.80.")
    return float((z_alpha + z_power) * sd_difference / sqrt(n_patients))


def illustrative_mde_table(n_patients: int) -> list[dict[str, float]]:
    return [
        {
            "assumed_patient_sd": sd,
            "mde_dice": minimum_detectable_effect(n_patients, sd),
        }
        for sd in MDE_ILLUSTRATIVE_DICE_SD
    ]
