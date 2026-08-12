"""Leakage-safe per-volume T1c normalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from sailor.contracts import assert_mask_contract, assert_volume_contract
from sailor.errors import StopProtocolError


def robust_scale_volume(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[np.ndarray, dict[str, float]]:
    assert_volume_contract(volume, name="T1c")
    assert_mask_contract(
        brain_mask,
        name="BrainExtractionMask",
        expected_shape=volume.shape,
    )
    inside = brain_mask.astype(bool)
    values = np.asarray(volume, dtype=np.float64)[inside]
    if values.size == 0:
        raise StopProtocolError(
            "BrainExtractionMask contains no foreground voxels.",
            "MRI normalization cannot be defined.",
            "Exclude or repair the brain mask through an approved preprocessing policy.",
        )
    low, high = np.percentile(values, [lower_percentile, upper_percentile])
    clipped = np.clip(values, low, high)
    median = float(np.median(clipped))
    q25, q75 = np.percentile(clipped, [25.0, 75.0])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 1e-8:
        raise StopProtocolError(
            "T1c robust intensity scale is zero or non-finite.",
            "The volume cannot be normalized reproducibly.",
            "Inspect the source volume and brain mask; do not substitute cohort statistics.",
        )
    output = np.zeros(volume.shape, dtype=np.float32)
    output[inside] = ((clipped - median) / iqr).astype(np.float32)
    assert_volume_contract(output, name="normalized_T1c", expected_shape=volume.shape)
    return output, {
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "clip_low": float(low),
        "clip_high": float(high),
        "median": median,
        "iqr": iqr,
        "scope": "single_volume_brain_mask",
    }


def normalization_manifest_entry(
    *,
    subject: str,
    session: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "subject": subject,
        "session": session,
        "parameters": parameters,
        "cohort_statistics_used": False,
        "outer_test_information_used": False,
    }
