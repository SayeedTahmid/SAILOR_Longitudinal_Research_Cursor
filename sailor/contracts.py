"""Shape and value contracts shared by all pipeline modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def assert_volume_contract(
    array: np.ndarray,
    *,
    name: str,
    expected_shape: Sequence[int] | None = None,
) -> None:
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D; received shape {array.shape}.")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} shape {array.shape} does not match expected {tuple(expected_shape)}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def assert_mask_contract(
    mask: np.ndarray,
    *,
    name: str,
    expected_shape: Sequence[int] | None = None,
) -> None:
    assert_volume_contract(mask, name=name, expected_shape=expected_shape)
    values = np.unique(mask)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} is not binary; values include {values[:10].tolist()}.")


def assert_aligned_geometry(
    *,
    reference_shape: Sequence[int],
    reference_affine: np.ndarray,
    candidate_shape: Sequence[int],
    candidate_affine: np.ndarray,
    candidate_name: str,
    affine_tolerance: float = 1e-5,
) -> None:
    if tuple(reference_shape) != tuple(candidate_shape):
        raise ValueError(
            f"{candidate_name} shape {tuple(candidate_shape)} does not match "
            f"reference {tuple(reference_shape)}."
        )
    if not np.allclose(
        np.asarray(reference_affine),
        np.asarray(candidate_affine),
        rtol=0.0,
        atol=affine_tolerance,
    ):
        raise ValueError(f"{candidate_name} affine does not match the reference grid.")
