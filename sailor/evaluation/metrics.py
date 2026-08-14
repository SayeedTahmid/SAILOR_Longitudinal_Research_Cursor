"""Mask overlap, volume, and surface metrics for Phase 3."""

from __future__ import annotations

from typing import Any

import numpy as np

from sailor.contracts import assert_mask_contract


def dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    assert_mask_contract(prediction, name="prediction")
    assert_mask_contract(target, name="target", expected_shape=prediction.shape)
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    intersection = float(np.count_nonzero(pred & truth))
    denom = float(np.count_nonzero(pred) + np.count_nonzero(truth))
    if denom == 0.0:
        return 1.0
    return 2.0 * intersection / denom


def relative_volume_error(prediction: np.ndarray, target: np.ndarray) -> float:
    assert_mask_contract(prediction, name="prediction")
    assert_mask_contract(target, name="target", expected_shape=prediction.shape)
    predicted = float(np.count_nonzero(prediction))
    truth = float(np.count_nonzero(target))
    if truth == 0.0:
        return 0.0 if predicted == 0.0 else float("inf")
    return abs(predicted - truth) / truth


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    padded = np.pad(binary, 1, mode="constant")
    neighbors = (
        padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
    )
    surface = binary & ~neighbors
    return np.argwhere(surface)


def hausdorff_95(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float | None:
    assert_mask_contract(prediction, name="prediction")
    assert_mask_contract(target, name="target", expected_shape=prediction.shape)
    pred_surface = _surface_voxels(prediction)
    truth_surface = _surface_voxels(target)
    if pred_surface.size == 0 or truth_surface.size == 0:
        return None
    scale = np.asarray(spacing, dtype=np.float64)
    pred_pts = pred_surface.astype(np.float64) * scale
    truth_pts = truth_surface.astype(np.float64) * scale
    if pred_pts.shape[0] * truth_pts.shape[0] > 25_000_000:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return None
        pred_bin = (np.asarray(prediction) > 0).astype(np.uint8)
        truth_bin = (np.asarray(target) > 0).astype(np.uint8)
        dt_pred = distance_transform_edt(~pred_bin.astype(bool), sampling=spacing)
        dt_truth = distance_transform_edt(~truth_bin.astype(bool), sampling=spacing)
        directed_pred = dt_truth[pred_bin.astype(bool)]
        directed_truth = dt_pred[truth_bin.astype(bool)]
        if directed_pred.size == 0 or directed_truth.size == 0:
            return None
        return float(
            max(np.percentile(directed_pred, 95), np.percentile(directed_truth, 95))
        )
    delta = pred_pts[:, None, :] - truth_pts[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    directed_pred = distances.min(axis=1)
    directed_truth = distances.min(axis=0)
    return float(max(np.percentile(directed_pred, 95), np.percentile(directed_truth, 95)))


def window_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    hd95 = hausdorff_95(prediction, target, spacing=spacing)
    return {
        "dice": dice_coefficient(prediction, target),
        "iou": iou_coefficient(prediction, target),
        "precision": precision_score(prediction, target),
        "recall": recall_score(prediction, target),
        "relative_volume_error": relative_volume_error(prediction, target),
        "hd95_mm": hd95,
        "predicted_voxels": int(np.count_nonzero(prediction)),
        "target_voxels": int(np.count_nonzero(target)),
    }


def iou_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    assert_mask_contract(prediction, name="prediction")
    assert_mask_contract(target, name="target", expected_shape=prediction.shape)
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    intersection = float(np.count_nonzero(pred & truth))
    union = float(np.count_nonzero(pred | truth))
    if union == 0.0:
        return 1.0
    return intersection / union


def precision_score(prediction: np.ndarray, target: np.ndarray) -> float:
    assert_mask_contract(prediction, name="prediction")
    assert_mask_contract(target, name="target", expected_shape=prediction.shape)
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    predicted = float(np.count_nonzero(pred))
    if predicted == 0.0:
        return 1.0 if np.count_nonzero(truth) == 0 else 0.0
    return float(np.count_nonzero(pred & truth)) / predicted


def recall_score(prediction: np.ndarray, target: np.ndarray) -> float:
    assert_mask_contract(prediction, name="prediction")
    assert_mask_contract(target, name="target", expected_shape=prediction.shape)
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    total = float(np.count_nonzero(truth))
    if total == 0.0:
        return 1.0 if np.count_nonzero(pred) == 0 else 0.0
    return float(np.count_nonzero(pred & truth)) / total
