"""Deterministic validation montages and simple training curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sailor.config import Settings
from sailor.paths import assert_writable_target


def _to_u8(array: np.ndarray) -> np.ndarray:
    finite = np.asarray(array, dtype=np.float64)
    finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros(finite.shape, dtype=np.uint8)
    scaled = np.clip((finite - low) / (high - low) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def _write_pgm(path: Path, image: np.ndarray, settings: Settings) -> None:
    assert_writable_target(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _to_u8(image)
    header = f"P5\n{array.shape[1]} {array.shape[0]}\n255\n"
    path.write_bytes(header.encode("ascii") + array.tobytes())


def save_validation_montages(
    run_dir: Path,
    panels: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    settings: Settings,
) -> None:
    viz_dir = run_dir / "viz"
    for index, (image, truth, prediction, name) in enumerate(panels[:3]):
        axis = int(image.shape[0] // 2)
        slice_image = image[axis]
        slice_truth = truth[axis]
        slice_pred = prediction[axis]
        overlay = np.clip(
            0.5 * _to_u8(slice_image).astype(np.float32)
            + 0.25 * slice_truth.astype(np.float32) * 255.0
            + 0.25 * slice_pred.astype(np.float32) * 255.0,
            0,
            255,
        )
        gap = np.full((slice_image.shape[0], 4), 255, dtype=np.uint8)
        montage = np.concatenate(
            [
                _to_u8(slice_image),
                gap,
                _to_u8(slice_truth),
                gap,
                _to_u8(slice_pred),
                gap,
                overlay.astype(np.uint8),
            ],
            axis=1,
        )
        _write_pgm(viz_dir / f"{index:02d}_{name}_t1c_gt_pred_overlay.pgm", montage, settings)


def save_training_curves(
    run_dir: Path,
    rows: list[dict[str, Any]],
    settings: Settings,
) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows if row.get("split") == "TRAINING"]
    if not epochs:
        return
    width = 320
    height = 120

    def _series(key: str, split: str) -> np.ndarray:
        values = [
            float(row[key])
            for row in rows
            if row.get("split") == split and row.get(key) not in {"", None}
        ]
        canvas = np.full((height, width), 255, dtype=np.uint8)
        if len(values) < 2:
            return canvas
        lo = min(values)
        hi = max(values)
        span = hi - lo if hi > lo else 1.0
        xs = np.linspace(0, width - 1, num=len(values)).astype(int)
        ys = [
            int((height - 1) * (1.0 - (value - lo) / span))
            for value in values
        ]
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                x = int(x0 + (x1 - x0) * step / steps)
                y = int(y0 + (y1 - y0) * step / steps)
                canvas[min(height - 1, max(0, y)), min(width - 1, max(0, x))] = 0
        return canvas

    _write_pgm(run_dir / "viz" / "curve_train_loss.pgm", _series("train_loss", "TRAINING"), settings)
    _write_pgm(run_dir / "viz" / "curve_train_dice.pgm", _series("dice", "TRAINING"), settings)
    _write_pgm(
        run_dir / "viz" / "curve_inner_val_dice.pgm",
        _series("dice", "INNER_VALIDATION"),
        settings,
    )
    _write_pgm(run_dir / "viz" / "curve_lr.pgm", _series("learning_rate", "TRAINING"), settings)
