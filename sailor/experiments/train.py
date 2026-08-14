"""Train and score the shared Phase-3 U-Net under frozen patient folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from sailor.baselines.io import (
    encode_delta,
    history_sessions,
    load_mask,
    load_mri,
)
from sailor.constants import (
    BASELINE_BATCH_SIZE,
    BASELINE_FOREGROUND_PATCH_FRACTION,
    BASELINE_INNER_EPOCHS,
    BASELINE_LR_GRID,
    BASELINE_OUTER_EPOCHS,
    BASELINE_PATCH_SIZE,
    MODEL_VERSION,
)
from sailor.errors import StopProtocolError
from sailor.evaluation.metrics import window_metrics
from sailor.models.unet3d import build_unet, require_torch


def delta_value(window: dict[str, Any], mode: str, constant_days: float | None) -> float:
    if mode == "C0":
        return 0.0
    if mode == "C1":
        return encode_delta(window["target_delta_days"])
    if mode == "C1_constant":
        if constant_days is None:
            raise StopProtocolError(
                "G4 constant Δt is missing.",
                "The constant-time control cannot be constructed.",
                "Compute the outer-training median Δt before training C1_constant.",
            )
        return encode_delta(constant_days)
    raise StopProtocolError(
        f"Unknown baseline rung {mode}.",
        "Only C0, C1, and C1_constant are approved in Phase 3.",
        "Restore the locked Phase-3 rung identifiers.",
    )


def build_input_volume(
    artefacts: dict[str, Any],
    window: dict[str, Any],
    *,
    dataset_root: Path,
    mode: str,
    constant_days: float | None = None,
    delta_perturbation_days: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    first, last = history_sessions(window)
    mri_a = np.asarray(
        load_mri(artefacts, window["subject"], first, dataset_root=dataset_root)
    )
    mri_b = np.asarray(
        load_mri(artefacts, window["subject"], last, dataset_root=dataset_root)
    )
    target = np.asarray(
        load_mask(
            artefacts,
            window["subject"],
            window["target_mni_session"],
            dataset_root=dataset_root,
        )
    )
    if mri_a.shape != mri_b.shape or mri_a.shape != target.shape:
        raise StopProtocolError(
            f"History/target geometry mismatch for {window['window_id']}.",
            "The shared U-Net cannot consume this window.",
            "Restore aligned p2.0 arrays.",
        )
    delta = delta_value(window, mode, constant_days)
    if mode == "C1" and delta_perturbation_days:
        delta = encode_delta(window["target_delta_days"] + delta_perturbation_days)
    dt_channel = np.full(mri_a.shape, delta, dtype=np.float32)
    stacked = np.stack(
        [mri_a.astype(np.float32), mri_b.astype(np.float32), dt_channel],
        axis=0,
    )
    from sailor.contracts import assert_baseline_input_contract

    assert_baseline_input_contract(stacked, name=f"{window['window_id']}/input")
    return stacked, target.astype(np.uint8)


def _sample_patch(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    spatial = volume.shape[1:]
    padded = np.pad(
        volume,
        ((0, 0),) + tuple((0, max(0, patch_size - size)) for size in spatial),
        mode="constant",
    )
    padded_mask = np.pad(
        mask,
        tuple((0, max(0, patch_size - size)) for size in spatial),
        mode="constant",
    )
    limits = [max(1, size - patch_size + 1) for size in padded_mask.shape]
    if (
        rng.random() < BASELINE_FOREGROUND_PATCH_FRACTION
        and np.any(padded_mask)
    ):
        coords = np.argwhere(padded_mask > 0)
        center = coords[int(rng.integers(0, len(coords)))]
        starts = [
            int(np.clip(center[axis] - patch_size // 2, 0, limits[axis] - 1))
            for axis in range(3)
        ]
    else:
        starts = [int(rng.integers(0, limit)) for limit in limits]
    slices = tuple(
        slice(start, start + patch_size) for start in starts
    )
    return padded[(slice(None), *slices)], padded_mask[slices]


def _dice_bce_loss(logits, target):  # type: ignore[no-untyped-def]
    torch, _ = require_torch()
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3, 4)
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    return dice.mean() + bce


def _device():
    torch, _ = require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forward_volume(model, volume: np.ndarray, patch_size: int):
    torch, _ = require_torch()
    device = next(model.parameters()).device
    spatial = volume.shape[1:]
    if all(size <= patch_size for size in spatial):
        padded = np.pad(
            volume,
            ((0, 0),) + tuple((0, patch_size - size) for size in spatial),
            mode="constant",
        )
        tensor = torch.from_numpy(padded[None]).to(device)
        with torch.no_grad():
            logits = model(tensor)[0, 0].cpu().numpy()
        return logits[tuple(slice(0, size) for size in spatial)]
    padded = np.pad(
        volume,
        ((0, 0),) + tuple((0, (patch_size - size % patch_size) % patch_size) for size in spatial),
        mode="constant",
    )
    acc = np.zeros(padded.shape[1:], dtype=np.float32)
    weight = np.zeros(padded.shape[1:], dtype=np.float32)
    step = max(1, patch_size // 2)
    for z in range(0, padded.shape[1] - patch_size + 1, step):
        for y in range(0, padded.shape[2] - patch_size + 1, step):
            for x in range(0, padded.shape[3] - patch_size + 1, step):
                patch = padded[:, z : z + patch_size, y : y + patch_size, x : x + patch_size]
                tensor = torch.from_numpy(patch[None]).to(device)
                with torch.no_grad():
                    logits = model(tensor)[0, 0].cpu().numpy()
                acc[z : z + patch_size, y : y + patch_size, x : x + patch_size] += logits
                weight[z : z + patch_size, y : y + patch_size, x : x + patch_size] += 1.0
    weight[weight == 0] = 1.0
    cropped = (acc / weight)[tuple(slice(0, size) for size in spatial)]
    return cropped


def train_unet(
    artefacts: dict[str, Any],
    train_windows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    mode: str,
    seed: int,
    constant_days: float | None = None,
    epochs: int = BASELINE_OUTER_EPOCHS,
    learning_rate: float,
    patch_size: int = BASELINE_PATCH_SIZE,
    validation_windows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    torch, _ = require_torch()
    if not train_windows:
        raise StopProtocolError(
            f"No training windows for {mode}.",
            "A fold would produce an empty model.",
            "Inspect the frozen patient lists.",
        )
    train_subjects = {window["subject"] for window in train_windows}
    if validation_windows:
        overlap = train_subjects & {window["subject"] for window in validation_windows}
        if overlap:
            raise StopProtocolError(
                f"Inner-loop leakage in {mode}: {sorted(overlap)}",
                "Hyperparameter selection would see validation patients.",
                "Use the frozen inner patient partitions.",
            )
    rng = np.random.default_rng(seed)
    device = _device()
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    ordered = list(train_windows)
    for _ in range(epochs):
        rng.shuffle(ordered)
        for start in range(0, len(ordered), BASELINE_BATCH_SIZE):
            batch = ordered[start : start + BASELINE_BATCH_SIZE]
            inputs = []
            targets = []
            for window in batch:
                volume, mask = build_input_volume(
                    artefacts,
                    window,
                    dataset_root=dataset_root,
                    mode=mode,
                    constant_days=constant_days,
                )
                patch, patch_mask = _sample_patch(volume, mask, patch_size, rng)
                inputs.append(patch)
                targets.append(patch_mask.astype(np.float32))
            tensor = torch.from_numpy(np.stack(inputs)).to(device)
            target = torch.from_numpy(np.stack(targets))[:, None].to(device)
            optimizer.zero_grad()
            loss = _dice_bce_loss(model(tensor), target)
            loss.backward()
            optimizer.step()
    model.eval()
    val_dice = None
    if validation_windows:
        scores = []
        for window in validation_windows:
            volume, mask = build_input_volume(
                artefacts,
                window,
                dataset_root=dataset_root,
                mode=mode,
                constant_days=constant_days,
            )
            logits = _forward_volume(model, volume, patch_size)
            prediction = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.uint8)
            scores.append(window_metrics(prediction, mask)["dice"])
        val_dice = float(np.mean(scores)) if scores else None
    return {
        "model": model,
        "mode": mode,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "validation_dice": val_dice,
        "n_train_windows": len(train_windows),
        "train_patients": sorted(train_subjects),
        "model_version": MODEL_VERSION,
        "describe": model.describe(),
    }


def select_learning_rate(
    artefacts: dict[str, Any],
    inner_folds: list[dict[str, Any]],
    all_windows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    mode: str,
    seed: int,
    constant_days: float | None,
    budget: dict[str, Any],
) -> float:
    grid = tuple(budget.get("lr_grid", BASELINE_LR_GRID))
    inner_epochs = int(budget.get("inner_epochs", BASELINE_INNER_EPOCHS))
    patch_size = int(budget.get("patch_size", BASELINE_PATCH_SIZE))
    means: dict[float, list[float]] = {lr: [] for lr in grid}
    for lr in grid:
        for inner in inner_folds:
            train = [
                window
                for window in all_windows
                if window["subject"] in set(inner["train_patients"])
            ]
            validation = [
                window
                for window in all_windows
                if window["subject"] in set(inner["validation_patients"])
            ]
            fitted = train_unet(
                artefacts,
                train,
                dataset_root=dataset_root,
                mode=mode,
                seed=seed,
                constant_days=constant_days,
                epochs=inner_epochs,
                learning_rate=lr,
                patch_size=patch_size,
                validation_windows=validation,
            )
            if fitted["validation_dice"] is not None:
                means[lr].append(fitted["validation_dice"])
    scored = {
        lr: float(np.mean(values)) if values else float("-inf")
        for lr, values in means.items()
    }
    return max(scored, key=lambda lr: (scored[lr], -lr))


def predict_windows(
    model,
    artefacts: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    mode: str,
    constant_days: float | None = None,
    patch_size: int = BASELINE_PATCH_SIZE,
    delta_perturbation_days: float = 0.0,
    rung: str,
) -> list[dict[str, Any]]:
    rows = []
    for window in windows:
        volume, target = build_input_volume(
            artefacts,
            window,
            dataset_root=dataset_root,
            mode=mode,
            constant_days=constant_days,
            delta_perturbation_days=delta_perturbation_days,
        )
        logits = _forward_volume(model, volume, patch_size)
        prediction = (1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20))) >= 0.5).astype(
            np.uint8
        )
        spacing = tuple(
            float(value)
            for value in artefacts["sessions"][
                (window["subject"], window["target_mni_session"])
            ].get("spacing", (1.0, 1.0, 1.0))
        )
        metrics = window_metrics(prediction, target, spacing=spacing)
        rows.append(
            {
                "window_id": window["window_id"],
                "subject": window["subject"],
                "rung": rung,
                "target_mni_session": window["target_mni_session"],
                **metrics,
            }
        )
    return rows


def make_predict_fn(
    artefacts: dict[str, Any],
    *,
    dataset_root: Path,
) -> Callable[[dict[str, Any]], np.ndarray]:
    def _predict(window: dict[str, Any]) -> np.ndarray:
        from sailor.baselines.persistence import persistence_prediction

        return persistence_prediction(
            artefacts, window, dataset_root=dataset_root
        )

    return _predict
