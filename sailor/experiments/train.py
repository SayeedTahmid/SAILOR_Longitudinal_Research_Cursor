"""Train and score the shared Phase-3 U-Net under frozen patient folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import time

import numpy as np

from sailor.config import Settings

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
    DATA_VERSION,
    MODEL_VERSION,
    PREPROCESSING_VERSION,
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
    dims = tuple(range(1, logits.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    dice_loss_mean = dice_loss.mean()
    return dice_loss_mean + bce, dice_loss_mean, bce, probs


def _device():
    torch, _ = require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gpu_info() -> dict[str, Any]:
    torch, _ = require_torch()
    if not torch.cuda.is_available():
        return {"gpu_name": "CPU", "peak_vram_bytes": 0}
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _patch_metrics(probs, target) -> dict[str, float]:
    torch, _ = require_torch()
    pred = (probs >= 0.5).to(target.dtype)
    dims = tuple(range(1, pred.ndim))
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "predicted_voxels": float(pred.sum().item()),
        "target_voxels": float(target.sum().item()),
    }


def _session_spacing(artefacts: dict[str, Any], window: dict[str, Any]) -> tuple[float, float, float]:
    session = artefacts["sessions"][(window["subject"], window["target_mni_session"])]
    spacing = session.get("spacing", (1.0, 1.0, 1.0))
    return tuple(float(value) for value in spacing)


def _volume_mm3(voxels: float, spacing: tuple[float, float, float]) -> float:
    return float(voxels) * float(np.prod(spacing))


def _volume_losses(logits: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    torch, _ = require_torch()
    logit_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32)[None, None])
    target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32)[None, None])
    loss, dice_loss, bce, _ = _dice_bce_loss(logit_tensor, target_tensor)
    return float(loss.item()), float(dice_loss.item()), float(bce.item())


def _score_window_set(
    model,
    artefacts: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    mode: str,
    constant_days: float | None,
    patch_size: int,
) -> dict[str, float]:
    empty = {
        "dice": float("nan"),
        "iou": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "predicted_voxels": 0.0,
        "target_voxels": 0.0,
        "predicted_volume_mm3": 0.0,
        "target_volume_mm3": 0.0,
        "loss": float("nan"),
        "dice_loss": float("nan"),
        "bce_loss": float("nan"),
    }
    if not windows:
        return empty
    rows = []
    losses = []
    for window in windows:
        volume, target = build_input_volume(
            artefacts,
            window,
            dataset_root=dataset_root,
            mode=mode,
            constant_days=constant_days,
        )
        logits = _forward_volume(model, volume, patch_size)
        prediction = (1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20))) >= 0.5).astype(
            np.uint8
        )
        spacing = _session_spacing(artefacts, window)
        metrics = window_metrics(prediction, target, spacing=spacing)
        metrics["predicted_volume_mm3"] = _volume_mm3(
            metrics["predicted_voxels"], spacing
        )
        metrics["target_volume_mm3"] = _volume_mm3(metrics["target_voxels"], spacing)
        loss, dice_loss, bce = _volume_losses(logits, target)
        metrics["loss"] = loss
        metrics["dice_loss"] = dice_loss
        metrics["bce_loss"] = bce
        rows.append(metrics)
        losses.append((loss, dice_loss, bce))
    return {
        "dice": float(np.mean([row["dice"] for row in rows])),
        "iou": float(np.mean([row["iou"] for row in rows])),
        "precision": float(np.mean([row["precision"] for row in rows])),
        "recall": float(np.mean([row["recall"] for row in rows])),
        "predicted_voxels": float(np.mean([row["predicted_voxels"] for row in rows])),
        "target_voxels": float(np.mean([row["target_voxels"] for row in rows])),
        "predicted_volume_mm3": float(
            np.mean([row["predicted_volume_mm3"] for row in rows])
        ),
        "target_volume_mm3": float(np.mean([row["target_volume_mm3"] for row in rows])),
        "loss": float(np.mean([item[0] for item in losses])),
        "dice_loss": float(np.mean([item[1] for item in losses])),
        "bce_loss": float(np.mean([item[2] for item in losses])),
    }


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
    settings: Settings | None = None,
    run_dir: Path | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from sailor.experiments.checkpointing import (
        load_checkpoint,
        resolve_resume_checkpoint,
        save_epoch_checkpoints,
        validate_checkpoint,
    )
    from sailor.experiments.logging import (
        append_csv_row,
        append_jsonl,
        log_failure,
        write_training_summary,
    )
    from sailor.experiments.visualize import save_training_curves, save_validation_montages

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
    if run_dir is None or identity is None or settings is None:
        raise StopProtocolError(
            "Training was requested without a checkpoint identity.",
            "C0/C1/G4 cannot resume after a Colab disconnect.",
            "Pass run_dir, identity, and settings from the Stage 3 orchestrator.",
        )
    rng = np.random.default_rng(seed)
    device = _device()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    best_monitor = float("-inf")
    best_epoch = None
    cumulative = 0.0
    history: list[dict[str, Any]] = []
    resume_path = resolve_resume_checkpoint(run_dir, epochs)
    if resume_path is not None:
        payload = load_checkpoint(resume_path)
        validate_checkpoint(payload, identity)
        completed = int(payload["epoch"])
        if completed < 0 or completed >= epochs:
            raise StopProtocolError(
                f"Checkpoint epoch {completed} is outside 0..{epochs - 1}.",
                "Resuming it would change the locked epoch budget.",
                "Inspect the checkpoint and do not improvise a new schedule.",
            )
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        if payload.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(payload["cuda_rng_state"])
        start_epoch = completed + 1
        best_monitor = float(payload.get("best_validation_metric", float("-inf")))
        best_epoch = payload.get("best_epoch")
        cumulative = float(payload.get("cumulative_seconds", 0.0))
        history = list(payload.get("history", []))
    train_spacing = _session_spacing(artefacts, train_windows[0])

    def _halt(message: str, found: str, what: str, *, epoch: int | None = None) -> None:
        log_failure(
            run_dir,
            {
                "mode": mode,
                "epoch": epoch,
                "error": message,
                "found": found,
                "what": what,
                "split_role": identity.get("split_role"),
            },
            settings,
        )
        raise StopProtocolError(message, found, what)

    ordered = list(train_windows)
    val_dice = None
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        rng.shuffle(ordered)
        epoch_losses = []
        epoch_parts = []
        try:
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
                loss, dice_loss, bce, probs = _dice_bce_loss(model(tensor), target)
                if not torch.isfinite(loss):
                    _halt(
                        f"Non-finite loss at {mode} epoch {epoch}.",
                        "Continuing would write NaN weights into later folds.",
                        "Stop, inspect the log, and do not resume this corrupted step.",
                        epoch=epoch,
                    )
                if float(loss.item()) > 50.0:
                    _halt(
                        f"Exploding loss {float(loss.item()):.3f} at {mode} epoch {epoch}.",
                        "The run is numerically unstable.",
                        "Stop rather than changing the locked learning rate.",
                        epoch=epoch,
                    )
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                epoch_parts.append(
                    (
                        float(dice_loss.item()),
                        float(bce.item()),
                        _patch_metrics(probs.detach(), target),
                    )
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                try:
                    _halt(
                        "GPU ran out of memory during a locked Phase-3 epoch.",
                        "Silently shrinking the batch would change the experiment.",
                        "Rerun on a larger GPU without changing patch size or batch size.",
                        epoch=epoch,
                    )
                except StopProtocolError as halted:
                    raise halted from exc
            raise
        train_loss = float(np.mean(epoch_losses))
        train_metrics = {
            "dice": float(np.mean([part[2]["dice"] for part in epoch_parts])),
            "iou": float(np.mean([part[2]["iou"] for part in epoch_parts])),
            "precision": float(np.mean([part[2]["precision"] for part in epoch_parts])),
            "recall": float(np.mean([part[2]["recall"] for part in epoch_parts])),
            "predicted_voxels": float(
                np.mean([part[2]["predicted_voxels"] for part in epoch_parts])
            ),
            "target_voxels": float(
                np.mean([part[2]["target_voxels"] for part in epoch_parts])
            ),
            "predicted_volume_mm3": _volume_mm3(
                float(np.mean([part[2]["predicted_voxels"] for part in epoch_parts])),
                train_spacing,
            ),
            "target_volume_mm3": _volume_mm3(
                float(np.mean([part[2]["target_voxels"] for part in epoch_parts])),
                train_spacing,
            ),
        }
        flags = []
        if train_metrics["predicted_voxels"] == 0:
            flags.append("all_zero_predictions")
        model.eval()
        val_metrics = None
        if validation_windows:
            val_metrics = _score_window_set(
                model,
                artefacts,
                validation_windows,
                dataset_root=dataset_root,
                mode=mode,
                constant_days=constant_days,
                patch_size=patch_size,
            )
            val_dice = val_metrics["dice"]
            if np.isfinite(val_dice) and val_dice > best_monitor:
                best_monitor = val_dice
                best_epoch = epoch
            if (
                np.isfinite(train_metrics["dice"])
                and np.isfinite(val_dice)
                and (train_metrics["dice"] - val_dice) > 0.30
                and epoch >= 2
            ):
                flags.append("severe_overfitting")
            if np.isfinite(val_dice) and val_dice < 0.05 and epoch >= 2:
                flags.append("severe_underfitting")
        elif train_metrics["dice"] > best_monitor:
            best_monitor = train_metrics["dice"]
            best_epoch = epoch
        elapsed = time.perf_counter() - started
        cumulative += elapsed
        gpu = _gpu_info()
        train_row = {
            "epoch": epoch,
            "split": "TRAINING",
            "loss": train_loss,
            "train_loss": train_loss,
            "dice_loss": float(np.mean([part[0] for part in epoch_parts])),
            "bce_loss": float(np.mean([part[1] for part in epoch_parts])),
            "learning_rate": learning_rate,
            "epoch_seconds": elapsed,
            "cumulative_seconds": cumulative,
            "flags": ",".join(flags),
            **train_metrics,
            **gpu,
        }
        history.append(train_row)
        append_csv_row(run_dir / "metrics.csv", train_row, settings)
        append_jsonl(run_dir / "training_log.jsonl", train_row, settings)
        if val_metrics is not None:
            val_row = {
                "epoch": epoch,
                "split": "INNER_VALIDATION",
                "train_loss": "",
                "learning_rate": learning_rate,
                "epoch_seconds": elapsed,
                "cumulative_seconds": cumulative,
                "flags": ",".join(flags),
                **val_metrics,
                **gpu,
            }
            history.append(val_row)
            append_csv_row(run_dir / "metrics.csv", val_row, settings)
            append_jsonl(run_dir / "training_log.jsonl", val_row, settings)
        is_final = epoch == epochs - 1
        is_best = best_epoch == epoch
        checkpoint = {
            "identity": identity,
            "experiment_configuration": identity,
            "model_version": MODEL_VERSION,
            "data_version": DATA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "best_validation_metric": best_monitor,
            "best_epoch": best_epoch,
            "fold": identity.get("outer_fold"),
            "repeat": identity.get("repeat"),
            "seed": identity.get("seed"),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
            "cumulative_seconds": cumulative,
            "history": history,
            "scientific_weights": "final_epoch_only",
        }
        save_epoch_checkpoints(
            run_dir=run_dir,
            settings=settings,
            payload=checkpoint,
            is_best=is_best,
            is_final=is_final,
        )
        if is_final:
            viz_windows = validation_windows or train_windows
            panels = []
            for window in sorted(viz_windows, key=lambda item: item["window_id"])[:3]:
                volume, target = build_input_volume(
                    artefacts,
                    window,
                    dataset_root=dataset_root,
                    mode=mode,
                    constant_days=constant_days,
                )
                logits = _forward_volume(model, volume, patch_size)
                prediction = (
                    1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20))) >= 0.5
                ).astype(np.uint8)
                panels.append((volume[1], target, prediction, window["window_id"][:12]))
            save_validation_montages(run_dir, panels, settings)
        save_training_curves(run_dir, history, settings)
        write_training_summary(
            run_dir / "training_summary.json",
            {
                "mode": mode,
                "epochs_completed": epoch + 1,
                "epochs_budget": epochs,
                "learning_rate": learning_rate,
                "best_monitor_metric": best_monitor,
                "best_epoch": best_epoch,
                "best_metric_source": (
                    "INNER_VALIDATION" if validation_windows else "TRAINING_MONITOR_ONLY"
                ),
                "scientific_checkpoint": "final.pt",
                "resumed_from_epoch": start_epoch,
                "cumulative_seconds": cumulative,
                "gpu": gpu,
                "flags": flags,
            },
            settings,
        )
    model.eval()
    if validation_windows and val_dice is None:
        val_dice = _score_window_set(
            model,
            artefacts,
            validation_windows,
            dataset_root=dataset_root,
            mode=mode,
            constant_days=constant_days,
            patch_size=patch_size,
        )["dice"]
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
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "checkpoint_used_for_eval": "final",
        "resumed_from_epoch": start_epoch,
        "gpu": _gpu_info(),
        "cumulative_seconds": cumulative,
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
    settings: Settings,
    fold_root: Path,
    identity_base: dict[str, Any],
) -> float:
    from sailor.experiments.checkpointing import build_identity, inner_run_dir
    from sailor.reporting import write_json

    grid = tuple(budget.get("lr_grid", BASELINE_LR_GRID))
    inner_epochs = int(budget.get("inner_epochs", BASELINE_INNER_EPOCHS))
    patch_size = int(budget.get("patch_size", BASELINE_PATCH_SIZE))
    selection_path = fold_root / "inner" / "selection.json"
    if selection_path.is_file():
        stored = json.loads(selection_path.read_text(encoding="utf-8"))
        if stored.get("identity_hash") != identity_base.get("selection_hash"):
            raise StopProtocolError(
                "Inner-loop selection file does not match this fold identity.",
                "Reusing it would change learning-rate selection after an interruption.",
                "Delete only if you intend to rerun the inner loop under the same locks.",
            )
        return float(stored["selected_lr"])
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
            identity = build_identity(
                mode=mode,
                split_role="INNER_TRAINING",
                repeat=int(identity_base["repeat"]),
                outer_fold=int(identity_base["outer_fold"]),
                inner_fold=int(inner["inner_fold"]),
                seed=seed,
                learning_rate=float(lr),
                epochs=inner_epochs,
                patch_size=patch_size,
                fold_scheme=str(identity_base["fold_scheme"]),
                train_patients=[window["subject"] for window in train],
                validation_patients=[window["subject"] for window in validation],
                test_patients=list(identity_base["test_patients"]),
            )
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
                settings=settings,
                run_dir=inner_run_dir(fold_root, int(inner["inner_fold"]), float(lr)),
                identity=identity,
            )
            if fitted["validation_dice"] is not None:
                means[lr].append(fitted["validation_dice"])
    scored = {
        lr: float(np.mean(values)) if values else float("-inf")
        for lr, values in means.items()
    }
    selected = max(scored, key=lambda lr: (scored[lr], -lr))
    write_json(
        selection_path,
        {
            "selected_lr": selected,
            "inner_validation_means": {str(lr): value for lr, value in scored.items()},
            "identity_hash": identity_base.get("selection_hash"),
            "metric_source": "INNER_VALIDATION",
            "outer_test_used": False,
        },
        settings,
    )
    return selected


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
