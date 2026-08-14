"""Atomic, identity-checked Phase-3 checkpoints on DATASET_ROOT."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.constants import DATA_VERSION, MODEL_VERSION, PREPROCESSING_VERSION
from sailor.errors import StopProtocolError
from sailor.paths import assert_writable_target


LOCKED_IDENTITY_FIELDS = (
    "mode",
    "split_role",
    "repeat",
    "outer_fold",
    "inner_fold",
    "seed",
    "learning_rate",
    "epochs",
    "patch_size",
    "model_version",
    "data_version",
    "preprocessing_version",
    "fold_scheme",
    "train_patients",
    "validation_patients",
    "test_patients",
)


def identity_hash(identity: dict[str, Any]) -> str:
    payload = {key: identity.get(key) for key in LOCKED_IDENTITY_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def checkpoint_root(settings: Settings, mode: str, repeat: int, outer_fold: int) -> Path:
    safe_mode = mode.replace("C-1", "Cminus1")
    return (
        settings.dataset_root
        / "CHECKPOINTS"
        / "p3.0"
        / safe_mode
        / f"repeat{repeat}_outer{outer_fold}"
    )


def inner_run_dir(fold_root: Path, inner_fold: int, learning_rate: float) -> Path:
    lr_token = f"{learning_rate:.0e}".replace("+", "")
    return fold_root / "inner" / f"inner{inner_fold}_lr{lr_token}"


def outer_run_dir(fold_root: Path) -> Path:
    return fold_root / "outer"


def build_identity(
    *,
    mode: str,
    split_role: str,
    repeat: int,
    outer_fold: int,
    inner_fold: int | None,
    seed: int,
    learning_rate: float,
    epochs: int,
    patch_size: int,
    fold_scheme: str,
    train_patients: list[str],
    validation_patients: list[str] | None,
    test_patients: list[str],
) -> dict[str, Any]:
    if split_role not in {"INNER_TRAINING", "OUTER_TRAINING"}:
        raise StopProtocolError(
            f"Invalid training split_role {split_role}.",
            "Checkpoint identity would mix training and evaluation roles.",
            "Use INNER_TRAINING or OUTER_TRAINING only.",
        )
    identity = {
        "mode": mode,
        "split_role": split_role,
        "repeat": int(repeat),
        "outer_fold": int(outer_fold),
        "inner_fold": inner_fold,
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "patch_size": int(patch_size),
        "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_scheme": fold_scheme,
        "train_patients": sorted(train_patients),
        "validation_patients": sorted(validation_patients or []),
        "test_patients": sorted(test_patients),
    }
    identity["identity_hash"] = identity_hash(identity)
    return identity


def atomic_torch_save(payload: dict[str, Any], path: Path, settings: Settings) -> None:
    from sailor.models.unet3d import require_torch

    torch, _ = require_torch()
    assert_writable_target(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".build")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    if temporary.exists():
        try:
            temporary.unlink()
        except OSError:
            pass


def resolve_resume_checkpoint(run_dir: Path, epochs: int) -> Path | None:
    latest = run_dir / "latest.pt"
    final = run_dir / "final.pt"
    best = run_dir / "best.pt"
    if latest.is_file():
        return latest
    if final.is_file():
        payload = load_checkpoint(final)
        completed = int(payload["epoch"])
        if completed == int(epochs) - 1:
            return final
        raise StopProtocolError(
            f"final.pt exists at epoch {completed} without latest.pt.",
            "The last completed epoch file is missing, so resume would be ambiguous.",
            "Restore latest.pt rather than guessing which epoch to continue from.",
        )
    if best.is_file():
        raise StopProtocolError(
            "best.pt exists without latest.pt.",
            "Resuming from best would silently skip or redo epochs.",
            "Restore latest.pt from Drive; do not improvise a resume point.",
        )
    return None


def load_checkpoint(path: Path) -> dict[str, Any]:
    from sailor.models.unet3d import require_torch

    torch, _ = require_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise StopProtocolError(
            f"Checkpoint is unreadable: {path}",
            "Resuming from a corrupted file could silently change the experiment.",
            "Delete only the corrupted .build file if present; do not improvise weights.",
        ) from exc
    if not isinstance(payload, dict) or "identity" not in payload:
        raise StopProtocolError(
            f"Checkpoint schema is invalid: {path}",
            "An incompatible file cannot be resumed.",
            "Inspect the checkpoint directory and stop this run.",
        )
    return payload


def validate_checkpoint(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    identity = payload.get("identity", {})
    if identity.get("identity_hash") != expected.get("identity_hash"):
        mismatches = [
            field
            for field in LOCKED_IDENTITY_FIELDS
            if identity.get(field) != expected.get(field)
        ]
        raise StopProtocolError(
            f"Checkpoint identity mismatch: {mismatches or ['identity_hash']}",
            "Loading it would mix two experiments into one fold.",
            "Start a new checkpoint directory or restore the matching run.",
        )
    if payload.get("model_version") != MODEL_VERSION:
        raise StopProtocolError(
            "Checkpoint model_version does not match b3.0.",
            "Resuming would change the locked architecture.",
            "Do not load this checkpoint.",
        )
    if payload.get("data_version") not in {None, DATA_VERSION}:
        raise StopProtocolError(
            f"Checkpoint data_version {payload.get('data_version')} does not match {DATA_VERSION}.",
            "Resuming would mix two dataset versions.",
            "Do not load this checkpoint.",
        )
    if payload.get("preprocessing_version") not in {None, PREPROCESSING_VERSION}:
        raise StopProtocolError(
            f"Checkpoint preprocessing_version {payload.get('preprocessing_version')} "
            f"does not match {PREPROCESSING_VERSION}.",
            "Resuming would mix two preprocessing versions.",
            "Do not load this checkpoint.",
        )


def save_epoch_checkpoints(
    *,
    run_dir: Path,
    settings: Settings,
    payload: dict[str, Any],
    is_best: bool,
    is_final: bool,
) -> None:
    atomic_torch_save(payload, run_dir / "latest.pt", settings)
    if is_best:
        atomic_torch_save(payload, run_dir / "best.pt", settings)
    if is_final:
        atomic_torch_save(payload, run_dir / "final.pt", settings)
