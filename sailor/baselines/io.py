"""Read-only access to frozen Phase-2 windows, folds, and arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sailor.config import Settings
from sailor.constants import (
    BASELINE_DT_SCALE_DAYS,
    BASELINE_HISTORY_SCANS,
    PREPROCESSING_VERSION,
)
from sailor.contracts import assert_mask_contract, assert_volume_contract
from sailor.errors import StopProtocolError


def phase2_paths(settings: Settings) -> dict[str, Path]:
    version = settings.preprocessing_version
    return {
        "preprocessing": settings.dataset_root
        / "02_PREPROCESSED_MRI"
        / version
        / "v2_preprocessing_manifest.json",
        "windows": settings.dataset_root
        / "04_LONGITUDINAL_WINDOWS"
        / version
        / "v2_windows_manifest.json",
        "cv": settings.dataset_root
        / "04_LONGITUDINAL_WINDOWS"
        / version
        / "v2_cv_manifest.json",
        "qc": settings.dataset_root / "06_QC_REPORTS" / "v2_phase2_qc_report.json",
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_array_path(raw: str, dataset_root: Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    relative = dataset_root / raw
    if relative.is_file():
        return relative
    raise StopProtocolError(
        f"Phase-2 array is missing: {raw}",
        "A baseline could train or score on an incomplete session.",
        "Restore the frozen p2.0 arrays before running Phase 3.",
    )


def load_phase2_artefacts(settings: Settings) -> dict[str, Any]:
    paths = phase2_paths(settings)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise StopProtocolError(
            f"Phase-2 artefacts are missing: {missing[:5]}",
            "Phase 3 cannot use frozen windows or folds.",
            "Complete Phase 2 on the authoritative DATASET_ROOT first.",
        )
    preprocessing = _read(paths["preprocessing"])
    windows = _read(paths["windows"])
    folds = _read(paths["cv"])
    if (
        preprocessing.get("preprocessing_version") != PREPROCESSING_VERSION
        or windows.get("timing_provenance") is None
        or folds.get("fold_scheme") != settings.fold_scheme
    ):
        raise StopProtocolError(
            "Phase-2 manifests do not match the locked Phase-3 inputs.",
            "Baseline results would not be attributable to p2.0.",
            "Restore the frozen p2.0 windows and CV manifests.",
        )
    sessions = {
        (record["subject"], record["mni_session"]): record
        for record in preprocessing["records"]
    }
    return {
        "preprocessing": preprocessing,
        "windows": windows,
        "folds": folds,
        "sessions": sessions,
        "paths": {key: str(path) for key, path in paths.items()},
    }


def history_sessions(window: dict[str, Any]) -> tuple[str, str]:
    sessions = window["history_mni_sessions"]
    if len(sessions) < BASELINE_HISTORY_SCANS:
        raise StopProtocolError(
            f"Window {window['window_id']} has fewer than two history scans.",
            "The locked Phase-3 baseline input is undefined.",
            "Rebuild Phase-2 windows with MIN_HISTORY_SCANS=2.",
        )
    return sessions[-BASELINE_HISTORY_SCANS], sessions[-1]


def load_mask(
    artefacts: dict[str, Any],
    subject: str,
    mni_session: str,
    *,
    dataset_root: Path,
) -> np.ndarray:
    record = artefacts["sessions"][(subject, mni_session)]
    mask = np.load(resolve_array_path(record["mask_output"], dataset_root), mmap_mode="r")
    array = np.asarray(mask)
    assert_mask_contract(array, name=f"{subject}/{mni_session}/mask")
    return array


def load_mri(
    artefacts: dict[str, Any],
    subject: str,
    mni_session: str,
    *,
    dataset_root: Path,
) -> np.ndarray:
    record = artefacts["sessions"][(subject, mni_session)]
    image = np.load(resolve_array_path(record["mri_output"], dataset_root), mmap_mode="r")
    array = np.asarray(image)
    assert_volume_contract(array, name=f"{subject}/{mni_session}/mri")
    return array


def encode_delta(days: float) -> float:
    return float(days) / BASELINE_DT_SCALE_DAYS


def windows_for_patients(
    windows: list[dict[str, Any]],
    patients: list[str],
) -> list[dict[str, Any]]:
    allowed = set(patients)
    selected = [window for window in windows if window["subject"] in allowed]
    extra = {window["subject"] for window in selected} - allowed
    if extra:
        raise StopProtocolError(
            f"Window selection leaked patients: {sorted(extra)}",
            "A fold could train or score on the wrong cohort.",
            "Filter windows by the frozen patient lists only.",
        )
    return selected
