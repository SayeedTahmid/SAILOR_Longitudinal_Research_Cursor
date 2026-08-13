"""Integrity audit for a staged or promoted SAILOR_READY package."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from sailor.config import Settings
from sailor.constants import QUARANTINE_NAMES
from sailor.guards import guard_g5_stage2
from sailor.packaging.common import (
    content_hash,
    read_json,
    safe_relative_path,
    sha256_file,
)


def _absolute_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            found.extend(_absolute_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_absolute_strings(child))
    elif isinstance(value, str):
        if Path(value).is_absolute() or "/content/drive/" in value or ":\\" in value:
            found.append(value)
    return found


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    return checksums


def verify_ready_package(
    data_root: Path,
    *,
    allow_pending: bool = False,
) -> dict[str, Any]:
    from sailor.data.ready import ReadyDataset

    root = data_root.resolve()
    failures: list[dict[str, Any]] = []
    required = [
        "manifests/package_manifest.json",
        "manifests/preprocessing_manifest.json",
        "manifests/longitudinal_windows.json",
        "manifests/folds.json",
        "metadata/treatment_manifest.json",
        "metadata/timing_cache.json",
        "metadata/session_metadata.csv",
        "README.md",
        "DATASET_VERSION.txt",
        "loader.py",
        "CHECKSUMS.sha256",
    ]
    for relative in required:
        if not (root / relative).is_file():
            failures.append({"type": "missing_required", "path": relative})
    if failures:
        return {"status": "FAIL", "failures": failures}

    package = read_json(root / "manifests/package_manifest.json")
    expected_status = package.get("package_status")
    if expected_status != "READY_TO_TRAIN" and not allow_pending:
        failures.append({"type": "package_status", "value": expected_status})
    if (
        package.get("data_version") != "v2.0"
        or package.get("preprocessing_version") != "p2.0"
    ):
        failures.append({"type": "version_mismatch"})
    for relative, expected in package.get("source_manifest_hashes", {}).items():
        try:
            path = root / safe_relative_path(relative)
        except ValueError:
            failures.append({"type": "unsafe_provenance_path", "path": relative})
            continue
        if not path.is_file() or sha256_file(path) != expected:
            failures.append({"type": "provenance_hash", "path": relative})
    for relative, expected in package.get("file_checksums", {}).items():
        try:
            path = root / safe_relative_path(relative)
        except ValueError:
            failures.append({"type": "unsafe_manifest_checksum_path", "path": relative})
            continue
        if not path.is_file() or sha256_file(path) != expected:
            failures.append({"type": "package_manifest_checksum", "path": relative})

    checksum_inventory = _read_checksums(root / "CHECKSUMS.sha256")
    for relative, expected in checksum_inventory.items():
        try:
            safe = safe_relative_path(relative)
        except ValueError:
            failures.append({"type": "unsafe_checksum_path", "path": relative})
            continue
        path = root / safe
        if not path.is_file():
            failures.append({"type": "checksum_file_missing", "path": relative})
        elif sha256_file(path) != expected:
            failures.append({"type": "checksum_mismatch", "path": relative})
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if set(checksum_inventory) != actual_files:
        failures.append(
            {
                "type": "checksum_inventory_coverage",
                "missing": sorted(actual_files - set(checksum_inventory)),
                "extra": sorted(set(checksum_inventory) - actual_files),
            }
        )

    preprocessing = read_json(root / "manifests/preprocessing_manifest.json")
    windows = read_json(root / "manifests/longitudinal_windows.json")
    folds = read_json(root / "manifests/folds.json")
    treatment = read_json(root / "metadata/treatment_manifest.json")
    timing = read_json(root / "metadata/timing_cache.json")

    operational = {
        "package": package,
        "preprocessing": preprocessing,
        "windows": windows,
        "folds": folds,
        "treatment": treatment,
        "timing": timing,
    }
    for name, payload in operational.items():
        absolute = _absolute_strings(payload)
        if absolute:
            failures.append(
                {"type": "absolute_operational_paths", "manifest": name, "paths": absolute[:20]}
            )

    if preprocessing.get("records_hash") != content_hash(
        preprocessing.get("records", [])
    ):
        failures.append({"type": "preprocessing_records_hash"})
    if windows.get("content_hash") != content_hash(windows.get("windows", [])):
        failures.append({"type": "windows_content_hash"})
    if folds.get("content_hash") != content_hash(folds.get("folds", [])):
        failures.append({"type": "folds_content_hash"})
    if treatment.get("content_hash") != content_hash(treatment.get("records", [])):
        failures.append({"type": "treatment_content_hash"})
    if timing.get("content_hash") != content_hash(timing.get("records", [])):
        failures.append({"type": "timing_content_hash"})

    if windows.get("preprocessing_manifest_sha256") != sha256_file(
        root / "manifests/preprocessing_manifest.json"
    ):
        failures.append({"type": "windows_preprocessing_parent"})
    if windows.get("treatment_manifest_sha256") != sha256_file(
        root / "metadata/treatment_manifest.json"
    ):
        failures.append({"type": "windows_treatment_parent"})
    if treatment.get("preprocessing_manifest_sha256") != sha256_file(
        root / "manifests/preprocessing_manifest.json"
    ):
        failures.append({"type": "treatment_preprocessing_parent"})
    if treatment.get("timing_cache_sha256") != sha256_file(
        root / "metadata/timing_cache.json"
    ):
        failures.append({"type": "treatment_timing_parent"})
    if folds.get("windows_manifest_sha256") != sha256_file(
        root / "manifests/longitudinal_windows.json"
    ):
        failures.append({"type": "folds_windows_parent"})

    session_keys: set[str] = set()
    for record in preprocessing.get("records", []):
        key = f"{record['subject']}/{record['mni_session']}"
        session_keys.add(key)
        try:
            image_path = root / safe_relative_path(record["mri_output"])
            mask_path = root / safe_relative_path(record["mask_output"])
        except ValueError as exc:
            failures.append({"type": "unsafe_array_path", "session": key, "error": str(exc)})
            continue
        if not image_path.is_file() or not mask_path.is_file():
            failures.append({"type": "array_missing", "session": key})
            continue
        image_checksum_ok = (
            sha256_file(image_path) == record["checksums"]["mri_sha256"]
        )
        mask_checksum_ok = (
            sha256_file(mask_path) == record["checksums"]["mask_sha256"]
        )
        if not image_checksum_ok:
            failures.append({"type": "image_checksum", "session": key})
        if not mask_checksum_ok:
            failures.append({"type": "mask_checksum", "session": key})
        if not image_checksum_ok or not mask_checksum_ok:
            continue
        try:
            image = np.load(image_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
        except (OSError, ValueError, EOFError) as exc:
            failures.append(
                {"type": "array_load_failure", "session": key, "error": str(exc)}
            )
            continue
        expected_shape = tuple(record["shape"])
        if image.shape != mask.shape or image.shape != expected_shape:
            failures.append({"type": "shape_mismatch", "session": key})
        if not np.isfinite(image).all() or not np.any(image):
            failures.append({"type": "invalid_image", "session": key})
        if not np.isin(np.unique(mask), (0, 1)).all() or not np.any(mask):
            failures.append({"type": "invalid_mask", "session": key})
        if not record.get("affine_hash"):
            failures.append({"type": "missing_affine_provenance", "session": key})

    for window in windows.get("windows", []):
        for session in [
            *window["history_mni_sessions"],
            window["target_mni_session"],
        ]:
            key = f"{window['subject']}/{session}"
            if key not in session_keys:
                failures.append(
                    {"type": "broken_window_reference", "window": window["window_id"], "session": key}
                )
    windows_per_patient = Counter(
        window["subject"] for window in windows.get("windows", [])
    )
    if dict(sorted(windows_per_patient.items())) != windows.get(
        "windows_per_patient"
    ):
        failures.append({"type": "windows_per_patient_mismatch"})
    for window in windows.get("windows", []):
        expected_weight = 1.0 / windows_per_patient[window["subject"]]
        if not np.isclose(window.get("patient_weight", -1.0), expected_weight):
            failures.append(
                {"type": "patient_weight", "window": window["window_id"]}
            )
    if len(windows_per_patient) != package.get("counts", {}).get("patients"):
        failures.append({"type": "patient_count_mismatch"})
    if sorted(windows_per_patient) != sorted(package.get("included_patients", [])):
        failures.append({"type": "included_patient_manifest_mismatch"})
    if folds.get("patient_window_counts") != dict(
        sorted(windows_per_patient.items())
    ):
        failures.append({"type": "fold_patient_window_counts"})
    if folds.get("n_patients") != len(windows_per_patient):
        failures.append({"type": "fold_patient_count"})

    settings = Settings.for_testing(root, root.parent / "_unused_legacy")
    leakage = guard_g5_stage2(settings, windows, folds, preprocessing)
    if leakage.status != "PASS":
        failures.append({"type": "G5", "details": leakage.details["failures"]})

    prohibited_suffixes = (
        ".nii",
        ".nii.gz",
        ".dcm",
        ".tar",
        ".tar.bz2",
        ".pt",
        ".pth",
        ".ckpt",
        ".npz",
        ".h5",
        ".hdf5",
    )
    prohibited_tokens = (
        "rawdata",
        "sourcedata",
        "tadiff",
        "checkpoint",
        "ckpt_",
        "00_quarantine",
    )
    prohibited: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().lower()
        if relative.startswith("manifests/provenance/"):
            continue
        if relative.endswith(prohibited_suffixes) or any(
            token in relative for token in prohibited_tokens
        ) or any(name.lower() in relative for name in QUARANTINE_NAMES):
            prohibited.append(relative)
    if prohibited:
        failures.append({"type": "prohibited_files", "paths": prohibited})

    try:
        ReadyDataset(root, allow_pending=allow_pending)
    except Exception as exc:
        failures.append({"type": "loader_validation", "error": str(exc)})

    counts = package.get("counts", {})
    if (
        len(preprocessing.get("records", [])) != counts.get("sessions")
        or len(windows.get("windows", [])) != counts.get("longitudinal_windows")
        or len(list(root.glob("images/T1c/sub-*/ses-*/T1c-icor.npy")))
        != counts.get("t1c_volumes")
        or len(
            list(
                root.glob(
                    "masks/CL/sub-*/ses-*/CL-enhancing-t1wc.npy"
                )
            )
        )
        != counts.get("cl_masks")
    ):
        failures.append({"type": "package_count_mismatch"})

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "counts": counts,
        "fold_hash": folds.get("content_hash"),
        "timing_provenance": timing.get("timing_provenance"),
        "package_status": package.get("package_status"),
    }
