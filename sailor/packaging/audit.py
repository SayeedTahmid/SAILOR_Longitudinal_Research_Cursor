"""Read-only audit and copy plan for frozen Phase-2 artefacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from sailor.constants import QUARANTINE_NAMES
from sailor.errors import StopProtocolError
from sailor.packaging.common import content_hash, read_json, sha256_file

SOURCE_RELATIVE_PATHS = {
    "phase1_dataset": "01_DATA_FOUNDATION/v2_dataset_manifest.json",
    "phase1_canonical": "01_DATA_FOUNDATION/v2_canonical_manifest.json",
    "phase1_qc": "06_QC_REPORTS/v2_stage1_qc_report.json",
    "preprocessing": "02_PREPROCESSED_MRI/p2.0/v2_preprocessing_manifest.json",
    "windows": "04_LONGITUDINAL_WINDOWS/p2.0/v2_windows_manifest.json",
    "folds": "04_LONGITUDINAL_WINDOWS/p2.0/v2_cv_manifest.json",
    "treatment": "05_TREATMENT_DATA/p2.0/v2_treatment_manifest.json",
    "timing": "05_TREATMENT_DATA/p2.0/v2_canonical_timing_cache.json",
    "phase2_qc": "06_QC_REPORTS/v2_phase2_qc_report.json",
    "leakage": "06_QC_REPORTS/v2_phase2_leakage_report.json",
    "montage": "06_QC_REPORTS/v2_phase2_t1c_montage.pgm",
    "overlay": "06_QC_REPORTS/v2_phase2_t1c_cl_overlay.png",
}


def _absolute_strings(value: Any, location: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_absolute_strings(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_absolute_strings(child, f"{location}[{index}]"))
    elif isinstance(value, str):
        path = Path(value)
        if path.is_absolute() or "/content/drive/" in value or ":\\" in value:
            found.append((location, value))
    return found


def _portable_array_path(record: dict[str, Any], kind: str) -> str:
    subject = record["subject"]
    session = record["mni_session"]
    if kind == "image":
        return f"images/T1c/{subject}/{session}/T1c-icor.npy"
    return f"masks/CL/{subject}/{session}/CL-enhancing-t1wc.npy"


def audit_frozen_source(
    source_root: Path,
    destination_root: Path,
    *,
    verify_array_hashes: bool = True,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if not source_root.is_dir():
        raise StopProtocolError(
            f"Frozen source root is missing: {source_root}",
            "No authoritative Phase-2 artefact can be packaged.",
            "Mount the correct Drive and restore the frozen source root.",
        )
    if destination_root.exists():
        raise StopProtocolError(
            f"Destination already exists: {destination_root}",
            "The no-overwrite distribution rule would be violated.",
            "Choose a new versioned destination or explicitly audit the existing folder.",
        )
    try:
        destination_root.relative_to(source_root)
        overlaps = True
    except ValueError:
        try:
            source_root.relative_to(destination_root)
            overlaps = True
        except ValueError:
            overlaps = False
    if overlaps:
        raise StopProtocolError(
            "Destination and authoritative source paths overlap.",
            "Packaging could add to, replace, or reorganize the frozen Phase-2 tree.",
            "Use a separate sibling destination outside the source root.",
        )

    source_paths = {
        key: source_root / relative
        for key, relative in SOURCE_RELATIVE_PATHS.items()
    }
    source_paths.update(
        {
            f"section_{section:02d}": source_root
            / "01_DATA_FOUNDATION"
            / "state"
            / f"section_{section:02d}_complete.json"
            for section in range(1, 14)
        }
    )
    missing = [key for key, path in source_paths.items() if not path.is_file()]
    if missing:
        raise StopProtocolError(
            f"Required frozen artefacts are missing: {missing}",
            "The package would be incomplete or lose provenance.",
            "Restore the exact artefacts; do not fabricate replacements.",
        )

    payloads = {
        key: read_json(source_paths[key])
        for key in (
            "phase1_dataset",
            "phase1_canonical",
            "phase1_qc",
            "preprocessing",
            "windows",
            "folds",
            "treatment",
            "timing",
            "phase2_qc",
            "leakage",
        )
    }
    preprocessing = payloads["preprocessing"]
    windows = payloads["windows"]
    folds = payloads["folds"]
    treatment = payloads["treatment"]
    timing = payloads["timing"]

    source_hash_failures: list[str] = []
    if preprocessing.get("records_hash") != content_hash(
        preprocessing.get("records", [])
    ):
        source_hash_failures.append("preprocessing.records_hash")
    if preprocessing.get("phase1_dataset_manifest_sha256") != sha256_file(
        source_paths["phase1_dataset"]
    ):
        source_hash_failures.append("preprocessing.phase1_dataset_manifest_sha256")
    if preprocessing.get("phase1_qc_report_sha256") != sha256_file(
        source_paths["phase1_qc"]
    ):
        source_hash_failures.append("preprocessing.phase1_qc_report_sha256")
    if preprocessing.get("phase1_canonical_manifest_sha256") != sha256_file(
        source_paths["phase1_canonical"]
    ):
        source_hash_failures.append("preprocessing.phase1_canonical_manifest_sha256")
    if treatment.get("content_hash") != content_hash(treatment.get("records", [])):
        source_hash_failures.append("treatment.content_hash")
    if treatment.get("preprocessing_manifest_sha256") != sha256_file(
        source_paths["preprocessing"]
    ):
        source_hash_failures.append("treatment.preprocessing_manifest_sha256")
    if treatment.get("phase1_dataset_manifest_sha256") != sha256_file(
        source_paths["phase1_dataset"]
    ):
        source_hash_failures.append("treatment.phase1_dataset_manifest_sha256")
    if treatment.get("timing_cache_sha256") != sha256_file(source_paths["timing"]):
        source_hash_failures.append("treatment.timing_cache_sha256")
    if windows.get("content_hash") != content_hash(windows.get("windows", [])):
        source_hash_failures.append("windows.content_hash")
    if windows.get("preprocessing_manifest_sha256") != sha256_file(
        source_paths["preprocessing"]
    ):
        source_hash_failures.append("windows.preprocessing_manifest_sha256")
    if windows.get("phase1_dataset_manifest_sha256") != sha256_file(
        source_paths["phase1_dataset"]
    ):
        source_hash_failures.append("windows.phase1_dataset_manifest_sha256")
    if windows.get("treatment_manifest_sha256") != sha256_file(
        source_paths["treatment"]
    ):
        source_hash_failures.append("windows.treatment_manifest_sha256")
    if folds.get("content_hash") != content_hash(folds.get("folds", [])):
        source_hash_failures.append("folds.content_hash")
    if folds.get("windows_manifest_sha256") != sha256_file(source_paths["windows"]):
        source_hash_failures.append("folds.windows_manifest_sha256")
    if timing.get("content_hash") != content_hash(timing.get("records", [])):
        source_hash_failures.append("timing.content_hash")
    if timing.get("phase1_dataset_manifest_sha256") != sha256_file(
        source_paths["phase1_dataset"]
    ):
        source_hash_failures.append("timing.phase1_dataset_manifest_sha256")
    if timing.get("phase1_canonical_manifest_sha256") != sha256_file(
        source_paths["phase1_canonical"]
    ):
        source_hash_failures.append("timing.phase1_canonical_manifest_sha256")
    if source_hash_failures:
        raise StopProtocolError(
            f"Frozen source hash chain is invalid: {source_hash_failures}",
            "Portable rewriting would conceal stale or mismatched parent artefacts.",
            "Restore the exact completed Phase-2 hash chain before packaging.",
        )

    expected_locks = {
        "data_version": "v2.0",
        "preprocessing_version": "p2.0",
    }
    lock_failures: dict[str, Any] = {}
    for key, expected in expected_locks.items():
        for name in ("preprocessing", "phase2_qc", "timing"):
            if payloads[name].get(key) != expected:
                lock_failures[f"{name}.{key}"] = payloads[name].get(key)
    if (
        preprocessing.get("PRIMARY_TARGET_MASK") != "CL"
        or preprocessing.get("PRIMARY_TARGET_COMPONENT") != "enhancing_t1wc"
        or preprocessing.get("selected_sequence") != "T1c-icor"
        or payloads["leakage"].get("status") != "PASS"
        or payloads["phase2_qc"].get("failed_guards")
    ):
        lock_failures["scientific_locks"] = "target/input/leakage mismatch"
    if payloads["phase1_qc"].get("failed_guards"):
        lock_failures["phase1_failed_guards"] = payloads["phase1_qc"][
            "failed_guards"
        ]
    invalid_completions: list[dict[str, Any]] = []
    for section in range(1, 14):
        record = read_json(source_paths[f"section_{section:02d}"])
        if (
            record.get("status") != "complete"
            or record.get("guards_failed")
            or record.get("git_dirty") is not False
        ):
            invalid_completions.append(
                {
                    "section": section,
                    "status": record.get("status"),
                    "guards_failed": record.get("guards_failed"),
                    "git_dirty": record.get("git_dirty"),
                }
            )
    if invalid_completions:
        lock_failures["completion_records"] = invalid_completions
    if lock_failures:
        raise StopProtocolError(
            f"Frozen locks are inconsistent: {lock_failures}",
            "The distribution could misstate its approved scientific scope.",
            "Restore the completed Phase-2 manifests.",
        )

    copy_items: list[dict[str, Any]] = []
    source_array_paths: set[Path] = set()
    session_keys: set[str] = set()
    for record in preprocessing.get("records", []):
        image = Path(record["mri_output"])
        mask = Path(record["mask_output"])
        session_key = f"{record['subject']}/{record['mni_session']}"
        if session_key in session_keys:
            raise StopProtocolError(
                f"Duplicate preprocessing session: {session_key}",
                "Portable paths would collide.",
                "Restore the frozen preprocessing manifest.",
            )
        session_keys.add(session_key)
        for kind, path, checksum_field in (
            ("image", image, "mri_sha256"),
            ("mask", mask, "mask_sha256"),
        ):
            approved_root = (
                source_root / "02_PREPROCESSED_MRI" / "p2.0"
                if kind == "image"
                else source_root / "03_TUMOR_MASKS" / "p2.0"
            ).resolve()
            try:
                path.resolve().relative_to(approved_root)
            except ValueError as exc:
                raise StopProtocolError(
                    f"Manifest {kind} path is outside its approved frozen root: {path}",
                    "A quarantined or unrelated file could be disguised in the package.",
                    "Restore the frozen preprocessing manifest.",
                ) from exc
            if any(token.lower() in str(path).lower() for token in QUARANTINE_NAMES):
                raise StopProtocolError(
                    f"Quarantined path entered preprocessing manifest: {path}",
                    "The package would mix prior experimental artefacts.",
                    "Restore the frozen preprocessing manifest.",
                )
            if not path.is_file():
                raise StopProtocolError(
                    f"Manifest-referenced {kind} is missing: {path}",
                    "The package would contain a broken session.",
                    "Restore the frozen array; do not regenerate silently.",
                )
            expected = record["checksums"][checksum_field]
            actual = sha256_file(path) if verify_array_hashes else expected
            if actual != expected:
                raise StopProtocolError(
                    f"Frozen {kind} checksum mismatch: {path}",
                    "The authoritative Phase-2 byte content changed.",
                    "Stop and investigate the source artefact.",
                )
            source_array_paths.add(path.resolve())
            copy_items.append(
                {
                    "category": kind,
                    "source": str(path.resolve()),
                    "destination": _portable_array_path(record, kind),
                    "bytes": path.stat().st_size,
                    "sha256": actual,
                }
            )

    actual_arrays = {
        path.resolve()
        for root in (
            source_root / "02_PREPROCESSED_MRI" / "p2.0",
            source_root / "03_TUMOR_MASKS" / "p2.0",
        )
        for path in root.rglob("*.npy")
    }
    unexpected_arrays = sorted(str(path) for path in actual_arrays - source_array_paths)
    if unexpected_arrays:
        raise StopProtocolError(
            f"Unexpected NPY files exist in frozen roots: {unexpected_arrays[:20]}",
            "The package boundary is ambiguous.",
            "Audit the extra files; do not copy or delete them silently.",
        )

    broken_windows: list[dict[str, str]] = []
    for window in windows.get("windows", []):
        for session in [
            *window.get("history_mni_sessions", []),
            window["target_mni_session"],
        ]:
            key = f"{window['subject']}/{session}"
            if key not in session_keys:
                broken_windows.append({"window": window["window_id"], "session": key})
    window_patients = set(windows.get("windows_per_patient", {}))
    fold_patients = set(folds.get("patient_window_counts", {}))
    if broken_windows or window_patients != fold_patients:
        raise StopProtocolError(
            "Window or fold references do not match preprocessed sessions.",
            "Training examples or evaluation groups would be broken.",
            f"Broken windows: {broken_windows[:10]}; patient difference: "
            f"{sorted(window_patients ^ fold_patients)}",
        )

    all_patients = set(payloads["phase1_dataset"]["overview"]["patients"])
    preprocessed_patients = {record["subject"] for record in preprocessing["records"]}
    excluded_patients = [
        {
            "patient_id": patient,
            "reason": (
                "no_phase1_eligible_sessions"
                if patient not in preprocessed_patients
                else "insufficient_history_for_two_plus_one_window"
            ),
        }
        for patient in sorted(all_patients - window_patients)
    ]
    excluded_sessions = payloads["phase1_dataset"]["target"].get(
        "excluded_primary_files", []
    )

    provenance_destinations = {
        "phase1_dataset": "manifests/provenance/v2_dataset_manifest.source.json",
        "phase1_canonical": "manifests/provenance/v2_canonical_manifest.source.json",
        "phase1_qc": "manifests/provenance/v2_stage1_qc_report.source.json",
        "preprocessing": "manifests/provenance/v2_preprocessing_manifest.source.json",
        "windows": "manifests/provenance/v2_windows_manifest.source.json",
        "folds": "manifests/provenance/v2_cv_manifest.source.json",
        "treatment": "manifests/provenance/v2_treatment_manifest.source.json",
        "timing": "manifests/provenance/v2_timing_cache.source.json",
        "phase2_qc": "manifests/provenance/v2_phase2_qc_report.source.json",
        "leakage": "manifests/provenance/v2_phase2_leakage_report.source.json",
    }
    for key, destination in provenance_destinations.items():
        path = source_paths[key]
        copy_items.append(
            {
                "category": "provenance_manifest",
                "source": str(path),
                "destination": destination,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for section in range(1, 14):
        key = f"section_{section:02d}"
        path = source_paths[key]
        copy_items.append(
            {
                "category": "completion_record",
                "source": str(path),
                "destination": f"manifests/provenance/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for key, destination in (
        ("phase2_qc", "QC/reports/v2_phase2_qc_report.json"),
        ("leakage", "QC/reports/v2_phase2_leakage_report.json"),
        ("montage", "QC/visualizations/v2_phase2_t1c_montage.pgm"),
        ("overlay", "QC/visualizations/v2_phase2_t1c_cl_overlay.png"),
    ):
        path = source_paths[key]
        copy_items.append(
            {
                "category": "qc",
                "source": str(path),
                "destination": destination,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    absolute_risks = {
        name: _absolute_strings(payload)
        for name, payload in payloads.items()
    }
    return {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "destination_exists": False,
        "data_version": "v2.0",
        "preprocessing_version": "p2.0",
        "n_sessions": preprocessing["n_records"],
        "n_images": sum(item["category"] == "image" for item in copy_items),
        "n_masks": sum(item["category"] == "mask" for item in copy_items),
        "n_windows": windows["n_windows"],
        "n_window_patients": windows["n_patients"],
        "included_patients": sorted(window_patients),
        "excluded_patients": excluded_patients,
        "excluded_sessions": excluded_sessions,
        "fold_scheme": folds["fold_scheme"],
        "fold_hash": folds["content_hash"],
        "timing_provenance": timing["timing_provenance"],
        "normalization": {
            "scope": preprocessing["normalization_scope"],
            "selected_sequence": preprocessing["selected_sequence"],
            "description": (
                "Per-volume brain-support robust scaling: clip 0.5/99.5 percentiles, "
                "subtract median, divide by IQR; no cohort statistics."
            ),
        },
        "target": {
            "mask": preprocessing["PRIMARY_TARGET_MASK"],
            "component": preprocessing["PRIMARY_TARGET_COMPONENT"],
        },
        "treatment_summary": {
            "records": treatment["n_records"],
            "missing": treatment["n_missing"],
            "dose_missing": treatment["n_dose_missing"],
            "status_counts": dict(
                Counter(
                    "MISSING" if item["missing"] else item["status"]
                    for item in treatment["records"]
                )
            ),
        },
        "absolute_path_risks": {
            name: {"count": len(matches), "samples": matches[:10]}
            for name, matches in absolute_risks.items()
        },
        "copy_items": copy_items,
        "copy_bytes": sum(item["bytes"] for item in copy_items),
        "source_payloads": payloads,
    }
