"""Portable manifest and metadata generation without changing source artefacts."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sailor.packaging.common import content_hash, sha256_file, write_json


def _portable_array_path(record: dict[str, Any], kind: str) -> str:
    subject = record["subject"]
    session = record["mni_session"]
    if kind == "image":
        return f"images/T1c/{subject}/{session}/T1c-icor.npy"
    return f"masks/CL/{subject}/{session}/CL-enhancing-t1wc.npy"


def _write_session_metadata(
    path: Path,
    preprocessing: dict[str, Any],
    treatment: dict[str, Any],
    timing: dict[str, Any],
    windows: dict[str, Any],
) -> None:
    treatment_lookup = {
        f"{item['subject']}/{item['mni_session']}": item
        for item in treatment["records"]
    }
    timing_lookup = {
        f"{item['subject']}/{item['mni_session']}": item
        for item in timing["records"]
    }
    window_counts: dict[str, dict[str, int]] = {}
    for window in windows["windows"]:
        for session in window["history_mni_sessions"]:
            key = f"{window['subject']}/{session}"
            window_counts.setdefault(key, {"history_uses": 0, "target_uses": 0})
            window_counts[key]["history_uses"] += 1
        key = f"{window['subject']}/{window['target_mni_session']}"
        window_counts.setdefault(key, {"history_uses": 0, "target_uses": 0})
        window_counts[key]["target_uses"] += 1

    fields = [
        "subject",
        "raw_subject",
        "raw_session",
        "mni_session",
        "image_path",
        "mask_path",
        "shape",
        "spacing",
        "affine_hash",
        "mri_sha256",
        "mask_sha256",
        "mask_original_positive_value",
        "approximate_day",
        "interval_from_previous_days",
        "timing_provenance",
        "treatment_status",
        "treatment_missing",
        "dose_available",
        "history_uses",
        "target_uses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            preprocessing["records"],
            key=lambda item: (item["subject"], item["mni_session"]),
        ):
            key = f"{record['subject']}/{record['mni_session']}"
            treatment_item = treatment_lookup[key]
            timing_item = timing_lookup[key]
            uses = window_counts.get(key, {"history_uses": 0, "target_uses": 0})
            writer.writerow(
                {
                    "subject": record["subject"],
                    "raw_subject": record["raw_subject"],
                    "raw_session": record["raw_session"],
                    "mni_session": record["mni_session"],
                    "image_path": record["mri_output"],
                    "mask_path": record["mask_output"],
                    "shape": json.dumps(record["shape"], separators=(",", ":")),
                    "spacing": json.dumps(record["spacing"], separators=(",", ":")),
                    "affine_hash": record["affine_hash"],
                    "mri_sha256": record["checksums"]["mri_sha256"],
                    "mask_sha256": record["checksums"]["mask_sha256"],
                    "mask_original_positive_value": record[
                        "mask_original_positive_value"
                    ],
                    "approximate_day": timing_item["approximate_day"],
                    "interval_from_previous_days": (
                        ""
                        if timing_item["interval_from_previous_days"] is None
                        else timing_item["interval_from_previous_days"]
                    ),
                    "timing_provenance": timing_item["timing_provenance"],
                    "treatment_status": treatment_item["status"] or "",
                    "treatment_missing": str(treatment_item["missing"]).lower(),
                    "dose_available": str(
                        not treatment_item["dose_missing"]
                    ).lower(),
                    "history_uses": uses["history_uses"],
                    "target_uses": uses["target_uses"],
                }
            )


def write_portable_manifests(
    staging_root: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    source = audit["source_payloads"]
    preprocessing = deepcopy(source["preprocessing"])
    treatment = deepcopy(source["treatment"])
    timing = deepcopy(source["timing"])
    windows = deepcopy(source["windows"])
    folds = deepcopy(source["folds"])

    for record in preprocessing["records"]:
        record["mri_output"] = _portable_array_path(record, "image")
        record["mask_output"] = _portable_array_path(record, "mask")
    preprocessing["visual_qc_montage"] = (
        "QC/visualizations/v2_phase2_t1c_montage.pgm"
    )
    preprocessing["records_hash"] = content_hash(preprocessing["records"])
    preprocessing["package_path_policy"] = "relative_to_DATA_ROOT"
    preprocessing_path = staging_root / "manifests" / "preprocessing_manifest.json"
    write_json(preprocessing_path, preprocessing)

    timing_path = staging_root / "metadata" / "timing_cache.json"
    write_json(timing_path, timing)

    for record in treatment["records"]:
        source_dose_available = not record.get("dose_missing", True)
        record["dose_reference"] = None
        record["dose_missing"] = True
        record["source_dose_available"] = source_dose_available
        record["dose_in_package"] = False
    treatment["n_dose_missing"] = len(treatment["records"])
    treatment["preprocessing_manifest_sha256"] = sha256_file(preprocessing_path)
    treatment["timing_cache_sha256"] = sha256_file(timing_path)
    treatment["content_hash"] = content_hash(treatment["records"])
    treatment["package_scope"] = "metadata_only_no_dose_arrays"
    treatment_path = staging_root / "metadata" / "treatment_manifest.json"
    write_json(treatment_path, treatment)

    windows["preprocessing_manifest_sha256"] = sha256_file(preprocessing_path)
    windows["treatment_manifest_sha256"] = sha256_file(treatment_path)
    windows["package_path_policy"] = "session_keys_resolve_via_preprocessing_manifest"
    windows_path = staging_root / "manifests" / "longitudinal_windows.json"
    write_json(windows_path, windows)

    folds["windows_manifest_sha256"] = sha256_file(windows_path)
    folds_path = staging_root / "manifests" / "folds.json"
    write_json(folds_path, folds)

    session_metadata_path = staging_root / "metadata" / "session_metadata.csv"
    _write_session_metadata(
        session_metadata_path,
        preprocessing,
        treatment,
        timing,
        windows,
    )

    source_root = Path(audit["source_root"])
    section13 = json.loads(
        (
            source_root
            / "01_DATA_FOUNDATION"
            / "state"
            / "section_13_complete.json"
        ).read_text(encoding="utf-8")
    )
    package_manifest = {
        "package_kind": "SAILOR_READY",
        "package_status": "STAGED_PENDING_INTEGRITY_AUDIT",
        "authoritative_source_policy": (
            "Derived distribution copy; frozen Phase-2 source remains authoritative."
        ),
        "data_version": audit["data_version"],
        "preprocessing_version": audit["preprocessing_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": section13["git_commit"],
        "source_git_dirty": section13["git_dirty"],
        "counts": {
            "patients": audit["n_window_patients"],
            "sessions": audit["n_sessions"],
            "t1c_volumes": audit["n_images"],
            "cl_masks": audit["n_masks"],
            "longitudinal_windows": audit["n_windows"],
        },
        "included_patients": audit["included_patients"],
        "excluded_patients": audit["excluded_patients"],
        "excluded_sessions": audit["excluded_sessions"],
        "target": audit["target"],
        "normalization": audit["normalization"],
        "window_definition": {
            "minimum_history_scans": windows["minimum_history_scans"],
            "history_policy": windows["history_policy"],
            "timing_provenance": windows["timing_provenance"],
            "content_hash": windows["content_hash"],
        },
        "cv": {
            "fold_scheme": audit["fold_scheme"],
            "fold_hash": audit["fold_hash"],
            "master_seed": folds["master_seed"],
            "repeat_seeds": folds["repeat_seeds"],
            "outer_folds": folds["outer_folds"],
            "outer_repeats": folds["outer_repeats"],
            "inner_folds": folds["inner_folds"],
        },
        "leakage_qc": {
            "status": source["leakage"]["status"],
            "failed_guards": source["phase2_qc"]["failed_guards"],
        },
        "treatment_limitations": {
            **audit["treatment_summary"],
            "dose_arrays_included": False,
            "dose_registration_verified": False,
        },
        "scientific_scope": {
            "approved": [
                "persistence_baseline",
                "mri_history_only_baseline",
                "mri_plus_approximate_delta_t_baseline",
            ],
            "not_approved": [
                "final_treatment_aware_claims",
                "dose_aware_modeling",
                "exact_time_from_surgery_claims",
                "causal_treatment_effect_claims",
            ],
        },
        "relative_paths": {
            "preprocessing_manifest": "manifests/preprocessing_manifest.json",
            "windows": "manifests/longitudinal_windows.json",
            "folds": "manifests/folds.json",
            "session_metadata": "metadata/session_metadata.csv",
            "treatment": "metadata/treatment_manifest.json",
            "timing": "metadata/timing_cache.json",
            "qc_reports": "QC/reports",
            "qc_visualizations": "QC/visualizations",
        },
        "source_manifest_hashes": {
            item["destination"]: item["sha256"]
            for item in audit["copy_items"]
            if item["category"] == "provenance_manifest"
        },
        "file_checksums": {},
    }
    package_manifest_path = staging_root / "manifests" / "package_manifest.json"
    write_json(package_manifest_path, package_manifest)
    return {
        "preprocessing": preprocessing,
        "treatment": treatment,
        "timing": timing,
        "windows": windows,
        "folds": folds,
        "package_manifest": package_manifest,
        "paths": {
            "preprocessing": preprocessing_path,
            "treatment": treatment_path,
            "timing": timing_path,
            "windows": windows_path,
            "folds": folds_path,
            "session_metadata": session_metadata_path,
            "package_manifest": package_manifest_path,
        },
    }
