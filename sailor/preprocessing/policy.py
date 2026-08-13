"""Measured Phase-2 input selection and geometry planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from sailor.config import Settings
from sailor.errors import StopProtocolError


def _basename(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name


def _affine_hash(affine: list[list[float]] | tuple[tuple[float, ...], ...]) -> str:
    normalized = json.dumps(affine, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _same_geometry(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return tuple(first["shape"]) == tuple(second["shape"]) and np.allclose(
        np.asarray(first["affine"], dtype=float),
        np.asarray(second["affine"], dtype=float),
        rtol=0.0,
        atol=1e-5,
    )


def build_preprocessing_plan(
    manifest: dict[str, Any],
    qc_report: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    locked_headers = {
        "data_version": settings.data_version,
        "implementation_id": settings.implementation_id,
        "PRIMARY_TARGET_MASK": settings.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": settings.primary_target_component,
    }
    mismatches = {
        key: {"manifest": manifest.get(key), "qc": qc_report.get(key), "expected": value}
        for key, value in locked_headers.items()
        if manifest.get(key) != value or qc_report.get(key) != value
    }
    if mismatches:
        raise StopProtocolError(
            f"Phase 1 manifest/QC headers do not match Phase 2 locks: {mismatches}",
            "Preprocessing could combine incompatible provenance or targets.",
            "Restore a matching Phase 1 manifest and QC report.",
        )
    if qc_report.get("failed_guards"):
        raise StopProtocolError(
            f"Phase 1 has failed guards: {qc_report['failed_guards']}",
            "Phase 2 inputs are not scientifically valid.",
            "Resolve Phase 1 guard failures before preprocessing.",
        )

    guards = {item["guard_id"]: item for item in qc_report.get("guards", [])}
    g9 = guards.get("G9", {})
    pairs = g9.get("details", {}).get("surviving_raw_mni_pairs", [])
    records = manifest.get("inventory", {}).get("records", [])
    by_session: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("subject") and record.get("session"):
            key = f"{record['subject']}/{record['session']}"
            by_session.setdefault(key, []).append(record)

    variant_report: dict[str, dict[str, Any]] = {}
    for variant in ("T1c", "T1c-icor", "T1c-icor-zscore"):
        matching = [
            record
            for record in records
            if record.get("classification") == "MRI"
            and record.get("sequence") == variant
        ]
        variant_report[variant] = {
            "n_files": len(matching),
            "n_finite": sum(record.get("finite") is True for record in matching),
            "geometry_count": len(
                {
                    (
                        tuple(record["shape"]),
                        tuple(record["spacing"]),
                        _affine_hash(record["affine"]),
                    )
                    for record in matching
                }
            ),
        }

    selected: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    planned_output_bytes = 0
    planned_staging_bytes = 0
    for pair in pairs:
        mni_key = pair["mni"]
        raw_subject, raw_session = pair["raw"].split("/", 1)
        session_records = by_session.get(mni_key, [])
        mri = [
            item
            for item in session_records
            if item.get("classification") == "MRI"
            and item.get("sequence") == settings.primary_input_sequence
        ]
        masks = [
            item
            for item in session_records
            if item.get("classification") == "CL:enhancing_t1wc"
            and item.get("nonzero_voxels", 0) > 0
        ]
        brains = [
            item
            for item in session_records
            if _basename(item["path"]) in {"BrainExtractionMask.nii", "BrainExtractionMask.nii.gz"}
        ]
        if len(mri) != 1 or len(masks) != 1 or len(brains) != 1:
            issues.append(
                {
                    "raw": pair["raw"],
                    "mni": mni_key,
                    "reason": "required_file_count",
                    "counts": {"mri": len(mri), "mask": len(masks), "brain": len(brains)},
                }
            )
            continue
        if any(
            not str(record.get("source", "")).startswith(
                "archive:derivatives.tar.bz2"
            )
            for record in (mri[0], masks[0], brains[0])
        ):
            issues.append({"mni": mni_key, "reason": "unsupported_source_mode"})
            continue
        if mri[0].get("finite") is not True:
            issues.append({"mni": mni_key, "reason": "nonfinite_primary_mri"})
            continue
        expected_mask_scale = float(masks[0].get("maximum", float("nan")))
        if (
            masks[0].get("finite") is not True
            or masks[0].get("minimum") not in {0, 0.0}
            or not np.isfinite(expected_mask_scale)
            or expected_mask_scale <= 0.0
            or brains[0].get("finite") is not True
            or float(brains[0].get("minimum", float("nan"))) < 0.0
            or float(brains[0].get("maximum", float("nan"))) > 1.0
            or not np.isclose(
                float(brains[0].get("maximum", float("nan"))),
                1.0,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            issues.append({"mni": mni_key, "reason": "nonbinary_mask_values"})
            continue
        if not _same_geometry(mri[0], masks[0]) or not _same_geometry(mri[0], brains[0]):
            issues.append({"mni": mni_key, "reason": "geometry_mismatch"})
            continue
        voxels = int(np.prod(mri[0]["shape"]))
        planned_output_bytes += voxels * (np.dtype(np.float32).itemsize + np.dtype(np.uint8).itemsize)
        planned_staging_bytes += voxels * (
            np.dtype(mri[0]["dtype"]).itemsize
            + np.dtype(masks[0]["dtype"]).itemsize
            + np.dtype(brains[0]["dtype"]).itemsize
        )
        selected.append(
            {
                "subject": mri[0]["subject"],
                "raw_subject": raw_subject,
                "raw_session": raw_session,
                "mni_session": mri[0]["session"],
                "mri_source": mri[0]["path"],
                "mask_source": masks[0]["path"],
                "brain_mask_source": brains[0]["path"],
                "shape": mri[0]["shape"],
                "spacing": mri[0]["spacing"],
                "affine": mri[0]["affine"],
                "affine_hash": _affine_hash(mri[0]["affine"]),
                "mri_dtype": mri[0]["dtype"],
                "mask_dtype": masks[0]["dtype"],
                "brain_mask_dtype": brains[0]["dtype"],
                "brain_mask_threshold": 0.5,
                "brain_mask_policy": "finite_0_1_support_threshold",
                "mask_positive_scale": expected_mask_scale,
                "mask_scale_policy": "relative_gap_checked_at_runtime",
            }
        )

    return {
        "data_version": settings.data_version,
        "preprocessing_version": settings.preprocessing_version,
        "implementation_id": settings.implementation_id,
        "PRIMARY_TARGET_MASK": settings.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": settings.primary_target_component,
        "selected_sequence": settings.primary_input_sequence,
        "variant_report": variant_report,
        "n_phase1_eligible_pairs": len(pairs),
        "n_selected_sessions": len(selected),
        "planned_output_bytes": planned_output_bytes,
        "planned_staging_bytes": planned_staging_bytes,
        "selected": selected,
        "issues": issues,
        "optional_modalities": "BLOCKED",
        "visual_qc_required": True,
        "dose_reference_summary": {
            "n_files": manifest.get("dose", {}).get("n_files", 0),
            "n_patients": manifest.get("dose", {}).get("n_patients", 0),
            "space_status": manifest.get("dose", {}).get("space_status", "UNVERIFIED"),
            "requires_registration": manifest.get("dose", {}).get(
                "requires_registration", "UNVERIFIED"
            ),
            "primary_cohort_requirement": False,
        },
    }


def assert_plan_ready(plan: dict[str, Any]) -> None:
    if plan.get("issues"):
        raise StopProtocolError(
            f"Phase 2 selection found {len(plan['issues'])} session issues.",
            "The approved T1c input path is incomplete or geometrically inconsistent.",
            "Inspect the dry-run report and request approval before changing the input policy.",
        )
    if plan.get("n_selected_sessions") != plan.get("n_phase1_eligible_pairs"):
        raise StopProtocolError(
            "Phase 2 selected-session count differs from the Phase 1 eligible cohort.",
            "Preprocessing would silently change the primary cohort.",
            "Resolve the selection discrepancy before extraction.",
        )
