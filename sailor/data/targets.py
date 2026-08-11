"""Locked target and dose-map inventory."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sailor.schemas import NiftiRecord


def mask_degeneracy_reason(record: NiftiRecord) -> str | None:
    if record.finite is not True:
        return "NONFINITE"
    if record.nonzero_voxels is None or record.total_voxels is None:
        return "UNMEASURED"
    if record.nonzero_voxels == 0:
        return "ALL_ZERO"
    if record.nonzero_voxels == record.total_voxels:
        return "ALL_ONE"
    if record.nonzero_voxels <= 10 or (
        record.nonzero_voxels / record.total_voxels <= 1e-6
    ):
        return "NEAR_EMPTY"
    return None


def target_inventory(records: list[NiftiRecord]) -> dict[str, Any]:
    by_classification: dict[str, list[NiftiRecord]] = defaultdict(list)
    for record in records:
        by_classification[record.classification].append(record)

    primary = by_classification["CL:enhancing_t1wc"]
    unresolved = by_classification["CL:unresolved_component"]
    excluded_primary = [
        {"path": record.path, "reason": reason}
        for record in primary
        if (reason := mask_degeneracy_reason(record))
    ]
    valid_primary = [
        record for record in primary if mask_degeneracy_reason(record) is None
    ]
    primary_sessions = sorted(
        {
            (record.subject, record.session)
            for record in valid_primary
            if record.subject and record.session
        }
    )
    return {
        "PRIMARY_TARGET_MASK": "CL",
        "PRIMARY_TARGET_COMPONENT": "enhancing_t1wc",
        "resolved_on_disk_as": sorted({record.path for record in primary}),
        "n_primary_files": len(primary),
        "n_valid_primary_files": len(valid_primary),
        "n_excluded_primary_files": len(excluded_primary),
        "excluded_primary_files": excluded_primary,
        "n_primary_patient_sessions": len(primary_sessions),
        "primary_patient_sessions": [list(item) for item in primary_sessions],
        "unresolved_cl_candidates": [record.path for record in unresolved],
        "inventory_only": {
            "CL:t2wflair_hyperintensity": len(
                by_classification["CL:t2wflair_hyperintensity"]
            ),
            "ONCO": len(by_classification["ONCO"]),
        },
        "cohort_selection_source": (
            "non-degenerate CL:enhancing_t1wc only; excluded labels remain reported"
        ),
    }


def dose_inventory(records: list[NiftiRecord]) -> dict[str, Any]:
    dose_records = [record for record in records if record.classification == "DOSE"]
    mri_geometries = {
        (
            record.shape,
            tuple(round(value, 5) for value in record.spacing),
            tuple(tuple(round(value, 5) for value in row) for row in record.affine),
        )
        for record in records
        if record.classification == "MRI"
    }
    patients = sorted({record.subject for record in dose_records if record.subject})
    geometries = sorted(
        {
            (record.shape, tuple(round(value, 5) for value in record.spacing))
            for record in dose_records
        }
    )
    grid_matches_mri = bool(dose_records) and all(
        (
            record.shape,
            tuple(round(value, 5) for value in record.spacing),
            tuple(tuple(round(value, 5) for value in row) for row in record.affine),
        )
        in mri_geometries
        for record in dose_records
    )
    isotropic_1mm = bool(dose_records) and all(
        all(abs(spacing - 1.0) < 1e-3 for spacing in record.spacing[:3])
        for record in dose_records
    )
    return {
        "n_files": len(dose_records),
        "n_patients": len(patients),
        "patients": patients,
        "files": [record.to_dict() for record in dose_records],
        "geometries": [
            {"shape": list(shape), "spacing": list(spacing)}
            for shape, spacing in geometries
        ],
        "space_status": (
            "MNI_GRID_AND_AFFINE_MATCH_OBSERVED_MRI"
            if grid_matches_mri and isotropic_1mm
            else "UNVERIFIED — registration requires affine/reference comparison"
        ),
        "requires_registration": (
            False
            if grid_matches_mri and isotropic_1mm
            else "UNVERIFIED — compare affine and reference grid before C3"
        ),
        "tmz_representation": (
            "Static CRT dose map plus explicit phase/timing variables; never represent "
            "the map as a time-varying TMZ dose."
        ),
        "interpretation": (
            "A static patient-level spatial prior modulated by time, subject to P2 "
            "cross-patient dose permutation."
        ),
    }
