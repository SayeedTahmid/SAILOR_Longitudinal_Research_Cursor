"""Automated Phase-1 integrity guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.constants import QUARANTINE_NAMES
from sailor.schemas import GuardResult, NiftiRecord


def guard_g1(records: list[NiftiRecord]) -> GuardResult:
    primary = [
        record for record in records if record.classification == "CL:enhancing_t1wc"
    ]
    secondary = [
        record
        for record in records
        if record.classification in {"CL:t2wflair_hyperintensity", "ONCO"}
    ]

    def degeneracy(record: NiftiRecord) -> str | None:
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

    primary_degenerate = [
        {"path": record.path, "reason": reason}
        for record in primary
        if (reason := degeneracy(record))
    ]
    inventory_only = [
        {"path": record.path, "classification": record.classification, "reason": reason}
        for record in secondary
        if (reason := degeneracy(record))
    ]
    if not primary:
        status = "FAIL"
        summary = "No CL / enhancing_t1wc target files were resolved."
    elif primary_degenerate:
        status = "FAIL"
        summary = f"{len(primary_degenerate)} primary masks are degenerate or unmeasured."
    else:
        status = "PASS"
        summary = f"{len(primary)} primary masks passed degeneracy checks."
    return GuardResult(
        "G1",
        status,
        summary,
        {
            "primary_degenerate": primary_degenerate,
            "secondary_inventory_only": inventory_only,
            "near_empty_rule": "<=10 voxels or <=1e-6 foreground fraction",
        },
    )


def guard_g5(settings: Settings, records: list[NiftiRecord]) -> GuardResult:
    contaminated = [
        record.path
        for record in records
        if any(name.lower() in record.path.lower() for name in QUARANTINE_NAMES)
    ]
    roots_separate = settings.dataset_root.resolve() != settings.legacy_root.resolve()
    if contaminated or not roots_separate:
        return GuardResult(
            "G5",
            "FAIL",
            "A quarantined artefact entered the inventory or roots overlap.",
            {
                "contaminated_paths": contaminated,
                "roots_separate": roots_separate,
            },
        )
    return GuardResult(
        "G5",
        "PASS",
        "No prior split or quarantined artefact was consumed in Stage 1.",
        {
            "scope": (
                "Stage-1 provenance only; patient-fold and target-window leakage "
                "checks become active when fresh CV splits are generated."
            )
        },
    )


def guard_g7(delta_report: dict[str, Any]) -> GuardResult:
    exact = delta_report.get("status") == "EXACT_SOURCE_FOUND"
    return GuardResult(
        "G7",
        "PASS",
        (
            "Exact raw/BIDS acquisition dates were discovered."
            if exact
            else "Only approximate Δt is available; sensitivity analysis is mandatory."
        ),
        {
            "delta_status": delta_report.get("status", "UNVERIFIED"),
            "n_exact_date_rows": len(delta_report.get("dates", [])),
            "source_attempts": delta_report.get("source_attempts", []),
            "limitation_required": not exact,
        },
    )


def guard_g8(
    link_report: dict[str, Any],
    overview_report: dict[str, Any],
) -> GuardResult:
    duplicates = link_report.get("duplicates", [])
    unresolved = link_report.get("unresolved_rows", [])
    links = link_report.get("links", [])
    unmatched: list[list[str]] = []
    mni_values = [link.get("mni", "") for link in links]
    for subject, session in overview_report.get("patient_sessions", []):
        if not any(subject in value and session in value for value in mni_values):
            unmatched.append([subject, session])
    failed = not links or bool(duplicates) or bool(unresolved)
    return GuardResult(
        "G8",
        "FAIL" if failed else "PASS",
        (
            "Raw-to-MNI correspondence is absent or not one-to-one."
            if failed
            else "Raw-to-MNI links are one-to-one for all parsed rows."
        ),
        {
            "n_links": len(links),
            "duplicates": duplicates,
            "unresolved_rows": unresolved,
            "overview_sessions_not_textually_matched": unmatched,
            "note": (
                "Unmatched overview sessions are reported, not assumed to align by "
                "ses-XX; descriptor versions can contain different session counts."
            ),
        },
    )


def guard_g9(
    missing_report: dict[str, Any],
    overview_report: dict[str, Any],
    records: list[NiftiRecord],
) -> GuardResult:
    if missing_report.get("n_rows", 0) == 0:
        return GuardResult(
            "G9",
            "FAIL",
            "missing.tsv is absent, empty, or unparseable.",
            {"surviving_patient_sessions": []},
        )
    excluded: set[tuple[str, str]] = set()
    for item in missing_report.get("sessions", []):
        text = " ".join(str(value) for value in item.get("missing", [])).lower()
        missing_fields = item.get("missing_fields", {})
        keyed_missing = any(
            ("t1wc" in key.lower() or "t1ce" in key.lower())
            and str(value).strip().lower()
            not in {"", "0", "false", "no", "present", "available"}
            for key, value in missing_fields.items()
        )
        if "t1wc" in text or "t1ce" in text or keyed_missing:
            subject, session = item.get("subject"), item.get("session")
            if subject and session:
                excluded.add((subject, session))
    primary_sessions = {
        (record.subject, record.session)
        for record in records
        if record.classification == "CL:enhancing_t1wc"
        and record.subject
        and record.session
    }
    overview_sessions = {
        tuple(item) for item in overview_report.get("patient_sessions", [])
    }
    surviving = sorted((overview_sessions & primary_sessions) - excluded)
    return GuardResult(
        "G9",
        "PASS",
        f"{len(surviving)} patient-sessions survive measured t1wc/CL requirements.",
        {
            "required_sequence": "t1wc",
            "excluded_by_missing_tsv": [list(item) for item in sorted(excluded)],
            "surviving_patient_sessions": [list(item) for item in surviving],
            "n_surviving_patients": len({subject for subject, _ in surviving}),
        },
    )


def guard_g10(records: list[NiftiRecord]) -> GuardResult:
    mri = [record for record in records if record.classification == "MRI"]
    if not mri:
        return GuardResult(
            "G10",
            "FAIL",
            "No MRI volumes were available for dtype and intensity measurement.",
            {},
        )
    nonfinite = [record.path for record in mri if record.finite is not True]
    observed_dtypes = sorted({record.dtype for record in mri})
    outside_descriptor_range = [
        record.path
        for record in mri
        if record.minimum is not None
        and record.maximum is not None
        and (record.minimum < 0 or record.maximum > 255)
    ]
    return GuardResult(
        "G10",
        "FAIL" if nonfinite else "PASS",
        (
            "Non-finite MRI intensities were detected."
            if nonfinite
            else "MRI dtype and intensity ranges were measured without assuming uint8."
        ),
        {
            "observed_dtypes": observed_dtypes,
            "nonfinite_paths": nonfinite,
            "outside_descriptor_0_255": outside_descriptor_range,
            "normalization_decision": "DEFERRED until measured fold-specific statistics",
        },
    )
