"""Automated Phase-1 integrity guards."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.constants import QUARANTINE_NAMES
from sailor.data.targets import mask_degeneracy_reason
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

    primary_degenerate = [
        {"path": record.path, "reason": reason}
        for record in primary
        if (reason := mask_degeneracy_reason(record))
    ]
    valid_primary_count = len(primary) - len(primary_degenerate)
    inventory_only = [
        {"path": record.path, "classification": record.classification, "reason": reason}
        for record in secondary
        if (reason := mask_degeneracy_reason(record))
    ]
    if not primary:
        status = "FAIL"
        summary = "No CL / enhancing_t1wc target files were resolved."
    elif valid_primary_count == 0:
        status = "FAIL"
        summary = "Every resolved CL / enhancing_t1wc mask is unusable."
    else:
        status = "PASS"
        summary = (
            f"{valid_primary_count} primary masks are valid; "
            f"{len(primary_degenerate)} are excluded as missing labels."
        )
    return GuardResult(
        "G1",
        status,
        summary,
        {
            "primary_degenerate": primary_degenerate,
            "exclusion_policy": (
                "Degenerate primary masks are missing labels and are excluded from "
                "cohort windows and scoring; they are never negative examples."
            ),
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
    records: list[NiftiRecord] | None = None,
) -> GuardResult:
    duplicates = link_report.get("duplicates", [])
    unresolved = link_report.get("unresolved_rows", [])
    links = link_report.get("links", [])
    explicitly_unmatched = link_report.get("explicitly_unmatched", [])
    mapped_raw = {link["raw"] for link in links}
    mapped_mni = {link["mni"] for link in links}
    explicit_raw = {
        item["raw"] for item in explicitly_unmatched if item.get("raw")
    }
    overview_sessions = {
        f"{subject}/{session}"
        for subject, session in overview_report.get("patient_sessions", [])
    }
    unexplained_overview = sorted(overview_sessions - mapped_raw - explicit_raw)

    observed_mni = {
        f"{record.subject}/{record.session}"
        for record in records or []
        if record.subject and record.session
    }
    observed_mni_unmapped = sorted(observed_mni - mapped_mni)
    mapped_mni_missing_on_disk = sorted(mapped_mni - observed_mni) if records else []
    failed = (
        not links
        or bool(duplicates)
        or bool(unresolved)
        or bool(unexplained_overview)
        or bool(observed_mni_unmapped)
        or bool(mapped_mni_missing_on_disk)
    )
    return GuardResult(
        "G8",
        "FAIL" if failed else "PASS",
        (
            "Raw-to-MNI correspondence is absent or not one-to-one."
            if failed
            else (
                "Raw-to-MNI links are one-to-one; raw sessions marked 'no' are "
                "retained as explicit unmatched records."
            )
        ),
        {
            "n_links": len(links),
            "n_explicitly_unmatched": len(explicitly_unmatched),
            "explicitly_unmatched": explicitly_unmatched,
            "duplicates": duplicates,
            "unresolved_rows": unresolved,
            "overview_sessions_unexplained": unexplained_overview,
            "observed_mni_sessions_unmapped": observed_mni_unmapped,
            "mapped_mni_sessions_missing_on_disk": mapped_mni_missing_on_disk,
            "note": (
                "A raw-session 'no' value is an official absent MNI correspondence, "
                "not a join failure and never an assumed ses-XX alignment."
            ),
        },
    )


def guard_g9(
    missing_report: dict[str, Any],
    overview_report: dict[str, Any],
    records: list[NiftiRecord],
    link_report: dict[str, Any],
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
        missing_sequences = {
            str(value).lower()
            for value in item.get("missing_sequences", item.get("missing", []))
        }
        if missing_sequences & {"t1wc", "t1ce", "t1c"}:
            subject, session = item.get("subject"), item.get("session")
            if subject and session:
                excluded.add((subject, session))
    primary_mni_sessions = {
        f"{record.subject}/{record.session}"
        for record in records
        if record.classification == "CL:enhancing_t1wc"
        and record.subject
        and record.session
        and mask_degeneracy_reason(record) is None
    }
    overview_raw_sessions = {
        f"{subject}/{session}"
        for subject, session in overview_report.get("patient_sessions", [])
    }
    excluded_raw_sessions = {
        f"{subject}/{session}" for subject, session in excluded
    }
    raw_to_mni = {
        link["raw"]: link["mni"] for link in link_report.get("links", [])
    }
    surviving = sorted(
        {
            (raw, mni)
            for raw, mni in raw_to_mni.items()
            if raw in overview_raw_sessions - excluded_raw_sessions
            and mni in primary_mni_sessions
        }
    )
    status = "PASS" if surviving else "FAIL"
    return GuardResult(
        "G9",
        status,
        f"{len(surviving)} patient-sessions survive measured t1wc/CL requirements.",
        {
            "required_sequence": "T1c",
            "excluded_by_missing_tsv_raw": sorted(excluded_raw_sessions),
            "surviving_raw_mni_pairs": [
                {"raw": raw, "mni": mni} for raw, mni in surviving
            ],
            "n_surviving_patients": len(
                {raw.split("/", 1)[0] for raw, _ in surviving}
            ),
            "join_policy": "raw and MNI sessions joined only through raw-mni-link.tsv",
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
    primary_mri = [
        record
        for record in mri
        if (record.sequence or "").lower().startswith(("t1c", "t1wc", "t1ce"))
    ]
    nonfinite_records = [record for record in mri if record.finite is not True]
    nonfinite = [record.path for record in nonfinite_records]
    primary_nonfinite = [
        record.path for record in primary_mri if record.finite is not True
    ]
    optional_nonfinite_by_sequence = dict(
        sorted(
            Counter(
                record.sequence or "UNRESOLVED"
                for record in nonfinite_records
                if record not in primary_mri
            ).items()
        )
    )
    observed_dtypes = sorted({record.dtype for record in mri})
    outside_descriptor_range = [
        record.path
        for record in mri
        if record.minimum is not None
        and record.maximum is not None
        and (record.minimum < 0 or record.maximum > 255)
    ]
    if not primary_mri:
        status = "FAIL"
        summary = "No T1c-family MRI volumes were resolved for the locked target."
    elif primary_nonfinite:
        status = "FAIL"
        summary = (
            f"{len(primary_nonfinite)} T1c-family MRI volumes contain NaN or Inf."
        )
    else:
        status = "PASS"
        summary = (
            f"{len(primary_mri)} T1c-family volumes are finite; "
            f"{len(nonfinite) - len(primary_nonfinite)} optional-modality volumes "
            "remain blocked pending preprocessing."
        )
    return GuardResult(
        "G10",
        status,
        summary,
        {
            "observed_dtypes": observed_dtypes,
            "nonfinite_paths": nonfinite,
            "n_primary_t1c_volumes": len(primary_mri),
            "primary_t1c_nonfinite_paths": primary_nonfinite,
            "blocked_optional_sequences": optional_nonfinite_by_sequence,
            "outside_descriptor_0_255": outside_descriptor_range,
            "normalization_decision": "DEFERRED until measured fold-specific statistics",
            "binding_policy": (
                "Optional sequences with non-finite values are ineligible for model "
                "input until Phase 2 defines, tests, and approves a deterministic "
                "finite-value policy. No values were repaired in Stage 1."
            ),
        },
    )
