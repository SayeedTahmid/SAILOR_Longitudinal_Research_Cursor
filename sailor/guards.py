"""Automated Phase-1 integrity guards."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.constants import QUARANTINE_NAMES
from sailor.data.splits import derive_repeat_seeds
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


def guard_g5_stage2(
    settings: Settings,
    windows_manifest: dict[str, Any],
    cv_manifest: dict[str, Any],
    preprocessing_manifest: dict[str, Any],
) -> GuardResult:
    failures: list[dict[str, Any]] = []
    windows = windows_manifest.get("windows", [])
    patients = {window["subject"] for window in windows}
    preprocessed_records = preprocessing_manifest.get("records", [])
    preprocessed_sessions = {
        f"{record['subject']}/{record['mni_session']}"
        for record in preprocessed_records
        if record.get("subject")
        and record.get("mni_session")
        and record.get("mri_output")
        and record.get("mask_output")
        and record.get("checksums")
    }
    unbound_windows: list[str] = []
    for window in windows:
        required = {
            f"{window['subject']}/{session}"
            for session in [
                *window.get("history_mni_sessions", []),
                window["target_mni_session"],
            ]
        }
        if not required <= preprocessed_sessions:
            unbound_windows.append(window["window_id"])
    if unbound_windows:
        failures.append(
            {
                "type": "window_without_preprocessed_record",
                "window_ids": unbound_windows[:50],
                "count": len(unbound_windows),
            }
        )
    targets_by_patient = {
        patient: {
            f"{patient}/{window['target_mni_session']}"
            for window in windows
            if window["subject"] == patient
        }
        for patient in patients
    }
    test_appearances: Counter[tuple[int, str]] = Counter()
    expected_seeds = derive_repeat_seeds(settings.seed, settings.outer_repeats)
    if (
        cv_manifest.get("outer_folds") != settings.outer_folds
        or cv_manifest.get("outer_repeats") != settings.outer_repeats
        or cv_manifest.get("inner_folds") != settings.inner_folds
        or cv_manifest.get("repeat_seeds") != expected_seeds
        or len(cv_manifest.get("folds", []))
        != settings.outer_folds * settings.outer_repeats
    ):
        failures.append({"type": "cv_topology_mismatch"})
    seen_outer: dict[int, set[int]] = {
        repeat: set() for repeat in range(settings.outer_repeats)
    }
    for fold in cv_manifest.get("folds", []):
        if fold.get("repeat") not in seen_outer:
            failures.append({"type": "unexpected_repeat", "repeat": fold.get("repeat")})
            continue
        seen_outer[fold["repeat"]].add(fold["outer_fold"])
        if fold.get("seed") != expected_seeds[fold["repeat"]]:
            failures.append(
                {"type": "repeat_seed_mismatch", "repeat": fold["repeat"]}
            )
        if len(fold.get("inner_folds", [])) != settings.inner_folds:
            failures.append(
                {
                    "type": "inner_fold_count",
                    "repeat": fold["repeat"],
                    "outer_fold": fold["outer_fold"],
                }
            )
        inner_ids: set[int] = set()
        validation_appearances: Counter[str] = Counter()
        train = set(fold["train_patients"])
        test = set(fold["test_patients"])
        overlap = sorted(train & test)
        if overlap:
            failures.append(
                {"type": "outer_patient_overlap", "fold": fold["outer_fold"], "patients": overlap}
            )
        if train | test != patients:
            failures.append(
                {
                    "type": "outer_patient_coverage",
                    "fold": fold["outer_fold"],
                    "missing": sorted(patients - train - test),
                }
            )
        for patient in test:
            test_appearances[(fold["repeat"], patient)] += 1
        train_targets = set().union(*(targets_by_patient[item] for item in train)) if train else set()
        test_targets = set().union(*(targets_by_patient[item] for item in test)) if test else set()
        if train_targets & test_targets:
            failures.append(
                {"type": "target_overlap", "fold": fold["outer_fold"]}
            )
        for inner in fold.get("inner_folds", []):
            inner_ids.add(inner["inner_fold"])
            inner_train = set(inner["train_patients"])
            validation = set(inner["validation_patients"])
            validation_appearances.update(validation)
            if inner_train & validation or (inner_train | validation) != train:
                failures.append(
                    {
                        "type": "inner_partition_invalid",
                        "outer_fold": fold["outer_fold"],
                        "inner_fold": inner["inner_fold"],
                    }
                )
            if (inner_train | validation) & test:
                failures.append(
                    {
                        "type": "outer_test_in_inner_loop",
                        "outer_fold": fold["outer_fold"],
                        "inner_fold": inner["inner_fold"],
                    }
                )
        if inner_ids != set(range(settings.inner_folds)):
            failures.append(
                {
                    "type": "inner_fold_ids",
                    "repeat": fold["repeat"],
                    "outer_fold": fold["outer_fold"],
                    "observed": sorted(inner_ids),
                }
            )
        invalid_validation_counts = {
            patient: validation_appearances[patient]
            for patient in train
            if validation_appearances[patient] != 1
        }
        if invalid_validation_counts:
            failures.append(
                {
                    "type": "inner_validation_appearance_count",
                    "repeat": fold["repeat"],
                    "outer_fold": fold["outer_fold"],
                    "counts": invalid_validation_counts,
                }
            )
    repeats = cv_manifest.get("outer_repeats", 0)
    for repeat in range(repeats):
        for patient in patients:
            if test_appearances[(repeat, patient)] != 1:
                failures.append(
                    {
                        "type": "test_appearance_count",
                        "repeat": repeat,
                        "patient": patient,
                        "count": test_appearances[(repeat, patient)],
                    }
                )
    for repeat, observed in seen_outer.items():
        if observed != set(range(settings.outer_folds)):
            failures.append(
                {
                    "type": "outer_fold_ids",
                    "repeat": repeat,
                    "observed": sorted(observed),
                }
            )
    contaminated = [
        record.get("mri_source", "")
        for record in preprocessing_manifest.get("records", [])
        if any(
            name.lower() in record.get("mri_source", "").lower()
            for name in QUARANTINE_NAMES
        )
    ]
    if contaminated:
        failures.append({"type": "quarantine_input", "paths": contaminated})
    if preprocessing_manifest.get("normalization_scope") != "single_volume_brain_mask":
        failures.append({"type": "normalization_scope_not_leakage_safe"})
    if settings.fold_scheme != cv_manifest.get("fold_scheme"):
        failures.append({"type": "fold_scheme_mismatch"})
    return GuardResult(
        "G5",
        "FAIL" if failures else "PASS",
        (
            f"Stage 2 leakage checks found {len(failures)} failures."
            if failures
            else "Patient, target, inner-loop, normalization, and provenance leakage checks passed."
        ),
        {
            "scope": "Stage-2 full leakage guard",
            "failures": failures,
            "n_patients": len(patients),
            "n_windows": len(windows),
            "fold_hash": cv_manifest.get("content_hash"),
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


def guard_g3(comparisons: dict[str, Any], *, required_models: tuple[str, ...]) -> GuardResult:
    failures: list[str] = []
    persistence = comparisons.get("rung_summaries", {}).get("C-1")
    if not persistence or "ci95" not in persistence:
        failures.append("persistence_ci_missing")
    for name in required_models:
        key = f"{name}_minus_C-1"
        if key not in comparisons.get("paired", {}):
            failures.append(f"missing_paired_{key}")
    if failures:
        return GuardResult(
            "G3",
            "FAIL",
            "Persistence was not reported as a patient-level bar against every proposed model.",
            {"failures": failures},
        )
    indistinguishable = [
        name
        for name in required_models
        if not comparisons["paired"][f"{name}_minus_C-1"]["beats"]
    ]
    if indistinguishable:
        summary = (
            "Persistence CIs are reported; "
            + ", ".join(indistinguishable)
            + " are statistically indistinguishable from copying the last mask."
        )
    else:
        summary = (
            "Persistence CIs are reported and every proposed model was compared "
            "with a paired patient-level test."
        )
    return GuardResult(
        "G3",
        "PASS",
        summary,
        {
            "persistence": persistence,
            "indistinguishable_from_persistence": indistinguishable,
            "rule": "Higher mean Dice is not a result; paired patient CIs are required.",
        },
    )


def guard_g4(comparisons: dict[str, Any]) -> GuardResult:
    paired = comparisons.get("paired", {})
    key = "C1_minus_C1_constant"
    if key not in paired:
        return GuardResult(
            "G4",
            "FAIL",
            "The constant-Δt ablation was not run.",
            {"failures": ["missing_constant_dt_control"]},
        )
    result = paired[key]
    decorative = not result["beats"]
    summary = (
        "Δt conditioning is decorative under the retrained constant-Δt control."
        if decorative
        else "C1 beats the retrained constant-Δt control outside the paired 95% CI."
    )
    return GuardResult(
        "G4",
        "PASS",
        summary,
        {
            "paired": result,
            "decorative": decorative,
            "rule": "The G4 control is retrained; inference-only zeroing is not the primary ablation.",
        },
    )
