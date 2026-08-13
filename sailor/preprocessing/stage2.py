"""Fresh-runtime orchestration for notebook sections 10–13."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.data.splits import generate_nested_cv_manifest
from sailor.data.timing import validate_timing_cache
from sailor.data.windows import build_longitudinal_windows
from sailor.errors import StopProtocolError
from sailor.guards import guard_g5_stage2
from sailor.preprocessing.pipeline import (
    execute_preprocessing,
    load_phase1_inputs,
    run_phase2_dry_run,
    validate_preprocessing_cache,
)
from sailor.reporting import (
    load_dashboard,
    persist_section_completion,
    write_json,
)
from sailor.schemas import GuardResult


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_windows(
    manifest: dict[str, Any],
    preprocessing_path: Path,
    treatment_path: Path,
) -> None:
    if (
        manifest.get("content_hash") != _hash(manifest.get("windows", []))
        or manifest.get("preprocessing_manifest_sha256")
        != _file_hash(preprocessing_path)
        or manifest.get("treatment_manifest_sha256") != _file_hash(treatment_path)
    ):
        raise StopProtocolError(
            "Longitudinal-window manifest hash is invalid.",
            "Stale or tampered targets could enter CV generation.",
            "Rebuild Section 10 from the preprocessing manifest.",
        )


def _validate_treatment(
    manifest: dict[str, Any],
    preprocessing_path: Path,
    phase1_path: Path,
    timing_path: Path | None,
) -> None:
    if (
        manifest.get("content_hash") != _hash(manifest.get("records", []))
        or manifest.get("preprocessing_manifest_sha256")
        != _file_hash(preprocessing_path)
        or manifest.get("phase1_dataset_manifest_sha256") != _file_hash(phase1_path)
        or (
            timing_path is not None
            and manifest.get("timing_cache_sha256") != _file_hash(timing_path)
        )
    ):
        raise StopProtocolError(
            "Treatment manifest is stale or has invalid parent hashes.",
            "Windows could receive incorrect treatment or missingness records.",
            "Rebuild Section 10 from the verified Phase 1 and preprocessing manifests.",
        )


def _write_section10_manifests(
    settings: Settings,
    *,
    windows_path: Path,
    windows: dict[str, Any],
    treatment_path: Path,
    treatment: dict[str, Any],
) -> None:
    builds = [
        (treatment_path.with_name(f".{treatment_path.name}.build"), treatment_path),
        (windows_path.with_name(f".{windows_path.name}.build"), windows_path),
    ]
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        write_json(builds[0][0], treatment, settings)
        windows["treatment_manifest_sha256"] = _file_hash(builds[0][0])
        write_json(builds[1][0], windows, settings)
        for _, final in builds:
            backup = final.with_name(f".{final.name}.backup")
            if backup.exists():
                backup.unlink()
            if final.exists():
                os.replace(final, backup)
                backups.append((backup, final))
        for build, final in builds:
            os.replace(build, final)
            promoted.append(final)
    except Exception:
        for final in promoted:
            if final.exists():
                final.unlink()
        for backup, final in backups:
            if backup.exists():
                os.replace(backup, final)
        for build, _ in builds:
            if build.exists():
                build.unlink()
        raise
    for backup, _ in backups:
        if backup.exists():
            try:
                backup.unlink()
            except OSError:
                continue


def _validate_cv(
    settings: Settings,
    manifest: dict[str, Any],
    windows_path: Path,
) -> None:
    if (
        manifest.get("fold_scheme") != settings.fold_scheme
        or manifest.get("content_hash") != _hash(manifest.get("folds", []))
        or manifest.get("outer_folds") != settings.outer_folds
        or manifest.get("outer_repeats") != settings.outer_repeats
        or manifest.get("inner_folds") != settings.inner_folds
        or manifest.get("master_seed") != settings.seed
        or manifest.get("windows_manifest_sha256") != _file_hash(windows_path)
    ):
        raise StopProtocolError(
            "Nested CV manifest is stale or does not match the locked topology.",
            "Later evaluation could use unapproved patient assignments.",
            "Regenerate Section 11 before running leakage checks.",
        )


def _paths(settings: Settings) -> dict[str, Path]:
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
        "treatment": settings.dataset_root
        / "05_TREATMENT_DATA"
        / version
        / "v2_treatment_manifest.json",
        "timing": settings.dataset_root
        / "05_TREATMENT_DATA"
        / version
        / "v2_canonical_timing_cache.json",
        "cv": settings.dataset_root
        / "04_LONGITUDINAL_WINDOWS"
        / version
        / "v2_cv_manifest.json",
        "leakage": settings.dataset_root
        / "06_QC_REPORTS"
        / "v2_phase2_leakage_report.json",
        "qc": settings.dataset_root / "06_QC_REPORTS" / "v2_phase2_qc_report.json",
    }


def _counts(windows: dict[str, Any]) -> tuple[int, int]:
    return int(windows.get("n_patients", 0)), int(windows.get("n_windows", 0))


def run_stage2_section(
    section_id: int,
    settings: Settings,
    *,
    execute: bool = False,
    extraction_approved: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if section_id not in range(10, 14):
        raise ValueError("Phase 2 supports section IDs 10–13 only.")
    settings.validate()
    paths = _paths(settings)

    if section_id == 10:
        if not execute:
            return {
                "section": 10,
                "mode": "DRY_RUN",
                **run_phase2_dry_run(settings),
            }
        if force or not paths["preprocessing"].is_file():
            preprocessing = execute_preprocessing(
                settings,
                extraction_approved=extraction_approved,
            )["manifest"]
        else:
            preprocessing = _read(paths["preprocessing"])
            validate_preprocessing_cache(settings, preprocessing)
        phase1, _ = load_phase1_inputs(settings)
        timing_cache = None
        if not phase1.get("delta_t", {}).get("dates"):
            if not paths["timing"].is_file():
                raise StopProtocolError(
                    "Verified approximate timing cache is missing.",
                    "Longitudinal windows cannot be ordered without dates or canonical intervals.",
                    "Persist v2_canonical_timing_cache.json from verified metadata.",
                )
            timing_cache = _read(paths["timing"])
            validate_timing_cache(settings, timing_cache)
        built = build_longitudinal_windows(
            phase1,
            preprocessing["records"],
            min_history_scans=settings.min_history_scans,
            timing_cache=timing_cache,
        )
        windows = built["windows"]
        windows["preprocessing_manifest_sha256"] = _file_hash(
            paths["preprocessing"]
        )
        windows["phase1_dataset_manifest_sha256"] = preprocessing[
            "phase1_dataset_manifest_sha256"
        ]
        treatment = built["treatment"]
        phase1_path = (
            settings.dataset_root
            / "01_DATA_FOUNDATION"
            / "v2_dataset_manifest.json"
        )
        treatment["preprocessing_manifest_sha256"] = _file_hash(
            paths["preprocessing"]
        )
        treatment["phase1_dataset_manifest_sha256"] = _file_hash(phase1_path)
        treatment["timing_cache_sha256"] = (
            _file_hash(paths["timing"]) if timing_cache is not None else None
        )
        treatment["content_hash"] = _hash(treatment["records"])
        _write_section10_manifests(
            settings,
            windows_path=paths["windows"],
            windows=windows,
            treatment_path=paths["treatment"],
            treatment=treatment,
        )
        patients, pairs = _counts(windows)
        guard = GuardResult(
            "P2_PREPROCESSING",
            "PASS",
            f"{preprocessing['n_records']} sessions produced {pairs} valid windows.",
            {
                "selected_sequence": preprocessing["selected_sequence"],
                "preprocessing_version": settings.preprocessing_version,
            },
        )
        persist_section_completion(
            settings,
            10,
            [guard],
            n_patients=patients,
            n_sessions=preprocessing["n_records"],
            n_pairs=pairs,
            preprocessing_version=settings.preprocessing_version,
            fold_scheme=settings.fold_scheme,
        )
        return {
            "section": 10,
            "mode": "EXECUTE",
            "preprocessing_manifest": str(paths["preprocessing"]),
            "windows_manifest": str(paths["windows"]),
            "treatment_manifest": str(paths["treatment"]),
            "n_windows": pairs,
            "n_patients": patients,
        }

    if (
        not paths["windows"].is_file()
        or not paths["preprocessing"].is_file()
        or not paths["treatment"].is_file()
    ):
        raise StopProtocolError(
            "Section 10 artefacts are missing.",
            f"Section {section_id} cannot run from a fresh runtime.",
            "Complete the approved Section 10 extraction and window build first.",
        )
    windows = _read(paths["windows"])
    preprocessing = _read(paths["preprocessing"])
    treatment = _read(paths["treatment"])
    validate_preprocessing_cache(settings, preprocessing)
    phase1_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json"
    )
    timing_path = paths["timing"] if paths["timing"].is_file() else None
    if timing_path is not None:
        validate_timing_cache(settings, _read(timing_path))
    _validate_treatment(
        treatment,
        paths["preprocessing"],
        phase1_path,
        timing_path,
    )
    _validate_windows(
        windows,
        paths["preprocessing"],
        paths["treatment"],
    )
    patients, pairs = _counts(windows)

    if section_id == 11:
        if force or not paths["cv"].is_file():
            cv = generate_nested_cv_manifest(
                windows,
                master_seed=settings.seed,
                outer_folds=settings.outer_folds,
                repeats=settings.outer_repeats,
                inner_folds=settings.inner_folds,
                fold_scheme=settings.fold_scheme,
            )
            cv["windows_manifest_sha256"] = _file_hash(paths["windows"])
            write_json(paths["cv"], cv, settings)
        else:
            cv = _read(paths["cv"])
            _validate_cv(settings, cv, paths["windows"])
        guard = GuardResult(
            "CV_STRUCTURE",
            "PASS",
            f"Frozen {settings.fold_scheme} topology for {cv['n_patients']} patients.",
            {"fold_hash": cv["content_hash"]},
        )
        persist_section_completion(
            settings,
            11,
            [guard],
            n_patients=patients,
            n_sessions=preprocessing["n_records"],
            n_pairs=pairs,
            preprocessing_version=settings.preprocessing_version,
            fold_scheme=settings.fold_scheme,
        )
        return {"section": 11, "cv_manifest": str(paths["cv"]), "cv": cv}

    if not paths["cv"].is_file():
        raise StopProtocolError(
            "Section 11 CV manifest is missing.",
            f"Section {section_id} cannot verify patient-level leakage.",
            "Generate and freeze Section 11 first.",
        )
    cv = _read(paths["cv"])
    _validate_cv(settings, cv, paths["windows"])
    leakage = guard_g5_stage2(settings, windows, cv, preprocessing)

    if section_id == 12:
        write_json(paths["leakage"], leakage.to_dict(), settings)
        persist_section_completion(
            settings,
            12,
            [leakage],
            n_patients=patients,
            n_sessions=preprocessing["n_records"],
            n_pairs=pairs,
            preprocessing_version=settings.preprocessing_version,
            fold_scheme=settings.fold_scheme,
        )
        if leakage.status == "FAIL":
            raise StopProtocolError(
                leakage.summary,
                "Phase 2 split or preprocessing leakage invalidates later evaluation.",
                "Inspect v2_phase2_leakage_report.json and rebuild affected manifests.",
            )
        return {"section": 12, "guard": leakage.to_dict()}

    if leakage.status == "FAIL":
        raise StopProtocolError(
            "G5 is not passing for Phase 2.",
            "Final Phase 2 QC cannot be completed.",
            "Run and resolve Section 12.",
        )
    qc = {
        "data_version": settings.data_version,
        "preprocessing_version": settings.preprocessing_version,
        "implementation_id": settings.implementation_id,
        "PRIMARY_TARGET_MASK": settings.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": settings.primary_target_component,
        "selected_sequence": settings.primary_input_sequence,
        "n_preprocessed_sessions": preprocessing["n_records"],
        "n_patients": patients,
        "n_windows": pairs,
        "timing_provenance": windows["timing_provenance"],
        "fold_scheme": settings.fold_scheme,
        "fold_hash": cv["content_hash"],
        "guards": [leakage.to_dict()],
        "failed_guards": [],
        "resource_profile": "UNMEASURED",
        "optional_modalities": "BLOCKED",
    }
    write_json(paths["qc"], qc, settings)
    complete = GuardResult(
        "P2_QC",
        "PASS",
        "Phase 2 QC, frozen windows, and nested patient folds are complete.",
        {"qc_report": str(paths["qc"])},
    )
    persist_section_completion(
        settings,
        13,
        [leakage, complete],
        n_patients=patients,
        n_sessions=preprocessing["n_records"],
        n_pairs=pairs,
        preprocessing_version=settings.preprocessing_version,
        fold_scheme=settings.fold_scheme,
    )
    return {
        "section": 13,
        "qc_report": str(paths["qc"]),
        "qc": qc,
        "dashboard": load_dashboard(settings),
    }
