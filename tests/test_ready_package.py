from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sailor.data.ready import ReadyDataset
from sailor.data.splits import generate_nested_cv_manifest
from sailor.errors import StopProtocolError
from sailor.packaging.audit import audit_frozen_source
from sailor.packaging.build import build_ready_package
from sailor.packaging.common import content_hash, sha256_file
from sailor.packaging.verify import verify_ready_package


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _ready_source(tmp_path: Path) -> Path:
    root = tmp_path / "frozen"
    records = []
    windows = []
    treatment_records = []
    timing_records = []
    patients = [f"sub-{index:02d}" for index in range(1, 6)]
    for patient_index, subject in enumerate(patients):
        for session_index in range(1, 4):
            session = f"ses-{session_index:02d}"
            image_path = (
                root
                / "02_PREPROCESSED_MRI"
                / "p2.0"
                / subject
                / session
                / "T1c-icor.npy"
            )
            mask_path = (
                root
                / "03_TUMOR_MASKS"
                / "p2.0"
                / subject
                / session
                / "CL-enhancing-t1wc.npy"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
            image = image + patient_index + session_index
            mask = np.zeros((4, 4, 4), dtype=np.uint8)
            mask[1:4, 1:4, 1:4] = 1
            np.save(image_path, image, allow_pickle=False)
            np.save(mask_path, mask, allow_pickle=False)
            records.append(
                {
                    "subject": subject,
                    "raw_subject": subject,
                    "raw_session": session,
                    "mni_session": session,
                    "mri_source": f"archive/{subject}/{session}/T1c-icor.nii.gz",
                    "mask_source": f"archive/{subject}/{session}/CL.nii.gz",
                    "brain_mask_source": f"archive/{subject}/{session}/brain.nii.gz",
                    "mri_output": str(image_path),
                    "mask_output": str(mask_path),
                    "shape": [4, 4, 4],
                    "spacing": [1.0, 1.0, 1.0],
                    "affine_hash": f"affine-{subject}",
                    "mask_original_positive_value": 1.0,
                    "scaling": {"scope": "single_volume_brain_mask"},
                    "checksums": {
                        "mri_sha256": sha256_file(image_path),
                        "mask_sha256": sha256_file(mask_path),
                    },
                }
            )
            treatment_records.append(
                {
                    "subject": subject,
                    "raw_subject": subject,
                    "raw_session": session,
                    "mni_session": session,
                    "status": "CRT",
                    "missing": False,
                    "dose_reference": None,
                    "dose_missing": True,
                }
            )
            timing_records.append(
                {
                    "subject": subject,
                    "mni_session": session,
                    "approximate_day": (session_index - 1) * 14,
                    "interval_from_previous_days": (
                        None if session_index == 1 else 14
                    ),
                    "timing_provenance": "approximate_mni_intervals",
                    "treatment_status": "CRT",
                    "treatment_missing": False,
                    "interval_source": f"archive/{subject}/intervals-days.txt",
                    "treatment_source": f"archive/{subject}/{session}/treatment.txt",
                }
            )
        windows.append(
            {
                "window_id": f"{subject}-window",
                "subject": subject,
                "history_raw_sessions": [
                    f"{subject}/ses-01",
                    f"{subject}/ses-02",
                ],
                "history_mni_sessions": ["ses-01", "ses-02"],
                "target_raw_session": f"{subject}/ses-03",
                "target_mni_session": "ses-03",
                "history_delta_days": [28.0, 14.0],
                "target_delta_days": 14.0,
                "timing_provenance": "approximate_mni_intervals",
                "treatment_status": "CRT",
                "treatment_missing": False,
                "patient_weight": 1.0,
            }
        )

    preprocessing = {
        "data_version": "v2.0",
        "preprocessing_version": "p2.0",
        "implementation_id": "cursor_primary",
        "PRIMARY_TARGET_MASK": "CL",
        "PRIMARY_TARGET_COMPONENT": "enhancing_t1wc",
        "selected_sequence": "T1c-icor",
        "n_records": len(records),
        "records": records,
        "records_hash": content_hash(records),
        "optional_modalities": "BLOCKED",
        "normalization_scope": "single_volume_brain_mask",
        "visual_qc_montage": str(
            root / "06_QC_REPORTS" / "v2_phase2_t1c_montage.pgm"
        ),
    }
    windows_manifest = {
        "minimum_history_scans": 2,
        "history_policy": "all earlier eligible scans; variable length",
        "timing_provenance": "approximate_mni_intervals",
        "n_windows": len(windows),
        "n_patients": len(patients),
        "windows_per_patient": {patient: 1 for patient in patients},
        "excluded_patients_insufficient_history": [],
        "windows": windows,
        "content_hash": content_hash(windows),
    }
    folds = generate_nested_cv_manifest(
        windows_manifest,
        master_seed=1337,
        outer_folds=5,
        repeats=3,
        inner_folds=4,
        fold_scheme="5fold_x3seeds_nested4",
    )
    treatment = {
        "unknown_policy": "missingness indicator; never a treatment class",
        "n_records": len(treatment_records),
        "n_missing": 0,
        "n_dose_missing": len(treatment_records),
        "records": treatment_records,
        "content_hash": content_hash(treatment_records),
    }
    timing = {
        "data_version": "v2.0",
        "preprocessing_version": "p2.0",
        "implementation_id": "cursor_primary",
        "timing_provenance": "approximate_mni_intervals",
        "n_patients": len(patients),
        "n_sessions": len(timing_records),
        "records": timing_records,
        "content_hash": content_hash(timing_records),
    }
    phase1 = {
        "overview": {"patients": patients},
        "target": {"excluded_primary_files": []},
    }
    canonical = {"verification": {"files": []}}
    phase1_qc = {"failed_guards": []}
    phase2_qc = {
        "data_version": "v2.0",
        "preprocessing_version": "p2.0",
        "failed_guards": [],
    }
    leakage = {"guard_id": "G5", "status": "PASS", "details": {"failures": []}}

    phase1_path = root / "01_DATA_FOUNDATION/v2_dataset_manifest.json"
    canonical_path = root / "01_DATA_FOUNDATION/v2_canonical_manifest.json"
    phase1_qc_path = root / "06_QC_REPORTS/v2_stage1_qc_report.json"
    _write_json(phase1_path, phase1)
    _write_json(canonical_path, canonical)
    _write_json(phase1_qc_path, phase1_qc)

    preprocessing["phase1_dataset_manifest_sha256"] = sha256_file(phase1_path)
    preprocessing["phase1_canonical_manifest_sha256"] = sha256_file(canonical_path)
    preprocessing["phase1_qc_report_sha256"] = sha256_file(phase1_qc_path)
    preprocessing_path = (
        root / "02_PREPROCESSED_MRI/p2.0/v2_preprocessing_manifest.json"
    )
    _write_json(preprocessing_path, preprocessing)

    timing["phase1_dataset_manifest_sha256"] = sha256_file(phase1_path)
    timing["phase1_canonical_manifest_sha256"] = sha256_file(canonical_path)
    timing_path = root / "05_TREATMENT_DATA/p2.0/v2_canonical_timing_cache.json"
    _write_json(timing_path, timing)
    treatment["preprocessing_manifest_sha256"] = sha256_file(preprocessing_path)
    treatment["timing_cache_sha256"] = sha256_file(timing_path)
    treatment["phase1_dataset_manifest_sha256"] = sha256_file(phase1_path)
    treatment_path = root / "05_TREATMENT_DATA/p2.0/v2_treatment_manifest.json"
    _write_json(treatment_path, treatment)

    windows_manifest["preprocessing_manifest_sha256"] = sha256_file(
        preprocessing_path
    )
    windows_manifest["treatment_manifest_sha256"] = sha256_file(treatment_path)
    windows_manifest["phase1_dataset_manifest_sha256"] = sha256_file(phase1_path)
    windows_path = root / "04_LONGITUDINAL_WINDOWS/p2.0/v2_windows_manifest.json"
    _write_json(windows_path, windows_manifest)
    folds["windows_manifest_sha256"] = sha256_file(windows_path)
    _write_json(root / "04_LONGITUDINAL_WINDOWS/p2.0/v2_cv_manifest.json", folds)
    _write_json(root / "06_QC_REPORTS/v2_phase2_qc_report.json", phase2_qc)
    _write_json(root / "06_QC_REPORTS/v2_phase2_leakage_report.json", leakage)
    (root / "06_QC_REPORTS" / "v2_phase2_t1c_montage.pgm").write_bytes(b"P5\n1 1\n255\n\x00")
    (root / "06_QC_REPORTS" / "v2_phase2_t1c_cl_overlay.png").write_bytes(b"png-fixture")
    for section in range(1, 14):
        _write_json(
            root
            / "01_DATA_FOUNDATION"
            / "state"
            / f"section_{section:02d}_complete.json",
            {
                "section": section,
                "status": "complete",
                "git_commit": "fixture-commit",
                "git_dirty": False,
                "guards_failed": [],
            },
        )
    return root


def test_ready_package_build_verify_and_loader(tmp_path: Path) -> None:
    source = _ready_source(tmp_path)
    destination = tmp_path / "SAILOR_READY_v2.0"
    audit = audit_frozen_source(source, destination)
    assert audit["n_images"] == 15
    assert audit["n_masks"] == 15
    result = build_ready_package(
        source,
        destination,
        execute=True,
        approve_copy=True,
    )
    assert result["audit"]["status"] == "PASS"
    assert result["source_bytes_unchanged"] is True
    assert result["authoritative_source"] == str(source.resolve())
    assert result["destination"] == str(destination.resolve())
    assert result["source_vs_package_checksum_comparison"]["mismatches"] == []
    assert "not moved" in result["source_preservation_confirmation"]
    assert result["promotion_mode"] in {
        "windows_rename_no_replace",
        "renameat2_noreplace",
        "exclusive_lock_drive_rename_fallback",
    }
    assert verify_ready_package(destination)["status"] == "PASS"
    dataset = ReadyDataset(destination)
    image, mask, record = dataset.load_session("sub-01", "ses-01")
    assert image.shape == mask.shape == (4, 4, 4)
    assert record["mri_output"].startswith("images/")
    assert len(list(dataset.iter_windows())) == 5
    assert dataset.get_outer_fold(0, 0)["test_patients"]
    with pytest.raises(ValueError):
        dataset.load_session("sub-01", "ses-01", mmap_mode="r+")
    assert not _absolute_operational_paths(destination)

    with pytest.raises(StopProtocolError):
        build_ready_package(
            source,
            destination,
            execute=True,
            approve_copy=True,
        )


def test_ready_audit_rejects_source_overlap_dirty_state_and_external_array(
    tmp_path: Path,
) -> None:
    source = _ready_source(tmp_path)
    with pytest.raises(StopProtocolError):
        audit_frozen_source(source, source / "SAILOR_READY_v2.0")

    section13 = (
        source
        / "01_DATA_FOUNDATION"
        / "state"
        / "section_13_complete.json"
    )
    record = json.loads(section13.read_text(encoding="utf-8"))
    record["git_dirty"] = True
    _write_json(section13, record)
    with pytest.raises(StopProtocolError):
        audit_frozen_source(source, tmp_path / "ready-dirty")

    source = _ready_source(tmp_path / "external-case")
    preprocessing_path = (
        source / "02_PREPROCESSED_MRI/p2.0/v2_preprocessing_manifest.json"
    )
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    external = tmp_path / "external.npy"
    np.save(external, np.ones((4, 4, 4), dtype=np.float32), allow_pickle=False)
    preprocessing["records"][0]["mri_output"] = str(external)
    preprocessing["records"][0]["checksums"]["mri_sha256"] = sha256_file(external)
    preprocessing["records_hash"] = content_hash(preprocessing["records"])
    _write_json(preprocessing_path, preprocessing)
    treatment_path = source / "05_TREATMENT_DATA/p2.0/v2_treatment_manifest.json"
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    treatment["preprocessing_manifest_sha256"] = sha256_file(preprocessing_path)
    _write_json(treatment_path, treatment)
    windows_path = source / "04_LONGITUDINAL_WINDOWS/p2.0/v2_windows_manifest.json"
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    windows["preprocessing_manifest_sha256"] = sha256_file(preprocessing_path)
    windows["treatment_manifest_sha256"] = sha256_file(treatment_path)
    _write_json(windows_path, windows)
    folds_path = source / "04_LONGITUDINAL_WINDOWS/p2.0/v2_cv_manifest.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    folds["windows_manifest_sha256"] = sha256_file(windows_path)
    _write_json(folds_path, folds)
    with pytest.raises(StopProtocolError):
        audit_frozen_source(source, tmp_path / "ready-external")


def _absolute_operational_paths(root: Path) -> list[str]:
    found: list[str] = []
    for relative in (
        "manifests/package_manifest.json",
        "manifests/preprocessing_manifest.json",
        "manifests/longitudinal_windows.json",
        "manifests/folds.json",
        "metadata/treatment_manifest.json",
        "metadata/timing_cache.json",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        if str(root.parent) in text or "/content/drive/" in text:
            found.append(relative)
    return found


def test_ready_verifier_detects_tamper_and_raw_file(tmp_path: Path) -> None:
    source = _ready_source(tmp_path)
    destination = tmp_path / "SAILOR_READY_v2.0"
    build_ready_package(source, destination, execute=True, approve_copy=True)
    image = next(destination.glob("images/T1c/**/*.npy"))
    image.write_bytes(b"tampered")
    assert verify_ready_package(destination)["status"] == "FAIL"

    destination2 = tmp_path / "SAILOR_READY_v2.0_second"
    build_ready_package(source, destination2, execute=True, approve_copy=True)
    (destination2 / "rawdata_BIDS.tar.bz2").write_bytes(b"forbidden")
    assert verify_ready_package(destination2)["status"] == "FAIL"


def test_ready_builder_rolls_back_staging_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ready_source(tmp_path)
    destination = tmp_path / "SAILOR_READY_v2.0"
    source_file = next(source.glob("02_PREPROCESSED_MRI/**/*.npy"))
    original_hash = sha256_file(source_file)
    calls = {"count": 0}
    real_copy = __import__("shutil").copyfile

    def failing_copy(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected copy failure")
        return real_copy(src, dst)

    monkeypatch.setattr("sailor.packaging.build.shutil.copyfile", failing_copy)
    with pytest.raises(OSError):
        build_ready_package(source, destination, execute=True, approve_copy=True)
    assert not destination.exists()
    assert not destination.with_name(f".{destination.name}.build").exists()
    assert sha256_file(source_file) == original_hash


def test_ready_builder_does_not_replace_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ready_source(tmp_path)
    destination = tmp_path / "SAILOR_READY_v2.0"
    from sailor.packaging import build as build_module

    real_rename = build_module._rename_no_replace

    def racing_rename(staging: Path, final: Path) -> None:
        final.mkdir()
        (final / "owner.txt").write_text("external", encoding="utf-8")
        real_rename(staging, final)

    monkeypatch.setattr(
        "sailor.packaging.build._rename_no_replace",
        racing_rename,
    )
    with pytest.raises((StopProtocolError, FileExistsError, OSError)):
        build_ready_package(source, destination, execute=True, approve_copy=True)
    assert (destination / "owner.txt").read_text(encoding="utf-8") == "external"
