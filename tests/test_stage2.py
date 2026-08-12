from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from sailor.config import Settings
from sailor.data.splits import generate_nested_cv_manifest
from sailor.data.windows import build_longitudinal_windows
from sailor.errors import StopProtocolError
from sailor.guards import guard_g5_stage2
from sailor.preprocessing.normalize import robust_scale_volume
from sailor.preprocessing.pipeline import (
    execute_preprocessing,
    validate_preprocessing_cache,
)
from sailor.preprocessing.policy import assert_plan_ready, build_preprocessing_plan
from sailor.preprocessing.stage2 import run_stage2_section


def _inventory_record(
    path: str,
    *,
    subject: str = "sub-01",
    session: str = "ses-01",
    sequence: str | None = None,
    classification: str = "MRI",
    dtype: str = "float32",
    nonzero: int | None = None,
) -> dict[str, object]:
    return {
        "path": path,
        "source": "archive:derivatives.tar.bz2",
        "subject": subject,
        "session": session,
        "sequence": sequence,
        "classification": classification,
        "shape": [4, 4, 4],
        "spacing": [1.0, 1.0, 1.0],
        "affine": np.eye(4).tolist(),
        "dtype": dtype,
        "minimum": 0.0,
        "maximum": 10.0 if classification == "MRI" else 1.0,
        "finite": True,
        "nonzero_voxels": nonzero,
        "total_voxels": 64 if nonzero is not None else None,
    }


def _phase1_selection_fixture() -> tuple[dict, dict]:
    records = [
        _inventory_record(
            "root/sub-01/ses-01/T1c-icor.nii.gz",
            sequence="T1c-icor",
        ),
        _inventory_record(
            "root/sub-01/ses-01/ContrastEnhancedMask-CL.nii.gz",
            classification="CL:enhancing_t1wc",
            dtype="uint8",
            nonzero=8,
        ),
        _inventory_record(
            "root/sub-01/ses-01/BrainExtractionMask.nii.gz",
            classification="AUXILIARY_MASK",
            dtype="uint8",
            nonzero=64,
        ),
    ]
    manifest = {
        "data_version": "v2.0",
        "implementation_id": "cursor_primary",
        "PRIMARY_TARGET_MASK": "CL",
        "PRIMARY_TARGET_COMPONENT": "enhancing_t1wc",
        "inventory": {"records": records},
        "dose": {"n_files": 0, "n_patients": 0, "files": []},
    }
    qc = {
        "data_version": "v2.0",
        "implementation_id": "cursor_primary",
        "PRIMARY_TARGET_MASK": "CL",
        "PRIMARY_TARGET_COMPONENT": "enhancing_t1wc",
        "failed_guards": [],
        "guards": [
            {
                "guard_id": "G9",
                "details": {
                    "surviving_raw_mni_pairs": [
                        {"raw": "sub-01/ses-01", "mni": "sub-01/ses-01"}
                    ]
                },
            }
        ],
    }
    return manifest, qc


def test_phase2_policy_selects_exact_t1c_and_geometry(tmp_path: Path) -> None:
    manifest, qc = _phase1_selection_fixture()
    settings = Settings.for_testing(tmp_path / "output", tmp_path / "legacy")
    plan = build_preprocessing_plan(manifest, qc, settings)
    assert_plan_ready(plan)
    assert plan["n_selected_sessions"] == 1
    assert plan["selected_sequence"] == "T1c-icor"
    assert plan["issues"] == []


def test_phase2_policy_reports_geometry_mismatch(tmp_path: Path) -> None:
    manifest, qc = _phase1_selection_fixture()
    manifest["inventory"]["records"][1]["affine"][0][3] = 4.0
    settings = Settings.for_testing(tmp_path / "output", tmp_path / "legacy")
    plan = build_preprocessing_plan(manifest, qc, settings)
    assert plan["issues"][0]["reason"] == "geometry_mismatch"
    with pytest.raises(StopProtocolError):
        assert_plan_ready(plan)


def test_phase2_policy_rejects_mismatched_provenance(tmp_path: Path) -> None:
    manifest, qc = _phase1_selection_fixture()
    qc["implementation_id"] = "other"
    settings = Settings.for_testing(tmp_path / "output", tmp_path / "legacy")
    with pytest.raises(StopProtocolError):
        build_preprocessing_plan(manifest, qc, settings)


def test_per_volume_normalization_is_finite_and_background_zero() -> None:
    volume = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    brain = np.zeros((4, 4, 4), dtype=np.uint8)
    brain[1:4, 1:4, 1:4] = 1
    normalized, parameters = robust_scale_volume(volume, brain)
    assert np.isfinite(normalized).all()
    assert np.count_nonzero(normalized[brain == 0]) == 0
    assert parameters["scope"] == "single_volume_brain_mask"


def test_variable_history_windows_link_treatment_and_dates() -> None:
    start = datetime(2020, 1, 1)
    preprocessed = []
    dates = []
    treatment = []
    for index in range(4):
        session = f"ses-{index + 1:02d}"
        preprocessed.append(
            {
                "subject": "sub-01",
                "raw_session": session,
                "mni_session": session,
            }
        )
        dates.append(
            {
                "subject": "sub-01",
                "session": session,
                "datetime": (start + timedelta(days=14 * index)).isoformat(),
            }
        )
        treatment.append(
            {
                "subject": "sub-01",
                "session": session,
                "status": "CRT",
                "missing": False,
            }
        )
    phase1 = {
        "delta_t": {"status": "EXACT_SOURCE_FOUND", "dates": dates},
        "overview": {"treatment_records": treatment},
    }
    built = build_longitudinal_windows(
        phase1,
        preprocessed,
        min_history_scans=2,
    )
    windows = built["windows"]
    assert windows["n_windows"] == 2
    assert len(windows["windows"][0]["history_mni_sessions"]) == 2
    assert len(windows["windows"][1]["history_mni_sessions"]) == 3
    assert windows["windows"][0]["target_delta_days"] == 14.0
    assert windows["windows"][0]["patient_weight"] == 0.5


def test_windows_refuse_to_guess_missing_dates() -> None:
    phase1 = {
        "delta_t": {"status": "APPROXIMATE_ONLY", "dates": []},
        "overview": {"treatment_records": []},
    }
    with pytest.raises(StopProtocolError):
        build_longitudinal_windows(
            phase1,
            [
                {
                    "subject": "sub-01",
                    "raw_session": "ses-01",
                    "mni_session": "ses-01",
                }
            ],
            min_history_scans=2,
        )


def test_windows_reject_unrecognized_treatment() -> None:
    phase1 = {
        "delta_t": {
            "status": "EXACT_SOURCE_FOUND",
            "dates": [
                {
                    "subject": "sub-01",
                    "session": f"ses-{index:02d}",
                    "datetime": f"2020-01-{index:02d}T00:00:00",
                }
                for index in range(1, 4)
            ],
        },
        "overview": {
            "treatment_records": [
                {
                    "subject": "sub-01",
                    "session": f"ses-{index:02d}",
                    "status": "UNRECOGNIZED:new",
                    "missing": False,
                }
                for index in range(1, 4)
            ]
        },
    }
    records = [
        {
            "subject": "sub-01",
            "raw_subject": "sub-01",
            "raw_session": f"ses-{index:02d}",
            "mni_session": f"ses-{index:02d}",
        }
        for index in range(1, 4)
    ]
    with pytest.raises(StopProtocolError):
        build_longitudinal_windows(phase1, records, min_history_scans=2)


def test_windows_preserve_different_raw_subject_identifier() -> None:
    phase1 = {
        "delta_t": {
            "status": "EXACT_SOURCE_FOUND",
            "dates": [
                {
                    "subject": "raw-01",
                    "session": f"ses-{index:02d}",
                    "datetime": f"2020-01-{index:02d}T00:00:00",
                }
                for index in range(1, 4)
            ],
        },
        "overview": {
            "treatment_records": [
                {
                    "subject": "raw-01",
                    "session": f"ses-{index:02d}",
                    "status": "CRT",
                    "missing": False,
                }
                for index in range(1, 4)
            ]
        },
    }
    records = [
        {
            "subject": "sub-01",
            "raw_subject": "raw-01",
            "raw_session": f"ses-{index:02d}",
            "mni_session": f"ses-{index:02d}",
        }
        for index in range(1, 4)
    ]
    built = build_longitudinal_windows(phase1, records, min_history_scans=2)
    assert built["windows"]["n_windows"] == 1
    assert built["windows"]["windows"][0]["target_raw_session"].startswith(
        "raw-01/"
    )


def _many_patient_windows() -> dict:
    windows = []
    for patient_index in range(10):
        subject = f"sub-{patient_index + 1:02d}"
        for window_index in range((patient_index % 3) + 1):
            windows.append(
                {
                    "window_id": f"{subject}-{window_index}",
                    "subject": subject,
                    "target_mni_session": f"ses-{window_index + 3:02d}",
                }
            )
    return {"windows": windows}


def test_nested_cv_is_reproducible_and_g5_catches_overlap(tmp_path: Path) -> None:
    windows = _many_patient_windows()
    cv_a = generate_nested_cv_manifest(
        windows,
        master_seed=1337,
        outer_folds=5,
        repeats=3,
        inner_folds=4,
        fold_scheme="5fold_x3seeds_nested4",
    )
    cv_b = generate_nested_cv_manifest(
        windows,
        master_seed=1337,
        outer_folds=5,
        repeats=3,
        inner_folds=4,
        fold_scheme="5fold_x3seeds_nested4",
    )
    assert cv_a["content_hash"] == cv_b["content_hash"]
    settings = Settings.for_testing(tmp_path / "output", tmp_path / "legacy")
    preprocessing_records = [
        {
            "subject": window["subject"],
            "mni_session": session,
            "mri_output": "mri.npy",
            "mask_output": "mask.npy",
            "checksums": {"mri_sha256": "x", "mask_sha256": "y"},
            "mri_source": "canonical",
        }
        for window in windows["windows"]
        for session in [window["target_mni_session"]]
    ]
    preprocessing = {
        "records": preprocessing_records,
        "normalization_scope": "single_volume_brain_mask",
    }
    assert guard_g5_stage2(settings, windows, cv_a, preprocessing).status == "PASS"
    assert guard_g5_stage2(
        settings,
        windows,
        cv_a,
        {"records": [], "normalization_scope": "single_volume_brain_mask"},
    ).status == "FAIL"

    invalid = deepcopy(cv_a)
    patient = invalid["folds"][0]["test_patients"][0]
    invalid["folds"][0]["train_patients"].append(patient)
    assert guard_g5_stage2(settings, windows, invalid, preprocessing).status == "FAIL"
    wrong_topology = deepcopy(cv_a)
    wrong_topology["outer_folds"] = 2
    assert guard_g5_stage2(
        settings, windows, wrong_topology, preprocessing
    ).status == "FAIL"
    duplicate_inner = deepcopy(cv_a)
    duplicate_inner["folds"][0]["inner_folds"] = [
        deepcopy(duplicate_inner["folds"][0]["inner_folds"][0])
        for _ in range(4)
    ]
    assert guard_g5_stage2(
        settings, windows, duplicate_inner, preprocessing
    ).status == "FAIL"


def _archive_member(name: str, array: np.ndarray) -> tuple[tarfile.TarInfo, io.BytesIO]:
    payload = gzip.compress(nib.Nifti1Image(array, np.eye(4)).to_bytes())
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info, io.BytesIO(payload)


def _write_canonical_manifest(output: Path, archive: Path) -> None:
    actual = hashlib.sha512(archive.read_bytes()).hexdigest()
    (output / "01_DATA_FOUNDATION" / "v2_canonical_manifest.json").write_text(
        json.dumps(
            {
                "verification": {
                    "files": [
                        {
                            "name": "derivatives.tar.bz2",
                            "status": "VERIFIED",
                            "actual_sha512": actual,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def test_selective_extraction_writes_only_approved_arrays(tmp_path: Path) -> None:
    output = tmp_path / "output"
    legacy = tmp_path / "legacy"
    (output / "01_DATA_FOUNDATION").mkdir(parents=True)
    (output / "06_QC_REPORTS").mkdir(parents=True)
    legacy.mkdir()
    settings = Settings.for_testing(output, legacy)
    manifest, qc = _phase1_selection_fixture()
    (output / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (output / "06_QC_REPORTS" / "v2_stage1_qc_report.json").write_text(
        json.dumps(qc),
        encoding="utf-8",
    )
    mri = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    brain = np.ones((4, 4, 4), dtype=np.uint8)
    with tarfile.open(legacy / "derivatives.tar.bz2", "w:bz2") as archive:
        for name, array in (
            ("root/sub-01/ses-01/T1c-icor.nii.gz", mri),
            ("root/sub-01/ses-01/ContrastEnhancedMask-CL.nii.gz", mask),
            ("root/sub-01/ses-01/BrainExtractionMask.nii.gz", brain),
        ):
            info, payload = _archive_member(name, array)
            archive.addfile(info, payload)
    _write_canonical_manifest(output, legacy / "derivatives.tar.bz2")
    result = execute_preprocessing(settings, extraction_approved=True)
    record = result["manifest"]["records"][0]
    assert Path(record["mri_output"]).is_file()
    assert Path(record["mask_output"]).is_file()
    assert np.load(record["mri_output"]).dtype == np.float32
    assert np.load(record["mask_output"]).dtype == np.uint8
    assert not (legacy / "02_PREPROCESSED_MRI").exists()
    repeated = execute_preprocessing(settings, extraction_approved=True)
    assert repeated["manifest"]["records"][0]["checksums"] == record["checksums"]
    validate_preprocessing_cache(settings, repeated["manifest"])
    original_bytes = Path(record["mri_output"]).read_bytes()
    with tarfile.open(legacy / "derivatives.tar.bz2", "w:bz2") as archive:
        for name, array in (
            ("root/sub-01/ses-01/T1c-icor.nii.gz", mri),
            ("root/sub-01/ses-01/ContrastEnhancedMask-CL.nii.gz", mask),
        ):
            info, payload = _archive_member(name, array)
            archive.addfile(info, payload)
    with pytest.raises(StopProtocolError):
        execute_preprocessing(settings, extraction_approved=True)
    assert Path(record["mri_output"]).read_bytes() == original_bytes
    Path(record["mri_output"]).write_bytes(b"tampered")
    with pytest.raises(StopProtocolError):
        validate_preprocessing_cache(settings, repeated["manifest"])


def test_phase2_modules_import_in_clean_interpreter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sailor.preprocessing.stage2 import run_stage2_section; "
            "assert callable(run_stage2_section)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_end_to_end_stage2_synthetic_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    legacy = tmp_path / "legacy"
    (output / "01_DATA_FOUNDATION").mkdir(parents=True)
    (output / "06_QC_REPORTS").mkdir(parents=True)
    legacy.mkdir()
    settings = Settings.for_testing(output, legacy)
    inventory = []
    pairs = []
    treatment = []
    dates = []
    archive_items: list[tuple[str, np.ndarray]] = []
    start = datetime(2020, 1, 1)
    for patient_index in range(10):
        subject = f"sub-{patient_index + 1:02d}"
        for session_index in range(3):
            session = f"ses-{session_index + 1:02d}"
            prefix = f"root/{subject}/{session}"
            inventory.extend(
                [
                    _inventory_record(
                        f"{prefix}/T1c-icor.nii.gz",
                        subject=subject,
                        session=session,
                        sequence="T1c-icor",
                    ),
                    _inventory_record(
                        f"{prefix}/ContrastEnhancedMask-CL.nii.gz",
                        subject=subject,
                        session=session,
                        classification="CL:enhancing_t1wc",
                        dtype="uint8",
                        nonzero=8,
                    ),
                    _inventory_record(
                        f"{prefix}/BrainExtractionMask.nii.gz",
                        subject=subject,
                        session=session,
                        classification="AUXILIARY_MASK",
                        dtype="uint8",
                        nonzero=64,
                    ),
                ]
            )
            pairs.append(
                {
                    "raw": f"{subject}/{session}",
                    "mni": f"{subject}/{session}",
                }
            )
            treatment.append(
                {
                    "subject": subject,
                    "session": session,
                    "status": "CRT",
                    "missing": False,
                }
            )
            dates.append(
                {
                    "subject": subject,
                    "session": session,
                    "datetime": (
                        start + timedelta(days=14 * session_index)
                    ).isoformat(),
                }
            )
            mri = (
                np.arange(64, dtype=np.float32).reshape(4, 4, 4)
                + patient_index
                + session_index
            )
            mask = np.zeros((4, 4, 4), dtype=np.uint8)
            mask[1:3, 1:3, 1:3] = 1
            archive_items.extend(
                [
                    (f"{prefix}/T1c-icor.nii.gz", mri),
                    (f"{prefix}/ContrastEnhancedMask-CL.nii.gz", mask),
                    (f"{prefix}/BrainExtractionMask.nii.gz", np.ones((4, 4, 4), dtype=np.uint8)),
                ]
            )
    manifest = {
        "data_version": "v2.0",
        "implementation_id": "cursor_primary",
        "PRIMARY_TARGET_MASK": "CL",
        "PRIMARY_TARGET_COMPONENT": "enhancing_t1wc",
        "inventory": {"records": inventory},
        "overview": {"treatment_records": treatment},
        "delta_t": {"status": "EXACT_SOURCE_FOUND", "dates": dates},
        "dose": {"n_files": 0, "n_patients": 0, "files": []},
    }
    qc = {
        "data_version": "v2.0",
        "implementation_id": "cursor_primary",
        "PRIMARY_TARGET_MASK": "CL",
        "PRIMARY_TARGET_COMPONENT": "enhancing_t1wc",
        "failed_guards": [],
        "guards": [
            {
                "guard_id": "G9",
                "details": {"surviving_raw_mni_pairs": pairs},
            }
        ],
    }
    (output / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (output / "06_QC_REPORTS" / "v2_stage1_qc_report.json").write_text(
        json.dumps(qc),
        encoding="utf-8",
    )
    with tarfile.open(legacy / "derivatives.tar.bz2", "w:bz2") as archive:
        for name, array in archive_items:
            info, payload = _archive_member(name, array)
            archive.addfile(info, payload)
    _write_canonical_manifest(output, legacy / "derivatives.tar.bz2")

    monkeypatch.setattr(
        "sailor.reporting.git_state",
        lambda: ("clean-commit", "main", False),
    )
    section10 = run_stage2_section(
        10,
        settings,
        execute=True,
        extraction_approved=True,
    )
    section11 = run_stage2_section(11, settings)
    section12 = run_stage2_section(12, settings)
    section13 = run_stage2_section(13, settings)
    assert section10["n_windows"] == 10
    assert section11["cv"]["n_patients"] == 10
    assert section12["guard"]["status"] == "PASS"
    assert section13["qc"]["failed_guards"] == []
    treatment_path = (
        output
        / "05_TREATMENT_DATA"
        / settings.preprocessing_version
        / "v2_treatment_manifest.json"
    )
    tampered = json.loads(treatment_path.read_text(encoding="utf-8"))
    tampered["records"][0]["status"] = "TMZ"
    treatment_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(StopProtocolError):
        run_stage2_section(11, settings)
