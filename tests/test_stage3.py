from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from sailor.baselines.io import load_phase2_artefacts
from sailor.baselines.persistence import persistence_prediction
from sailor.config import Settings
from sailor.constants import FOLD_SCHEME, PREPROCESSING_VERSION
from sailor.data.splits import generate_nested_cv_manifest
from sailor.errors import StopProtocolError
from sailor.evaluation.metrics import dice_coefficient, relative_volume_error
from sailor.evaluation.patient_stats import (
    aggregate_patient_scores,
    bootstrap_patient_mean,
    holm_adjust,
    minimum_detectable_effect,
    paired_patient_bootstrap,
)
from sailor.experiments.stage3 import _assert_fold_topology, run_stage3_section
from sailor.experiments.train import build_input_volume, delta_value
from sailor.guards import guard_g3, guard_g4
from sailor.paths import create_output_tree


def _mask(shape: tuple[int, int, int], offset: int = 0) -> np.ndarray:
    array = np.zeros(shape, dtype=np.uint8)
    array[1 + offset : 4 + offset, 1:4, 1:4] = 1
    return array


def write_phase2_fixture(tmp_path: Path) -> Settings:
    settings = Settings.for_testing(tmp_path / "output", tmp_path / "legacy")
    create_output_tree(settings)
    shape = (8, 8, 8)
    version = PREPROCESSING_VERSION
    mri_root = settings.dataset_root / "02_PREPROCESSED_MRI" / version
    mask_root = settings.dataset_root / "03_TUMOR_MASKS" / version
    records = []
    windows = []
    for patient_index in range(1, 5):
        subject = f"sub-{patient_index:02d}"
        sessions = []
        for session_index in range(1, 4):
            session = f"ses-{session_index:02d}"
            mri_path = mri_root / subject / session / "T1c-icor.npy"
            mask_path = mask_root / subject / session / "CL-enhancing-t1wc.npy"
            mri_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mri = np.full(shape, 0.2 * session_index, dtype=np.float32)
            mri[1:4, 1:4, 1:4] = 1.0 + 0.1 * session_index
            mask = _mask(shape, offset=0 if session_index < 3 else min(patient_index, 2))
            np.save(mri_path, mri)
            np.save(mask_path, mask)
            records.append(
                {
                    "subject": subject,
                    "raw_subject": subject,
                    "raw_session": session,
                    "mni_session": session,
                    "mri_output": str(mri_path),
                    "mask_output": str(mask_path),
                    "shape": list(shape),
                    "spacing": [1.0, 1.0, 1.0],
                    "affine_hash": "synthetic",
                }
            )
            sessions.append(session)
        windows.append(
            {
                "window_id": f"win-{subject}",
                "subject": subject,
                "history_raw_sessions": [f"{subject}/{sessions[0]}", f"{subject}/{sessions[1]}"],
                "history_mni_sessions": sessions[:2],
                "target_raw_session": f"{subject}/{sessions[2]}",
                "target_mni_session": sessions[2],
                "history_delta_days": [60.0, 30.0],
                "target_delta_days": 30.0 + patient_index,
                "timing_provenance": "approximate_mni_intervals",
                "treatment_status": "TMZ",
                "treatment_missing": False,
                "patient_weight": 1.0,
            }
        )
    windows_manifest = {
        "timing_provenance": "approximate_mni_intervals",
        "n_windows": len(windows),
        "n_patients": 4,
        "windows": windows,
        "content_hash": "synthetic",
    }
    folds = generate_nested_cv_manifest(
        windows_manifest,
        master_seed=1337,
        outer_folds=2,
        repeats=1,
        inner_folds=2,
        fold_scheme=FOLD_SCHEME,
    )
    preprocessing = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "n_records": len(records),
        "records": records,
        "selected_sequence": "T1c-icor",
    }
    paths = {
        "preprocessing": mri_root / "v2_preprocessing_manifest.json",
        "windows": settings.dataset_root
        / "04_LONGITUDINAL_WINDOWS"
        / version
        / "v2_windows_manifest.json",
        "cv": settings.dataset_root
        / "04_LONGITUDINAL_WINDOWS"
        / version
        / "v2_cv_manifest.json",
        "qc": settings.dataset_root / "06_QC_REPORTS" / "v2_phase2_qc_report.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                preprocessing
                if path == paths["preprocessing"]
                else windows_manifest
                if path == paths["windows"]
                else folds
                if path == paths["cv"]
                else {"status": "PASS"},
                indent=2,
            ),
            encoding="utf-8",
        )
    return settings


def test_dice_and_volume_error() -> None:
    target = _mask((8, 8, 8))
    assert dice_coefficient(target, target) == 1.0
    assert relative_volume_error(target, target) == 0.0
    empty = np.zeros((8, 8, 8), dtype=np.uint8)
    other = _mask((8, 8, 8), offset=1)
    assert dice_coefficient(empty, target) == 0.0
    assert dice_coefficient(other, target) < 1.0


def test_patient_bootstrap_does_not_use_sessions() -> None:
    scores = {"sub-01": 0.2, "sub-02": 0.8}
    result = bootstrap_patient_mean(scores, replicates=200, seed=1337)
    assert result["n_patients"] == 2
    assert result["unit"] == "patient"
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def test_paired_bootstrap_detects_clear_gain() -> None:
    left = {f"sub-{index:02d}": 0.9 for index in range(1, 8)}
    right = {f"sub-{index:02d}": 0.2 for index in range(1, 8)}
    result = paired_patient_bootstrap(left, right, replicates=500, seed=1337)
    assert result["beats"] is True
    assert result["ci95"][0] > 0


def test_paired_bootstrap_indistinguishable_when_equal() -> None:
    scores = {f"sub-{index:02d}": 0.5 for index in range(1, 8)}
    result = paired_patient_bootstrap(scores, scores, replicates=500, seed=1337)
    assert result["beats"] is False
    assert result["ci95"][0] <= 0 <= result["ci95"][1]


def test_holm_and_mde() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    mde = minimum_detectable_effect(25, 0.15)
    assert mde == pytest.approx(0.084, abs=0.01)


def test_g3_requires_persistence_bar_but_allows_negative_result() -> None:
    comparisons = {
        "rung_summaries": {"C-1": {"mean": 0.7, "ci95": [0.6, 0.8]}},
        "paired": {
            "C0_minus_C-1": {
                "beats": False,
                "ci95": [-0.1, 0.05],
                "p_bootstrap": 0.4,
            }
        },
    }
    guard = guard_g3(comparisons, required_models=("C0",))
    assert guard.status == "PASS"
    assert "indistinguishable" in guard.summary.lower() or "statistically" in guard.summary.lower()
    missing = guard_g3({"rung_summaries": {}, "paired": {}}, required_models=("C0",))
    assert missing.status == "FAIL"


def test_g4_decorative_is_a_finding_not_a_failed_guard() -> None:
    comparisons = {
        "paired": {
            "C1_minus_C1_constant": {
                "beats": False,
                "ci95": [-0.02, 0.03],
                "p_bootstrap": 0.8,
            }
        }
    }
    guard = guard_g4(comparisons)
    assert guard.status == "PASS"
    assert guard.details["decorative"] is True
    assert guard_g4({"paired": {}}).status == "FAIL"


def test_persistence_copies_last_history_mask(tmp_path: Path) -> None:
    settings = write_phase2_fixture(tmp_path)
    artefacts = load_phase2_artefacts(settings)
    window = artefacts["windows"]["windows"][0]
    prediction = persistence_prediction(
        artefacts, window, dataset_root=settings.dataset_root
    )
    last = np.load(
        artefacts["sessions"][(window["subject"], window["history_mni_sessions"][-1])][
            "mask_output"
        ]
    )
    target = np.load(
        artefacts["sessions"][(window["subject"], window["target_mni_session"])][
            "mask_output"
        ]
    )
    assert np.array_equal(prediction, last)
    assert not np.array_equal(prediction, target)


def test_c0_delta_channel_is_zero_and_c1_uses_days(tmp_path: Path) -> None:
    settings = write_phase2_fixture(tmp_path)
    artefacts = load_phase2_artefacts(settings)
    window = artefacts["windows"]["windows"][0]
    volume_c0, _ = build_input_volume(
        artefacts, window, dataset_root=settings.dataset_root, mode="C0"
    )
    volume_c1, _ = build_input_volume(
        artefacts, window, dataset_root=settings.dataset_root, mode="C1"
    )
    assert volume_c0.shape[0] == 3
    assert np.allclose(volume_c0[2], 0.0)
    assert np.allclose(volume_c1[2], delta_value(window, "C1", None))
    constant = build_input_volume(
        artefacts,
        window,
        dataset_root=settings.dataset_root,
        mode="C1_constant",
        constant_days=10.0,
    )[0]
    assert np.allclose(constant[2], 10.0 / 365.0)


def test_fold_topology_rejects_train_test_overlap() -> None:
    fold = {
        "train_patients": ["sub-01", "sub-02"],
        "test_patients": ["sub-01"],
        "inner_folds": [
            {
                "train_patients": ["sub-02"],
                "validation_patients": ["sub-01"],
            }
        ],
    }
    with pytest.raises(StopProtocolError):
        _assert_fold_topology(fold)


def test_patient_aggregation_equals_patients_not_windows() -> None:
    rows = [
        {"subject": "sub-01", "dice": 1.0},
        {"subject": "sub-01", "dice": 0.0},
        {"subject": "sub-02", "dice": 0.5},
    ]
    scores = aggregate_patient_scores(rows)
    assert scores == {"sub-01": 0.5, "sub-02": 0.5}


def test_phase3_dry_run_does_not_train(tmp_path: Path) -> None:
    settings = write_phase2_fixture(tmp_path)
    result = run_stage3_section(14, settings, execute=False)
    assert result["mode"] == "DRY_RUN"
    assert result["n_patients"] == 4
    assert not (settings.dataset_root / "07_BASELINE_RESULTS" / "p3.0" / "c0_window_metrics.json").exists()
    assert (
        settings.dataset_root / "07_BASELINE_RESULTS" / "p3.0" / "persistence_baseline.json"
    ).exists() is False


def test_section15_requires_section14(tmp_path: Path) -> None:
    settings = write_phase2_fixture(tmp_path)
    with pytest.raises(StopProtocolError):
        run_stage3_section(15, settings, execute=True, budget={"outer_epochs": 1})


@pytest.mark.skipif(find_spec("torch") is None, reason="C0/C1 require torch")
def test_section14_runs_tiny_learned_baseline(tmp_path: Path) -> None:
    settings = write_phase2_fixture(tmp_path)
    budget = {
        "lr_grid": (1e-3,),
        "inner_epochs": 1,
        "outer_epochs": 1,
        "patch_size": 8,
        "bootstrap_replicates": 200,
    }
    result = run_stage3_section(14, settings, execute=True, budget=budget)
    assert result["mode"] == "EXECUTE"
    assert result["guard"]["status"] == "PASS"
    assert "C-1" in result["comparisons"]
    assert "C0" in result["comparisons"]
    assert (
        settings.dataset_root / "07_BASELINE_RESULTS" / "p3.0" / "persistence_baseline.json"
    ).exists() is False
    assert result["mde"]["empirical_persistence_sd"] >= 0
    assert "empirical_mde_dice" in result["mde"]
