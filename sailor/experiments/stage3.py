"""Fresh-runtime orchestration for notebook sections 14–15."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sailor.baselines.io import load_phase2_artefacts, windows_for_patients
from sailor.baselines.persistence import iter_persistence_scores
from sailor.config import Settings
from sailor.constants import (
    BASELINE_DT_PERTURBATION_DAYS,
    BASELINE_INNER_EPOCHS,
    BASELINE_LR_GRID,
    BASELINE_OUTER_EPOCHS,
    BASELINE_PATCH_SIZE,
    DATA_VERSION,
    MODEL_VERSION,
    PATIENT_BOOTSTRAP_REPLICATES,
    PREPROCESSING_VERSION,
)
from sailor.errors import StopProtocolError
from sailor.evaluation.patient_stats import (
    aggregate_patient_scores,
    bootstrap_patient_mean,
    holm_adjust,
    illustrative_mde_table,
    minimum_detectable_effect,
    paired_patient_bootstrap,
)
from sailor.experiments.checkpointing import (
    build_identity,
    checkpoint_root,
    outer_run_dir,
)
from sailor.experiments.train import (
    predict_windows,
    select_learning_rate,
    train_unet,
)
from sailor.guards import guard_g3, guard_g4
from sailor.reporting import load_dashboard, persist_section_completion, write_json
from sailor.schemas import GuardResult


FORBIDDEN_RESULT_NAMES = {"persistence_baseline.json"}


def _result_root(settings: Settings) -> Path:
    return settings.dataset_root / "07_BASELINE_RESULTS" / "p3.0"


def _budget(settings: Settings, override: dict[str, Any] | None) -> dict[str, Any]:
    locked = {
        "lr_grid": BASELINE_LR_GRID,
        "inner_epochs": BASELINE_INNER_EPOCHS,
        "outer_epochs": BASELINE_OUTER_EPOCHS,
        "patch_size": BASELINE_PATCH_SIZE,
        "bootstrap_replicates": PATIENT_BOOTSTRAP_REPLICATES,
    }
    if override is None:
        return locked
    if settings.production_lock:
        raise StopProtocolError(
            "A runtime budget override was supplied under production lock.",
            "Phase-3 epoch or bootstrap locks could silently change.",
            "Use Settings.for_testing for synthetic budgets.",
        )
    merged = dict(locked)
    merged.update(override)
    return merged


def _assert_safe_outputs(root: Path) -> None:
    for name in FORBIDDEN_RESULT_NAMES:
        if (root / name).exists():
            raise StopProtocolError(
                f"Refusing to use quarantined baseline filename {name}.",
                "Old TaDiff persistence outputs could be mistaken for Phase 3.",
                "Write C−1 results as cminus1_window_metrics.json.",
            )


def _assert_fold_topology(fold: dict[str, Any]) -> None:
    train = set(fold["train_patients"])
    test = set(fold["test_patients"])
    if not train or not test:
        raise StopProtocolError(
            "An outer fold has an empty train or test patient list.",
            "Patient-level evaluation would be undefined.",
            "Restore the frozen Phase-2 CV manifest.",
        )
    if train & test:
        raise StopProtocolError(
            "Outer-fold train and test patients overlap.",
            "G5 leakage would invalidate every baseline comparison.",
            "Restore the frozen Phase-2 CV manifest.",
        )
    for inner in fold["inner_folds"]:
        inner_train = set(inner["train_patients"])
        inner_val = set(inner["validation_patients"])
        if inner_train & inner_val or inner_train & test or inner_val & test:
            raise StopProtocolError(
                "Inner-loop patients overlap validation or outer test.",
                "Hyperparameter selection would leak.",
                "Restore the frozen nested CV manifest.",
            )
        if not inner_train or not inner_val:
            raise StopProtocolError(
                "An inner fold has an empty train or validation patient list.",
                "Learning-rate selection would be undefined.",
                "Restore the frozen nested CV manifest.",
            )
        if inner_train | inner_val != train:
            raise StopProtocolError(
                "An inner fold does not partition the outer training patients.",
                "Learning-rate selection would leak or drop patients.",
                "Restore the frozen nested CV manifest.",
            )


def _median_delta(windows: list[dict[str, Any]]) -> float:
    return float(np.median([window["target_delta_days"] for window in windows]))


def _score_persistence(
    artefacts: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    return list(
        iter_persistence_scores(artefacts, windows, dataset_root=dataset_root)
    )


def _score_learned(
    artefacts: dict[str, Any],
    fold: dict[str, Any],
    all_windows: list[dict[str, Any]],
    *,
    settings: Settings,
    mode: str,
    budget: dict[str, Any],
    seed: int,
    perturbations: tuple[tuple[str, float], ...] = (),
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    _assert_fold_topology(fold)
    dataset_root = settings.dataset_root
    train_windows = windows_for_patients(all_windows, fold["train_patients"])
    test_windows = windows_for_patients(all_windows, fold["test_patients"])
    train_subjects = {window["subject"] for window in train_windows}
    test_subjects = {window["subject"] for window in test_windows}
    if train_subjects & test_subjects:
        raise StopProtocolError(
            f"{mode} train/test window subjects overlap.",
            "A learned baseline would leak patient identity.",
            "Filter windows by the frozen patient lists.",
        )
    fold_root = checkpoint_root(
        settings, mode, int(fold["repeat"]), int(fold["outer_fold"])
    )
    complete_path = fold_root / "fold_complete.json"
    selection_identity = {
        "mode": mode,
        "repeat": int(fold["repeat"]),
        "outer_fold": int(fold["outer_fold"]),
        "seed": int(seed),
        "lr_grid": [float(value) for value in budget["lr_grid"]],
        "inner_epochs": int(budget["inner_epochs"]),
        "outer_epochs": int(budget["outer_epochs"]),
        "patch_size": int(budget["patch_size"]),
        "train_patients": sorted(fold["train_patients"]),
        "test_patients": sorted(fold["test_patients"]),
        "inner_folds": fold["inner_folds"],
        "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    selection_hash = hashlib.sha256(
        json.dumps(selection_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if complete_path.is_file() and not force:
        stored = json.loads(complete_path.read_text(encoding="utf-8"))
        if stored.get("selection_hash") != selection_hash:
            raise StopProtocolError(
                f"Completed-fold marker for {mode} repeat {fold['repeat']} "
                f"outer {fold['outer_fold']} does not match the locked identity.",
                "Skipping it could mix two experiments.",
                "Inspect CHECKPOINTS/p3.0 and do not delete valid markers to chase a better score.",
            )
        return stored["predictions"]
    constant_days = _median_delta(train_windows) if mode == "C1_constant" else None
    learning_rate = select_learning_rate(
        artefacts,
        fold["inner_folds"],
        train_windows,
        dataset_root=dataset_root,
        mode=mode,
        seed=seed,
        constant_days=constant_days,
        budget=budget,
        settings=settings,
        fold_root=fold_root,
        identity_base={
            "repeat": int(fold["repeat"]),
            "outer_fold": int(fold["outer_fold"]),
            "fold_scheme": settings.fold_scheme,
            "test_patients": list(fold["test_patients"]),
            "selection_hash": selection_hash,
        },
    )
    outer_identity = build_identity(
        mode=mode,
        split_role="OUTER_TRAINING",
        repeat=int(fold["repeat"]),
        outer_fold=int(fold["outer_fold"]),
        inner_fold=None,
        seed=seed,
        learning_rate=float(learning_rate),
        epochs=int(budget["outer_epochs"]),
        patch_size=int(budget["patch_size"]),
        fold_scheme=settings.fold_scheme,
        train_patients=list(fold["train_patients"]),
        validation_patients=None,
        test_patients=list(fold["test_patients"]),
    )
    fitted = train_unet(
        artefacts,
        train_windows,
        dataset_root=dataset_root,
        mode=mode,
        seed=seed,
        constant_days=constant_days,
        epochs=int(budget["outer_epochs"]),
        learning_rate=learning_rate,
        patch_size=int(budget["patch_size"]),
        settings=settings,
        run_dir=outer_run_dir(fold_root),
        identity=outer_identity,
    )
    if set(fitted["train_patients"]) & set(fold["test_patients"]):
        raise StopProtocolError(
            f"{mode} trained on outer-test patients.",
            "The comparison against persistence would be invalid.",
            "Retrain using only outer-training patients.",
        )
    primary_rung = "C1_constant" if mode == "C1_constant" else mode
    outputs = {
        "primary": predict_windows(
            fitted["model"],
            artefacts,
            test_windows,
            dataset_root=dataset_root,
            mode=mode,
            constant_days=constant_days,
            patch_size=int(budget["patch_size"]),
            rung=primary_rung,
        )
    }
    for row in outputs["primary"]:
        row["split"] = "OUTER_TEST"
    for label, shift in perturbations:
        outputs[label] = predict_windows(
            fitted["model"],
            artefacts,
            test_windows,
            dataset_root=dataset_root,
            mode=mode,
            constant_days=constant_days,
            patch_size=int(budget["patch_size"]),
            delta_perturbation_days=shift,
            rung=label,
        )
        for row in outputs[label]:
            row["split"] = "OUTER_TEST"
    primary = outputs["primary"]
    write_json(
        fold_root / "fold_summary.json",
        {
            "status": "complete",
            "mode": mode,
            "repeat": fold["repeat"],
            "outer_fold": fold["outer_fold"],
            "seed": seed,
            "selected_lr": learning_rate,
            "selected_lr_source": "INNER_VALIDATION",
            "best_monitor_score": fitted.get("validation_dice"),
            "best_monitor_epoch": fitted.get("best_epoch"),
            "best_metric_source": (
                "INNER_VALIDATION"
                if fitted.get("validation_dice") is not None
                else "TRAINING_MONITOR_ONLY"
            ),
            "scientific_checkpoint": "final.pt",
            "checkpoint_used": "final.pt",
            "outer_test_metrics": {
                "dice_mean": float(np.mean([row["dice"] for row in primary])),
                "iou_mean": float(np.mean([row["iou"] for row in primary])),
                "precision_mean": float(np.mean([row["precision"] for row in primary])),
                "recall_mean": float(np.mean([row["recall"] for row in primary])),
                "n_windows": len(primary),
                "split": "OUTER_TEST",
            },
            "training_seconds": fitted.get("cumulative_seconds"),
            "gpu": fitted.get("gpu"),
            "resumed_from_epoch": fitted.get("resumed_from_epoch"),
            "outer_test_used_for_selection": False,
            "outer_test_used_for_checkpoint_selection": False,
            "outer_test_used_for_early_stopping": False,
        },
        settings,
    )
    write_json(
        complete_path,
        {
            "status": "complete",
            "selection_hash": selection_hash,
            "identity": selection_identity,
            "predictions": outputs,
        },
        settings,
    )
    return outputs


def _oof_rows(
    artefacts: dict[str, Any],
    *,
    settings: Settings,
    mode: str,
    budget: dict[str, Any],
    seed: int,
    perturbations: tuple[tuple[str, float], ...] = (),
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    windows = artefacts["windows"]["windows"]
    collected: dict[str, list[dict[str, Any]]] = {"primary": []}
    for label, _ in perturbations:
        collected[label] = []
    for fold in artefacts["folds"]["folds"]:
        _assert_fold_topology(fold)
        test_windows = windows_for_patients(windows, fold["test_patients"])
        if mode == "C-1":
            scored = {
                "primary": _score_persistence(
                    artefacts, test_windows, dataset_root=settings.dataset_root
                )
            }
        else:
            scored = _score_learned(
                artefacts,
                fold,
                windows,
                settings=settings,
                mode=mode,
                budget=budget,
                seed=seed + 17 * int(fold["repeat"]) + int(fold["outer_fold"]),
                perturbations=perturbations,
                force=force,
            )
        for key, rows in scored.items():
            for row in rows:
                row["repeat"] = fold["repeat"]
                row["outer_fold"] = fold["outer_fold"]
            collected.setdefault(key, []).extend(rows)
    return collected


def _patient_table(rows: list[dict[str, Any]]) -> dict[str, float]:
    return aggregate_patient_scores(rows, metric="dice")


def _summarize(
    rows_by_rung: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    patient_scores = {
        rung: _patient_table(rows) for rung, rows in rows_by_rung.items()
    }
    summaries = {
        rung: bootstrap_patient_mean(scores, replicates=replicates, seed=seed)
        for rung, scores in patient_scores.items()
    }
    pair_spec = [
        ("C0", "C-1"),
        ("C1", "C0"),
        ("C1", "C-1"),
        ("C1", "C1_constant"),
    ]
    paired: dict[str, Any] = {}
    for left, right in pair_spec:
        if left in patient_scores and right in patient_scores:
            paired[f"{left}_minus_{right}"] = paired_patient_bootstrap(
                patient_scores[left],
                patient_scores[right],
                replicates=replicates,
                seed=seed,
            )
    adjusted = holm_adjust(
        {name: item["p_bootstrap"] for name, item in paired.items()}
    )
    for name, value in adjusted.items():
        paired[name]["p_holm"] = value
    return {
        "patient_scores": patient_scores,
        "rung_summaries": summaries,
        "paired": paired,
    }


def _mde_payload(
    n_patients: int,
    persistence_scores: dict[str, float] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_patients": n_patients,
        "alpha": 0.05,
        "power": 0.80,
        "unit": "patient",
        "approximation": "normal_two_sided_paired",
        "illustrative": illustrative_mde_table(n_patients),
        "note": (
            "The normal approximation is slightly optimistic at n=25. "
            "A hoped-for Dice gain below the empirical MDE is not interpretable "
            "as an improvement."
        ),
    }
    if persistence_scores:
        values = list(persistence_scores.values())
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        payload["empirical_persistence_sd"] = sd
        payload["empirical_mde_dice"] = (
            minimum_detectable_effect(n_patients, sd) if sd > 0 else 0.0
        )
        if sd == 0.0:
            payload["empirical_mde_note"] = (
                "Patient-level persistence Dice has zero observed variance, "
                "so the empirical MDE is 0 and is not a useful power bound."
            )
    return payload


def _experiment_summary(
    settings: Settings,
    mode: str,
    comparisons: dict[str, Any],
    *,
    n_expected_folds: int,
) -> dict[str, Any]:
    safe_mode = mode.replace("C-1", "Cminus1")
    ckpt = settings.dataset_root / "CHECKPOINTS" / "p3.0" / safe_mode
    folds = []
    failures: list[dict[str, Any]] = []
    if ckpt.exists():
        for path in sorted(ckpt.glob("repeat*_outer*/fold_summary.json")):
            folds.append(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(ckpt.glob("repeat*_outer*/**/failures.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    failures.append(json.loads(line))
    resume_events = [
        {
            "repeat": item.get("repeat"),
            "outer_fold": item.get("outer_fold"),
            "resumed_from_epoch": item.get("resumed_from_epoch"),
        }
        for item in folds
        if int(item.get("resumed_from_epoch") or 0) > 0
    ]
    gpu_names = sorted(
        {
            str(item.get("gpu", {}).get("gpu_name"))
            for item in folds
            if item.get("gpu")
        }
    )
    peak_vram = max(
        (int(item.get("gpu", {}).get("peak_vram_bytes") or 0) for item in folds),
        default=0,
    )
    training_seconds = float(
        sum(float(item.get("training_seconds") or 0.0) for item in folds)
    )
    status = "complete" if len(folds) == n_expected_folds and not failures else (
        "failed" if failures else "incomplete"
    )
    return {
        "mode": mode,
        "status": status,
        "n_completed_folds": len(folds),
        "n_expected_folds": n_expected_folds,
        "folds": folds,
        "patient_level_outer_test": comparisons.get("patient_scores"),
        "confidence_intervals": comparisons.get("rung_summaries"),
        "paired": comparisons.get("paired"),
        "primary_metric": "patient_macro_dice",
        "metric_roles": {
            "TRAINING": "monitoring only",
            "INNER_VALIDATION": "LR selection and monitoring; never scientific reporting",
            "OUTER_TEST": "final evaluation only; never used for LR, epoch, checkpoint, or early stopping",
        },
        "outer_test_used_for_selection": False,
        "training_seconds_total": training_seconds,
        "gpu_names": gpu_names,
        "peak_vram_bytes": peak_vram,
        "resume_events": resume_events,
        "n_resume_events": len(resume_events),
        "failures": failures,
        "checkpoint_policy": {
            "latest": "every completed epoch",
            "best": "inner-validation monitor only",
            "final": "scientific weights after the locked epoch budget",
        },
    }


def _write_rung(path: Path, rows: list[dict[str, Any]], settings: Settings) -> None:
    if path.name in FORBIDDEN_RESULT_NAMES:
        raise StopProtocolError(
            f"Refusing to write {path.name}.",
            "That filename is quarantined from prior implementations.",
            "Use the Phase-3 p3.0 result names.",
        )
    write_json(path, {"rung_rows": rows, "n_rows": len(rows)}, settings)


def run_stage3_section(
    section_id: int,
    settings: Settings,
    *,
    execute: bool = False,
    force: bool = False,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if section_id not in {14, 15}:
        raise ValueError("Phase 3 supports section IDs 14–15 only.")
    settings.validate()
    artefacts = load_phase2_artefacts(settings)
    root = _result_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    _assert_safe_outputs(root)
    active_budget = _budget(settings, budget)
    n_patients = int(artefacts["windows"]["n_patients"])
    n_windows = int(artefacts["windows"]["n_windows"])
    plan = {
        "section": section_id,
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "fold_scheme": settings.fold_scheme,
        "n_patients": n_patients,
        "n_windows": n_windows,
        "timing_provenance": artefacts["windows"].get("timing_provenance"),
        "history_policy": "most_recent_two_eligible_scans",
        "rungs": ["C-1", "C0"] if section_id == 14 else ["C1", "C1_constant"],
        "mde": _mde_payload(n_patients, None),
        "budget": {
            key: value
            for key, value in active_budget.items()
            if key != "bootstrap_replicates" or True
        },
        "scientific_boundary": {
            "approved": ["persistence", "mri_history_only", "mri_plus_approximate_dt"],
            "not_approved": [
                "treatment_aware",
                "dose_aware",
                "exact_time",
                "causal_effect",
            ],
        },
    }
    write_json(root / "phase3_plan.json", plan, settings)
    if not execute:
        return {
            "section": section_id,
            "mode": "DRY_RUN",
            "plan": str(root / "phase3_plan.json"),
            "n_patients": n_patients,
            "n_windows": n_windows,
        }

    replicates = int(active_budget["bootstrap_replicates"])
    cminus1_path = root / "cminus1_window_metrics.json"
    if section_id == 14:
        if force or not cminus1_path.is_file():
            cminus1 = _oof_rows(
                artefacts,
                settings=settings,
                mode="C-1",
                budget=active_budget,
                seed=settings.seed,
                force=force,
            )["primary"]
            _write_rung(cminus1_path, cminus1, settings)
        else:
            cminus1 = json.loads(cminus1_path.read_text(encoding="utf-8"))["rung_rows"]
        c0 = _oof_rows(
            artefacts,
            settings=settings,
            mode="C0",
            budget=active_budget,
            seed=settings.seed,
            force=force,
        )["primary"]
        _write_rung(root / "c0_window_metrics.json", c0, settings)
        rows_by_rung = {"C-1": cminus1, "C0": c0}
        comparisons = _summarize(
            rows_by_rung, seed=settings.seed, replicates=replicates
        )
        mde = _mde_payload(n_patients, comparisons["patient_scores"]["C-1"])
        write_json(root / "mde.json", mde, settings)
        write_json(root / "section14_comparisons.json", comparisons, settings)
        experiment = _experiment_summary(
            settings,
            "C0",
            comparisons,
            n_expected_folds=len(artefacts["folds"]["folds"]),
        )
        write_json(root / "experiment_summary_C0.json", experiment, settings)
        guard = guard_g3(comparisons, required_models=("C0",))
        persist_section_completion(
            settings,
            14,
            [guard],
            n_patients=n_patients,
            n_sessions=int(artefacts["preprocessing"].get("n_records", 0)),
            n_pairs=n_windows,
            preprocessing_version=PREPROCESSING_VERSION,
            fold_scheme=settings.fold_scheme,
            model_version=MODEL_VERSION,
            conditioning_rung="C-1+C0",
        )
        return {
            "section": 14,
            "mode": "EXECUTE",
            "guard": guard.to_dict(),
            "mde": mde,
            "comparisons": comparisons["rung_summaries"],
            "paired": comparisons["paired"],
            "experiment_summary": experiment,
            "dashboard": load_dashboard(settings),
        }

    required = [
        cminus1_path,
        root / "c0_window_metrics.json",
        root / "section14_comparisons.json",
    ]
    if any(not path.is_file() for path in required):
        raise StopProtocolError(
            "Section 14 artefacts are missing.",
            "C1 cannot be interpreted without the persistence and MRI-only floor.",
            "Execute section 14 before section 15.",
        )
    cminus1 = json.loads(cminus1_path.read_text(encoding="utf-8"))["rung_rows"]
    c0 = json.loads((root / "c0_window_metrics.json").read_text(encoding="utf-8"))[
        "rung_rows"
    ]
    c1_scored = _oof_rows(
        artefacts,
        settings=settings,
        mode="C1",
        budget=active_budget,
        seed=settings.seed,
        force=force,
        perturbations=(
            ("C1_dt_minus7", -BASELINE_DT_PERTURBATION_DAYS),
            ("C1_dt_plus7", BASELINE_DT_PERTURBATION_DAYS),
        ),
    )
    c1 = c1_scored["primary"]
    _write_rung(root / "c1_window_metrics.json", c1, settings)
    c1_constant = _oof_rows(
        artefacts,
        settings=settings,
        mode="C1_constant",
        budget=active_budget,
        seed=settings.seed,
        force=force,
    )["primary"]
    _write_rung(root / "c1_constant_window_metrics.json", c1_constant, settings)
    rows_by_rung = {
        "C-1": cminus1,
        "C0": c0,
        "C1": c1,
        "C1_constant": c1_constant,
    }
    comparisons = _summarize(
        rows_by_rung, seed=settings.seed, replicates=replicates
    )
    g7_rows = {
        "minus7": c1_scored["C1_dt_minus7"],
        "plus7": c1_scored["C1_dt_plus7"],
    }
    for label, rows in g7_rows.items():
        _write_rung(root / f"c1_dt_{label}_window_metrics.json", rows, settings)
    comparisons["g7_sensitivity"] = {
        label: bootstrap_patient_mean(
            _patient_table(rows),
            replicates=replicates,
            seed=settings.seed,
        )
        for label, rows in g7_rows.items()
    }
    mde = _mde_payload(n_patients, comparisons["patient_scores"]["C-1"])
    write_json(root / "mde.json", mde, settings)
    write_json(root / "section15_comparisons.json", comparisons, settings)
    experiment_c1 = _experiment_summary(
        settings,
        "C1",
        comparisons,
        n_expected_folds=len(artefacts["folds"]["folds"]),
    )
    experiment_g4 = _experiment_summary(
        settings,
        "C1_constant",
        comparisons,
        n_expected_folds=len(artefacts["folds"]["folds"]),
    )
    write_json(root / "experiment_summary_C1.json", experiment_c1, settings)
    write_json(root / "experiment_summary_C1_constant.json", experiment_g4, settings)
    guards = [
        guard_g3(comparisons, required_models=("C0", "C1")),
        guard_g4(comparisons),
    ]
    persist_section_completion(
        settings,
        15,
        guards,
        n_patients=n_patients,
        n_sessions=int(artefacts["preprocessing"].get("n_records", 0)),
        n_pairs=n_windows,
        preprocessing_version=PREPROCESSING_VERSION,
        fold_scheme=settings.fold_scheme,
        model_version=MODEL_VERSION,
        conditioning_rung="C1",
    )
    qc = {
        "guards": [guard.to_dict() for guard in guards],
        "failed_guards": [guard.guard_id for guard in guards if guard.status == "FAIL"],
        "timing_provenance": artefacts["windows"].get("timing_provenance"),
        "mde": mde,
        "rung_summaries": comparisons["rung_summaries"],
        "paired": comparisons["paired"],
    }
    write_json(root / "v2_phase3_qc_report.json", qc, settings)
    return {
        "section": 15,
        "mode": "EXECUTE",
        "guards": [guard.to_dict() for guard in guards],
        "mde": mde,
        "comparisons": comparisons["rung_summaries"],
        "paired": comparisons["paired"],
        "experiment_summary_c1": experiment_c1,
        "experiment_summary_g4": experiment_g4,
        "dashboard": load_dashboard(settings),
    }
