"""Deterministic patient-level nested cross-validation manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np

from sailor.errors import StopProtocolError
from sailor.schemas import FoldRecord


def derive_repeat_seeds(master_seed: int, repeats: int) -> list[int]:
    sequence = np.random.SeedSequence(master_seed)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(repeats)
    ]


def _balanced_assignments(
    patient_counts: dict[str, int],
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    if len(patient_counts) < folds:
        raise StopProtocolError(
            f"Only {len(patient_counts)} patients are available for {folds} folds.",
            "At least one test fold would be empty.",
            "Reduce folds before model results or restore eligible patients.",
        )
    rng = np.random.default_rng(seed)
    tie_break = {patient: float(rng.random()) for patient in patient_counts}
    ordered = sorted(
        patient_counts,
        key=lambda patient: (-patient_counts[patient], tie_break[patient], patient),
    )
    totals = [0] * folds
    sizes = [0] * folds
    assignments: dict[str, int] = {}
    for patient in ordered:
        fold = min(range(folds), key=lambda index: (totals[index], sizes[index], index))
        assignments[patient] = fold
        totals[fold] += patient_counts[patient]
        sizes[fold] += 1
    return assignments


def generate_nested_cv_manifest(
    windows_manifest: dict[str, Any],
    *,
    master_seed: int,
    outer_folds: int,
    repeats: int,
    inner_folds: int,
    fold_scheme: str,
) -> dict[str, Any]:
    windows = windows_manifest.get("windows", [])
    patient_counts = Counter(item["subject"] for item in windows)
    if not patient_counts:
        raise StopProtocolError(
            "No patients are available for CV generation.",
            "Patient-level evaluation cannot be constructed.",
            "Build and validate longitudinal windows first.",
        )
    seeds = derive_repeat_seeds(master_seed, repeats)
    records: list[FoldRecord] = []
    patients = set(patient_counts)
    for repeat, seed in enumerate(seeds):
        assignments = _balanced_assignments(
            dict(patient_counts),
            folds=outer_folds,
            seed=seed,
        )
        for outer_fold in range(outer_folds):
            test_patients = sorted(
                patient for patient, fold in assignments.items() if fold == outer_fold
            )
            train_patients = sorted(patients - set(test_patients))
            inner_counts = {
                patient: patient_counts[patient] for patient in train_patients
            }
            inner_assignments = _balanced_assignments(
                inner_counts,
                folds=inner_folds,
                seed=seed + outer_fold + 1,
            )
            nested: list[dict[str, Any]] = []
            for inner_fold in range(inner_folds):
                validation = sorted(
                    patient
                    for patient, fold in inner_assignments.items()
                    if fold == inner_fold
                )
                training = sorted(set(train_patients) - set(validation))
                nested.append(
                    {
                        "inner_fold": inner_fold,
                        "train_patients": training,
                        "validation_patients": validation,
                    }
                )
            records.append(
                FoldRecord(
                    repeat=repeat,
                    seed=seed,
                    outer_fold=outer_fold,
                    train_patients=train_patients,
                    test_patients=test_patients,
                    inner_folds=nested,
                )
            )
    payload = {
        "fold_scheme": fold_scheme,
        "master_seed": master_seed,
        "repeat_seeds": seeds,
        "outer_folds": outer_folds,
        "outer_repeats": repeats,
        "inner_folds": inner_folds,
        "balancing_inputs": "eligible_window_counts_only",
        "n_patients": len(patient_counts),
        "patient_window_counts": dict(sorted(patient_counts.items())),
        "folds": [record.to_dict() for record in records],
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(
            payload["folds"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload
