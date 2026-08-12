"""Centralized, locked configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sailor.constants import (
    DATA_VERSION,
    FOLD_SCHEME,
    IMPLEMENTATION_ID,
    INNER_FOLDS,
    MIN_HISTORY_SCANS,
    OUTER_FOLDS,
    OUTER_REPEATS,
    PREPROCESSING_VERSION,
    PRIMARY_INPUT_SEQUENCE,
    PRIMARY_TARGET_COMPONENT,
    PRIMARY_TARGET_MASK,
    PRODUCTION_DATASET_ROOT,
    PRODUCTION_LEGACY_ROOT,
)
from sailor.errors import StopProtocolError


@dataclass(frozen=True)
class Settings:
    dataset_root: Path
    legacy_root: Path
    data_version: str = DATA_VERSION
    implementation_id: str = IMPLEMENTATION_ID
    primary_target_mask: str = PRIMARY_TARGET_MASK
    primary_target_component: str = PRIMARY_TARGET_COMPONENT
    preprocessing_version: str = PREPROCESSING_VERSION
    primary_input_sequence: str = PRIMARY_INPUT_SEQUENCE
    fold_scheme: str = FOLD_SCHEME
    outer_folds: int = OUTER_FOLDS
    outer_repeats: int = OUTER_REPEATS
    inner_folds: int = INNER_FOLDS
    min_history_scans: int = MIN_HISTORY_SCANS
    seed: int = 1337
    production_lock: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        dataset_root = Path(
            os.environ.get("SAILOR_DATASET_ROOT", PRODUCTION_DATASET_ROOT)
        )
        legacy_root = Path(
            os.environ.get("SAILOR_LEGACY_ROOT", PRODUCTION_LEGACY_ROOT)
        )
        settings = cls(dataset_root=dataset_root, legacy_root=legacy_root)
        settings.validate()
        return settings

    @classmethod
    def for_testing(cls, dataset_root: Path, legacy_root: Path) -> "Settings":
        return cls(
            dataset_root=dataset_root,
            legacy_root=legacy_root,
            production_lock=False,
        )

    def validate(self) -> None:
        if self.primary_target_mask != PRIMARY_TARGET_MASK or (
            self.primary_target_component != PRIMARY_TARGET_COMPONENT
        ):
            raise StopProtocolError(
                "Primary target differs from CL / enhancing_t1wc.",
                "The run violates the pre-registered target lock and is invalid.",
                "Restore the locked target constants before running the audit.",
            )
        if self.implementation_id != IMPLEMENTATION_ID:
            raise StopProtocolError(
                "Implementation ID is not cursor_primary.",
                "Artefacts cannot be attributed to the designated primary implementation.",
                "Use the locked implementation ID and regenerate the artefacts.",
            )
        if self.preprocessing_version != PREPROCESSING_VERSION:
            raise StopProtocolError(
                f"Preprocessing version is {self.preprocessing_version}, not {PREPROCESSING_VERSION}.",
                "Phase 2 outputs could mix incompatible preprocessing definitions.",
                "Restore the locked preprocessing version and rebuild Phase 2 artefacts.",
            )
        if (
            self.primary_input_sequence != PRIMARY_INPUT_SEQUENCE
            or self.fold_scheme != FOLD_SCHEME
            or self.min_history_scans != MIN_HISTORY_SCANS
            or self.outer_folds != OUTER_FOLDS
            or self.outer_repeats != OUTER_REPEATS
            or self.inner_folds != INNER_FOLDS
            or self.seed != 1337
        ):
            raise StopProtocolError(
                "Phase 2 input, fold, or history locks were changed.",
                "Window and evaluation manifests would no longer match the approved plan.",
                "Restore the Phase 2 locks before generating artefacts.",
            )
        if self.production_lock and self.dataset_root.as_posix() != PRODUCTION_DATASET_ROOT:
            raise StopProtocolError(
                f"DATASET_ROOT is {self.dataset_root}, not {PRODUCTION_DATASET_ROOT}.",
                "Outputs could mix with another implementation or personal path.",
                "Unset SAILOR_DATASET_ROOT or set it to the locked production root.",
            )
        if self.production_lock and self.legacy_root.as_posix() != PRODUCTION_LEGACY_ROOT:
            raise StopProtocolError(
                f"LEGACY_ROOT is {self.legacy_root}, not {PRODUCTION_LEGACY_ROOT}.",
                "The provenance firewall would no longer reference the approved legacy input.",
                "Unset SAILOR_LEGACY_ROOT or set it to the locked read-only root.",
            )
        try:
            self.dataset_root.resolve().relative_to(self.legacy_root.resolve())
        except ValueError:
            return
        raise StopProtocolError(
            "DATASET_ROOT is inside LEGACY_ROOT.",
            "The audit could modify or mix immutable legacy inputs.",
            "Use separate read and write roots.",
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset_root"] = str(self.dataset_root)
        result["legacy_root"] = str(self.legacy_root)
        return result
