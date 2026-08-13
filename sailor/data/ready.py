"""Read-only loader for a portable SAILOR_READY_v2.0 package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ReadyDataset:
    """Load frozen package artefacts relative to a configurable DATA_ROOT."""

    def __init__(self, data_root: str | Path, *, allow_pending: bool = False) -> None:
        self.root = Path(data_root).expanduser().resolve()
        self.package_manifest = self._read_json("manifests/package_manifest.json")
        if self.package_manifest.get("package_kind") != "SAILOR_READY":
            raise ValueError("DATA_ROOT is not a SAILOR_READY package.")
        status = self.package_manifest.get("package_status")
        if status != "READY_TO_TRAIN" and not allow_pending:
            raise ValueError(f"Package is not READY_TO_TRAIN: {status}")
        if (
            self.package_manifest.get("data_version") != "v2.0"
            or self.package_manifest.get("preprocessing_version") != "p2.0"
        ):
            raise ValueError("Unsupported dataset or preprocessing version.")

        relative = self.package_manifest["relative_paths"]
        self.preprocessing = self._read_json(relative["preprocessing_manifest"])
        self.windows_manifest = self._read_json(relative["windows"])
        self.folds_manifest = self._read_json(relative["folds"])
        self.treatment_manifest = self._read_json(relative["treatment"])
        self.timing_cache = self._read_json(relative["timing"])
        self.sessions = {
            (record["subject"], record["mni_session"]): record
            for record in self.preprocessing["records"]
        }
        self.windows = {
            record["window_id"]: record
            for record in self.windows_manifest["windows"]
        }
        self._validate_metadata()

    def _resolve(self, relative: str | Path) -> Path:
        path = (self.root / Path(relative)).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path escapes DATA_ROOT: {relative}") from exc
        return path

    def _read_json(self, relative: str | Path) -> dict[str, Any]:
        path = self._resolve(relative)
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_metadata(self) -> None:
        counts = self.package_manifest["counts"]
        if len(self.sessions) != counts["sessions"]:
            raise ValueError("Session count differs from package manifest.")
        if len(self.windows) != counts["longitudinal_windows"]:
            raise ValueError("Window count differs from package manifest.")
        for record in self.preprocessing["records"]:
            for field in ("mri_output", "mask_output"):
                value = record[field]
                if Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ValueError(f"Non-portable path in preprocessing manifest: {value}")
        for window in self.windows.values():
            required = [
                *window["history_mni_sessions"],
                window["target_mni_session"],
            ]
            for session in required:
                if (window["subject"], session) not in self.sessions:
                    raise ValueError(
                        f"Window {window['window_id']} references missing session."
                    )

    def load_session(
        self,
        subject: str,
        mni_session: str,
        *,
        mmap_mode: str | None = "r",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if mmap_mode not in {None, "r"}:
            raise ValueError("ReadyDataset only permits read-only mmap_mode='r' or None.")
        record = self.sessions[(subject, mni_session)]
        image = np.load(self._resolve(record["mri_output"]), mmap_mode=mmap_mode)
        mask = np.load(self._resolve(record["mask_output"]), mmap_mode=mmap_mode)
        return image, mask, record

    def get_window(self, window_id: str) -> dict[str, Any]:
        return self.windows[window_id]

    def iter_windows(self) -> Iterator[dict[str, Any]]:
        return iter(self.windows.values())

    def get_outer_fold(self, repeat: int, outer_fold: int) -> dict[str, Any]:
        return next(
            fold
            for fold in self.folds_manifest["folds"]
            if fold["repeat"] == repeat and fold["outer_fold"] == outer_fold
        )

    def session_metadata(self) -> list[dict[str, str]]:
        relative = self.package_manifest["relative_paths"]["session_metadata"]
        with self._resolve(relative).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle))
