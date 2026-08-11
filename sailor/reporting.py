"""Atomic JSON reporting and completion-state helpers."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.constants import SECTION_STAGE
from sailor.paths import assert_writable_target
from sailor.schemas import CompletionRecord, GuardResult


def write_json(path: Path, payload: Any, settings: Settings) -> None:
    assert_writable_target(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_value(args: list[str], default: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def git_state() -> tuple[str, str, bool]:
    commit = _git_value(["rev-parse", "HEAD"], "UNCOMMITTED")
    branch = _git_value(["branch", "--show-current"], "UNKNOWN")
    dirty = bool(_git_value(["status", "--porcelain"], ""))
    return commit, branch, dirty


def build_completion_record(
    section: int,
    settings: Settings,
    guards: list[GuardResult],
    *,
    n_patients: int | None,
    n_sessions: int | None,
) -> CompletionRecord:
    commit, branch, dirty = git_state()
    failed = [g.guard_id for g in guards if g.status == "FAIL"]
    passed = [g.guard_id for g in guards if g.status == "PASS"]
    return CompletionRecord(
        section=section,
        stage=SECTION_STAGE[section],
        status="complete" if not failed else "failed",
        owner="primary_implementation",
        data_version=settings.data_version,
        model_version="NOT_APPLICABLE_PHASE_1",
        preprocessing_version="UNSET",
        feature_shape=[],
        primary_target_mask=settings.primary_target_mask,
        primary_target_component=settings.primary_target_component,
        conditioning_rung="NOT_APPLICABLE_PHASE_1",
        fold_scheme="UNSET",
        guards_passed=passed,
        guards_failed=failed,
        n_patients=n_patients,
        n_sessions=n_sessions,
        n_pairs=None,
        seed=settings.seed,
        gpu="CPU-only",
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        implementation_id=settings.implementation_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def persist_completion_records(
    settings: Settings,
    guards_by_section: dict[int, list[GuardResult]],
    *,
    n_patients: int | None,
    n_sessions: int | None,
) -> None:
    state_dir = settings.dataset_root / "01_DATA_FOUNDATION" / "state"
    for section in range(1, 10):
        record = build_completion_record(
            section,
            settings,
            guards_by_section.get(section, []),
            n_patients=n_patients,
            n_sessions=n_sessions,
        )
        suffix = "complete" if record.status == "complete" else "failed"
        write_json(
            state_dir / f"section_{section:02d}_{suffix}.json",
            record.to_dict(),
            settings,
        )


def load_dashboard(settings: Settings) -> dict[str, Any]:
    state_dir = settings.dataset_root / "01_DATA_FOUNDATION" / "state"
    records: list[dict[str, Any]] = []
    for path in sorted(state_dir.glob("section_??_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    failures = [
        {"section": record["section"], "guards": record["guards_failed"]}
        for record in records
        if record.get("guards_failed")
    ]
    return {
        "failed_guards": failures,
        "sections": records,
        "source": str(state_dir),
    }
