from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sailor.config import Settings
from sailor.contracts import assert_mask_contract, assert_volume_contract
from sailor.errors import StopProtocolError
from sailor.paths import assert_writable_target, create_output_tree
from sailor.profiling import profile_stage
from sailor.reporting import (
    load_dashboard,
    reconcile_completion_records,
    write_json,
)


def test_output_tree_is_idempotent_and_separate_from_legacy(
    synthetic_project: tuple[Settings, Path],
) -> None:
    settings, _ = synthetic_project
    first = create_output_tree(settings)
    second = create_output_tree(settings)
    assert first == second
    assert (settings.dataset_root / "00_CANONICAL" / "v2_pointers.json").is_file()
    assert not (settings.legacy_root / "00_CANONICAL").exists()


def test_write_firewall_rejects_legacy_path(
    synthetic_project: tuple[Settings, Path],
) -> None:
    settings, _ = synthetic_project
    with pytest.raises(StopProtocolError):
        assert_writable_target(settings.legacy_root / "forbidden.json", settings)


def test_volume_and_mask_contracts() -> None:
    volume = np.ones((4, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    assert_volume_contract(volume, name="volume")
    assert_mask_contract(mask, name="mask")
    with pytest.raises(ValueError):
        assert_volume_contract(np.ones((4, 4)), name="not_3d")
    with pytest.raises(ValueError):
        assert_mask_contract(np.full((4, 4, 4), 2), name="not_binary")


def test_target_lock_cannot_be_changed(tmp_path: Path) -> None:
    settings = Settings(
        dataset_root=tmp_path / "output",
        legacy_root=tmp_path / "legacy",
        primary_target_mask="ONCO",
        production_lock=False,
    )
    with pytest.raises(StopProtocolError):
        settings.validate()

    fold_settings = Settings(
        dataset_root=tmp_path / "output",
        legacy_root=tmp_path / "legacy",
        outer_folds=2,
        production_lock=False,
    )
    with pytest.raises(StopProtocolError):
        fold_settings.validate()


def test_profiler_labels_synthetic_measurement_honestly() -> None:
    result = profile_stage(
        3,
        lambda: "ok",
        measurement_scope="synthetic_fixture",
    )
    assert result.value == "ok"
    assert result.measurements["profiled"] is True
    assert result.measurements["measurement_scope"] == "synthetic_fixture"
    assert result.measurements["compute_mode"] == "CPU-only"


def test_guard_module_imports_in_clean_interpreter_without_cycle() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sailor.guards import guard_g1; assert callable(guard_g1)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_dashboard_uses_latest_state_and_reconciles_stale_failure(
    synthetic_project: tuple[Settings, Path],
) -> None:
    settings, _ = synthetic_project
    create_output_tree(settings)
    state_dir = settings.dataset_root / "01_DATA_FOUNDATION" / "state"
    failed = {
        "section": 5,
        "status": "failed",
        "guards_failed": ["G10"],
        "timestamp": "2026-08-12T00:00:00+00:00",
    }
    complete = {
        "section": 5,
        "status": "complete",
        "guards_failed": [],
        "timestamp": "2026-08-12T01:00:00+00:00",
    }
    stale_path = state_dir / "section_05_failed.json"
    write_json(stale_path, failed, settings)
    write_json(state_dir / "section_05_complete.json", complete, settings)

    dashboard = load_dashboard(settings)
    assert dashboard["failed_guards"] == []
    assert len(dashboard["sections"]) == 1
    assert dashboard["sections"][0]["status"] == "complete"

    removed = reconcile_completion_records(settings)
    assert removed == [str(stale_path)]
    assert not stale_path.exists()
