"""Thin-notebook section dispatch for Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sailor.bootstrap import bootstrap_runtime
from sailor.config import Settings
from sailor.data.audit import run_stage1_audit
from sailor.reporting import load_dashboard


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase1_section(
    section_id: int,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    active = settings or Settings.from_env()
    if section_id == 1:
        state = bootstrap_runtime(settings=active, mount_drive=True, install_missing=True)
        return {
            "section": 1,
            "runtime": state.runtime,
            "dependencies": state.dependencies,
            "resource_card": {
                "compute_mode": "CPU-only",
                "profiled": False,
                "requirements": "UNMEASURED",
                "fresh_runtime_safe": True,
            },
        }
    if section_id == 2:
        active.validate()
        return {
            "section": 2,
            "configuration": active.to_dict(),
            "resource_card": {
                "compute_mode": "CPU-only",
                "profiled": False,
                "requirements": "UNMEASURED",
                "fresh_runtime_safe": True,
            },
        }
    if section_id not in range(3, 10):
        raise ValueError("Phase 1 supports section IDs 01–09 only.")

    foundation = active.dataset_root / "01_DATA_FOUNDATION"
    qc = active.dataset_root / "06_QC_REPORTS"
    manifest_path = foundation / "v2_dataset_manifest.json"
    qc_path = qc / "v2_stage1_qc_report.json"
    if force or not manifest_path.exists() or not qc_path.exists():
        run_stage1_audit(active)

    manifest = _read(manifest_path)
    qc_report = _read(qc_path)
    section_payloads = {
        3: _read(foundation / "v2_canonical_manifest.json"),
        4: _read(foundation / "v2_canonical_manifest.json"),
        5: {
            "overview": manifest["overview"],
            "missing": manifest["missing"],
            "inventory": manifest["inventory"],
        },
        6: {
            "target": manifest["target"],
            "guard": next(g for g in qc_report["guards"] if g["guard_id"] == "G1"),
        },
        7: {
            "raw_mni_links": manifest["raw_mni_links"],
            "guard": next(g for g in qc_report["guards"] if g["guard_id"] == "G8"),
        },
        8: {
            "delta_t": manifest["delta_t"],
            "guard": next(g for g in qc_report["guards"] if g["guard_id"] == "G7"),
        },
        9: {
            "dose": manifest["dose"],
            "guard": next(
                g for g in qc_report["guards"] if g["guard_id"] == "DOSE_PREREQ"
            ),
        },
    }
    return {
        "section": section_id,
        "payload": section_payloads[section_id],
        "dashboard": load_dashboard(active),
        "resource_card": {
            "compute_mode": "CPU-only",
            "profiled": False,
            "requirements": "UNMEASURED",
            "fresh_runtime_safe": True,
        },
    }


def run_phase2_section(
    section_id: int,
    *,
    settings: Settings | None = None,
    execute: bool = False,
    extraction_approved: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    from sailor.preprocessing.stage2 import run_stage2_section

    active = settings or Settings.from_env()
    return run_stage2_section(
        section_id,
        active,
        execute=execute,
        extraction_approved=extraction_approved,
        force=force,
    )
