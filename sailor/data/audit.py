"""CPU-only Stage-1 data-foundation audit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.constants import CANONICAL_FILES
from sailor.data.inventory import inventory_nifti, summarize_inventory
from sailor.data.metadata import (
    discover_exact_dates,
    read_canonical_tsv,
    read_tsv,
    summarize_missing,
    summarize_overview,
    summarize_raw_mni_links,
)
from sailor.data.provenance import (
    classify_legacy_root,
    storage_report,
    verify_canonical_files,
)
from sailor.data.targets import dose_inventory, target_inventory
from sailor.errors import StopProtocolError
from sailor.guards import (
    guard_g1,
    guard_g5,
    guard_g7,
    guard_g8,
    guard_g9,
    guard_g10,
)
from sailor.paths import (
    create_output_tree,
    snapshot_tree,
    verify_snapshot_unchanged,
)
from sailor.reporting import persist_completion_records, write_json
from sailor.schemas import GuardResult


def _provenance_guard(report: dict[str, Any]) -> GuardResult:
    invalid = [
        item
        for item in report["files"]
        if item["status"] in {"MISSING", "MISMATCH"}
    ]
    return GuardResult(
        "PROVENANCE",
        "FAIL" if invalid else "PASS",
        (
            f"{len(invalid)} canonical files are missing or checksum-invalid."
            if invalid
            else "All listed canonical files are present and checksum-valid where indexed."
        ),
        {"invalid_files": invalid},
    )


def _metadata_guard(
    overview: dict[str, Any],
    missing: dict[str, Any],
) -> GuardResult:
    failures: list[str] = []
    if overview.get("n_rows", 0) == 0:
        failures.append("overview.tsv")
    if missing.get("n_rows", 0) == 0:
        failures.append("missing.tsv")
    return GuardResult(
        "METADATA",
        "FAIL" if failures else "PASS",
        (
            f"Required metadata could not be parsed: {', '.join(failures)}."
            if failures
            else "Required overview and exclusion metadata were parsed."
        ),
        {"unparseable_or_empty": failures},
    )


def _dose_guard(report: dict[str, Any]) -> GuardResult:
    available = report.get("n_files", 0) > 0
    return GuardResult(
        "DOSE_PREREQ",
        "PASS" if available else "FAIL",
        (
            f"Dose maps were measured for {report.get('n_patients', 0)} patients."
            if available
            else "No canonical dose maps were resolved; C3 cannot proceed."
        ),
        {
            "space_status": report.get("space_status"),
            "requires_registration": report.get("requires_registration"),
        },
    )


def _rank_gaps(
    canonical: dict[str, Any],
    target: dict[str, Any],
    delta: dict[str, Any],
    dose: dict[str, Any],
    legacy_root: Path,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    missing = [
        item["name"]
        for item in canonical["files"]
        if item["status"] == "MISSING"
    ]
    if missing:
        gaps.append(
            {
                "rank": len(gaps) + 1,
                "gap": "canonical files absent",
                "minimum_candidate": missing,
                "reason": "required by provenance and Stage-1 guards",
                "download_size": "UNVERIFIED",
                "approval_required": True,
            }
        )
    if target.get("n_primary_files", 0) == 0:
        gaps.append(
            {
                "rank": len(gaps) + 1,
                "gap": "locked CL/enhancing_t1wc masks unavailable",
                "minimum_candidate": ["derivatives.tar.bz2"],
                "reason": "G1 and every primary endpoint require MNI-only CL masks",
                "download_size": "UNVERIFIED",
                "approval_required": True,
            }
        )
    if delta.get("status") != "EXACT_SOURCE_FOUND":
        gaps.append(
            {
                "rank": len(gaps) + 1,
                "gap": "exact inter-exam dates unavailable after local-source audit",
                "minimum_candidate": ["rawdata.tar.bz2", "sourcedata.tar.bz2"],
                "reason": (
                    "Try rawdata first; fetch sourcedata only if the smaller exact-date "
                    "source cannot close G7."
                ),
                "download_size": "UNVERIFIED",
                "approval_required": True,
            }
        )
    if dose.get("n_files", 0) == 0:
        ambiguous = legacy_root / "dosemaps.tar"
        gaps.append(
            {
                "rank": len(gaps) + 1,
                "gap": "dose-map prerequisites unresolved",
                "minimum_candidate": (
                    ["verify existing dosemaps.tar provenance"]
                    if ambiguous.exists()
                    else ["minimum EBRAINS dose-map artefact — identify before download"]
                ),
                "reason": "C3/P2 require patient-specific spatial dose maps",
                "download_size": (
                    ambiguous.stat().st_size if ambiguous.is_file() else "UNVERIFIED"
                ),
                "approval_required": True,
            }
        )
    return gaps


def run_stage1_audit(
    settings: Settings | None = None,
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    active = settings or Settings.from_env()
    active.validate()
    before = snapshot_tree(active.legacy_root)
    directories = create_output_tree(active)

    classification = classify_legacy_root(active.legacy_root)
    canonical = verify_canonical_files(active.legacy_root)
    overview = summarize_overview(read_tsv(active.legacy_root / "overview.tsv"))
    missing = summarize_missing(read_tsv(active.legacy_root / "missing.tsv"))
    link_rows, link_source = read_canonical_tsv(
        active.legacy_root,
        "raw-mni-link.tsv",
        archive_candidates=("derivatives.tar.bz2",),
    )
    links = summarize_raw_mni_links(link_rows)
    links["source"] = link_source
    records, inventory_process = inventory_nifti(active.legacy_root)
    inventory = summarize_inventory(records)
    target = target_inventory(records)
    dose = dose_inventory(records)
    delta = discover_exact_dates(active.legacy_root)
    storage = storage_report(active.dataset_root, active.legacy_root)

    guards = [
        _provenance_guard(canonical),
        _metadata_guard(overview, missing),
        guard_g1(records),
        guard_g5(active, records),
        guard_g7(delta),
        guard_g8(links, overview),
        guard_g9(missing, overview, records),
        guard_g10(records),
        _dose_guard(dose),
    ]
    failed = [guard for guard in guards if guard.status == "FAIL"]

    foundation_dir = active.dataset_root / "01_DATA_FOUNDATION"
    qc_dir = active.dataset_root / "06_QC_REPORTS"
    canonical_manifest = {
        "data_version": active.data_version,
        "implementation_id": active.implementation_id,
        "PRIMARY_TARGET_MASK": active.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": active.primary_target_component,
        "legacy_root": str(active.legacy_root),
        "copy_policy": "read-only references; no canonical data copied",
        "classification": classification,
        "verification": canonical,
    }
    dataset_manifest = {
        "data_version": active.data_version,
        "implementation_id": active.implementation_id,
        "PRIMARY_TARGET_MASK": active.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": active.primary_target_component,
        "overview": overview,
        "missing": missing,
        "raw_mni_links": links,
        "inventory": inventory,
        "target": target,
        "dose": dose,
        "delta_t": delta,
    }
    qc_report = {
        "data_version": active.data_version,
        "implementation_id": active.implementation_id,
        "PRIMARY_TARGET_MASK": active.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": active.primary_target_component,
        "compute_mode": "CPU-only",
        "profiled": False,
        "resources": "UNMEASURED",
        "inventory_process": inventory_process,
        "guards": [guard.to_dict() for guard in guards],
        "failed_guards": [guard.guard_id for guard in failed],
    }
    gap_report = {
        "download_performed": False,
        "policy": "No download without a separate justification and approval.",
        "source_attempts_for_exact_delta_t": delta.get("source_attempts", []),
        "storage": storage,
        "ranked_gaps": _rank_gaps(
            canonical,
            target,
            delta,
            dose,
            active.legacy_root,
        ),
        "canonical_names_required": list(CANONICAL_FILES),
    }

    write_json(
        foundation_dir / "v2_canonical_manifest.json",
        canonical_manifest,
        active,
    )
    write_json(
        foundation_dir / "v2_dataset_manifest.json",
        dataset_manifest,
        active,
    )
    write_json(qc_dir / "v2_stage1_qc_report.json", qc_report, active)
    write_json(qc_dir / "v2_gap_report.json", gap_report, active)

    guards_by_section = {
        3: [guards[0]],
        4: [guards[0]],
        5: [guards[1], guards[6], guards[7]],
        6: [guards[2]],
        7: [guards[5]],
        8: [guards[4]],
        9: [guards[8]],
    }
    persist_completion_records(
        active,
        guards_by_section,
        n_patients=overview.get("n_patients"),
        n_sessions=overview.get("n_sessions"),
    )
    verify_snapshot_unchanged(before, snapshot_tree(active.legacy_root))

    result = {
        "directories": directories,
        "canonical_manifest": str(foundation_dir / "v2_canonical_manifest.json"),
        "dataset_manifest": str(foundation_dir / "v2_dataset_manifest.json"),
        "qc_report": str(qc_dir / "v2_stage1_qc_report.json"),
        "gap_report": str(qc_dir / "v2_gap_report.json"),
        "failed_guards": [guard.guard_id for guard in failed],
    }
    if failed and raise_on_failure:
        first = failed[0]
        raise StopProtocolError(
            first.summary,
            (
                "Stage 1 is not valid; downstream preprocessing, cohort construction, "
                "and modelling must not begin."
            ),
            (
                "Inspect v2_stage1_qc_report.json and v2_gap_report.json, apply only "
                "the minimum approved fix, then rerun Stage 1."
            ),
        )
    return result


def main() -> int:
    try:
        result = run_stage1_audit()
    except StopProtocolError as exc:
        print(exc.render(), file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
