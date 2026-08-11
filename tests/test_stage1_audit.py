from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from sailor.data.audit import run_stage1_audit
from sailor.data.inventory import classify_nifti, inventory_nifti
from sailor.data.metadata import (
    read_canonical_tsv,
    read_tsv,
    summarize_missing,
    summarize_overview,
    summarize_raw_mni_links,
)
from sailor.errors import StopProtocolError
from sailor.guards import guard_g1, guard_g8, guard_g9, guard_g10
from sailor.paths import snapshot_tree


def test_classification_resolves_locked_and_inventory_only_targets() -> None:
    assert (
        classify_nifti("sub-01_ses-01_CL_t1wc_enhancing_mask.nii.gz")
        == "CL:enhancing_t1wc"
    )
    assert (
        classify_nifti("sub-01_ses-01_CL_t2wflair_mask.nii.gz")
        == "CL:t2wflair_hyperintensity"
    )
    assert classify_nifti("sub-01_ses-01_ONCO_mask.nii.gz") == "ONCO"
    assert classify_nifti("sub-01_ses-01_dosemap.nii.gz") == "DOSE"


def test_metadata_parser_counts_patients_sessions_and_treatment(
    synthetic_project: tuple[object, Path],
) -> None:
    _, legacy = synthetic_project
    report = summarize_overview(read_tsv(legacy / "overview.tsv"))
    assert report["n_patients"] == 1
    assert report["n_sessions"] == 2
    assert report["treatment_counts"] == {"CRT": 1, "TMZ": 1}
    assert report["unknown_semantics"].startswith("missingness")


def test_raw_mni_guard_rejects_duplicate_mapping(
    synthetic_project: tuple[object, Path],
) -> None:
    _, legacy = synthetic_project
    rows = read_tsv(legacy / "raw-mni-link.tsv")
    rows.append(dict(rows[0]))
    links = summarize_raw_mni_links(rows)
    overview = summarize_overview(read_tsv(legacy / "overview.tsv"))
    result = guard_g8(links, overview)
    assert result.status == "FAIL"
    assert result.details["duplicates"]


def test_raw_mni_links_stream_from_canonical_derivatives_archive(
    tmp_path: Path,
) -> None:
    payload = (
        b"raw_session\tmni_session\n"
        b"sub-01/ses-raw01\tsub-01/ses-01\n"
    )
    archive_path = tmp_path / "derivatives.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        info = tarfile.TarInfo(
            "sailor_ebrains_pseud/derivatives/mni2009c-n-s/raw-mni-link.tsv"
        )
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    rows, source = read_canonical_tsv(
        tmp_path,
        "raw-mni-link.tsv",
        archive_candidates=("derivatives.tar.bz2",),
    )
    assert len(rows) == 1
    assert source is not None and source.endswith("raw-mni-link.tsv")
    assert summarize_raw_mni_links(rows)["links"] == [
        {"raw": "sub-01/ses-raw01", "mni": "sub-01/ses-01"}
    ]


def test_missing_tsv_excludes_required_t1wc_session(
    synthetic_project: tuple[object, Path],
) -> None:
    _, legacy = synthetic_project
    missing_path = legacy / "missing.tsv"
    missing_path.write_text(
        "participant_id\tsession_id\tmissing_sequence\n"
        "sub-01\tses-02\tt1wc\n",
        encoding="utf-8",
    )
    records, _ = inventory_nifti(legacy)
    result = guard_g9(
        summarize_missing(read_tsv(missing_path)),
        summarize_overview(read_tsv(legacy / "overview.tsv")),
        records,
    )
    assert result.status == "PASS"
    assert result.details["surviving_patient_sessions"] == [["sub-01", "ses-01"]]


def test_intensity_guard_reports_measured_range(
    synthetic_project: tuple[object, Path],
) -> None:
    _, legacy = synthetic_project
    records, _ = inventory_nifti(legacy)
    result = guard_g10(records)
    assert result.status == "PASS"
    assert result.details["observed_dtypes"] == ["float32"]
    assert result.details["normalization_decision"].startswith("DEFERRED")


def test_end_to_end_audit_writes_locked_manifests_without_touching_legacy(
    synthetic_project: tuple[object, Path],
) -> None:
    settings, legacy = synthetic_project
    before = snapshot_tree(legacy)
    result = run_stage1_audit(settings, raise_on_failure=False)
    after = snapshot_tree(legacy)

    assert result["failed_guards"] == []
    assert before == after
    manifest = json.loads(Path(result["dataset_manifest"]).read_text(encoding="utf-8"))
    qc = json.loads(Path(result["qc_report"]).read_text(encoding="utf-8"))
    gap = json.loads(Path(result["gap_report"]).read_text(encoding="utf-8"))
    assert manifest["PRIMARY_TARGET_MASK"] == "CL"
    assert manifest["PRIMARY_TARGET_COMPONENT"] == "enhancing_t1wc"
    assert manifest["implementation_id"] == "cursor_primary"
    assert manifest["target"]["n_primary_patient_sessions"] == 2
    assert manifest["dose"]["n_patients"] == 1
    assert manifest["delta_t"]["status"] == "EXACT_SOURCE_FOUND"
    assert manifest["delta_t"]["inter_session_gaps"][0]["gap_days"] == 14.0
    assert len(manifest["overview"]["treatment_records"]) == 2
    assert manifest["inventory"]["sequences_by_patient_session"]
    assert qc["compute_mode"] == "CPU-only"
    assert qc["profiled"] is False
    assert gap["download_performed"] is False


def test_degenerate_primary_mask_triggers_stop_protocol(
    synthetic_project: tuple[object, Path],
) -> None:
    settings, legacy = synthetic_project
    mask_path = next(legacy.rglob("*CL_t1wc_enhancing_mask.nii.gz"))
    image = nib.load(str(mask_path))
    nib.save(
        nib.Nifti1Image(np.zeros(image.shape, dtype=np.uint8), image.affine),
        str(mask_path),
    )

    records, _ = inventory_nifti(legacy)
    assert guard_g1(records).status == "FAIL"
    with pytest.raises(StopProtocolError) as exc:
        run_stage1_audit(settings, raise_on_failure=True)
    rendered = exc.value.render()
    assert "PROBLEM:" in rendered
    assert "IMPACT:" in rendered
    assert "RECOMMENDED FIX:" in rendered


def test_quarantined_nifti_is_not_in_inventory(
    synthetic_project: tuple[object, Path],
) -> None:
    _, legacy = synthetic_project
    quarantine = legacy / "tadiff_npy" / "sub-99_ses-99_t1wc.nii.gz"
    quarantine.parent.mkdir()
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)),
        str(quarantine),
    )
    records, _ = inventory_nifti(legacy)
    assert all("tadiff_npy" not in record.path for record in records)
