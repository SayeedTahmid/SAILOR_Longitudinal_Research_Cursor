"""Phase-2 dry run and selective preprocessing execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from sailor.config import Settings
from sailor.contracts import (
    assert_aligned_geometry,
    assert_mask_contract,
    assert_volume_contract,
)
from sailor.data.provenance import sha512_file
from sailor.errors import StopProtocolError
from sailor.paths import (
    assert_writable_target,
    snapshot_tree,
    verify_snapshot_unchanged,
)
from sailor.preprocessing.normalize import robust_scale_volume
from sailor.preprocessing.policy import assert_plan_ready, build_preprocessing_plan
from sailor.reporting import write_json
from sailor.schemas import PreprocessingRecord


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_verified_binary_mask(
    mask: np.ndarray,
    *,
    expected_positive_value: float,
    name: str,
) -> np.ndarray:
    assert_volume_contract(mask, name=name)
    positive = np.asarray(mask)[np.asarray(mask) > 0]
    values = np.unique(positive)
    if values.size != 1 or not np.isclose(
        float(values[0]),
        expected_positive_value,
        rtol=1e-6,
        atol=1e-9,
    ):
        raise StopProtocolError(
            f"{name} does not match its verified binary foreground scale.",
            "Binarization could change an interpolated or corrupted tumour boundary.",
            "Stop and inspect the exact positive-value distribution.",
        )
    normalized = (np.asarray(mask) > 0).astype(np.uint8)
    assert_mask_contract(normalized, name=name, expected_shape=mask.shape)
    return normalized


def normalize_brain_support_mask(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    name: str,
) -> np.ndarray:
    assert_volume_contract(mask, name=name)
    minimum = float(np.min(mask))
    maximum = float(np.max(mask))
    if minimum < 0.0 or maximum > 1.0:
        raise StopProtocolError(
            f"{name} has values outside the verified 0–1 range.",
            "The normalization support could include invalid anatomy.",
            "Inspect the brain-mask provenance before changing its threshold.",
        )
    normalized = (np.asarray(mask) >= threshold).astype(np.uint8)
    fraction = float(np.count_nonzero(normalized) / normalized.size)
    if not 0.01 <= fraction <= 0.90:
        raise StopProtocolError(
            f"{name} foreground fraction {fraction:.6f} is implausible.",
            "MRI normalization would use an empty or nearly whole-volume support.",
            "Inspect the brain mask and registration; do not continue automatically.",
        )
    assert_mask_contract(normalized, name=name, expected_shape=mask.shape)
    return normalized


def load_phase1_inputs(settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    foundation = settings.dataset_root / "01_DATA_FOUNDATION"
    qc_dir = settings.dataset_root / "06_QC_REPORTS"
    manifest_path = foundation / "v2_dataset_manifest.json"
    qc_path = qc_dir / "v2_stage1_qc_report.json"
    if not manifest_path.is_file() or not qc_path.is_file():
        raise StopProtocolError(
            "Required Phase 1 manifests are missing.",
            "Phase 2 cannot identify the locked cohort or upstream guards.",
            "Complete Phase 1 sections 01–09 before Phase 2.",
        )
    return _read_json(manifest_path), _read_json(qc_path)


def run_phase2_dry_run(settings: Settings) -> dict[str, Any]:
    manifest, qc_report = load_phase1_inputs(settings)
    plan = build_preprocessing_plan(manifest, qc_report, settings)
    phase1_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json"
    )
    phase1_qc_path = (
        settings.dataset_root / "06_QC_REPORTS" / "v2_stage1_qc_report.json"
    )
    canonical_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_canonical_manifest.json"
    )
    if not canonical_path.is_file():
        raise StopProtocolError(
            "Phase 1 canonical manifest is missing.",
            "The derivatives archive cannot be bound to its verified checksum.",
            "Restore the successful Phase 1 canonical manifest.",
        )
    canonical = _read_json(canonical_path)
    derivatives_record = next(
        (
            item
            for item in canonical.get("verification", {}).get("files", [])
            if item.get("name") == "derivatives.tar.bz2"
        ),
        None,
    )
    if not derivatives_record or derivatives_record.get("status") != "VERIFIED":
        raise StopProtocolError(
            "derivatives.tar.bz2 is not checksum-verified in Phase 1.",
            "Selective extraction lacks canonical provenance.",
            "Re-run the Phase 1 provenance guard.",
        )
    archive_path = settings.legacy_root / "derivatives.tar.bz2"
    archive_stat = archive_path.stat()
    plan["phase1_dataset_manifest_sha256"] = _sha256(phase1_path)
    plan["phase1_qc_report_sha256"] = _sha256(phase1_qc_path)
    plan["phase1_canonical_manifest_sha256"] = _sha256(canonical_path)
    plan["source_derivatives_sha512"] = derivatives_record["actual_sha512"]
    plan["source_derivatives_size"] = archive_stat.st_size
    plan["source_derivatives_mtime_ns"] = archive_stat.st_mtime_ns
    usage = shutil.disk_usage(settings.dataset_root)
    required = int(
        (plan["planned_output_bytes"] + plan["planned_staging_bytes"]) * 1.10
    )
    plan["free_bytes"] = usage.free
    plan["required_bytes_with_10pct_margin"] = required
    plan["space_sufficient"] = usage.free >= required
    plan["extraction_approved"] = False
    path = (
        settings.dataset_root
        / "06_QC_REPORTS"
        / "v2_phase2_preprocessing_dry_run.json"
    )
    write_json(path, plan, settings)
    return {"plan": plan, "report": str(path)}


def validate_preprocessing_cache(
    settings: Settings,
    manifest: dict[str, Any],
) -> None:
    expected_headers = {
        "data_version": settings.data_version,
        "preprocessing_version": settings.preprocessing_version,
        "implementation_id": settings.implementation_id,
        "PRIMARY_TARGET_MASK": settings.primary_target_mask,
        "PRIMARY_TARGET_COMPONENT": settings.primary_target_component,
        "selected_sequence": settings.primary_input_sequence,
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected_headers.items()
        if manifest.get(key) != value
    }
    phase1_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json"
    )
    if not phase1_path.is_file() or manifest.get(
        "phase1_dataset_manifest_sha256"
    ) != _sha256(phase1_path):
        mismatches["phase1_dataset_manifest_sha256"] = "missing or stale"
    qc_path = settings.dataset_root / "06_QC_REPORTS" / "v2_stage1_qc_report.json"
    canonical_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_canonical_manifest.json"
    )
    if not qc_path.is_file() or manifest.get("phase1_qc_report_sha256") != _sha256(
        qc_path
    ):
        mismatches["phase1_qc_report_sha256"] = "missing or stale"
    if not canonical_path.is_file() or manifest.get(
        "phase1_canonical_manifest_sha256"
    ) != _sha256(canonical_path):
        mismatches["phase1_canonical_manifest_sha256"] = "missing or stale"
    elif canonical_path.is_file():
        canonical = _read_json(canonical_path)
        verified = next(
            (
                item
                for item in canonical.get("verification", {}).get("files", [])
                if item.get("name") == "derivatives.tar.bz2"
            ),
            {},
        )
        if (
            verified.get("status") != "VERIFIED"
            or verified.get("actual_sha512")
            != manifest.get("source_derivatives_sha512")
        ):
            mismatches["source_derivatives_sha512"] = "canonical checksum mismatch"
    archive = settings.legacy_root / "derivatives.tar.bz2"
    if not archive.is_file():
        mismatches["source_derivatives_archive"] = "missing"
    else:
        stat = archive.stat()
        if (
            stat.st_size != manifest.get("source_derivatives_size")
            or stat.st_mtime_ns != manifest.get("source_derivatives_mtime_ns")
        ):
            mismatches["source_derivatives_archive"] = "size or mtime changed"
    records = manifest.get("records", [])
    if manifest.get("n_records") != len(records) or manifest.get(
        "records_hash"
    ) != _content_hash(records):
        mismatches["records"] = "count or content hash mismatch"
    invalid_outputs: list[str] = []
    for record in records:
        for field, checksum_field in (
            ("mri_output", "mri_sha256"),
            ("mask_output", "mask_sha256"),
        ):
            path = Path(record.get(field, ""))
            expected = record.get("checksums", {}).get(checksum_field)
            if not path.is_file() or not expected or _sha256(path) != expected:
                invalid_outputs.append(str(path))
    if invalid_outputs:
        mismatches["outputs"] = invalid_outputs[:20]
    if mismatches:
        raise StopProtocolError(
            f"Preprocessing cache validation failed: {mismatches}",
            "Stale or tampered arrays could be reused as valid Phase 2 inputs.",
            "Rebuild the cache from the verified archive with explicit approval.",
        )


def _write_pgm(path: Path, arrays: list[np.ndarray]) -> None:
    if not arrays:
        return
    slices: list[np.ndarray] = []
    for array in arrays[:12]:
        image = array[:, :, array.shape[2] // 2]
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        scale = high - low if high > low else 1.0
        slices.append((np.clip((image - low) / scale, 0, 1) * 255).astype(np.uint8))
    height = max(item.shape[0] for item in slices)
    width = sum(item.shape[1] for item in slices)
    montage = np.zeros((height, width), dtype=np.uint8)
    offset = 0
    for item in slices:
        montage[: item.shape[0], offset : offset + item.shape[1]] = item
        offset += item.shape[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(montage.tobytes())


def _promote_directories(
    builds_and_finals: list[tuple[Path, Path]],
) -> list[tuple[Path, Path]]:
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for _, final in builds_and_finals:
            backup = final.with_name(f".{final.name}_backup")
            if backup.exists():
                shutil.rmtree(backup)
            if final.exists():
                os.replace(final, backup)
                backups.append((backup, final))
        for build, final in builds_and_finals:
            os.replace(build, final)
            promoted.append(final)
    except Exception:
        for final in promoted:
            if final.exists():
                shutil.rmtree(final)
        for backup, final in backups:
            if backup.exists():
                os.replace(backup, final)
        raise
    return backups


def _rollback_promotions(
    finals: list[Path],
    backups: list[tuple[Path, Path]],
) -> None:
    for final in finals:
        if final.exists():
            shutil.rmtree(final)
    for backup, final in backups:
        if backup.exists():
            os.replace(backup, final)


def _discard_backups(backups: list[tuple[Path, Path]]) -> None:
    for backup, _ in backups:
        if backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError:
                # A validated new cache is already active. Retaining an old backup is
                # safer than turning cleanup friction into rollback data loss.
                continue


def execute_preprocessing(
    settings: Settings,
    *,
    extraction_approved: bool,
) -> dict[str, Any]:
    if not extraction_approved:
        raise StopProtocolError(
            "Selective extraction has not been explicitly approved.",
            "Phase 2 would write medical arrays before the disk/file plan was reviewed.",
            "Review the dry-run report and rerun with extraction_approved=True.",
        )
    before = snapshot_tree(settings.legacy_root)
    dry_run = run_phase2_dry_run(settings)
    plan = dry_run["plan"]
    assert_plan_ready(plan)
    if not plan["space_sufficient"]:
        raise StopProtocolError(
            "Drive free space is below the measured Phase 2 requirement.",
            "Selective extraction could fail mid-write or corrupt an incomplete cache.",
            "Free space or revise the approved storage policy before extraction.",
        )
    actual_archive_sha512 = sha512_file(
        settings.legacy_root / "derivatives.tar.bz2"
    )
    if actual_archive_sha512 != plan["source_derivatives_sha512"]:
        raise StopProtocolError(
            "The current derivatives archive does not match its Phase 1 SHA-512 checksum.",
            "Phase 2 would extract from changed or corrupted canonical input.",
            "Restore the verified archive and rerun Phase 1 provenance checks.",
        )

    version = settings.preprocessing_version
    mri_parent = settings.dataset_root / "02_PREPROCESSED_MRI"
    mask_parent = settings.dataset_root / "03_TUMOR_MASKS"
    mri_root = mri_parent / version
    mask_root = mask_parent / version
    build_mri = mri_parent / f".{version}_build"
    build_mask = mask_parent / f".{version}_build"
    staging = build_mri / "_source"
    for path in (build_mri, build_mask):
        assert_writable_target(path, settings)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    staging.mkdir()

    montage = settings.dataset_root / "06_QC_REPORTS" / "v2_phase2_t1c_montage.pgm"
    montage_build = montage.with_name(f".{montage.name}.build")
    montage_backup = montage.with_name(f".{montage.name}.backup")
    promotion_backups: list[tuple[Path, Path]] = []
    promoted_finals: list[Path] = []
    try:
        wanted: dict[str, tuple[str, dict[str, Any]]] = {}
        for item in plan["selected"]:
            wanted[item["mri_source"]] = ("mri", item)
            wanted[item["mask_source"]] = ("mask", item)
            wanted[item["brain_mask_source"]] = ("brain", item)

        staged: dict[tuple[str, str, str], Path] = {}
        found_sources: set[str] = set()
        archive_path = settings.legacy_root / "derivatives.tar.bz2"
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                selected = wanted.get(member.name)
                if selected is None or not member.isfile():
                    continue
                component, item = selected
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                key = (item["subject"], item["mni_session"], component)
                suffix = ".nii.gz" if member.name.endswith(".nii.gz") else ".nii"
                destination = staging / (
                    f"{item['subject']}_{item['mni_session']}_{component}{suffix}"
                )
                destination.write_bytes(handle.read())
                staged[key] = destination
                found_sources.add(member.name)

        if len(staged) != len(wanted):
            missing = sorted(set(wanted) - found_sources)
            raise StopProtocolError(
                f"Selective extraction found {len(staged)} of {len(wanted)} required files.",
                "The preprocessing cache is incomplete.",
                f"Inspect archive member matching; unresolved examples: {missing[:10]}",
            )

        output_records: list[PreprocessingRecord] = []
        qc_arrays: list[np.ndarray] = []
        for item in plan["selected"]:
            key = (item["subject"], item["mni_session"])
            mri_image = nib.load(str(staged[(*key, "mri")]))
            mask_image = nib.load(str(staged[(*key, "mask")]))
            brain_image = nib.load(str(staged[(*key, "brain")]))
            mri = np.asanyarray(mri_image.dataobj)
            mask = np.asanyarray(mask_image.dataobj)
            brain = np.asanyarray(brain_image.dataobj)
            assert_volume_contract(mri, name=f"{key}/T1c")
            binary_mask = normalize_verified_binary_mask(
                mask,
                expected_positive_value=item["mask_positive_scale"],
                name=f"{key}/CL",
            )
            binary_brain = normalize_brain_support_mask(
                brain,
                threshold=item["brain_mask_threshold"],
                name=f"{key}/brain",
            )
            assert_aligned_geometry(
                reference_shape=mri.shape,
                reference_affine=mri_image.affine,
                candidate_shape=mask.shape,
                candidate_affine=mask_image.affine,
                candidate_name=f"{key}/CL",
            )
            assert_aligned_geometry(
                reference_shape=mri.shape,
                reference_affine=mri_image.affine,
                candidate_shape=brain.shape,
                candidate_affine=brain_image.affine,
                candidate_name=f"{key}/brain",
            )
            normalized, scaling = robust_scale_volume(mri, binary_brain)
            scaling["brain_mask_threshold"] = item["brain_mask_threshold"]
            build_subject = build_mri / item["subject"] / item["mni_session"]
            build_mask_dir = build_mask / item["subject"] / item["mni_session"]
            build_subject.mkdir(parents=True, exist_ok=True)
            build_mask_dir.mkdir(parents=True, exist_ok=True)
            built_mri = build_subject / "T1c-icor.npy"
            built_mask = build_mask_dir / "CL-enhancing-t1wc.npy"
            final_mri = mri_root / item["subject"] / item["mni_session"] / built_mri.name
            final_mask = mask_root / item["subject"] / item["mni_session"] / built_mask.name
            np.save(built_mri, normalized, allow_pickle=False)
            np.save(built_mask, binary_mask, allow_pickle=False)
            output_records.append(
                PreprocessingRecord(
                    subject=item["subject"],
                    raw_subject=item["raw_subject"],
                    raw_session=item["raw_session"],
                    mni_session=item["mni_session"],
                    mri_source=item["mri_source"],
                    mask_source=item["mask_source"],
                    brain_mask_source=item["brain_mask_source"],
                    mri_output=str(final_mri),
                    mask_output=str(final_mask),
                    shape=tuple(int(value) for value in mri.shape),
                    spacing=tuple(float(value) for value in mri_image.header.get_zooms()[:3]),
                    affine_hash=item["affine_hash"],
                    mask_original_positive_value=item["mask_positive_scale"],
                    scaling=scaling,
                    checksums={
                        "mri_sha256": _sha256(built_mri),
                        "mask_sha256": _sha256(built_mask),
                    },
                )
            )
            if len(qc_arrays) < 12:
                qc_arrays.append(normalized)

        shutil.rmtree(staging)
        _write_pgm(montage_build, qc_arrays)
        records_payload = [record.to_dict() for record in output_records]
        manifest = {
            "data_version": settings.data_version,
            "preprocessing_version": version,
            "implementation_id": settings.implementation_id,
            "PRIMARY_TARGET_MASK": settings.primary_target_mask,
            "PRIMARY_TARGET_COMPONENT": settings.primary_target_component,
            "selected_sequence": settings.primary_input_sequence,
            "phase1_dataset_manifest_sha256": plan["phase1_dataset_manifest_sha256"],
            "phase1_qc_report_sha256": plan["phase1_qc_report_sha256"],
            "phase1_canonical_manifest_sha256": plan[
                "phase1_canonical_manifest_sha256"
            ],
            "source_derivatives_sha512": plan["source_derivatives_sha512"],
            "source_derivatives_size": plan["source_derivatives_size"],
            "source_derivatives_mtime_ns": plan["source_derivatives_mtime_ns"],
            "n_records": len(output_records),
            "records": records_payload,
            "records_hash": _content_hash(records_payload),
            "optional_modalities": "BLOCKED",
            "normalization_scope": "single_volume_brain_mask",
            "visual_qc_montage": str(montage),
        }
        write_json(build_mri / "v2_preprocessing_manifest.json", manifest, settings)
        verify_snapshot_unchanged(before, snapshot_tree(settings.legacy_root))
        if montage_backup.exists():
            montage_backup.unlink()
        if montage.exists():
            os.replace(montage, montage_backup)
        promotion_backups = _promote_directories(
            [(build_mri, mri_root), (build_mask, mask_root)]
        )
        promoted_finals = [mri_root, mask_root]
        if montage_build.exists():
            os.replace(montage_build, montage)
        validate_preprocessing_cache(settings, manifest)
        _discard_backups(promotion_backups)
        if montage_backup.exists():
            try:
                montage_backup.unlink()
            except OSError:
                pass
        return {
            "manifest": manifest,
            "manifest_path": str(mri_root / "v2_preprocessing_manifest.json"),
        }
    except Exception:
        if promoted_finals:
            _rollback_promotions(promoted_finals, promotion_backups)
        if promoted_finals and montage.exists():
            montage.unlink()
        if montage_backup.exists():
            os.replace(montage_backup, montage)
        for path in (build_mri, build_mask):
            if path.exists():
                shutil.rmtree(path)
        if montage_build.exists():
            montage_build.unlink()
        raise
