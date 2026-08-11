"""Streaming NIfTI inventory without extracting canonical archives."""

from __future__ import annotations

import gzip
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import nibabel as nib
import numpy as np

from sailor.constants import QUARANTINE_NAMES
from sailor.schemas import NiftiRecord

SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)", re.IGNORECASE)
SESSION_RE = re.compile(r"(ses-[A-Za-z0-9]+)", re.IGNORECASE)


def _tokens(name: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", name.lower())
        if token
    }


def classify_nifti(path: str) -> str:
    lowered = path.lower()
    basename = PurePosixPath(path.replace("\\", "/")).name.lower()
    tokens = _tokens(lowered)
    is_cl = "cl" in tokens
    if basename in {"contrastenhancedmask-cl.nii", "contrastenhancedmask-cl.nii.gz"}:
        return "CL:enhancing_t1wc"
    if basename in {"edemamask-cl.nii", "edemamask-cl.nii.gz"}:
        return "CL:t2wflair_hyperintensity"
    if is_cl and ("t1wc" in tokens or "t1ce" in tokens) and (
        {"enhancing", "enhancement", "tumor", "tumour", "mask"} & tokens
    ):
        return "CL:enhancing_t1wc"
    if is_cl and ({"t2wflair", "flair"} & tokens):
        return "CL:t2wflair_hyperintensity"
    if "onco" in tokens or "oncohabitats" in lowered:
        return "ONCO"
    if "dose" in tokens or "dosemap" in lowered or "rtdose" in lowered:
        return "DOSE"
    if is_cl:
        return "CL:unresolved_component"
    stem = basename.removesuffix(".nii.gz").removesuffix(".nii")
    if stem in {
        "brainextractionmask",
        "nawmask",
        "fastsurfer-segmentation",
        "tumormask",
        "mask",
    }:
        return "AUXILIARY_MASK"
    return "MRI"


def _subject_session(path: str) -> tuple[str | None, str | None]:
    subject_match = SUBJECT_RE.search(path)
    session_match = SESSION_RE.search(path)
    return (
        subject_match.group(1) if subject_match else None,
        session_match.group(1) if session_match else None,
    )


def _sequence(path: str, classification: str) -> str | None:
    if classification != "MRI":
        return None
    name = PurePosixPath(path.replace("\\", "/")).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name.split("_")[-1] if name else None


def _image_to_record(image: nib.spatialimages.SpatialImage, path: str, source: str) -> NiftiRecord:
    classification = classify_nifti(path)
    subject, session = _subject_session(path)
    array = np.asanyarray(image.dataobj)
    finite_mask = np.isfinite(array)
    all_finite = bool(finite_mask.all())
    minimum = float(np.min(array[finite_mask])) if finite_mask.any() else None
    maximum = float(np.max(array[finite_mask])) if finite_mask.any() else None
    count_mask = classification != "MRI"
    return NiftiRecord(
        path=path,
        source=source,
        subject=subject,
        session=session,
        sequence=_sequence(path, classification),
        classification=classification,
        shape=tuple(int(value) for value in image.shape),
        spacing=tuple(float(value) for value in image.header.get_zooms()[:3]),
        affine=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(image.affine)
        ),
        dtype=str(image.get_data_dtype()),
        minimum=minimum,
        maximum=maximum,
        finite=all_finite,
        nonzero_voxels=int(np.count_nonzero(array)) if count_mask else None,
        total_voxels=int(array.size) if count_mask else None,
    )


def inspect_nifti_file(path: Path, source: str = "extracted") -> NiftiRecord:
    image = nib.load(str(path))
    return _image_to_record(image, str(path), source)


def _image_from_archive_bytes(name: str, payload: bytes) -> nib.spatialimages.SpatialImage:
    raw = gzip.decompress(payload) if name.endswith(".nii.gz") else payload
    try:
        return nib.Nifti1Image.from_bytes(raw)
    except Exception:
        return nib.Nifti2Image.from_bytes(raw)


def inspect_nifti_archive(path: Path) -> list[NiftiRecord]:
    records: list[NiftiRecord] = []
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not (
                member.name.endswith(".nii") or member.name.endswith(".nii.gz")
            ):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            image = _image_from_archive_bytes(member.name, handle.read())
            records.append(
                _image_to_record(
                    image,
                    member.name,
                    f"archive:{path.name}",
                )
            )
    return records


def _is_quarantined(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return any(name.lower() in lowered_parts for name in QUARANTINE_NAMES)


def inventory_nifti(legacy_root: Path) -> tuple[list[NiftiRecord], dict[str, Any]]:
    extracted_paths: list[Path] = []
    if legacy_root.exists():
        for path in legacy_root.rglob("*.nii*"):
            lowered_parts = {part.lower() for part in path.parts}
            canonical_mni = (
                "derivatives" in lowered_parts
                and "mni2009c-n-s" in lowered_parts
            )
            if path.is_file() and canonical_mni and not _is_quarantined(path):
                extracted_paths.append(path)

    records: list[NiftiRecord] = []
    errors: list[dict[str, str]] = []
    if extracted_paths:
        for path in sorted(extracted_paths):
            try:
                records.append(inspect_nifti_file(path))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        source_mode = "extracted"
    else:
        archive = legacy_root / "derivatives.tar.bz2"
        source_mode = "archive"
        if archive.is_file():
            try:
                records = inspect_nifti_archive(archive)
            except Exception as exc:
                errors.append({"path": str(archive), "error": str(exc)})
        else:
            source_mode = "none"

    return records, {
        "source_mode": source_mode,
        "n_records": len(records),
        "errors": errors,
        "archive_policy": "streamed in place; no extraction",
    }


def summarize_inventory(records: list[NiftiRecord]) -> dict[str, Any]:
    patients = sorted({record.subject for record in records if record.subject})
    sessions = sorted(
        {
            (record.subject, record.session)
            for record in records
            if record.subject and record.session
        }
    )
    classifications: dict[str, int] = {}
    sequences: dict[str, int] = {}
    session_sequences: dict[str, set[str]] = {}
    for record in records:
        classifications[record.classification] = (
            classifications.get(record.classification, 0) + 1
        )
        if record.sequence:
            sequences[record.sequence] = sequences.get(record.sequence, 0) + 1
            if record.subject and record.session:
                key = f"{record.subject}/{record.session}"
                session_sequences.setdefault(key, set()).add(record.sequence)
    return {
        "n_patients": len(patients),
        "n_sessions": len(sessions),
        "patients": patients,
        "classifications": dict(sorted(classifications.items())),
        "sequences": dict(sorted(sequences.items())),
        "sequences_by_patient_session": {
            key: sorted(values) for key, values in sorted(session_sequences.items())
        },
        "records": [record.to_dict() for record in records],
    }
