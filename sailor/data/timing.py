"""Verified approximate MNI timing and treatment cache."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.errors import StopProtocolError
from sailor.reporting import write_json

ALLOWED_TREATMENTS = {"CRT", "TMZ", "no", "unknown"}


def _normalize_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _session_number(session: str) -> int:
    match = re.fullmatch(r"ses-(\d+)", session)
    if not match:
        raise StopProtocolError(
            f"Unrecognized MNI session label {session!r}.",
            "Approximate intervals cannot be assigned to a deterministic order.",
            "Inspect the canonical interval metadata and session labels.",
        )
    return int(match.group(1))


def build_verified_timing_cache(
    settings: Settings,
    *,
    canonical_intervals: dict[str, bytes],
    canonical_treatments: dict[str, bytes],
) -> dict[str, Any]:
    phase1_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json"
    )
    canonical_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_canonical_manifest.json"
    )
    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    canonical_manifest = json.loads(canonical_path.read_text(encoding="utf-8"))
    verified_archive = next(
        (
            item
            for item in canonical_manifest.get("verification", {}).get("files", [])
            if item.get("name") == "derivatives.tar.bz2"
            and item.get("status") == "VERIFIED"
        ),
        None,
    )
    if verified_archive is None:
        raise StopProtocolError(
            "Canonical derivatives checksum is not verified.",
            "Timing metadata cannot be attributed to immutable input.",
            "Restore the successful Phase 1 canonical manifest.",
        )

    raw_intervals: dict[str, bytes] = {}
    raw_treatments: dict[str, bytes] = {}
    with tarfile.open(settings.legacy_root / "raw_needed.tar", "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            normalized = _normalize_name(member.name)
            if normalized.endswith("intervals-days.txt"):
                handle = archive.extractfile(member)
                if handle:
                    raw_intervals[normalized] = handle.read()
            elif normalized.endswith("treatment.txt"):
                handle = archive.extractfile(member)
                if handle:
                    raw_treatments[normalized] = handle.read()

    canonical_intervals = {
        _normalize_name(name): payload for name, payload in canonical_intervals.items()
    }
    canonical_treatments = {
        _normalize_name(name): payload for name, payload in canonical_treatments.items()
    }
    if set(raw_intervals) != set(canonical_intervals) or any(
        raw_intervals[name] != canonical_intervals[name]
        for name in canonical_intervals
    ):
        raise StopProtocolError(
            "raw_needed interval files are not byte-identical to canonical metadata.",
            "The convenient metadata cache cannot be trusted.",
            "Read intervals directly from the verified derivatives archive.",
        )
    if set(raw_treatments) != set(canonical_treatments) or any(
        raw_treatments[name] != canonical_treatments[name]
        for name in canonical_treatments
    ):
        raise StopProtocolError(
            "raw_needed treatment files are not byte-identical to canonical metadata.",
            "Treatment conditioning could use repackaged or changed labels.",
            "Read treatment files directly from the verified derivatives archive.",
        )

    mni_sessions: dict[str, set[str]] = defaultdict(set)
    for record in phase1.get("inventory", {}).get("records", []):
        if (
            record.get("classification") == "MRI"
            and record.get("sequence") == settings.primary_input_sequence
            and record.get("subject")
            and record.get("session")
        ):
            mni_sessions[record["subject"]].add(record["session"])

    interval_by_subject: dict[str, list[int]] = {}
    interval_hashes: dict[str, str] = {}
    for name, payload in canonical_intervals.items():
        match = re.search(r"(sub-\d+)/intervals-days\.txt$", name)
        if not match:
            continue
        subject = match.group(1)
        values = [
            int(line.strip())
            for line in payload.decode("utf-8", errors="strict").splitlines()
            if line.strip()
        ]
        if any(value <= 0 for value in values):
            raise StopProtocolError(
                f"Non-positive interval found for {subject}.",
                "Approximate chronology is invalid.",
                "Inspect the canonical interval file.",
            )
        interval_by_subject[subject] = values
        interval_hashes[name] = _sha256_bytes(payload)

    treatment_lookup: dict[str, tuple[str, str, bytes]] = {}
    treatment_hashes: dict[str, str] = {}
    for name, payload in canonical_treatments.items():
        match = re.search(r"(sub-\d+)/(ses-\d+)/treatment\.txt$", name)
        if not match:
            continue
        subject, session = match.groups()
        label = payload.decode("utf-8", errors="strict").strip()
        if label not in ALLOWED_TREATMENTS:
            raise StopProtocolError(
                f"Unrecognized treatment label {label!r} in {name}.",
                "An unregistered treatment class would enter Phase 2.",
                "Correct or explicitly revise the treatment protocol.",
            )
        treatment_lookup[f"{subject}/{session}"] = (label, name, payload)
        treatment_hashes[name] = _sha256_bytes(payload)

    records: list[dict[str, Any]] = []
    for subject, session_set in sorted(mni_sessions.items()):
        sessions = sorted(session_set, key=_session_number)
        intervals = interval_by_subject.get(subject)
        if intervals is None or len(intervals) != len(sessions) - 1:
            raise StopProtocolError(
                f"Interval count does not match MNI sessions for {subject}.",
                "Approximate days cannot be assigned one-to-one.",
                "Inspect the canonical interval file and MNI inventory.",
            )
        cumulative = [0]
        for value in intervals:
            cumulative.append(cumulative[-1] + value)
        for index, session in enumerate(sessions):
            key = f"{subject}/{session}"
            if key not in treatment_lookup:
                raise StopProtocolError(
                    f"Canonical treatment file is missing for {key}.",
                    "Treatment missingness would be fabricated.",
                    "Restore the canonical treatment metadata.",
                )
            label, treatment_source, _ = treatment_lookup[key]
            records.append(
                {
                    "subject": subject,
                    "mni_session": session,
                    "approximate_day": cumulative[index],
                    "interval_from_previous_days": (
                        None if index == 0 else intervals[index - 1]
                    ),
                    "timing_provenance": "approximate_mni_intervals",
                    "treatment_status": None if label == "unknown" else label,
                    "treatment_missing": label == "unknown",
                    "interval_source": next(
                        name
                        for name in canonical_intervals
                        if name.endswith(f"{subject}/intervals-days.txt")
                    ),
                    "treatment_source": treatment_source,
                }
            )

    expected_sessions = sum(len(value) for value in mni_sessions.values())
    if len(records) != expected_sessions:
        raise StopProtocolError(
            "Timing cache does not cover every MNI session.",
            "Window eligibility could change silently.",
            "Resolve missing timing or treatment records.",
        )
    payload = {
        "data_version": settings.data_version,
        "preprocessing_version": settings.preprocessing_version,
        "implementation_id": settings.implementation_id,
        "timing_provenance": "approximate_mni_intervals",
        "exact_dates_available": False,
        "source_derivatives_sha512": verified_archive["actual_sha512"],
        "phase1_dataset_manifest_sha256": _sha256_file(phase1_path),
        "phase1_canonical_manifest_sha256": _sha256_file(canonical_path),
        "raw_needed_verification": {
            "interval_files": len(raw_intervals),
            "treatment_files": len(raw_treatments),
            "all_byte_identical": True,
        },
        "interval_file_hashes": dict(sorted(interval_hashes.items())),
        "treatment_file_hashes": dict(sorted(treatment_hashes.items())),
        "n_patients": len(mni_sessions),
        "n_sessions": len(records),
        "records": records,
    }
    payload["content_hash"] = _content_hash(records)
    return payload


def persist_verified_timing_cache(
    settings: Settings,
    *,
    canonical_intervals: dict[str, bytes],
    canonical_treatments: dict[str, bytes],
) -> dict[str, Any]:
    payload = build_verified_timing_cache(
        settings,
        canonical_intervals=canonical_intervals,
        canonical_treatments=canonical_treatments,
    )
    path = (
        settings.dataset_root
        / "05_TREATMENT_DATA"
        / settings.preprocessing_version
        / "v2_canonical_timing_cache.json"
    )
    write_json(path, payload, settings)
    return {"path": str(path), "cache": payload}


def validate_timing_cache(
    settings: Settings,
    payload: dict[str, Any],
) -> None:
    phase1_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_dataset_manifest.json"
    )
    canonical_path = (
        settings.dataset_root / "01_DATA_FOUNDATION" / "v2_canonical_manifest.json"
    )
    invalid = (
        payload.get("data_version") != settings.data_version
        or payload.get("preprocessing_version") != settings.preprocessing_version
        or payload.get("implementation_id") != settings.implementation_id
        or payload.get("timing_provenance") != "approximate_mni_intervals"
        or payload.get("content_hash") != _content_hash(payload.get("records", []))
        or payload.get("phase1_dataset_manifest_sha256") != _sha256_file(phase1_path)
        or payload.get("phase1_canonical_manifest_sha256")
        != _sha256_file(canonical_path)
        or payload.get("raw_needed_verification", {}).get("all_byte_identical")
        is not True
    )
    if invalid:
        raise StopProtocolError(
            "Approximate timing cache failed provenance validation.",
            "Windows could use stale intervals or treatment labels.",
            "Regenerate the timing cache from verified canonical metadata.",
        )
