"""Provenance-aware longitudinal windows and treatment linkage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any

from sailor.errors import StopProtocolError
from sailor.schemas import WindowRecord


def _key(subject: str, session: str) -> str:
    return f"{subject}/{session}"


def build_treatment_manifest(
    phase1_manifest: dict[str, Any],
    preprocessed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    treatment_lookup = {
        _key(item["subject"], item["session"]): item
        for item in phase1_manifest.get("overview", {}).get("treatment_records", [])
        if item.get("subject") and item.get("session")
    }
    dose_lookup = {
        item["subject"]: item["path"]
        for item in phase1_manifest.get("dose", {}).get("files", [])
        if item.get("subject") and item.get("path")
    }
    records: list[dict[str, Any]] = []
    for item in preprocessed_records:
        raw_subject = item.get("raw_subject", item["subject"])
        raw_key = _key(raw_subject, item["raw_session"])
        treatment = treatment_lookup.get(raw_key, {})
        status = treatment.get("status")
        missing = bool(treatment.get("missing", True)) or status in {
            None,
            "MISSING",
            "UNKNOWN",
        }
        if not missing and status not in {"CRT", "TMZ", "no"}:
            raise StopProtocolError(
                f"Unrecognized treatment status {status!r} for {raw_key}.",
                "A new treatment class would enter conditioning without pre-registration.",
                "Correct the metadata parser or explicitly revise the treatment protocol.",
            )
        records.append(
            {
                "subject": item["subject"],
                "raw_subject": raw_subject,
                "raw_session": item["raw_session"],
                "mni_session": item["mni_session"],
                "status": None if missing else status,
                "missing": missing,
                "dose_reference": dose_lookup.get(item["subject"]),
                "dose_missing": item["subject"] not in dose_lookup,
            }
        )
    return {
        "unknown_policy": "missingness indicator; never a treatment class",
        "n_records": len(records),
        "n_missing": sum(item["missing"] for item in records),
        "n_dose_missing": sum(item["dose_missing"] for item in records),
        "records": records,
    }


def build_longitudinal_windows(
    phase1_manifest: dict[str, Any],
    preprocessed_records: list[dict[str, Any]],
    *,
    min_history_scans: int,
) -> dict[str, Any]:
    delta = phase1_manifest.get("delta_t", {})
    provenance = (
        "exact"
        if delta.get("status") == "EXACT_SOURCE_FOUND"
        else "approximate"
        if delta.get("status") == "APPROXIMATE_ONLY"
        else "unavailable"
    )
    date_lookup: dict[str, datetime] = {}
    for item in delta.get("dates", []):
        subject, session = item.get("subject"), item.get("session")
        if not subject or not session or "UNRESOLVED" in {subject, session}:
            continue
        parsed = datetime.fromisoformat(item["datetime"])
        key = _key(subject, session)
        date_lookup[key] = min(parsed, date_lookup.get(key, parsed))

    treatment = build_treatment_manifest(phase1_manifest, preprocessed_records)
    treatment_lookup = {
        _key(item.get("raw_subject", item["subject"]), item["raw_session"]): item
        for item in treatment["records"]
    }
    by_subject: dict[str, list[dict[str, Any]]] = {}
    missing_dates: list[str] = []
    for item in preprocessed_records:
        raw_subject = item.get("raw_subject", item["subject"])
        raw_key = _key(raw_subject, item["raw_session"])
        date = date_lookup.get(raw_key)
        if date is None:
            missing_dates.append(raw_key)
            continue
        enriched = dict(item)
        enriched["raw_subject"] = raw_subject
        enriched["_date"] = date
        by_subject.setdefault(item["subject"], []).append(enriched)
    if missing_dates:
        raise StopProtocolError(
            f"Exact/qualified acquisition dates are missing for {len(missing_dates)} selected sessions.",
            "Longitudinal order or Δt would require guessed session numbering.",
            f"Resolve timing provenance for: {missing_dates[:10]}",
        )

    windows: list[WindowRecord] = []
    excluded_patients: list[str] = []
    for subject, sessions in sorted(by_subject.items()):
        ordered = sorted(sessions, key=lambda item: item["_date"])
        if len(ordered) <= min_history_scans:
            excluded_patients.append(subject)
            continue
        for target_index in range(min_history_scans, len(ordered)):
            history = ordered[:target_index]
            target = ordered[target_index]
            deltas = [
                (target["_date"] - item["_date"]).total_seconds() / 86400.0
                for item in history
            ]
            if any(value <= 0 for value in deltas):
                raise StopProtocolError(
                    f"Non-positive longitudinal interval detected for {subject}.",
                    "Session chronology is invalid for prediction windows.",
                    "Inspect duplicate acquisition dates and raw-to-MNI mappings.",
                )
            treatment_item = treatment_lookup[
                _key(target["raw_subject"], target["raw_session"])
            ]
            identity = {
                "subject": subject,
                "history": [item["mni_session"] for item in history],
                "target": target["mni_session"],
            }
            window_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
            windows.append(
                WindowRecord(
                    window_id=window_id,
                    subject=subject,
                    history_raw_sessions=[
                        _key(item["raw_subject"], item["raw_session"])
                        for item in history
                    ],
                    history_mni_sessions=[item["mni_session"] for item in history],
                    target_raw_session=_key(
                        target["raw_subject"], target["raw_session"]
                    ),
                    target_mni_session=target["mni_session"],
                    history_delta_days=deltas,
                    target_delta_days=deltas[-1],
                    timing_provenance=provenance,
                    treatment_status=treatment_item["status"],
                    treatment_missing=treatment_item["missing"],
                )
            )

    counts = Counter(window.subject for window in windows)
    for window in windows:
        window.patient_weight = 1.0 / counts[window.subject]
    if not windows:
        raise StopProtocolError(
            "No longitudinal windows satisfy the two-history requirement.",
            "Phase 3 would have no valid prediction examples.",
            "Review measured per-patient session counts without weakening target guards.",
        )
    payload = {
        "minimum_history_scans": min_history_scans,
        "history_policy": "all earlier eligible scans; variable length",
        "timing_provenance": provenance,
        "n_windows": len(windows),
        "n_patients": len(counts),
        "windows_per_patient": dict(sorted(counts.items())),
        "excluded_patients_insufficient_history": excluded_patients,
        "windows": [window.to_dict() for window in windows],
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(
            payload["windows"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {"windows": payload, "treatment": treatment}
