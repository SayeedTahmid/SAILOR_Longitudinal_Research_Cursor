"""Robust TSV parsing and longitudinal metadata discovery."""

from __future__ import annotations

import csv
import io
import re
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SUBJECT_COLUMNS = ("subject", "subject_id", "participant_id", "patient", "sub")
SESSION_COLUMNS = ("session", "session_id", "ses")
TREATMENT_COLUMNS = ("treatment", "treatment_status", "therapy", "treatmentstatus")
DATE_COLUMNS = ("acq_time", "acquisition_time", "exam_date", "scan_date", "date")
SUBJECT_PATTERN = re.compile(r"(sub-[A-Za-z0-9]+)", re.IGNORECASE)
SESSION_PATTERN = re.compile(r"(ses-[A-Za-z0-9]+)", re.IGNORECASE)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [
            {str(key).strip(): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def read_canonical_tsv(
    legacy_root: Path,
    filename: str,
    *,
    archive_candidates: tuple[str, ...] = (),
) -> tuple[list[dict[str, str]], str | None]:
    direct = legacy_root / filename
    if direct.is_file():
        return read_tsv(direct), str(direct)

    for archive_name in archive_candidates:
        archive_path = legacy_root / archive_name
        if not archive_path.is_file():
            continue
        try:
            with tarfile.open(archive_path, mode="r|*") as archive:
                for member in archive:
                    if (
                        member.isfile()
                        and PurePosixPath(member.name).name == filename
                    ):
                        handle = archive.extractfile(member)
                        if handle is not None:
                            return (
                                _rows_from_bytes(handle.read()),
                                f"{archive_path}!{member.name}",
                            )
        except (OSError, tarfile.TarError):
            continue
    return [], None


def _normalized_key(record: dict[str, str], candidates: Iterable[str]) -> str | None:
    by_lower = {key.lower().strip(): key for key in record}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def value_from(record: dict[str, str], candidates: Iterable[str]) -> str | None:
    key = _normalized_key(record, candidates)
    value = record.get(key, "").strip() if key else ""
    return value or None


def normalize_subject(value: str | None) -> str | None:
    if not value:
        return None
    return value if value.startswith("sub-") else f"sub-{value}"


def normalize_session(value: str | None) -> str | None:
    if not value:
        return None
    return value if value.startswith("ses-") else f"ses-{value}"


def summarize_overview(rows: list[dict[str, str]]) -> dict[str, Any]:
    subjects: set[str] = set()
    sessions: set[tuple[str, str]] = set()
    treatments: Counter[str] = Counter()
    treatment_records: list[dict[str, Any]] = []
    unknown_rows: list[int] = []
    for index, row in enumerate(rows, start=2):
        subject = normalize_subject(value_from(row, SUBJECT_COLUMNS))
        session = normalize_session(value_from(row, SESSION_COLUMNS))
        if subject:
            subjects.add(subject)
        if subject and session:
            sessions.add((subject, session))
        treatment = value_from(row, TREATMENT_COLUMNS)
        if treatment:
            normalized = treatment.upper()
            if normalized == "NO":
                normalized = "no"
            if normalized not in {"CRT", "TMZ", "no", "UNKNOWN"}:
                normalized = f"UNRECOGNIZED:{treatment}"
            treatments[normalized] += 1
        else:
            normalized = "MISSING"
            treatments["MISSING"] += 1
            unknown_rows.append(index)
        treatment_records.append(
            {
                "subject": subject,
                "session": session,
                "status": normalized,
                "missing": normalized in {"MISSING", "UNKNOWN"},
            }
        )
    return {
        "n_rows": len(rows),
        "n_patients": len(subjects),
        "n_sessions": len(sessions),
        "patients": sorted(subjects),
        "patient_sessions": [list(item) for item in sorted(sessions)],
        "treatment_counts": dict(sorted(treatments.items())),
        "treatment_records": treatment_records,
        "treatment_missing_rows": unknown_rows,
        "unknown_semantics": "missingness indicator; never a treatment category",
    }


def summarize_missing(rows: list[dict[str, str]]) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    true_flags = {"y", "yes", "1", "true", "missing", "absent"}
    false_flags = {"n", "no", "0", "false", "present", "available", ""}
    for row in rows:
        subject = normalize_subject(value_from(row, SUBJECT_COLUMNS))
        session = normalize_session(value_from(row, SESSION_COLUMNS))
        missing_sequences: list[str] = []
        unrecognized_flags: dict[str, str] = {}
        missing_fields: dict[str, str] = {}
        for key, value in row.items():
            lowered_key = key.lower().strip()
            normalized_value = value.strip().lower()
            if lowered_key in SUBJECT_COLUMNS + SESSION_COLUMNS:
                continue
            missing_fields[key] = value
            if lowered_key in {"missing_sequence", "missing_sequences"}:
                missing_sequences.extend(
                    token
                    for token in re.split(r"[,;\s]+", normalized_value)
                    if token
                )
            elif normalized_value in true_flags:
                missing_sequences.append(lowered_key)
            elif normalized_value not in false_flags:
                unrecognized_flags[key] = value
        sessions.append(
            {
                "subject": subject,
                "session": session,
                "missing": sorted(set(missing_sequences)),
                "missing_sequences": sorted(set(missing_sequences)),
                "missing_fields": missing_fields,
                "unrecognized_flags": unrecognized_flags,
                "raw": row,
            }
        )
    return {
        "n_rows": len(rows),
        "sessions": sessions,
        "flag_semantics": {"missing": sorted(true_flags), "present": sorted(false_flags)},
    }


def summarize_raw_mni_links(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_rows": 0,
            "links": [],
            "explicitly_unmatched": [],
            "duplicates": [],
            "unresolved_rows": [],
        }
    keys = list(rows[0])
    subject_key = next(
        (k for k in keys if k.lower().strip() in SUBJECT_COLUMNS),
        None,
    )
    raw_key = next((k for k in keys if "raw" in k.lower()), None)
    mni_key = next((k for k in keys if "mni" in k.lower()), None)
    links: list[dict[str, str]] = []
    explicitly_unmatched: list[dict[str, str | int | None]] = []
    unresolved: list[int] = []

    def qualify(subject: str | None, value: str) -> str:
        normalized = value.replace("\\", "/")
        if "/" in normalized or normalized.lower().startswith("sub-"):
            return normalized
        session = normalize_session(normalized)
        return f"{subject}/{session}" if subject and session else normalized

    sentinels = {"", "no", "none", "na", "n/a", "missing", "unknown"}
    for index, row in enumerate(rows, start=2):
        subject = normalize_subject(row.get(subject_key, "").strip()) if subject_key else None
        raw_value = row.get(raw_key, "").strip() if raw_key else ""
        mni_value = row.get(mni_key, "").strip() if mni_key else ""
        raw_missing = raw_value.lower() in sentinels
        mni_missing = mni_value.lower() in sentinels
        if raw_key is None or mni_key is None:
            unresolved.append(index)
        elif raw_missing or mni_missing:
            explicitly_unmatched.append(
                {
                    "row": index,
                    "subject": subject,
                    "raw": None if raw_missing else qualify(subject, raw_value),
                    "mni": None if mni_missing else qualify(subject, mni_value),
                }
            )
        else:
            links.append(
                {
                    "raw": qualify(subject, raw_value),
                    "mni": qualify(subject, mni_value),
                }
            )
    raw_counts = Counter(link["raw"] for link in links)
    mni_counts = Counter(link["mni"] for link in links)
    duplicates = sorted(
        [f"raw:{key}" for key, count in raw_counts.items() if count > 1]
        + [f"mni:{key}" for key, count in mni_counts.items() if count > 1]
    )
    return {
        "n_rows": len(rows),
        "links": links,
        "explicitly_unmatched": explicitly_unmatched,
        "duplicates": duplicates,
        "unresolved_rows": unresolved,
        "columns": {"subject": subject_key, "raw": raw_key, "mni": mni_key},
    }


def _parse_datetime(value: str) -> datetime | None:
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        for pattern in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(cleaned[:10], pattern)
            except ValueError:
                continue
    return None


def _rows_from_bytes(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig", errors="replace")
    return [
        {str(key).strip(): (value or "").strip() for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(text), delimiter="\t")
    ]


def discover_exact_dates(legacy_root: Path) -> dict[str, Any]:
    candidates: list[tuple[str, list[dict[str, str]]]] = []
    if legacy_root.exists():
        for path in legacy_root.rglob("*scans.tsv"):
            lowered_parts = {part.lower() for part in path.parts}
            approved_tree = bool(
                {"rawdata_bids", "rawdata", "sourcedata"} & lowered_parts
            )
            if approved_tree:
                candidates.append((str(path), read_tsv(path)))

    archive_path = legacy_root / "rawdata_BIDS.tar.bz2"
    if not candidates and archive_path.is_file():
        try:
            with tarfile.open(archive_path, mode="r|*") as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith("scans.tsv"):
                        handle = archive.extractfile(member)
                        if handle:
                            candidates.append(
                                (
                                    f"{archive_path}!{member.name}",
                                    _rows_from_bytes(handle.read()),
                                )
                            )
        except (OSError, tarfile.TarError) as exc:
            return {
                "status": "UNVERIFIED",
                "source_attempts": [str(archive_path)],
                "error": str(exc),
                "dates": [],
            }

    dates: list[dict[str, str]] = []
    for source, rows in candidates:
        for row in rows:
            date_value = value_from(row, DATE_COLUMNS)
            parsed = _parse_datetime(date_value) if date_value else None
            if parsed:
                combined = f"{source}/{value_from(row, ('filename', 'file')) or ''}"
                subject_match = SUBJECT_PATTERN.search(combined)
                session_matches = SESSION_PATTERN.findall(combined)
                dates.append(
                    {
                        "source": source,
                        "filename": value_from(row, ("filename", "file")) or "",
                        "datetime": parsed.isoformat(),
                        "provenance": "raw/BIDS acquisition metadata",
                        "subject": (
                            subject_match.group(1) if subject_match else "UNRESOLVED"
                        ),
                        "session": (
                            session_matches[-1] if session_matches else "UNRESOLVED"
                        ),
                    }
                )

    additional_attempts = [
        str(path)
        for path in legacy_root.rglob("*")
        if path.is_file()
        and (
            path.name in {"history.txt", "raw_needed.tar"}
            or path.name.endswith("meta-data.txt")
        )
    ] if legacy_root.exists() else []
    session_dates: dict[tuple[str, str], datetime] = {}
    for item in dates:
        subject, session = item["subject"], item["session"]
        parsed = datetime.fromisoformat(item["datetime"])
        key = (subject, session)
        if "UNRESOLVED" not in key:
            session_dates[key] = min(parsed, session_dates.get(key, parsed))
    gaps: list[dict[str, Any]] = []
    by_subject: dict[str, list[tuple[str, datetime]]] = {}
    for (subject, session), parsed in session_dates.items():
        by_subject.setdefault(subject, []).append((session, parsed))
    for subject, sessions in sorted(by_subject.items()):
        ordered = sorted(sessions, key=lambda item: item[1])
        for (previous_session, previous_date), (session, date) in zip(
            ordered, ordered[1:]
        ):
            gaps.append(
                {
                    "subject": subject,
                    "from_session": previous_session,
                    "to_session": session,
                    "gap_days": (date - previous_date).total_seconds() / 86400.0,
                    "provenance": "raw/BIDS acquisition metadata",
                }
            )
    return {
        "status": "EXACT_SOURCE_FOUND" if dates else "APPROXIMATE_ONLY",
        "source_attempts": [source for source, _ in candidates] + additional_attempts,
        "dates": dates,
        "inter_session_gaps": gaps,
        "note": (
            "Intervals are exact only when derived from raw/source acquisition dates; "
            "MNI-derived intervals remain approximate."
        ),
    }
