"""Canonical-file classification, checksums, and storage accounting."""

from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

from sailor.constants import (
    AMBIGUOUS_NAMES,
    CANONICAL_FILES,
    KNOWN_MISSING_CANONICAL,
    QUARANTINE_NAMES,
)

SHA512_PATTERN = re.compile(r"^([0-9a-fA-F]{128})\s+\*?(.+?)\s*$")


def parse_sha512(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not path.exists():
        return checksums
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SHA512_PATTERN.match(raw_line.strip())
        if match:
            digest, name = match.groups()
            checksums[Path(name).name] = digest.lower()
    return checksums


def sha512_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def classify_legacy_root(legacy_root: Path) -> dict[str, Any]:
    present = {path.name: path for path in legacy_root.iterdir()} if legacy_root.exists() else {}
    return {
        "canonical": {
            name: str(present[name]) if name in present else None
            for name in CANONICAL_FILES
        },
        "quarantine": {
            name: str(present[name]) if name in present else None
            for name in QUARANTINE_NAMES
        },
        "ambiguous": {
            name: str(present[name]) if name in present else None
            for name in AMBIGUOUS_NAMES
        },
        "known_missing_candidates": {
            name: str(present[name]) if name in present else None
            for name in KNOWN_MISSING_CANONICAL
        },
    }


def verify_canonical_files(legacy_root: Path) -> dict[str, Any]:
    checksum_file = legacy_root / "SHA512.txt"
    expected = parse_sha512(checksum_file)
    results: list[dict[str, Any]] = []
    for name in CANONICAL_FILES:
        path = legacy_root / name
        record: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha512": expected.get(name),
            "actual_sha512": None,
            "status": "MISSING",
        }
        if path.is_file():
            if name == "SHA512.txt":
                record["status"] = "PRESENT_CHECKSUM_INDEX"
            elif expected.get(name):
                actual = sha512_file(path)
                record["actual_sha512"] = actual
                record["status"] = "VERIFIED" if actual == expected[name] else "MISMATCH"
            else:
                record["status"] = "PRESENT_NO_PUBLISHED_CHECKSUM"
        results.append(record)
    return {
        "checksum_index": str(checksum_file),
        "parsed_checksum_count": len(expected),
        "files": results,
    }


def archive_uncompressed_size(path: Path) -> tuple[int | None, str]:
    if not path.is_file() or not tarfile.is_tarfile(path):
        return None, "UNVERIFIED — archive absent or unreadable"
    try:
        total = 0
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if member.isfile():
                    total += member.size
        return total, "measured from archive member headers"
    except (OSError, tarfile.TarError) as exc:
        return None, f"UNVERIFIED — {exc}"


def storage_report(dataset_root: Path, legacy_root: Path) -> dict[str, Any]:
    usage_anchor = dataset_root.parent if dataset_root.parent.exists() else legacy_root
    try:
        usage = shutil.disk_usage(usage_anchor)
        free_bytes: int | None = usage.free
    except OSError:
        free_bytes = None

    archives: list[dict[str, Any]] = []
    projected = 0
    complete_projection = True
    for name in ("derivatives.tar.bz2", "rawdata_BIDS.tar.bz2", "code.tar.bz2"):
        path = legacy_root / name
        unpacked, basis = archive_uncompressed_size(path)
        if unpacked is None:
            complete_projection = False
        else:
            projected += unpacked
        archives.append(
            {
                "name": name,
                "compressed_bytes": path.stat().st_size if path.is_file() else None,
                "projected_uncompressed_bytes": unpacked,
                "projection_basis": basis,
            }
        )
    return {
        "free_bytes": free_bytes,
        "projected_uncompressed_bytes": projected if complete_projection else None,
        "archives": archives,
        "policy": "selective or streaming inspection; no extraction performed",
    }
