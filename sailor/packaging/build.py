"""Transactional, copy-only construction of SAILOR_READY_v2.0."""

from __future__ import annotations

import json
import ctypes
import errno
import os
import shutil
from pathlib import Path
from typing import Any

from sailor.errors import StopProtocolError
from sailor.packaging.audit import audit_frozen_source
from sailor.packaging.common import (
    file_hash_map,
    safe_relative_path,
    sha256_file,
    write_json,
)
from sailor.packaging.manifests import write_portable_manifests
from sailor.packaging.verify import verify_ready_package


def _rename_no_replace(source: Path, destination: Path) -> str:
    if os.name == "nt":
        os.rename(source, destination)
        return "windows_rename_no_replace"
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StopProtocolError(
            "Atomic no-replace rename is unavailable on this runtime.",
            "The package cannot guarantee destination preservation.",
            "Use a Linux/Colab runtime exposing renameat2.",
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise StopProtocolError(
                f"Destination appeared during promotion: {destination}",
                "Atomic no-overwrite protection blocked replacement.",
                "Inspect the existing destination before any new package attempt.",
            )
        if error in {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "EOPNOTSUPP", 95),
        }:
            if destination.exists():
                raise StopProtocolError(
                    f"Destination exists before Drive-compatible promotion: {destination}",
                    "The no-overwrite package rule would be violated.",
                    "Inspect the destination and do not continue automatically.",
                )
            os.rename(source, destination)
            return "exclusive_lock_drive_rename_fallback"
        raise OSError(error, os.strerror(error), str(destination))
    return "renameat2_noreplace"


def _release_lock(fd: int, path: Path) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _write_checksums(root: Path) -> dict[str, str]:
    checksums = file_hash_map(root, exclude={"CHECKSUMS.sha256"})
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksums


def _readme(audit: dict[str, Any]) -> str:
    counts = {
        "patients": audit["n_window_patients"],
        "sessions": audit["n_sessions"],
        "windows": audit["n_windows"],
    }
    return f"""# SAILOR_READY v2.0

This is a frozen, derived distribution copy of the approved SAILOR Phase-2
dataset. The original Phase-2 project root remains authoritative.

## Versions

- DATA_VERSION: v2.0
- PREPROCESSING_VERSION: p2.0
- Input: normalized T1c-icor
- Target: CL / enhancing_t1wc

## Contents

- Patients contributing longitudinal windows: {counts['patients']}
- Preprocessed MRI/mask sessions: {counts['sessions']}
- Longitudinal prediction windows: {counts['windows']}
- Fold scheme: {audit['fold_scheme']}
- Timing: {audit['timing_provenance']} (not exact acquisition dates)

## Scientific scope

Approved:

- persistence baseline
- MRI-history-only baseline
- MRI plus approximate delta-t baseline

Not approved:

- final treatment-aware claims
- dose-aware modeling
- exact time-from-surgery claims
- causal treatment-effect claims

Dose-map binaries are not included. Missing treatment labels remain missing.
Team members must not independently re-preprocess or modify this folder.

## Python / Colab use

Set DATA_ROOT to this folder, regardless of where Drive is mounted:

```python
from pathlib import Path
import sys

DATA_ROOT = Path("/path/to/SAILOR_READY_v2.0")
sys.path.insert(0, str(DATA_ROOT))

from loader import ReadyDataset

dataset = ReadyDataset(DATA_ROOT)
window = next(dataset.iter_windows())
image, mask, session = dataset.load_session(
    window["subject"],
    window["target_mni_session"],
)
fold = dataset.get_outer_fold(repeat=0, outer_fold=0)
```

All operational paths are relative to DATA_ROOT. Run the package verifier before
training and compare `CHECKSUMS.sha256` after any transfer.

## Important limitations

- Only 25 independent patients contribute longitudinal windows.
- MNI interval timing is approximate.
- There is no verified surgery-date anchor.
- Treatment labels contain missing values.
- Dose-map registration is not validated for final modeling.
- Optional non-finite modalities remain excluded.
"""


def build_ready_package(
    source_root: Path,
    destination_root: Path,
    *,
    execute: bool = False,
    approve_copy: bool = False,
    verify_array_hashes: bool = True,
) -> dict[str, Any]:
    audit = audit_frozen_source(
        source_root,
        destination_root,
        verify_array_hashes=verify_array_hashes,
    )
    if not execute:
        return {"mode": "DRY_RUN", "audit": audit}
    if not approve_copy:
        raise StopProtocolError(
            "Ready-dataset copy has not been explicitly approved.",
            "Approximately 9+ GiB of medical arrays would be written.",
            "Review the dry-run plan and rerun with approve_copy=True.",
        )

    destination_root = destination_root.resolve()
    staging = destination_root.with_name(f".{destination_root.name}.build")
    if staging.exists():
        raise StopProtocolError(
            f"Packaging staging directory already exists: {staging}",
            "A previous incomplete attempt may require forensic review.",
            "Inspect it manually; the builder will not clean or overwrite it.",
        )
    free = shutil.disk_usage(destination_root.parent).free
    required = int(audit["copy_bytes"] * 1.10)
    if free < required:
        raise StopProtocolError(
            f"Only {free} bytes free; package requires {required} bytes with margin.",
            "The copy could fail part-way.",
            "Free space or choose a different approved destination.",
        )

    lock_path = destination_root.with_name(f".{destination_root.name}.lock")
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise StopProtocolError(
            f"Packaging lock already exists: {lock_path}",
            "Another build or unresolved prior attempt may be active.",
            "Inspect the lock and destination; do not overwrite either.",
        ) from exc
    source_pre_hashes = {
        item["source"]: item["sha256"] for item in audit["copy_items"]
    }
    try:
        staging.mkdir(parents=False)
        for item in audit["copy_items"]:
            relative = safe_relative_path(item["destination"])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source"], destination)
            if sha256_file(destination) != item["sha256"]:
                raise StopProtocolError(
                    f"Copied byte checksum mismatch: {relative.as_posix()}",
                    "The staged package differs from the authoritative source.",
                    "Stop and inspect Drive transfer integrity.",
                )

        generated = write_portable_manifests(staging, audit)
        loader_source = Path(__file__).resolve().parents[1] / "data" / "ready.py"
        shutil.copyfile(loader_source, staging / "loader.py")
        (staging / "README.md").write_text(_readme(audit), encoding="utf-8")
        (staging / "DATASET_VERSION.txt").write_text(
            "DATA_VERSION=v2.0\n"
            "PREPROCESSING_VERSION=p2.0\n"
            "PACKAGE_KIND=SAILOR_READY\n",
            encoding="utf-8",
        )

        package_manifest_path = generated["paths"]["package_manifest"]
        package_manifest = json.loads(
            package_manifest_path.read_text(encoding="utf-8")
        )
        package_manifest["file_checksums"] = file_hash_map(
            staging,
            exclude={
                "CHECKSUMS.sha256",
                "manifests/package_manifest.json",
            },
        )
        write_json(package_manifest_path, package_manifest)
        _write_checksums(staging)

        pending_audit = verify_ready_package(staging, allow_pending=True)
        if pending_audit["status"] != "PASS":
            raise StopProtocolError(
                f"Staged ready-package audit failed: {pending_audit['failures']}",
                "The distribution is not safe to promote.",
                "Correct package code or source issues; do not alter frozen artefacts.",
            )

        source_post_hashes = {
            source: sha256_file(Path(source)) for source in source_pre_hashes
        }
        changed_sources = [
            source
            for source, before in source_pre_hashes.items()
            if source_post_hashes[source] != before
        ]
        if changed_sources:
            raise StopProtocolError(
                f"Authoritative source bytes changed during packaging: {changed_sources[:20]}",
                "The frozen Phase-2 artefact is no longer proven unchanged.",
                "Stop, preserve evidence, and restore from the authoritative snapshot.",
            )

        package_manifest["package_status"] = "READY_TO_TRAIN"
        package_manifest["integrity_audit"] = {
            "status": "PASS",
            "source_files_rehashed": len(source_post_hashes),
            "source_bytes_unchanged": True,
            "staged_failures": [],
        }
        write_json(package_manifest_path, package_manifest)
        _write_checksums(staging)
        final_audit = verify_ready_package(staging)
        if final_audit["status"] != "PASS":
            raise StopProtocolError(
                f"Final ready-package audit failed: {final_audit['failures']}",
                "The distribution cannot be marked READY-TO-TRAIN.",
                "Do not promote the staged directory.",
            )
        promotion_mode = _rename_no_replace(staging, destination_root)
        _release_lock(lock_fd, lock_path)
        return {
            "mode": "EXECUTE",
            "authoritative_source": str(Path(audit["source_root"]).resolve()),
            "destination": str(destination_root),
            "audit": final_audit,
            "source_bytes_unchanged": True,
            "source_preservation_confirmation": (
                "Original Phase-2 frozen artefacts were not moved, renamed, "
                "modified, deleted, or overwritten."
            ),
            "source_vs_package_checksum_comparison": {
                "source_files_rehashed": len(source_post_hashes),
                "copied_files_matched": len(audit["copy_items"]),
                "mismatches": [],
            },
            "copy_bytes": audit["copy_bytes"],
            "promotion_mode": promotion_mode,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        _release_lock(lock_fd, lock_path)
        raise
