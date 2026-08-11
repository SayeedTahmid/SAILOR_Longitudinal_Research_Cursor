"""Output-tree creation and provenance-firewall checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sailor.config import Settings
from sailor.constants import OUTPUT_DIRECTORIES
from sailor.errors import StopProtocolError


def assert_writable_target(path: Path, settings: Settings) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(settings.dataset_root.resolve())
    except ValueError as exc:
        raise StopProtocolError(
            f"Attempted write outside DATASET_ROOT: {resolved}",
            "The provenance firewall cannot guarantee legacy inputs remain immutable.",
            "Redirect the write beneath the configured DATASET_ROOT.",
        ) from exc
    try:
        resolved.relative_to(settings.legacy_root.resolve())
    except ValueError:
        return
    raise StopProtocolError(
        f"Attempted write beneath LEGACY_ROOT: {resolved}",
        "Immutable canonical or quarantined input could be altered.",
        "Write only beneath DATASET_ROOT.",
    )


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def verify_snapshot_unchanged(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> None:
    if before != after:
        changed = sorted(set(before) ^ set(after))
        changed.extend(
            key
            for key in set(before) & set(after)
            if before[key] != after[key]
        )
        raise StopProtocolError(
            f"LEGACY_ROOT changed during audit: {sorted(set(changed))[:20]}",
            "Input provenance and read-only guarantees are invalidated.",
            "Restore the legacy tree from its verified source and rerun read-only.",
        )


def create_output_tree(settings: Settings) -> dict[str, str]:
    settings.validate()
    assert_writable_target(settings.dataset_root, settings)
    settings.dataset_root.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}
    for relative in OUTPUT_DIRECTORIES:
        path = settings.dataset_root / relative
        assert_writable_target(path, settings)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".sailor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        created[relative] = str(path)

    readme = settings.dataset_root / "README.md"
    if not readme.exists():
        payload = (
            "# SAILOR data and artefact root\n\n"
            f"Created: {datetime.now(timezone.utc).isoformat()}\n\n"
            f"Data version: `{settings.data_version}`\n\n"
            "Primary target: `CL / enhancing_t1wc`\n\n"
            f"Legacy source (read-only): `{settings.legacy_root}`\n"
        )
        readme.write_text(payload, encoding="utf-8")

    pointer_manifest = settings.dataset_root / "00_CANONICAL" / "v2_pointers.json"
    pointer_manifest.write_text(
        json.dumps(
            {
                "legacy_root": str(settings.legacy_root),
                "copy_policy": "references_only",
                "data_version": settings.data_version,
                "implementation_id": settings.implementation_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return created
