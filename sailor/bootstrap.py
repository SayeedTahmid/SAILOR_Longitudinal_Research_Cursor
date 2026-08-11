"""Single idempotent bootstrap entry point for Colab and headless runs."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.paths import create_output_tree
from sailor.reporting import load_dashboard

PINNED_PACKAGES = {
    "nibabel": "5.3.2",
    "numpy": "2.2.6",
    "PyYAML": "6.0.2",
}


@dataclass
class BootstrapState:
    settings: Settings
    dependencies: dict[str, str]
    dashboard: dict[str, Any]
    runtime: str


def _mount_drive() -> str:
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        return "non_colab"
    drive.mount("/content/drive", force_remount=False)
    return "colab"


def _dependency_status() -> tuple[dict[str, str], list[str]]:
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in PINNED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        observed[package] = actual
        if actual != expected:
            mismatches.append(f"{package}=={expected}")
    return observed, mismatches


def bootstrap_runtime(
    *,
    settings: Settings | None = None,
    mount_drive: bool = True,
    install_missing: bool = False,
    requirements_file: Path | None = None,
) -> BootstrapState:
    runtime = _mount_drive() if mount_drive else "headless"
    active = settings or Settings.from_env()
    active.validate()

    dependencies, mismatches = _dependency_status()
    if mismatches and install_missing:
        requirements = requirements_file or Path("requirements.txt")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        )
        dependencies, mismatches = _dependency_status()
    if mismatches:
        formatted = ", ".join(mismatches)
        raise RuntimeError(
            f"Pinned dependencies are absent or mismatched: {formatted}. "
            "Run bootstrap_runtime(install_missing=True)."
        )

    create_output_tree(active)
    return BootstrapState(
        settings=active,
        dependencies=dependencies,
        dashboard=load_dashboard(active),
        runtime=runtime,
    )
