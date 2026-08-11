"""Serializable Phase-1 records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GuardStatus = Literal["PASS", "FAIL", "INCONCLUSIVE"]


@dataclass
class GuardResult:
    guard_id: str
    status: GuardStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NiftiRecord:
    path: str
    source: str
    subject: str | None
    session: str | None
    sequence: str | None
    classification: str
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    affine: tuple[tuple[float, ...], ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    finite: bool | None
    nonzero_voxels: int | None = None
    total_voxels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompletionRecord:
    section: int
    stage: int
    status: str
    owner: str
    data_version: str
    model_version: str
    preprocessing_version: str
    feature_shape: list[int]
    primary_target_mask: str
    primary_target_component: str
    conditioning_rung: str
    fold_scheme: str
    guards_passed: list[str]
    guards_failed: list[str]
    n_patients: int | None
    n_sessions: int | None
    n_pairs: int | None
    seed: int
    gpu: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    implementation_id: str
    timestamp: str
    profiled: bool = False
    resource_measurements: dict[str, Any] = field(
        default_factory=lambda: {
            "compute_mode": "CPU-only",
            "system_ram": "UNMEASURED",
            "disk": "UNMEASURED",
            "wall_time": "UNMEASURED",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
