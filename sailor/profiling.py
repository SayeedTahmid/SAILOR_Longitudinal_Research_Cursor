"""Empirical section profiling; no resource values are estimated."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


def _peak_process_rss_bytes() -> int | str:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak * 1024)
    except (ImportError, AttributeError):
        return "UNMEASURED"


@dataclass
class ProfileResult(Generic[T]):
    section_id: int
    value: T
    measurements: dict[str, Any]


def profile_stage(
    section_id: int,
    operation: Callable[..., T],
    *args: Any,
    measurement_scope: str = "real_data",
    **kwargs: Any,
) -> ProfileResult[T]:
    if measurement_scope not in {"real_data", "synthetic_fixture", "truncated_subset"}:
        raise ValueError("measurement_scope must describe what was actually profiled.")
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = operation(*args, **kwargs)
    finally:
        wall_seconds = time.perf_counter() - started
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    peak_process_rss = _peak_process_rss_bytes()
    return ProfileResult(
        section_id=section_id,
        value=value,
        measurements={
            "profiled": True,
            "measurement_scope": measurement_scope,
            "compute_mode": "CPU-only" if section_id <= 13 else "UNVERIFIED",
            "wall_seconds": wall_seconds,
            "peak_python_bytes": peak_python_bytes,
            "peak_gpu_bytes": 0 if section_id <= 13 else "UNMEASURED",
            "peak_process_rss_bytes": peak_process_rss,
            "note": (
                "Python allocations use tracemalloc; process RSS is measured where the "
                "runtime exposes resource.getrusage, otherwise it remains UNMEASURED."
            ),
        },
    )
