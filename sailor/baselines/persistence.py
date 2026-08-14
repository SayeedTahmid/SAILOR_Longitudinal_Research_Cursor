"""C−1 persistence baseline: copy the last valid CL mask forward."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from sailor.baselines.io import history_sessions, load_mask
from sailor.errors import StopProtocolError
from sailor.evaluation.metrics import window_metrics


def persistence_prediction(
    artefacts: dict[str, Any],
    window: dict[str, Any],
    *,
    dataset_root,
) -> np.ndarray:
    _, last_history = history_sessions(window)
    return np.array(
        load_mask(
            artefacts,
            window["subject"],
            last_history,
            dataset_root=dataset_root,
        ),
        copy=True,
    )


def iter_persistence_scores(
    artefacts: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    dataset_root,
) -> Iterator[dict[str, Any]]:
    for window in windows:
        prediction = persistence_prediction(
            artefacts, window, dataset_root=dataset_root
        )
        target = load_mask(
            artefacts,
            window["subject"],
            window["target_mni_session"],
            dataset_root=dataset_root,
        )
        if prediction.shape != target.shape:
            raise StopProtocolError(
                f"Persistence shape mismatch for {window['window_id']}.",
                "C−1 cannot be scored on a shifted grid.",
                "Restore aligned p2.0 MRI/mask arrays.",
            )
        spacing = tuple(
            float(value)
            for value in artefacts["sessions"][
                (window["subject"], window["target_mni_session"])
            ].get("spacing", (1.0, 1.0, 1.0))
        )
        metrics = window_metrics(prediction, target, spacing=spacing)
        yield {
            "window_id": window["window_id"],
            "subject": window["subject"],
            "rung": "C-1",
            "target_mni_session": window["target_mni_session"],
            **metrics,
        }
