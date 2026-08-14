"""Persistent Phase-3 training logs on Google Drive."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sailor.config import Settings
from sailor.paths import assert_writable_target
from sailor.reporting import write_json

CSV_FIELDS = [
    "epoch",
    "split",
    "loss",
    "train_loss",
    "dice_loss",
    "bce_loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "predicted_voxels",
    "target_voxels",
    "predicted_volume_mm3",
    "target_volume_mm3",
    "learning_rate",
    "epoch_seconds",
    "cumulative_seconds",
    "gpu_name",
    "peak_vram_bytes",
    "flags",
]


def append_csv_row(path: Path, row: dict[str, Any], settings: Settings) -> None:
    assert_writable_target(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def append_jsonl(path: Path, row: dict[str, Any], settings: Settings) -> None:
    assert_writable_target(path, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_training_summary(
    path: Path,
    payload: dict[str, Any],
    settings: Settings,
) -> None:
    write_json(path, payload, settings)


def log_failure(run_dir: Path, row: dict[str, Any], settings: Settings) -> None:
    payload = dict(row)
    payload["event"] = "FAILURE"
    append_jsonl(run_dir / "failures.jsonl", payload, settings)
    append_jsonl(run_dir / "training_log.jsonl", payload, settings)
