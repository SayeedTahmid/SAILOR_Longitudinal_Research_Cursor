"""Dry-run or execute the copy-only SAILOR_READY_v2.0 builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sailor.errors import StopProtocolError  # noqa: E402
from sailor.packaging.build import build_ready_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/content/drive/MyDrive/SAILOR_Longitudinal_Research_Cursor"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/content/drive/MyDrive/SAILOR_READY_v2.0"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-copy", action="store_true")
    args = parser.parse_args()
    try:
        result = build_ready_package(
            args.source,
            args.destination,
            execute=args.execute,
            approve_copy=args.approve_copy,
            verify_array_hashes=True,
        )
    except StopProtocolError as exc:
        print(exc.render(), file=sys.stderr)
        return 2
    if result["mode"] == "DRY_RUN":
        audit = result["audit"]
        result = {
            "mode": "DRY_RUN",
            "source": audit["source_root"],
            "destination": audit["destination_root"],
            "data_version": audit["data_version"],
            "preprocessing_version": audit["preprocessing_version"],
            "patients": audit["n_window_patients"],
            "sessions": audit["n_sessions"],
            "images": audit["n_images"],
            "masks": audit["n_masks"],
            "windows": audit["n_windows"],
            "copy_files": len(audit["copy_items"]),
            "copy_bytes": audit["copy_bytes"],
            "excluded_patients": audit["excluded_patients"],
            "absolute_path_risks": audit["absolute_path_risks"],
        }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
