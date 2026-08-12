"""Headless Phase-2 dry-run or approved selective preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sailor.config import Settings  # noqa: E402
from sailor.errors import StopProtocolError  # noqa: E402
from sailor.preprocessing.stage2 import run_stage2_section  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, default=10, choices=range(10, 14))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-extraction", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run_stage2_section(
            args.section,
            Settings.from_env(),
            execute=args.execute,
            extraction_approved=args.approve_extraction,
            force=args.force,
        )
    except StopProtocolError as exc:
        print(exc.render(), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
