"""Headless Colab entry point for the CPU-only Stage-1 audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sailor.bootstrap import bootstrap_runtime  # noqa: E402
from sailor.data.audit import main  # noqa: E402


if __name__ == "__main__":
    bootstrap_runtime(
        mount_drive=True,
        install_missing=False,
        requirements_file=ROOT / "requirements.txt",
    )
    raise SystemExit(main())
