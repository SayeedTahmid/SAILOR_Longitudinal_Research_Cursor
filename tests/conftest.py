from __future__ import annotations

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from sailor.config import Settings
from sailor.constants import CANONICAL_FILES


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, np.eye(4))
    nib.save(image, str(path))


def _sha512(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_project(tmp_path: Path) -> tuple[Settings, Path]:
    legacy = tmp_path / "legacy"
    output = tmp_path / "output"
    legacy.mkdir()

    overview = (
        "participant_id\tsession_id\ttreatment_status\n"
        "sub-01\tses-01\tCRT\n"
        "sub-01\tses-02\tTMZ\n"
    )
    (legacy / "overview.tsv").write_text(overview, encoding="utf-8")
    (legacy / "missing.tsv").write_text(
        "participant_id\tsession_id\tmissing_sequence\n"
        "sub-01\tses-01\tadc\n",
        encoding="utf-8",
    )
    (legacy / "raw-mni-link.tsv").write_text(
        "raw_session\tmni_session\n"
        "sub-01/ses-raw01\tsub-01/ses-01\n"
        "sub-01/ses-raw02\tsub-01/ses-02\n",
        encoding="utf-8",
    )
    (legacy / "src-to-raw.yaml").write_text("{}\n", encoding="utf-8")
    (legacy / "README.txt").write_text("synthetic fixture\n", encoding="utf-8")
    (legacy / "data-descriptor_a866425efff8.pdf").write_bytes(b"%PDF-synthetic")
    for name in (
        "code.tar.bz2",
        "rawdata_BIDS.tar.bz2",
        "derivatives.tar.bz2",
    ):
        (legacy / name).write_bytes(f"synthetic {name}".encode())

    shape = (8, 8, 8)
    for session, offset in (("ses-01", 0.0), ("ses-02", 1.0)):
        anat = (
            legacy
            / "derivatives"
            / "mni2009c-n-s"
            / "sub-01"
            / session
            / "anat"
        )
        _write_nifti(
            anat / f"sub-01_{session}_t1wc.nii.gz",
            np.full(shape, 100.0 + offset, dtype=np.float32),
        )
        mask = np.zeros(shape, dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        _write_nifti(
            anat / f"sub-01_{session}_CL_t1wc_enhancing_mask.nii.gz",
            mask,
        )
        _write_nifti(
            anat / f"sub-01_{session}_CL_t2wflair_mask.nii.gz",
            mask,
        )
        _write_nifti(
            anat / f"sub-01_{session}_ONCO_mask.nii.gz",
            mask,
        )
        _write_nifti(
            anat / f"sub-01_{session}_dosemap.nii.gz",
            np.full(shape, 30.0, dtype=np.float32),
        )
        scans = legacy / "rawdata_BIDS" / "sub-01" / session
        scans.mkdir(parents=True, exist_ok=True)
        day = "01" if session == "ses-01" else "15"
        (scans / f"sub-01_{session}_scans.tsv").write_text(
            "filename\tacq_time\n"
            f"anat/sub-01_{session}_t1wc.nii.gz\t2020-01-{day}T10:00:00\n",
            encoding="utf-8",
        )

    checksum_lines = []
    for name in CANONICAL_FILES:
        if name == "SHA512.txt":
            continue
        checksum_lines.append(f"{_sha512(legacy / name)}  {name}")
    (legacy / "SHA512.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    return Settings.for_testing(output, legacy), legacy
