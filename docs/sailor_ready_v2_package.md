# SAILOR_READY v2.0 Distribution Package

## Purpose

`SAILOR_READY_v2.0` is a portable, read-only distribution copy of the frozen
Phase-2 dataset. It is intended for three collaborating researchers to run the
approved baseline experiments from their own Google Drive or Colab paths.

The original Phase-2 root remains authoritative. The package builder never
moves, renames, deletes, rewrites, or reorganizes source artefacts.

## Included

- 230 normalized `T1c-icor` arrays
- 230 validated binary `CL / enhancing_t1wc` masks
- 178 longitudinal windows from 25 patients
- frozen 5-fold × 3-repeat nested patient-level folds
- portable preprocessing, window, fold, treatment, and timing manifests
- one-row-per-session CSV metadata
- Phase-2 QC reports and visualizations
- byte-preserved source manifests under `manifests/provenance/`
- package-level SHA-256 inventory
- self-contained read-only `loader.py`

## Excluded

- raw/BIDS/DICOM/NIfTI data
- `derivatives.tar.bz2` and other canonical archives
- `raw_needed.tar`
- dose-map binaries
- non-finite optional modalities
- old TaDiff arrays, checkpoints, and splits
- all quarantined artefacts

## Build safety

The builder defaults to a read-only dry run. Execution requires both
`--execute` and `--approve-copy`.

The package is built under a hidden sibling staging directory. Every copied
array is checked against its frozen Phase-2 SHA-256. Operational manifests are
rewritten with relative paths while exact source manifests are copied unchanged
for provenance. The source files are rehashed after copying.

The staging folder is promoted only if the package integrity audit passes. An
existing `SAILOR_READY_v2.0` destination is never overwritten.

The execution report explicitly records:

- confirmation that source Phase-2 files were not moved, renamed, modified,
  deleted, or overwritten;
- the number of source and destination checksums compared and any mismatch;
- the exact untouched authoritative source root; and
- the exact promoted distribution-package root.

These absolute roots appear only in the local build report, not as required
operational paths inside the portable package.

## Commands in Colab

Dry run:

```bash
python scripts/build_sailor_ready.py
```

Approved copy:

```bash
python scripts/build_sailor_ready.py --execute --approve-copy
```

Independent verification:

```bash
python scripts/verify_sailor_ready.py \
  /content/drive/MyDrive/SAILOR_READY_v2.0
```

## Teammate loader

```python
from pathlib import Path
import sys

DATA_ROOT = Path("/your/drive/path/SAILOR_READY_v2.0")
sys.path.insert(0, str(DATA_ROOT))

from loader import ReadyDataset

dataset = ReadyDataset(DATA_ROOT)
window = next(dataset.iter_windows())
image, mask, session = dataset.load_session(
    window["subject"],
    window["target_mni_session"],
)
fold = dataset.get_outer_fold(0, 0)
```

The loader only reads package-relative paths and never modifies the package.

## Scientific boundary

Approved:

- persistence baseline
- MRI-history-only baseline
- MRI plus approximate Δt baseline

Not approved:

- final treatment-aware claims
- dose-aware modeling
- exact time-from-surgery claims
- causal treatment-effect claims

Timing remains `approximate_mni_intervals`; missing treatment information
remains missing.
