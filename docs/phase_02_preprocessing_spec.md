# Phase 2 Preprocessing — Easy Explanation

## Goal

Phase 2 turns the verified Phase 1 inventory into a small, consistent, and
leakage-safe dataset for later baseline models. It does not train a model.

## Locked choices

- MRI input: `T1c-icor`
- Ground truth: `CL / enhancing_t1wc`
- History: at least two earlier valid scans, followed by one future target
- Evaluation topology: five patient folds, repeated three times
- Inner selection topology: four patient folds inside each outer training set
- Preprocessing version: `p2.0`

If `T1c-icor` does not cover and align with every selected Phase 1 session, the
pipeline stops. It does not silently switch to another MRI variant.

## Dry run first

Section 10 first creates a dry-run report. It lists:

- the exact files planned for extraction
- missing or duplicate files
- shape and affine mismatches
- expected output and staging disk space
- available Drive space
- T1c variant coverage

No medical array is written until selective extraction is explicitly approved.

## What gets written

Only the approved T1c image and CL mask are copied out of the verified
derivatives archive. The legacy `sailor_v1` folder remains read-only.

The MRI is stored as a memory-mappable float32 `.npy` array. The mask is stored
as a uint8 `.npy` array. Each output receives a checksum and a manifest entry.

Six valid CL masks use a constant foreground value other than 1. Their exact
positive values were measured from the canonical archive and pre-registered by
session. Phase 2 accepts only those six exact session/value combinations,
converts positive foreground to 1, and records the original scale. Any mask
with multiple positive values or a different scale still triggers STOP.

## Normalization

Normalization uses only the current MRI volume and its aligned brain mask:

1. measure intensities inside the brain
2. clip the lowest 0.5% and highest 0.5%
3. subtract the clipped median
4. divide by the clipped interquartile range
5. keep background voxels at zero

This does not use other patients, validation patients, or test patients.
Cohort-wide normalization is forbidden. If later models need a cohort-level
scaler, it must be fitted separately inside each outer training fold.

## Data that remains excluded

- Seven all-zero CL masks remain missing labels.
- ONCO never becomes primary ground truth.
- CL edema remains a sensitivity target.
- The 416 optional MRI volumes containing NaN or infinity remain blocked.
- Old TaDiff arrays, splits, and checkpoints remain quarantined.

## Longitudinal windows

Sessions are ordered using verified raw acquisition dates and joined to MNI
sessions only through `raw-mni-link.tsv`.

For each patient:

- the third eligible scan can be the first target
- all earlier eligible scans become its history
- later targets receive longer histories
- every interval stores its number of days and timing provenance
- treatment `unknown` is stored as missingness, not a treatment class

Patients with fewer than three eligible scans cannot create a window and are
reported.

## Patient-level cross-validation

The same patient can never appear in both training and testing for one fold.

Phase 2 creates:

- five outer folds
- three deterministic repeats derived from seed 1337
- four inner folds inside every outer training group

Eligible-window counts may balance the folds. Tumour outcomes and model scores
may not influence fold assignment.

## Leakage guard

Section 12 fails if it finds:

- the same patient in train and test
- the same qualified target in train and test
- outer-test patients inside an inner fold
- incomplete patient coverage
- normalization using cohort or test information
- old quarantined paths
- a changed fold scheme

## Notebook sections

- Section 10: dry run, approved selective preprocessing, and windows
- Section 11: nested patient-level CV manifest
- Section 12: full G5 leakage checks
- Section 13: final Phase 2 QC and dashboard state

Each section can run from a fresh runtime using persisted Drive artefacts.

## Phase 2 gate

Phase 2 is complete only when:

- preprocessing is deterministic
- output checksums and manifests exist
- masks and MRI grids align
- no invalid primary labels enter windows
- patient folds and inner folds are frozen
- G5 passes
- sections 10–13 are complete from a clean commit
- the dashboard has no failed guards

Model training starts only after separate approval for Phase 3.
