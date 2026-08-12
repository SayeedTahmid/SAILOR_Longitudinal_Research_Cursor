# SAILOR Dataset and Phase 1 — Explained in Easy Words

## 1. What this project is trying to do

The long-term goal is to use a patient's earlier brain MRI scans and treatment
information to predict what the enhancing tumour may look like at a later time.

The project is not ready to train a model yet. Phase 1 was about understanding
and checking the data first. This is important because a model can produce
convincing but incorrect results if sessions are matched incorrectly, missing
masks are treated as real labels, or old experimental files leak into the new
pipeline.

## 2. What the SAILOR dataset contains

SAILOR contains longitudinal data from **27 patients with high-grade glioma**.
Longitudinal means that each patient was scanned multiple times instead of only
once.

The official overview contains **337 raw/source sessions**. The processed MNI
archive contains **270 sessions**. These numbers differ because not every raw
session has a processed MNI version.

Phase 1 inspected **6,423 NIfTI files** inside the canonical derivatives archive.
This number includes MRI volumes, tumour masks, brain masks, segmentations, dose
maps, and other derived files. It does not mean there are 6,423 independent
training examples.

### MRI data

The processed archive contains the main structural MRI types:

- `T1`
- `T1c` — contrast-enhanced T1 MRI
- `T2`
- `Flair`

It also contains registered and normalized variants, such as:

- `T1c-icor`
- `T1c-icor-zscore`
- `T1-icor`
- `T2-icor`
- `Flair-icor`

There are also functional or derived volumes such as `rCBF` and `rCBV`, plus
brain-extraction masks, normal-appearing white-matter masks, and FastSurfer
segmentations.

For the primary task, the most important MRI family is **T1c** because the
locked target is the contrast-enhancing tumour. Phase 1 measured **810
T1c-family volumes** across the plain, registered, and z-score variants. All 810
were finite: none contained NaN or infinity.

Phase 2 still has to decide which one of these T1c variants will be the
canonical model input. That decision must use measured geometry and intensity
information, not convenience or model results.

### Tumour masks

Two CL mask filenames were verified directly from the archive:

- `ContrastEnhancedMask-CL.nii.gz`  
  This is the locked primary target: `CL / enhancing_t1wc`.
- `EdemaMask-CL.nii.gz`  
  This is a secondary or sensitivity target and cannot replace the primary
  target.

The audit found **240 primary contrast-enhancing CL masks**:

- **233 valid masks**
- **7 all-zero masks**

The seven all-zero masks are:

- `sub-25/ses-03`
- `sub-25/ses-04`
- `sub-25/ses-05`
- `sub-25/ses-06`
- `sub-25/ses-07`
- `sub-25/ses-08`
- `sub-25/ses-10`

Under the locked project rules, an all-zero mask is treated as a **missing
label**, not as proof that the patient has no tumour. These seven sessions are
excluded from training windows and scoring. They remain listed in the audit so
the exclusion is never hidden.

The archive also contains **ONCO** masks. ONCO is an automatic segmentation
output, not the manually annotated primary ground truth. Phase 1 inventoried
these files, but they did not influence patient selection, preprocessing,
hyperparameters, or the primary target.

### Treatment and clinical information

The official dataset descriptor says the data includes:

- treatment status such as `CRT`, `TMZ`, `no`, and `unknown`
- radiation dose maps
- time between scans
- age and sex
- overall survival
- RANO response information

`unknown` treatment status means the information is missing. It must not be
treated as a real fourth treatment type.

Treatment status is strongly connected to time because most patients follow a
similar treatment schedule. A model could appear “treatment-aware” while only
learning how many weeks have passed. Later experiments must compare treatment
against time-only controls before making a treatment-awareness claim.

### Radiation dose maps

The canonical derivatives archive contains **26 dose-map files**, covering 26
patients. The Phase 1 dose prerequisite guard passed.

A radiation dose map is mostly a static patient-level spatial map from CRT. It
must not be described as a changing TMZ dose. Later experiments must also
shuffle dose maps between patients to test whether a model is using real dose
information instead of tumour location or patient identity.

## 3. How sessions are matched

The raw and MNI archives do not use interchangeable session numbers. For
example, raw `ses-03` is not automatically MNI `ses-03`.

The official mapping file was found inside the derivatives archive:

`derivatives/mni2009c-n-s/raw-mni-link.tsv`

It contains **337 rows**. Phase 1 parsed it using the subject, raw-session, and
MNI-session columns.

Some rows have `mni session = no`. This is an official statement that the raw
session has no MNI counterpart. It is not an error and must never be replaced
by a guessed matching session.

After applying the mapping file, the official missing-data file, and the valid
primary-mask requirement, **230 patient-sessions remain eligible** for the
primary data foundation.

This is still not the number of independent samples. The independent unit is
the patient, so the effective sample size remains 27 patients.

## 4. What `missing.tsv` means

`missing.tsv` has one row per raw session and one column per sequence.

- `y` means the sequence is missing.
- `n` means the sequence is present.

For `t1wc`:

- 15 raw sessions are marked missing.
- 322 raw sessions are marked present.

The first version of the audit did not understand this matrix correctly. Phase
1 diagnosed the problem from the real file and corrected the parser. The
current pipeline now excludes only sessions explicitly marked `t1wc = y`.

## 5. Data problems discovered in Phase 1

### Seven unusable primary masks

Seven `ContrastEnhancedMask-CL` files are all zero. They are now recorded as
missing labels and excluded. They are never used as negative examples.

### Non-finite values in optional MRI volumes

Phase 1 found **416 optional-modality volumes** containing NaN or infinity:

- 206 `T2-icor`
- 206 `T2-icor-zscore`
- 1 `T1-icor`
- 1 `T1-icor-zscore`
- 1 `Flair-icor`
- 1 `Flair-icor-zscore`

These files are blocked from model input for now. Phase 2 must define and test a
deterministic handling policy before any of them can be used. Phase 1 did not
silently replace, clip, or repair their values.

The locked T1c family does not have this problem.

### Intensity values are not simply 0–255

The descriptor discusses MNI intensity scaling, but measured files include
`uint8`, `float32`, and `float64`, and many derived volumes fall outside the
0–255 range.

Therefore the project cannot safely divide every image by 255. Phase 2 must
choose normalization from measured training-fold statistics.

### Raw and MNI session counts differ

There are 337 raw sessions and 270 MNI sessions. Guessing session matches would
cause wrong dates, treatments, or targets to be attached to an MRI. The
pipeline now uses only the official mapping file.

### MNI preprocessing is heterogeneous

The official descriptor says preprocessing choices differed between patients.
Brain extraction, registration order, and reference sessions were not
identical for everyone. Even though images are in a common MNI space, Phase 2
must still check shape, affine, spacing, and visual quality.

### Timing must keep its provenance

The descriptor warns that MNI time intervals may be approximate. Phase 1 ran
the G7 timing-provenance guard and stored the result in the dataset manifest.
Any later code must read that recorded provenance rather than treating every
interval as exact.

### Treatment status is confounded with time

CRT and TMZ usually happen in a similar order for most patients. Treatment
status may therefore be predictable from weeks since surgery. A simple
“with-treatment versus without-treatment” comparison is not enough.

The project has a separate, pre-written confound plan using:

- mutual information
- a time-only treatment classifier
- treatment permutation
- dose-map permutation
- patient-level confidence intervals

### The dataset is small

There are only 27 independent patients. Hundreds of sessions cannot be treated
as hundreds of independent people.

The project must use patient-level cross-validation, patient-level bootstrap
confidence intervals, and inner-loop hyperparameter selection. A single
train/validation/test split would give unreliable conclusions.

## 6. Files that were deliberately not used

The `sailor_v1` folder mixes official EBRAINS files with artefacts from earlier
TaDiff experiments.

The following kinds of files were quarantined and never used as Phase 1 inputs:

- old checkpoints
- old latent arrays
- old pair files
- old session whitelists
- old train/test splits
- prior persistence results
- the old TaDiff working directory

Reusing an old split could leak information from earlier experiments into the
new result.

Two files remain ambiguous:

- `raw_needed.tar`
- `dosemaps.tar`

They are repackaged `.tar` files rather than official `.tar.bz2` archives.
Phase 1 recorded them but did not trust or use them. Canonical dose maps were
already found in the verified derivatives archive, so `dosemaps.tar` was not
needed.

## 7. What Phase 1 actually did

Phase 1 completed the following work:

1. Locked the primary target to `CL / enhancing_t1wc`.
2. Locked this repository as the primary implementation.
3. Kept `sailor_v1` read-only.
4. Created a separate output root:
   `/content/drive/MyDrive/SAILOR_Longitudinal_Research_Cursor`.
5. Verified the SHA-512 checksums of the three main archives:
   `code.tar.bz2`, `rawdata_BIDS.tar.bz2`, and `derivatives.tar.bz2`.
6. Separated canonical, quarantined, and ambiguous files.
7. Streamed the large archives without extracting them into the legacy folder.
8. Inventoried 6,423 NIfTI files, including shapes, spacing, dtype, ranges, and
   finite-value status.
9. Resolved the real CL mask filenames.
10. Counted and excluded the seven invalid primary masks.
11. Parsed the official raw-to-MNI mapping without guessing session numbers.
12. Parsed the `missing.tsv` `y`/`n` matrix correctly.
13. Verified 230 eligible primary patient-sessions.
14. Inventoried treatment and dose prerequisites.
15. Ran integrity guards G1, G5, G7, G8, G9, and G10.
16. Wrote manifests, QC reports, gap reports, and section-completion records to
    Drive.
17. Reconciled stale failure records so the dashboard now reports:
    Sections 01–09 complete and no failed guards.
18. Added automated tests. The current local suite has 19 passing tests.

The successful full audit was produced from clean Git commit `dc6beba`. The
later commit `f6be399` only corrected dashboard state reconciliation; it did
not change the measured scientific results.

## 8. What Phase 1 did not do

Phase 1 did not:

- preprocess or normalize model inputs
- create training windows
- create patient-level cross-validation folds
- train a neural network
- choose a foundation encoder
- tune hyperparameters
- calculate prediction accuracy
- claim treatment awareness
- download extra EBRAINS data
- use ONCO as primary ground truth
- reuse old checkpoints or old splits

There are no model results yet. Any accuracy or clinical claim would be
premature.

## 9. Where the Phase 1 outputs are stored

The main generated files are under:

`/content/drive/MyDrive/SAILOR_Longitudinal_Research_Cursor`

Important outputs include:

- `01_DATA_FOUNDATION/v2_canonical_manifest.json`
- `01_DATA_FOUNDATION/v2_dataset_manifest.json`
- `01_DATA_FOUNDATION/state/section_XX_complete.json`
- `06_QC_REPORTS/v2_stage1_qc_report.json`
- `06_QC_REPORTS/v2_gap_report.json`

These reports are generated from measured data. They are the source of truth,
not handwritten notes.

## 10. What happens in Phase 2

Phase 2 is deterministic preprocessing. It should:

1. Select the exact canonical T1c input variant from measured QC.
2. Verify image, mask, and dose-map geometry and affines.
3. Define normalization using training-fold statistics only.
4. Keep the 416 non-finite optional volumes blocked unless an approved,
   tested policy makes them usable.
5. Keep the seven all-zero CL masks excluded.
6. Join raw timing/treatment information to MNI images only through the
   official mapping.
7. Build longitudinal windows without patient or target overlap.
8. Generate and freeze patient-level cross-validation manifests.
9. Write versioned preprocessing outputs and fresh QC reports.

No model training should begin until the Phase 2 gate passes.

## 11. Short version

The dataset is useful because it contains repeated brain MRI scans, expert CL
tumour masks, treatment information, and dose maps. It is difficult because it
is small, sessions differ between raw and MNI versions, some labels are
missing, some optional MRI files contain invalid numbers, preprocessing varies
between patients, and treatment is strongly tied to time.

Phase 1 did not solve these problems by hiding them. It measured them, recorded
them, excluded only the invalid primary labels, blocked unsafe optional data,
and created a clean, reproducible foundation for Phase 2.
