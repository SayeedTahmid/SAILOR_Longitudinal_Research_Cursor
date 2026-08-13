# SAILOR Phase 2 Results — Explained in Easy Words

## 1. What Phase 2 was for

Phase 1 checked what data exists and which files are safe to use. Phase 2 took
that verified inventory and created a consistent dataset for later baseline
models.

Phase 2 did **not** train a model or calculate prediction accuracy. Its job was
to:

- select one MRI input type;
- preprocess it consistently;
- preserve only valid tumour masks;
- connect sessions to timing and treatment information;
- build longitudinal prediction windows;
- freeze patient-level cross-validation folds; and
- prove that no patient or target leaks between training and testing.

## 2. The locked input and target

### MRI input

The selected MRI input is:

`T1c-icor`

This is a contrast-enhanced T1 MRI variant. It was selected before any model
result was seen.

The pre-zscored `T1c-icor-zscore` variant was not selected because Phase 2
applies its own documented normalization. This avoids depending on hidden or
inconsistent earlier normalization.

### Tumour target

The primary target remains:

`CL / enhancing_t1wc`

Its canonical filename is:

`ContrastEnhancedMask-CL.nii.gz`

The target was not changed to ONCO, edema, or another mask after seeing the
data.

## 3. The Phase 2 dry run

Before writing medical arrays, Phase 2 performed a dry run.

The dry run measured:

- Phase 1 eligible sessions: **230**
- Selected sessions after the final verified policy: **230**
- Selection issues after correction: **none**
- Required temporary/output space with margin: about **55.4 GB**
- Free Drive space before extraction: about **157.6 GB**

This meant extraction was feasible without filling the Drive.

No medical array was written until the dry-run report was reviewed and
extraction was explicitly approved.

## 4. How MRI volumes were normalized

Each T1c volume was normalized independently using its aligned brain-support
mask.

The procedure was:

1. keep intensities inside the brain support;
2. measure the 0.5th and 99.5th percentiles;
3. clip values outside that range;
4. subtract the clipped median;
5. divide by the clipped interquartile range; and
6. leave voxels outside the brain at zero.

The processed MRI is stored as `float32`.

No whole-cohort mean or standard deviation was used. No validation or test
patient contributed statistics to another patient. This prevents normalization
leakage.

If a later model needs a cohort-level scaler, it must be fitted separately
inside each outer training fold.

## 5. Brain-mask interpolation

Some `BrainExtractionMask` files contained values between 0 and 1. They were
not corrupted; they were interpolated support masks.

These masks are used only to define the MRI normalization region. They are not
tumour ground truth.

Phase 2 therefore:

- requires every brain-support value to be finite and within 0–1;
- uses values `>= 0.5` as brain support;
- verifies the result is neither empty nor almost the whole volume; and
- never applies this rule to the CL tumour label.

## 6. CL tumour-mask scaling and numerical noise

The CL masks did not all use the same numerical foreground value.

Six masks used unusual constant foreground scales:

- four used approximately `0.001`;
- one used approximately `0.5046`; and
- one used `5000`.

Many otherwise normal masks also contained tiny positive resampling noise.

All **233 valid primary masks** were inspected. The dataset-wide diagnostic
showed:

- all 233 masks were inspected successfully;
- median foreground retention under relative thresholding was 100%;
- no mask retained less than 25%;
- only five retained less than 50% of all merely-positive voxels; and
- thresholds from normalized 0.1 through 0.9 gave the same foreground counts
  in the worst examples, showing a clear gap between numerical noise and true
  foreground.

The locked conversion therefore:

1. checks that the source maximum matches the Phase 1 manifest;
2. divides mask values by that maximum;
3. stops if any value lies in the ambiguous normalized `[0.1, 0.9)` band;
4. converts normalized values `>= 0.5` to foreground;
5. removes tiny values below `0.1` as numerical noise; and
6. reruns the G1 degeneracy checks.

The original foreground maximum is recorded for every output mask.

The seven all-zero masks found in Phase 1 remain missing labels. They were not
restored or treated as negative examples.

## 7. Safe selective extraction

Phase 2 did not extract the entire derivatives archive.

It selected only the required MRI, CL mask, and brain mask for the approved
cohort.

Before extraction it:

- recomputed the derivatives archive SHA-512 checksum;
- compared it with the verified Phase 1 checksum;
- checked current archive size and modification time;
- verified Phase 1 manifest and QC hashes; and
- confirmed enough Drive space remained.

Outputs were written to temporary build directories first. They were checked
before being promoted to their final locations. If a STOP occurred, the
temporary build was removed and any previously valid cache remained active.

Each final MRI and mask has its own SHA-256 checksum.

## 8. Phase 2 output arrays

Phase 2 produced:

- **230 normalized T1c MRI arrays**
- **230 validated binary CL masks**

MRI arrays are stored as memory-mappable `float32` `.npy` files.

Tumour masks are stored as `uint8` `.npy` files containing only 0 and 1.

The main output locations are:

- `02_PREPROCESSED_MRI/p2.0/`
- `03_TUMOR_MASKS/p2.0/`

The main preprocessing manifest is:

`02_PREPROCESSED_MRI/p2.0/v2_preprocessing_manifest.json`

## 9. Timing problem discovered in Phase 2

Exact acquisition dates were not available.

The canonical BIDS archive contained:

- no `scans.tsv`;
- no `sessions.tsv`; and
- only 14 JSON files with `AcquisitionTime`, which is time-of-day rather than a
  longitudinal date.

Therefore the project cannot claim exact scan dates or an exact
weeks-since-surgery value.

### Canonical approximate intervals

The checksum-verified derivatives archive contains:

- 27 `intervals-days.txt` files, one per patient; and
- 270 `treatment.txt` files, one per MNI session.

The repackaged `raw_needed.tar` contains copies of these files.

Phase 2 verified:

- all 27 interval copies were byte-identical to canonical files;
- all 270 treatment copies were byte-identical to canonical files;
- every interval count equals the number of MNI sessions minus one;
- every interval is a positive integer;
- no treatment file is missing or extra; and
- treatment labels are limited to `CRT`, `TMZ`, `no`, and `unknown`.

A versioned timing cache was then created:

`05_TREATMENT_DATA/p2.0/v2_canonical_timing_cache.json`

Its provenance is explicitly:

`approximate_mni_intervals`

The cache content hash is:

`a184d992bd3432cff5686e26258596c496986f24ce6f747302d4e4b60d3b6a15`

These intervals are valid for relative chronological windows, but they must
never be described as exact acquisition dates.

## 10. Treatment information

Among the 230 selected sessions:

- `CRT`: **95**
- `TMZ`: **90**
- `no`: **15**
- missing/`unknown`: **30**

`unknown` is stored as missingness. It is not treated as a fourth treatment
class.

Three selected session records do not have a dose-map reference. Dose is not a
requirement for the primary MRI-only or MRI+Δt baselines.

Dose-map registration remains unverified for later treatment-aware
experiments.

## 11. Longitudinal-window construction

The approved history rule is:

- at least two earlier valid scans;
- followed by one future target;
- with all earlier eligible scans kept as a variable-length history.

Phase 2 produced:

- **178 longitudinal windows**
- from **25 patients**

### Patients excluded from windows

`sub-24` has no Phase 1 eligible T1c/CL session.

`sub-25` has only two valid preprocessed sessions after its all-zero labels were
excluded. It cannot provide two history scans plus a future target.

Neither patient was silently replaced, and no eligibility rule was weakened.

### Windows per patient

The number of windows varies substantially:

- minimum: 1
- maximum: 14

Patients with many sessions must not dominate training. Every window stores a
patient-equal weight of:

`1 / number_of_windows_for_that_patient`

## 12. Visual quality control

Phase 2 generated a central-slice T1c montage and a one-patient-per-panel T1c/CL
overlay review.

The review found:

- no blank or corrupted preprocessed volume;
- no systematic tumour-mask displacement;
- consistent orientation;
- brains contained within the field of view; and
- expected anatomical and contrast heterogeneity, including surgical cavities.

This was preprocessing QC, not clinical validation.

The reviewed overlay was saved as:

`06_QC_REPORTS/v2_phase2_t1c_cl_overlay.png`

## 13. Frozen patient-level cross-validation

The locked fold scheme is:

`5fold_x3seeds_nested4`

This means:

- five outer patient folds;
- three deterministic repeats; and
- four inner patient folds inside every outer training partition.

The repeat seeds are:

- `3923535749`
- `3212275105`
- `4130349263`

Each outer test fold contains exactly five patients.

Test-window counts are tightly balanced:

- minimum: 35
- maximum: 37

The folds were balanced using eligible-window counts only. No tumour outcome,
model score, or future performance was used.

The frozen fold hash is:

`05292cf2aa19d8e2fb19c3ca2c36a1acd70bf45dde104745d4dac63b41eead27`

Changing any patient assignment changes this hash and invalidates later
comparisons.

## 14. G5 leakage guard

The full Stage 2 G5 leakage guard passed.

It checked:

- no patient appears in both outer training and testing;
- no target crosses train/test partitions;
- every patient appears in exactly one test fold per repeat;
- inner folds contain no outer-test patient;
- every outer-training patient appears once in inner validation;
- normalization uses no cohort/test statistics;
- every window resolves to validated preprocessing outputs;
- no quarantined path enters the manifests;
- seed, fold, parent, and content hashes match; and
- treatment and timing caches match their parents.

Result:

`PASS — no failures`

## 15. Final Phase 2 results

The final Phase 2 QC report records:

- data version: `v2.0`
- preprocessing version: `p2.0`
- selected sequence: `T1c-icor`
- preprocessed sessions: `230`
- longitudinal patients: `25`
- windows: `178`
- timing: `approximate_mni_intervals`
- fold scheme: `5fold_x3seeds_nested4`
- failed guards: none

Sections 10–13 are complete.

All section records were produced from clean Git commit:

`c70dda6b`

No completion record is dirty, and the dashboard reports no failed guards.

## 16. What Phase 2 did not do

Phase 2 did not:

- train a prediction model;
- tune neural-network hyperparameters;
- measure Dice or Hausdorff performance;
- claim treatment awareness;
- claim exact acquisition dates;
- use ONCO as primary ground truth;
- repair or use the 416 blocked optional-modality volumes;
- reuse old TaDiff splits or checkpoints; or
- change the locked target after seeing data.

## 17. Remaining limitations

The main limitations entering Phase 3 are:

- only 25 independent patients produce windows;
- exact scan dates are unavailable;
- there is no verified surgery-date anchor;
- 30 selected treatment labels are missing;
- dose registration is still unverified;
- session/window counts are highly uneven across patients;
- MNI preprocessing remains heterogeneous; and
- optional T1/T2/FLAIR derivatives with non-finite values remain blocked.

These limitations must stay visible in every later result.

## 18. What comes next

Phase 3 is the baseline floor.

It should evaluate:

- `C−1`: persistence — copy the last valid CL mask forward;
- `C0`: MRI-history-only baseline; and
- `C1`: MRI history plus approximate Δt.

All three must use the frozen folds and patient-level inference.

Phase 3 must also:

- report patient-level confidence intervals;
- compare learned models against persistence;
- perform the G4 constant-Δt ablation;
- preserve approximate timing limitations; and
- avoid treatment-awareness claims.

No complex encoder, treatment branch, residual model, or diffusion model should
be introduced before these baseline gates are complete.

## 19. Short version

Phase 2 converted the verified SAILOR inventory into a safe modeling dataset:

- 230 normalized T1c scans;
- 230 valid binary CL masks;
- 178 prediction windows;
- 25 patients;
- approximate but canonical relative timing;
- verified treatment labels with missingness;
- frozen nested patient folds; and
- no detected leakage.

The dataset is now ready for baseline evaluation, but not yet for complex model
claims.
