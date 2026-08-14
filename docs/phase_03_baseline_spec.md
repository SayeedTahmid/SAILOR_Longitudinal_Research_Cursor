# Phase 3 Baseline Floor — Specification

Phase 3 evaluates the locked conditioning rungs **C−1**, **C0**, and **C1**
on the frozen Phase-2 windows and nested patient-level folds. It does not
introduce a foundation encoder, treatment status, dose maps, residual heads,
or diffusion.

Writes go only under the authoritative project `DATASET_ROOT`. The
`SAILOR_READY_v2.0` package remains a read-only distribution copy.

## Locked choices

- Target: `CL / enhancing_t1wc`
- Input MRI: normalized `T1c-icor`
- Windows and folds: Phase-2 `p2.0` manifests, never regenerated
- Model version: `b3.0`
- History used by learned baselines: the **most recent two** eligible scans
- Shared learned architecture: compact 3D U-Net, 3 input channels
- Channel 0–1: T1c at history `t-2` and `t-1`
- Channel 2: Δt conditioning
  - **C0:** zeros
  - **C1:** `target_delta_days / 365`
  - **G4:** outer-training median `target_delta_days / 365`, retrained
- Persistence **C−1:** copy the last history CL mask forward; no MRI, no Δt
- Primary metric: patient-macro Dice
- Secondary metrics: relative volume error; 95% Hausdorff when computable
- Uncertainty: 10,000 patient-cluster bootstrap replicates; sessions are never
  bootstrap units
- Inner selection: learning rate from `{1e-4, 3e-4}` using the four inner
  patient folds; outer test patients are never used
- G7 sensitivity: add `{−7, +7}` days to C1 test Δt at inference only
- Timing provenance remains `approximate_mni_intervals`

Using only the last two scans is an explicit Phase-3 limit. Variable-length
history belongs to the later temporal encoder, not this baseline floor.

## What "beats" means

A rung beats another only if the paired patient-bootstrap 95% CI for the Dice
difference excludes zero in the favourable direction. A higher mean Dice is
not a result. Holm-adjusted and unadjusted comparisons are both stored.

If C0 or C1 is statistically indistinguishable from persistence, that is the
G3 finding. The run still completes.

If C1 does not beat the retrained constant-Δt control, temporal conditioning
is decorative. That is the G4 finding. The run still completes.

## Minimum detectable effect

Before training, Phase 3 records the paired Dice MDE at n = 25 patients for
illustrative within-patient SDs `{0.05, 0.10, 0.15, 0.20}`. After C−1, it
records the empirical MDE using the observed patient-level persistence SD.
If a hoped-for gain is below that MDE, the report says so before interpreting
C0 or C1 as an improvement.

## Scientific boundary

Approved to evaluate:

- persistence
- MRI-history-only prediction of the future CL mask
- MRI plus approximate Δt
- whether Δt is decorative under G4

Not approved:

- treatment-aware claims
- dose-aware claims
- exact time-from-surgery claims
- causal effects
- encoder, fusion, residual, or diffusion claims

## Outputs

All results are written under `07_BASELINE_RESULTS/p3.0/`. The quarantined
filename `persistence_baseline.json` is never reused.

## Checkpointing, resume, and monitoring

C0, C1, and G4 (`C1_constant`) are resumable. Checkpoints and logs are written
under `DATASET_ROOT/CHECKPOINTS/p3.0/<rung>/repeatR_outerF/` after every
completed epoch. A Colab disconnect must not change epochs, learning rates,
folds, seeds, data, or the model.

Each run keeps atomic `latest.pt`, `best.pt`, and `final.pt`. Scientific
evaluation always uses `final.pt` after the locked epoch budget. `best.pt` is
a monitor only.

On restart, `latest.pt` is loaded if present and its identity is verified
against the locked experiment (rung, fold, repeat, seed, LR, epoch budget,
model/data/preprocessing versions). Training continues at the next unfinished
epoch. A completed fold writes `fold_complete.json` and is skipped. A
corrupt or mismatched checkpoint **stops**; it is never loaded with a warning.

Training/inner-validation metrics are for monitoring and LR selection.
Outer-test metrics are stored only after a fold finishes and are never used
for learning-rate selection, epoch selection, checkpoint selection, or early
stopping. The primary scientific metric remains patient-macro Dice.
