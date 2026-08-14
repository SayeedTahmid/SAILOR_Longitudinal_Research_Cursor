# SAILOR Phase 3 — What We Are About To Do

This note explains Phase 3 in plain language: the GPU, the exact model,
whether it is pretrained, how long it trains, and what claims are allowed.

Phase 3 is the **baseline floor**. It asks a simple question first:

> If we just copy the last tumour mask forward, how good is that — and can a
> small MRI model beat it?

It does **not** train Triad, BrainFound, MedNeXt, a longitudinal encoder,
treatment models, dose models, or diffusion.

## 1. Preferred GPU

Use a **Colab GPU** runtime. CPU can score persistence, but C0 and C1 need
PyTorch and a GPU.

Preferred order:

1. **L4** — best practical Colab Pro choice for this baseline
2. **T4** — acceptable and common; this is the minimum GPU we should plan around
3. **A100** — fine if Colab gives it, but we must **not** assume it
4. Local **RTX 3060 12 GB** — allowed later, but we must not assume it can match Colab

Do not start section 14/15 execute on CPU. Persistence C−1 would still run,
but the U-Net training would be extremely slow and is likely to disconnect.

VRAM is **UNMEASURED** until this run profiles it. We do not guess memory.

Colab idle disconnects will stop training. Keep the tab awake, and prefer a
runtime that can stay connected for several hours.

## 2. What model we are using

There are two kinds of “model” in this phase.

### C−1 — Persistence (not a neural net)

Copy the last valid `CL / enhancing_t1wc` mask to the future session.

- no MRI
- no Δt
- no training
- no pretrained weights

This is the G3 bar. Every learned model must be compared with it using
patient-level confidence intervals. A higher mean Dice is not enough.

### C0 and C1 — Compact 3D U-Net, version `b3.0`

The learned baseline is a **small 3D U-Net written for this project**.

Exact locked details:

- name / version: `BaselineUNet3D`, `model_version = b3.0`
- task: predict the future binary CL enhancing mask
- input MRI: normalized `T1c-icor`
- history: the **most recent two** eligible scans
- input shape: 3 channels × 64³ training patches
  - channel 0: T1c at `t-2`
  - channel 1: T1c at `t-1`
  - channel 2: time channel
- output: one full-resolution mask, thresholded at 0.5
- depth: 3 encoder / decoder stages
- base width: 8 channels, then 16, then 32
- normalization: InstanceNorm
- loss: Dice + binary cross-entropy
- optimizer: Adam

C0 and C1 use the **same** U-Net. Only the time channel changes:

| Rung | Time channel | Question |
|---|---|---|
| **C0** | all zeros | Do the last two MRIs predict change? |
| **C1** | `target_delta_days / 365` | Does approximate Δt add anything? |
| **G4** | training-fold median Δt, retrained | Is Δt actually used, or decorative? |

This U-Net is **not** a foundation model and **not** nnU-Net.

## 3. Is it pretrained?

**No.** C0, C1, and the G4 control are trained **from scratch** on SAILOR
windows only.

They do **not** load:

- Triad
- BrainFound
- MedNeXt
- old TaDiff checkpoints
- any ImageNet, medical, or previous SAILOR weights

Random initialization is intentional. Phase 3 is a simple floor. Pretrained
encoders are a later phase, after this floor exists.

C−1 has no weights at all.

## 4. How many epochs?

Locked budget:

| Setting | Value | Used for |
|---|---|---|
| Inner-loop epochs | **5** | choose learning rate inside training patients |
| Outer-loop epochs | **20** | final model for each outer fold |
| Learning rates tried | `1e-4` and `3e-4` | inner selection only |
| Batch size | 2 patches | training |
| Patch size | 64 × 64 × 64 | training and sliding-window inference |
| Outer folds | 5 | frozen from Phase 2 |
| Repeats | 3 | frozen from Phase 2 |
| Inner folds | 4 | frozen from Phase 2 |

What this means in practice:

- For each outer fold, the code tries both learning rates on the 4 inner
  patient splits using 5 epochs each.
- It picks the better learning rate **without looking at test patients**.
- It then retrains on all outer-training patients for **20 epochs**.
- It scores only the outer-test patients.

Section 14 does this for **C0**. Section 15 does it again for **C1** and for
the retrained constant-Δt control.

That is a long GPU job. It is supposed to be. We are not reducing epochs after
seeing results.

## 5. What data it uses

Frozen Phase-2 artefacts only:

- 230 normalized T1c volumes
- 230 binary CL masks
- 178 prediction windows
- 25 patients
- frozen `5fold_x3seeds_nested4` splits
- approximate MNI intervals

It will not:

- rebuild windows
- change the CL target
- use ONCO as ground truth
- use dose maps
- use treatment labels
- use `SAILOR_READY_v2.0` as the write location

Results are written under:

`/content/drive/MyDrive/SAILOR_Longitudinal_Research_Cursor/07_BASELINE_RESULTS/p3.0/`

## 6. How we will judge success

Primary metric: **patient-macro Dice**.

That means:

1. score every window;
2. average windows inside each patient;
3. give every patient equal weight;
4. bootstrap **patients**, 10,000 times;
5. never treat 178 windows as 178 independent samples.

A model “beats” another only if the paired 95% confidence interval for the
Dice difference is entirely above zero.

If C0 does not beat copying the last mask, that is a real finding. We report
it. We do not add a bigger network to rescue it in this phase.

If C1 does not beat the constant-Δt control, Δt is decorative. That is also a
real finding.

## 7. Power warning already in the plan

With 25 patients, small Dice gains are hard to trust:

- if patient scatter is 0.05, we can detect about **0.028 Dice**
- if patient scatter is 0.10, about **0.056**
- if patient scatter is 0.15, about **0.084**
- if patient scatter is 0.20, about **0.112**

After C−1 runs, the empirical MDE replaces these assumed numbers. A hoped-for
gain below that MDE is not an improvement.

## 8. What this phase will not claim

Not approved in Phase 3:

- treatment awareness
- dose-aware modelling
- exact time since surgery
- causal treatment effects
- foundation-model performance
- that the U-Net is the final SAILOR architecture

Approved to evaluate:

- persistence
- MRI-history-only prediction
- MRI plus approximate Δt
- whether Δt is actually used

## 9. The two notebook steps

**Section 14 — already dry-run complete**

Execute on GPU:

```python
section_14 = run_phase3_section(14, execute=True)
```

This runs C−1 and C0.

**Section 15 — only after 14 finishes**

```python
section_15 = run_phase3_section(15, execute=True)
```

This runs C1, the G4 constant-Δt retrain, and a ±7 day timing sensitivity.

## 11. Checkpointing and resume

C0, C1, and G4 write to Google Drive after **every epoch**:

`CHECKPOINTS/p3.0/<rung>/repeatR_outerF/`

That path is under `DATASET_ROOT`, so it survives a Colab runtime reset.
Each run keeps atomic `latest.pt`, `best.pt`, and `final.pt`, plus
`metrics.csv`, `training_log.jsonl`, `training_summary.json`, and a small
fixed set of T1c / ground-truth CL / prediction / overlay montages with
loss, Dice, and learning-rate curves. Inner-loop LR search has its own
subdirectory. After a fold finishes, `fold_complete.json` and
`fold_summary.json` are written so a disconnect does not rerun completed
folds.

Each checkpoint stores model and optimizer state, epoch, learning rate,
best monitor metric/epoch, experiment identity, `model_version`,
`DATA_VERSION`, `PREPROCESSING_VERSION`, fold/repeat/seed, and RNG state.

Resume rules:

- a valid `latest.pt` continues at the next unfinished epoch
- a valid `fold_complete.json` skips that fold
- if `latest.pt` is missing and `final.pt` already finished the locked
  budget, training is not repeated
- a mismatched version, fold, seed, LR, or model identity **stops**
- a corrupt file or `best.pt` without `latest.pt` **stops**
- NaN/Inf, exploding loss, and GPU OOM **stop** after writing `failures.jsonl`
- outer-test Dice is never used to pick LR, epoch, or checkpoint
- the scientific weights are always `final.pt` after the locked 20 epochs
- an interruption does not change epochs, LR, folds, seeds, data, or the model

Metric roles stay distinct: **TRAINING** and **INNER_VALIDATION** are for
monitoring and learning-rate selection; **OUTER_TEST** is only for final
evaluation. Patient-macro Dice remains the primary scientific metric.

## 12. Short version

Phase 3 trains a **small 3D U-Net from scratch**, not a pretrained foundation
model. Persistence copies the last mask and needs no GPU. The U-Net needs a
Colab **T4 or L4** GPU. Inner selection uses **5 epochs**; the real fold models
use **20 epochs**. The question is not “can a fancy model segment tumours?”
The question is “does MRI, and then approximate time, beat copying the last
scan in this 25-patient cohort?”
