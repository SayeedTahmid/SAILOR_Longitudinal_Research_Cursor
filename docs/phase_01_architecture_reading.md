# Phase 1 Architecture Reading

## Scope and governing lock

This reading compares the uploaded “Treatment-Aware Longitudinal Neuro-Oncology Framework with Residual Diffusion” diagram against the verified EBRAINS `sailor` v1 descriptor facts recorded in `Research_Operating_System.md`. Diagram labels are proposals, not evidence that data, checkpoints, dimensions, runtimes, or performance are available.

The primary target is locked to:

- `PRIMARY_TARGET_MASK = "CL"`
- `PRIMARY_TARGET_COMPONENT = "enhancing_t1wc"`

`ONCO` is inventory-only until the primary result is final. The `CL` t2wflair component is a pre-specified sensitivity target. Neither may define the primary cohort or influence primary preprocessing, model selection, or hyperparameters.

Availability states below mean:

- **Verified by descriptor:** the descriptor says the data type exists, but actual usable files and counts still require Stage 1 inspection.
- **Partly verified:** only some required fields or conditions are descriptor-confirmed.
- **Unverified:** the descriptor does not establish the claim.
- **Conflict:** the diagram contradicts a binding project rule or verified source statement.

## Blocks 1–10

| Block | Diagram function | Required SAILOR inputs | Descriptor availability | Conflicts, corrections, and unverified claims |
|---|---|---|---|---|
| **1. Input: one patient** | Use 3+ longitudinal MRI time points to predict a future time point. | Patient identity; ordered sessions; selected structural MRI sequence(s), principally t1wc for the locked target; session correspondence; acquisition timing; current and future `CL/enhancing_t1wc` masks. | **Partly verified.** There are 27 patients, with diagnostic/immediate postoperative imaging and 3–19 time points during CRT/TMZ. Structural sequences and `CL` masks are descriptor-confirmed. Actual per-patient usable histories after `missing.tsv`, G1, and modality filtering are unverified. | “3+ time points” is a proposed eligibility rule, not descriptor-verified. Sessions are not independent. MNI and raw `ses-XX` labels cannot be aligned directly; use `raw-mni-link.tsv`. All primary windows must use `CL/enhancing_t1wc`. |
| **2. 3D spatial feature extractor** | Encode each MRI into hierarchical features. | Preprocessed MRI volumes; a fixed modality policy; encoder code and downloadable weights; licence; expected input shape/intensity; preprocessing and weights hashes. | **Partly verified.** MNI derivatives are skull-stripped, nominally 1 mm isotropic, and include structural MRI. Actual sequence availability, shapes, dtype/range, and compatibility with any encoder require inspection. | The pictured input size **`192^3` is unverified** and must not be adopted until measured. MNI processing is heterogeneous and the stated 0–255 uint8 representation is internally inconsistent; G10 must measure it. Correct model facts: **Triad** is a 3D-MRI foundation model using autoencoder/SimMIM-style pretraining on about 131K volumes, with open weights; arXiv 2502.14064 and a Medical Image Analysis journal version, **not NeurIPS 2024**. **BrainFound** extends DINOv2 to brain MRI but is slice-wise 2D rather than native 3D; arXiv 2510.23415, and weights must be confirmed downloadable before selection. **MedNeXt** is a MICCAI 2023 segmentation architecture, **not ECCV 2024** and not a large-scale pretrained foundation model; it implies from-scratch or task-specific training. **MiM** and **BrainNext** remain **[VERIFY]** for existence, venue, architecture, licence, and public weights; do not repeat the diagram’s venue claims. Default regime is frozen/cached features, with from-scratch 3D CNN and Swin baselines. |
| **3. Longitudinal temporal encoder** | Model ordered feature sequences with positional/time embeddings. | Per-session spatial features; patient-level order; inter-exam `Δt`; missing-session handling; sequence masks. | **Partly verified.** Number of days between time points exists, but the descriptor warns MNI-derived intervals may be inaccurate. Exact dates may be recoverable only from source/raw or present metadata and must be audited. | A pure ordinal/positional embedding cannot substitute for irregular `Δt`. G7 provenance is binding; approximate intervals must be labelled and sensitivity-tested. Transformer, Hyena, Longformer, and Performer suitability and memory use are architecture proposals, not verified SAILOR facts. |
| **4. Treatment data and encoder** | Encode treatment type, timing, dose/fractionation, duration, and clinical variables. | Per-time-point treatment status; explicit missingness indicator; weeks since surgery/target `Δt`; spatial dose maps and dose provenance; optional age, sex, RANO and other pre-specified covariates. | **Partly verified.** Status values `CRT`, `TMZ`, `no`, `unknown`; radiation dose distribution maps; age, sex, overall survival, days between time points, and RANO are descriptor-confirmed. Actual completeness, dose-map space/resolution/registration, surgery timing, and usable covariate timing require inspection. | `unknown` is **missingness**, not a fourth treatment class. The descriptor does not verify the diagram’s generic surgery/combo labels, fractionation, treatment duration, or all “other clinical covariates.” Status is likely highly predictable from time; status-only treatment-awareness is constrained by G2. Dose maps are static CRT-derived spatial information unless inspection shows otherwise. TabTransformer, FT-Transformer, and MLP-Mixer are candidate choices only. |
| **5. Cross-attention fusion** | Use longitudinal history as query and treatment representation as key/value. | `Z_history`, `Z_treat`, attention masks, fixed dimensions, and missing-treatment behavior. | **Not a descriptor availability question.** Inputs can exist only after Blocks 3–4 pass their audits and contracts. | Cross-attention is not automatically superior. It must beat concatenation under the same patient-level CV or be dropped. Removing the treatment branch alone is not a valid G2 test because it also removes a time-correlated signal. |
| **6. Conditioned representation** | Combine fused history with target-time embedding. | `Z_fused`; target `Δt`/weeks-since-surgery with provenance; shape contracts. | **Partly verified.** Timing fields exist, but exactness in MNI is not assured. | The interface contract requires current state and `Δt` in the residual head and `Z_cond` in diffusion. A decorative time embedding fails G4 if replacing `Δt` with a constant does not change results. |
| **7. Residual conditional diffusion model** | Predict change from current MRI/mask, conditioned on history, treatment, and `Δt`. | Current t1wc (or pre-specified MRI channels); current `CL/enhancing_t1wc` mask; future locked mask; aligned longitudinal volumes; `Z_cond`; `Δt`; sufficient GPU memory. | **Partly verified.** Relevant MRI and `CL` annotations are descriptor-confirmed in the MNI derivatives, but actual paired availability and array validity are unverified. Hardware feasibility is unmeasured. | The diagram mixes future-MRI synthesis and tumour-mask prediction. The primary supervised endpoint remains future `CL/enhancing_t1wc`; synthesized MRI can only be an explicitly designated auxiliary endpoint. DDPM, rectified flow, consistency models, and latent diffusion are alternatives, not verified improvements. Conditional diffusion proceeds only if it beats simpler rungs and fits measured VRAM. |
| **8. Reconstruct future** | Add predicted residual to the current MRI and mask. | Spatially aligned current and target volumes/masks; residual definition; valid output constraints. | **Partly verified.** MNI-space derivatives support common-space operations in principle, but registration quality and heterogeneous preprocessing require QC. | Directly adding a residual to a binary mask can create invalid values; the implementation must define whether residuals operate on logits/probabilities and how masks are thresholded. “Residual diffusion is easier because it improves learning efficiency and accuracy” is an unverified outcome claim and requires the A3/A4 ablations. |
| **9. Supervision (ground truth)** | Supervise future MRI and tumour mask at `t_i + Δt`. | Future structural MRI; future `CL/enhancing_t1wc`; valid non-degenerate target; patient-safe window assignment. | **Partly verified.** Longitudinal MRI and `CL` annotations exist in the descriptor, with masks only in MNI derivatives. Exact counts and component filenames require Stage 1 inspection. | Ground truth for the primary claim is specifically `CL/enhancing_t1wc`, not generic “tumour mask” and not ONCO. Zero/near-empty masks require G1 handling. Any future-MRI target is secondary unless separately pre-specified. |
| **10. Evaluation** | Report segmentation, volume/growth, image-similarity, calibration, uncertainty, and interval metrics. | Patient-level predictions and targets; per-patient aggregation; exact/qualified `Δt`; predictive distributions; frozen outer-CV outputs. | **Partly verified.** Masks and timing are descriptor-confirmed in principle; the descriptor does not verify that every proposed metric is estimable after exclusions. | Dice, Hausdorff, and volume error are relevant to the locked mask. Growth velocity requires trustworthy `Δt`, so G7 applies. SSIM/PSNR concern an auxiliary MRI-synthesis task, not the primary mask endpoint. ECE, AUROC, and prediction-interval coverage require a pre-specified uncertainty target and construction. All metrics need patient-level outer CV and patient-bootstrap CIs; session-level CIs are invalid. |

## End-to-end pipeline reading

The pictured 11 steps are: input MRI history → foundation encoder → temporal encoder → treatment encoder → cross-attention → conditioned representation → residual diffusion → reconstruction → ground-truth supervision → evaluation → baseline comparison. This ordering is broadly compatible with the interface contracts, but it is not the execution order for scientific claims. The conditioning ladder C−1 through C4 must establish whether each information source adds value before the architectural ladder adds temporal, attention, residual, diffusion, and foundation-model complexity. Each stage must run from persisted upstream artefacts on a fresh runtime and write target, provenance, fold, and completion metadata.

The diagram’s architecture cannot bypass the gates: data/provenance first; persistence and MRI/time baselines next; G2 treatment controls before a treatment-awareness claim; then architectural ablations under the same patient-level CV. No component is retained merely because it appears in the end-to-end drawing.

## Baselines

| Diagram item | Required input/availability | Reading |
|---|---|---|
| Persistence (copy last MRI and mask) | Current and future aligned observations; locked masks. Descriptor-confirmed in principle, exact pairs unverified. | Mandatory C−1 floor. Report patient-bootstrap CIs and paired comparisons under the outer CV scheme (G3). For the primary result, mask persistence is the key baseline. |
| TaDiff reproduction | Quarantined prior TaDiff artefacts and/or a clean reproduction from canonical inputs. | Prior artefacts are comparison-only, read-only, and may never define splits or manifests. A new reproduction must use canonical v2 inputs and frozen patient-level folds. |
| 3D-GlioPREDICT | Public specification, code/weights, compatible inputs and target. | **[VERIFY]** existence, exact method, licence, checkpoint availability, target compatibility, and fair preprocessing before inclusion. |
| Other state-of-the-art methods | Reproducible implementations compatible with 3D longitudinal data and the locked target. | **[VERIFY]** individually. “SOTA” is not a baseline definition. Include the required from-scratch 3D CNN and Swin baselines for encoder comparison. |

## Ablations

The diagram lists “without treatment,” “without cross-attention,” “without residual,” “without longitudinal encoder,” and “without foundation pretraining.” These are useful only within the locked ladders:

- “Without treatment” alone is **insufficient and conflicts with G2** if treated as proof of treatment-awareness. Use C1 versus C2 plus P1 and P3; use C1 versus C3 plus P2 for dose.
- Cross-attention must be compared with concatenation (A7), not only deleted.
- Residual prediction must beat direct prediction (A3).
- Longitudinal encoding must beat the simpler fixed conditioning rung (A1), with `Δt` separately ablated to a constant under G4.
- Foundation encoding must be compared with from-scratch CNN/Swin (A5), and frozen versus fine-tuned regimes compared only with nested selection (A6).
- Conditional diffusion must beat the non-diffusion residual formulation (A4) and fit measured hardware.

All comparisons use the same frozen patient-level outer folds, inner-loop selection only, per-patient outputs, paired tests, and patient-bootstrap confidence intervals. Negative results remain reportable.

## Data and training assumptions

| Diagram claim | Status |
|---|---|
| SAILOR/EBRAINS dataset | **Verified.** |
| 27 patients | **Verified.** |
| Approximately 270 sessions | **Unverified.** The descriptor gives 3–19 treatment-era time points plus diagnostic/immediate postoperative scans, but actual unique usable sessions must be counted after correspondence, exclusions, modalities, and G1. |
| Patient-wise split | **Directionally correct but incomplete.** A single split is inadequate at n=27. Use repeated patient-level outer CV (or leave-one-patient-out), nested inner selection, and frozen fold manifests. |
| Mixed-precision training | Implementation option; not a descriptor fact. Validate numerical stability. |
| Gradient checkpointing | Implementation option; not evidence of feasibility. |

Masks required for the primary task exist only in `derivatives/mni2009c-n-s`. `missing.tsv` is binding. MNI/raw sessions must be joined only by `raw-mni-link.tsv`. Preprocessing heterogeneity, mask degeneracy, actual shapes/spacings/dtypes, dose-map coverage, and accurate timing all remain Stage 1 measurements.

## Hardware estimate

The project targets Colab Pro hardware (T4/L4/A100 when available) and a local RTX 3060 12 GB, so the diagram’s RTX 3060 reference is consistent with the intended test hardware. Its batch size 1–2, feature-extraction time of 0.5–2 hours, experiment time of 1–2 hours, and “4–12 hours per experiment typical” are **unverified**. VRAM, RAM, disk, wall time, and safe batch size must be empirically profiled on real inputs and shown as `UNMEASURED` until then. An A100 must never be assumed.

## Expected outcome

The diagram’s claims of more accurate future tumour segmentation/volume, better growth prediction, calibrated uncertainty, and clinical interpretability are hypotheses, not expected facts. They become supportable only if:

1. the model beats persistence and simpler baselines with patient-level uncertainty;
2. claimed treatment information passes G2 rather than merely improving a branch-deletion ablation;
3. architectural components beat their paired ablations;
4. uncertainty and interpretability are operationally defined and evaluated; and
5. negative, null, and cross-implementation disagreement results are reported without target or model switching.

At 27 patients, wide confidence intervals and negative results are plausible. No accuracy, novelty, treatment-awareness, or clinical-utility claim is verified by the diagram.
