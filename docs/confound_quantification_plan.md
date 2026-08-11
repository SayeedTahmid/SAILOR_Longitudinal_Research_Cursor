# Treatment–Time Confound Quantification Plan

## Question and claim boundary

The diagnostic question is: **how predictable is observed treatment status from weeks since surgery when the patient, not the session, is the independent unit?** This is measured before interpreting C1–C4.

The analysis does not estimate a causal treatment effect. High predictability shows that status can act as a protocol-time proxy; it does not, by itself, prove that a downstream model is confounded. Low predictability also does not validate treatment-awareness. Guard G2 remains binding regardless of the diagnostic results.

The primary prediction target remains `CL/enhancing_t1wc`.

## Analysis population and variables

1. Build records only from canonical data, honour `missing.tsv`, and join MNI to raw/source sessions only through `raw-mni-link.tsv`.
2. Use one row per eligible patient-session with:
   - patient ID;
   - session ID and within-patient order;
   - weeks since surgery, `W`;
   - treatment status, `T`;
   - explicit status-missingness indicator, `M_T`;
   - timing provenance (`exact`, `approximate`, or unavailable).
3. Treat `T ∈ {CRT, TMZ, no}` as the observed status classes.
4. Treat `unknown` as **missing data**, never as a treatment class:
   - exclude `unknown` rows from mutual-information estimation and treatment-status classification;
   - report missing counts and rates by patient, session order, and `W`;
   - carry `M_T = 1` into downstream C2/C4 so the outcome model can distinguish missing status from an observed status without learning an “unknown treatment” semantics;
   - do not impute an unobserved class label for the diagnostic classifier.
5. The surgery anchor and exactness of `W` must be verified. If exact timing cannot be recovered, label the diagnostic timing approximate and repeat it under the G7 timing-perturbation sensitivity analysis. If surgery anchoring is unavailable, report the analysis as not estimable rather than substituting session index.

## Patient-aware mutual-information estimate

Estimate the dependence between continuous `W` and categorical `T` with the mixed discrete–continuous k-nearest-neighbour estimator of Ross, using pre-registered `k = 3`. If any observed treatment class has too few distinct observations for that estimator, report MI as not estimable at `k = 3`; do not tune `k` to maximize MI.

Repeated sessions must not create a fictitious large sample. Use this patient-aware Monte Carlo procedure:

1. For each of 2,000 deterministic seeded draws, select one eligible observed-status session uniformly within each patient who has at least one such session.
2. Compute `I(T; W)` in nats on that one-row-per-patient draw.
3. Use the mean across draws as the point estimate. Also report normalized MI, `I(T; W) / H(T)`, when `H(T) > 0`.
4. Produce a 95% patient-bootstrap interval with 10,000 cluster replicates. Each replicate samples patients with replacement, then repeats the within-patient one-session draw; all records belonging to a selected patient remain a cluster for missingness summaries.
5. For a null reference, permute complete patient treatment trajectories across patient IDs while preserving each trajectory’s internal order, recompute the same estimator, and report the permutation distribution and two-sided Monte Carlo p-value. This null calibration is descriptive and is not a replacement for P1.

This design gives each patient equal expected influence and incorporates both between-patient sampling uncertainty and within-patient visit selection. Also report the number of contributing patients, classes, eligible sessions, and per-class support so the estimate cannot be read without its small-sample context.

As a sensitivity analysis, repeat the calculation using all eligible sessions with each session weighted by `1 / n_i`, where `n_i` is that patient’s eligible-session count. Agreement supports robustness; disagreement is reported, not resolved by choosing the larger estimate.

## Time-only treatment-status classifier

The classifier is separate from outcome-model control P3. Its sole target is treatment status.

- **Inputs:** `W` only. Do not include patient ID, MRI, session index, treatment history, dose, age, or future information.
- **Model:** multinomial logistic regression over a natural cubic spline expansion of `W`.
- **Outer evaluation:** frozen repeated stratified group CV, 5 folds × 3 pre-registered seeds, grouped strictly by patient. If class support makes a five-fold training or test partition invalid, reduce the number of folds before seeing results and record the change in the fold manifest.
- **Inner selection:** within each outer-training partition only, group CV selects spline degrees of freedom from `{3, 4, 5}` and inverse regularization `C` from `{0.01, 0.1, 1, 10}` using macro log loss. No outer prediction is used for selection.
- **Preprocessing:** knot placement, scaling, and class handling are fitted on the outer-training patients only.
- **Metrics:** primary metric is patient-macro accuracy: calculate accuracy within each outer-test patient, then average patients equally. Also report pooled accuracy for transparency, balanced accuracy, macro-F1, class-wise recall, log loss, and the confusion matrix.
- **Uncertainty:** first aggregate repeated out-of-fold predictions to one score per patient. Obtain 95% CIs by resampling patients 10,000 times. Sessions are never bootstrap units.
- **Comparator:** report majority-class and training-fold class-prior classifiers under the identical outer folds.

The fold assignment is frozen before fitting and reused by C1–C4 and P1–P3 wherever their eligibility sets permit. Any reduced eligibility set is documented; folds are not rearranged to improve balance after results are seen.

## Pre-specified 90% warning rule

If outer-CV patient-macro accuracy for predicting `T` from `W` alone is **greater than 0.90**, the following are mandatory:

1. State prominently that treatment status is more than 90% predictable from protocol time in this cohort.
2. Treat an unqualified claim that a status-conditioned model is “treatment-aware” as untenable unless the complete G2 criteria below pass.
3. Do not use a simple “with treatment” versus “without treatment” ablation as evidence, because removing status also removes a time proxy.
4. Restrict any passing result to an incremental predictive association in this cohort; do not call it a causal treatment effect.
5. Report the point estimate and patient-bootstrap CI. A CI crossing 0.90 signals uncertainty around the warning boundary and is stated explicitly; it is not converted into a pass by choosing one side of the interval.

The 0.90 threshold is a warning and claim constraint, **not conclusive proof of confounding**. Conversely, accuracy at or below 0.90 is not evidence that confounding is absent. G2 is the decision rule for treatment-awareness.

## Exact conditioning and control wiring

All rungs use the same locked target, eligible windows, frozen patient-level outer folds, architecture, optimization budget, encoder regime, and inner-loop selection policy. Only the listed conditioning information changes.

| ID | Exact inputs | Required comparison and interpretation |
|---|---|---|
| **C1** | MRI history + target interval `Δt`; no treatment status and no dose map. | Time-aware imaging baseline. It is the direct baseline for C2, C3, and C4. |
| **C2** | C1 inputs + observed treatment status encoded over `{CRT, TMZ, no}` + explicit `M_T`; no dose map. | Tests whether status adds information beyond C1. C2 must beat C1 and P3 outside the paired patient-bootstrap CIs and must degrade under P1 for G2. |
| **C3** | C1 inputs + spatial CRT dose map; no treatment status. | Tests dose beyond MRI and `Δt`. C3 must beat C1 for a useful increment and must degrade under P2 for any dose-based treatment claim. A static dose map is described as a spatial prior modulated by time, not as time-varying treatment. |
| **C4** | C1 inputs + observed status + `M_T` + spatial dose map. | Tests complementarity. Compare C4 with C1, C2, and C3. Interpret a gain only after the status evidence from C2/P1/P3 and dose evidence from C3/P2 have passed; C4 cannot rescue a failed C2 or C3 control. |
| **P1** | A C2 replica in which complete treatment records are reassigned across patients while each donor trajectory’s session order is preserved; MRI, `Δt`, target, architecture, and missingness policy stay fixed. | Generate donor mappings separately inside each outer-training and outer-test partition so no information crosses folds. Refit the model for each pre-registered shuffle. C2 must outperform its P1 distribution; failure means the status branch has not demonstrated patient-specific treatment information. |
| **P2** | A C3 replica in which each patient receives another patient’s static dose map in MNI space; MRI, `Δt`, target, and architecture stay fixed. | Reassign maps separately within outer-training and outer-test partitions and refit. C3 must degrade under P2. Before running, Stage 1 must verify dose coverage, units, resolution, MNI registration, and CRT/TMZ representation. |
| **P3** | Outcome model conditioned on weeks since surgery only; no MRI, treatment status, dose, or patient ID. | This is not the diagnostic status classifier. It predicts the locked future outcome under the same outer folds and evaluation protocol. C2 must beat P3 outside paired patient-bootstrap CIs; otherwise status is not distinguished from protocol timing. |

P1/P2 mappings and seeds are generated before model results, persisted in manifests, and reused across paired comparisons. A control changes only its designated information source. Every controlled model is retrained; merely shuffling an input at inference time is not the primary control.

## Patient-level inference

For every C/P comparison:

1. Retain outer-fold predictions only.
2. Compute each metric per patient, aggregating that patient’s eligible target windows without treating windows as independent.
3. For repeated outer CV, average repeated out-of-fold results to one value per patient and model.
4. Compute paired differences using the same patient set.
5. Draw 10,000 patient-bootstrap samples with replacement, using the same sampled patient indices for both members of each pair.
6. Report the paired mean/median difference and percentile 95% CI. “Outside the confidence intervals” means the paired 95% CI for the required improvement excludes zero in the favorable direction.
7. Report exact contributing patient/window counts and per-patient results. Do not bootstrap sessions or pool repeated predictions as independent observations.

Permutation-control distributions are summarized alongside these patient-bootstrap CIs. Multiplicity across the pre-specified core comparisons is reported and controlled with Holm adjustment; unadjusted and adjusted results are both shown.

## Binding interpretation matrix

- **G2 status claim passes only if all hold:** C2 beats C1 outside the paired CI; C2 beats P3 outside the paired CI; and C2 degrades under P1.
- **Dose claim additionally requires:** C3 degrades under P2. A useful dose increment is supported by C3 beating C1 outside the paired CI.
- **If C2 fails any required comparison:** state that the model is not treatment-aware through status in this cohort, even if mean Dice rises.
- **If C3 fails P2:** do not call dose conditioning treatment-aware; it may be encoding anatomy, tumour location, or patient identity.
- **If C4 improves while a component control fails:** report prediction improvement, if statistically supported, but do not attribute that improvement to the failed treatment component.
- **If time-only status accuracy exceeds 0.90:** apply the warning language and claim restriction above; still evaluate G2.
- **If time-only status accuracy does not exceed 0.90:** do not relax G2.

No threshold, mutual-information estimate, architecture ablation, or mean metric overrides G2.
