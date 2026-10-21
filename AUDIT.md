# WaferGuard Audit

## Decision

**NO-GO for production.** The current model is a research prototype. Test data has already been inspected, so the reported test metrics are exploratory and must not be described as an untouched final estimate.

## 1. Dataset and natural prevalence

The current notebook's latest reported split is:

| Partition | Records |
|---|---:|
| Training | 1,096 |
| Validation | 235 |
| Test | 236 |
| **Total** | **1,567** |

The latest run reported 9 defects in the test partition (3.81% prevalence). The dataset is imbalanced; metrics must be interpreted against this natural prevalence, not a balanced test set.

## 2. Required audit question: What were the three silent bugs, and how were they fixed?

### Bug 1 — Data leakage and temporal causality

The audit target is to prevent information from validation/test rows or later timestamps influencing training. The repaired notebook sorts records chronologically, creates separate train/validation/test partitions, and fits the median imputer and scaler on training rows only. Validation data is used for model selection and threshold selection; the test set is not used to select the checkpoint or threshold.

**Evidence requirement:** A measured before/after leakage comparison has not been supplied in the current evidence. Do not claim a numeric improvement until the original buggy pipeline and repaired pipeline have both been run under a documented, comparable protocol.

### Bug 2 — Reconstruction-loss broadcasting

The reconstruction tensor and input target must both have shape `[batch, features]`. The repaired model asserts that reconstruction and input shapes match before computing MSE, and that logits and labels both have shape `[batch]`. This prevents a silent broadcast such as `[B, 1, F]` against `[B, F]`, which can produce an unintended `[B, B, F]` result.

Example shape evidence:
- Intended reconstruction and target: `[B, F]` and `[B, F]`.
- Buggy reconstruction and target: `[B, 1, F]` and `[B, F]`.
- Broadcasted result: `[B, B, F]`.

### Bug 3 — Legacy PS-L invariant blindness

Legacy semantics must not be treated as ordinary continuous measurements. `sensor_0 == 0.0` is a sentinel; `sensor_1` codes 100, 200, and 300 are categorical codes. The repaired preprocessing excludes these raw columns from continuous imputation/scaling and appends explicit indicator features.

The supplied review reports 146 wafers with `sensor_0 == 0.0`, none defective. This count should be confirmed by the current dataset before treating it as a reproduced measurement. The invariant is documented as a data property, not a production guarantee.

## 3. Required audit question: What are the true evaluation metrics at natural prevalence?

All metrics below are on the chronological test partition (last 15%: 236 records, 9 defects, 3.81% prevalence). The test set has already been inspected, so these numbers are **exploratory**, not an untouched final estimate. With nine defects, one extra catch changes recall by 11 points.

### 3a. Notebook run (`notebooks/wafer_guard_fixed.ipynb`, outputs saved in the notebook)

Model: shared `WaferAutoencoder` (hidden 128, latent 32, 592 features), seed 1337, best checkpoint at epoch 8 by validation average precision. Threshold 0.00067027786 from validation scores for a nominal 10% alert rate.

| Metric | Validation (235 rows, 17 defects) | Test (236 rows, 9 defects) |
|---|---:|---:|
| ROC-AUC | 0.510 | 0.679 (95% CI 0.48–0.85) |
| Average precision | 0.078 | 0.092 (95% CI 0.03–0.27) |
| Alert rate | 10.21% | 6.36% |
| Confusion matrix (TN / FP / FN / TP) | 195 / 23 / 16 / 1 | 214 / 13 / 7 / 2 |
| Precision / recall / F1 | 0.042 / 0.059 / 0.049 | 0.133 / 0.222 / 0.167 |

Lift over prevalence on test is 3.5x, but it rests on two true positives, and the lower bootstrap bound of average precision (0.033) is below the natural prevalence (0.038). The notebook's reference gate is therefore **NOT MET**.

### 3b. Currently deployed model (`artifacts/wafer_model.pt`, 472 features, threshold 0.001003)

Evaluated with `scripts/evaluation/evaluate_model.py` on the same test partition:

| Metric | Test |
|---|---:|
| ROC-AUC | 0.747 |
| Average precision | 0.096 |
| TN / FP / FN / TP | 215 / 12 / 8 / 1 |
| Precision / recall / F1 | 0.077 / 0.111 / 0.091 |

### 3c. Earlier supplied run (historical, not reproducible)

An earlier run reported test ROC-AUC 0.631 and average precision 0.060 with TP=8, FP=196, FN=1 (86% of the test set flagged). It came from a previous architecture and a near-zero F1-maximising threshold. It is kept only to document why the threshold policy was changed, and must not be cited as current performance.

## 4. Threshold and overfitting controls

The notebook revision selects the threshold from validation scores for a nominal 10% alert rate, reports the actual validation/test alert rates, and selects the model checkpoint by validation average precision with patience-based early stopping. The test partition is not used for checkpoint selection or threshold selection. If tied scores cause the alert rate to exceed the target, report the actual rate rather than hiding the difference.

## 5. Findings on signal and drift

These were produced by read-only analyses of the training and validation rows (and, where stated, test rows). They explain why every model scores near chance.

**Chronological cross-validation** (4 expanding folds over the first 1,331 rows; held-out blocks hold 6, 9, 2 and 15 defects; prevalence about 7%):

| Model | Mean average precision | Mean ROC-AUC |
|---|---:|---:|
| Logistic regression (C=0.01) | 0.106 | 0.545 |
| Gradient boosting | 0.122 | 0.513 |
| Autoencoder as in the notebook | 0.091 | 0.540 |
| Autoencoder, 8 epochs | 0.089 | 0.513 |
| Autoencoder, small and regularised (hidden 32, latent 8) | 0.111 | 0.625 |
| Same, with input noise | 0.099 | 0.566 |

Fold-to-fold swings are larger than the differences between models, and every model sits within a few points of chance.

**Per-sensor diagnostic** (456 sensors with enough data):

- Among the 20 sensors most associated with defects in training, only 45% point the same direction in validation (chance is 50%). Across all sensors the correlation between the training and validation associations is 0.008 (0.08 against the test period).
- The defect rate drifts over time: 12.7%, 15.3%, 8.9%, 4.5%, 1.3%, 5.7%, 1.3%, 6.4%, 8.3%, 1.9% across ten equal time blocks.
- All 146 legacy wafers (`sensor_0 == 0`) are non-defective, and the model already has an indicator for them; this can rule wafers out but cannot find defects.
- Missing-cell rate is 4.3% for defects and 4.6% for normal wafers, so missingness carries no signal.

**Conclusion:** with 104 defects and sensor-defect relationships that do not persist over time, further tuning is expected to fit noise. Improvement would need more or better data, a label review, or a drift-aware approach, not more model work on this dataset. These analyses were run in scratch scripts and are not committed.

## 6. Reproducibility and scope

- Model: PyTorch `WaferAutoencoder` (`src/models/wafer_model.py`), shared by the notebook and the app.
- Split: one chronological 70/15/15 definition in `src/data/splits.py`, used by the notebook, the Streamlit app, and the evaluation scripts (test = last 15%, 236 rows, 9 defects).
- Artifacts: the notebook saves `artifacts/wafer_model.pt` and `artifacts/preprocessor.joblib` in the format the app loads, backs up any existing copies, and ends with a parity check that reloads them through the app's inference code.
- Tests: `python -m pytest` covers the split, legacy-sensor feature rules, model shapes, and train/serve parity on synthetic data. It does not measure model quality.
- Notebook: `notebooks/wafer_guard_fixed.ipynb`, with outputs saved from a run on 2026-10-22. Re-running it with the default settings replaces `artifacts/wafer_model.pt` (after making a timestamped backup).
- Dataset is not tracked in Git. Obtain an authorized copy and place it at `data/raw/secom_telemetry_modified.csv`.
- Run all cells from a fresh kernel. Notebook outputs must be generated by the code being submitted.
- The test set has already been inspected; no future evaluation should be described as an untouched final test without a newly held-out dataset.
- No claim of production readiness is made.

## 7. Limitations

- Only nine defects are present in the reported test partition, so uncertainty is substantial.
- The ROC-AUC and average-precision intervals are wide.
- A high alert burden and low average precision make this unsuitable for production triage without further independent validation.
- Numeric before/after evidence for leakage and a fresh measurement of the 146-row legacy invariant must be recorded only after the relevant checks are run.
