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

The supplied current-run result is:

| Metric | Reported value |
|---|---:|
| Test records | 236 |
| Test defects | 9 |
| Test defect prevalence | 3.81% |
| ROC-AUC | 0.631 |
| ROC-AUC bootstrap 95% CI | 0.43–0.79 |
| Average precision | 0.060 |
| Average precision bootstrap 95% CI | 0.03–0.13 |

The reported confusion matrix values are TP=8, FP=196, FN=1. Given 236 records and 9 defects, these imply TN=31. This corresponds to 204 flagged records (86.4% of the test set), so it is not an acceptable alert policy. The threshold policy must be selected using validation data and should target a fixed nominal alert rate (the current notebook revision uses 10%) rather than maximizing validation F1 at a near-zero threshold.

**Important:** the metrics above describe the supplied run before the fixed-alert-rate/early-stopping revision is rerun, and before the notebook switched to the shared `WaferAutoencoder` class, so they are not reproducible from the current notebook. They must not be presented as the metrics of that revised run. After rerunning the revised notebook, replace this table with its printed validation metrics, test metrics, threshold, alert rate, confusion matrix, and bootstrap confidence intervals.

## 4. Threshold and overfitting controls

The notebook revision selects the threshold from validation scores for a nominal 10% alert rate, reports the actual validation/test alert rates, and selects the model checkpoint by validation average precision with patience-based early stopping. The test partition is not used for checkpoint selection or threshold selection. If tied scores cause the alert rate to exceed the target, report the actual rate rather than hiding the difference.

## 5. Reproducibility and scope

- Model: PyTorch `WaferAutoencoder` (`src/models/wafer_model.py`), shared by the notebook and the app.
- Split: one chronological 70/15/15 definition in `src/data/splits.py`, used by the notebook, the Streamlit app, and the evaluation scripts (test = last 15%, 236 rows, 9 defects).
- Artifacts: the notebook saves `artifacts/wafer_model.pt` and `artifacts/preprocessor.joblib` in the format the app loads, backs up any existing copies, and ends with a parity check that reloads them through the app's inference code.
- Tests: `python -m pytest` covers the split, legacy-sensor feature rules, model shapes, and train/serve parity on synthetic data. It does not measure model quality.
- Notebook: `notebooks/wafer_guard_fixed.ipynb` (use the revised notebook after merging it).
- Dataset is not tracked in Git. Obtain an authorized copy and place it at `data/raw/secom_telemetry_modified.csv`.
- Run all cells from a fresh kernel. Notebook outputs must be generated by the code being submitted.
- The test set has already been inspected; no future evaluation should be described as an untouched final test without a newly held-out dataset.
- No claim of production readiness is made.

## 6. Limitations

- Only nine defects are present in the reported test partition, so uncertainty is substantial.
- The ROC-AUC and average-precision intervals are wide.
- A high alert burden and low average precision make this unsuitable for production triage without further independent validation.
- Numeric before/after evidence for leakage and a fresh measurement of the 146-row legacy invariant must be recorded only after the relevant checks are run.
