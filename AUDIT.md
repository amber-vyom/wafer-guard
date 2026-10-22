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

The original notebook is preserved as `notebooks/original_bnypi.ipynb` (git commit `ad30d10`, `notebooks/bnypi.ipynb`). Cell numbers below refer to it.

**Location (cell 8).**

```python
X_imp    = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X_raw)   # all 1,567 rows
X_scaled = StandardScaler().fit_transform(X_imp)                                           # all 1,567 rows
X_bal, y_bal = SMOTE(...).fit_resample(X_scaled, y_raw)                                    # before any split
X_train, X_test, ... = train_test_split(X_bal, y_bal, test_size=0.20, stratify=y_bal, random_state=SEED)
```

**Mechanism.**
1. Imputation means and scaling statistics are computed from the evaluation population, so every test row has already influenced preprocessing.
2. SMOTE runs before the split, so synthetic defects interpolated between real defects land on both sides of the split. A synthetic test defect sits next to its real "parent" in training.
3. The split is a stratified random shuffle. The `timestamp` column is never read, so training sees later wafers than the test set despite documented equipment ageing and recipe drift.
4. The test set is the balanced SMOTE output (293 of 586 rows defective, 50%) rather than the natural ~6.6% (104 of 1,567).

**Evidence** (`scripts/audit/reproduce_original_defects.py`, seed 1337, original model and hyper-parameters, threshold 0.5):

| Run | Test set | F1 | Precision | Recall | ROC-AUC | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---|
| A. Original pipeline as shipped | 586 rows, 50.0% defective | 0.9915 | 0.9832 | 1.0000 | 1.0000 | 288 / 5 / 0 / 293 |
| B. Leaky split, loss bug fixed | 586 rows, 50.0% defective | 0.9865 | 0.9734 | 1.0000 | 0.9991 | 285 / 8 / 0 / 293 |
| C. Chronological, train-only fit, no SMOTE, loss bug kept | 236 rows, 3.81% defective | 0.0000 | 0.0000 | 0.0000 | 0.6686 | 226 / 1 / 9 / 0 |
| D. Chronological, train-only fit, no SMOTE, loss fixed | 236 rows, 3.81% defective | 0.0000 | 0.0000 | 0.0000 | 0.6304 | 226 / 1 / 9 / 0 |

Run A reproduces the notebook's "F1 ≥ 0.90" release gate. Run B shows the leakage, not the loss bug, produces that score. Removing the leakage (C, D) drops F1 from about 0.99 to 0, with the model catching none of the 9 test defects.

**Repair.** Records are sorted by timestamp with a stable sort and cut by position into 70% / 15% / 15% train / validation / test (`src/data/splits.py`, shared by the notebook, the app and the evaluation scripts). The imputer and scaler are fitted on training rows only. SMOTE is not used. The validation set picks the checkpoint and the threshold, and the test set is read once for reporting.

### Bug 2 — Reconstruction-loss broadcasting

**Location (cells 10, 12, 14).**

```python
Xt = torch.from_numpy(X).float().unsqueeze(1)                      # cell 10: batch_x is [B, 1, F]
...
x = x.flatten(start_dim=1)                                         # cell 12: model output is [B, F]
...
recon_loss = torch.mean((reconstructed - batch_x) ** 2)            # cell 14: [B, F] - [B, 1, F]
```

**Mechanism.** The model flattens its input internally, so `reconstructed` is `[B, F]` while `batch_x` is `[B, 1, F]`. PyTorch broadcasts the subtraction to `[B, B, F]` (checked: `torch.Size([64, 64, 590])`). Every reconstruction is compared with every sample in the batch, and the value that minimises that loss is the batch mean for all samples. The loss stays finite and decreases, and nothing raises, so training looks healthy while the encoder receives no per-sample reconstruction signal. `clip_grad_norm_(..., 1.0)` further hides how wrong the gradients are. The classification term was unaffected (`logits` and labels are both `[B]`), which is why the headline metric did not reveal the bug.

**Evidence** (same script, 40 epochs, measured on the first 256 training rows):

| Training loss | Per-sample MSE of the reconstruction | MSE of always predicting the batch mean | Cross-sample spread kept by the reconstruction |
|---|---:|---:|---:|
| Original (broadcast) | 0.6121 | 0.6124 | 0.6% |
| Corrected | 0.1969 | 0.6124 | 81.8% |

With the original loss the autoencoder does no better than a constant batch-mean prediction and its outputs barely vary between wafers. With the corrected loss it reconstructs individual wafers.

**Repair.** Inputs are 2-D `[B, F]` tensors, the model rejects any other rank, and the training loop asserts `reconstructed.shape == batch_x.shape` and `batch_logits.shape == batch_y.shape` before the loss is computed.

### Bug 3 — Legacy PS-L invariant blindness

**Location (cell 8).** The original passes all 590 sensor columns, including `sensor_0` and `sensor_1`, through the same mean imputer and `StandardScaler`. The brief says `sensor_0 == 0.0` is a sentinel for Probe Station PS-L (legacy) wafers and that `sensor_1` on those wafers holds categorical firmware codes (100, 200, 300). The original has no code that looks at either fact.

**Mechanism ("invariant blindness").** The model is blind to the invariant that those rows come from a different tool format:
- The error codes are fed in as magnitudes, which invents an ordering (100 < 200 < 300) that does not exist.
- The sentinel `0.0` is treated as a real `sensor_0` reading.
- The 146 codes stretch the `sensor_1` scale, so the real variation is crushed.

**Evidence** (computed from `data/raw/secom_telemetry_modified.csv`, same script):

| Quantity | Value |
|---|---:|
| Wafers with `sensor_0 == 0.0` | 146 of 1,567 (9.3%) |
| Defects among them | 0 |
| `sensor_1` codes on those wafers | 100: 61, 200: 35, 300: 50 |
| `sensor_1` on the other 1,417 wafers (mean / std) | 2,495.7 / 79.8 |
| `sensor_1` std the original scaler used (all rows) | 675.1 |
| Real `sensor_1` spread after the original scaling | about ±0.12 σ |
| Legacy codes after the original scaling | -3.2, -3.1, -2.9 σ |
| Legacy codes on the real wafers' scale | -30.0, -28.8, -27.5 σ |
| Sentinel `sensor_0 = 0.0` against real wafers (mean 3,014.0, std 74.3) | -40.6 σ |

The original scaler makes the codes look like mildly low readings, about -3σ, instead of a separate category, and it squeezes the genuine `sensor_1` signal of normal wafers into about one eighth of a standard deviation. No test in the original notebook could detect this.

**Repair.** Legacy wafers are identified with `sensor_0 == 0.0`, and an `is_legacy_ps_l` indicator plus one indicator per code (`legacy_sensor_1_code_100/200/300`) are added. The sentinel and the codes are masked before imputation and scaling, so they never reach the continuous features. A code outside 100/200/300 sets no code indicator, and the training notebook and the app share one feature layout (covered by `tests/test_features.py` and `tests/test_artifacts.py`).

**Not yet demonstrated.**
- No invariance test shows that the model's score is unaffected by changing the code value on a legacy row. Indicator columns existing is not proof that the blindness is gone.
- All 146 legacy wafers are non-defective. The legacy indicator can only rule wafers out; it cannot find defects.
- The notebook-run model (`artifacts/notebook_run`) excludes `sensor_0` and `sensor_1` from the continuous features altogether, so the real `sensor_1` readings of the 1,417 non-legacy wafers are not used.

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

### Why 3a and 3b differ: two separate trainings

They are not the same model, so their metrics must not be mixed. The dashboard evaluates 3b; the committed notebook produces 3a and writes its artifacts to a separate folder, so re-running the notebook does not change what the dashboard shows.

| | 3a. Notebook run | 3b. Deployed model (dashboard) |
|---|---|---|
| Checkpoint | `artifacts/notebook_run/wafer_model.pt` | `artifacts/wafer_model.pt` |
| Written | 2026-10-22 00:05 | 2026-10-21 21:07 |
| Input features | 592 | 472 |
| Threshold | 0.00067028 | 0.00100333 |
| Test ROC-AUC / TP of 9 | 0.679 / 2 | 0.747 / 1 |
| SHA-256 (model) | `245b348f6e676fef…bf30e1d` | `5ccee7132d9f2180…c030e444` |

Hashes are of the files on disk at the time of writing. The 3b column was confirmed by loading the app and clicking "Run Model": it showed 236 records, 9 defects, 1 detected, 12 false alarms, ROC-AUC 0.7469 and average precision 0.0961. With 9 test defects, the 0.68 versus 0.75 AUC difference is well inside the notebook run's bootstrap interval (0.48-0.85), so neither run is shown to be better.

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
- Original defects: `notebooks/original_bnypi.ipynb` is the supplied notebook, kept unmodified for reference (a leading byte-order mark was removed so Jupyter can open it). `python scripts/audit/reproduce_original_defects.py` re-runs it under four ablations and prints the numbers quoted in section 2 (about 3 minutes on CPU; needs `imbalanced-learn`).
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
