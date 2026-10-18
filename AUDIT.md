# WaferGuard - Audit and Evaluation Report

## 1. Objective

WaferGuard is an experimental wafer anomaly detection application. This report documents the dataset, preprocessing safeguards, recorded evaluation results, and limitations relevant to submission.

The current model is frozen for this submission. These results do not establish production readiness.

## 2. Dataset and Evaluation Split

The dataset contains 1,567 records, split chronologically.

| Partition | Records | Defect prevalence |
|---|---:|---:|
| Training | 1,096 | 7.12% |
| Validation | 157 | 5.73% |
| Test | 314 | 5.41% |

The held-out test partition contains 17 defective records and retains its natural class prevalence.

## 3. Pipeline Safeguards

### Data splitting and preprocessing

The documented workflow uses chronological training, validation, and test partitions. The imputer and scaler are fitted on training data only, and threshold selection is documented as a validation-stage operation. These safeguards do not independently prove that all possible data leakage has been eliminated.

### Tensor shapes and training objective

The documented model checks input dimensions, reconstruction output shape, and classification output shape. Its training objective combines mean squared reconstruction loss with weighted binary cross-entropy. These checks do not guarantee useful defect detection performance.

### Legacy PS-L sensor handling

The prediction service treats `sensor_0 == 0.0` as a legacy-record sentinel. It masks the affected `sensor_0` and `sensor_1` measurements, creates a legacy-record indicator, and represents `sensor_1` codes 100, 200, and 300 using separate indicator features.

## 4. Recorded Evaluation Results

The repository contains different recorded evaluation results. Their exact model checkpoints and evaluation runs have not been reconciled. The results below must be interpreted separately.

### 4.1 Dashboard-displayed evaluation

| Metric | Displayed result |
|---|---:|
| Test records | 314 |
| Actual defects | 17 |
| Accuracy | 94.27% |
| Precision | 0.00 |
| Recall | 0.00 |
| F1 score | 0.00 |
| Average precision | 0.0948 |
| ROC-AUC | 0.6324 |

The dashboard displays this confusion matrix, with actual labels as rows and predicted labels as columns:

```text
                 Predicted Normal  Predicted Defective
Actual Normal                296                    1
Actual Defective              17                    0
```

This matrix implies 296 true negatives, one false positive, 17 false negatives, and zero true positives. Accuracy is 296 / 314 = 94.27%.

The model misses all 17 defective records in this displayed classification result. The matrix verifies the classification counts and derived classification metrics, but does not independently verify average precision or ROC-AUC.

### 4.2 Previously documented audit evaluation

The earlier audit recorded a separate result:

| Metric | Previously recorded result |
|---|---:|
| Test records | 314 |
| Actual defects | 17 |
| Decision threshold | 0.01 |
| Accuracy | 93.31% |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 score | 0.0000 |
| Average precision | 0.0948 |
| ROC-AUC | 0.6778 |

Previously recorded confusion matrix:

```text
[[293   4]
 [ 17   0]]
```

This matrix implies 293 true negatives, four false positives, 17 false negatives, and zero true positives. Accuracy is 293 / 314 = 93.31%.

These results differ from the dashboard-displayed evaluation. They must not be combined as one run. The ROC-AUC values also differ; changing only a classification threshold cannot explain this because ROC-AUC uses scores across thresholds. The underlying model, scores, or evaluation run must be checked to reconcile them.

## 5. Experimental Model Results

A separate Top-20-feature Random Forest candidate recorded the following results:

- Validation average precision: 0.14122
- Test average precision: 0.06160
- Test ROC-AUC: 0.53832
- Test recall: 0.58824
- Test precision: 0.06410
- False positives: 146

These results belong to an experimental candidate, not the deployed autoencoder. They indicate substantial false-alarm risk and are not evidence that the deployed model has improved.

## 6. Limitations

- The dashboard-displayed classification result detects none of the 17 defective test records.
- Accuracy is misleading when the model fails to detect the minority defect class.
- The dashboard's recorded metrics are not recalculated for uploaded datasets.
- The dashboard and audit results have not been reconciled to one evaluation run.
- Dataset distribution shifts and sensor missingness may affect generalization.
- Preprocessing and tensor-shape checks alone do not prove that all leakage, gradient-flow, or invariant-handling issues have been eliminated.

The current model is an academic prototype, not a production-ready wafer inspection system.

## 7. Submission and Reproducibility

The model and preprocessor are frozen for the current submission. Do not retrain the model or overwrite its artifacts during documentation cleanup.

The notebook `notebooks/wafer_guard_fixed.ipynb` contains the documented training workflow. Running it may regenerate model artifacts; do not run it during finalization unless retraining is intentionally approved.

Keep raw datasets and generated model artifacts out of Git commits, in accordance with `.gitignore`.

For future model development, evaluate changes using validation data, record the exact checkpoint and metrics for each run, and reserve a clearly identified held-out test evaluation for the final candidate.