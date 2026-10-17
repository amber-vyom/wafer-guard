# Wafer Guard — Audit Report

## 1. Objective

Audit and repair the wafer anomaly detection pipeline, with particular attention to data leakage, tensor shapes, gradient flow, and legacy sensor semantics.

## 2. Data Inspection

- Total records: 1,567
- Sensor features before preprocessing: 590
- Normal wafers: 1,463
- Defective wafers: 104
- Defect prevalence: approximately 6.64%
- Missing sensor values: 41,948

The dataset is imbalanced. Accuracy alone is insufficient to evaluate defect detection.

## 3. Identified Risks and Repairs

### A. Data Leakage

**Risk:** Fitting preprocessing transformations or selecting features using validation or test data allows information from those records to influence the model.

**Repair:** Split records chronologically. Fit imputation, scaling, and feature selection using training data only. Use validation data for threshold selection and reserve the test set for final evaluation.

### B. Tensor Shapes and Gradients

**Risk:** Incompatible tensor shapes can cause unintended broadcasting during reconstruction loss calculation. Incorrect gradient handling can also prevent model components from learning as intended.

**Repair:** Keep reconstruction and input tensors at shape `[batch_size, features]`. Check model output shapes and ensure training losses retain their gradient paths.

### C. Legacy Sensor Semantics

**Risk:** Treating legacy sentinel values as ordinary measurements can introduce misleading features.

**Repair:** Treat `sensor_0 == 0.0` according to the legacy sentinel rule and preserve the categorical meaning of `sensor_1` values 100, 200, and 300 for legacy records.

## 4. Evaluation Protocol

The data is split chronologically into training, validation, and test sets. Preprocessing is fitted on training data. The validation set is used for threshold selection, and the test set is reserved for final evaluation.

## 5. Measured Results

Results from the current training script:

| Metric | Test result |
|---|---:|
| Test records | 314 |
| Actual defective records | 17 |
| Accuracy | 94.27% |
| Precision | 0.00 |
| Recall | 0.00 |
| F1 score | 0.00 |
| Average precision | 0.0948 |
| ROC-AUC | 0.6324 |

The test confusion matrix is:

- True negatives: 296
- False positives: 1
- False negatives: 17
- True positives: 0

## 6. Limitations

The current model fails to identify any defective wafers in the test set. Its accuracy is misleading because most wafers are normal. The model is not ready for deployment.

The validation set contains only nine defective wafers, so its selected threshold is uncertain. Improvements must be evaluated without tuning against the test set.

## 7. Reproducibility

The executed notebook is `notebooks/wafer_guard_fixed.ipynb`. The notebook completed without cell execution errors. Model execution success does not imply successful defect detection.

## 8. Conclusion

The audit documents the identified risks, preprocessing safeguards, legacy sensor handling, and measured evaluation results. Further work is needed to improve defective-wafer recall while maintaining strict chronological separation and preventing leakage.