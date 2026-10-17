# Wafer Guard — Audit and Repair

## 1. Objective

Audit and repair the wafer anomaly detection pipeline, focusing on data leakage, tensor shapes, gradient flow, legacy sensor semantics, and evaluation at natural defect prevalence.

## 2. Dataset and Evaluation

The dataset contains 1,567 records, split chronologically:

| Partition | Records | Defect prevalence |
|---|---:|---:|
| Training | 1,096 | 7.12% |
| Validation | 157 | 5.73% |
| Test | 314 | 5.41% |

The held-out test set contains 17 defects. It was not artificially balanced.

## 3. Repairs and Checks

### Data leakage

The notebook sorts records chronologically and splits them into training, validation, and test partitions. The imputer and scaler are fitted only on training data. The classification threshold is selected using validation data, not the test set.

### Tensor shapes and loss

The model checks that inputs are two-dimensional. Reconstruction output matches the input shape, and the classifier returns one logit per record. Assertions verify reconstruction and classification shapes before calculating losses.

The training objective combines mean squared reconstruction loss with weighted binary cross-entropy. Backpropagation and optimizer updates are performed for each training batch.

### Legacy PS-L sensor semantics

For legacy records, `sensor_0 == 0.0` is treated as a sentinel. The notebook masks the affected `sensor_0` measurement and does not treat `sensor_1` as a continuous feature. Firmware codes 100, 200, and 300 are represented using separate indicator features, alongside a legacy-record indicator.

### Natural defect prevalence

The test partition retains its natural prevalence. Accuracy is reported alongside precision, recall, F1, average precision, ROC-AUC, and the confusion matrix.

## 4. Measured Test Results

| Metric | Result |
|---|---:|
| Test records | 314 |
| Actual defects | 17 |
| Test prevalence | 5.41% |
| Decision threshold | 0.01 |
| Accuracy | 0.9331 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 score | 0.0000 |
| Average precision | 0.0948 |
| ROC-AUC | 0.6778 |

Confusion matrix, with rows representing actual labels and columns representing predicted labels:

```text
[[293   4]
 [ 17   0]]
```

The model detected none of the 17 defective test records. Its accuracy is therefore misleading as a standalone measure. These results do not support deployment.

## 5. Limitations and Next Steps

Investigate the classification scores, training objective, temporal distribution shifts, and threshold behavior. Evaluate any changes using validation data before performing a final held-out test evaluation.

The notebook's preprocessing and tensor-shape checks do not, by themselves, establish that every original defect has been eliminated. The original data-leakage, corrupted-gradient, and invariant-blindness defects must be traced to their original implementation and verified against the repaired code before claiming complete resolution.

## 6. Reproducibility

Run `notebooks/wafer_guard_fixed.ipynb` from top to bottom with the required dataset and dependencies installed. The notebook saves the trained model and preprocessing objects under `artifacts/`. Keep raw datasets and generated model artifacts out of Git commits.
