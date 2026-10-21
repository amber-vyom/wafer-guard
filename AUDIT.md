# WaferGuard — Final PyTorch Model Audit

## 1. Executive Summary

WaferGuard is a semiconductor wafer anomaly-detection prototype developed for the GDG Tech Wildcard AI/ML task.

The project processes sensor telemetry, applies a preprocessing pipeline, and uses a PyTorch model to generate predictions about potentially defective wafers. The application provides a Streamlit interface for running inference and viewing results.

The audit focuses on the three silent failure categories identified in the GDG brief:

1. Data leakage and violation of temporal causality.
2. Corrupted loss gradients caused by incorrect tensor shapes and broadcasting.
3. Invariant blindness caused by incorrect handling of legacy Probe Station PS-L sensor semantics.

The project uses the existing PyTorch model as its final application model. The current repository cleanup is restricted to documentation and repository hygiene; it does not retrain the model, modify inference logic, or regenerate model artifacts.

**Overall assessment:** The project demonstrates a working application workflow, but the recorded defect-detection performance is limited. It should be treated as an academic prototype, not a production-ready wafer inspection system.

## 2. Problem Statement and Objectives

Semiconductor manufacturing produces large volumes of sensor telemetry. Detecting unusual sensor patterns can help identify wafers that may require additional inspection.

A model can appear to perform well while containing serious implementation defects. Data leakage can make evaluation overly optimistic, incorrect tensor broadcasting can change the intended training objective, and mishandling special sensor values can hide meaningful manufacturing information.

The objectives of this project are to:

- Process semiconductor sensor telemetry.
- Respect chronological ordering and prevent future information from influencing earlier predictions.
- Separate training, validation, and test data.
- Fit preprocessing transformations using training data only.
- Ensure model outputs and loss tensors have compatible shapes.
- Preserve the meaning of legacy Probe Station PS-L measurements and error codes.
- Evaluate the final PyTorch model at the dataset's natural defect prevalence.
- Document limitations and avoid unsupported claims of production readiness.

## 3. Dataset and Natural Defect Prevalence

The documented dataset is `data/raw/secom_telemetry_modified.csv`.

It contains 1,567 records divided into chronological partitions.

| Partition | Records | Defects | Defect prevalence |
|---|---:|---:|---:|
| Training | 1,096 | 78 | 7.12% |
| Validation | 157 | 9 | 5.73% |
| Test | 314 | 17 | 5.41% |
| **Total** | **1,567** | **104** | **6.64%** |

The target labels are documented as:

- `-1`: Normal
- `+1`: Defect

The held-out test partition contains 17 defective records among 314 total records. The test partition retains its natural class distribution rather than being artificially balanced.

This distinction matters because an imbalanced dataset can produce high accuracy even when the model fails to identify the minority defect class. Precision, recall, F1-score, average precision, ROC-AUC, and confusion-matrix counts therefore provide important information alongside accuracy.

The test partition's prevalence is approximately 5.41%; the full dataset's defect prevalence is approximately 6.64%. These figures describe different populations and should not be interchanged.

## 4. Silent Bug 1 — Data Leakage and Temporal Causality

### 4.1 The problem

Data leakage occurs when information that should be unavailable during training or prediction influences the model or its preprocessing.

In a manufacturing setting, future observations must not influence a prediction made at an earlier time. Similarly, statistics calculated from validation or test observations must not influence preprocessing fitted for training.

Potential sources of leakage include:

- Randomly mixing temporally ordered observations before splitting.
- Fitting an imputer or scaler on the entire dataset before separating the partitions.
- Selecting a classification threshold using test labels.
- Using the held-out test set to choose model configurations or features.

These practices can make evaluation results appear better than the model's performance on genuinely unseen data.

### 4.2 Required repair

The corrected workflow should:

1. Preserve chronological ordering when creating training, validation, and test partitions.
2. Fit the imputer using training data only.
3. Fit the scaler using the training data after imputation.
4. Transform validation and test data using the already-fitted training transformations.
5. Select model configurations and classification thresholds using training and validation information, as appropriate.
6. Keep the test partition out of model selection and use it only for the designated final evaluation.

For a fitted transformation, the distinction is:

- Training data: `fit_transform`
- Validation data: `transform`
- Test data: `transform`

The validation and test partitions must not refit the preprocessing objects.

### 4.3 Verification and remaining limitations

The documented workflow uses chronological partitions and training-only imputer/scaler fitting. Threshold selection is documented as a validation-stage operation.

These are appropriate safeguards, but their existence does not prove that every historical experiment avoided test-set reuse. Earlier experiments may have examined the same test partition.

Consequently, the recorded test metrics should be presented as the existing evaluation results for this submission, not as a newly conducted independent evaluation on previously unseen data.

## 5. Silent Bug 2 — Corrupted Loss Gradients and Tensor Broadcasting

### 5.1 The problem

PyTorch operations can broadcast tensors with compatible but unintended shapes. A loss calculation may therefore execute without raising an error while calculating a different objective from the one intended.

For example, suppose a batch contains \(B\) records and the model predicts one logit per record.

The expected classification shapes are:

- Logits: `(B,)` or `(B, 1)`
- Targets: the same shape as the logits

If logits have shape `(B, 1)` while targets have shape `(B,)`, some elementwise operations can broadcast them to `(B, B)`. The resulting loss can compare predictions against unrelated records in the batch instead of matching each prediction with its corresponding label.

This can silently corrupt the training objective and its gradients.

Reconstruction loss has a similar requirement: the reconstruction tensor must match the input tensor's shape.

### 5.2 Required repair

The model and training loop should enforce these invariants:

1. Sensor input tensors have the expected batch and feature dimensions.
2. Reconstruction outputs have the same shape as the corresponding input tensors.
3. Classification produces one logit per record.
4. Classification targets have the same shape as the logits when passed to the loss function.
5. Loss values are calculated using correctly aligned tensors.
6. Backpropagation and optimizer updates operate on the intended loss.

For a batch of \(B\) records with \(F\) features, reconstruction input and output should both have shape `(B, F)`. Classification logits and labels should each represent exactly \(B\) records.

The training objective documented for the model combines mean squared reconstruction loss with weighted binary cross-entropy. Shape checks are important for both components because an incorrect shape can change the loss being optimized.

### 5.3 Verification and remaining limitations

The repaired workflow documents input-dimensionality checks and assertions for reconstruction and classification output shapes. These checks are intended to detect incompatible tensor dimensions before the loss is calculated.

A complete verification of gradient correctness also requires checking the actual training path, loss inputs, and optimizer updates. Shape assertions alone do not establish that every gradient is mathematically correct.

No new training or gradient experiment is performed during this documentation cleanup.

## 6. Silent Bug 3 — Invariant Blindness and Legacy PS-L Semantics

### 6.1 The problem

The GDG task describes a legacy Probe Station PS-L that produces sensor values with special meanings.

For legacy wafers:

- `sensor_0 == 0.0` is a sentinel indicating a legacy record, not a genuine continuous measurement.
- `sensor_1` contains categorical firmware error codes `100.0`, `200.0`, or `300.0`.
- These error codes must not be treated as ordinary continuous measurements.

If these values are handled as normal sensor readings, the pipeline can lose information about the measurement system. Scaling the categorical codes as continuous values imposes a misleading numeric relationship between categories. Imputing fabricated continuous values can also obscure the distinction between a genuine measurement and a special code.

This is invariant blindness: the preprocessing pipeline fails to preserve information about the underlying measurement process.

### 6.2 Required repair

The documented preprocessing approach distinguishes legacy records from ordinary measurements.

The intended handling is:

1. Identify legacy records using the `sensor_0 == 0.0` sentinel.
2. Mask the affected `sensor_0` measurement rather than treating the sentinel as a genuine measurement.
3. Avoid treating legacy `sensor_1` error codes as continuous sensor values.
4. Represent codes `100`, `200`, and `300` with separate indicator features.
5. Preserve a legacy-record indicator so the model can distinguish these records from ordinary measurements.
6. Keep categorical indicators separate from continuous-feature scaling.

The resulting representation preserves both the missing-measurement information and the categorical meaning of the legacy firmware codes.

### 6.3 Verification and remaining limitations

The existing prediction service documents the legacy sentinel and code-indicator handling. The saved preprocessor contains the configured sensor and indicator feature lists.

The inspected preprocessor contains 472 expected features, including 468 usable sensor columns and four indicator columns.

That configuration is evidence of the saved feature layout; it is not, by itself, proof that every legacy record is handled correctly in every possible input dataset. The model and preprocessing artifacts remain unchanged during the current cleanup.

## 7. Final PyTorch Model and Application Architecture

The final application model is the existing PyTorch model.

| Component | Location |
|---|---|
| Streamlit application | `app.py` |
| PyTorch model definition | `src/models/wafer_model.py` |
| Prediction service | `src/services/prediction_service.py` |
| Final model artifact | `artifacts/wafer_model.pt` |
| Saved preprocessor | `artifacts/preprocessor.joblib` |
| Corrected notebook | `notebooks/wafer_guard_fixed.ipynb` |

The application workflow is:

1. Load sensor data.
2. Prepare numeric and categorical features according to the existing preprocessing pipeline.
3. Load the saved preprocessor and PyTorch checkpoint.
4. Generate predictions.
5. Present the output through the Streamlit interface.

The current application expects its model and preprocessor at the configured local paths. Generated artifacts are not tracked in Git and may need to be supplied separately when setting up a fresh checkout.

No Gradient Boosting, Random Forest, hybrid, or Top-20 candidate is designated as the final application model in this audit.

## 8. Final PyTorch Evaluation Results

The final application model for this submission is PyTorch. This section records the previously documented PyTorch result; it does not describe a new evaluation run.

The notebook's saved output records a test ROC-AUC of approximately `0.65993`. The previously documented PyTorch evaluation also reports the following classification result:

| Metric | Recorded result |
|---|---:|
| Test records | 314 |
| Actual defective records | 17 |
| Defects detected | 2 |
| Defects missed | 15 |
| False alarms | 17 |
| True negatives | 280 |
| Precision | 10.53% |
| Recall | 11.76% |
| F1-score | 11.11% |
| ROC-AUC | Approximately 0.65993 |

The corresponding confusion matrix, with actual labels as rows and predicted labels as columns, is:

```text
                 Predicted Normal  Predicted Defective
Actual Normal                 280                   17
Actual Defective               15                    2
```

These classification counts imply:

- True negatives: 280
- False positives: 17
- False negatives: 15
- True positives: 2

The model identifies only 2 of the 17 defective records. Its recall is therefore low, and it also produces 17 false alarms. These limitations matter more than a high overall accuracy would suggest.

**Metric provenance:** These are previously recorded results, not newly generated results. The notebook's saved ROC-AUC output and the previously documented classification metrics should be confirmed to originate from the same final PyTorch checkpoint and evaluation run before they are represented as one fully reconciled metrics report. No new evaluation is performed here.

The test partition retains its natural prevalence of approximately 5.41%; it is not artificially balanced.

## 9. Interpretation and Go/No-Go Assessment

### 9.1 Technical interpretation

The project addresses the three categories of silent failure described in the GDG brief:

- Causality and leakage: chronological splitting and training-only preprocessing are documented safeguards.
- Tensor mechanics: shape assertions and correctly aligned loss inputs are required to avoid unintended broadcasting.
- Legacy semantics: the PS-L sentinel and firmware codes are represented separately from continuous measurements.

These safeguards are important, but they should not be confused with evidence of strong predictive performance.

### 9.2 Model performance

The recorded classification result detects only 2 of 17 defective test records and misses 15. Recall of 11.76% is insufficient for a use case in which missing a defective wafer could have substantial consequences.

The 17 false alarms also indicate that predictions can create additional inspection work.

### 9.3 Final decision

**Production deployment: NO-GO.**

The current evidence does not justify using the model for autonomous manufacturing decisions. The project can be presented as an academic prototype demonstrating a sensor-data pipeline, PyTorch inference, and evaluation methodology.

This decision does not require changing the existing model during the current static cleanup.

## 10. Verification and Reproducibility

The following checks were previously recorded in the development environment:

- The standalone smoke test passed seven checks.
- The dependency check was reported clean.
- The Streamlit interface was previously confirmed working.
- The saved preprocessor was inspected and found to contain a 472-feature configuration.

These checks verify selected application behaviors and configuration. They do not independently prove high defect-detection performance or production readiness.

The application requires:

- `data/raw/secom_telemetry_modified.csv`
- `artifacts/wafer_model.pt`
- `artifacts/preprocessor.joblib`
- The project's required Python dependencies

The standalone smoke test can be run using:

```powershell
python .\src\evaluation\smoke_test.py
```

The Streamlit application can be launched using:

```powershell
streamlit run app.py
```

The notebook is the documented training and evaluation workflow. Running it may regenerate artifacts; it is not required for the current documentation cleanup and should not be run as part of these static changes.

## 11. Limitations

- The final PyTorch model has low recorded defect recall.
- The classification metrics and saved ROC-AUC output must be tied to the same checkpoint and run before being treated as a fully reconciled evaluation report.
- The test partition contains only 17 defective records, making individual classification errors influential.
- Historical experiments may have examined the test partition, limiting claims that it represents an untouched independent evaluation.
- Sensor missingness and distribution changes may affect generalization.
- Shape assertions and preprocessing safeguards do not independently prove the absence of every implementation defect.
- Model artifacts may need to be obtained separately because they are not tracked in Git.
- A successful smoke test or working dashboard does not establish production readiness.

## 12. Scope of the Current Cleanup

The current work is limited to static documentation and repository hygiene.

It does not include:

- Retraining the PyTorch model.
- Changing model architecture or inference behavior.
- Modifying preprocessing logic.
- Regenerating model artifacts.
- Running alternative model experiments.
- Performing a new model evaluation.

The existing PyTorch model and preprocessor remain unchanged.

## 13. Conclusion

WaferGuard demonstrates a local semiconductor sensor-analysis workflow using preprocessing, PyTorch inference, and Streamlit visualization.

This audit addresses the GDG brief's three silent-bug categories: data leakage, corrupted loss gradients, and invariant blindness. It describes the corresponding safeguards, legacy PS-L sensor semantics, natural-prevalence evaluation, reproducibility requirements, and limitations.

The existing PyTorch model remains the final application model. Its previously recorded result detects 2 of 17 defective test records, demonstrating that the current defect-detection performance is limited.

The final recommendation is **NO-GO for production deployment**. The current submission should be presented as an academic prototype, without claiming that static documentation changes improve model performance.
---

## Latest Evaluation Evidence — Exploratory Results

The following values were copied from the notebook execution output shared on 21 October 2026. The test set has already been inspected, so these results are exploratory and must not be presented as an untouched final test estimate.

### Dataset split

| Partition | Records | Defect prevalence |
|---|---:|---:|
| Training | 1,096 | 7.12% |
| Validation | 157 | 5.73% |
| Test | 314 | 5.41% |

### Forward-chaining evaluation

The reported pooled out-of-fold evaluation covers 627 records and 22 defects, with a defect prevalence of 3.51%.

- Pooled average precision: 0.0401
- Pooled ROC-AUC: 0.4067
- Fold ROC-AUC values: 0.495, 0.524, and 0.402
- Fold average precision values: 0.078, 0.040, and 0.040

These results do not demonstrate reliable ranking of future defects.

### Exploratory test results

The notebook output reports 314 test records, including 17 defects.

- Average precision: 0.09348
- ROC-AUC: 0.65993
- Primary reported threshold: 0.0010033317
- Flagged records: 19 of 314
- True positives: 2
- False positives: 17
- True negatives: 280
- False negatives: 15
- Precision: 10.53%
- Recall: 11.76%
- F1 score: 11.11%

Confusion matrix, with rows representing actual classes and columns representing predicted classes:

| | Predicted non-defect | Predicted defect |
|---|---:|---:|
| Actual non-defect | 280 | 17 |
| Actual defect | 15 | 2 |

### Interpretation and release decision

The reported exploratory test result detects only 2 of 17 defects and produces 17 false alarms. The forward-chaining ROC-AUC of 0.4067 is also concerning. The release decision remains **NO-GO for production**.

The execution output also contains a configuration with `max_iter`, `learning_rate`, and `max_leaf_nodes`. These values suggest a non-PyTorch estimator may be involved. Until the executed notebook and estimator are verified, the metrics in this addendum are attributed to the supplied execution output, not specifically to the final PyTorch model.

No bootstrap confidence intervals were present in the supplied output; none are claimed here.

