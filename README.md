# WaferGuard: Wafer Defect Detection and Monitoring

WaferGuard is an experimental wafer anomaly detection application built to explore semiconductor manufacturing data, sensor preprocessing, machine learning, and interactive monitoring.

The project combines a saved PyTorch model, a preprocessing pipeline, and a Streamlit dashboard for wafer-level predictions and evaluation.

**Project status:** Academic prototype prepared for submission. The current model is frozen. The application runs, but its defect detection performance is not suitable for production use.

## 1. Project Objectives

- Explore semiconductor wafer sensor data.
- Build a preprocessing pipeline for high-dimensional sensor measurements.
- Handle missing sensor values and legacy PS-L sensor records.
- Generate anomaly probabilities and reconstruction errors.
- Present results through an interactive dashboard.
- Evaluate performance using measures suitable for imbalanced classification.
- Document limitations and maintain reproducible experiments.

## 2. Dataset Overview

The dataset contains 1,567 wafer records and 590 sensor columns.

| Dataset characteristic | Value |
|---|---:|
| Total records | 1,567 |
| Sensor columns | 590 |
| Normal records | 1,463 |
| Defective records | 104 |
| Missing sensor cells | 41,948 |
| Defect prevalence | Approximately 6.64% |

Defects are a minority class, so accuracy alone is not a reliable measure of model quality.

### Chronological data split

| Partition | Records | Defect prevalence |
|---|---:|---:|
| Training | 1,096 | 7.12% |
| Validation | 157 | 5.73% |
| Test | 314 | 5.41% |

The held-out test partition contains 17 defective records.

## 3. Technology Stack

- **Python** — application and data processing
- **PyTorch** — autoencoder and classification model
- **scikit-learn** — preprocessing and evaluation utilities
- **Pandas and NumPy** — data manipulation
- **Streamlit** — interactive web dashboard
- **Plotly** — data visualization
- **Joblib** — saved preprocessing objects

## 4. Application Features

The Streamlit dashboard contains five pages.

### Overview
Provides a high-level introduction to the application and its current model limitations.

### Dataset Explorer
Supports exploration of the available dataset and sensor measurements.

### Live Prediction
Accepts sensor records for inference and displays model probabilities, the classification threshold, reconstruction errors, and available dataset labels for reference.

### Model Evaluation
Displays recorded test evaluation metrics and a confusion matrix. These are fixed recorded results; they are not recalculated for uploaded datasets.

### Audit
Documents preprocessing safeguards, legacy sensor handling, evaluation limitations, and future improvement areas.

## 5. System Architecture

The application follows this inference workflow:

1. Receive wafer sensor data.
2. Validate the required sensor columns.
3. Convert sensor values to numeric data.
4. Apply legacy PS-L sensor handling.
5. Impute missing values using the saved preprocessor.
6. Scale measurements and prepare indicator features.
7. Load the saved PyTorch model.
8. Generate classification probabilities and reconstruction errors.
9. Apply the saved classification threshold.
10. Display results in the Streamlit dashboard.

The deployed model and preprocessing artifacts are stored locally under `artifacts/`.

## 6. Model Evaluation and Limitations

The repository contains multiple recorded evaluation results. The dashboard and the earlier audit report have not been reconciled to a single model checkpoint and evaluation run. Their metrics are therefore documented separately.

### Dashboard-displayed evaluation

| Metric | Recorded result |
|---|---:|
| Test records | 314 |
| Actual defects | 17 |
| Accuracy | 94.27% |
| Precision | 0.00 |
| Recall | 0.00 |
| F1 score | 0.00 |
| Average precision | 0.0948 |
| ROC-AUC | 0.6324 |

Dashboard-displayed confusion matrix:

```text
                 Predicted Normal  Predicted Defective
Actual Normal                296                    1
Actual Defective              17                    0
```

The model missed all 17 defective records in this displayed result. The 94.27% accuracy is misleading because the model did not identify any defects.

The confusion matrix supports the classification counts and derived classification metrics. Average precision and ROC-AUC are recorded values that have not been independently verified from the displayed matrix.

### Previously documented audit evaluation

The earlier audit recorded a separate evaluation at a threshold of 0.01:

| Metric | Previously recorded result |
|---|---:|
| Accuracy | 93.31% |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 score | 0.0000 |
| Average precision | 0.0948 |
| ROC-AUC | 0.6778 |

Its confusion matrix was:

```text
[[293   4]
 [ 17   0]]
```

These results differ from the dashboard's displayed evaluation. They must not be combined as if they came from the same run. The different ROC-AUC values also require investigation of the underlying model scores or evaluation runs.

### Separate experimental Random Forest

A separate Top-20-feature Random Forest experiment recorded:

- Validation average precision: 0.14122
- Test average precision: 0.06160
- Test ROC-AUC: 0.53832
- Test recall: 0.58824
- Test precision: 0.06410
- False positives: 146

These results belong to an experimental candidate, not the deployed autoencoder. They indicate a substantial false-alarm risk and do not demonstrate an improvement to the deployed model.

### Overall assessment

WaferGuard is an academic prototype, not a production-ready wafer inspection system. The current recorded dashboard result has zero defect recall. Further model development would require careful threshold selection, investigation of temporal distribution shifts and missingness, and a clearly documented final evaluation.

## 7. Installation

Use a compatible Python environment and install the dependencies from the project root.

```powershell
python -m pip install -r requirements.txt
```

If the project virtual environment is already configured, activate it before running the application.

## 8. Run the Application

From the project root:

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## 9. Validation and Smoke Tests

The project includes a smoke-test script for basic application and inference checks.

Run:

```powershell
python src/evaluation/final_smoke_test.py
```

The previously recorded smoke-test result was:

```text
7 passed, 0 failed
```

The checks cover dataset loading, model and preprocessor loading, single-record prediction, batch prediction, required sensor validation, nonnumeric sensor handling, and inference output shape.

A passing smoke test confirms basic operational behavior; it does not establish predictive accuracy or production readiness.

## 10. Project Structure

```text
wafer-guard/
├── app.py
├── README.md
├── AUDIT.md
├── requirements.txt
├── artifacts/
│   ├── wafer_model.pt
│   └── preprocessor.joblib
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── wafer_guard_fixed.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── services/
│   └── evaluation/
└── tests/
```

Additional experiment scripts and reports may also be present in the repository.

## 11. Data and Artifact Policy

- Raw datasets are kept out of Git commits.
- Generated model artifacts are kept out of Git commits.
- The current model and preprocessor are frozen for this submission.
- Documentation cleanup must not retrain the model or overwrite its artifacts.
- Experimental scripts must be reviewed before deciding whether to include them in the final submission.

## 12. Future Improvements

- Reconcile evaluation metrics against the exact saved checkpoint and test predictions.
- Improve defect recall while controlling false alarms.
- Investigate temporal drift and sensor missingness.
- Compare model candidates using a documented validation process.
- Preserve a clearly identified held-out test evaluation for a future final candidate.
- Improve prediction explanations and monitoring.
- Add automated regression tests and clearer experiment tracking.

## 13. Conclusion

WaferGuard demonstrates an end-to-end academic workflow for sensor-data preprocessing, machine learning inference, evaluation reporting, and interactive monitoring.

The application and its basic inference workflow have passed the previously recorded smoke tests. However, the recorded model evaluation shows that defect detection remains inadequate. The project should therefore be presented honestly as an experimental prototype with documented limitations, rather than as a validated production solution.