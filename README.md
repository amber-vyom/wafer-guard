# WaferGuard — Wafer Defect Anomaly Detection

## 1. Overview
WaferGuard is a Python-based prototype for analyzing semiconductor wafer sensor data and identifying potentially defective records using machine learning. The application provides a Streamlit interface with a Run Model button, results table, and graph.

## 2. Problem Statement
Semiconductor manufacturing generates large amounts of sensor telemetry. Detecting unusual patterns can help identify wafers that require further inspection. WaferGuard demonstrates a machine-learning workflow for analyzing sensor measurements and displaying prediction results.

## 3. Objectives
- Load and preprocess sensor telemetry.
- Apply a PyTorch-based anomaly detection model.
- Display predictions through a Streamlit dashboard.
- Evaluate model performance using held-out test data.
- Keep experimental model artifacts separate from application artifacts.

## 4. Dataset
File: `data/raw/secom_telemetry_modified.csv`

- Total records: 1,567
- Normal label: -1
- Defect label: +1
- Training: 1,096 records, 78 defects
- Validation: 157 records, 9 defects
- Testing: 314 records, 17 defects

The data is split chronologically into training, validation, and test sets using a 70/10/20 split.

## 5. Technology Stack
- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib
- Pytest

## 6. Architecture
1. Load sensor data from CSV.
2. Convert numeric values and handle missing measurements.
3. Scale sensor features and construct additional sensor flags.
4. Load the saved PyTorch model and preprocessor.
5. Generate prediction scores and classifications.
6. Display results in the Streamlit interface.
7. Evaluate the model using separate evaluation scripts.

## 7. Main Files
- `app.py` — Streamlit application.
- `src/models/wafer_model.py` — PyTorch model definition.
- `src/services/prediction_service.py` — Prediction service.
- `artifacts/wafer_model.pt` — Existing application model.
- `artifacts/preprocessor.joblib` — Existing application preprocessor.
- `artifacts/experiments/` — Experimental models and reports.

## 8. Existing Model Evaluation
Test records: 314
Actual defects: 17

| Metric | Result |
|---|---:|
| Defects detected | 2 of 17 |
| Defects missed | 15 |
| False alarms | 17 |
| Precision | 10.53% |
| Recall | 11.76% |
| F1-score | 11.11% |
| ROC-AUC | 0.6599 |

A separate experimental candidate detected 0 of 17 test defects and was not promoted. The existing PyTorch model remains the application model.

## 9. Installation and Execution
Activate the existing virtual environment and start Streamlit:

```powershell
cd A:\05_Wafer\wafer-guard
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL shown in the terminal. Click **Run Model** to view the results.

## 10. Verification
- Automated tests: 7 passed.
- Dependency check: no broken requirements.
- Existing model and preprocessor files: verified present.
- Experimental artifacts: saved separately.
- Streamlit interface: previously confirmed working.

Seven Pytest warnings were reported because smoke-test functions return strings instead of `None`. They did not cause test failures.

## 11. Limitations
The existing model detected only 2 of the 17 actual test defects. Its defect-detection performance is weak, and it is not suitable for production manufacturing decisions without further development and validation.

## 12. Future Scope
- Improve defect detection and recall.
- Investigate sensor-label alignment and anomaly-score orientation.
- Compare alternative detection and classification methods.
- Validate with additional representative data.
- Improve threshold selection and model monitoring.

## Conclusion
WaferGuard demonstrates a working prototype that combines sensor-data preprocessing, PyTorch inference, and Streamlit visualization. The application passes its current automated tests, but the model requires further improvement before it can be considered reliable for real manufacturing use.
