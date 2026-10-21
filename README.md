# WaferGuard — Wafer Defect Anomaly Detection

## 1. Overview

WaferGuard is a Python-based prototype for analyzing semiconductor wafer sensor data and identifying potentially defective records using a PyTorch model. The application provides a Streamlit interface for running inference and viewing prediction results.

The project combines sensor-data processing, feature preparation, model inference, and dashboard visualization.

**Project status:** Academic prototype. The model's evaluation results and limitations must be considered before using its predictions for any manufacturing decision.

## 2. Problem Statement

Semiconductor manufacturing generates large volumes of sensor telemetry. Identifying unusual patterns in these measurements can help flag wafers that may require further inspection.

WaferGuard demonstrates a local application workflow for analyzing sensor data and presenting model predictions. It is intended as an academic project, not a validated production inspection system.

## 3. Objectives

- Load semiconductor sensor telemetry.
- Prepare sensor features for inference.
- Apply the existing PyTorch model.
- Display predictions through a Streamlit dashboard.
- Evaluate model performance using held-out data.
- Keep application artifacts separate from experimental artifacts.
- Document limitations and reproducibility requirements.

## 4. Dataset

**Primary dataset:** `data/raw/secom_telemetry_modified.csv`

The documented dataset contains 1,567 records.

| Partition | Records | Defects | Defect prevalence |
|---|---:|---:|---:|
| Training | 1,096 | 78 | 7.12% |
| Validation | 157 | 9 | 5.73% |
| Test | 314 | 17 | 5.41% |
| **Total** | **1,567** | **104** | **6.64%** |

The documented split is chronological, with approximately 70% of records assigned to training, 10% to validation, and 20% to testing.

The documented target labels are:

- `-1`: Normal
- `+1`: Defect

The test partition contains 17 defective records. Since the dataset is imbalanced, accuracy alone is not sufficient to assess defect-detection performance.

## 5. Technology Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub

## 6. Architecture

The application follows this general workflow:

1. Load sensor telemetry from the configured CSV file.
2. Convert sensor measurements into the expected numeric representation.
3. Apply the existing preprocessing pipeline.
4. Load the saved PyTorch model and preprocessor.
5. Generate prediction scores and classifications.
6. Present the results in the Streamlit interface.
7. Use the project's evaluation utilities to inspect model behavior.

The saved model and preprocessor determine the inference behavior. This documentation does not claim that the model or preprocessing pipeline has been modified during the current repository cleanup.

## 7. Main Files and Directories

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application entry point |
| `src/models/wafer_model.py` | PyTorch model definition |
| `src/services/prediction_service.py` | Prediction and preprocessing service |
| `src/evaluation/smoke_test.py` | Standalone application smoke test |
| `notebooks/wafer_guard_fixed.ipynb` | Notebook containing the training and evaluation workflow |
| `data/raw/secom_telemetry_modified.csv` | Primary sensor dataset |
| `artifacts/wafer_model.pt` | Existing application model artifact |
| `artifacts/preprocessor.joblib` | Existing application preprocessing artifact |
| `artifacts/experiments/` | Intended location for experimental artifacts and reports |

The model and preprocessor are expected at their configured local paths. Generated artifacts are not tracked in Git, so a fresh checkout may require the authorized artifact files to be supplied separately.

The actual repository tree should be consulted to confirm the presence of individual files and directories.

## 8. Final PyTorch Model Evaluation

This section is reserved for the **final PyTorch model's evaluation results**.

Only results corresponding to the selected final PyTorch checkpoint and its identified evaluation run should be reported here. Older dashboard outputs and experimental model results from different runs are not treated as final results.

The final evaluation should identify the test dataset, checkpoint, classification threshold where applicable, and relevant metrics. Useful metrics include:

- Precision
- Recall
- F1-score
- Average precision
- ROC-AUC
- Confusion matrix

Because the dataset is imbalanced, defect recall and false alarms are important when interpreting the results.

The final metrics should be copied from the verified final PyTorch evaluation report. This README does not substitute unverified historical results for the final evaluation.

## 9. Installation and Execution

### 9.1 Prerequisites

- A compatible Python installation.
- Required project dependencies.
- The primary dataset at the configured path.
- The existing model and preprocessor artifacts at their configured local paths.

### 9.2 Activate the virtual environment

Open PowerShell in the project directory:

```powershell
cd A:\05_Wafer\wafer-guard
.\.venv\Scripts\Activate.ps1
```

If the virtual environment is unavailable, create one using a compatible Python version and install the project's required dependencies.

### 9.3 Check dependencies

```powershell
python -m pip check
```

This command checks installed package dependency compatibility. It does not verify model performance.

### 9.4 Run the standalone smoke test

```powershell
python .\src\evaluation\smoke_test.py
```

The smoke test checks selected application behaviors, including data loading, model and preprocessor loading, inference, and input validation.

The previously recorded result was seven checks passed. This refers to the standalone smoke-test script; it should not be described as seven separate Pytest tests unless a corresponding Pytest suite is verified.

### 9.5 Launch the Streamlit application

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal and click **Run Model** to view the application results.

Successful application startup does not, by itself, establish that the model reliably detects defective wafers.

## 10. Verification Status

The following checks have previously been recorded for the development environment:

- **Standalone smoke test:** Seven checks passed.
- **Dependency check:** `pip check` was reported clean.
- **Streamlit interface:** Previously confirmed working.
- **Preprocessor configuration:** The inspected local preprocessor contains 472 expected features.
- **Final model evaluation:** Final PyTorch metrics should be taken from the verified final evaluation report.
- **Production validation:** Not established.

These statements describe previously recorded checks. They should not be interpreted as proof that every environment, artifact, or model evaluation has been independently validated.

## 11. Limitations

- Model performance must be judged using the verified final PyTorch evaluation results.
- Imbalanced labels can make accuracy misleading.
- Sensor missingness and distribution changes may affect generalization.
- Preprocessing and tensor-shape checks alone do not establish reliable defect detection.
- Generated model artifacts may need to be obtained separately.
- Successful application execution does not establish suitability for production manufacturing decisions.

WaferGuard remains an academic prototype rather than a validated wafer inspection system.

## 12. Scope of the Current Cleanup

The current repository cleanup is restricted to static documentation and repository hygiene.

It does not include:

- Retraining or replacing the model.
- Changing the model architecture or inference behavior.
- Changing preprocessing logic.
- Regenerating model artifacts.
- Running training or threshold experiments.
- Claiming new model-performance results.

The existing PyTorch model and preprocessor remain unchanged during this cleanup.

## 13. Future Scope

Potential future work, outside the current static-cleanup scope, includes:

- Improving defect detection and recall.
- Investigating sensor-label alignment and anomaly-score orientation.
- Comparing alternative detection and classification methods.
- Evaluating model changes using validation data.
- Recording the exact checkpoint and metrics for each evaluation run.
- Reserving a clearly identified held-out test evaluation for the final candidate.
- Validating the system with additional representative data.
- Improving threshold selection and monitoring.

These items describe possible future development, not completed changes.

## Conclusion

WaferGuard demonstrates a local workflow combining semiconductor sensor telemetry, preprocessing, PyTorch inference, and Streamlit visualization.

The current cleanup focuses on accurate documentation and repository hygiene while preserving the existing model and its artifacts. The final PyTorch evaluation results should be reported consistently from the verified final evaluation run.

The project should be treated as an academic prototype until its defect-detection performance and suitability for its intended use are adequately validated.
## Dataset Setup

The dataset is not tracked in this repository.

Before running the notebook, obtain an authorized copy of the dataset and place it at:

`data/raw/secom_telemetry_modified.csv`

The notebook also supports `secom_telemetry_modified.csv` beside the notebook.

Do not commit private, restricted, or evaluator-only data to Git.

## Run the Notebook

Open `notebooks/wafer_guard_fixed.ipynb` in VS Code, select the project's Python environment, and run all cells from a fresh kernel.

Previously inspected test results are exploratory and must not be presented as an untouched final test estimate.

