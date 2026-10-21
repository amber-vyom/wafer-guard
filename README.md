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
| Validation | 235 | 17 | 7.23% |
| Test | 236 | 9 | 3.81% |
| **Total** | **1,567** | **104** | **6.64%** |

The split is chronological: the earliest 70% of records (ordered by timestamp, stable sort) are training, the next 15% are validation, and the latest 15% are the held-out test partition. It is defined once in `src/data/splits.py` and shared by the notebook, the Streamlit app, and the evaluation scripts, so every reported number refers to the same rows.

The saved model in `artifacts/` was trained under an earlier 70/10/20 split (first 70% for training), so the final 15% is also unseen by it.

The documented target labels are:

- `-1`: Normal
- `+1`: Defect

The test partition contains 9 defective records, so any test metric carries wide uncertainty. Since the dataset is imbalanced, accuracy alone is not sufficient to assess defect-detection performance.

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
| `src/services/prediction_service.py` | Prediction and preprocessing service; validates the artifacts it loads |
| `src/data/splits.py` | The single chronological train/validation/test split definition |
| `tests/` | Pytest suite (synthetic data only; does not need the dataset or artifacts) |
| `scripts/training/`, `scripts/evaluation/`, `scripts/maintenance/` | Gradient-boosting trainer, evaluation and threshold scripts, preprocessor metadata repair |
| `archive/experiments/` | Retired one-off experiments (not maintained) |
| `src/evaluation/smoke_test.py` | Standalone application smoke test |
| `notebooks/wafer_guard_fixed.ipynb` | Training and evaluation workflow; saves `artifacts/wafer_model.pt` and `artifacts/preprocessor.joblib` in the format the app loads |
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

The recorded result is seven checks passed. The smoke test is a standalone script, separate from the pytest suite below.

### 9.5 Run the pytest suite

```powershell
python -m pytest
```

The suite uses small synthetic data. It covers the split, the legacy-sensor feature rules, the model class, and train/serve parity (what the notebook saves is what the app loads). It does not evaluate model quality.

### 9.6 Launch the Streamlit application

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal and click **Run Model** to view the application results.

Successful application startup does not, by itself, establish that the model reliably detects defective wafers.

### 9.7 Regenerate artifacts and reports

| Goal | Command |
|---|---|
| Retrain the PyTorch model and preprocessor | Run all cells of `notebooks/wafer_guard_fixed.ipynb` from a fresh kernel (backs up existing artifacts as timestamped `.bak` files, then checks app parity) |
| Add missing metadata to an older `preprocessor.joblib` | `python scripts/maintenance/repair_preprocessor.py` |
| Retrain the gradient-boosting candidate | `python scripts/training/train_model.py` |
| Evaluate the saved PyTorch model on the test partition | `python scripts/evaluation/evaluate_model.py` |
| Write the gradient-boosting evaluation report | `python scripts/evaluation/generate_evaluation_report.py` |
| Threshold diagnostic on the validation partition | `python scripts/evaluation/validate_threshold.py` |

Set `WAFERGUARD_ARTIFACT_DIR` before running the notebook to write the new artifacts somewhere other than `artifacts/`.

## 10. Verification Status

The following checks have previously been recorded for the development environment:

- **Standalone smoke test:** Seven checks passed.
- **Dependency check:** `pip check` was reported clean.
- **Pytest suite:** 23 tests passed (synthetic data; does not measure model quality).
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

The current repository cleanup is restricted to static changes: documentation, repository layout, a shared split and model definition, tests, and aligning the notebook's saved artifacts with the app's loader.

It does not include:

- Retraining or replacing the model.
- Changing the model architecture or inference behavior.
- Changing preprocessing logic.
- Regenerating model artifacts.
- Running training or threshold experiments.
- Claiming new model-performance results.

The saved model weights and the preprocessor's fitted imputer and scaler are unchanged. The only artifact edit was adding the missing `sensor_cols` and `feature_cols` metadata to `preprocessor.joblib` with `scripts/maintenance/repair_preprocessor.py`; the original was kept as `artifacts/preprocessor_before_metadata_fix.joblib`.

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

The notebook trains the same `WaferAutoencoder` class the app serves and saves `artifacts/wafer_model.pt` and `artifacts/preprocessor.joblib`. Any existing copies are first backed up with a timestamp, and the notebook finishes by reloading the saved files through the app's inference code and checking that features, scores and alerts match.

Previously inspected test results are exploratory and must not be presented as an untouched final test estimate.

