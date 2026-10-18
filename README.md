# Wafer Guard

**Wafer Guard** is an experimental semiconductor wafer anomaly-monitoring application built with PyTorch, scikit-learn, pandas, and Streamlit. It explores sensor telemetry, identifies data-quality issues, evaluates defect-detection models, and provides a local dashboard for inspecting wafers and running predictions.

> **Status:** The application and prediction pipeline are operational. Defect detection is experimental and is not suitable for production manufacturing decisions.

## 1. Project Goals

- Prevent temporal and preprocessing data leakage.
- Correct reconstruction-loss tensor shapes.
- Handle legacy PS-L sentinel values and firmware codes.
- Evaluate defects at their natural prevalence.
- Investigate sensor missingness and temporal distribution drift.
- Report measured model performance honestly.
- Preserve the existing application model while evaluating alternatives separately.

## 2. Technology Stack

- **Language:** Python
- **Machine learning:** PyTorch, scikit-learn
- **Data processing:** pandas, NumPy
- **Dashboard:** Streamlit
- **Visualizations:** Plotly
- **Model artifact loading:** joblib and PyTorch checkpoints

## 3. Dataset

The application uses the SECOM-style wafer telemetry dataset supplied as:

`data/raw/secom_telemetry_modified.csv`

The current dataset contains:

| Metric | Value |
|---|---:|
| Wafer records | 1,567 |
| Sensor columns | 590 |
| Normal records | 1,463 |
| Defective records | 104 |
| Missing sensor readings | 41,948 |

The labels are `-1` for normal records and `+1` for defective records.

The dataset is imbalanced, and some sensors contain extensive missing values or little variation.

## 4. Application Features

Run the Streamlit dashboard to access five pages:

1. **Overview** — wafer counts, sensor counts, missing readings, defect distribution, and dataset status.
2. **Dataset Explorer** — record previews, column summaries, sensor distributions, and CSV preview downloads.
3. **Live Prediction** — model probability, classification threshold, reconstruction error, and the available dataset label for reference.
4. **Model Evaluation** — the existing model's recorded evaluation metrics and confusion matrix.
5. **Audit** — data-leakage safeguards, legacy sensor handling, model-integrity notes, and current limitations.

Predictions are model outputs, not verified wafer-quality measurements.

## 5. Architecture

The main application pipeline is:

1. Load the wafer telemetry CSV.
2. Prepare sensor inputs using the saved preprocessing configuration.
3. Apply the existing saved PyTorch model.
4. Return defect probabilities, threshold-based predictions, and reconstruction errors.
5. Display the results in the Streamlit dashboard.

The prediction service is implemented in `src/services/prediction_service.py`. The dashboard entry point is `app.py`.

## 6. Installation

Use Python 3.14 or another Python version compatible with the pinned dependencies in `requirements.txt`.

From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the supplied dataset at:

`data/raw/secom_telemetry_modified.csv`

The model artifacts required for inference must also be present in the expected `artifacts/` directory.

## 7. Run the Application

From the repository root:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## 8. Data Inspection and Model Scripts

Inspect the dataset:

```powershell
python src/data/inspect_data.py
```

Run preprocessing checks:

```powershell
python src/preprocessing/prepare_data.py
```

Train the existing model using its training script:

```powershell
python src/models/train_model.py
```

**Caution:** Training scripts may write model artifacts. Do not run training against the working application artifacts unless you have reviewed the script and made backups.

## 9. Smoke Tests

The application smoke-test script is:

`src/evaluation/final_smoke_test.py`

Run it with:

```powershell
python src/evaluation/final_smoke_test.py
```

The recorded smoke-test run passed all seven checks:

- Dataset loading
- Model and preprocessor loading
- Single-record prediction
- Batch prediction
- Missing required sensor handling
- Non-numeric sensor input handling
- Prediction output shape

These tests establish basic pipeline functionality, not predictive accuracy.

## 10. Evaluation and Limitations

The chronological split used in the experiments contains:

| Split | Records | Defects |
|---|---:|---:|
| Training | 1,096 | 78 |
| Validation | 157 | 9 |
| Test | 314 | 17 |

The existing application's recorded test evaluation has recall of `0.00` and F1 of `0.00` at its selected threshold. It failed to identify the 17 defective test records in that evaluation.

A separate Top 20 sensor Random Forest experiment reported:

- Validation average precision: `0.14122`
- Test average precision: `0.06160`
- Test ROC-AUC: `0.53832`
- Test recall: `0.58824`
- Test precision: `0.06410`
- Test false positives: `146`

This candidate detected 10 of the 17 defective test records but also incorrectly flagged 146 normal records. Its performance is insufficient for practical manufacturing decisions.

The test split has already been examined during experimentation, so these results should not be described as an independent final benchmark.

## 11. Data-Quality and Drift Findings

Exploratory diagnostics found:

- 33 repeated timestamp values.
- 28 sensors with more than 50% missing values across the dataset.
- 116 sensors with zero or one observed unique value across the dataset.
- Large changes in missingness and value distributions for some sensors across chronological periods.
- Weekly defect rates that vary substantially over the observation period.

These findings suggest potential temporal or data-quality issues. They do not independently establish the cause of poor predictive performance.

The diagnostic scripts are exploratory and should not be confused with production monitoring.

## 12. Repository Structure

Important files include:

- `app.py` — Streamlit dashboard
- `src/services/prediction_service.py` — model loading and prediction pipeline
- `src/services/data_service.py` — dataset loading and summary functions
- `src/models/` — model architecture, training scripts, and experiments
- `src/evaluation/` — evaluation utilities and smoke tests
- `notebooks/wafer_guard_fixed.ipynb` — repaired analysis notebook
- `AUDIT.md` — audit notes and measured limitations
- `requirements.txt` — Python dependencies

## 13. Data and Artifact Policy

Do not commit raw datasets, evaluator-only files, virtual environments, or generated model artifacts unless explicitly authorized.

Experimental candidates should remain separate from the live model until their preprocessing, interface compatibility, and predictive performance have been verified.

## 14. Future Improvements

- Investigate temporal sensor drift and label consistency.
- Improve defect recall without generating excessive false alarms.
- Validate feature selection using time-aware evaluation.
- Add reproducible automated tests for model and preprocessing behavior.
- Track model versions and evaluation results separately.
- Re-evaluate on genuinely unseen data before considering manufacturing use.

## License and Data

Use the dataset and third-party components according to their respective licenses and distribution terms. Add a project license only if one has been selected and approved.