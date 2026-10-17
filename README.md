
# Wafer Guard

An audit and repair of a PyTorch wafer anomaly detector for semiconductor manufacturing telemetry.

## Project Goals

- Prevent temporal and preprocessing leakage.
- Correct reconstruction-loss tensor shapes.
- Handle legacy PS-L sentinel values and firmware codes.
- Evaluate defects at their natural prevalence.
- Document measured model performance honestly.

## Requirements

Python 3.14 or a compatible Python version, with packages listed in `requirements.txt`.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the supplied dataset at:

`data/raw/secom_telemetry_modified.csv`

## Run

Inspect the dataset:

```powershell
python src/data/inspect_data.py
```

Run preprocessing checks:

```powershell
python src/preprocessing/prepare_data.py
```

Train and evaluate the model:

```powershell
python src/models/train_model.py
```

## Current Results

The chronological test set contains 314 wafers, including 17 defects. The current model has F1 = 0.00 and recall = 0.00 at the validation-selected threshold. It is not production-ready.

## Deliverables

- `notebooks/wafer_guard_fixed.ipynb`
- `AUDIT.md`
- Source code under `src/`

## Data and Artifacts

Do not commit raw datasets, evaluator-only files, virtual environments, or generated model artifacts unless explicitly authorized.
