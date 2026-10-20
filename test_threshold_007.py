from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.services.prediction_service import (
    load_final_model,
    prepare_final_features,
)

DATA_PATH = Path("data/raw/secom_telemetry_modified.csv")
THRESHOLD = 0.07

df = pd.read_csv(DATA_PATH)

# Identify the target column without guessing silently.
target_candidates = [
    c for c in df.columns
    if c.lower().replace(" ", "").replace("_", "") in
    {"pass/fail", "passfail", "target", "label", "class"}
]

if len(target_candidates) != 1:
    print("Could not identify exactly one target column.")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

target_col = target_candidates[0]

# Identify a timestamp column for the chronological split.
time_candidates = [
    c for c in df.columns
    if c.lower() in {"time", "timestamp", "date", "datetime"}
]

if len(time_candidates) != 1:
    print("Could not identify exactly one timestamp column.")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

time_col = time_candidates[0]
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

if df[time_col].isna().any():
    print("Some timestamps could not be parsed. Stopping to avoid a wrong split.")
    sys.exit(1)

df = df.sort_values(time_col).reset_index(drop=True)

# Chronological 70% train / 10% validation / 20% test.
n = len(df)
validation_start = int(n * 0.70)
test_start = int(n * 0.80)
validation = df.iloc[validation_start:test_start].copy()

# SECOM labels: -1 = normal, +1 = defect.
y_raw = pd.to_numeric(validation[target_col], errors="coerce")
if y_raw.isna().any() or not set(y_raw.unique()).issubset({-1, 1}):
    print("Unexpected target labels:", sorted(y_raw.dropna().unique().tolist()))
    sys.exit(1)

y_true = (y_raw.to_numpy() == 1).astype(int)

artifact = load_final_model()
features = prepare_final_features(validation, artifact)
probabilities = artifact["model"].predict_proba(features)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

tp = int(((predictions == 1) & (y_true == 1)).sum())
fn = int(((predictions == 0) & (y_true == 1)).sum())
fp = int(((predictions == 1) & (y_true == 0)).sum())
tn = int(((predictions == 0) & (y_true == 0)).sum())

print("\n=== VALIDATION RESULTS: THRESHOLD 0.07 ===")
print("Model:", artifact.get("model_name", "Unknown"))
print("Threshold tested:", THRESHOLD)
print("Validation records:", len(validation))
print("Actual defects:", int(y_true.sum()))
print("Defects detected:", tp)
print("Defects missed:", fn)
print("False alarms:", fp)
print("True negatives:", tn)
print(f"Recall: {tp / (tp + fn):.1%}" if tp + fn else "Recall: N/A")
print(f"Precision: {tp / (tp + fp):.1%}" if tp + fp else "Precision: 0.0%")
print("\nNo project files or saved model artifacts were modified.")
