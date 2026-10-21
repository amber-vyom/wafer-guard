from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.splits import sort_chronologically, split_bounds
from src.services.prediction_service import predict_records

path = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
df = pd.read_csv(path)

# Identify the label and timestamp columns.
normalize = lambda c: str(c).lower().replace("_", "").replace(" ", "").replace("-", "")
targets = [c for c in df.columns if normalize(c) in
           {"pass/fail", "passfail", "target", "label", "class"}]
times = [c for c in df.columns if normalize(c) in
         {"time", "timestamp", "date", "datetime"}]

if len(targets) != 1 or len(times) != 1:
    raise ValueError(f"Could not identify label/time columns. Columns: {df.columns.tolist()}")

target_col, time_col = targets[0], times[0]
# Evaluate only the held-out test partition of the shared chronological split.
df = sort_chronologically(df, time_col)
_, validation_end = split_bounds(len(df))
test = df.iloc[validation_end:].copy()
labels = pd.to_numeric(test[target_col], errors="coerce")

if labels.isna().any() or not set(labels.unique()).issubset({-1, 1}):
    raise ValueError(f"Expected labels -1 and +1; got {sorted(labels.dropna().unique())}")

y_true = (labels.to_numpy() == 1).astype(int)

# Uses the saved PyTorch model and its saved threshold.
result = predict_records(test)
scores = np.asarray(result["probabilities"], dtype=float)
pred = np.asarray(result["predictions"], dtype=int)
threshold = float(result["threshold"])

if len(pred) != len(test):
    raise RuntimeError("Prediction count differs from test record count.")

tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

print("\n=== PYTORCH FINAL TEST RESULTS ===")
print("Test records:", len(test))
print("Actual defects:", int(y_true.sum()))
print("Saved threshold:", threshold)
print("Defects detected:", int(tp))
print("Defects missed:", int(fn))
print("False alarms:", int(fp))
print("Correctly unflagged normal records:", int(tn))
print(f"Recall: {recall_score(y_true, pred, zero_division=0):.4f}")
print(f"Precision: {precision_score(y_true, pred, zero_division=0):.4f}")
print(f"F1: {f1_score(y_true, pred, zero_division=0):.4f}")

if len(np.unique(y_true)) == 2:
    print(f"ROC-AUC: {roc_auc_score(y_true, scores):.4f}")
    print(f"Average precision: {average_precision_score(y_true, scores):.4f}")

print("\nNo model artifacts were changed.")
