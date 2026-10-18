
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.wafer_model import WaferAutoencoder
from src.services.prediction_service import (
    load_model,
    prepare_features,
)

DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

# Load data and reproduce the chronological split.
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

a = int(len(df) * 0.70)
b = int(len(df) * 0.80)
validation = df.iloc[a:b].copy()

y = (validation["label"] == 1).astype(int).to_numpy()

# Load the existing model and training-fitted preprocessor.
model, preprocessor, saved_threshold = load_model()
X = prepare_features(validation, preprocessor)

with torch.no_grad():
    tensor = torch.tensor(X, dtype=torch.float32)
    _, logits, _ = model(tensor)
    probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)

print("VALIDATION DIAGNOSTICS")
print("Records:", len(y))
print("Actual defects:", int(y.sum()))
print("Defect prevalence:", round(float(y.mean()), 4))
print("Saved threshold:", saved_threshold)
print("Probability min:", round(float(probabilities.min()), 6))
print("Probability max:", round(float(probabilities.max()), 6))
print("Normal mean:", round(float(probabilities[y == 0].mean()), 6))
print("Defect mean:", round(float(probabilities[y == 1].mean()), 6))
print("Normal median:", round(float(np.median(probabilities[y == 0])), 6))
print("Defect median:", round(float(np.median(probabilities[y == 1])), 6))

if len(np.unique(y)) == 2:
    print("Average precision:", round(float(average_precision_score(y, probabilities)), 6))
    print("ROC-AUC:", round(float(roc_auc_score(y, probabilities)), 6))

print("\nTHRESHOLD DIAGNOSTICS (VALIDATION ONLY)")
print("Threshold | Precision | Recall | F1 | TP | FP | FN | TN")

for threshold in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:9.3f} | "
        f"{precision_score(y, predictions, zero_division=0):9.3f} | "
        f"{recall_score(y, predictions, zero_division=0):6.3f} | "
        f"{f1_score(y, predictions, zero_division=0):.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
    )
