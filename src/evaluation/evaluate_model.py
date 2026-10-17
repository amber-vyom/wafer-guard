
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.services.prediction_service import (
    load_model,
    prepare_features,
    predict_anomaly,
)
from src.services.data_service import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = load_dataset(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

split_index = int(len(df) * 0.80)
test = df.iloc[split_index:].copy()

y_test = (test["label"] == 1).astype(int).to_numpy()

model, preprocessor, threshold = load_model()
features = prepare_features(test, preprocessor)
results = predict_anomaly(model, features, threshold)

probabilities = np.asarray(results["probabilities"])
predictions = np.asarray(results["predictions"])
errors = np.asarray(results["reconstruction_errors"])

print("\nNEURAL MODEL EVALUATION")
print("Test records:", len(test))
print("Actual defects:", int(y_test.sum()))
print("Defect prevalence:", round(float(y_test.mean()), 4))
print("Classification threshold:", threshold)

print("\nPROBABILITY COMPARISON")
print(
    "Normal mean probability:",
    round(float(probabilities[y_test == 0].mean()), 6),
)
print(
    "Defect mean probability:",
    round(float(probabilities[y_test == 1].mean()), 6),
)
print(
    "Normal median probability:",
    round(float(np.median(probabilities[y_test == 0])), 6),
)
print(
    "Defect median probability:",
    round(float(np.median(probabilities[y_test == 1])), 6),
)

print("\nRECONSTRUCTION ERROR COMPARISON")
print(
    "Normal mean error:",
    round(float(errors[y_test == 0].mean()), 6),
)
print(
    "Defect mean error:",
    round(float(errors[y_test == 1].mean()), 6),
)

print("\nRANKING METRICS")
print("Average precision:", round(float(average_precision_score(y_test, probabilities)), 6))
print("ROC-AUC:", round(float(roc_auc_score(y_test, probabilities)), 6))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Defective"],
        zero_division=0,
    )
)

print("\nTHRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1 | TP | FP | FN | TN")

for value in (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
    predicted = (probabilities >= value).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test, predicted, labels=[0, 1]
    ).ravel()

    from sklearn.metrics import precision_score, recall_score, f1_score

    print(
        f"{value:9.2f} | "
        f"{precision_score(y_test, predicted, zero_division=0):9.3f} | "
        f"{recall_score(y_test, predicted, zero_division=0):6.3f} | "
        f"{f1_score(y_test, predicted, zero_division=0):.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
    )
