
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUTPUT_DIR = ROOT / "artifacts" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a label column.")

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

df = df[df["label"].isin([-1, 1])].copy().reset_index(drop=True)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_train_raw = train[sensor_cols].apply(pd.to_numeric, errors="coerce")
X_val_raw = val[sensor_cols].apply(pd.to_numeric, errors="coerce")
X_test_raw = test[sensor_cols].apply(pd.to_numeric, errors="coerce")

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()
y_test = (test["label"] == 1).astype(int).to_numpy()

missing_rate = X_train_raw.isna().mean()
unique_count = X_train_raw.nunique(dropna=True)

eligible = [
    c for c in sensor_cols
    if unique_count[c] >= 2 and missing_rate[c] <= 0.50
]

X_train_raw = X_train_raw[eligible]
X_val_raw = X_val_raw[eligible]
X_test_raw = X_test_raw[eligible]

imputer = SimpleImputer(strategy="median")
X_train_all = imputer.fit_transform(X_train_raw)
X_val_all = imputer.transform(X_val_raw)
X_test_all = imputer.transform(X_test_raw)

valid_indices = [
    i for i in range(X_train_all.shape[1])
    if np.isfinite(X_train_all[:, i]).all()
    and np.unique(X_train_all[:, i]).size >= 2
]

X_train_all = X_train_all[:, valid_indices]
X_val_all = X_val_all[:, valid_indices]
X_test_all = X_test_all[:, valid_indices]
eligible = [eligible[i] for i in valid_indices]

mi = mutual_info_classif(
    X_train_all,
    y_train,
    random_state=42
)

selected_indices = np.argsort(mi)[::-1][:20]
selected_sensors = [eligible[i] for i in selected_indices]

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_all[:, selected_indices], y_train)

val_prob = model.predict_proba(
    X_val_all[:, selected_indices]
)[:, 1]

test_prob = model.predict_proba(
    X_test_all[:, selected_indices]
)[:, 1]

# Choose a threshold on validation data only.
# Among thresholds reaching at least 70% validation recall,
# choose the one with the highest precision.
threshold_candidates = np.unique(val_prob)
valid_thresholds = []

for threshold in threshold_candidates:
    pred = (val_prob >= threshold).astype(int)
    recall = recall_score(y_val, pred, zero_division=0)

    if recall >= 0.70:
        precision = precision_score(y_val, pred, zero_division=0)
        valid_thresholds.append((precision, threshold, recall))

if valid_thresholds:
    _, threshold, validation_recall = max(
        valid_thresholds,
        key=lambda item: (item[0], item[1])
    )
else:
    threshold = 0.5
    validation_recall = recall_score(
        y_val,
        (val_prob >= threshold).astype(int),
        zero_division=0
    )

test_pred = (test_prob >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

results = {
    "features": 20,
    "selected_sensors": selected_sensors,
    "threshold_selected_on_validation": float(threshold),
    "validation_average_precision": float(
        average_precision_score(y_val, val_prob)
    ),
    "validation_roc_auc": float(roc_auc_score(y_val, val_prob)),
    "validation_recall_at_threshold": float(validation_recall),
    "test_average_precision": float(
        average_precision_score(y_test, test_prob)
    ),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_precision": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "test_confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    },
    "test_records": int(len(y_test)),
    "test_defects": int(y_test.sum()),
    "warning": (
        "Test metrics are exploratory because prior experiments have "
        "already examined this dataset's test split. Do not describe "
        "these results as an independent final evaluation."
    ),
}

result_path = OUTPUT_DIR / "top20_evaluation.json"
result_path.write_text(json.dumps(results, indent=2))

print("=" * 60)
print("TOP 20 SENSOR MODEL EVALUATION")
print("=" * 60)
print(f"Selected features: {len(selected_sensors)}")
print(f"Validation AP: {results['validation_average_precision']:.5f}")
print(f"Validation ROC-AUC: {results['validation_roc_auc']:.5f}")
print(f"Threshold selected on validation: {threshold:.6f}")
print(f"Test AP: {results['test_average_precision']:.5f}")
print(f"Test ROC-AUC: {results['test_roc_auc']:.5f}")
print(f"Test precision: {results['test_precision']:.5f}")
print(f"Test recall: {results['test_recall']:.5f}")
print(f"Test F1: {results['test_f1']:.5f}")
print(f"Test confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Saved: {result_path}")
print("Existing application model was not changed.")
