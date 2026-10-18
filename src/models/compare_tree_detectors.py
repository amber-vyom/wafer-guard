
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUT = ROOT / "artifacts" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensors = [c for c in df.columns if c.startswith("sensor_")]
n = len(df)
a, b = int(n * 0.70), int(n * 0.80)
train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()
y_test = (test["label"] == 1).astype(int).to_numpy()

# Select features using training data only.
scores = []
for col in sensors:
    normal = pd.to_numeric(
        train.loc[train["label"] == -1, col], errors="coerce"
    ).dropna()
    defect = pd.to_numeric(
        train.loc[train["label"] == 1, col], errors="coerce"
    ).dropna()

    if len(normal) < 20 or len(defect) < 10:
        continue

    std = normal.std()
    if not np.isfinite(std) or std <= 0:
        continue

    score = abs(defect.median() - normal.median()) / std
    if np.isfinite(score):
        scores.append((col, float(score)))

scores.sort(key=lambda item: item[1], reverse=True)
selected = [col for col, _ in scores[:50]]

if not selected:
    raise RuntimeError("No usable sensors selected.")

X_train = train[selected].apply(pd.to_numeric, errors="coerce")
X_val = val[selected].apply(pd.to_numeric, errors="coerce")
X_test = test[selected].apply(pd.to_numeric, errors="coerce")

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

fitted = {}
print("Training and comparing models on validation data...")

for name, model in models.items():
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_prob)
    auc = roc_auc_score(y_val, val_prob)

    fitted[name] = {
        "model": model,
        "validation_probabilities": val_prob,
        "validation_ap": float(ap),
        "validation_auc": float(auc),
    }

    print(f"{name}: validation AP={ap:.5f}, ROC-AUC={auc:.5f}")

# Choose model by validation average precision only.
best_name = max(
    fitted,
    key=lambda name: fitted[name]["validation_ap"],
)
best = fitted[best_name]
val_prob = best["validation_probabilities"]

print("\nSelected using validation AP:", best_name)

# Choose threshold on validation data to maximize precision
# while catching at least 80% of validation defects.
threshold_candidates = np.unique(
    np.r_[np.arange(0.01, 0.51, 0.01), val_prob]
)
eligible = []

for threshold in threshold_candidates:
    pred = (val_prob >= threshold).astype(int)
    precision = precision_score(y_val, pred, zero_division=0)
    recall = recall_score(y_val, pred, zero_division=0)

    if recall >= 0.80:
        eligible.append((precision, float(threshold), recall))

if eligible:
    _, threshold, val_recall = max(eligible, key=lambda x: (x[0], x[2], x[1]))
else:
    choices = []
    for threshold0 in threshold_candidates:
        pred = (val_prob >= threshold0).astype(int)
        recall = recall_score(y_val, pred, zero_division=0)
        precision = precision_score(y_val, pred, zero_division=0)
        choices.append((recall, precision, float(threshold0)))
    val_recall, _, threshold = max(choices)

print("Validation threshold:", round(threshold, 6))
print("Validation recall:", round(val_recall, 4))

# Evaluate the selected model and threshold on test data.
model = best["model"]
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

results = {
    "selected_model": best_name,
    "validation_average_precision": best["validation_ap"],
    "validation_roc_auc": best["validation_auc"],
    "threshold_selected_on_validation": float(threshold),
    "validation_recall": float(val_recall),
    "test_records": int(len(y_test)),
    "test_defects": int(y_test.sum()),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "average_precision": float(average_precision_score(y_test, test_prob)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_negatives": int(tn),
    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
}

print("\nTEST RESULTS")
print(json.dumps(results, indent=2))

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "selected_sensors": selected,
        "threshold": float(threshold),
        "selected_model": best_name,
    },
    OUT / "tree_detector_candidate.joblib",
)

(OUT / "tree_detector_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

print("\nCandidate saved separately.")
print("Existing application artifacts were not modified.")
