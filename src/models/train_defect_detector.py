from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUT = ROOT / "artifacts" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensors = [c for c in df.columns if c.startswith("sensor_")]
n = len(df)
a, b = int(n * 0.70), int(n * 0.80)
train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()
y_test = (test["label"] == 1).astype(int).to_numpy()

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

model = RandomForestClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
print("Train defects:", int(y_train.sum()))
print("Validation defects:", int(y_val.sum()))
print("Test defects:", int(y_test.sum()))
print("Validation AP:", round(average_precision_score(y_val, val_prob), 4))

# Threshold is chosen on validation data only.
candidates = np.unique(np.r_[val_prob, 0.0, 1.0])
choices = []
for threshold0 in candidates:
    pred = (val_prob >= threshold0).astype(int)
    recall = recall_score(y_val, pred, zero_division=0)
    precision = precision_score(y_val, pred, zero_division=0)
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if (4 * precision + recall) > 0 else 0.0
    )
    if recall >= 0.70:
        choices.append((f2, precision, recall, float(threshold0)))

if choices:
    _, _, _, threshold = max(choices)
else:
    threshold = min(
        candidates,
        key=lambda t: (
            -recall_score(y_val, (val_prob >= t).astype(int), zero_division=0),
            -f1_score(y_val, (val_prob >= t).astype(int), zero_division=0),
        ),
    )

print("Selected validation threshold:", float(threshold))
print("Validation recall:", recall_score(
    y_val, (val_prob >= threshold).astype(int), zero_division=0
))

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

results = {
    "threshold": float(threshold),
    "test_records": int(len(y_test)),
    "test_defects": int(y_test.sum()),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "average_precision": float(average_precision_score(y_test, test_prob)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
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
    },
    OUT / "random_forest_defect_candidate.joblib",
)
(OUT / "random_forest_defect_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print("\nCandidate saved separately. Existing app model unchanged.")
