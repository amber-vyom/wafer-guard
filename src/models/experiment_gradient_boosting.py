
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

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
    raise RuntimeError("No usable features selected.")

X_train = train[selected].apply(pd.to_numeric, errors="coerce")
X_val = val[selected].apply(pd.to_numeric, errors="coerce")
X_test = test[selected].apply(pd.to_numeric, errors="coerce")

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Compare the baseline with HistGradientBoosting.
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    ),
}

fitted = {}
for name, model in models.items():
    print("Training:", name)

    if name == "HistGradientBoosting":
        # Balance the rare defect class without duplicating observations.
        weights = np.where(
            y_train == 1,
            (y_train == 0).sum() / max((y_train == 1).sum(), 1),
            1.0,
        )
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_prob)
    auc = roc_auc_score(y_val, val_prob)

    fitted[name] = {
        "model": model,
        "validation_prob": val_prob,
        "ap": float(ap),
        "auc": float(auc),
    }

    print(f"{name}: validation AP={ap:.5f}, ROC-AUC={auc:.5f}")

best_name = max(fitted, key=lambda name: fitted[name]["ap"])
best = fitted[best_name]
print("\nSelected by validation average precision:", best_name)

# Choose threshold on validation data only.
from sklearn.metrics import precision_score, recall_score, confusion_matrix

val_prob = best["validation_prob"]
thresholds = np.unique(np.r_[np.arange(0.01, 0.51, 0.01), val_prob])
eligible = []

for t in thresholds:
    pred = (val_prob >= t).astype(int)
    recall = recall_score(y_val, pred, zero_division=0)
    precision = precision_score(y_val, pred, zero_division=0)
    if recall >= 0.80:
        eligible.append((precision, float(t), recall))

if eligible:
    _, threshold, val_recall = max(eligible, key=lambda item: (item[0], item[2], item[1]))
else:
    fallback = []
    for t in thresholds:
        pred = (val_prob >= t).astype(int)
        recall = recall_score(y_val, pred, zero_division=0)
        precision = precision_score(y_val, pred, zero_division=0)
        fallback.append((recall, precision, float(t)))
    val_recall, _, threshold = max(fallback)

print("Validation threshold:", round(threshold, 6))
print("Validation recall:", round(val_recall, 4))

# One test evaluation for the selected candidate.
test_prob = best["model"].predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

results = {
    "selected_model": best_name,
    "validation_average_precision": best["ap"],
    "validation_roc_auc": best["auc"],
    "threshold_selected_on_validation": float(threshold),
    "validation_recall": float(val_recall),
    "test_records": int(len(y_test)),
    "test_defects": int(y_test.sum()),
    "test_average_precision": float(average_precision_score(y_test, test_prob)),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "test_true_positives": int(tp),
    "test_false_positives": int(fp),
    "test_false_negatives": int(fn),
    "test_true_negatives": int(tn),
    "test_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
}

print("\nRESULTS")
print(json.dumps(results, indent=2))

joblib.dump(
    {
        "model": best["model"],
        "imputer": imputer,
        "selected_sensors": selected,
        "threshold": float(threshold),
        "model_name": best_name,
    },
    OUT / "gradient_boosting_candidate.joblib",
)

(OUT / "gradient_boosting_results.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

print("\nCandidate saved separately; live application model unchanged.")
