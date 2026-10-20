
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
EXPERIMENT_PATH = ROOT / "artifacts" / "experiments" / "wafer_rf_recall_candidate.joblib"
RESULT_PATH = ROOT / "artifacts" / "experiments" / "wafer_rf_recall_results.json"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()


def prepare_raw(part):
    X = part[sensor_cols].apply(pd.to_numeric, errors="coerce").copy()

    legacy = X["sensor_0"].eq(0.0).fillna(False)
    sensor1 = X["sensor_1"].copy()

    X.loc[legacy, "sensor_0"] = np.nan
    X["sensor_1"] = sensor1.mask(legacy, np.nan)

    flags = pd.DataFrame(
        {
            "is_legacy_ps_l": legacy.astype(float),
            "legacy_sensor_1_code_100": (
                legacy & sensor1.eq(100.0)
            ).astype(float),
            "legacy_sensor_1_code_200": (
                legacy & sensor1.eq(200.0)
            ).astype(float),
            "legacy_sensor_1_code_300": (
                legacy & sensor1.eq(300.0)
            ).astype(float),
        },
        index=X.index,
    )

    return pd.concat([X, flags], axis=1)


X_train = prepare_raw(train_df)
X_val = prepare_raw(val_df)
X_test = prepare_raw(test_df)

y_train = (train_df["label"] == 1).astype(int).to_numpy()
y_val = (val_df["label"] == 1).astype(int).to_numpy()
y_test = (test_df["label"] == 1).astype(int).to_numpy()

numeric_cols = [c for c in sensor_cols if c != "sensor_1"]

usable_cols = [
    c for c in numeric_cols
    if X_train[c].notna().any()
    and X_train[c].nunique(dropna=True) > 1
]
usable_cols.append("sensor_1")

flag_cols = [
    "is_legacy_ps_l",
    "legacy_sensor_1_code_100",
    "legacy_sensor_1_code_200",
    "legacy_sensor_1_code_300",
]

feature_cols = usable_cols + flag_cols

X_train = X_train[feature_cols]
X_val = X_val[feature_cols]
X_test = X_test[feature_cols]

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Training classifier...")
pipeline.fit(X_train, y_train)

val_prob = pipeline.predict_proba(X_val)[:, 1]

print("\nVALIDATION THRESHOLD SEARCH")
print("Threshold | Precision | Recall | F2 | TP | FP | FN | TN")

thresholds = np.unique(
    np.concatenate(
        [
            np.arange(0.01, 0.51, 0.01),
            val_prob,
        ]
    )
)

eligible = []

for threshold_candidate in thresholds:
    val_pred_candidate = (
        val_prob >= threshold_candidate
    ).astype(int)

    precision = precision_score(
        y_val, val_pred_candidate, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred_candidate, zero_division=0
    )
    f2 = fbeta_score(
        y_val, val_pred_candidate, beta=2, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, val_pred_candidate, labels=[0, 1]
    ).ravel()

    if recall >= 0.80:
        eligible.append(
            (precision, f2, float(threshold_candidate), recall)
        )

    if any(
        abs(threshold_candidate - value) < 1e-9
        for value in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    ):
        print(
            f"{threshold_candidate:9.3f} | "
            f"{precision:9.3f} | {recall:6.3f} | {f2:.3f} | "
            f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
        )

if eligible:
    # Prioritize precision, subject to at least 80% validation recall.
    precision, f2, threshold, val_recall = max(
        eligible, key=lambda item: (item[0], item[1], item[2])
    )
else:
    # If 80% recall is impossible, prioritize validation recall first.
    fallback = []
    for t in thresholds:
        pred = (val_prob >= t).astype(int)
        recall = recall_score(y_val, pred, zero_division=0)
        precision = precision_score(y_val, pred, zero_division=0)
        f2 = fbeta_score(y_val, pred, beta=2, zero_division=0)
        fallback.append((recall, precision, f2, float(t)))

    val_recall, precision, f2, threshold = max(fallback)

print("\nVALIDATION RESULTS")
print("Validation defects:", int(y_val.sum()))
print("Validation AP:", round(average_precision_score(y_val, val_prob), 5))
print("Selected threshold:", float(threshold))
print("Validation precision:", round(float(precision), 5))
print("Validation recall:", round(float(val_recall), 5))
print("Validation F2:", round(float(f2), 5))

# Evaluate the selected threshold on the chronological test partition.
test_prob = pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

results = {
    "threshold_selected_on_validation": float(threshold),
    "validation_recall": float(val_recall),
    "test_records": int(len(y_test)),
    "test_defects": int(y_test.sum()),
    "accuracy": float(accuracy_score(y_test, test_pred)),
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

print("\nHELD-OUT TEST RESULTS")
print(pd.Series(results).to_string())

EXPERIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "pipeline": pipeline,
        "sensor_cols": sensor_cols,
        "feature_cols": feature_cols,
        "threshold": float(threshold),
    },
    EXPERIMENT_PATH,
)

RESULT_PATH.write_text(
    __import__("json").dumps(results, indent=2),
    encoding="utf-8",
)

print("\nCandidate saved:", EXPERIMENT_PATH)
print("Results saved:", RESULT_PATH)
print("Existing application artifacts were not modified.")
