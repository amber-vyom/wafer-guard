
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
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
EXPERIMENT_PATH = ROOT / "artifacts" / "wafer_rf_experiment.joblib"

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
    X = part[sensor_cols].copy()

    for column in sensor_cols:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    legacy = X["sensor_0"].eq(0.0).fillna(False)
    sensor1 = X["sensor_1"].copy()

    X.loc[legacy, "sensor_0"] = np.nan
    X["sensor_1"] = sensor1.mask(legacy, np.nan)

    X["is_legacy_ps_l"] = legacy.astype(float)

    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    return X


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

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

val_prob = pipeline.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.05)

threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_val,
        (val_prob >= t).astype(int),
        zero_division=0,
    ),
)

val_pred = (val_prob >= threshold).astype(int)

print("\nVALIDATION RESULTS")
print("Validation defects:", int(y_val.sum()))
print("Predicted defects:", int(val_pred.sum()))
print("Selected threshold:", float(threshold))
print("Validation precision:", precision_score(y_val, val_pred, zero_division=0))
print("Validation recall:", recall_score(y_val, val_pred, zero_division=0))
print("Validation F1:", f1_score(y_val, val_pred, zero_division=0))
print("Validation confusion matrix:")
print(confusion_matrix(y_val, val_pred))

test_prob = pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

print("\nHELD-OUT TEST RESULTS")
print("Test defects:", int(y_test.sum()))
print("Test prevalence:", round(float(y_test.mean()), 4))
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, zero_division=0))
print("Recall:", recall_score(y_test, test_pred, zero_division=0))
print("F1:", f1_score(y_test, test_pred, zero_division=0))
print("Average precision:", average_precision_score(y_test, test_prob))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", roc_auc_score(y_test, test_prob))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))

EXPERIMENT_PATH.parent.mkdir(exist_ok=True)

joblib.dump({
    "pipeline": pipeline,
    "sensor_cols": sensor_cols,
    "feature_cols": feature_cols,
    "threshold": float(threshold),
}, EXPERIMENT_PATH)

print("\nExperiment saved:", EXPERIMENT_PATH)
