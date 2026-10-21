
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
OUTPUT_PATH = ROOT / "artifacts" / "wafer_logistic_experiment.joblib"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

a = int(len(df) * 0.70)
b = int(len(df) * 0.80)

train = df.iloc[:a].copy()
val = df.iloc[a:b].copy()
test = df.iloc[b:].copy()


def prepare(part):
    X = part[sensor_cols].apply(pd.to_numeric, errors="coerce").copy()

    legacy = X["sensor_0"].eq(0.0).fillna(False)
    sensor1 = X["sensor_1"].copy()

    X["sensor_0"] = X["sensor_0"].mask(legacy, np.nan)
    X["sensor_1"] = X["sensor_1"].mask(legacy, np.nan)
    X["is_legacy_ps_l"] = legacy.astype(float)

    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    return X


X_train = prepare(train)
X_val = prepare(val)
X_test = prepare(test)

flag_cols = [
    "is_legacy_ps_l",
    "legacy_sensor_1_code_100",
    "legacy_sensor_1_code_200",
    "legacy_sensor_1_code_300",
]

numeric_cols = [
    c for c in sensor_cols
    if c != "sensor_1"
    and X_train[c].notna().any()
    and X_train[c].nunique(dropna=True) > 1
]

usable_cols = numeric_cols + ["sensor_1"]

X_train_num = X_train[usable_cols]
X_val_num = X_val[usable_cols]
X_test_num = X_test[usable_cols]

imputer = SimpleImputer(strategy="median")

train_imputed = imputer.fit_transform(X_train_num)
val_imputed = imputer.transform(X_val_num)
test_imputed = imputer.transform(X_test_num)

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_imputed)
val_scaled = scaler.transform(val_imputed)
test_scaled = scaler.transform(test_imputed)

train_flags = X_train[flag_cols].to_numpy(dtype=np.float32)
val_flags = X_val[flag_cols].to_numpy(dtype=np.float32)
test_flags = X_test[flag_cols].to_numpy(dtype=np.float32)

X_train_final = np.column_stack([train_scaled, train_flags])
X_val_final = np.column_stack([val_scaled, val_flags])
X_test_final = np.column_stack([test_scaled, test_flags])

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()
y_test = (test["label"] == 1).astype(int).to_numpy()

model = LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    random_state=42,
)

model.fit(X_train_final, y_train)

val_prob = model.predict_proba(X_val_final)[:, 1]

print("\nVALIDATION PROBABILITY DIAGNOSTICS")
print("Normal mean probability:", round(float(val_prob[y_val == 0].mean()), 6))
print("Defect mean probability:", round(float(val_prob[y_val == 1].mean()), 6))
print("Normal median probability:", round(float(np.median(val_prob[y_val == 0])), 6))
print("Defect median probability:", round(float(np.median(val_prob[y_val == 1])), 6))
print("Average precision:", round(float(average_precision_score(y_val, val_prob)), 6))
print("ROC-AUC:", round(float(roc_auc_score(y_val, val_prob)), 6))

thresholds = (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)

print("\nTHRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1 | TP | FP | FN | TN")

for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_val, pred, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:9.2f} | "
        f"{precision_score(y_val, pred, zero_division=0):9.3f} | "
        f"{recall_score(y_val, pred, zero_division=0):6.3f} | "
        f"{f1_score(y_val, pred, zero_division=0):.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
    )

threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_val,
        (val_prob >= t).astype(int),
        zero_division=0,
    ),
)

test_prob = model.predict_proba(X_test_final)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

print("\nHELD-OUT TEST")
print("Selected threshold:", threshold)
print("Precision:", precision_score(y_test, test_pred, zero_division=0))
print("Recall:", recall_score(y_test, test_pred, zero_division=0))
print("F1:", f1_score(y_test, test_pred, zero_division=0))
print("Average precision:", average_precision_score(y_test, test_prob))
print("ROC-AUC:", roc_auc_score(y_test, test_prob))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "sensor_cols": sensor_cols,
        "usable_cols": usable_cols,
        "flag_cols": flag_cols,
        "feature_cols": usable_cols + flag_cols,
        "threshold": float(threshold),
    },
    OUTPUT_PATH,
)

print("\nExperiment saved:", OUTPUT_PATH)
