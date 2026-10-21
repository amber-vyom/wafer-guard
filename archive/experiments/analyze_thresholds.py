
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.80)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()


def prepare(part):
    X = part[sensor_cols].copy()

    for col in sensor_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    legacy = X["sensor_0"].eq(0.0).fillna(False)
    sensor1 = X["sensor_1"].copy()

    X.loc[legacy, "sensor_0"] = np.nan
    X["sensor_1"] = sensor1.mask(legacy, np.nan)

    extra = {"is_legacy_ps_l": legacy.astype(float)}

    for code in (100.0, 200.0, 300.0):
        extra[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    return pd.concat(
        [X, pd.DataFrame(extra, index=X.index)],
        axis=1,
    )


X_train = prepare(train)
X_val = prepare(val)

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

X_train_num = X_train[usable_cols]
X_val_num = X_val[usable_cols]

train_flags = X_train[flag_cols].to_numpy(dtype=np.float32)
val_flags = X_val[flag_cols].to_numpy(dtype=np.float32)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_num))
val_scaled = scaler.transform(imputer.transform(X_val_num))

X_train_final = np.column_stack([train_scaled, train_flags])
X_val_final = np.column_stack([val_scaled, val_flags])

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)

model.fit(X_train_final, y_train)
probabilities = model.predict_proba(X_val_final)[:, 1]

print("Validation defects:", int(y_val.sum()))
print("Validation normal wafers:", int((y_val == 0).sum()))
print("Average precision:", average_precision_score(y_val, probabilities))
print("ROC-AUC:", roc_auc_score(y_val, probabilities))

print("\nTHRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1 | TP | FP | FN | TN")

for threshold in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:2d}"
    )
