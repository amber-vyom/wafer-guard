from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUT = ROOT / "artifacts" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
flag_cols = [
    "is_legacy_ps_l",
    "legacy_sensor_1_code_100",
    "legacy_sensor_1_code_200",
    "legacy_sensor_1_code_300",
]

def get_features(part):
    X = part[sensor_cols].apply(pd.to_numeric, errors="coerce").copy()
    legacy = X["sensor_0"].eq(0.0).fillna(False)
    sensor1 = X["sensor_1"].copy()
    X["is_legacy_ps_l"] = legacy.astype(np.float32)
    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(np.float32)
    X.loc[legacy, "sensor_0"] = np.nan
    X["sensor_1"] = sensor1.mask(legacy, np.nan)
    y = (part["label"] == 1).astype(int).to_numpy()
    return X, y

# Fit on the original training partition only.
train_end = int(len(df) * 0.70)
X_raw, y = get_features(df.iloc[:train_end])
sensor_cols_used = [
    c for c in sensor_cols
    if X_raw[c].notna().any() and X_raw[c].nunique(dropna=True) > 1
]

imputer = SimpleImputer(strategy="median")
X_num = imputer.fit_transform(X_raw[sensor_cols_used])
X = np.column_stack([
    X_num, X_raw[flag_cols].to_numpy(dtype=np.float32)
])

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.05,
    max_leaf_nodes=7,
    l2_regularization=5.0,
    class_weight="balanced",
    random_state=42,
)
model.fit(X, y)

artifact = {
    "model": model,
    "imputer": imputer,
    "sensor_cols": sensor_cols,
    "usable_cols": sensor_cols_used,
    "flag_cols": flag_cols,
    "input_dim": X.shape[1],
    "threshold": 0.5,
    "model_name": "Tuned Gradient Boosting",
    "parameters": {
        "max_iter": 100,
        "learning_rate": 0.05,
        "max_leaf_nodes": 7,
        "l2_regularization": 5.0,
    },
    "training_records": len(y),
    "training_defects": int(y.sum()),
}

path = OUT / "wafer_guard_final_gb.joblib"
joblib.dump(artifact, path)
print("Saved:", path)
print("Training records:", len(y))
print("Training defects:", int(y.sum()))
print("Features:", X.shape[1])
print("Threshold: 0.5 (provisional; not yet validated for deployment)")
