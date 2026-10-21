
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

sys.path.insert(0, str(ROOT))
from src.data.splits import split_bounds

# 1. Load and order the observations.
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

# 2. Chronological train / validation / test split.
n = len(df)
train_end, val_end = split_bounds(n)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# 3. Separate the target and remove metadata from the features.
def separate_features(part):
    X = part[sensor_cols].copy()
    y = (part["label"] == 1).astype(np.int64).to_numpy()

    # PS-L sentinel: sensor_0 == 0.0.
    legacy = X["sensor_0"].eq(0.0).fillna(False)

    # Preserve station identity as a feature.
    X["is_legacy_ps_l"] = legacy.astype(float)

    # A legacy sensor_0 zero is not a genuine measurement.
    X.loc[legacy, "sensor_0"] = np.nan

    # sensor_1 is categorical only for legacy wafers.
    sensor1 = X["sensor_1"].copy()
    X["sensor_1"] = sensor1.mask(legacy, np.nan)

    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    return X, y

X_train_raw, y_train = separate_features(train_df)
X_val_raw, y_val = separate_features(val_df)
X_test_raw, y_test = separate_features(test_df)

# 4. Fit feature selection using TRAINING DATA ONLY.
# Drop sensors with no variation or no observed values in training.
numeric_sensor_cols = sensor_cols.copy()
numeric_sensor_cols.remove("sensor_1")

train_sensor_view = X_train_raw[numeric_sensor_cols]
usable_cols = [
    col
    for col in numeric_sensor_cols
    if train_sensor_view[col].notna().any()
    and train_sensor_view[col].nunique(dropna=True) > 1
]

# sensor_0 remains a continuous sensor after replacing the legacy sentinel.
# sensor_1's non-legacy values remain continuous; legacy codes are one-hot.
usable_cols.append("sensor_1")

special_cols = [
    "is_legacy_ps_l",
    "legacy_sensor_1_code_100",
    "legacy_sensor_1_code_200",
    "legacy_sensor_1_code_300",
]
feature_cols = usable_cols + special_cols

X_train = X_train_raw[feature_cols].copy()
X_val = X_val_raw[feature_cols].copy()
X_test = X_test_raw[feature_cols].copy()

# 5. Fit imputation and scaling on training data only.
continuous_cols = usable_cols
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_cont = imputer.fit_transform(X_train[continuous_cols])
X_val_cont = imputer.transform(X_val[continuous_cols])
X_test_cont = imputer.transform(X_test[continuous_cols])

X_train_scaled = scaler.fit_transform(X_train_cont)
X_val_scaled = scaler.transform(X_val_cont)
X_test_scaled = scaler.transform(X_test_cont)

# Add categorical flags without scaling them.
X_train_final = np.column_stack(
    [X_train_scaled, X_train[special_cols].to_numpy(dtype=np.float32)]
).astype(np.float32)

X_val_final = np.column_stack(
    [X_val_scaled, X_val[special_cols].to_numpy(dtype=np.float32)]
).astype(np.float32)

X_test_final = np.column_stack(
    [X_test_scaled, X_test[special_cols].to_numpy(dtype=np.float32)]
).astype(np.float32)

print("Preprocessing complete.")
print("Feature count:", X_train_final.shape[1])
print("Train:", X_train_final.shape, "defects:", int(y_train.sum()))
print("Validation:", X_val_final.shape, "defects:", int(y_val.sum()))
print("Test:", X_test_final.shape, "defects:", int(y_test.sum()))
print("Test defect prevalence:", round(float(y_test.mean()), 4))
print("All feature values finite:",
      np.isfinite(X_train_final).all()
      and np.isfinite(X_val_final).all()
      and np.isfinite(X_test_final).all())
