
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = pd.read_csv(DATA_PATH)

# Reproduce the chronological split used by the training pipeline.
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()

y_train = train_df["label"].eq(1).astype(int)
y_val = val_df["label"].eq(1).astype(int)

sensor_cols = [
    c for c in df.columns
    if c.startswith("sensor_")
]

X_train = train_df[sensor_cols].apply(pd.to_numeric, errors="coerce")
X_val = val_df[sensor_cols].apply(pd.to_numeric, errors="coerce")

# Reproduce legacy PS-L handling.
legacy_train = X_train["sensor_0"].eq(0).fillna(False)
legacy_val = X_val["sensor_0"].eq(0).fillna(False)

X_train.loc[legacy_train, "sensor_0"] = np.nan
X_val.loc[legacy_val, "sensor_0"] = np.nan

codes = [100, 200, 300]

for code in codes:
    X_train[f"legacy_code_{code}"] = (
        legacy_train & X_train["sensor_1"].eq(code)
    ).astype(int)
    X_val[f"legacy_code_{code}"] = (
        legacy_val & X_val["sensor_1"].eq(code)
    ).astype(int)

X_train.loc[legacy_train, "sensor_1"] = np.nan
X_val.loc[legacy_val, "sensor_1"] = np.nan

# Select usable sensors using training data only.
usable_cols = [
    c for c in sensor_cols
    if c != "sensor_1"
    and X_train[c].notna().any()
    and X_train[c].nunique(dropna=True) > 1
]

usable_cols.append("sensor_1")
flag_cols = [f"legacy_code_{code}" for code in codes]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_num = imputer.fit_transform(X_train[usable_cols])
X_val_num = imputer.transform(X_val[usable_cols])

X_train_num = scaler.fit_transform(X_train_num)
X_val_num = scaler.transform(X_val_num)

X_train_final = np.column_stack([
    X_train_num,
    X_train[flag_cols].to_numpy(),
    legacy_train.astype(int).to_numpy(),
])

X_val_final = np.column_stack([
    X_val_num,
    X_val[flag_cols].to_numpy(),
    legacy_val.astype(int).to_numpy(),
])

print("Training shape:", X_train_final.shape)
print("Validation shape:", X_val_final.shape)
print("Training defects:", int(y_train.sum()))
print("Validation defects:", int(y_val.sum()))

# Train only on the training partition.
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
model.fit(X_train_final, y_train)

# Use validation labels only for this diagnostic.
probabilities = model.predict_proba(X_val_final)[:, 1]

print("\nVALIDATION BASELINE")
print("Average precision:", round(average_precision_score(y_val, probabilities), 5))
print("ROC-AUC:", round(roc_auc_score(y_val, probabilities), 5))

print("\nThreshold | Precision | Recall | F1 | TP | FP | FN | TN")

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        predictions,
        average="binary",
        zero_division=0,
    )

    tp = int(((predictions == 1) & (y_val.to_numpy() == 1)).sum())
    fp = int(((predictions == 1) & (y_val.to_numpy() == 0)).sum())
    fn = int(((predictions == 0) & (y_val.to_numpy() == 1)).sum())
    tn = int(((predictions == 0) & (y_val.to_numpy() == 0)).sum())

    print(
        f"{threshold:8.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:4.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
    )

print("\nDiagnostic complete. No model artifacts were changed.")
