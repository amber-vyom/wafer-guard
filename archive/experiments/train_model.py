
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

# Save this baseline separately; do not overwrite the PyTorch artifacts.
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "wafer_logistic_baseline.joblib"

RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load and chronologically sort the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

n = len(df)
a = int(n * 0.70)
b = int(n * 0.80)

train_df = df.iloc[:a].copy()
val_df = df.iloc[a:b].copy()
test_df = df.iloc[b:].copy()


# --------------------------------------------------
# 2. Prepare sensor features and legacy indicators
# --------------------------------------------------

def get_features(part):
    X = part[sensor_cols].copy()

    legacy = X["sensor_0"].eq(0.0).fillna(False)

    X["is_legacy_ps_l"] = legacy.astype(float)
    X.loc[legacy, "sensor_0"] = np.nan

    sensor1 = X["sensor_1"].copy()
    X["sensor_1"] = sensor1.mask(legacy, np.nan)

    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    y = (part["label"] == 1).astype(np.int64).to_numpy()

    return X, y


X_train_raw, y_train = get_features(train_df)
X_val_raw, y_val = get_features(val_df)
X_test_raw, y_test = get_features(test_df)


# --------------------------------------------------
# 3. Select features using training data only
# --------------------------------------------------

numeric_cols = [c for c in sensor_cols if c != "sensor_1"]

usable_cols = [
    c for c in numeric_cols
    if X_train_raw[c].notna().any()
    and X_train_raw[c].nunique(dropna=True) > 1
]

usable_cols.append("sensor_1")

flag_cols = [
    "is_legacy_ps_l",
    "legacy_sensor_1_code_100",
    "legacy_sensor_1_code_200",
    "legacy_sensor_1_code_300",
]

feature_cols = usable_cols + flag_cols

# Fit preprocessing only on the training partition.
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

train_imputed = imputer.fit_transform(X_train_raw[usable_cols])
val_imputed = imputer.transform(X_val_raw[usable_cols])
test_imputed = imputer.transform(X_test_raw[usable_cols])

train_scaled = scaler.fit_transform(train_imputed)
val_scaled = scaler.transform(val_imputed)
test_scaled = scaler.transform(test_imputed)


def combine_features(scaled, raw):
    flags = raw[flag_cols].to_numpy(dtype=np.float32)

    return np.column_stack([
        scaled,
        flags,
    ]).astype(np.float32)


X_train = combine_features(train_scaled, X_train_raw)
X_val = combine_features(val_scaled, X_val_raw)
X_test = combine_features(test_scaled, X_test_raw)


# --------------------------------------------------
# 4. Print split information
# --------------------------------------------------

print("=" * 55)
print("WAFER DEFECT DETECTION - LOGISTIC REGRESSION BASELINE")
print("=" * 55)

print("Input features:", X_train.shape[1])
print("Training:", X_train.shape, "defects:", int(y_train.sum()))
print("Validation:", X_val.shape, "defects:", int(y_val.sum()))
print("Test:", X_test.shape, "defects:", int(y_test.sum()))
print("Test prevalence:", round(float(y_test.mean()), 4))


# --------------------------------------------------
# 5. Train classification-only baseline
# --------------------------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=RANDOM_STATE,
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 6. Select threshold using validation data only
# --------------------------------------------------

val_prob = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION RESULTS")
print("Average precision:", round(
    average_precision_score(y_val, val_prob), 5
))
print("ROC-AUC:", round(
    roc_auc_score(y_val, val_prob), 5
))

print("\nTHRESHOLD DIAGNOSTICS - VALIDATION ONLY")
print("Threshold | Precision | Recall | F1 | TP | FP | FN | TN")

best_threshold = 0.5
best_f1 = -1.0

for threshold in np.arange(0.01, 1.00, 0.01):
    val_pred = (val_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        val_pred,
        average="binary",
        zero_division=0,
    )

    tp = int(((val_pred == 1) & (y_val == 1)).sum())
    fp = int(((val_pred == 1) & (y_val == 0)).sum())
    fn = int(((val_pred == 0) & (y_val == 1)).sum())
    tn = int(((val_pred == 0) & (y_val == 0)).sum())

    if f1 > best_f1:
        best_f1 = float(f1)
        best_threshold = float(threshold)

    # Print every 10th threshold to keep the output readable.
    if round(threshold * 100) % 10 == 0:
        print(
            f"{threshold:9.2f} | {precision:9.3f} | "
            f"{recall:6.3f} | {f1:4.3f} | "
            f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
        )

print("\nSelected threshold:", best_threshold)
print("Best validation F1:", round(best_f1, 5))


# --------------------------------------------------
# 7. Evaluate the untouched test partition once
# --------------------------------------------------

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Threshold:", best_threshold)
print("Test prevalence:", round(float(y_test.mean()), 5))
print("Accuracy:", round(accuracy_score(y_test, test_pred), 5))
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 5
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 5
))
print("F1:", round(
    f1_score(y_test, test_pred, zero_division=0), 5
))
print("Average precision:", round(
    average_precision_score(y_test, test_prob), 5
))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 5))
else:
    print("ROC-AUC unavailable: test contains only one class")

print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))


# --------------------------------------------------
# 8. Save the baseline separately
# --------------------------------------------------

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "sensor_cols": sensor_cols,
        "usable_cols": usable_cols,
        "flag_cols": flag_cols,
        "feature_cols": feature_cols,
        "imputer": imputer,
        "scaler": scaler,
        "threshold": best_threshold,
        "model_type": "logistic_regression_baseline",
    },
    MODEL_PATH,
)

print("\nBaseline saved:", MODEL_PATH)
print("Original PyTorch model and preprocessor were not overwritten.")
