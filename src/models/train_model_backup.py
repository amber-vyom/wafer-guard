from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.wafer_model import WaferAutoencoder

DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
ARTIFACT_DIR = ROOT / "artifacts"

torch.manual_seed(42)
np.random.seed(42)

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
    return np.column_stack([scaled, flags]).astype(np.float32)


X_train = combine_features(train_scaled, X_train_raw)
X_val = combine_features(val_scaled, X_val_raw)
X_test = combine_features(test_scaled, X_test_raw)

print("Input features:", X_train.shape[1])
print("Train:", X_train.shape, "defects:", int(y_train.sum()))
print("Validation:", X_val.shape, "defects:", int(y_val.sum()))
print("Test:", X_test.shape, "defects:", int(y_test.sum()))
print("Test prevalence:", round(float(y_test.mean()), 4))

train_data = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
)

model = WaferAutoencoder(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
reconstruction_loss_fn = nn.MSELoss()

positive_count = max(int(y_train.sum()), 1)
negative_count = len(y_train) - positive_count

positive_weight = torch.tensor([
    negative_count / positive_count
])

classification_loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=positive_weight
)

for epoch in range(1, 21):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        reconstructed, logits, _ = model(batch_x)

        reconstruction_loss = reconstruction_loss_fn(
            reconstructed, batch_x
        )
        classification_loss = classification_loss_fn(
            logits, batch_y
        )

        loss = reconstruction_loss + classification_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_x)

    print(
        f"Epoch {epoch:02d}/20 - "
        f"loss: {total_loss / len(train_data):.4f}"
    )

model.eval()

with torch.no_grad():
    val_logits = model(torch.tensor(X_val, dtype=torch.float32))[1]
    val_prob = torch.sigmoid(val_logits).numpy()

thresholds = np.arange(0.05, 0.96, 0.05)

threshold = max(
    thresholds,
    key=lambda value: f1_score(
        y_val,
        (val_prob >= value).astype(int),
        zero_division=0,
    ),
)

val_pred = (val_prob >= threshold).astype(int)

print("\nValidation threshold:", float(threshold))
print("Validation defects:", int(y_val.sum()))
print("Validation predicted defects:", int(val_pred.sum()))
print("Validation confusion matrix:")
print(confusion_matrix(y_val, val_pred))

with torch.no_grad():
    test_logits = model(torch.tensor(X_test, dtype=torch.float32))[1]
    test_prob = torch.sigmoid(test_logits).numpy()

test_pred = (test_prob >= threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Test prevalence:", float(y_test.mean()))
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, zero_division=0))
print("Recall:", recall_score(y_test, test_pred, zero_division=0))
print("F1:", f1_score(y_test, test_pred, zero_division=0))
print("Average precision:", average_precision_score(y_test, test_prob))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", roc_auc_score(y_test, test_prob))
else:
    print("ROC-AUC: unavailable; test contains only one class")

print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))

ARTIFACT_DIR.mkdir(exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "threshold": float(threshold),
    },
    ARTIFACT_DIR / "wafer_model.pt",
)

joblib.dump(
    {
        "sensor_cols": sensor_cols,
        "usable_cols": usable_cols,
        "flag_cols": flag_cols,
        "feature_cols": feature_cols,
        "imputer": imputer,
        "scaler": scaler,
    },
    ARTIFACT_DIR / "preprocessor.joblib",
)

print("\nModel saved:", ARTIFACT_DIR / "wafer_model.pt")
print("Preprocessor saved:", ARTIFACT_DIR / "preprocessor.joblib")
