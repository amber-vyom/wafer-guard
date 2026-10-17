
from pathlib import Path

import numpy as np
import pandas as pd

from src.services.prediction_service import load_model, prepare_features
from src.services.data_service import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = load_dataset(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

train_end = int(len(df) * 0.70)
test_start = int(len(df) * 0.80)

train = df.iloc[:train_end]
test = df.iloc[test_start:].copy()

model, preprocessor, threshold = load_model()
features = prepare_features(test, preprocessor)

feature_names = preprocessor["feature_cols"]

print("Feature count:", len(feature_names))
print("Matrix feature count:", features.shape[1])

print("\nFEATURES WITH LARGEST ABSOLUTE VALUES")

max_values = np.max(np.abs(features), axis=0)
largest = np.argsort(max_values)[-15:][::-1]

for index in largest:
    print(
        feature_names[index],
        "maximum absolute value:",
        float(max_values[index]),
    )

print("\nTRAINING DISTRIBUTION CHECK")

sensor_cols = preprocessor["sensor_cols"]
usable_cols = preprocessor["usable_cols"]
imputer = preprocessor["imputer"]
scaler = preprocessor["scaler"]

raw_train = train[sensor_cols].copy()
raw_test = test[sensor_cols].copy()

for part in (raw_train, raw_test):
    for column in sensor_cols:
        part[column] = pd.to_numeric(part[column], errors="coerce")

    legacy = part["sensor_0"].eq(0.0).fillna(False)
    part.loc[legacy, "sensor_0"] = np.nan
    part.loc[legacy, "sensor_1"] = np.nan

train_values = scaler.transform(
    imputer.transform(raw_train[usable_cols])
)
test_values = scaler.transform(
    imputer.transform(raw_test[usable_cols])
)

train_max = np.max(np.abs(train_values), axis=0)
test_max = np.max(np.abs(test_values), axis=0)

print("\nTop 15 features by test/train scale ratio")

ratios = test_max / np.maximum(train_max, 1e-8)

for index in np.argsort(ratios)[-15:][::-1]:
    print(
        usable_cols[index],
        "train max:",
        float(train_max[index]),
        "test max:",
        float(test_max[index]),
        "ratio:",
        float(ratios[index]),
    )
