
from pathlib import Path

import numpy as np
import pandas as pd

from src.services.data_service import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = load_dataset(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

split = int(len(df) * 0.80)
train = df.iloc[:int(len(df) * 0.70)]
test = df.iloc[split:].copy()

sensors = [
    "sensor_140",
    "sensor_275",
    "sensor_4",
    "sensor_516",
    "sensor_382",
    "sensor_244",
]

print("TRAINING RAW VALUE RANGES")

for sensor in sensors:
    values = pd.to_numeric(train[sensor], errors="coerce").dropna()

    print(
        sensor,
        "min:", float(values.min()),
        "median:", float(values.median()),
        "99th percentile:", float(values.quantile(0.99)),
        "max:", float(values.max()),
    )

print("\nLARGEST RAW TEST VALUES")

for sensor in sensors:
    test[sensor] = pd.to_numeric(test[sensor], errors="coerce")

    rows = test.nlargest(3, sensor)[
        ["timestamp", "label", sensor]
    ]

    print("\n", sensor)
    print(rows.to_string(index=False))

print("\nMISSING VALUES IN SELECTED TEST SENSORS")

print(test[sensors].isna().sum().to_string())

print("\nLABEL COUNTS FOR EXTREME SENSOR VALUES")

for sensor in sensors:
    train_values = pd.to_numeric(train[sensor], errors="coerce")
    low = train_values.quantile(0.01)
    high = train_values.quantile(0.99)

    test_values = pd.to_numeric(test[sensor], errors="coerce")
    extreme = (test_values < low) | (test_values > high)

    print(
        sensor,
        "training 1%-99% range:", (float(low), float(high)),
        "extreme test rows:", int(extreme.sum()),
        "normal:", int((extreme & (test["label"] == -1)).sum()),
        "defective:", int((extreme & (test["label"] == 1)).sum()),
    )
