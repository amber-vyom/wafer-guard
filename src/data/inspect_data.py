from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = pd.read_csv(DATA_PATH)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

print("Dataset shape:", df.shape)
print("Number of sensor columns:", len(sensor_cols))
print("\nColumns:", df.columns.tolist()[-5:])
print("\nLabels:")
print(df["label"].value_counts(dropna=False))
print("\nDefect prevalence:", (df["label"] == 1).mean())
print("\nTimestamp range:")
print(df["timestamp"].min(), "to", df["timestamp"].max())
print("\nMissing sensor cells:", df[sensor_cols].isna().sum().sum())
print("\nSensor zero counts:")
print((df[sensor_cols] == 0).sum().sort_values(ascending=False).head(10))
print("\nSensor 0 values:")
print(df["sensor_0"].value_counts(dropna=False).head(10))

legacy_rows = df["sensor_0"] == 0.0

print("\nRows where sensor_0 is zero:", legacy_rows.sum())
print("\nsensor_1 values for these rows:")
print(df.loc[legacy_rows, "sensor_1"].value_counts(dropna=False))

print("\nMissing values in top 15 sensors:")
print(df[sensor_cols].isna().sum().sort_values(ascending=False).head(15))

print("\nSensors with at most one unique non-missing value:")
unique_counts = df[sensor_cols].nunique(dropna=True)
print(unique_counts[unique_counts <= 1])

print("\nTimestamps sorted already:", df["timestamp"].is_monotonic_increasing)