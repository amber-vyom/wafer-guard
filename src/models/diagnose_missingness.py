
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    sensor_cols = [
        col for col in df.columns if col.startswith("sensor_")
    ]

    train_end = int(len(df) * 0.70)
    train = df.iloc[:train_end].copy()

    normal = train[train["label"] == -1]
    defect = train[train["label"] == 1]

    print("=" * 65)
    print("WAFERGUARD MISSINGNESS AND TIMESTAMP DIAGNOSTICS")
    print("=" * 65)

    # Duplicate timestamp inspection
    duplicate_mask = df["timestamp"].duplicated(keep=False)
    duplicates = df.loc[duplicate_mask].sort_values("timestamp")

    print("\nDUPLICATE TIMESTAMPS")
    print("Rows sharing a timestamp:", len(duplicates))
    print(
        "Distinct duplicated timestamps:",
        int(duplicates["timestamp"].nunique()),
    )

    if not duplicates.empty:
        print("\nFirst 15 duplicate-timestamp records:")
        columns = ["timestamp", "label"] + sensor_cols[:5]
        print(duplicates[columns].head(15).to_string(index=False))

    # Missingness statistics from training data only
    records = []

    for col in sensor_cols:
        normal_values = normal[col]
        defect_values = defect[col]

        normal_missing = float(normal_values.isna().mean())
        defect_missing = float(defect_values.isna().mean())

        records.append({
            "sensor": col,
            "normal_missing_rate": normal_missing,
            "defect_missing_rate": defect_missing,
            "difference": defect_missing - normal_missing,
            "normal_observed": int(normal_values.notna().sum()),
            "defect_observed": int(defect_values.notna().sum()),
        })

    quality = pd.DataFrame(records)

    print("\nTRAINING LABEL COUNTS")
    print("Normal:", len(normal))
    print("Defect:", len(defect))

    print("\nSENSORS WITH HIGHEST DEFECT-VS-NORMAL MISSINGNESS")
    print(
        quality.assign(
            absolute_difference=quality["difference"].abs()
        )
        .sort_values("absolute_difference", ascending=False)
        .drop(columns="absolute_difference")
        .head(20)
        .to_string(index=False, formatters={
            "normal_missing_rate": "{:.3f}".format,
            "defect_missing_rate": "{:.3f}".format,
            "difference": "{:+.3f}".format,
        })
    )

    print("\nMOST SPARSE SENSORS IN TRAINING DATA")
    print(
        quality.sort_values(
            "normal_observed"
        ).head(15).to_string(index=False)
    )

    # Legacy sentinel diagnostic
    if "sensor_0" in train.columns:
        legacy = train["sensor_0"].eq(0).fillna(False)

        print("\nLEGACY SENSOR_0 ZERO SENTINEL")
        print("Training rows with sensor_0 == 0:", int(legacy.sum()))
        print(
            "Normal rows with sentinel:",
            int((legacy & train["label"].eq(-1)).sum()),
        )
        print(
            "Defect rows with sentinel:",
            int((legacy & train["label"].eq(1)).sum()),
        )

    print("\nDiagnostic complete.")
    print("Training data only was used for sensor comparisons.")
    print("No files or model artifacts were changed.")


if __name__ == "__main__":
    main()
