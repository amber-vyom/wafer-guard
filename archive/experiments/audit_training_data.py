
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    required = {"timestamp", "label"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )
    df = df.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    sensor_cols = [
        c for c in df.columns if c.startswith("sensor_")
    ]

    n = len(df)
    a = int(n * 0.70)
    b = int(n * 0.80)

    splits = {
        "TRAIN": df.iloc[:a],
        "VALIDATION": df.iloc[a:b],
        "TEST": df.iloc[b:],
    }

    print("=" * 60)
    print("WAFERGUARD TRAINING DATA AUDIT")
    print("=" * 60)
    print("Rows:", len(df))
    print("Sensor columns:", len(sensor_cols))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Duplicate timestamps:", int(df["timestamp"].duplicated().sum()))
    print("Timestamp range:", df["timestamp"].min(), "to", df["timestamp"].max())

    print("\nLABEL DISTRIBUTION")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("\nCHRONOLOGICAL SPLITS")
    for name, part in splits.items():
        defects = int((part["label"] == 1).sum())
        normal = int((part["label"] == -1).sum())

        print(f"\n{name}")
        print("Rows:", len(part))
        print("Normal:", normal)
        print("Defects:", defects)
        print("Defect rate:", round(defects / max(len(part), 1), 5))
        print("Start:", part["timestamp"].min())
        print("End:", part["timestamp"].max())

    print("\nSENSOR QUALITY")
    sensor_data = df[sensor_cols]
    missing_counts = sensor_data.isna().sum()
    unique_counts = sensor_data.nunique(dropna=True)

    print("Total missing sensor cells:", int(missing_counts.sum()))
    print("Sensors with over 50% missing:", int(
        (missing_counts > len(df) * 0.50).sum()
    ))
    print("Sensors with zero or one observed unique value:", int(
        (unique_counts <= 1).sum()
    ))

    print("\nMOST MISSING SENSORS")
    quality = pd.DataFrame({
        "missing": missing_counts,
        "missing_rate": missing_counts / len(df),
        "unique_values": unique_counts,
    })
    print(quality.sort_values("missing_rate", ascending=False).head(15))

    print("\nLABEL VALIDATION")
    unexpected = sorted(
        set(df["label"].dropna().unique()) - {-1, 1}
    )
    print("Unexpected labels:", unexpected)
    print("Missing labels:", int(df["label"].isna().sum()))

    print("\nAudit complete. No files or model artifacts changed.")


if __name__ == "__main__":
    main()
