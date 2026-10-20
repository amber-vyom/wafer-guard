"""
WaferGuard sensor drift diagnostic (training vs validation only).

Uses the chronological first 70% as train and next 10% as validation.
Does NOT inspect the final 20% test period, train a model, or modify artifacts.

Run from repository root:
    python src/models/audit_sensor_drift.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUT_DIR = ROOT / "reports" / "experiments"


def safe_num(series):
    return pd.to_numeric(series, errors="coerce")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()

    records = []
    for col in sensor_cols:
        tr = safe_num(train[col])
        va = safe_num(val[col])
        tr_missing = float(tr.isna().mean())
        va_missing = float(va.isna().mean())

        tr_obs = tr.dropna()
        va_obs = va.dropna()
        if len(tr_obs) and len(va_obs):
            tr_median = float(tr_obs.median())
            va_median = float(va_obs.median())
            tr_mean = float(tr_obs.mean())
            va_mean = float(va_obs.mean())
            tr_std = float(tr_obs.std(ddof=0))
            va_std = float(va_obs.std(ddof=0))
            pooled_scale = float(np.sqrt((tr_std**2 + va_std**2) / 2.0))
            if pooled_scale > 0 and np.isfinite(pooled_scale):
                standardized_mean_shift = float((va_mean - tr_mean) / pooled_scale)
            else:
                standardized_mean_shift = None
            median_ratio = (
                float(va_median / tr_median)
                if tr_median != 0 and np.isfinite(tr_median) and np.isfinite(va_median)
                else None
            )
        else:
            tr_median = va_median = tr_mean = va_mean = None
            standardized_mean_shift = median_ratio = None

        records.append({
            "sensor": col,
            "train_missing_pct": round(tr_missing * 100, 3),
            "validation_missing_pct": round(va_missing * 100, 3),
            "missing_pct_point_change": round((va_missing - tr_missing) * 100, 3),
            "train_unique_observed": int(tr_obs.nunique()),
            "validation_unique_observed": int(va_obs.nunique()),
            "train_median": tr_median,
            "validation_median": va_median,
            "train_mean": tr_mean,
            "validation_mean": va_mean,
            "standardized_mean_shift": standardized_mean_shift,
            "median_ratio_val_over_train": median_ratio,
        })

    audit = pd.DataFrame(records)
    audit["abs_missing_shift"] = audit["missing_pct_point_change"].abs()
    audit["abs_standardized_mean_shift"] = audit["standardized_mean_shift"].abs()

    # Label prevalence by chronological partition.
    train_y = (train["label"] == 1).astype(int)
    val_y = (val["label"] == 1).astype(int)
    label_summary = {
        "dataset_rows": int(n),
        "train_rows": int(len(train)),
        "train_defects": int(train_y.sum()),
        "train_defect_rate": float(train_y.mean()),
        "validation_rows": int(len(val)),
        "validation_defects": int(val_y.sum()),
        "validation_defect_rate": float(val_y.mean()),
        "sensor_count": int(len(sensor_cols)),
        "total_missing_sensor_cells_train": int(
            train[sensor_cols].apply(pd.to_numeric, errors="coerce").isna().sum().sum()
        ),
        "total_sensor_cells_train": int(len(train) * len(sensor_cols)),
        "total_missing_sensor_cells_validation": int(
            val[sensor_cols].apply(pd.to_numeric, errors="coerce").isna().sum().sum()
        ),
        "total_sensor_cells_validation": int(len(val) * len(sensor_cols)),
        "note": (
            "Diagnostic only. Mean shifts can be affected by outliers and special codes; "
            "investigate sensors rather than automatically dropping them. Final test period "
            "was not loaded into the analysis."
        ),
    }

    csv_path = OUT_DIR / "sensor_drift_train_vs_validation.csv"
    audit.drop(columns=["abs_missing_shift", "abs_standardized_mean_shift"]).to_csv(
        csv_path, index=False
    )
    json_path = OUT_DIR / "sensor_drift_train_vs_validation_summary.json"
    json_path.write_text(json.dumps(label_summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("WAFERGUARD SENSOR DRIFT AUDIT (TRAIN VS VALIDATION)")
    print("=" * 72)
    print(f"Train: {len(train)} rows; defects={int(train_y.sum())}; rate={train_y.mean():.4f}")
    print(f"Validation: {len(val)} rows; defects={int(val_y.sum())}; rate={val_y.mean():.4f}")
    print(f"Sensors: {len(sensor_cols)}")
    print("\nTOP 20 SENSORS BY CHANGE IN MISSINGNESS")
    cols = [
        "sensor", "train_missing_pct", "validation_missing_pct",
        "missing_pct_point_change", "train_unique_observed",
        "validation_unique_observed",
    ]
    print(
        audit.sort_values("abs_missing_shift", ascending=False)[cols]
        .head(20).to_string(index=False)
    )
    print("\nTOP 20 SENSORS BY ABSOLUTE STANDARDIZED MEAN SHIFT")
    cols2 = [
        "sensor", "standardized_mean_shift", "train_mean",
        "validation_mean", "train_median", "validation_median",
    ]
    print(
        audit.dropna(subset=["standardized_mean_shift"])
        .sort_values("abs_standardized_mean_shift", ascending=False)[cols2]
        .head(20).to_string(index=False)
    )
    print(f"\nCSV report: {csv_path}")
    print(f"JSON summary: {json_path}")
    print("No model artifacts were modified.")


if __name__ == "__main__":
    main()
