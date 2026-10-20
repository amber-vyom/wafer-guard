"""
WaferGuard targeted sensor usefulness audit.
Compares selected drifting sensors between normal and defective samples
in chronological training and validation splits only.
Does not train models or modify deployed artifacts.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
REPORT_DIR = ROOT / "reports" / "experiments"

SENSORS = [
    "sensor_244", "sensor_245", "sensor_246", "sensor_111",
    "sensor_109", "sensor_519", "sensor_247", "sensor_385",
    "sensor_393", "sensor_255", "sensor_527", "sensor_440",
    "sensor_130", "sensor_169", "sensor_480", "sensor_129",
]

def pct_missing(series):
    return float(series.isna().mean() * 100)

def standardized_difference(a, b):
    """Cohen's d for observed values; returns NaN if too few/constant."""
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    return float((b.mean() - a.mean()) / np.sqrt(pooled_var))

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Source SECOM convention: -1 = normal, +1 = defect. Convert to 0/1.
    df["target"] = (pd.to_numeric(df["label"], errors="coerce") == 1).astype(int)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()

    sensors = [s for s in SENSORS if s in df.columns]
    rows = []
    for split_name, split in [("train", train), ("validation", val)]:
        for sensor in sensors:
            values = pd.to_numeric(split[sensor], errors="coerce")
            for target, class_name in [(0, "normal"), (1, "defect")]:
                subset = split.loc[split["target"] == target, sensor]
                subset = pd.to_numeric(subset, errors="coerce")
                observed = subset.dropna().astype(float)
                rows.append({
                    "split": split_name,
                    "sensor": sensor,
                    "class": class_name,
                    "samples": int(len(subset)),
                    "observed": int(observed.size),
                    "missing_pct": pct_missing(subset),
                    "mean": float(observed.mean()) if len(observed) else np.nan,
                    "median": float(observed.median()) if len(observed) else np.nan,
                    "std": float(observed.std(ddof=1)) if len(observed) > 1 else np.nan,
                    "unique_observed": int(observed.nunique()),
                })

    detail = pd.DataFrame(rows)
    summary_rows = []
    for split_name, split in [("train", train), ("validation", val)]:
        for sensor in sensors:
            normal = pd.to_numeric(split.loc[split["target"] == 0, sensor], errors="coerce")
            defect = pd.to_numeric(split.loc[split["target"] == 1, sensor], errors="coerce")
            normal_obs = normal.dropna().astype(float)
            defect_obs = defect.dropna().astype(float)
            summary_rows.append({
                "split": split_name,
                "sensor": sensor,
                "normal_n": int((split["target"] == 0).sum()),
                "defect_n": int((split["target"] == 1).sum()),
                "normal_missing_pct": pct_missing(normal),
                "defect_missing_pct": pct_missing(defect),
                "defect_minus_normal_missing_pp": pct_missing(defect) - pct_missing(normal),
                "normal_observed_mean": float(normal_obs.mean()) if len(normal_obs) else np.nan,
                "defect_observed_mean": float(defect_obs.mean()) if len(defect_obs) else np.nan,
                "normal_observed_median": float(normal_obs.median()) if len(normal_obs) else np.nan,
                "defect_observed_median": float(defect_obs.median()) if len(defect_obs) else np.nan,
                "mean_effect_size_defect_vs_normal": standardized_difference(normal_obs, defect_obs),
            })

    summary = pd.DataFrame(summary_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORT_DIR / "targeted_sensor_class_detail.csv"
    summary_path = REPORT_DIR / "targeted_sensor_class_summary.csv"
    json_path = REPORT_DIR / "targeted_sensor_class_summary.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=" * 76)
    print("WAFERGUARD TARGETED SENSOR USEFULNESS AUDIT")
    print("=" * 76)
    print(f"Train: {len(train)} rows; defects={int(train['target'].sum())}")
    print(f"Validation: {len(val)} rows; defects={int(val['target'].sum())}")
    print("No test-period rows used. No model or preprocessor artifacts modified.")
    print("\nDEFECT VS NORMAL: MISSINGNESS DIFFERENCE (percentage points)")
    miss = summary[["split", "sensor", "normal_missing_pct", "defect_missing_pct",
                    "defect_minus_normal_missing_pp"]].copy()
    print(miss.round(2).to_string(index=False))
    print("\nDEFECT VS NORMAL: OBSERVED VALUE EFFECT SIZE")
    effects = summary[["split", "sensor", "normal_observed_mean", "defect_observed_mean",
                       "normal_observed_median", "defect_observed_median",
                       "mean_effect_size_defect_vs_normal"]].copy()
    print(effects.round(4).to_string(index=False))
    payload = {
        "train_rows": len(train),
        "train_defects": int(train["target"].sum()),
        "validation_rows": len(val),
        "validation_defects": int(val["target"].sum()),
        "sensors": sensors,
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
        "note": "Descriptive diagnostics only; association is not proof of predictive usefulness."
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nDetailed report: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"JSON summary: {json_path}")

if __name__ == "__main__":
    main()
