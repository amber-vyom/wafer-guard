
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUTPUT_DIR = ROOT / "artifacts" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

if not sensor_cols:
    raise ValueError("No sensor_* columns found.")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        print(
            "WARNING:",
            int(df["timestamp"].isna().sum()),
            "records have invalid timestamps."
        )
    df = df.sort_values(
        "timestamp", kind="mergesort", na_position="last"
    ).reset_index(drop=True)

df = df[df["label"].isin([-1, 1])].copy().reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

periods = {
    "train": df.iloc[:train_end].copy(),
    "validation": df.iloc[train_end:val_end].copy(),
    "test": df.iloc[val_end:].copy(),
}

def defect_rate(frame):
    return float((frame["label"] == 1).mean()) if len(frame) else 0.0

def population_stability_index(reference, comparison):
    """PSI for non-missing numeric values using reference quantile bins."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]

    if len(reference) < 20 or len(comparison) < 5:
        return np.nan

    quantiles = np.linspace(0, 1, 11)
    inner_edges = np.unique(np.quantile(reference, quantiles))
    inner_edges = inner_edges[1:-1]

    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))

    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cmp_pct = cmp_counts / cmp_counts.sum()

    # Smoothing prevents division by zero and log(0).
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cmp_pct = np.clip(cmp_pct, epsilon, None)

    return float(np.sum(
        (cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)
    ))

print("=" * 68)
print("WAFERGUARD TEMPORAL DRIFT DIAGNOSTIC")
print("=" * 68)
print(f"Records: {len(df)}")
print(f"Sensor columns: {len(sensor_cols)}")
print(f"Duplicate rows: {int(df.duplicated().sum())}")

if "timestamp" in df.columns:
    valid_times = df["timestamp"].dropna()
    duplicate_timestamp_rows = int(
        df["timestamp"].duplicated(keep=False).sum()
    )
    duplicate_timestamp_values = int(
        df.loc[df["timestamp"].notna(), "timestamp"].duplicated().sum()
    )

    print(f"Rows sharing a duplicated timestamp: {duplicate_timestamp_rows}")
    print(f"Repeated timestamp values: {duplicate_timestamp_values}")

print("\nCHRONOLOGICAL PERIODS")
period_summary = {}

for name, frame in periods.items():
    period_summary[name] = {
        "rows": int(len(frame)),
        "normal_records": int((frame["label"] == -1).sum()),
        "defect_records": int((frame["label"] == 1).sum()),
        "defect_rate": defect_rate(frame),
        "start": (
            str(frame["timestamp"].min())
            if "timestamp" in frame.columns else None
        ),
        "end": (
            str(frame["timestamp"].max())
            if "timestamp" in frame.columns else None
        ),
    }

    info = period_summary[name]
    print(
        f"{name.upper():12s} rows={info['rows']:4d} "
        f"defects={info['defect_records']:3d} "
        f"defect_rate={info['defect_rate']:.5f} "
        f"start={info['start']} end={info['end']}"
    )

train = periods["train"]
validation = periods["validation"]
test = periods["test"]

X_train = train[sensor_cols].apply(pd.to_numeric, errors="coerce")
X_val = validation[sensor_cols].apply(pd.to_numeric, errors="coerce")
X_test = test[sensor_cols].apply(pd.to_numeric, errors="coerce")

rows = []

for col in sensor_cols:
    tr = X_train[col].to_numpy(dtype=float, na_value=np.nan)
    va = X_val[col].to_numpy(dtype=float, na_value=np.nan)
    te = X_test[col].to_numpy(dtype=float, na_value=np.nan)

    train_missing = float(np.isnan(tr).mean())
    val_missing = float(np.isnan(va).mean())
    test_missing = float(np.isnan(te).mean())

    train_observed = tr[np.isfinite(tr)]
    val_observed = va[np.isfinite(va)]
    test_observed = te[np.isfinite(te)]

    train_unique = int(np.unique(train_observed).size)

    if train_unique < 2:
        continue

    psi_val = population_stability_index(train_observed, val_observed)
    psi_test = population_stability_index(train_observed, test_observed)

    rows.append({
        "sensor": col,
        "train_missing_rate": train_missing,
        "validation_missing_rate": val_missing,
        "test_missing_rate": test_missing,
        "validation_missing_rate_change": val_missing - train_missing,
        "test_missing_rate_change": test_missing - train_missing,
        "train_unique_values": train_unique,
        "psi_validation": psi_val,
        "psi_test": psi_test,
    })

drift = pd.DataFrame(rows)

if drift.empty:
    raise ValueError("No variable sensors available for drift analysis.")

drift["max_psi"] = drift[["psi_validation", "psi_test"]].max(axis=1)
drift["max_missing_rate_change"] = drift[
    ["validation_missing_rate_change", "test_missing_rate_change"]
].abs().max(axis=1)

drift = drift.sort_values("max_psi", ascending=False, na_position="last")

csv_path = OUTPUT_DIR / "temporal_sensor_drift.csv"
drift.to_csv(csv_path, index=False)

missing_path = OUTPUT_DIR / "temporal_missingness_drift.csv"
drift.sort_values(
    "max_missing_rate_change", ascending=False
).to_csv(missing_path, index=False)

def top_rows(frame, columns, count=15):
    result = []
    for _, row in frame.head(count).iterrows():
        item = {}
        for col in columns:
            value = row[col]
            item[col] = (
                None if pd.isna(value) else
                float(value) if isinstance(value, (np.floating, float)) else
                int(value) if isinstance(value, (np.integer, int)) else
                str(value)
            )
        result.append(item)
    return result

weekly = []
if "timestamp" in df.columns:
    dated = df.dropna(subset=["timestamp"]).copy()
    if not dated.empty:
        dated["week"] = dated["timestamp"].dt.to_period("W").astype(str)
        for week, group in dated.groupby("week", sort=True):
            weekly.append({
                "week": week,
                "records": int(len(group)),
                "defects": int((group["label"] == 1).sum()),
                "defect_rate": defect_rate(group),
            })

summary = {
    "dataset_rows": int(len(df)),
    "sensor_columns": int(len(sensor_cols)),
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_timestamp_rows": (
        int(df["timestamp"].duplicated(keep=False).sum())
        if "timestamp" in df.columns else None
    ),
    "periods": period_summary,
    "weekly_defect_rates": weekly,
    "top_distribution_shifts": top_rows(
        drift,
        [
            "sensor", "psi_validation", "psi_test",
            "train_missing_rate", "validation_missing_rate",
            "test_missing_rate"
        ],
    ),
    "top_missingness_shifts": top_rows(
        drift.sort_values("max_missing_rate_change", ascending=False),
        [
            "sensor", "train_missing_rate", "validation_missing_rate",
            "test_missing_rate", "validation_missing_rate_change",
            "test_missing_rate_change"
        ],
    ),
    "interpretation_note": (
        "PSI is a descriptive drift signal, not proof of a causal change "
        "or proof that a sensor predicts defects. PSI above 0.10 can be "
        "used as a screening flag and above 0.25 as a stronger flag, "
        "not as universal decision boundaries. Test results are exploratory "
        "because this test set has been examined in earlier experiments. "
        "No records were deleted and no model was trained or changed."
    ),
}

json_path = OUTPUT_DIR / "temporal_drift_summary.json"
json_path.write_text(json.dumps(summary, indent=2))

print("\nTOP 15 SENSOR DISTRIBUTION SHIFTS")
print(
    drift[
        [
            "sensor", "psi_validation", "psi_test",
            "train_missing_rate", "validation_missing_rate",
            "test_missing_rate"
        ]
    ].head(15).to_string(index=False)
)

print("\nTOP 15 MISSINGNESS SHIFTS")
missing_drift = drift.sort_values(
    "max_missing_rate_change", ascending=False
)
print(
    missing_drift[
        [
            "sensor", "train_missing_rate", "validation_missing_rate",
            "test_missing_rate", "validation_missing_rate_change",
            "test_missing_rate_change"
        ]
    ].head(15).to_string(index=False)
)

print("\nWEEKLY DEFECT RATES")
if weekly:
    print(pd.DataFrame(weekly).to_string(index=False))
else:
    print("Weekly analysis unavailable: no valid timestamps.")

print("\nSAVED REPORTS")
print(csv_path)
print(missing_path)
print(json_path)
print("\nDiagnostic complete. No model or dataset was modified.")
