
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUTPUT_DIR = ROOT / "artifacts" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

if not sensor_cols:
    raise ValueError("No sensor_* columns found.")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

df = df[df["label"].isin([-1, 1])].copy()
split = int(len(df) * 0.70)
train = df.iloc[:split].copy()

X = train[sensor_cols].apply(pd.to_numeric, errors="coerce")
y = (train["label"] == 1).astype(int)

# Exclude sensors that are constant, entirely missing, or almost entirely missing.
missing_rate = X.isna().mean()
unique_count = X.nunique(dropna=True)

eligible = [
    c for c in sensor_cols
    if unique_count[c] >= 2
    and missing_rate[c] <= 0.50
]

if len(eligible) < 2:
    raise ValueError(
        f"Only {len(eligible)} eligible sensors. "
        "Inspect the dataset before continuing."
    )

X = X[eligible]

# Fit all transformations on training data only.
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Remove any columns that remain non-finite or constant after imputation.
valid_indices = []
for i, col in enumerate(eligible):
    values = X_imputed[:, i]
    if np.isfinite(values).all() and np.unique(values).size >= 2:
        valid_indices.append(i)

X_imputed = X_imputed[:, valid_indices]
eligible = [eligible[i] for i in valid_indices]

if len(eligible) < 2:
    raise ValueError("Too few usable sensors after cleaning.")

# Mutual information measures dependence on the binary label.
mi = mutual_info_classif(
    X_imputed,
    y.to_numpy(),
    discrete_features=False,
    random_state=42
)

# Standardized mean difference gives a second, interpretable ranking.
normal_mask = y.to_numpy() == 0
defect_mask = y.to_numpy() == 1

normal_mean = X_imputed[normal_mask].mean(axis=0)
defect_mean = X_imputed[defect_mask].mean(axis=0)
normal_std = X_imputed[normal_mask].std(axis=0)
defect_std = X_imputed[defect_mask].std(axis=0)

pooled_std = np.sqrt(
    (normal_std ** 2 + defect_std ** 2) / 2
)

effect_size = np.divide(
    np.abs(defect_mean - normal_mean),
    pooled_std,
    out=np.zeros_like(pooled_std),
    where=pooled_std > 1e-12
)

report = pd.DataFrame({
    "sensor": eligible,
    "missing_rate_train": [missing_rate[c] for c in eligible],
    "unique_values_train": [unique_count[c] for c in eligible],
    "mutual_information": mi,
    "standardized_mean_difference": effect_size,
})

report = report.sort_values(
    ["mutual_information", "standardized_mean_difference"],
    ascending=False
).reset_index(drop=True)

csv_path = OUTPUT_DIR / "feature_quality_report.csv"
json_path = OUTPUT_DIR / "feature_quality_summary.json"

report.to_csv(csv_path, index=False)

summary = {
    "dataset_rows": int(len(df)),
    "training_rows": int(len(train)),
    "training_normal": int((y == 0).sum()),
    "training_defects": int((y == 1).sum()),
    "sensor_columns": int(len(sensor_cols)),
    "eligible_sensors": int(len(eligible)),
    "excluded_sensors": int(len(sensor_cols) - len(eligible)),
    "top_20_sensors": report.head(20).to_dict(orient="records"),
    "warning": (
        "Sensor rankings are exploratory, based on training data only, "
        "and do not establish predictive performance. Confirm any choices "
        "on validation data before evaluating once on a held-out test set."
    ),
}

json_path.write_text(json.dumps(summary, indent=2, default=float))

print("=" * 60)
print("WAFERGUARD FEATURE QUALITY REPORT")
print("=" * 60)
print(f"Training rows: {len(train)}")
print(f"Training normal records: {(y == 0).sum()}")
print(f"Training defect records: {(y == 1).sum()}")
print(f"Original sensors: {len(sensor_cols)}")
print(f"Eligible sensors: {len(eligible)}")
print(f"Excluded sensors: {len(sensor_cols) - len(eligible)}")
print()
print("TOP 20 SENSORS — TRAINING DATA ONLY")
print(report.head(20).to_string(index=False))
print()
print(f"CSV report: {csv_path}")
print(f"Summary: {json_path}")
print()
print("No dataset, application model, or existing model artifact was changed.")
