
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUTPUT_DIR = ROOT / "artifacts" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

if "label" not in df.columns:
    raise ValueError("Dataset has no label column.")

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

df = df[df["label"].isin([-1, 1])].copy().reset_index(drop=True)

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]

X_train_raw = train[sensor_cols].apply(pd.to_numeric, errors="coerce")
X_val_raw = val[sensor_cols].apply(pd.to_numeric, errors="coerce")

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()

missing_rate = X_train_raw.isna().mean()
unique_count = X_train_raw.nunique(dropna=True)

eligible = [
    c for c in sensor_cols
    if unique_count[c] >= 2 and missing_rate[c] <= 0.50
]

X_train_raw = X_train_raw[eligible]
X_val_raw = X_val_raw[eligible]

imputer = SimpleImputer(strategy="median")
X_train_all = imputer.fit_transform(X_train_raw)
X_val_all = imputer.transform(X_val_raw)

valid = [
    i for i in range(X_train_all.shape[1])
    if np.isfinite(X_train_all[:, i]).all()
    and np.unique(X_train_all[:, i]).size >= 2
]

X_train_all = X_train_all[:, valid]
X_val_all = X_val_all[:, valid]
eligible = [eligible[i] for i in valid]

mi = mutual_info_classif(
    X_train_all,
    y_train,
    random_state=42
)

ranking = np.argsort(mi)[::-1]
results = []

for count in [20, 50, 100]:
    chosen_indices = ranking[:min(count, len(ranking))]
    chosen_sensors = [eligible[i] for i in chosen_indices]

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_all[:, chosen_indices], y_train)
    probabilities = model.predict_proba(
        X_val_all[:, chosen_indices]
    )[:, 1]

    ap = average_precision_score(y_val, probabilities)
    auc = roc_auc_score(y_val, probabilities)

    results.append({
        "feature_count": count,
        "validation_average_precision": float(ap),
        "validation_roc_auc": float(auc),
        "validation_defects": int(y_val.sum()),
        "selected_sensors": chosen_sensors
    })

    print(
        f"Top {count:3d} sensors | "
        f"Validation AP: {ap:.5f} | "
        f"Validation ROC-AUC: {auc:.5f}"
    )

results.sort(
    key=lambda item: item["validation_average_precision"],
    reverse=True
)

output_path = OUTPUT_DIR / "feature_set_comparison.json"
output_path.write_text(json.dumps(results, indent=2))

print("\nBEST VALIDATION RESULT")
print(f"Feature count: {results[0]['feature_count']}")
print(f"Validation AP: {results[0]['validation_average_precision']:.5f}")
print(f"Validation ROC-AUC: {results[0]['validation_roc_auc']:.5f}")
print(f"Saved results: {output_path}")
print("\nNo application model or existing model artifact was changed.")
print("Validation contains only a small number of defects; interpret results cautiously.")
