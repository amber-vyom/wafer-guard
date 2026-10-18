
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

RANDOM_STATE = 42


def prepare_raw_features(part, sensor_cols):
    X = part[sensor_cols].copy()

    for col in sensor_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    legacy = X["sensor_0"].eq(0.0).fillna(False)
    sensor1 = X["sensor_1"].copy()

    X["sensor_0"] = X["sensor_0"].mask(legacy, np.nan)
    X["sensor_1"] = sensor1.mask(legacy, np.nan)

    X["is_legacy_ps_l"] = legacy.astype(float)

    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    y = (part["label"] == 1).astype(int).to_numpy()
    return X, y


def main():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    sensor_cols = [
        col for col in df.columns if col.startswith("sensor_")
    ]

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()

    X_train_raw, y_train = prepare_raw_features(train, sensor_cols)
    X_val_raw, y_val = prepare_raw_features(val, sensor_cols)

    flag_cols = [
        "is_legacy_ps_l",
        "legacy_sensor_1_code_100",
        "legacy_sensor_1_code_200",
        "legacy_sensor_1_code_300",
    ]

    numeric_cols = [c for c in sensor_cols if c != "sensor_1"]

    # Feature eligibility is determined from training data only.
    usable_cols = [
        col for col in numeric_cols
        if X_train_raw[col].notna().any()
        and X_train_raw[col].nunique(dropna=True) > 1
    ]
    usable_cols.append("sensor_1")

    # Missingness indicators are created after legacy sentinel handling.
    missing_train = X_train_raw[usable_cols].isna().astype(np.float32)
    missing_val = X_val_raw[usable_cols].isna().astype(np.float32)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(X_train_raw[usable_cols])
    val_imputed = imputer.transform(X_val_raw[usable_cols])

    train_scaled = scaler.fit_transform(train_imputed)
    val_scaled = scaler.transform(val_imputed)

    train_flags = X_train_raw[flag_cols].to_numpy(dtype=np.float32)
    val_flags = X_val_raw[flag_cols].to_numpy(dtype=np.float32)

    X_train_baseline = np.column_stack([train_scaled, train_flags])
    X_val_baseline = np.column_stack([val_scaled, val_flags])

    X_train_missing = np.column_stack([
        X_train_baseline,
        missing_train.to_numpy(),
    ])
    X_val_missing = np.column_stack([
        X_val_baseline,
        missing_val.to_numpy(),
    ])

    print("=" * 60)
    print("WAFERGUARD MISSINGNESS INDICATOR EXPERIMENT")
    print("=" * 60)
    print("Training rows:", len(train))
    print("Training defects:", int(y_train.sum()))
    print("Validation rows:", len(val))
    print("Validation defects:", int(y_val.sum()))
    print("Sensor value features:", len(usable_cols))
    print("Baseline feature count:", X_train_baseline.shape[1])
    print("Missingness-aware feature count:", X_train_missing.shape[1])

    experiments = [
        ("BASELINE", X_train_baseline, X_val_baseline),
        ("WITH MISSINGNESS INDICATORS", X_train_missing, X_val_missing),
    ]

    for name, Xtr, Xva in experiments:
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        model.fit(Xtr, y_train)
        probabilities = model.predict_proba(Xva)[:, 1]

        print(f"\n{name}")
        print("-" * 45)
        print("Average precision:", round(
            average_precision_score(y_val, probabilities), 5
        ))
        print("ROC-AUC:", round(
            roc_auc_score(y_val, probabilities), 5
        ))

        ranked = np.argsort(probabilities)[::-1]
        for k in (10, 20, 50):
            indices = ranked[:min(k, len(ranked))]
            print(
                f"Defects in top {len(indices)}:",
                int(y_val[indices].sum()),
            )

    print("\nExperiment complete.")
    print("No artifacts saved.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
