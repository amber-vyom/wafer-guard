"""
WaferGuard supervised candidate trainer.

Reads the existing dataset and writes candidate models ONLY under
artifacts/experiments/. It never loads the test split for model selection
and never overwrites the deployed PyTorch model or preprocessor.

Run from the repository root:
    python src/models/train_supervised_candidates.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUT_DIR = ROOT / "artifacts" / "experiments"
REPORT_DIR = ROOT / "reports" / "experiments"
RANDOM_STATE = 42


def split_features(part: pd.DataFrame, sensor_cols: list[str]):
    """Apply the existing WaferGuard sensor/sentinel rules."""
    X = part[sensor_cols].copy()
    y = (part["label"] == 1).astype(np.int64).to_numpy()

    # Existing project rule: sensor_0 == 0 marks legacy PS-L records.
    legacy = X["sensor_0"].eq(0.0).fillna(False)
    X["is_legacy_ps_l"] = legacy.astype(float)
    X.loc[legacy, "sensor_0"] = np.nan

    # sensor_1 is continuous for non-legacy records; legacy codes are flags.
    sensor1 = X["sensor_1"].copy()
    X["sensor_1"] = sensor1.mask(legacy, np.nan)
    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    for col in sensor_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray):
    """Select threshold using validation F1; ties prefer precision then recall."""
    candidates = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 197),
        np.asarray(probabilities, dtype=float),
    ]))
    best = None
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(np.int64)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        key = (float(f1), float(precision), float(recall), float(threshold))
        if best is None or key > best["key"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "key": key,
            }
    return best


def metrics_at_threshold(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "predicted_defects": int(pred.sum()),
    }
    if len(np.unique(y_true)) == 2:
        result["average_precision"] = float(average_precision_score(y_true, probabilities))
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["average_precision"] = None
        result["roc_auc"] = None
    return result


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if "timestamp" not in df or "label" not in df:
        raise ValueError("Dataset must contain 'timestamp' and 'label' columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    required = {"sensor_0", "sensor_1"}
    if not required.issubset(sensor_cols):
        raise ValueError(f"Required sensors missing: {sorted(required - set(sensor_cols))}")

    # Same chronological 70/10/20 boundaries as the current project.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    X_train_raw, y_train = split_features(train_df, sensor_cols)
    X_val_raw, y_val = split_features(val_df, sensor_cols)

    # Feature selection is based on training rows only.
    numeric_sensor_cols = [c for c in sensor_cols if c != "sensor_1"]
    usable_cols = [
        col for col in numeric_sensor_cols
        if X_train_raw[col].notna().any()
        and X_train_raw[col].nunique(dropna=True) > 1
    ]
    usable_cols.append("sensor_1")
    flag_cols = [
        "is_legacy_ps_l",
        "legacy_sensor_1_code_100",
        "legacy_sensor_1_code_200",
        "legacy_sensor_1_code_300",
    ]
    feature_cols = usable_cols + flag_cols

    # Fit imputer/scaler on training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_cont = imputer.fit_transform(X_train_raw[usable_cols])
    val_cont = imputer.transform(X_val_raw[usable_cols])
    train_scaled = scaler.fit_transform(train_cont)
    val_scaled = scaler.transform(val_cont)

    X_train = np.column_stack([
        train_scaled, X_train_raw[flag_cols].to_numpy(dtype=np.float32)
    ]).astype(np.float32)
    X_val = np.column_stack([
        val_scaled, X_val_raw[flag_cols].to_numpy(dtype=np.float32)
    ]).astype(np.float32)

    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite feature values remain after preprocessing.")
    if X_train.shape[1] != len(feature_cols):
        raise ValueError("Feature count does not match feature metadata.")
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training split must contain both classes.")
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        warnings.warn("Validation split has only one class; metrics are unreliable.")

    print("=" * 72)
    print("WAFERGUARD SUPERVISED CANDIDATE TRAINING (NO TEST EVALUATION)")
    print("=" * 72)
    print(f"Dataset: {len(df)} rows; {len(sensor_cols)} sensor columns")
    print(f"Train: {len(y_train)} rows; defects={int(y_train.sum())}")
    print(f"Validation: {len(y_val)} rows; defects={int(y_val.sum())}")
    print(f"Features: {X_train.shape[1]}")
    print(f"scikit-learn: {sklearn.__version__}")
    print("Test split is deliberately not loaded or evaluated by this script.")

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=4000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
    }

    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    sample_weights = np.where(
        y_train == 1,
        len(y_train) / (2.0 * max(n_positive, 1)),
        len(y_train) / (2.0 * max(n_negative, 1)),
    )

    results = []
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == "hist_gradient_boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_prob)["threshold"]
        val_metrics = metrics_at_threshold(y_val, val_prob, threshold)

        artifact_path = OUT_DIR / f"wafer_{name}_candidate.joblib"
        joblib.dump({
            "model": model,
            "model_type": name,
            "sensor_cols": sensor_cols,
            "usable_cols": usable_cols,
            "flag_cols": flag_cols,
            "feature_cols": feature_cols,
            "imputer": imputer,
            "scaler": scaler,
            "threshold": threshold,
            "random_state": RANDOM_STATE,
            "split": "chronological 70/10/20; threshold selected on validation",
            "input_dim": int(X_train.shape[1]),
        }, artifact_path)

        row = {
            "model": name,
            "threshold": threshold,
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_average_precision": val_metrics["average_precision"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_tp": val_metrics["tp"],
            "validation_fp": val_metrics["fp"],
            "validation_fn": val_metrics["fn"],
            "validation_tn": val_metrics["tn"],
            "candidate_artifact": str(artifact_path.relative_to(ROOT)),
        }
        results.append(row)
        ap = row["validation_average_precision"]
        auc = row["validation_roc_auc"]
        print(
            f"Validation: precision={row['validation_precision']:.3f}, "
            f"recall={row['validation_recall']:.3f}, F1={row['validation_f1']:.3f}, "
            f"AP={ap if ap is not None else float('nan'):.4f}, "
            f"ROC-AUC={auc if auc is not None else float('nan'):.4f}"
        )
        print(
            f"Confusion counts: TP={row['validation_tp']} FP={row['validation_fp']} "
            f"FN={row['validation_fn']} TN={row['validation_tn']}"
        )
        print(f"Candidate saved: {artifact_path}")

    report = {
        "purpose": "validation-only candidate comparison; no test evaluation",
        "dataset_rows": int(len(df)),
        "sensor_columns": int(len(sensor_cols)),
        "train_rows": int(len(y_train)),
        "train_defects": int(y_train.sum()),
        "validation_rows": int(len(y_val)),
        "validation_defects": int(y_val.sum()),
        "feature_count": int(X_train.shape[1]),
        "scikit_learn_version": sklearn.__version__,
        "candidates": results,
        "caution": (
            "Validation contains few defects, so threshold and metrics are unstable. "
            "Do not promote a candidate based on this report alone. The test split "
            "must remain untouched until candidate/threshold selection is complete."
        ),
    }
    json_path = REPORT_DIR / "supervised_candidate_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = REPORT_DIR / "supervised_candidate_comparison.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print("\n" + "=" * 72)
    print("CANDIDATE COMPARISON (VALIDATION ONLY)")
    print("=" * 72)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nJSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print("\nNo deployed artifacts were overwritten.")
    print("Do not treat the validation winner as proof of final test performance.")


if __name__ == "__main__":
    main()
