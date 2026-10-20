"""
WaferGuard chronological cross-validation diagnostic.

Uses ONLY the first 70% chronological training period. The existing validation
(70%-80%) and test (80%-100%) periods are not used by this script.

Outputs only to artifacts/experiments/ and reports/experiments/.
It does not overwrite deployed model/preprocessor files.
Run from repository root:
    python src/models/compare_chronological_cv.py
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUT_DIR = ROOT / "artifacts" / "experiments"
REPORT_DIR = ROOT / "reports" / "experiments"
RANDOM_STATE = 42


def prepare_raw(part: pd.DataFrame, sensor_cols: list[str]):
    X = part[sensor_cols].copy()
    for col in sensor_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = (part["label"] == 1).astype(np.int64).to_numpy()

    legacy = X["sensor_0"].eq(0.0).fillna(False)
    X["is_legacy_ps_l"] = legacy.astype(float)
    X.loc[legacy, "sensor_0"] = np.nan

    sensor1 = X["sensor_1"].copy()
    X["sensor_1"] = sensor1.mask(legacy, np.nan)
    for code in (100.0, 200.0, 300.0):
        X[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)
    return X, y


def fit_transform_fold(train_df, val_df, sensor_cols):
    X_train_raw, y_train = prepare_raw(train_df, sensor_cols)
    X_val_raw, y_val = prepare_raw(val_df, sensor_cols)

    numeric_cols = [c for c in sensor_cols if c != "sensor_1"]
    usable_cols = [
        c for c in numeric_cols
        if X_train_raw[c].notna().any()
        and X_train_raw[c].nunique(dropna=True) > 1
    ]
    usable_cols.append("sensor_1")
    flag_cols = [
        "is_legacy_ps_l",
        "legacy_sensor_1_code_100",
        "legacy_sensor_1_code_200",
        "legacy_sensor_1_code_300",
    ]

    # Fit imputation and scaling on the fold's training rows only.
    imputer = SimpleImputer(strategy="median")
    xtr = imputer.fit_transform(X_train_raw[usable_cols])
    xva = imputer.transform(X_val_raw[usable_cols])
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xva = scaler.transform(xva)

    X_train = np.column_stack(
        [xtr, X_train_raw[flag_cols].to_numpy(dtype=np.float32)]
    ).astype(np.float32)
    X_val = np.column_stack(
        [xva, X_val_raw[flag_cols].to_numpy(dtype=np.float32)]
    ).astype(np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Non-finite values remain after fold preprocessing.")
    return X_train, y_train, X_val, y_val


def make_models(y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    weights = np.where(
        y_train == 1,
        len(y_train) / (2.0 * max(pos, 1)),
        len(y_train) / (2.0 * max(neg, 1)),
    )
    models = {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced", max_iter=4000, random_state=RANDOM_STATE
            ),
            None,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            None,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=200, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=RANDOM_STATE,
            ),
            weights,
        ),
    }
    return models


def safe_auc(metric_fn, y, score):
    if len(np.unique(y)) < 2:
        return None
    return float(metric_fn(y, score))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not {"sensor_0", "sensor_1"}.issubset(sensor_cols):
        raise ValueError("Dataset is missing sensor_0 or sensor_1.")

    # Restrict CV to the original first 70% training period.
    train_end = int(len(df) * 0.70)
    train_period = df.iloc[:train_end].reset_index(drop=True)
    y_all = (train_period["label"] == 1).astype(int).to_numpy()
    splitter = TimeSeriesSplit(n_splits=4)

    # Preallocate OOF predictions only for rows that appear in a validation fold.
    fold_records = []
    oof_by_model = {
        "logistic_regression": [],
        "random_forest": [],
        "hist_gradient_boosting": [],
    }

    print("=" * 72)
    print("WAFERGUARD CHRONOLOGICAL CROSS-VALIDATION")
    print("=" * 72)
    print(f"Using first 70% only: {len(train_period)} chronological rows")
    print(f"Defects in training period: {int(y_all.sum())}")
    print("The 70%-80% validation and final 20% test periods are not used.")
    print(f"scikit-learn: {sklearn.__version__}")

    for fold_no, (train_idx, val_idx) in enumerate(splitter.split(train_period), start=1):
        fold_train = train_period.iloc[train_idx].copy()
        fold_val = train_period.iloc[val_idx].copy()
        X_train, y_train, X_val, y_val = fit_transform_fold(
            fold_train, fold_val, sensor_cols
        )
        if len(np.unique(y_train)) < 2:
            warnings.warn(f"Fold {fold_no}: training fold lacks both classes; skipped.")
            continue

        print(
            f"\nFold {fold_no}: train={len(y_train)} "
            f"(defects={int(y_train.sum())}), validation={len(y_val)} "
            f"(defects={int(y_val.sum())}), features={X_train.shape[1]}"
        )
        models = make_models(y_train)
        for name, (model, weights) in models.items():
            if weights is None:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, sample_weight=weights)
            probabilities = model.predict_proba(X_val)[:, 1]
            ap = (
                float(average_precision_score(y_val, probabilities))
                if y_val.sum() > 0 else None
            )
            auc = safe_auc(roc_auc_score, y_val, probabilities)
            fold_records.append({
                "fold": fold_no,
                "model": name,
                "train_rows": int(len(y_train)),
                "train_defects": int(y_train.sum()),
                "validation_rows": int(len(y_val)),
                "validation_defects": int(y_val.sum()),
                "average_precision": ap,
                "roc_auc": auc,
            })
            for row_index, label, prob in zip(val_idx, y_val, probabilities):
                oof_by_model[name].append({
                    "row_index": int(row_index),
                    "label": int(label),
                    "probability": float(prob),
                    "fold": fold_no,
                })
            print(
                f"  {name:24s} AP={ap if ap is not None else float('nan'):.4f} "
                f"ROC-AUC={auc if auc is not None else float('nan'):.4f}"
            )

    folds_df = pd.DataFrame(fold_records)
    folds_path = REPORT_DIR / "chronological_cv_fold_metrics.csv"
    folds_df.to_csv(folds_path, index=False)

    summaries = []
    oof_rows = []
    for name, rows in oof_by_model.items():
        if not rows:
            continue
        oof = pd.DataFrame(rows).sort_values("row_index")
        y = oof["label"].to_numpy(dtype=int)
        prob = oof["probability"].to_numpy(dtype=float)
        ap = float(average_precision_score(y, prob)) if y.sum() else None
        auc = safe_auc(roc_auc_score, y, prob)
        summaries.append({
            "model": name,
            "oof_rows": int(len(y)),
            "oof_defects": int(y.sum()),
            "oof_prevalence": float(y.mean()),
            "oof_average_precision": ap,
            "oof_roc_auc": auc,
            "fold_ap_mean": float(
                folds_df.loc[folds_df["model"] == name, "average_precision"].dropna().mean()
            ),
            "fold_ap_std": float(
                folds_df.loc[folds_df["model"] == name, "average_precision"].dropna().std(ddof=0)
            ),
            "fold_auc_mean": float(
                folds_df.loc[folds_df["model"] == name, "roc_auc"].dropna().mean()
            ),
            "fold_auc_std": float(
                folds_df.loc[folds_df["model"] == name, "roc_auc"].dropna().std(ddof=0)
            ),
        })
        for row in rows:
            oof_rows.append({"model": name, **row})

    summary_df = pd.DataFrame(summaries)
    summary_path = REPORT_DIR / "chronological_cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    oof_path = REPORT_DIR / "chronological_cv_oof_predictions.csv"
    pd.DataFrame(oof_rows).to_csv(oof_path, index=False)

    report = {
        "purpose": "time-aware diagnostic on first 70% training period only",
        "dataset_rows": int(len(df)),
        "training_period_rows": int(len(train_period)),
        "training_period_defects": int(y_all.sum()),
        "n_splits": 4,
        "fold_metrics_csv": str(folds_path.relative_to(ROOT)),
        "summary_csv": str(summary_path.relative_to(ROOT)),
        "oof_predictions_csv": str(oof_path.relative_to(ROOT)),
        "summary": summaries,
        "caution": (
            "OOF scores are diagnostic, not a final unbiased estimate. Thresholds "
            "were not tuned here. Keep the original 70%-80% validation and final "
            "20% test period for later steps. Do not promote based on this report alone."
        ),
    }
    json_path = REPORT_DIR / "chronological_cv_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("CROSS-VALIDATION SUMMARY (TRAINING PERIOD ONLY)")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"\nFold metrics: {folds_path}")
    print(f"Summary CSV:  {summary_path}")
    print(f"OOF scores:   {oof_path}")
    print(f"JSON report:  {json_path}")
    print("\nNo deployed artifacts were overwritten.")


if __name__ == "__main__":
    main()
