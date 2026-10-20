
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.wafer_model import WaferAutoencoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
MODEL_PATH = ROOT / "artifacts" / "wafer_model.pt"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.joblib"
OUTPUT_DIR = ROOT / "artifacts" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RECALL_TARGET = 0.70
BATCH_SIZE = 128


def load_and_split():
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a label column.")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    if not sensor_cols:
        raise ValueError("No sensor columns were found.")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
        df = df.sort_values("timestamp", kind="mergesort")

    df = df[df["label"].isin([-1, 1])].reset_index(drop=True)

    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.80)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    y_train = (train["label"] == 1).astype(int).to_numpy()
    y_val = (val["label"] == 1).astype(int).to_numpy()
    y_test = (test["label"] == 1).astype(int).to_numpy()

    return sensor_cols, train, val, test, y_train, y_val, y_test


def prepare_autoencoder_features(data, preprocessor):
    """Apply the deployed preprocessing contract and PS-L handling."""
    data = data.copy()

    sensor_cols = preprocessor["sensor_cols"]
    usable_cols = preprocessor["usable_cols"]
    flag_cols = preprocessor["flag_cols"]
    feature_cols = preprocessor["feature_cols"]

    missing = [c for c in sensor_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing[:10]}")

    raw = data[sensor_cols].copy()

    for column in sensor_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    legacy = raw["sensor_0"].eq(0.0).fillna(False)
    sensor1 = raw["sensor_1"].copy()

    raw["sensor_0"] = raw["sensor_0"].mask(legacy, np.nan)
    raw["sensor_1"] = raw["sensor_1"].mask(legacy, np.nan)

    extra = {"is_legacy_ps_l": legacy.astype(float)}

    for code in (100.0, 200.0, 300.0):
        extra[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    raw = pd.concat(
        [raw, pd.DataFrame(extra, index=raw.index)],
        axis=1,
    )

    imputed = preprocessor["imputer"].transform(raw[usable_cols])
    scaled = preprocessor["scaler"].transform(imputed)
    flags = raw[flag_cols].to_numpy(dtype=np.float32)

    features = np.column_stack([scaled, flags]).astype(np.float32)

    if features.shape[1] != len(feature_cols):
        raise ValueError(
            f"Expected {len(feature_cols)} features; "
            f"got {features.shape[1]}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Prepared features contain non-finite values.")

    return features


def extract_features(model, x):
    """Extract latent features, reconstruction error, and AE probability."""
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.tensor(
                x[start:start + BATCH_SIZE],
                dtype=torch.float32,
            )

            reconstruction, logits, latent = model(batch)

            reconstruction_error = torch.mean(
                (batch - reconstruction) ** 2,
                dim=1,
            ).unsqueeze(1)

            probability = torch.sigmoid(logits).reshape(-1, 1)

            combined = torch.cat(
                [latent, reconstruction_error, probability],
                dim=1,
            )

            outputs.append(combined.cpu().numpy())

    if not outputs:
        raise ValueError("Cannot extract features from an empty dataset.")

    return np.vstack(outputs)


def score_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "predicted_defects": int(predictions.sum()),
    }


def choose_threshold(y_true, probabilities):
    """
    Choose a positive threshold using validation data only.

    First require the target recall. Among qualifying thresholds,
    maximize precision, then F1, then threshold. Never use zero.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    positive_scores = np.unique(
        probabilities[np.isfinite(probabilities) & (probabilities > 0)]
    )

    if len(positive_scores) == 0:
        fallback = 0.5
        return fallback, score_at_threshold(y_true, probabilities, fallback), []

    # Test thresholds at observed positive scores. Using >= means
    # samples tied at the selected score are all predicted positive.
    candidates = positive_scores.tolist()

    # Also test 0.5 if it is in the meaningful probability range.
    candidates.append(0.5)
    candidates = sorted(set(
        threshold for threshold in candidates
        if 0 < threshold <= 1
    ))

    evaluated = [
        score_at_threshold(y_true, probabilities, threshold)
        for threshold in candidates
    ]

    acceptable = [
        row for row in evaluated
        if row["recall"] >= RECALL_TARGET
    ]

    if acceptable:
        best = max(
            acceptable,
            key=lambda row: (
                row["precision"],
                row["f1"],
                row["threshold"],
            ),
        )
    else:
        # If no positive threshold reaches the recall target, choose
        # the threshold with highest validation F1 rather than zero.
        best = max(
            evaluated,
            key=lambda row: (
                row["f1"],
                row["precision"],
                row["threshold"],
            ),
        )

    return best["threshold"], best, evaluated


def describe_scores(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)

    return {
        "count": int(len(probabilities)),
        "min": float(np.min(probabilities)),
        "max": float(np.max(probabilities)),
        "mean": float(np.mean(probabilities)),
        "median": float(np.median(probabilities)),
        "unique_score_count": int(len(np.unique(probabilities))),
        "zero_score_count": int(np.sum(probabilities == 0)),
        "positive_score_count": int(np.sum(probabilities > 0)),
        "quantiles": {
            str(q): float(np.quantile(probabilities, q))
            for q in (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
        },
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Existing model and preprocessor artifacts are required."
        )

    (
        _sensor_cols,
        train,
        val,
        test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split()

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split must contain both classes.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    x_train = prepare_autoencoder_features(train, preprocessor)
    x_val = prepare_autoencoder_features(val, preprocessor)
    x_test = prepare_autoencoder_features(test, preprocessor)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    autoencoder = WaferAutoencoder(
        input_dim=checkpoint["input_dim"]
    )
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    # The deployed model and preprocessor are read-only.
    f_train = extract_features(autoencoder, x_train)
    f_val = extract_features(autoencoder, x_val)
    f_test = extract_features(autoencoder, x_test)

    forest = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(f_train, y_train)

    val_prob = forest.predict_proba(f_val)[:, 1]
    test_prob = forest.predict_proba(f_test)[:, 1]

    threshold, validation_choice, validation_candidates = choose_threshold(
        y_val, val_prob
    )

    test_metrics = score_at_threshold(y_test, test_prob, threshold)

    results = {
        "model": "Autoencoder latent features + Random Forest",
        "test_records": int(len(y_test)),
        "test_defects": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "feature_count": int(f_train.shape[1]),
        "threshold_selected_on_validation": float(threshold),
        "validation_selected_metrics": validation_choice,
        "validation_score_distribution": describe_scores(val_prob),
        "test_score_distribution": describe_scores(test_prob),
        "validation_average_precision": float(
            average_precision_score(y_val, val_prob)
        ),
        "validation_roc_auc": (
            float(roc_auc_score(y_val, val_prob))
            if len(np.unique(y_val)) == 2
            else None
        ),
        "test_average_precision": float(
            average_precision_score(y_test, test_prob)
        ),
        "test_roc_auc": (
            float(roc_auc_score(y_test, test_prob))
            if len(np.unique(y_test)) == 2
            else None
        ),
        "test_metrics_at_validation_threshold": test_metrics,
        "validation_threshold_candidates": validation_candidates,
        "warning": (
            "Exploratory test result only. The dataset test split has "
            "been examined in prior experiments. Do not call this an "
            "independent final evaluation."
        ),
    }

    output_path = OUTPUT_DIR / "hybrid_evaluation.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("WAFERGUARD HYBRID DIAGNOSTIC EXPERIMENT")
    print("=" * 60)
    print(f"Train records: {len(y_train)}; defects: {int(y_train.sum())}")
    print(f"Validation records: {len(y_val)}; defects: {int(y_val.sum())}")
    print(f"Test records: {len(y_test)}; defects: {int(y_test.sum())}")
    print(f"Feature count: {f_train.shape[1]}")
    print(f"Validation AP: {results['validation_average_precision']:.5f}")
    print(f"Validation ROC-AUC: {results['validation_roc_auc']}")
    print(f"Validation score range: {val_prob.min():.6f} to {val_prob.max():.6f}")
    print(f"Unique validation scores: {len(np.unique(val_prob))}")
    print(f"Zero validation scores: {int(np.sum(val_prob == 0))}")
    print(f"Selected positive threshold: {threshold:.6f}")
    print(f"Validation metrics: {validation_choice}")
    print(f"Test AP: {results['test_average_precision']:.5f}")
    print(f"Test ROC-AUC: {results['test_roc_auc']}")
    print(f"Test metrics: {test_metrics}")
    print(f"Results saved to: {output_path}")
    print("Deployed model and preprocessor were not modified.")


if __name__ == "__main__":
    main()
