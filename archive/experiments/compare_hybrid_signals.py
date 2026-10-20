
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
        raise ValueError("No sensor columns found.")

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

    return train, val, test, y_train, y_val, y_test


def prepare_features(data, preprocessor):
    """Apply existing preprocessing and PS-L handling."""
    data = data.copy()

    sensor_cols = preprocessor["sensor_cols"]
    usable_cols = preprocessor["usable_cols"]
    flag_cols = preprocessor["flag_cols"]
    feature_cols = preprocessor["feature_cols"]

    missing = [c for c in sensor_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing[:10]}")

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
            f"Expected {len(feature_cols)} features, got {features.shape[1]}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Prepared features contain non-finite values.")

    return features


def extract_signals(model, x):
    """Return classifier probabilities, reconstruction errors, and latents."""
    model.eval()
    probabilities = []
    errors = []
    latent_vectors = []

    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.tensor(
                x[start:start + BATCH_SIZE],
                dtype=torch.float32,
            )

            reconstruction, logits, latent = model(batch)

            error = torch.mean(
                (batch - reconstruction) ** 2,
                dim=1,
            )
            probability = torch.sigmoid(logits.reshape(-1))

            probabilities.append(probability.cpu().numpy())
            errors.append(error.cpu().numpy())
            latent_vectors.append(latent.cpu().numpy())

    if not probabilities:
        raise ValueError("Cannot process an empty dataset.")

    return (
        np.concatenate(probabilities),
        np.concatenate(errors),
        np.vstack(latent_vectors),
    )


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if len(np.unique(y_true)) == 2
            else None
        ),
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


def choose_threshold(y_true, scores):
    """Choose a positive threshold using validation data only."""
    scores = np.asarray(scores, dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values.")

    positive_scores = np.unique(scores[scores > 0])

    if len(positive_scores) == 0:
        raise ValueError(
            "All validation scores are zero; no useful threshold exists."
        )

    candidates = positive_scores.tolist()
    candidates.append(float(np.nextafter(positive_scores.max(), np.inf)))

    evaluated = [
        calculate_metrics(y_true, scores, threshold)
        for threshold in sorted(set(candidates))
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
        reason = (
            f"Highest precision among thresholds with recall >= "
            f"{RECALL_TARGET:.2f}"
        )
    else:
        best = max(
            evaluated,
            key=lambda row: (
                row["f1"],
                row["precision"],
                row["threshold"],
            ),
        )
        reason = (
            "Target recall not reached; selected highest validation F1"
        )

    return best["threshold"], reason, evaluated


def evaluate_signal(name, y_val, val_scores, y_test, test_scores):
    threshold, reason, candidates = choose_threshold(y_val, val_scores)

    return {
        "name": name,
        "threshold_selected_on_validation": float(threshold),
        "threshold_selection_reason": reason,
        "validation_metrics": calculate_metrics(
            y_val, val_scores, threshold
        ),
        "test_metrics": calculate_metrics(
            y_test, test_scores, threshold
        ),
        "validation_score_distribution": {
            "min": float(np.min(val_scores)),
            "median": float(np.median(val_scores)),
            "max": float(np.max(val_scores)),
            "unique_scores": int(len(np.unique(val_scores))),
            "zero_scores": int(np.sum(val_scores == 0)),
        },
        "test_score_distribution": {
            "min": float(np.min(test_scores)),
            "median": float(np.median(test_scores)),
            "max": float(np.max(test_scores)),
            "unique_scores": int(len(np.unique(test_scores))),
            "zero_scores": int(np.sum(test_scores == 0)),
        },
        "validation_threshold_candidates": candidates,
    }


def main():
    for path in (DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train, val, test, y_train, y_val, y_test = load_and_split()

    if len(np.unique(y_train)) != 2:
        raise ValueError("Training split must contain both classes.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    x_train = prepare_features(train, preprocessor)
    x_val = prepare_features(val, preprocessor)
    x_test = prepare_features(test, preprocessor)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    autoencoder = WaferAutoencoder(input_dim=checkpoint["input_dim"])
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    train_prob, train_error, train_latent = extract_signals(
        autoencoder, x_train
    )
    val_prob, val_error, val_latent = extract_signals(autoencoder, x_val)
    test_prob, test_error, test_latent = extract_signals(autoencoder, x_test)

    # Train the experimental forest only on training records and labels.
    train_hybrid = np.column_stack([
        train_latent, train_error, train_prob
    ])
    val_hybrid = np.column_stack([
        val_latent, val_error, val_prob
    ])
    test_hybrid = np.column_stack([
        test_latent, test_error, test_prob
    ])

    forest = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(train_hybrid, y_train)

    val_rf = forest.predict_proba(val_hybrid)[:, 1]
    test_rf = forest.predict_proba(test_hybrid)[:, 1]

    signals = [
        evaluate_signal(
            "Autoencoder classifier probability",
            y_val, val_prob, y_test, test_prob,
        ),
        evaluate_signal(
            "Autoencoder reconstruction error",
            y_val, val_error, y_test, test_error,
        ),
        evaluate_signal(
            "Autoencoder features + Random Forest",
            y_val, val_rf, y_test, test_rf,
        ),
    ]

    results = {
        "experiment": "Controlled comparison of three anomaly signals",
        "split": "Chronological 70% train / 10% validation / 20% test",
        "train_records": int(len(y_train)),
        "train_defects": int(y_train.sum()),
        "validation_records": int(len(y_val)),
        "validation_defects": int(y_val.sum()),
        "test_records": int(len(y_test)),
        "test_defects": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "recall_target": RECALL_TARGET,
        "signals": signals,
        "limitations": [
            "The dataset test split was examined in previous experiments.",
            "Test results are exploratory, not an independent final benchmark.",
            "Thresholds are selected using validation labels only.",
            "The deployed Autoencoder and preprocessor are not modified.",
            "The experimental Random Forest is not saved for deployment.",
        ],
    }

    output_path = OUTPUT_DIR / "signal_comparison.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("WAFERGUARD: CONTROLLED SIGNAL COMPARISON")
    print("=" * 65)
    print(f"Train: {len(y_train)} records; defects={int(y_train.sum())}")
    print(f"Validation: {len(y_val)} records; defects={int(y_val.sum())}")
    print(f"Test: {len(y_test)} records; defects={int(y_test.sum())}")

    for result in signals:
        v = result["validation_metrics"]
        t = result["test_metrics"]

        print("\n" + "-" * 65)
        print(result["name"])
        print(
            f"Validation threshold: "
            f"{result['threshold_selected_on_validation']:.8g}"
        )
        print(f"Selection: {result['threshold_selection_reason']}")
        print(
            f"Validation: AP={v['average_precision']:.5f}, "
            f"AUC={v['roc_auc']}, Precision={v['precision']:.5f}, "
            f"Recall={v['recall']:.5f}, F1={v['f1']:.5f}"
        )
        print(
            f"Test: AP={t['average_precision']:.5f}, "
            f"AUC={t['roc_auc']}, Precision={t['precision']:.5f}, "
            f"Recall={t['recall']:.5f}, F1={t['f1']:.5f}"
        )
        print(
            f"Test confusion matrix: TN={t['TN']}, FP={t['FP']}, "
            f"FN={t['FN']}, TP={t['TP']}"
        )
        print("Validation score distribution:",
              result["validation_score_distribution"])

    print(f"\nResults saved to: {output_path}")
    print("Deployed model and preprocessor were not modified.")
    print("Results are exploratory, not an independent final benchmark.")


if __name__ == "__main__":
    main()
