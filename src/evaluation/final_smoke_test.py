
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.data_service import load_dataset
from src.services.prediction_service import (
    load_model,
    prepare_features,
    predict_anomaly,
    predict_records,
)

passed = 0
failed = 0


def check(name, function):
    global passed, failed

    try:
        result = function()
        print(f"[PASS] {name}")
        if result is not None:
            print(f"       {result}")
        passed += 1
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failed += 1


def test_dataset():
    df = load_dataset()

    if df.empty:
        raise AssertionError("Dataset is empty.")

    sensors = [c for c in df.columns if c.startswith("sensor_")]

    if not sensors:
        raise AssertionError("No sensor_* columns found.")

    if "label" not in df.columns:
        raise AssertionError("Expected label column is missing.")

    return f"{len(df)} rows; {len(sensors)} sensor columns"


def test_model_artifacts():
    model, preprocessor, threshold = load_model()

    if not isinstance(threshold, (int, float)):
        raise AssertionError("Threshold is not numeric.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise AssertionError(f"Invalid threshold: {threshold}")

    for key in ("sensor_cols", "usable_cols", "flag_cols", "feature_cols"):
        if key not in preprocessor:
            raise AssertionError(f"Preprocessor missing key: {key}")

    return (
        f"Model loaded; threshold={threshold:.6f}; "
        f"expected features={len(preprocessor['feature_cols'])}"
    )


def test_single_record_prediction():
    df = load_dataset()
    result = predict_records(df.iloc[[0]])

    if len(result["probabilities"]) != 1:
        raise AssertionError("Expected one probability.")

    if len(result["predictions"]) != 1:
        raise AssertionError("Expected one prediction.")

    if len(result["reconstruction_errors"]) != 1:
        raise AssertionError("Expected one reconstruction error.")

    probability = float(result["probabilities"][0])
    prediction = int(result["predictions"][0])
    error = float(result["reconstruction_errors"][0])
    threshold = float(result["threshold"])

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise AssertionError(f"Invalid probability: {probability}")

    if prediction not in (0, 1):
        raise AssertionError(f"Invalid prediction: {prediction}")

    if not np.isfinite(error) or error < 0:
        raise AssertionError(f"Invalid reconstruction error: {error}")

    expected = int(probability >= threshold)
    if prediction != expected:
        raise AssertionError("Prediction does not match the threshold.")

    return (
        f"probability={probability:.6f}; prediction={prediction}; "
        f"reconstruction_error={error:.6f}"
    )


def test_batch_prediction():
    df = load_dataset()
    batch = df.iloc[:3]
    result = predict_records(batch)

    for key in ("probabilities", "predictions", "reconstruction_errors"):
        if len(result[key]) != len(batch):
            raise AssertionError(
                f"{key} length {len(result[key])} "
                f"does not match batch size {len(batch)}."
            )

    probabilities = np.asarray(result["probabilities"], dtype=float)
    errors = np.asarray(result["reconstruction_errors"], dtype=float)

    if not np.isfinite(probabilities).all():
        raise AssertionError("Batch probabilities contain invalid values.")

    if not np.isfinite(errors).all():
        raise AssertionError("Batch reconstruction errors are invalid.")

    if not np.isin(result["predictions"], [0, 1]).all():
        raise AssertionError("Batch predictions are not binary.")

    return f"{len(batch)} records predicted successfully"


def test_missing_sensor_rejected():
    df = load_dataset()
    _, preprocessor, _ = load_model()

    required = preprocessor["sensor_cols"]

    if not required:
        raise AssertionError("No required sensor columns recorded.")

    missing_column = required[0]
    incomplete = df.iloc[[0]].drop(columns=[missing_column])

    try:
        prepare_features(incomplete, preprocessor)
    except ValueError as exc:
        if "Missing sensor columns" in str(exc):
            return f"Correctly rejected missing {missing_column}"
        raise

    raise AssertionError("Input without a required sensor was accepted.")


def test_invalid_numeric_values_handled():
    df = load_dataset()
    _, preprocessor, _ = load_model()

    required = preprocessor["sensor_cols"]
    sensor = required[0]
    invalid = df.iloc[[0]].copy()
    invalid[sensor] = "not-a-number"

    try:
        features = prepare_features(invalid, preprocessor)
    except ValueError:
        return f"Invalid value in {sensor} rejected safely"

    if not np.isfinite(features).all():
        raise AssertionError("Prepared features contain non-finite values.")

    return (
        f"Non-numeric {sensor} was converted and preprocessing "
        "returned finite features"
    )


def test_prediction_output_shape():
    df = load_dataset()
    model, preprocessor, threshold = load_model()
    features = prepare_features(df.iloc[:2], preprocessor)
    result = predict_anomaly(model, features, threshold)

    if len(result["probabilities"]) != 2:
        raise AssertionError("Unexpected output shape.")

    return "Direct model inference returned two records"


if __name__ == "__main__":
    print("=" * 64)
    print("WAFERGUARD FINAL SMOKE TEST")
    print("=" * 64)

    checks = [
        ("Dataset can be loaded", test_dataset),
        ("Existing model and preprocessor load", test_model_artifacts),
        ("Single-record prediction", test_single_record_prediction),
        ("Batch prediction", test_batch_prediction),
        ("Missing required sensor is rejected", test_missing_sensor_rejected),
        ("Invalid numeric sensor value is handled", test_invalid_numeric_values_handled),
        ("Direct inference output shape", test_prediction_output_shape),
    ]

    for name, function in checks:
        check(name, function)

    print("\n" + "=" * 64)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 64)

    if failed:
        sys.exit(1)
