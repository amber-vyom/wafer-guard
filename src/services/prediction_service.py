from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.wafer_model import WaferAutoencoder

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "wafer_model.pt"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.joblib"
FINAL_MODEL_PATH = (
    ROOT / "artifacts" / "experiments" / "wafer_guard_final_gb.joblib"
)

REQUIRED_CHECKPOINT_KEYS = ("model_state_dict", "input_dim", "threshold")
REQUIRED_PREPROCESSOR_KEYS = (
    "imputer", "scaler", "sensor_cols", "usable_cols", "flag_cols", "feature_cols"
)


def _require_keys(artifact, keys, path, kind):
    missing = [key for key in keys if key not in artifact]
    if missing:
        raise ValueError(
            f"{kind} {path.name} is missing {missing}. Re-run the training notebook; "
            "an older preprocessor can be fixed with "
            "scripts/maintenance/repair_preprocessor.py."
        )


def load_model(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH):
    """Load the PyTorch autoencoder and preprocessor (default: artifacts/).

    Returns ``(model, preprocessor, threshold)``. The model is in eval mode and is
    built from the checkpoint's ``input_dim``, ``latent_dim`` and ``hidden_dim``
    (older checkpoints without ``hidden_dim`` use 128). The artifact format is
    described in docs/artifacts.md.

    Raises ``FileNotFoundError`` if either file is missing and ``ValueError`` if a
    required key is absent.
    """
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True
    )
    preprocessor = joblib.load(preprocessor_path)
    _require_keys(checkpoint, REQUIRED_CHECKPOINT_KEYS, model_path, "Checkpoint")
    _require_keys(preprocessor, REQUIRED_PREPROCESSOR_KEYS, preprocessor_path, "Preprocessor")
    model = WaferAutoencoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint.get("latent_dim", 32),
        hidden_dim=checkpoint.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, preprocessor, float(checkpoint["threshold"])


def _to_dataframe(data):
    """Accept a DataFrame, Series (one row), dict (one row) or array-like."""
    if isinstance(data, pd.Series):
        return data.to_frame().T
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _prepare_sensor_data(data, sensor_cols):
    """Coerce sensors to numbers and apply the legacy PS-L rules.

    ``sensor_0 == 0.0`` marks a legacy wafer: both ``sensor_0`` and ``sensor_1`` are
    set to missing on those rows and four indicator columns are appended
    (``is_legacy_ps_l`` and one flag per ``sensor_1`` code 100/200/300, which only
    fires on legacy rows). Raises ``ValueError`` if a required sensor is absent.
    """
    data = _to_dataframe(data)
    missing = [col for col in sensor_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing[:10]}")

    raw = data[sensor_cols].copy()
    for column in sensor_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    legacy = raw["sensor_0"].eq(0.0).fillna(False)
    sensor1 = raw["sensor_1"].copy()
    raw["sensor_0"] = raw["sensor_0"].mask(legacy, np.nan)
    raw["sensor_1"] = sensor1.mask(legacy, np.nan)

    flags = {"is_legacy_ps_l": legacy.astype(float)}
    for code in (100.0, 200.0, 300.0):
        flags[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    return pd.concat(
        [raw, pd.DataFrame(flags, index=raw.index)], axis=1
    )


def prepare_features(data, preprocessor):
    """Turn raw sensor rows into the float32 feature matrix the model expects.

    Applies the legacy rules, imputes and scales ``usable_cols`` with the fitted
    preprocessor, then appends the unscaled ``flag_cols``. Raises ``ValueError`` on
    a feature-count mismatch or non-finite values.
    """
    raw = _prepare_sensor_data(data, preprocessor["sensor_cols"])
    usable_cols = preprocessor["usable_cols"]
    flag_cols = preprocessor["flag_cols"]

    imputed = preprocessor["imputer"].transform(raw[usable_cols])
    scaled = preprocessor["scaler"].transform(imputed)
    flags = raw[flag_cols].to_numpy(dtype=np.float32)
    features = np.column_stack([scaled, flags]).astype(np.float32)

    expected = len(preprocessor["feature_cols"])
    if features.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("Prepared features contain invalid values.")
    return features


def predict_anomaly(model, features, threshold=0.05):
    """Score a feature matrix.

    Returns a dict with ``probabilities`` (sigmoid of the classifier logit),
    ``predictions`` (1 where probability >= threshold), ``threshold`` and
    ``reconstruction_errors`` (per-row mean squared error). The probabilities are
    scores for ranking, not calibrated defect probabilities.
    """
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite 1D or 2D array.")

    tensor = torch.tensor(values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstruction, logits, latent = model(tensor)
        probabilities = torch.sigmoid(logits).reshape(-1).numpy()
        errors = torch.mean((tensor - reconstruction) ** 2, dim=1).numpy()

    predictions = (probabilities >= threshold).astype(int)
    return {
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "threshold": float(threshold),
        "reconstruction_errors": errors.tolist(),
    }


def predict_records(data):
    """Load the saved artifacts and score raw sensor rows end to end."""
    model, preprocessor, threshold = load_model()
    features = prepare_features(data, preprocessor)
    return predict_anomaly(model, features, threshold)


def load_final_model():
    """Load the tuned Gradient Boosting model."""
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Final model artifact not found: {FINAL_MODEL_PATH}")
    return joblib.load(FINAL_MODEL_PATH)


def prepare_final_features(data, artifact):
    """Apply the preprocessing saved with the tuned model."""
    raw = _prepare_sensor_data(data, artifact["sensor_cols"])
    usable_cols = artifact["usable_cols"]
    flag_cols = artifact["flag_cols"]

    imputed = artifact["imputer"].transform(raw[usable_cols])
    flags = raw[flag_cols].to_numpy(dtype=np.float32)
    features = np.column_stack([imputed, flags]).astype(np.float32)

    expected = artifact["input_dim"]
    if features.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("Prepared features contain invalid values.")
    return features


def predict_final_records(data):
    """Predict inspection priority with the tuned Gradient Boosting model."""
    artifact = load_final_model()
    features = prepare_final_features(data, artifact)
    probabilities = artifact["model"].predict_proba(features)[:, 1]
    threshold = float(artifact["threshold"])
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model_name": artifact["model_name"],
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "threshold": threshold,
        "parameters": artifact["parameters"],
    }
