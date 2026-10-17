
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.wafer_model import WaferAutoencoder

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "wafer_model.pt"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = WaferAutoencoder(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, preprocessor, float(checkpoint["threshold"])


def prepare_features(data, preprocessor):
    if isinstance(data, pd.Series):
        data = data.to_frame().T
    elif isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    sensor_cols = preprocessor["sensor_cols"]
    usable_cols = preprocessor["usable_cols"]
    flag_cols = preprocessor["flag_cols"]

    missing = [c for c in sensor_cols if c not in data.columns]

    if missing:
        raise ValueError(f"Missing sensor columns: {missing[:10]}")

    raw = data[sensor_cols].copy()

    for column in sensor_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    legacy = raw["sensor_0"].eq(0.0).fillna(False)
    sensor1 = raw["sensor_1"].copy()

    raw["sensor_0"] = raw["sensor_0"].mask(legacy, np.nan)
    raw["sensor_1"] = sensor1.mask(legacy, np.nan)

    extra_columns = {
        "is_legacy_ps_l": legacy.astype(float)
    }

    for code in (100.0, 200.0, 300.0):
        extra_columns[f"legacy_sensor_1_code_{int(code)}"] = (
            legacy & sensor1.eq(code)
        ).astype(float)

    raw = pd.concat(
        [raw, pd.DataFrame(extra_columns, index=raw.index)],
        axis=1,
    )

    imputed = preprocessor["imputer"].transform(raw[usable_cols])
    scaled = preprocessor["scaler"].transform(imputed)
    flags = raw[flag_cols].to_numpy(dtype=np.float32)

    features = np.column_stack([scaled, flags]).astype(np.float32)

    expected_features = len(preprocessor["feature_cols"])

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {features.shape[1]}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Prepared features contain invalid values.")

    return features


def predict_anomaly(model, features, threshold=0.05):
    values = np.asarray(features, dtype=np.float32)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError("Features must be a 1D or 2D array.")

    if not np.isfinite(values).all():
        raise ValueError("Features contain invalid values.")

    tensor = torch.tensor(values, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        reconstruction, logits, latent = model(tensor)
        probabilities = torch.sigmoid(logits).numpy()
        errors = torch.mean(
            (tensor - reconstruction) ** 2,
            dim=1,
        ).numpy()

    return {
        "probabilities": probabilities.tolist(),
        "predictions": (probabilities >= threshold).astype(int).tolist(),
        "threshold": float(threshold),
        "reconstruction_errors": errors.tolist(),
    }


def predict_records(data):
    model, preprocessor, threshold = load_model()
    features = prepare_features(data, preprocessor)
    return predict_anomaly(model, features, threshold)
