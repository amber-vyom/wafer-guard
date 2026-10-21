"""Train/serve parity: what the notebook saves must be what the app loads."""

import joblib
import numpy as np
import pytest
import torch

from src.models.wafer_model import WaferAutoencoder
from src.services.prediction_service import (
    load_model,
    predict_anomaly,
    prepare_features,
)

THRESHOLD = 0.4


def save_artifacts(directory, preprocessor, hidden_dim=32, latent_dim=8):
    input_dim = len(preprocessor["feature_cols"])
    torch.manual_seed(0)
    model = WaferAutoencoder(input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    model_path = directory / "wafer_model.pt"
    preprocessor_path = directory / "preprocessor.joblib"
    # Same keys the notebook writes.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "threshold": THRESHOLD,
            "usable_cols": list(preprocessor["usable_cols"]),
            "flag_cols": list(preprocessor["flag_cols"]),
        },
        model_path,
    )
    joblib.dump(preprocessor, preprocessor_path)
    return model, model_path, preprocessor_path


def test_saved_artifacts_reproduce_direct_inference(tmp_path, raw_frame, preprocessor):
    model, model_path, preprocessor_path = save_artifacts(tmp_path, preprocessor)

    loaded, loaded_preprocessor, threshold = load_model(model_path, preprocessor_path)
    assert threshold == THRESHOLD

    features = prepare_features(raw_frame, loaded_preprocessor)
    result = predict_anomaly(loaded, features, threshold)

    model.eval()
    with torch.no_grad():
        _, logits, _ = model(torch.tensor(features))
    expected = torch.sigmoid(logits).numpy()

    np.testing.assert_allclose(result["probabilities"], expected, atol=1e-6)
    np.testing.assert_array_equal(result["predictions"], (expected >= THRESHOLD).astype(int))
    assert len(result["reconstruction_errors"]) == len(raw_frame)


def test_old_checkpoints_without_hidden_dim_still_load(tmp_path, preprocessor):
    input_dim = len(preprocessor["feature_cols"])
    legacy_checkpoint = {
        "model_state_dict": WaferAutoencoder(input_dim).state_dict(),
        "input_dim": input_dim,
        "latent_dim": 32,
        "threshold": THRESHOLD,
    }
    torch.save(legacy_checkpoint, tmp_path / "wafer_model.pt")
    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    model, _, _ = load_model(tmp_path / "wafer_model.pt", tmp_path / "preprocessor.joblib")
    assert isinstance(model, WaferAutoencoder)


def test_incomplete_preprocessor_gets_a_clear_error(tmp_path, preprocessor):
    _, model_path, preprocessor_path = save_artifacts(tmp_path, preprocessor)
    incomplete = {k: v for k, v in preprocessor.items() if k not in {"sensor_cols", "feature_cols"}}
    joblib.dump(incomplete, preprocessor_path)
    with pytest.raises(ValueError, match="sensor_cols"):
        load_model(model_path, preprocessor_path)


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pt", tmp_path / "nope.joblib")


def test_wrong_feature_count_is_rejected(raw_frame, preprocessor):
    broken = dict(preprocessor, feature_cols=list(preprocessor["feature_cols"]) + ["extra"])
    with pytest.raises(ValueError, match="Expected"):
        prepare_features(raw_frame, broken)
