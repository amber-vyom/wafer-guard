import pytest
import torch

from src.models.wafer_model import WaferAutoencoder


def test_forward_shapes():
    model = WaferAutoencoder(input_dim=20)
    reconstruction, logits, latent = model(torch.zeros(5, 20))
    assert reconstruction.shape == (5, 20)
    assert logits.shape == (5,)
    assert latent.shape == (5, 32)


def test_rejects_non_2d_input():
    model = WaferAutoencoder(input_dim=20)
    with pytest.raises(ValueError):
        model(torch.zeros(5, 1, 20))


def test_default_width_matches_saved_checkpoints():
    # Checkpoints saved before hidden_dim existed used 128 -> 64 -> latent.
    state = WaferAutoencoder(input_dim=20).state_dict()
    assert state["encoder.0.weight"].shape == (128, 20)
    assert state["encoder.2.weight"].shape == (64, 128)
    assert state["encoder.4.weight"].shape == (32, 64)
    assert state["decoder.4.weight"].shape == (20, 128)
    assert state["classifier.weight"].shape == (1, 32)


def test_hidden_width_is_configurable_and_loadable():
    source = WaferAutoencoder(input_dim=20, latent_dim=8, hidden_dim=64)
    clone = WaferAutoencoder(input_dim=20, latent_dim=8, hidden_dim=64)
    clone.load_state_dict(source.state_dict())  # strict: same keys and shapes
    with pytest.raises(RuntimeError):
        WaferAutoencoder(input_dim=20).load_state_dict(source.state_dict())
