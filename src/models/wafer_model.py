
import torch
from torch import nn


class WaferAutoencoder(nn.Module):
    """Supervised autoencoder shared by the training notebook and the app.

    ``hidden_dim`` defaults to 128 (128 -> 64 -> latent), so checkpoints saved
    before the width was configurable load unchanged.
    """

    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        mid_dim = hidden_dim // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                f"Expected [batch_size, features], got {tuple(x.shape)}"
            )

        z = self.encoder(x)
        reconstruction = self.decoder(z)
        logits = self.classifier(z).squeeze(-1)

        # Guard against silent broadcasting in reconstruction/classification losses.
        if reconstruction.shape != x.shape:
            raise ValueError(
                f"Reconstruction {tuple(reconstruction.shape)} does not match "
                f"input {tuple(x.shape)}"
            )
        if logits.shape != (x.shape[0],):
            raise ValueError(
                f"Expected logits of shape ({x.shape[0]},), got {tuple(logits.shape)}"
            )

        return reconstruction, logits, z
