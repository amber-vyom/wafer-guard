
import torch
from torch import nn


class WaferAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
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

        return reconstruction, logits, z
