
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.services.prediction_service import (
    load_model,
    prepare_features,
)
from src.services.data_service import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

df = load_dataset(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

test = df.iloc[int(len(df) * 0.80):].copy()

model, preprocessor, threshold = load_model()
features = prepare_features(test, preprocessor)

tensor = torch.tensor(features, dtype=torch.float32)

model.eval()

with torch.no_grad():
    reconstructed, logits, latent = model(tensor)
    errors = torch.mean((tensor - reconstructed) ** 2, dim=1).numpy()
    probabilities = torch.sigmoid(logits).numpy()

labels = (test["label"] == 1).to_numpy()

print("\nFEATURE CHECK")
print("Feature shape:", features.shape)
print("All features finite:", bool(np.isfinite(features).all()))
print("Feature minimum:", float(features.min()))
print("Feature maximum:", float(features.max()))
print("Feature mean:", float(features.mean()))
print("Feature standard deviation:", float(features.std()))

print("\nRECONSTRUCTION CHECK")
print("Reconstruction finite:", bool(torch.isfinite(reconstructed).all()))
print("Reconstruction minimum:", float(reconstructed.min()))
print("Reconstruction maximum:", float(reconstructed.max()))
print("Error minimum:", float(errors.min()))
print("Error median:", float(np.median(errors)))
print("Error maximum:", float(errors.max()))

print("\nHIGHEST ERROR RECORDS")
for index in np.argsort(errors)[-5:][::-1]:
    print(
        "Row:", int(index),
        "Label:", int(test.iloc[index]["label"]),
        "Error:", float(errors[index]),
        "Probability:", float(probabilities[index]),
    )

print("\nERROR BY LABEL")
for label, name in [(False, "Normal"), (True, "Defective")]:
    group = errors[labels == label]
    print(
        name,
        "count:", len(group),
        "mean:", float(group.mean()),
        "median:", float(np.median(group)),
        "maximum:", float(group.max()),
    )
