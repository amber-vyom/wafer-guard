
from pathlib import Path

import joblib
import numpy as np
import torch

from src.models.compare_hybrid_signals import (
    DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    load_and_split,
    prepare_features,
    extract_signals,
)
from src.models.wafer_model import WaferAutoencoder


def describe_scores(name, labels, scores):
    print(f"\n{name}")

    for label, description in [
        (0, "NORMAL"),
        (1, "DEFECT"),
    ]:
        values = scores[labels == label]

        if len(values) == 0:
            print(f"  {description}: no records")
            continue

        print(
            f"  {description}: n={len(values)}, "
            f"min={values.min():.10g}, "
            f"median={np.median(values):.10g}, "
            f"mean={values.mean():.10g}, "
            f"max={values.max():.10g}"
        )


def main():
    for path in (DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    _, val, test, _, y_val, y_test = load_and_split()
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    x_val = prepare_features(val, preprocessor)
    x_test = prepare_features(test, preprocessor)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = WaferAutoencoder(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_probability, val_error, _ = extract_signals(model, x_val)
    test_probability, test_error, _ = extract_signals(model, x_test)

    for split_name, labels, probability, error in [
        ("VALIDATION CLASSIFIER PROBABILITY", y_val,
         val_probability, None),
        ("VALIDATION RECONSTRUCTION ERROR", y_val,
         None, val_error),
        ("TEST CLASSIFIER PROBABILITY", y_test,
         test_probability, None),
        ("TEST RECONSTRUCTION ERROR", y_test,
         None, test_error),
    ]:
        if probability is not None:
            describe_scores(split_name, labels, probability)
        else:
            describe_scores(split_name, labels, error)

    print("\nRead-only diagnostic complete.")
    print("Frozen model and preprocessor were not modified.")


if __name__ == "__main__":
    main()
