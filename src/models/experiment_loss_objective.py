from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)

from src.models.wafer_model import WaferAutoencoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
OUTPUT_DIR = ROOT / "artifacts" / "experiments"

SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    if not set(df["label"].dropna().unique()).issubset({-1, 1}):
        raise ValueError("Expected labels -1 for normal and 1 for defects.")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not {"sensor_0", "sensor_1"}.issubset(sensor_cols):
        raise ValueError("Required legacy sensor columns are missing.")

    n = len(df)
    a = int(n * 0.70)
    b = int(n * 0.80)

    train_df = df.iloc[:a].copy()
    val_df = df.iloc[a:b].copy()
    test_df = df.iloc[b:].copy()

    print("Records:", n)
    for name, part in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        defects = int((part["label"] == 1).sum())
        print(
            f"{name}: {len(part)} records, {defects} defects, "
            f"prevalence={(part['label'] == 1).mean():.4f}"
        )

    return sensor_cols, train_df, val_df, test_df


def build_features(part, sensor_cols):
    raw = part[sensor_cols].copy()

    for column in sensor_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    legacy = raw["sensor_0"].eq(0.0).fillna(False)
    sensor1 = raw["sensor_1"].copy()

    raw["is_legacy_ps_l"] = legacy.astype(np.float32)

    for code in (100, 200, 300):
        raw[f"legacy_sensor_1_code_{code}"] = (
            legacy & sensor1.eq(float(code))
        ).astype(np.float32)

    raw.loc[legacy, "sensor_0"] = np.nan
    raw["sensor_1"] = sensor1.mask(legacy, np.nan)

    y = (part["label"] == 1).astype(np.int64).to_numpy()
    return raw, y


def prepare_data(sensor_cols, train_df, val_df, test_df):
    train_raw, y_train = build_features(train_df, sensor_cols)
    val_raw, y_val = build_features(val_df, sensor_cols)
    test_raw, y_test = build_features(test_df, sensor_cols)

    flag_cols = [
        "is_legacy_ps_l",
        "legacy_sensor_1_code_100",
        "legacy_sensor_1_code_200",
        "legacy_sensor_1_code_300",
    ]

    numeric_cols = [c for c in sensor_cols if c != "sensor_1"]

    usable_cols = [
        c for c in numeric_cols
        if train_raw[c].notna().any()
        and train_raw[c].nunique(dropna=True) > 1
    ]

    if not usable_cols:
        raise ValueError("No usable continuous sensor features found.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imp = imputer.fit_transform(train_raw[usable_cols])
    val_imp = imputer.transform(val_raw[usable_cols])
    test_imp = imputer.transform(test_raw[usable_cols])

    train_scaled = scaler.fit_transform(train_imp)
    val_scaled = scaler.transform(val_imp)
    test_scaled = scaler.transform(test_imp)

    def combine(scaled, raw):
        flags = raw[flag_cols].to_numpy(dtype=np.float32)
        return np.column_stack([scaled, flags]).astype(np.float32)

    X_train = combine(train_scaled, train_raw)
    X_val = combine(val_scaled, val_raw)
    X_test = combine(test_scaled, test_raw)

    for name, values in [
        ("Train", X_train),
        ("Validation", X_val),
        ("Test", X_test),
    ]:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} features contain non-finite values.")

    preprocessor = {
        "sensor_cols": sensor_cols,
        "usable_cols": usable_cols,
        "flag_cols": flag_cols,
        "feature_cols": usable_cols + flag_cols,
        "imputer": imputer,
        "scaler": scaler,
    }

    print("Continuous features:", len(usable_cols))
    print("Total model features:", X_train.shape[1])
    print("Preprocessing fitted on training data only.")

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


def get_probabilities(model, X):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32)
        _, logits, _ = model(tensor)
        return torch.sigmoid(logits).cpu().numpy()


def select_threshold(y, probabilities):
    candidates = np.unique(
        np.concatenate(([0.0], probabilities, [1.0]))
    )

    results = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(y, predictions, zero_division=0)
        precision = precision_score(y, predictions, zero_division=0)
        results.append((f1, precision, float(threshold)))

    # Prefer F1, then precision, then the higher threshold.
    best_f1, best_precision, best_threshold = max(results)
    return best_threshold, best_f1, best_precision


def train_candidate(name, X_train, y_train, X_val, y_val, combined_loss):
    # Reset the seed so both approaches start from the same initialization.
    torch.manual_seed(SEED)
    model = WaferAutoencoder(input_dim=X_train.shape[1])

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    positive_count = max(int(y_train.sum()), 1)
    negative_count = len(y_train) - int(y_train.sum())
    pos_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
    )

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_ap = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, logits, _ = model(batch_x)

            if reconstruction.shape != batch_x.shape:
                raise RuntimeError("Reconstruction shape mismatch.")
            if logits.shape != batch_y.shape:
                raise RuntimeError("Classifier output shape mismatch.")

            classification_loss = bce(logits, batch_y)

            if combined_loss:
                reconstruction_loss = mse(reconstruction, batch_x)
                loss = classification_loss + reconstruction_loss
            else:
                loss = classification_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(batch_x)

        val_prob = get_probabilities(model, X_val)
        val_ap = float(average_precision_score(y_val, val_prob))

        history.append({
            "epoch": epoch,
            "training_loss": running_loss / len(dataset),
            "validation_average_precision": val_ap,
        })

        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{name} | epoch {epoch}/{EPOCHS} | "
                f"loss={running_loss / len(dataset):.5f} | "
                f"validation AP={val_ap:.5f}"
            )

    if best_state is None:
        raise RuntimeError(f"No checkpoint created for {name}.")

    model.load_state_dict(best_state)
    val_prob = get_probabilities(model, X_val)
    threshold, val_f1, val_precision = select_threshold(y_val, val_prob)

    metrics = {
        "name": name,
        "best_epoch": best_epoch,
        "validation_average_precision": float(
            average_precision_score(y_val, val_prob)
        ),
        "validation_roc_auc": (
            float(roc_auc_score(y_val, val_prob))
            if len(np.unique(y_val)) == 2 else None
        ),
        "validation_f1_at_selected_threshold": float(val_f1),
        "validation_precision_at_selected_threshold": float(val_precision),
        "selected_threshold": float(threshold),
        "history": history,
    }

    print(
        f"{name} BEST: epoch={best_epoch}, "
        f"validation AP={metrics['validation_average_precision']:.5f}, "
        f"threshold={threshold:.6f}, validation F1={val_f1:.4f}"
    )

    return model, metrics


def main():
    (
        sensor_cols,
        train_df,
        val_df,
        test_df,
    ) = load_and_split()

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        preprocessor,
    ) = prepare_data(sensor_cols, train_df, val_df, test_df)

    candidates = []

    for name, combined_loss in [
        ("classification_only", False),
        ("combined_loss", True),
    ]:
        model, metrics = train_candidate(
            name, X_train, y_train, X_val, y_val, combined_loss
        )
        candidates.append((model, metrics))

    print("\nVALIDATION COMPARISON")
    for _, metrics in candidates:
        print(
            f"{metrics['name']}: "
            f"AP={metrics['validation_average_precision']:.5f}, "
            f"ROC-AUC={metrics['validation_roc_auc']}, "
            f"threshold={metrics['selected_threshold']:.6f}, "
            f"F1={metrics['validation_f1_at_selected_threshold']:.4f}"
        )

    # Select the objective using validation AP only.
    selected_model, selected_metrics = max(
        candidates,
        key=lambda item: item[1]["validation_average_precision"],
    )
    selected_name = selected_metrics["name"]
    threshold = selected_metrics["selected_threshold"]

    model_path = OUTPUT_DIR / "wafer_model_candidate.pt"
    preprocessor_path = OUTPUT_DIR / "preprocessor_candidate.joblib"

    torch.save(
        {
            "model_state_dict": selected_model.state_dict(),
            "input_dim": X_train.shape[1],
            "latent_dim": 32,
            "threshold": threshold,
            "selected_objective": selected_name,
            "selected_epoch": selected_metrics["best_epoch"],
        },
        model_path,
    )
    joblib.dump(preprocessor, preprocessor_path)

    # Final test evaluation: only the validation-selected candidate is evaluated.
    test_prob = get_probabilities(selected_model, X_test)
    test_pred = (test_prob >= threshold).astype(int)

    test_metrics = {
        "selected_objective": selected_name,
        "selected_epoch": selected_metrics["best_epoch"],
        "threshold": float(threshold),
        "test_records": int(len(y_test)),
        "test_defects": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, test_prob)),
        "roc_auc": (
            float(roc_auc_score(y_test, test_prob))
            if len(np.unique(y_test)) == 2 else None
        ),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    report = {
        "seed": SEED,
        "epochs": EPOCHS,
        "split": "chronological 70/10/20",
        "selection_metric": "validation average precision",
        "validation_candidates": [metrics for _, metrics in candidates],
        "final_test_evaluation": test_metrics,
        "candidate_model_path": str(model_path.relative_to(ROOT)),
        "candidate_preprocessor_path": str(preprocessor_path.relative_to(ROOT)),
    }

    report_path = OUTPUT_DIR / "loss_objective_experiment.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL TEST EVALUATION — SELECTED CANDIDATE ONLY")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\nExperimental artifacts saved:")
    print(model_path)
    print(preprocessor_path)
    print(report_path)
    print("\nExisting application artifacts were not modified.")


if __name__ == "__main__":
    main()
