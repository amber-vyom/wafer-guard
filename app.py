from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from src.data.splits import (
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    sort_chronologically,
    split_bounds,
)
from src.services.prediction_service import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    load_model,
    predict_anomaly,
    prepare_features,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
TEST_SHARE = 1 - TRAIN_FRACTION - VALIDATION_FRACTION


# The modification times are cache keys: replacing a file reloads it.
@st.cache_data
def load_data(path, modified):
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts(model_modified, preprocessor_modified):
    return load_model()


def modified_time(path):
    return path.stat().st_mtime if path.exists() else None

st.set_page_config(page_title="WaferGuard", page_icon="🔬", layout="centered")

st.title("WaferGuard")
st.subheader("Problem Statement")
st.write(
    "To detect potentially defective semiconductor wafers using machine "
    "learning on sensor telemetry data."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime)

st.button("Run Model", type="primary", key="run_model")

if st.session_state.get("run_model", False):
    # Use the held-out chronological test partition (see src/data/splits.py).
    time_cols = [
        c for c in df.columns
        if str(c).lower().replace("_", "").replace(" ", "")
        in {"time", "timestamp", "date", "datetime"}
    ]
    target_cols = [
        c for c in df.columns
        if str(c).lower().replace("_", "").replace(" ", "").replace("-", "")
        in {"pass/fail", "passfail", "target", "label", "class"}
    ]

    if len(time_cols) != 1 or len(target_cols) != 1:
        st.error("Could not identify the dataset timestamp and target columns.")
        st.stop()

    time_col, target_col = time_cols[0], target_cols[0]
    try:
        df = sort_chronologically(df, time_col)
    except ValueError:
        st.error("Invalid timestamps found. Test evaluation was stopped.")
        st.stop()

    _, validation_end = split_bounds(len(df))
    test = df.iloc[validation_end:].copy()
    labels = pd.to_numeric(test[target_col], errors="coerce")

    if labels.isna().any() or not set(labels.unique()).issubset({-1, 1}):
        st.error("Expected target labels -1 and +1; cannot evaluate safely.")
        st.stop()

    y_true = (labels.to_numpy() == 1).astype(int)

    try:
        model, preprocessor, saved_threshold = load_artifacts(
            modified_time(MODEL_PATH), modified_time(PREPROCESSOR_PATH)
        )
        result = predict_anomaly(
            model, prepare_features(test, preprocessor), saved_threshold
        )
        scores = np.asarray(result["probabilities"], dtype=float)
        y_pred = np.asarray(result["predictions"], dtype=int)
        threshold = float(result["threshold"])
    except Exception as exc:
        st.error(f"Model evaluation failed: {exc}")
        st.stop()

    if len(scores) != len(test) or len(y_pred) != len(test):
        st.error("Prediction count does not match test dataset size.")
        st.stop()

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    metrics = [
        ("Test records", len(test)),
        ("Actual defects", int(y_true.sum())),
        ("Defects detected", f"{tp}/{int(y_true.sum())}"),
        ("Defects missed", fn),
        ("False alarms", fp),
        ("Alert rate", f"{y_pred.mean():.2%}"),
        ("Defect prevalence", f"{y_true.mean():.2%}"),
        ("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}"),
        ("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}"),
        ("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2%}"),
        ("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.4f}"),
        ("ROC-AUC", f"{roc_auc_score(y_true, scores):.4f}"
         if len(np.unique(y_true)) == 2 else "Unavailable"),
        ("Average Precision", f"{average_precision_score(y_true, scores):.4f}"),
        ("Saved threshold", f"{threshold:.8f}"),
    ]

    st.subheader("Final Model — Test Results")
    st.caption(
        f"Results are calculated from the saved model on the final chronological "
        f"{TEST_SHARE:.0%} of the dataset, which is held out from training and validation."
    )
    st.table(pd.DataFrame(
        [(name, str(value)) for name, value in metrics],
        columns=["Evaluation Metric", "Result"],
    ))

    st.subheader("Model Performance Graph")
    graph_values = {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0) * 100,
    }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(graph_values.keys(), graph_values.values())
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Saved PyTorch Model — Test Metrics")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("About Our Model")
    st.write(
        "WaferGuard uses a saved PyTorch autoencoder to identify potentially "
        "anomalous sensor records. Its predictions are compared with known "
        "test labels to calculate the reported evaluation metrics."
    )
    st.warning(
        f"The model missed {fn} of {int(y_true.sum())} actual defects in this "
        "evaluation. It is an experimental prototype, not a production-ready "
        "inspection system. An unflagged record is not guaranteed defect-free."
    )
