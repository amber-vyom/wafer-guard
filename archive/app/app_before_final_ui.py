from pathlib import Path

import pandas as pd
import streamlit as st

from src.services.prediction_service import predict_records

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

st.set_page_config(
    page_title="WaferGuard",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 WaferGuard")
st.caption("Wafer inspection support | Experimental research prototype")

st.warning(
    "Experimental predictions only. A flagged record is not a confirmed defect. "
    "Do not use this prototype alone to make manufacturing decisions."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

st.metric("Available records", f"{len(df):,}")

st.divider()
st.subheader("Inspect a dataset record")

index = st.number_input(
    "Dataset record index",
    min_value=0,
    max_value=max(0, len(df) - 1),
    value=0,
    step=1,
)

if st.button("Run Prediction", type="primary"):
    try:
        result = predict_records(df.iloc[[int(index)]])
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check the saved PyTorch model, preprocessor, and sensor feature "
            "columns in the artifacts folder."
        )
        st.stop()

    probability = float(result["probabilities"][0])
    prediction = int(result["predictions"][0])
    threshold = float(result["threshold"])

    st.subheader(f"Result for record {int(index)}")

    left, right = st.columns(2)
    left.metric("Model score", f"{probability:.2%}")
    right.metric("Saved decision threshold", f"{threshold:.6f}")

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction == 1:
        st.error("Flagged for further inspection")
    else:
        st.success("Not flagged by the model")

    st.caption(
        "The score is the model's output, not a verified real-world defect rate. "
        "A record that is not flagged is not guaranteed to be defect-free."
    )

    if "reconstruction_errors" in result:
        with st.expander("Technical details"):
            st.write(
                "Reconstruction error:",
                f"{float(result['reconstruction_errors'][0]):.6f}",
            )

with st.expander("About WaferGuard"):
    st.write(
        "WaferGuard uses the saved PyTorch model and its preprocessing pipeline "
        "to score sensor records. Its predictions are experimental and require "
        "independent validation before operational use."
    )

st.divider()
st.caption("WaferGuard | Experimental prototype")
