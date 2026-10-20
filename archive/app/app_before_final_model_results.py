import streamlit as st

st.set_page_config(
    page_title="WaferGuard - Model Comparison",
    layout="centered",
)

st.title("WaferGuard")
st.subheader("Problem Statement")
st.write(
    "To detect defective semiconductor wafers using machine learning "
    "and compare our model with the previous model."
)

st.divider()

st.subheader("Model Performance Comparison")

comparison = [
    {
        "Evaluation Metric": "Defects detected",
        "Previous Model": "0 / 17",
        "Our Model": "2 / 17",
    },
    {
        "Evaluation Metric": "F1 Score",
        "Previous Model": "Not available",
        "Our Model": "0.1111",
    },
    {
        "Evaluation Metric": "Accuracy",
        "Previous Model": "Not available",
        "Our Model": "89.81%",
    },
    {
        "Evaluation Metric": "Recall",
        "Previous Model": "0%",
        "Our Model": "11.76%",
    },
]

st.table(comparison)

st.caption(
    "Test set: 314 records, including 17 actual defects. "
    "WaferGuard detected 2 defects, missed 15, and produced 17 false alarms. "
    "The model is an experimental prototype, not production-ready."
)
