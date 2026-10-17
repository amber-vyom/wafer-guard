import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Wafer Guard",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
}
[data-testid="stMetric"] {
    background: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263449;
}
h1, h2, h3 {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🔬 Wafer Guard")
st.sidebar.caption("AI-powered wafer anomaly detection")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Dataset Explorer", "Model Evaluation", "Audit"]
)

st.sidebar.divider()
st.sidebar.caption("SECOM wafer telemetry")

data_path = Path("data/raw/secom_telemetry_modified.csv")

uploaded_file = st.sidebar.file_uploader(
    "Upload wafer CSV",
    type=["csv"]
)

df = None

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif data_path.exists():
        df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Could not read dataset: {e}")

st.sidebar.divider()
st.sidebar.caption("Wafer Guard · AI/ML Challenge")

if df is None:
    st.title("Wafer Anomaly Monitor")
    st.warning("Upload your wafer telemetry CSV to begin.")
    st.stop()

label_column = next(
    (
        c for c in ["label", "pass_fail", "Pass/Fail", "target", "y"]
        if c in df.columns
    ),
    None
)

if label_column is None:
    possible_labels = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 3
        and set(df[c].dropna().unique()).issubset({-1, 1, 0})
    ]
    if possible_labels:
        label_column = possible_labels[-1]

sensor_columns = [
    c for c in df.columns
    if c != label_column and pd.api.types.is_numeric_dtype(df[c])
]

total_wafers = len(df)
sensor_count = len(sensor_columns)
missing_count = (
    int(df[sensor_columns].isna().sum().sum())
    if sensor_columns else 0
)

labels = None
defect_count = None
normal_count = None

if label_column is not None:
    labels = pd.to_numeric(df[label_column], errors="coerce")
    unique_labels = set(labels.dropna().unique())

    if unique_labels.issubset({-1, 1}):
        defect_count = int((labels == 1).sum())
        normal_count = int((labels == -1).sum())
    elif unique_labels.issubset({0, 1}):
        defect_count = int((labels == 1).sum())
        normal_count = int((labels == 0).sum())

st.title("Wafer Anomaly Monitor")
st.caption("Semiconductor manufacturing · Sensor analytics · Defect detection")

if page == "Overview":
    st.subheader("Production Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Wafers", f"{total_wafers:,}")
    c2.metric("Sensor Features", f"{sensor_count:,}")
    c3.metric("Missing Readings", f"{missing_count:,}")

    if defect_count is not None:
        prevalence = defect_count / total_wafers if total_wafers else 0
        c4.metric(
            "Defective Wafers",
            f"{defect_count:,}",
            f"{prevalence:.2%} prevalence"
        )
    else:
        c4.metric("Defective Wafers", "Labels unavailable")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Dataset Snapshot")
        st.write(f"**Rows:** {total_wafers:,}")
        st.write(f"**Numeric sensor columns:** {sensor_count:,}")
        st.write(f"**Missing sensor cells:** {missing_count:,}")
        st.write(f"**Label column:** {label_column or 'Not detected'}")

    with right:
        st.subheader("Defect Distribution")

        if normal_count is not None and defect_count is not None:
            chart_data = pd.DataFrame({
                "Class": ["Normal", "Defective"],
                "Count": [normal_count, defect_count]
            })

            fig = px.bar(
                chart_data,
                x="Class",
                y="Count",
                color="Class",
                color_discrete_map={
                    "Normal": "#22c55e",
                    "Defective": "#ef4444"
                },
                template="plotly_dark",
                text="Count"
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not identify the label encoding automatically.")

    st.subheader("Sensor Missingness")

    if sensor_columns:
        missing_by_sensor = (
            df[sensor_columns]
            .isna()
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )

        missing_by_sensor.columns = ["Sensor", "Missing Values"]

        fig = px.bar(
            missing_by_sensor,
            x="Sensor",
            y="Missing Values",
            color="Missing Values",
            template="plotly_dark",
            color_continuous_scale="Teal"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Status")

    st.warning(
        "The current measured model has zero recall and zero F1 on the "
        "held-out test set. It currently fails to detect defective wafers."
    )

    st.write(
        "The dashboard displays dataset statistics and previously measured "
        "model results. It does not claim live predictions."
    )

elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")

    st.write("Preview your wafer telemetry dataset.")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Column Summary")

    summary = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Unique Values": df.nunique(dropna=True)
    })

    st.dataframe(summary, use_container_width=True)

    st.subheader("Explore a Sensor")

    if sensor_columns:
        selected_sensor = st.selectbox(
            "Select a sensor",
            sensor_columns
        )

        values = pd.to_numeric(
            df[selected_sensor],
            errors="coerce"
        ).dropna()

        if not values.empty:
            a, b, c = st.columns(3)
            a.metric("Mean", f"{values.mean():.4f}")
            b.metric("Standard Deviation", f"{values.std():.4f}")
            c.metric(
                "Missing Values",
                f"{df[selected_sensor].isna().sum():,}"
            )

            fig = px.histogram(
                values,
                nbins=40,
                labels={"value": selected_sensor},
                title=f"Distribution of {selected_sensor}",
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Dataset Preview",
        df.head(1000).to_csv(index=False).encode("utf-8"),
        file_name="wafer_dataset_preview.csv",
        mime="text/csv"
    )

elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    st.warning(
        "The current trained model failed to identify defective wafers "
        "in the held-out test set. Accuracy alone is misleading."
    )

    metrics = [
        ("Accuracy", "94.27%"),
        ("Precision", "0.00"),
        ("Recall", "0.00"),
        ("F1 Score", "0.00"),
        ("Average Precision", "0.0948"),
        ("ROC-AUC", "0.6324")
    ]

    columns = st.columns(3)

    for i, (name, value) in enumerate(metrics):
        columns[i % 3].metric(name, value)

    st.subheader("Test Confusion Matrix")

    matrix = pd.DataFrame(
        [[296, 1], [17, 0]],
        index=["Actual Normal", "Actual Defective"],
        columns=["Predicted Normal", "Predicted Defective"]
    )

    st.dataframe(matrix, use_container_width=True)

    st.caption(
        "These are results from the previously measured test run. "
        "They are not recalculated for uploaded datasets."
    )

    st.subheader("Interpretation")

    st.write("- The model correctly classified 296 normal wafers.")
    st.write("- It incorrectly flagged 1 normal wafer.")
    st.write("- It missed all 17 defective wafers.")
    st.write("- The model requires improvement before practical use.")

elif page == "Audit":
    st.subheader("Audit and Safeguards")

    st.markdown("""
    ### 1. Data Leakage

    - Split data chronologically.
    - Fit preprocessing transformations on training data only.
    - Use validation data for threshold selection.
    - Keep the test set for final evaluation.

    ### 2. Legacy Sensor Handling

    - Treat `sensor_0 == 0.0` according to the legacy sentinel rule.
    - Preserve categorical meanings of `sensor_1` values 100, 200,
      and 300 for legacy wafers.

    ### 3. Model Integrity

    - Match reconstruction and input tensor shapes.
    - Preserve valid gradient paths during training.
    - Avoid using accuracy as the only performance measure.

    ### 4. Current Limitation

    The current measured model has zero recall and zero F1 on the
    held-out test set. It is not ready for production use.
    """)

    audit_path = Path("AUDIT.md")

    if audit_path.exists() and audit_path.stat().st_size > 0:
        st.download_button(
            "Download Audit Report",
            audit_path.read_text(encoding="utf-8"),
            file_name="AUDIT.md",
            mime="text/markdown"
        )

st.sidebar.success("Dashboard running locally")
