from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "secom_telemetry_modified.csv"


def load_dataset(file_path=None):
    path = Path(file_path) if file_path else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def get_dataset_summary(df):
    label_column = next(
        (
            column for column in ["label", "pass_fail", "Pass/Fail", "target", "y"]
            if column in df.columns
        ),
        None
    )

    sensor_columns = [
        column for column in df.columns
        if column != label_column
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "sensor_count": len(sensor_columns),
        "missing_values": int(df[sensor_columns].isna().sum().sum()),
        "label_column": label_column
    }

    if label_column:
        labels = pd.to_numeric(df[label_column], errors="coerce")
        summary["defective_wafers"] = int((labels == 1).sum())
        summary["normal_wafers"] = int((labels == -1).sum())

    return summary
