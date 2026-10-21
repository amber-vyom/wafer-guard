from datetime import datetime
from pathlib import Path
import json
import sys
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score
)

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.data.splits import (
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    sort_chronologically,
    split_bounds,
)
from src.services.prediction_service import FINAL_MODEL_PATH, predict_final_records

df = sort_chronologically(
    pd.read_csv(root / "data" / "raw" / "secom_telemetry_modified.csv")
)

# Identify the true label column.
candidates = ["label", "target", "pass_fail", "pass/fail", "class"]
label_col = next(
    (c for c in candidates if c in [str(x).lower() for x in df.columns]),
    None
)
if label_col is None:
    raise ValueError(
        "Could not identify the true label column. Available columns: "
        + ", ".join(map(str, df.columns))
    )

label_col = next(c for c in df.columns if str(c).lower() == label_col)
labels = df[label_col]

# SECOM labels commonly use -1 for pass and +1 for fail.
if set(labels.dropna().unique()).issubset({-1, 1}):
    y_all = (labels == 1).astype(int)
elif set(labels.dropna().unique()).issubset({0, 1}):
    y_all = labels.astype(int)
else:
    raise ValueError(
        f"Unexpected labels in {label_col}: "
        f"{labels.value_counts(dropna=False).to_dict()}. "
        "Confirm which label means defective before evaluating."
    )

# Held-out test partition of the shared chronological split.
_, test_start = split_bounds(len(df))
test_df = df.iloc[test_start:].copy()
y = y_all.iloc[test_start:].to_numpy()

if pd.isna(y).any():
    raise ValueError("Final test labels contain missing values.")

result = predict_final_records(test_df)
prob = result["probabilities"]
pred = result["predictions"]

tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

report = {
    "model": result["model_name"],
    "test_records": len(y),
    "actual_defects": int(y.sum()),
    "true_positives_detected": int(tp),
    "false_negatives_missed": int(fn),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "defect_detection_rate_percent": round(100 * recall_score(y, pred, zero_division=0), 2),
    "precision_percent": round(100 * precision_score(y, pred, zero_division=0), 2),
    "f1_score": round(f1_score(y, pred, zero_division=0), 4),
    "threshold": float(result["threshold"]),
    "split": {
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "test_start_row": int(test_start),
        "test_start_timestamp": str(test_df["timestamp"].iloc[0]),
        "test_end_timestamp": str(test_df["timestamp"].iloc[-1]),
    },
    "model_artifact": FINAL_MODEL_PATH.relative_to(root).as_posix(),
    "model_artifact_modified": datetime.fromtimestamp(
        FINAL_MODEL_PATH.stat().st_mtime
    ).isoformat(timespec="seconds"),
}

if len(set(y)) == 2:
    report["average_precision"] = round(float(average_precision_score(y, prob)), 4)
    report["roc_auc"] = round(float(roc_auc_score(y, prob)), 4)

out = root / "artifacts" / "experiments" / "final_evaluation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== FINAL MODEL EVALUATION ===")
print(json.dumps(report, indent=2))
print(f"\nSaved report: {out}")
print("\nThis is an evaluation, not another training or tuning run.")
