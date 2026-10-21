from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.data.splits import sort_chronologically, split_bounds
from src.services.prediction_service import predict_final_records

df = sort_chronologically(
    pd.read_csv(root / "data" / "raw" / "secom_telemetry_modified.csv")
)

label_col = next(
    (c for c in df.columns if str(c).lower() in
     ["label", "target", "pass_fail", "pass/fail", "class"]),
    None
)
if label_col is None:
    raise ValueError(f"Label column not found. Columns: {list(df.columns)}")

labels = df[label_col]
unique = set(labels.dropna().unique())
if unique.issubset({-1, 1}):
    y = (labels == 1).astype(int).to_numpy()
elif unique.issubset({0, 1}):
    y = labels.astype(int).to_numpy()
else:
    raise ValueError(f"Unexpected labels: {labels.value_counts().to_dict()}")

# Validation partition of the shared chronological split.
start, end = split_bounds(len(df))
valid_df = df.iloc[start:end]
yv = y[start:end]

result = predict_final_records(valid_df)
p = np.asarray(result["probabilities"], dtype=float)

print(f"\nValidation records: {len(yv)}")
print(f"Actual defects: {int(yv.sum())}")
print(f"Model: {result['model_name']}")

if len(set(yv)) == 2:
    print(f"ROC-AUC: {roc_auc_score(yv, p):.4f}")
    print(f"Average precision: {average_precision_score(yv, p):.4f}")

print("\nThreshold | Detected | Missed | False alarms | Recall | Precision")
for t in [.01, .02, .05, .10, .15, .20, .30, .40, .50]:
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (yv == 1)).sum())
    fn = int(((pred == 0) & (yv == 1)).sum())
    fp = int(((pred == 1) & (yv == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    print(f"{t:8.2f} | {tp:8} | {fn:6} | {fp:12} | {recall:6.1%} | {precision:9.1%}")

print("\nValidation-only threshold diagnostic complete.")
