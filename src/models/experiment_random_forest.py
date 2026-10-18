
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/secom_telemetry_modified.csv"

TOP_SENSOR_COUNT = 50

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

n = len(df)
a, b = int(n * 0.70), int(n * 0.80)

train = df.iloc[:a].copy()
val = df.iloc[a:b].copy()

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()

# Select sensors from training data only.
scores = []

for col in sensor_cols:
    normal = train.loc[train["label"] == -1, col].dropna()
    defect = train.loc[train["label"] == 1, col].dropna()

    if len(normal) < 20 or len(defect) < 10:
        continue

    std = normal.std()
    if not np.isfinite(std) or std <= 0:
        continue

    score = abs(defect.median() - normal.median()) / std
    if np.isfinite(score):
        scores.append((col, float(score)))

scores.sort(key=lambda item: item[1], reverse=True)
selected = [col for col, _ in scores[:TOP_SENSOR_COUNT]]

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(train[selected])
X_val = imputer.transform(val[selected])

model = RandomForestClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

prob = model.predict_proba(X_val)[:, 1]

print("Validation AP:", round(average_precision_score(y_val, prob), 5))
print("Validation ROC-AUC:", round(roc_auc_score(y_val, prob), 5))

print("\nTHRESHOLD | PRECISION | RECALL | F1 | TP | FP | FN | TN")

for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    pred = (prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, pred, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:4.3f} | "
        f"{tp:2d} | {fp:2d} | {fn:2d} | {tn:3d}"
    )

print("\nDiagnostic only: no artifacts saved.")
print("The test partition was not used.")
