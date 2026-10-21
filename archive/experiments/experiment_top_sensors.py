
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

RANDOM_STATE = 42
TOP_SENSOR_COUNT = 50

# --------------------------------------------------
# 1. Load and chronologically split the data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

y_train = (train["label"] == 1).astype(int).to_numpy()
y_val = (val["label"] == 1).astype(int).to_numpy()

print("=" * 55)
print("TOP 50 SENSOR EXPERIMENT")
print("=" * 55)
print("Training rows:", len(train))
print("Training defects:", int(y_train.sum()))
print("Validation rows:", len(val))
print("Validation defects:", int(y_val.sum()))

# --------------------------------------------------
# 2. Select sensors using training data only
# --------------------------------------------------

scores = []

for col in sensor_cols:
    normal = train.loc[train["label"] == -1, col].dropna()
    defect = train.loc[train["label"] == 1, col].dropna()

    # Require enough observations in both classes.
    if len(normal) < 20 or len(defect) < 10:
        continue

    normal_std = normal.std()

    if not np.isfinite(normal_std) or normal_std <= 0:
        continue

    # Standardized difference between class medians.
    score = abs(defect.median() - normal.median()) / normal_std

    if np.isfinite(score):
        scores.append((col, float(score)))

scores.sort(key=lambda item: item[1], reverse=True)

selected = [col for col, _ in scores[:TOP_SENSOR_COUNT]]

if len(selected) < 2:
    raise RuntimeError("Not enough usable sensors were selected.")

print("\nSelected sensor count:", len(selected))
print("Selected sensors:")
for col, score in scores[:TOP_SENSOR_COUNT]:
    print(f"{col}: score={score:.4f}")

# --------------------------------------------------
# 3. Preprocess using training data only
# --------------------------------------------------

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = imputer.fit_transform(train[selected])
X_val = imputer.transform(val[selected])

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# --------------------------------------------------
# 4. Train baseline classifier
# --------------------------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=RANDOM_STATE,
)

model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]

# --------------------------------------------------
# 5. Validation evaluation
# --------------------------------------------------

print("\nVALIDATION RESULTS")
print("Average precision:", round(
    average_precision_score(y_val, val_prob), 5
))
print("ROC-AUC:", round(
    roc_auc_score(y_val, val_prob), 5
))

# Ranking-based checks: no threshold selection here.
ranked_indices = np.argsort(val_prob)[::-1]

for k in (10, 20, 50):
    top_k = ranked_indices[:min(k, len(ranked_indices))]
    defects_found = int(y_val[top_k].sum())

    print(
        f"Defects among top {len(top_k)} scores:",
        defects_found,
    )

print("\nExperiment complete.")
print("No model artifacts were saved or overwritten.")
print("The test partition was not used.")
