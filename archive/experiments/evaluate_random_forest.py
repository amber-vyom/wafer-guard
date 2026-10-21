
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/secom_telemetry_modified.csv"

TOP_SENSOR_COUNT = 50
THRESHOLD = 0.10

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

n = len(df)
a, b = int(n * 0.70), int(n * 0.80)

train = df.iloc[:a].copy()
val = df.iloc[a:b].copy()
test = df.iloc[b:].copy()

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

y_train = (train["label"] == 1).astype(int).to_numpy()
y_test = (test["label"] == 1).astype(int).to_numpy()

# Select sensors using training data only.
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

# Fit imputation on training data only.
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(train[selected])
X_test = imputer.transform(test[selected])

model = RandomForestClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Evaluate the frozen threshold once on the held-out test set.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= THRESHOLD).astype(int)

print("=" * 50)
print("RANDOM FOREST - HELD-OUT TEST RESULTS")
print("=" * 50)
print("Sensors:", len(selected))
print("Frozen threshold:", THRESHOLD)
print("Test samples:", len(test))
print("Actual defects:", int(y_test.sum()))
print("Predicted defects:", int(test_pred.sum()))
print("Accuracy:", round(accuracy_score(y_test, test_pred), 5))
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 5
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 5
))
print("F1:", round(
    f1_score(y_test, test_pred, zero_division=0), 5
))
print("Average precision:", round(
    average_precision_score(y_test, test_prob), 5
))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(roc_auc_score(y_test, test_prob), 5))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nEvaluation complete. No artifacts were saved.")
