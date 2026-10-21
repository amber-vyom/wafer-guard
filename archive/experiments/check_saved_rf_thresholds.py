
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"
MODEL = ROOT / "artifacts" / "experiments" / "wafer_rf_recall_candidate.joblib"

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

artifact = joblib.load(MODEL)
pipeline = artifact["pipeline"]
feature_cols = artifact["feature_cols"]
sensor_cols = artifact["sensor_cols"]

end_train = int(len(df) * 0.70)
end_val = int(len(df) * 0.80)
val = df.iloc[end_train:end_val].copy()

X = val[sensor_cols].apply(pd.to_numeric, errors="coerce")
legacy = X["sensor_0"].eq(0.0).fillna(False)
sensor1 = X["sensor_1"].copy()

X.loc[legacy, "sensor_0"] = np.nan
X["sensor_1"] = sensor1.mask(legacy, np.nan)

flags = pd.DataFrame(
    {
        "is_legacy_ps_l": legacy.astype(float),
        "legacy_sensor_1_code_100": (
            legacy & sensor1.eq(100.0)
        ).astype(float),
        "legacy_sensor_1_code_200": (
            legacy & sensor1.eq(200.0)
        ).astype(float),
        "legacy_sensor_1_code_300": (
            legacy & sensor1.eq(300.0)
        ).astype(float),
    },
    index=X.index,
)

X = pd.concat([X, flags], axis=1)[feature_cols]
y = (val["label"] == 1).astype(int).to_numpy()
prob = pipeline.predict_proba(X)[:, 1]

print("Validation records:", len(y))
print("Actual validation defects:", int(y.sum()))
print("\nTHRESHOLD | PRECISION | RECALL | TP | FP | FN | TN")

for threshold in np.arange(0.05, 0.201, 0.01):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:9.2f} | "
        f"{precision_score(y, pred, zero_division=0):9.3f} | "
        f"{recall_score(y, pred, zero_division=0):6.3f} | "
        f"{tp:2d} | {fp:3d} | {fn:2d} | {tn:3d}"
    )

print("\nDiagnostic only: validation data used; no artifacts changed.")
