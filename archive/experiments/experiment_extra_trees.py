
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "secom_telemetry_modified.csv"

TOP_SENSOR_COUNT = 50
RANDOM_STATE = 42


def select_sensors(train, sensor_cols):
    scores = []

    for col in sensor_cols:
        normal = train.loc[train["label"] == -1, col].dropna()
        defect = train.loc[train["label"] == 1, col].dropna()

        if len(normal) < 20 or len(defect) < 10:
            continue

        normal_std = normal.std()

        if not np.isfinite(normal_std) or normal_std <= 0:
            continue

        score = abs(defect.median() - normal.median()) / normal_std

        if np.isfinite(score):
            scores.append((col, float(score)))

    scores.sort(key=lambda item: item[1], reverse=True)
    return [col for col, _ in scores[:TOP_SENSOR_COUNT]]


def evaluate(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"\n{name}")
    print("-" * 45)
    print("Average precision:", round(
        average_precision_score(y_val, probabilities), 5
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_val, probabilities), 5
    ))

    ranked = np.argsort(probabilities)[::-1]

    for k in (10, 20, 50):
        indices = ranked[:min(k, len(ranked))]
        print(
            f"Defects in top {len(indices)}:",
            int(y_val[indices].sum())
        )


def main():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()

    sensor_cols = [
        col for col in df.columns if col.startswith("sensor_")
    ]

    y_train = (train["label"] == 1).astype(int).to_numpy()
    y_val = (val["label"] == 1).astype(int).to_numpy()

    selected = select_sensors(train, sensor_cols)

    if len(selected) < 2:
        raise RuntimeError("Not enough usable sensors were selected.")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[selected])
    X_val = imputer.transform(val[selected])

    print("=" * 55)
    print("EXTRA TREES VS RANDOM FOREST")
    print("=" * 55)
    print("Selected sensors:", len(selected))
    print("Training rows:", len(train))
    print("Training defects:", int(y_train.sum()))
    print("Validation rows:", len(val))
    print("Validation defects:", int(y_val.sum()))
    print(
        "Validation defect prevalence:",
        round(float(y_val.mean()), 5)
    )

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    evaluate(
        "RANDOM FOREST",
        random_forest,
        X_train, y_train, X_val, y_val
    )

    evaluate(
        "EXTRA TREES",
        extra_trees,
        X_train, y_train, X_val, y_val
    )

    print("\nExperiment complete.")
    print("No artifacts saved.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
