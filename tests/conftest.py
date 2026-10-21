"""Shared synthetic fixtures. Tests never touch artifacts/ or the real dataset."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

N_ROWS = 60
SENSOR_COLS = [f"sensor_{i}" for i in range(6)]
CONTINUOUS_COLS = [c for c in SENSOR_COLS if c not in {"sensor_0", "sensor_1"}]
FLAG_COLS = [
    "is_legacy_ps_l",
    "legacy_sensor_1_code_100",
    "legacy_sensor_1_code_200",
    "legacy_sensor_1_code_300",
]


@pytest.fixture
def raw_frame():
    """Small dataset with legacy rows, repeated timestamps and some missing cells."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(N_ROWS, len(SENSOR_COLS))), columns=SENSOR_COLS
    )
    frame["sensor_0"] = 1.0
    frame["sensor_1"] = 7.0
    # Rows 0-5 are legacy: sensor_0 is the 0.0 sentinel, sensor_1 holds a code.
    frame.loc[0:5, "sensor_0"] = 0.0
    frame.loc[0:1, "sensor_1"] = 100.0
    frame.loc[2:3, "sensor_1"] = 200.0
    frame.loc[4:5, "sensor_1"] = 300.0
    # A non-legacy row that happens to hold a code value must not raise a code flag.
    frame.loc[10, "sensor_1"] = 100.0
    frame.loc[3, "sensor_2"] = np.nan
    frame["label"] = -1
    frame.loc[[7, 40, 55], "label"] = 1
    # Pairs of rows share a timestamp, like the real data.
    frame["timestamp"] = pd.date_range("2008-01-01", periods=N_ROWS // 2, freq="h").repeat(2)
    return frame


@pytest.fixture
def preprocessor(raw_frame):
    """A preprocessor in the format the notebook saves and the app loads."""
    legacy = raw_frame["sensor_0"].eq(0.0)
    continuous = raw_frame[CONTINUOUS_COLS]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(continuous)
    scaler = StandardScaler().fit(imputed)
    assert legacy.sum() == 6
    return {
        "imputer": imputer,
        "scaler": scaler,
        "sensor_cols": list(SENSOR_COLS),
        "usable_cols": list(CONTINUOUS_COLS),
        "flag_cols": list(FLAG_COLS),
        "feature_cols": list(CONTINUOUS_COLS) + list(FLAG_COLS),
    }
