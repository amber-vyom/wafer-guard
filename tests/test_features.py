import numpy as np
import pytest

from src.services.prediction_service import _prepare_sensor_data, prepare_features
from tests.conftest import FLAG_COLS, SENSOR_COLS


def test_legacy_sentinel_is_masked_and_flagged(raw_frame):
    prepared = _prepare_sensor_data(raw_frame, SENSOR_COLS)
    legacy_rows = prepared.index[:6]
    assert prepared.loc[legacy_rows, "sensor_0"].isna().all()
    assert prepared.loc[legacy_rows, "sensor_1"].isna().all()
    assert prepared.loc[legacy_rows, "is_legacy_ps_l"].eq(1.0).all()
    assert prepared["is_legacy_ps_l"].sum() == 6


def test_code_flags_follow_the_sensor_1_code(raw_frame):
    prepared = _prepare_sensor_data(raw_frame, SENSOR_COLS)
    assert prepared["legacy_sensor_1_code_100"].sum() == 2
    assert prepared["legacy_sensor_1_code_200"].sum() == 2
    assert prepared["legacy_sensor_1_code_300"].sum() == 2


def test_code_value_outside_legacy_rows_raises_no_flag(raw_frame):
    # Row 10 has sensor_1 == 100 but sensor_0 != 0, so it is not a legacy wafer.
    prepared = _prepare_sensor_data(raw_frame, SENSOR_COLS)
    assert prepared.loc[10, "legacy_sensor_1_code_100"] == 0.0
    assert prepared.loc[10, "sensor_1"] == 100.0


def test_non_numeric_values_become_missing(raw_frame):
    dirty = raw_frame.copy()
    dirty["sensor_3"] = dirty["sensor_3"].astype(object)
    dirty.loc[2, "sensor_3"] = "bad"
    prepared = _prepare_sensor_data(dirty, SENSOR_COLS)
    assert np.isnan(prepared.loc[2, "sensor_3"])


def test_missing_sensor_column_is_rejected(raw_frame):
    with pytest.raises(ValueError, match="Missing sensor columns"):
        _prepare_sensor_data(raw_frame.drop(columns=["sensor_4"]), SENSOR_COLS)


def test_features_have_expected_layout(raw_frame, preprocessor):
    features = prepare_features(raw_frame, preprocessor)
    assert features.shape == (len(raw_frame), len(preprocessor["feature_cols"]))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    # The indicator columns come last and are not scaled.
    flags = features[:, -len(FLAG_COLS):]
    assert set(np.unique(flags)) <= {0.0, 1.0}


def test_single_row_and_series_inputs(raw_frame, preprocessor):
    one_row = prepare_features(raw_frame.iloc[[0]], preprocessor)
    from_series = prepare_features(raw_frame.iloc[0], preprocessor)
    assert one_row.shape[0] == 1
    np.testing.assert_allclose(one_row, from_series)
