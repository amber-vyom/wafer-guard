import pandas as pd
import pytest

from src.data.splits import (
    chronological_split,
    sort_chronologically,
    split_bounds,
)


def test_bounds_match_the_documented_split():
    # 1,567 rows -> 1,096 training, 235 validation, 236 test (see README / AUDIT).
    assert split_bounds(1567) == (1096, 1331)


def test_bounds_reject_tiny_datasets():
    with pytest.raises(ValueError):
        split_bounds(3)


def test_partitions_are_disjoint_and_complete(raw_frame):
    train, validation, test = chronological_split(raw_frame)
    assert len(train) + len(validation) + len(test) == len(raw_frame)
    assert train["timestamp"].max() <= validation["timestamp"].min()
    assert validation["timestamp"].max() <= test["timestamp"].min()


def test_shuffled_input_is_reordered_before_splitting(raw_frame):
    shuffled = raw_frame.sample(frac=1.0, random_state=1)
    ordered = sort_chronologically(shuffled)
    assert ordered["timestamp"].is_monotonic_increasing
    assert list(ordered.index) == list(range(len(ordered)))


def test_sort_is_stable_for_repeated_timestamps(raw_frame):
    tagged = raw_frame.assign(order=range(len(raw_frame)))
    ordered = sort_chronologically(tagged)
    for _, group in ordered.groupby("timestamp"):
        assert list(group["order"]) == sorted(group["order"])


def test_invalid_timestamp_is_rejected(raw_frame):
    broken = raw_frame.copy()
    broken["timestamp"] = broken["timestamp"].astype(str)
    broken.loc[3, "timestamp"] = "not a date"
    with pytest.raises(ValueError):
        sort_chronologically(broken)


def test_input_frame_is_not_modified(raw_frame):
    before = raw_frame.copy()
    chronological_split(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)
