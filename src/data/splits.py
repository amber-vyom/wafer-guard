"""Single definition of the chronological train / validation / test split.

The notebook, the Streamlit app, and the evaluation scripts all import from here,
so every reported number refers to the same rows. Data is ordered by timestamp
(stable sort, because timestamps repeat) and cut by position: the earliest rows
train the model, the next rows choose the checkpoint and threshold, and the latest
rows are the held-out test set.
"""

import pandas as pd

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15  # the test partition gets the remaining rows
TIME_COLUMN = "timestamp"
LABEL_COLUMN = "label"


def split_bounds(
    n_rows,
    train_fraction=TRAIN_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
):
    """Return ``(train_end, validation_end)`` row positions for ordered data.

    Rows ``[0, train_end)`` are training, ``[train_end, validation_end)`` are
    validation, and ``[validation_end, n_rows)`` are test.
    """
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    if not 0 < train_end < validation_end < n_rows:
        raise ValueError(
            f"{n_rows} rows are too few for a train/validation/test split "
            f"({train_fraction:.0%}/{validation_fraction:.0%}/rest)."
        )
    return train_end, validation_end


def sort_chronologically(df, time_col=TIME_COLUMN):
    """Return ``df`` ordered by ``time_col`` with a fresh 0..n-1 index.

    Raises ``ValueError`` if any timestamp is missing or unparseable, because an
    unreliable order would silently corrupt the split.
    """
    ordered = df.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
    if ordered[time_col].isna().any():
        raise ValueError(
            f"{time_col} contains missing or invalid values; resolve them before splitting."
        )
    return ordered.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def chronological_split(df, time_col=TIME_COLUMN):
    """Return ``(train, validation, test)`` frames from chronologically sorted ``df``."""
    ordered = sort_chronologically(df, time_col)
    train_end, validation_end = split_bounds(len(ordered))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
