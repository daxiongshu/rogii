from __future__ import annotations

import numpy as np

from limited_query_cv.prepare_v5_batch3_run2_combined_stack_stats import (
    PARENT,
    load_parent_fold,
)
from limited_query_cv.train_v5_batch3_nested_linear_stack import (
    L2_GRID,
    fit_stack,
)
from limited_query_cv.v5_batch3_run2_reverse_residual_bank import (
    load_reverse_directions,
)
from limited_query_cv.v5_batch3_run2_secondary_reverse_bank import (
    load_secondary_reverse_directions,
)


RAMPED_REVERSE_TOTAL_CAPS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def hidden_progress(lengths: np.ndarray) -> np.ndarray:
    pieces = []
    for length in np.asarray(lengths).astype(int):
        if length < 1:
            raise ValueError("well has no hidden rows")
        pieces.append(
            np.zeros(1, dtype=np.float64)
            if length == 1
            else np.linspace(0.0, 1.0, length, dtype=np.float64)
        )
    return np.concatenate(pieces)


def fit_ramped_reverse_bank(
    cross: np.ndarray,
    gram: np.ndarray,
    relative_l2: float,
    total_cap: float,
) -> np.ndarray:
    if total_cap == 0.0:
        return np.zeros(len(cross), dtype=np.float64)
    return fit_stack(cross, gram, relative_l2, total_cap)


def load_ramped_reverse_directions(
    fold: int,
    expected_well_ids: np.ndarray,
    clean_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    ids, parent_indices, parent_starts, parent_clean = load_parent_fold(
        fold, PARENT
    )
    if (
        not np.array_equal(ids.astype(str), expected_well_ids.astype(str))
        or not np.array_equal(
            parent_clean.astype(np.float32),
            np.asarray(clean_prediction).astype(np.float32),
        )
    ):
        raise ValueError("ramped reverse parent layout changed")
    lengths = np.asarray(
        [
            parent_starts[index + 1] - parent_starts[index]
            for index in parent_indices
        ],
        dtype=np.int64,
    )
    ramp = hidden_progress(lengths)
    primary_names, primary, primary_records = load_reverse_directions(
        fold, ids, clean_prediction
    )
    secondary_names, secondary, secondary_records = (
        load_secondary_reverse_directions(
            fold, ids, clean_prediction
        )
    )
    names = np.char.add(
        "ramped_",
        np.concatenate([primary_names, secondary_names]),
    )
    directions = np.concatenate([primary, secondary]) * ramp[None]
    if (
        directions.shape != (10, len(clean_prediction))
        or len(ramp) != len(clean_prediction)
        or not np.isfinite(directions).all()
        or np.min(ramp) != 0.0
        or np.max(ramp) != 1.0
    ):
        raise ValueError("ramped reverse direction schema changed")
    return (
        names,
        directions,
        lengths,
        primary_records + secondary_records,
    )
