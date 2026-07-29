from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


def fold_indices(
    identifiers: np.ndarray,
    fold: int,
    splits: int = 5,
    fold_file: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return whole-well train/validation indices from a saved assignment."""
    if not 0 <= fold < splits:
        raise ValueError(f"fold must be in [0, {splits}), got {fold}")
    if fold_file is None:
        return list(
            GroupKFold(splits).split(identifiers, groups=identifiers)
        )[fold]

    payload = json.loads(fold_file.read_text())
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError(f"{fold_file}: missing assignment mapping")
    expected = {str(identifier) for identifier in identifiers}
    observed = set(assignments)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{fold_file}: well mismatch; missing={missing[:5]} extra={extra[:5]}"
        )
    values = np.asarray(
        [int(assignments[str(identifier)]) for identifier in identifiers],
        dtype=np.int64,
    )
    if np.any((values < 0) | (values >= splits)):
        raise ValueError(f"{fold_file}: assignments must be in [0, {splits})")
    counts = np.bincount(values, minlength=splits)
    if np.any(counts == 0):
        raise ValueError(f"{fold_file}: empty fold in counts={counts.tolist()}")
    valid = np.flatnonzero(values == fold)
    train = np.flatnonzero(values != fold)
    return train, valid
