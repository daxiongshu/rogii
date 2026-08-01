from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HISTORICAL_AMPLITUDES = (0.0, 0.25, 0.5, 0.75, 1.0)
HISTORICAL_ARCHIVE = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "clean_multimodel_ridge_clean.npz"
)
HISTORICAL_RESULT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "clean_multimodel_ridge_result.json"
)
EXPECTED_ARCHIVE_SHA256 = (
    "82cc3319a6752273973599bfcc0725790a24936e39d04139a422f6f10ffffbd1"
)
EXPECTED_RESULT_SHA256 = (
    "27cb0aa5af7670ff32adf942aac4c4ba0ab9763e75d42904e7aebb7c206d7105"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_historical_result(path: Path = HISTORICAL_RESULT) -> dict:
    if path == HISTORICAL_RESULT and sha256(path) != EXPECTED_RESULT_SHA256:
        raise ValueError("historical result hash changed")
    result = json.loads(path.read_text())
    expected_components = [
        "protected_posterior_incremental",
        "protected_mode_bank",
    ]
    if (
        result.get("status") != "complete_selection_clean_F0_F3"
        or not result.get("selection_clean_nested_selector_run")
        or result.get("audit_fold_loaded")
        or result.get("confirmation_regroupings_loaded")
        or result.get("forbidden_stratified_parent_loaded")
        or result.get("removed_component") != "D072"
        or result.get("components") != expected_components
    ):
        raise ValueError("historical result is not the frozen clean lineage")
    return result


def load_historical_prediction(
    path: Path = HISTORICAL_ARCHIVE,
    *,
    enforce_hash: bool = True,
) -> dict[str, np.ndarray]:
    if (
        enforce_hash
        and path == HISTORICAL_ARCHIVE
        and sha256(path) != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("historical prediction hash changed")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "well_ids",
            "folds",
            "row_starts",
            "prediction",
            "selection_clean_nested_selector_run",
            "audit_fold_loaded",
            "confirmation_regroupings_loaded",
            "forbidden_stratified_parent_loaded",
        }
        if not required.issubset(archive.files):
            raise ValueError("historical prediction schema changed")
        if (
            not bool(archive["selection_clean_nested_selector_run"])
            or bool(archive["audit_fold_loaded"])
            or bool(archive["confirmation_regroupings_loaded"])
            or bool(archive["forbidden_stratified_parent_loaded"])
        ):
            raise ValueError("historical prediction firewall failed")
        result = {
            "well_ids": archive["well_ids"].astype(str),
            "folds": archive["folds"].astype(np.int64),
            "row_starts": archive["row_starts"].astype(np.int64),
            "prediction": archive["prediction"].astype(np.float64),
        }
    if (
        len(result["row_starts"]) != len(result["well_ids"]) + 1
        or int(result["row_starts"][0]) != 0
        or int(result["row_starts"][-1]) != len(result["prediction"])
        or np.any(np.diff(result["row_starts"]) <= 0)
        or not np.isfinite(result["prediction"]).all()
    ):
        raise ValueError("historical prediction layout changed")
    return result


def extract_fold_prediction(
    archive: dict[str, np.ndarray],
    fold: int,
    expected_ids: np.ndarray,
) -> np.ndarray:
    indices = np.flatnonzero(archive["folds"] == int(fold))
    ids = archive["well_ids"][indices]
    if not np.array_equal(ids, np.asarray(expected_ids).astype(str)):
        raise ValueError(f"historical fold {fold} well order changed")
    starts = archive["row_starts"]
    return np.concatenate(
        [
            archive["prediction"][starts[index] : starts[index + 1]]
            for index in indices
        ]
    )
