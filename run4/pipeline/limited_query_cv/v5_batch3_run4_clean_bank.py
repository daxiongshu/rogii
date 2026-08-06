from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from limited_query_cv import promote_v5_batch3_run2_r094 as legacy_r094
from limited_query_cv import v5_batch3_run2_reverse_residual_bank as primary


RUN_ID = "v5_batch3_run_004_goal_020"
RUN_ROOT = Path("limited_query_cv/runs") / RUN_ID
PARENT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit/sealed/outer_predictions.npz"
)
PARENT_SHA256 = (
    "737711c483cc3ce81ba94fa8e5dfb9c307f75c7691fd06be33631d80c355fad1"
)
LINEAGE_INVALIDATION = Path(
    "limited_query_cv/runs/v5_batch3_run_003_goal_020/"
    "r094_robust_lineage_invalidation.json"
)
LINEAGE_INVALIDATION_SHA256 = (
    "eb1605f352679372fd498154e2050276e6379ae9ae0acc93188ff157521cdb3d"
)
SAFE_MODE_ARCHIVE = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "original_safe_mode_bank_clean.npz"
)
SAFE_MODE_ARCHIVE_SHA256 = (
    "1dc2c98f971f5b66a31b3499685e6aadf5eddfe0c63a9be365d43457228673df"
)
SAFE_MODE_RESULT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "original_safe_mode_bank_result.json"
)
SAFE_MODE_RESULT_SHA256 = (
    "04fa4abf518ea1b67ace07328b32c250608b3bd7333c9be5bb0ecb73a3614ec3"
)


def verify_run4_lineage() -> list[dict[str, Any]]:
    if primary.sha256(LINEAGE_INVALIDATION) != LINEAGE_INVALIDATION_SHA256:
        raise ValueError("Run 3 lineage invalidation changed")
    invalidation = json.loads(LINEAGE_INVALIDATION.read_text())
    if (
        invalidation.get("status")
        != "r094_robust_invalidated_before_protected_metric"
        or invalidation["sealed_attempt"]["protected_metric_computed"]
        or invalidation["sealed_attempt"]["protected_reveal_spent"]
        or invalidation["clean_reassessment"]["mode_replacement"]
        != str(SAFE_MODE_ARCHIVE)
        or invalidation["clean_reassessment"]["mode_replacement_sha256"]
        != SAFE_MODE_ARCHIVE_SHA256
        or invalidation["clean_reassessment"][
            "historical_replacement_parent"
        ]
        != str(PARENT)
        or invalidation["clean_reassessment"][
            "historical_replacement_parent_sha256"
        ]
        != PARENT_SHA256
    ):
        raise ValueError("Run 3 clean replacement contract changed")
    if (
        primary.sha256(SAFE_MODE_ARCHIVE) != SAFE_MODE_ARCHIVE_SHA256
        or primary.sha256(SAFE_MODE_RESULT) != SAFE_MODE_RESULT_SHA256
    ):
        raise ValueError("original-fold-safe mode repair changed")
    safe_result = json.loads(SAFE_MODE_RESULT.read_text())
    if (
        safe_result.get("family")
        != "original_fold_safe_protected_mode_bank"
        or safe_result.get("output") != str(SAFE_MODE_ARCHIVE)
        or safe_result.get("output_sha256") != SAFE_MODE_ARCHIVE_SHA256
        or safe_result.get("forbidden_stratified_parent_loaded")
        or safe_result.get("audit_fold_loaded")
        or safe_result.get("confirmation_regroupings_loaded")
    ):
        raise ValueError("original-fold-safe mode attestation changed")
    if primary.sha256(PARENT) != PARENT_SHA256:
        raise ValueError("promoted C016 parent changed")

    # The legacy verifier deliberately fails because its fourth primary
    # archive is quarantined. Verify the remaining records directly, then
    # substitute the checksum-pinned original-fold-safe repair under the same
    # semantic candidate name.
    records: list[dict[str, Any]] = []
    for name, manifest_key, path, expected_hash in primary.ARCHIVES:
        if name == "reverse_protected_mode_maximin":
            continue
        if primary.sha256(path) != expected_hash:
            raise ValueError(f"{name}: primary archive changed")
        records.append(
            {
                "name": name,
                "manifest_key": manifest_key,
                "path": str(path),
                "sha256": expected_hash,
                "field": "prediction",
                "clean_replacement": False,
            }
        )
    records.insert(
        3,
        {
            "name": "reverse_protected_mode_maximin",
            "manifest_key": "original_fold_safe_mode_bank",
            "path": str(SAFE_MODE_ARCHIVE),
            "sha256": SAFE_MODE_ARCHIVE_SHA256,
            "field": "prediction",
            "clean_replacement": True,
            "result": str(SAFE_MODE_RESULT),
            "result_sha256": SAFE_MODE_RESULT_SHA256,
        },
    )
    expected_names = [record[0] for record in primary.ARCHIVES]
    if [record["name"] for record in records] != expected_names:
        raise ValueError("clean primary replacement order changed")
    return records


def production_historical_prediction() -> dict[str, np.ndarray]:
    if primary.sha256(PARENT) != PARENT_SHA256:
        raise ValueError("promoted C016 parent changed")
    with np.load(PARENT, allow_pickle=False) as parent:
        required = {
            "well_ids",
            "folds",
            "row_starts",
            "D570",
            "prediction",
            "hidden_truth_stored",
            "f4_metric_computed",
        }
        if (
            not required.issubset(parent.files)
            or "truth" in parent.files
            or bool(parent["hidden_truth_stored"])
            or bool(parent["f4_metric_computed"])
        ):
            raise ValueError("promoted C016 parent firewall changed")
        d570 = parent["D570"].astype(np.float64)
        c016 = parent["prediction"].astype(np.float64)
        result = {
            "well_ids": parent["well_ids"].astype(str),
            "folds": parent["folds"].astype(np.int64),
            "row_starts": parent["row_starts"].astype(np.int64),
            "prediction": d570 + 2.0 * (c016 - d570),
        }
    if (
        len(result["well_ids"]) != 773
        or set(result["folds"].tolist()) != set(range(5))
        or not np.isfinite(result["prediction"]).all()
    ):
        raise ValueError("production historical view is incomplete")
    return result


@contextmanager
def clean_run4_overrides() -> Iterator[None]:
    clean_primary_records = verify_run4_lineage()
    original_verifier = primary.verify_clean_lineage
    original_historical_loader = legacy_r094.load_historical_prediction
    original_historical_verifier = legacy_r094.verify_historical_result
    primary.verify_clean_lineage = lambda: clean_primary_records
    legacy_r094.load_historical_prediction = production_historical_prediction
    legacy_r094.verify_historical_result = lambda: {
        "status": "production_parity_full_ridge_from_promoted_C016"
    }
    try:
        yield
    finally:
        primary.verify_clean_lineage = original_verifier
        legacy_r094.load_historical_prediction = original_historical_loader
        legacy_r094.verify_historical_result = original_historical_verifier


def load_clean_fold_bank(
    fold: int,
    *,
    truth_allowed: bool,
) -> dict[str, Any]:
    if fold not in range(4):
        raise ValueError("Run 4 development bank is restricted to F0-F3")
    with clean_run4_overrides():
        bank = legacy_r094.load_fold_bank(
            fold,
            truth_allowed=truth_allowed,
        )
    expected = Path(
        "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
        "original_safe_mode_bank_clean.npz"
    )
    matching = [
        record
        for record in bank["records"]
        if record.get("name") == "reverse_protected_mode_maximin"
    ]
    if (
        not matching
        or any(
            Path(record["path"]) != expected
            or record["sha256"] != SAFE_MODE_ARCHIVE_SHA256
            for record in matching
        )
    ):
        raise ValueError("Run 4 clean mode replacement was not loaded")
    return bank
