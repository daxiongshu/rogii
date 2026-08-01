from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from limited_query_cv.develop_v5_batch3_run4_well_utility_router import (
    NON_SURFACE_INDICES,
    STRUCTURAL_REVERSE_LOCAL,
    SURFACE_INDICES,
    base_weight,
    evaluate,
    select_single_route,
)
from limited_query_cv.v5_batch3_run4_clean_bank import RUN_ID, RUN_ROOT


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
FEATURES = RUN_ROOT / "router_features.npz"
ASSIGNMENT = RUN_ROOT / "promotion_confirmation_assignment.json"
PREREGISTRATION = RUN_ROOT / "boosted_router_confirmation_preregistration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = json.loads(PREREGISTRATION.read_text())
    assignment = json.loads(ASSIGNMENT.read_text())
    if (
        preregistration["status"]
        != "frozen_before_fresh_confirmation_metric"
        or preregistration["run_id"] != RUN_ID
        or preregistration["assignment_sha256"] != sha256(ASSIGNMENT)
        or preregistration["confirmation_source_sha256"]
        != sha256(Path(__file__))
        or preregistration["well_stats_sha256"] != sha256(WELL_STATS)
        or preregistration["features_sha256"] != sha256(FEATURES)
        or assignment["status"] != "generated_metrics_unopened"
        or assignment["firewall"]["F4_target_loaded"] is not False
        or assignment["firewall"]["candidate_metrics_opened"] is not False
        or assignment["firewall"]["protected_reveal_spent"] is not False
    ):
        raise RuntimeError("Run 4 confirmation contract changed")
    return preregistration, assignment


def qnorm_target(
    labels: np.ndarray,
    held_group: int,
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    weight = base_weight(labels, held_group, cross, gram)
    diagonal = np.stack(
        [gram[:, index, index] for index in candidate_indices], axis=1
    )
    numerator = cross[:, candidate_indices] + np.einsum(
        "nij,j->ni", gram[:, candidate_indices, :], weight
    )
    return -numerator / np.sqrt(
        np.maximum(diagonal * rows[:, None], 1e-12)
    )


def crossfit_routes(
    features: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    route_weight = np.zeros((len(labels), 49), dtype=np.float64)
    for held_group in sorted(set(labels.tolist())):
        training = labels != held_group
        held = ~training

        surface_target = qnorm_target(
            labels,
            held_group,
            rows,
            cross,
            gram,
            SURFACE_INDICES,
        )
        lower = np.quantile(surface_target[training], 0.01, axis=0)
        upper = np.quantile(surface_target[training], 0.99, axis=0)
        surface_target = np.clip(surface_target, lower, upper)
        surface_model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.025,
            max_depth=1,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.6,
            reg_lambda=15.0,
            reg_alpha=0.1,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=16,
            random_state=20265600 + held_group,
        )
        surface_model.fit(
            features[training],
            surface_target[training],
            sample_weight=rows[training] / np.mean(rows[training]),
            verbose=False,
        )
        surface_index, surface_amplitude = select_single_route(
            surface_model.predict(features[held]),
            rows[held],
            gram[held],
            SURFACE_INDICES,
            allowed_local_indices=np.arange(4, dtype=np.int64),
            amplitude_cap=0.30,
            utility_threshold=0.20,
        )

        second_target = qnorm_target(
            labels,
            held_group,
            rows,
            cross,
            gram,
            NON_SURFACE_INDICES,
        )
        lower = np.quantile(second_target[training], 0.01, axis=0)
        upper = np.quantile(second_target[training], 0.99, axis=0)
        second_target = np.clip(second_target, lower, upper)
        second_model = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=16,
            max_features=0.7,
            random_state=20265216 + held_group,
            n_jobs=16,
        )
        second_model.fit(
            features[training],
            second_target[training],
            sample_weight=rows[training] / np.mean(rows[training]),
        )
        second_index, second_amplitude = select_single_route(
            second_model.predict(features[held]),
            rows[held],
            gram[held],
            NON_SURFACE_INDICES,
            allowed_local_indices=STRUCTURAL_REVERSE_LOCAL,
            amplitude_cap=0.20,
            utility_threshold=0.01,
        )
        held_rows = np.flatnonzero(held)
        route_weight[held_rows, surface_index] += 0.625 * surface_amplitude
        route_weight[held_rows, second_index] += 1.25 * second_amplitude
    return route_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "promotion_confirmation_result.json",
    )
    args = parser.parse_args()
    preregistration, assignment = verify_contract()

    with np.load(WELL_STATS, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise RuntimeError("Run 4 well-stat firewall changed")
        well_ids = stats["well_ids"].astype(str)
        original_folds = stats["original_folds"].astype(np.int64)
        rows = stats["rows"].astype(np.float64)
        clean_sse = stats["clean_sse"].astype(np.float64)
        cross = stats["cross"].astype(np.float64)
        gram = stats["gram"].astype(np.float64)
    with np.load(FEATURES, allow_pickle=False) as feature_archive:
        if (
            bool(feature_archive["target_fields_read"])
            or bool(feature_archive["F4_loaded"])
            or bool(feature_archive["F4_metric_computed"])
            or bool(feature_archive["protected_reveal_spent"])
            or not np.array_equal(
                feature_archive["well_ids"].astype(str), well_ids
            )
        ):
            raise RuntimeError("Run 4 feature firewall changed")
        features = feature_archive["features"].astype(np.float64)
    lookup = {
        str(well_id): int(group)
        for well_id, group in assignment["assignments"].items()
    }
    if set(lookup) != set(well_ids):
        raise RuntimeError("fresh confirmation assignment is incomplete")
    labels = np.asarray([lookup[well_id] for well_id in well_ids])
    route_weight = crossfit_routes(
        features, labels, rows, cross, gram
    )
    geometry, all_group_gains = evaluate(
        {"fresh_confirmation": labels},
        rows,
        clean_sse,
        cross,
        gram,
        route_weight,
    )
    result = geometry["fresh_confirmation"]
    gates = {
        "minimum_pooled_gain_0p10": (
            result["gain_vs_clean_C016"] >= 0.10
        ),
        "maximum_held_group_regression_0p01": (
            max(0.0, -min(all_group_gains)) <= 0.01
        ),
        "all_five_groups_positive": all(
            gain > 0.0 for gain in all_group_gains
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "stage": "promotion_fresh_confirmation_result",
        "status": (
            "fresh_confirmation_passed"
            if passed
            else "fresh_confirmation_failed"
        ),
        "candidate": "clean boosted whole-well utility router",
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "assignment": str(ASSIGNMENT),
        "assignment_sha256": sha256(ASSIGNMENT),
        "result": result,
        "gates": gates,
        "fresh_confirmation_passed": passed,
        "exception_entry_threshold": 0.17,
        "stored_effective_readiness_target": preregistration[
            "stored_effective_readiness_target"
        ],
        "stored_goal_changed": False,
        "firewall": {
            "development_original_folds_loaded": [0, 1, 2, 3],
            "F4_loaded": False,
            "F4_metric_computed": False,
            "protected_reveal_spent": False,
            "kaggle_dataset_authorized": False,
            "kaggle_kernel_authorized": False,
            "kaggle_submission_authorized": False,
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
