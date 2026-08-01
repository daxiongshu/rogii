from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from limited_query_cv.develop_v5_batch3_run4_well_utility_router import (
    SURFACE_INDICES,
    base_weight,
    evaluate,
    grouping_labels,
    select_single_route,
)
from limited_query_cv.v5_batch3_run4_clean_bank import RUN_ID, RUN_ROOT


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
FEATURES = RUN_ROOT / "router_features.npz"
PRIOR_ROUTE = RUN_ROOT / "clean_well_utility_router.npz"
PRIOR_ROUTE_SHA256 = (
    "d9658a3a5cdb0e5ad3e47485d3c87f7ffec56dfde7dc4b09271c9d6a91f93666"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train_surface_qnorm_oof(
    features: np.ndarray,
    folds: np.ndarray,
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros(
        (len(folds), len(SURFACE_INDICES)), dtype=np.float64
    )
    diagonal = np.stack(
        [gram[:, index, index] for index in SURFACE_INDICES],
        axis=1,
    )
    for held_fold in range(4):
        training = folds != held_fold
        held = ~training
        weight = base_weight(folds, held_fold, cross, gram)
        numerator = (
            cross[:, SURFACE_INDICES]
            + np.einsum(
                "nij,j->ni",
                gram[:, SURFACE_INDICES, :],
                weight,
            )
        )
        target = -numerator / np.sqrt(
            np.maximum(diagonal * rows[:, None], 1e-12)
        )
        lower = np.quantile(target[training], 0.01, axis=0)
        upper = np.quantile(target[training], 0.99, axis=0)
        target = np.clip(target, lower, upper)
        model = XGBRegressor(
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
            random_state=20265600 + held_fold,
        )
        model.fit(
            features[training],
            target[training],
            sample_weight=rows[training] / np.mean(rows[training]),
            verbose=False,
        )
        prediction[held] = model.predict(features[held])
    return prediction


def load_inputs(
    well_stats_path: Path,
    features_path: Path,
    prior_route_path: Path,
) -> dict[str, Any]:
    if sha256(prior_route_path) != PRIOR_ROUTE_SHA256:
        raise ValueError("Run 4 prior clean route changed")
    with np.load(well_stats_path, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise ValueError("Run 4 well-stat firewall changed")
        data: dict[str, Any] = {
            "well_ids": stats["well_ids"].astype(str),
            "folds": stats["original_folds"].astype(np.int64),
            "rows": stats["rows"].astype(np.float64),
            "clean_sse": stats["clean_sse"].astype(np.float64),
            "cross": stats["cross"].astype(np.float64),
            "gram": stats["gram"].astype(np.float64),
            "candidate_names": stats["candidate_names"].astype(str),
        }
    with np.load(features_path, allow_pickle=False) as feature_archive:
        if (
            bool(feature_archive["target_fields_read"])
            or bool(feature_archive["F4_loaded"])
            or bool(feature_archive["F4_metric_computed"])
            or bool(feature_archive["protected_reveal_spent"])
            or not np.array_equal(
                feature_archive["well_ids"].astype(str),
                data["well_ids"],
            )
        ):
            raise ValueError("Run 4 router-feature firewall changed")
        data["features"] = feature_archive["features"].astype(np.float64)
    with np.load(prior_route_path, allow_pickle=False) as prior:
        if (
            bool(prior["F4_loaded"])
            or bool(prior["F4_metric_computed"])
            or bool(prior["protected_reveal_spent"])
            or not np.array_equal(
                prior["well_ids"].astype(str), data["well_ids"]
            )
        ):
            raise ValueError("Run 4 prior-route firewall changed")
        data["second_index"] = prior["second_index"].astype(np.int64)
        data["second_amplitude"] = prior["second_amplitude"].astype(
            np.float64
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--well-stats", type=Path, default=WELL_STATS)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--prior-route", type=Path, default=PRIOR_ROUTE)
    parser.add_argument(
        "--route-output",
        type=Path,
        default=RUN_ROOT / "clean_boosted_well_utility_router.npz",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=(
            RUN_ROOT / "clean_boosted_well_utility_router_development.json"
        ),
    )
    args = parser.parse_args()
    if args.route_output.exists() or args.result_output.exists():
        raise FileExistsError(
            args.route_output
            if args.route_output.exists()
            else args.result_output
        )

    data = load_inputs(args.well_stats, args.features, args.prior_route)
    surface_qnorm = train_surface_qnorm_oof(
        data["features"],
        data["folds"],
        data["rows"],
        data["cross"],
        data["gram"],
    )
    surface_index, surface_amplitude = select_single_route(
        surface_qnorm,
        data["rows"],
        data["gram"],
        SURFACE_INDICES,
        allowed_local_indices=np.arange(4, dtype=np.int64),
        amplitude_cap=0.30,
        utility_threshold=0.20,
    )

    # Both scales were fixed after bounded development on the exposed
    # geometries. The conservative surface scale gives up 0.00039 worst-
    # geometry gain in exchange for 0.0151 more worst-group margin.
    surface_scale = 0.625
    second_scale = 1.25
    route_weight = np.zeros(
        (len(data["well_ids"]), 49), dtype=np.float64
    )
    route_weight[
        np.arange(len(data["well_ids"])), surface_index
    ] += surface_scale * surface_amplitude
    route_weight[
        np.arange(len(data["well_ids"])), data["second_index"]
    ] += second_scale * data["second_amplitude"]

    results, all_group_gains = evaluate(
        grouping_labels(data["well_ids"], data["folds"]),
        data["rows"],
        data["clean_sse"],
        data["cross"],
        data["gram"],
        route_weight,
    )
    pooled_gains = [
        result["gain_vs_clean_C016"] for result in results.values()
    ]

    np.savez_compressed(
        args.route_output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray("clean_boosted_well_utility_router"),
        well_ids=data["well_ids"],
        original_folds=data["folds"].astype(np.int8),
        candidate_names=data["candidate_names"],
        route_weight=route_weight.astype(np.float32),
        surface_index=surface_index.astype(np.int16),
        surface_amplitude=surface_amplitude.astype(np.float32),
        surface_scale=np.asarray(surface_scale, dtype=np.float32),
        surface_qnorm=surface_qnorm.astype(np.float32),
        second_index=data["second_index"].astype(np.int16),
        second_amplitude=data["second_amplitude"].astype(np.float32),
        second_scale=np.asarray(second_scale, dtype=np.float32),
        well_stats_sha256=np.asarray(sha256(args.well_stats)),
        feature_sha256=np.asarray(sha256(args.features)),
        prior_route_sha256=np.asarray(sha256(args.prior_route)),
        raw_hidden_target_stored=np.asarray(False),
        F4_loaded=np.asarray(False),
        F4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
    )
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "stage": "development_exposed_clean_boosted_well_router",
        "status": (
            "development_goal_passed"
            if min(pooled_gains) >= 0.20
            else "development_goal_not_yet_passed"
        ),
        "candidate": "clean boosted whole-well utility router",
        "fixed_recipe": {
            "base": {
                "historical_amplitude": 0.75,
                "relative_l2": 1e-4,
                "original_cap": 0.75,
                "reverse_cap": 0.75,
                "global_shrinkage": 0.875,
                "nuisance_index_suppressed": 17,
            },
            "surface_route": {
                "model": "XGBRegressor",
                "cross_fit": "four original folds",
                "n_estimators": 600,
                "learning_rate": 0.025,
                "max_depth": 1,
                "min_child_weight": 10,
                "subsample": 0.85,
                "colsample_bytree": 0.6,
                "reg_lambda": 15.0,
                "reg_alpha": 0.1,
                "candidate_indices": SURFACE_INDICES.tolist(),
                "allowed_local_indices": [0, 1, 2, 3],
                "amplitude_cap": 0.30,
                "utility_threshold": 0.20,
                "applied_scale": surface_scale,
            },
            "second_route": {
                "source": str(args.prior_route),
                "source_sha256": sha256(args.prior_route),
                "component": "ExtraTrees structural/reverse route",
                "applied_scale": second_scale,
            },
            "selection_rationale": (
                "The conservative surface scale sacrifices 0.00039 "
                "minimum-geometry gain versus the exposed optimum and "
                "improves the worst held-group margin by 0.0151."
            ),
            "target": (
                "negative clean-stacked-base error projection normalized "
                "by candidate energy and well rows"
            ),
            "held_group_truth_selects_own_route": False,
        },
        "geometries": results,
        "minimum_geometry_gain": min(pooled_gains),
        "mean_geometry_gain": float(np.mean(pooled_gains)),
        "minimum_held_group_gain": min(all_group_gains),
        "positive_held_groups": int(
            np.sum(np.asarray(all_group_gains) > 0.0)
        ),
        "held_groups": len(all_group_gains),
        "prior_minimum_geometry_gain": 0.16647490047932223,
        "gain_over_prior_minimum_geometry": (
            min(pooled_gains) - 0.16647490047932223
        ),
        "development_goal": 0.20,
        "readiness_shortfall": max(0.0, 0.20 - min(pooled_gains)),
        "route": str(args.route_output),
        "route_sha256": sha256(args.route_output),
        "well_stats": str(args.well_stats),
        "well_stats_sha256": sha256(args.well_stats),
        "features": str(args.features),
        "features_sha256": sha256(args.features),
        "prior_route": str(args.prior_route),
        "prior_route_sha256": sha256(args.prior_route),
        "firewall": {
            "development_original_folds_loaded": [0, 1, 2, 3],
            "F4_loaded": False,
            "F4_metric_computed": False,
            "protected_reveal_spent": False,
        },
        "kaggle_dataset_authorized": False,
        "kaggle_kernel_authorized": False,
        "kaggle_submission_authorized": False,
    }
    args.result_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
