from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from limited_query_cv import promote_v5_batch3_run4_boosted_router as router
from limited_query_cv.develop_v5_batch3_run4_well_utility_router import (
    evaluate,
    grouping_labels,
)
from limited_query_cv.v5_batch3_run4_clean_bank import RUN_ID, RUN_ROOT


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
FEATURES = RUN_ROOT / "router_features.npz"
KEPT_REVERSE = frozenset(
    {
        "reverse_physics_prior_mixed",
        "ramped_reverse_physics_prior_mixed",
        "reverse_protected_mode_maximin",
        "ramped_reverse_protected_mode_maximin",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--well-stats", type=Path, default=WELL_STATS)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--route-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    if args.route_output.exists() or args.result_output.exists():
        raise FileExistsError("refusing to overwrite uniform repair output")

    with np.load(args.well_stats, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise RuntimeError("development well-stat firewall changed")
        well_ids = stats["well_ids"].astype(str)
        folds = stats["original_folds"].astype(np.int64)
        rows = stats["rows"].astype(np.float64)
        clean_sse = stats["clean_sse"].astype(np.float64)
        cross = stats["cross"].astype(np.float64)
        gram = stats["gram"].astype(np.float64)
        names = stats["candidate_names"].astype(str)
    with np.load(args.features, allow_pickle=False) as feature_archive:
        if (
            bool(feature_archive["target_fields_read"])
            or bool(feature_archive["F4_loaded"])
            or bool(feature_archive["F4_metric_computed"])
            or bool(feature_archive["protected_reveal_spent"])
            or not np.array_equal(
                feature_archive["well_ids"].astype(str), well_ids
            )
        ):
            raise RuntimeError("development feature firewall changed")
        features = feature_archive["features"].astype(np.float64)

    disabled = []
    for index, name in enumerate(names):
        if (
            name.startswith("reverse_")
            or name.startswith("ramped_reverse_")
        ) and name not in KEPT_REVERSE:
            disabled.append(index)
    disabled_indices = np.asarray(disabled, dtype=np.int64)
    cross[:, disabled_indices] = 0.0
    gram[:, disabled_indices, :] = 0.0
    gram[:, :, disabled_indices] = 0.0

    route_weight = np.zeros((len(well_ids), 49), dtype=np.float64)
    for held_fold in range(4):
        held = folds == held_fold
        _, held_route = router.fit_route(
            features,
            rows,
            cross,
            gram,
            ~held,
            held,
            held_fold,
        )
        route_weight[held] = held_route
    results, all_group_gains = evaluate(
        grouping_labels(well_ids, folds),
        rows,
        clean_sse,
        cross,
        gram,
        route_weight,
    )
    minimum_gain = min(
        result["gain_vs_clean_C016"] for result in results.values()
    )
    worst_group = min(all_group_gains)
    passed = minimum_gain >= 0.17 and worst_group >= 0.0

    np.savez_compressed(
        args.route_output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray("uniform_deployable_physics_router"),
        well_ids=well_ids,
        original_folds=folds.astype(np.int8),
        candidate_names=names,
        route_weight=route_weight.astype(np.float32),
        disabled_indices=disabled_indices.astype(np.int16),
        disabled_names=names[disabled_indices],
        kept_reverse_names=np.asarray(sorted(KEPT_REVERSE)),
        F4_loaded=np.asarray(False),
        F4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
    )
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "stage": "development_exposed_uniform_deployment_repair",
        "status": "readiness_passed" if passed else "readiness_failed",
        "authorized_candidate_specific_threshold": 0.17,
        "candidate": "uniform deployable physics router",
        "kept_reverse_directions": sorted(KEPT_REVERSE),
        "uniformly_disabled_directions": names[disabled_indices].tolist(),
        "geometries": results,
        "minimum_exposed_geometry_gain": minimum_gain,
        "worst_held_group_gain": worst_group,
        "route_output": str(args.route_output),
        "route_output_sha256": router.sha256(args.route_output),
        "well_stats_sha256": router.sha256(args.well_stats),
        "features_sha256": router.sha256(args.features),
        "source_sha256": router.sha256(Path(__file__)),
        "firewall": {
            "F4_loaded": False,
            "F4_metric_computed": False,
            "protected_result_used_for_selection": False,
        },
    }
    args.result_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "minimum_exposed_geometry_gain": minimum_gain,
                "worst_held_group_gain": worst_group,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise RuntimeError("uniform deployable repair missed readiness")


if __name__ == "__main__":
    main()
