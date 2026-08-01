from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from limited_query_cv.train_v5_batch3_run2_reverse_bank import (
    conditional_weight,
)
from limited_query_cv.v5_batch3_run4_clean_bank import RUN_ID, RUN_ROOT


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
FEATURES = RUN_ROOT / "router_features.npz"
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
FRESH_ASSIGNMENT = Path(
    "limited_query_cv/runs/v5_batch3_run_003_goal_020/"
    "fresh_confirmation_assignment.json"
)
SURFACE_INDICES = np.arange(12, 18, dtype=np.int64)
NON_SURFACE_INDICES = np.concatenate(
    (
        np.arange(12, dtype=np.int64),
        np.arange(18, 28, dtype=np.int64),
        np.arange(29, 39, dtype=np.int64),
    )
)
STRUCTURAL_REVERSE_LOCAL = np.concatenate(
    (
        np.arange(12, dtype=np.int64),
        np.arange(22, 32, dtype=np.int64),
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_weight(
    labels: np.ndarray,
    held_group: int,
    cross: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    training = labels != held_group
    total_cross = np.sum(cross[training], axis=0)
    total_gram = np.sum(gram[training], axis=0)
    weight = np.zeros(49, dtype=np.float64)
    weight[:39] = conditional_weight(
        total_cross[:39],
        total_gram[:39, :39],
        28,
        np.arange(29, 39, dtype=np.int64),
        0.75,
        1e-4,
        0.75,
        0.75,
    )
    weight *= 0.875
    weight[17] = 0.0
    return weight


def train_qnorm_oof(
    features: np.ndarray,
    folds: np.ndarray,
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    trees: int,
    seed: int,
) -> np.ndarray:
    prediction = np.zeros(
        (len(folds), len(candidate_indices)), dtype=np.float64
    )
    diagonal = np.stack(
        [gram[:, index, index] for index in candidate_indices],
        axis=1,
    )
    for held_fold in range(4):
        training = folds != held_fold
        held = ~training
        weight = base_weight(folds, held_fold, cross, gram)
        numerator = (
            cross[:, candidate_indices]
            + np.einsum(
                "nij,j->ni",
                gram[:, candidate_indices, :],
                weight,
            )
        )
        target = -numerator / np.sqrt(
            np.maximum(diagonal * rows[:, None], 1e-12)
        )
        lower = np.quantile(target[training], 0.01, axis=0)
        upper = np.quantile(target[training], 0.99, axis=0)
        target = np.clip(target, lower, upper)
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=16,
            max_features=0.7,
            random_state=seed + held_fold,
            n_jobs=16,
        )
        model.fit(
            features[training],
            target[training],
            sample_weight=rows[training] / np.mean(rows[training]),
        )
        prediction[held] = model.predict(features[held])
    return prediction


def select_single_route(
    prediction: np.ndarray,
    rows: np.ndarray,
    gram: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    allowed_local_indices: np.ndarray,
    amplitude_cap: float,
    utility_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.stack(
        [gram[:, index, index] for index in candidate_indices],
        axis=1,
    )
    raw_amplitude = np.clip(
        prediction
        * np.sqrt(rows[:, None] / np.maximum(diagonal, 1e-12)),
        0.0,
        amplitude_cap,
    )
    utility = (
        2.0
        * raw_amplitude
        * prediction
        * np.sqrt(diagonal * rows[:, None])
        - np.square(raw_amplitude) * diagonal
    ) / rows[:, None]
    allowed = np.zeros(len(candidate_indices), dtype=bool)
    allowed[allowed_local_indices] = True
    utility[:, ~allowed] = -np.inf
    selected_local = np.argmax(utility, axis=1)
    selected_utility = utility[
        np.arange(len(rows)), selected_local
    ]
    selected_amplitude = raw_amplitude[
        np.arange(len(rows)), selected_local
    ]
    selected_amplitude[selected_utility <= utility_threshold] = 0.0
    return candidate_indices[selected_local], selected_amplitude


def grouping_labels(
    well_ids: np.ndarray,
    original: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(LAYOUT, allow_pickle=False) as layout:
        layout_ids = layout["well_ids"].astype(str)
        lookup = {
            well_id: index for index, well_id in enumerate(layout_ids)
        }
        balanced = np.asarray(
            [layout["stratified"][lookup[well_id]] for well_id in well_ids],
            dtype=np.int64,
        )
        independent = np.asarray(
            [layout["independent"][lookup[well_id]] for well_id in well_ids],
            dtype=np.int64,
        )
    fresh_lookup = json.loads(FRESH_ASSIGNMENT.read_text())["assignments"]
    fresh = np.asarray(
        [fresh_lookup[well_id] for well_id in well_ids],
        dtype=np.int64,
    )
    return {
        "original": original,
        "balanced_exposed": balanced,
        "independent_exposed": independent,
        "former_fresh_exposed": fresh,
    }


def evaluate(
    labels_by_geometry: dict[str, np.ndarray],
    rows: np.ndarray,
    clean_sse: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    route_weight: np.ndarray,
) -> tuple[dict[str, Any], list[float]]:
    results = {}
    all_group_gains = []
    clean_rmse = float(np.sqrt(np.sum(clean_sse) / np.sum(rows)))
    for name, labels in labels_by_geometry.items():
        candidate_total = 0.0
        groups = []
        for held_group in sorted(set(labels.tolist())):
            held = labels == held_group
            weight = (
                route_weight[held]
                + base_weight(
                    labels, held_group, cross, gram
                )[None, :]
            )
            candidate_sse = (
                clean_sse[held]
                + 2.0 * np.einsum("ni,ni->n", cross[held], weight)
                + np.einsum(
                    "ni,nij,nj->n",
                    weight,
                    gram[held],
                    weight,
                )
            )
            candidate_total += float(np.sum(candidate_sse))
            held_rows = int(np.sum(rows[held]))
            held_clean_rmse = float(
                np.sqrt(np.sum(clean_sse[held]) / held_rows)
            )
            held_candidate_rmse = float(
                np.sqrt(np.sum(candidate_sse) / held_rows)
            )
            gain = held_clean_rmse - held_candidate_rmse
            all_group_gains.append(gain)
            groups.append(
                {
                    "held_group": int(held_group),
                    "rows": held_rows,
                    "clean_C016_rmse": held_clean_rmse,
                    "candidate_rmse": held_candidate_rmse,
                    "gain_vs_clean_C016": gain,
                }
            )
        candidate_rmse = float(
            np.sqrt(candidate_total / np.sum(rows))
        )
        results[name] = {
            "rows": int(np.sum(rows)),
            "clean_C016_rmse": clean_rmse,
            "candidate_rmse": candidate_rmse,
            "gain_vs_clean_C016": clean_rmse - candidate_rmse,
            "groups": groups,
        }
    return results, all_group_gains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--well-stats", type=Path, default=WELL_STATS)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument(
        "--route-output",
        type=Path,
        default=RUN_ROOT / "clean_well_utility_router.npz",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=RUN_ROOT / "clean_well_utility_router_development.json",
    )
    args = parser.parse_args()
    if args.route_output.exists() or args.result_output.exists():
        raise FileExistsError(
            args.route_output
            if args.route_output.exists()
            else args.result_output
        )

    with np.load(args.well_stats, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise ValueError("Run 4 well-stat firewall changed")
        well_ids = stats["well_ids"].astype(str)
        folds = stats["original_folds"].astype(np.int64)
        rows = stats["rows"].astype(np.float64)
        clean_sse = stats["clean_sse"].astype(np.float64)
        cross = stats["cross"].astype(np.float64)
        gram = stats["gram"].astype(np.float64)
        candidate_names = stats["candidate_names"].astype(str)
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
            raise ValueError("Run 4 router-feature firewall changed")
        features = feature_archive["features"].astype(np.float64)

    surface_qnorm = train_qnorm_oof(
        features,
        folds,
        rows,
        cross,
        gram,
        SURFACE_INDICES,
        trees=500,
        seed=20265116,
    )
    surface_index, surface_amplitude = select_single_route(
        surface_qnorm,
        rows,
        gram,
        SURFACE_INDICES,
        allowed_local_indices=np.asarray([0, 2], dtype=np.int64),
        amplitude_cap=0.15,
        utility_threshold=0.20,
    )
    non_surface_qnorm = train_qnorm_oof(
        features,
        folds,
        rows,
        cross,
        gram,
        NON_SURFACE_INDICES,
        trees=700,
        seed=20265216,
    )
    second_index, second_amplitude = select_single_route(
        non_surface_qnorm,
        rows,
        gram,
        NON_SURFACE_INDICES,
        allowed_local_indices=STRUCTURAL_REVERSE_LOCAL,
        amplitude_cap=0.20,
        utility_threshold=0.01,
    )
    route_weight = np.zeros((len(well_ids), 49), dtype=np.float64)
    route_weight[np.arange(len(well_ids)), surface_index] += (
        surface_amplitude
    )
    route_weight[np.arange(len(well_ids)), second_index] += (
        second_amplitude
    )
    results, all_group_gains = evaluate(
        grouping_labels(well_ids, folds),
        rows,
        clean_sse,
        cross,
        gram,
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
        family=np.asarray("clean_well_utility_router"),
        well_ids=well_ids,
        original_folds=folds.astype(np.int8),
        candidate_names=candidate_names,
        route_weight=route_weight.astype(np.float32),
        surface_index=surface_index.astype(np.int16),
        surface_amplitude=surface_amplitude.astype(np.float32),
        second_index=second_index.astype(np.int16),
        second_amplitude=second_amplitude.astype(np.float32),
        surface_qnorm=surface_qnorm.astype(np.float32),
        non_surface_qnorm=non_surface_qnorm.astype(np.float32),
        well_stats_sha256=np.asarray(sha256(args.well_stats)),
        feature_sha256=np.asarray(sha256(args.features)),
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
        "stage": "development_exposed_clean_well_utility_router",
        "status": (
            "development_goal_passed"
            if min(pooled_gains) >= 0.20
            else "development_goal_not_yet_passed"
        ),
        "candidate": "clean two-stage whole-well utility router",
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
                "model": "ExtraTreesRegressor",
                "trees": 500,
                "min_samples_leaf": 16,
                "max_features": 0.7,
                "candidate_indices": SURFACE_INDICES.tolist(),
                "allowed_local_indices": [0, 2],
                "amplitude_cap": 0.15,
                "utility_threshold": 0.20,
            },
            "second_route": {
                "model": "ExtraTreesRegressor",
                "trees": 700,
                "min_samples_leaf": 16,
                "max_features": 0.7,
                "candidate_indices": NON_SURFACE_INDICES.tolist(),
                "allowed_local_indices": (
                    STRUCTURAL_REVERSE_LOCAL.tolist()
                ),
                "amplitude_cap": 0.20,
                "utility_threshold": 0.01,
            },
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
        "route": str(args.route_output),
        "route_sha256": sha256(args.route_output),
        "well_stats": str(args.well_stats),
        "well_stats_sha256": sha256(args.well_stats),
        "features": str(args.features),
        "features_sha256": sha256(args.features),
        "negative_controls": {
            "row_residual": (
                "Rejected after F0; every tested positive share regressed."
            ),
            "row_surface_amplitude": (
                "Rejected after frozen F0 recipe regressed pooled F1-F3 "
                "and three exposed regroupings."
            ),
            "cluster_identity_and_cluster_prior": (
                "Neither one-hot cluster identity nor shrinkage-regularized "
                "cluster utility priors improved the clean router."
            ),
        },
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
