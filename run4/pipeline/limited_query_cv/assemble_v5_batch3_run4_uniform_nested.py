from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv import promote_v5_batch3_run4_boosted_router as router
from limited_query_cv.select_v5_batch3_run4_physics_nested import (
    extract_fold,
    role_name,
)
from limited_query_cv.v5_batch3_run4_clean_bank import load_clean_fold_bank


RUN_ID = "v5_batch3_run_004_goal_020"
FAMILY = "uniform_deployable_physics_router"
PHYSICS_INDEX = 29
RAMPED_PHYSICS_INDEX = 39
DISABLED_INDICES = np.asarray(
    [30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 43, 44, 45, 46, 47, 48],
    dtype=np.int64,
)


def ramp(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(1, dtype=np.float64)
            if length == 1
            else np.linspace(0.0, 1.0, int(length), dtype=np.float64)
            for length in lengths
        ]
    )


def base_bank(fold: int) -> dict[str, Any]:
    bank = (
        load_clean_fold_bank(fold, truth_allowed=False)
        if fold < 4
        else router.load_promotion_bank(4, truth_allowed=False)
    )
    bank = dict(bank)
    bank["correction"] = bank["correction"].astype(np.float64).copy()
    bank["correction"][DISABLED_INDICES] = 0.0
    return bank


def physics_from_pair(
    physics_root: Path,
    outer: int,
    inner: int,
    recipe: dict[str, Any],
    expected_ids: np.ndarray,
) -> np.ndarray:
    role = role_name(outer, inner)
    snapshot = int(recipe["snapshot"])
    temperature = float(recipe["temperature"])
    weight = float(recipe["physics_blend_weight"])
    path = physics_root / role / f"epoch{snapshot:03d}_prediction.npz"
    ids, _, baseline, predictions = extract_fold(path, inner)
    if not np.array_equal(ids, expected_ids.astype(str)):
        raise RuntimeError(f"{role}: physics-bank well order changed")
    return baseline + weight * (predictions[temperature] - baseline)


def physics_from_outer(
    physics_prediction: Path,
    fold: int,
    expected_ids: np.ndarray,
) -> np.ndarray:
    with np.load(physics_prediction, allow_pickle=False) as archive:
        if (
            bool(archive["hidden_truth_stored"])
            or bool(archive["protected_metric_computed"])
            or bool(archive["F4_metric_recomputed"])
            or "truth" in archive.files
        ):
            raise RuntimeError("physics outer prediction firewall changed")
        ids = archive["well_ids"].astype(str)
        folds = archive["folds"].astype(np.int64)
        starts = archive["row_starts"].astype(np.int64)
        values = archive["prediction"].astype(np.float64)
    pieces = []
    for well_id in expected_ids.astype(str):
        found = np.flatnonzero((ids == well_id) & (folds == fold))
        if len(found) != 1:
            raise RuntimeError(f"fold {fold}: physics outer well missing")
        index = int(found[0])
        pieces.append(values[starts[index] : starts[index + 1]])
    return np.concatenate(pieces)


def install_physics(
    bank: dict[str, Any],
    physics_prediction: np.ndarray,
) -> dict[str, Any]:
    if physics_prediction.shape != bank["clean"].shape:
        raise RuntimeError("physics prediction shape changed")
    bank["correction"][PHYSICS_INDEX] = (
        bank["clean"] - physics_prediction
    )
    lengths = np.diff(bank["starts"]).astype(np.int64)
    bank["correction"][RAMPED_PHYSICS_INDEX] = (
        bank["correction"][PHYSICS_INDEX] * ramp(lengths)
    )
    if not np.isfinite(bank["correction"]).all():
        raise RuntimeError("uniform candidate bank is nonfinite")
    return bank


def selector_lookup(path: Path) -> dict[int, dict[str, Any]]:
    record = json.loads(path.read_text())
    if (
        record["status"] != "complete_nested_selection_without_outer_metric"
        or record["F4_metric_recomputed"]
    ):
        raise RuntimeError("physics selector contract changed")
    return {int(item["outer_fold"]): item for item in record["roles"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--physics-prediction", type=Path, required=True)
    parser.add_argument("--physics-selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest_output.exists():
        raise FileExistsError("refusing to overwrite uniform nested assembly")

    recipes = selector_lookup(args.physics_selector)
    with np.load(router.PARENT, allow_pickle=False) as parent:
        well_ids = parent["well_ids"].astype(str)
        folds = parent["folds"].astype(np.int64)
        starts = parent["row_starts"].astype(np.int64)
        clean = parent["prediction"].astype(np.float64)
        baseline = parent["baseline"].astype(np.float32)
    with np.load(router.LAYOUT, allow_pickle=False) as layout:
        if (
            not np.array_equal(layout["well_ids"].astype(str), well_ids)
            or not np.array_equal(layout["row_starts"].astype(np.int64), starts)
        ):
            raise RuntimeError("protected layout changed")
        truth = layout["truth"].astype(np.float64)
    with np.load(router.DEV_FEATURES, allow_pickle=False) as dev:
        dev_features = {
            well_id: dev["features"][index].astype(np.float64)
            for index, well_id in enumerate(dev["well_ids"].astype(str))
        }
    with np.load(router.F4_FEATURES, allow_pickle=False) as f4:
        if bool(f4["F4_target_loaded"]) or bool(f4["F4_metric_computed"]):
            raise RuntimeError("F4 router feature firewall changed")
        f4_features = {
            well_id: f4["features"][index].astype(np.float64)
            for index, well_id in enumerate(f4["well_ids"].astype(str))
        }
    features = np.stack(
        [
            (dev_features if fold < 4 else f4_features)[well_id]
            for well_id, fold in zip(well_ids, folds)
        ]
    )
    rows = np.diff(starts).astype(np.float64)
    prediction = clean.copy()
    outer_bases = np.zeros((5, 49), dtype=np.float64)
    outer_routes = np.zeros((len(well_ids), 49), dtype=np.float32)
    outer_records = []
    for outer in range(5):
        cross = np.zeros((len(well_ids), 49), dtype=np.float64)
        gram = np.zeros((len(well_ids), 49, 49), dtype=np.float64)
        for inner in range(5):
            if inner == outer:
                continue
            bank = base_bank(inner)
            pair_physics = physics_from_pair(
                args.physics_root,
                outer,
                inner,
                recipes[outer],
                bank["ids"],
            )
            bank = install_physics(bank, pair_physics)
            for local_index, parent_index in enumerate(bank["parent_indices"]):
                left, right = bank["starts"][local_index : local_index + 2]
                target_left, target_right = starts[
                    parent_index : parent_index + 2
                ]
                error = bank["clean"][left:right] - truth[
                    target_left:target_right
                ]
                correction = bank["correction"][:, left:right]
                cross[parent_index] = correction @ error
                gram[parent_index] = correction @ correction.T

        held_bank = base_bank(outer)
        held_physics = physics_from_outer(
            args.physics_prediction, outer, held_bank["ids"]
        )
        held_bank = install_physics(held_bank, held_physics)
        for local_index, parent_index in enumerate(held_bank["parent_indices"]):
            left, right = held_bank["starts"][local_index : local_index + 2]
            correction = held_bank["correction"][:, left:right]
            gram[parent_index] = correction @ correction.T

        training = folds != outer
        held = ~training
        base, route = router.fit_route(
            features,
            rows,
            cross,
            gram,
            training,
            held,
            outer,
        )
        outer_bases[outer] = base
        outer_routes[held] = route.astype(np.float32)
        route_by_parent = {
            parent_index: route[index]
            for index, parent_index in enumerate(np.flatnonzero(held))
        }
        for local_index, parent_index in enumerate(held_bank["parent_indices"]):
            left, right = held_bank["starts"][local_index : local_index + 2]
            target_left, target_right = starts[parent_index : parent_index + 2]
            weight = base + route_by_parent[parent_index]
            prediction[target_left:target_right] = (
                held_bank["clean"][left:right]
                + weight @ held_bank["correction"][:, left:right]
            )
        outer_records.append(
            {
                "outer_fold": outer,
                "training_folds": np.flatnonzero(training).tolist(),
                "physics_recipe": recipes[outer],
                "positive_base_weights": int(np.sum(base > 1e-10)),
                "routed_wells": int(np.sum(np.any(route > 1e-10, axis=1))),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray(FAMILY),
        well_ids=well_ids,
        folds=folds.astype(np.int8),
        row_starts=starts,
        baseline=baseline,
        clean_C016=clean.astype(np.float32),
        prediction=prediction.astype(np.float32),
        outer_base_weights=outer_bases.astype(np.float32),
        outer_route_weights=outer_routes,
        hidden_truth_stored=np.asarray(False),
        protected_metric_computed=np.asarray(False),
        F4_metric_recomputed=np.asarray(False),
    )
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "uniform_nested_outer_predictions_frozen_without_metric",
        "outer_roles": outer_records,
        "prediction": str(args.output),
        "prediction_sha256": router.sha256(args.output),
        "physics_prediction_sha256": router.sha256(args.physics_prediction),
        "physics_selector_sha256": router.sha256(args.physics_selector),
        "source_sha256": router.sha256(Path(__file__)),
        "hidden_truth_stored": False,
        "protected_metric_computed": False,
        "F4_metric_recomputed": False,
        "kaggle_submission_authorized": False,
    }
    args.manifest_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "prediction_sha256": payload["prediction_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
