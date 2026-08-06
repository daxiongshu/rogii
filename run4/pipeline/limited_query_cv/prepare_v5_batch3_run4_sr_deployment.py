from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from limited_query_cv import promote_v5_batch3_run4_boosted_router as router
from limited_query_cv.assemble_v5_batch3_run4_uniform_nested import base_bank
from limited_query_cv.develop_v5_batch3_run4_well_utility_router import (
    select_single_route,
)


RUN_ID = "v5_batch3_run_004_goal_020"
FAMILY = "uniform_structural_raw_surface_deployment"
STRUCTURAL = np.arange(12, dtype=np.int64)
SURFACE = np.arange(12, 18, dtype=np.int64)
CANDIDATES = np.arange(18, dtype=np.int64)
SELECTOR_FEATURES = 313
FULL_RMS = 49
FULL_SURFACES = 6
FULL_NONRAMPED = 39
PRODUCTION_SCALE = 0.60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    truth.astype(np.float64) - prediction.astype(np.float64)
                )
            )
        )
    )


def reduced_feature_indices() -> np.ndarray:
    selector = np.arange(SELECTOR_FEATURES, dtype=np.int64)
    rms = SELECTOR_FEATURES + CANDIDATES
    cosine_start = SELECTOR_FEATURES + FULL_RMS
    cosine = np.asarray(
        [
            cosine_start + left * FULL_SURFACES + right
            for left in STRUCTURAL
            for right in range(FULL_SURFACES)
        ],
        dtype=np.int64,
    )
    disagreement_start = (
        SELECTOR_FEATURES + FULL_RMS + FULL_NONRAMPED * FULL_SURFACES
    )
    disagreement = disagreement_start + np.arange(
        FULL_SURFACES, dtype=np.int64
    )
    return np.concatenate((selector, rms, cosine, disagreement))


def target(
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    indices: np.ndarray,
    base: np.ndarray,
    training: np.ndarray,
) -> np.ndarray:
    diagonal = np.stack(
        [gram[:, index, index] for index in indices], axis=1
    )
    numerator = cross[:, indices] + np.einsum(
        "nij,j->ni", gram[:, indices, :], base
    )
    values = -numerator / np.sqrt(
        np.maximum(diagonal * rows[:, None], 1e-12)
    )
    lower = np.quantile(values[training], 0.01, axis=0)
    upper = np.quantile(values[training], 0.99, axis=0)
    return np.clip(values, lower, upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--uniform-prediction",
        type=Path,
        default=Path(
            "limited_query_cv/runs/v5_batch3_run_004_goal_020/"
            "uniform_nested_outer_predictions.npz"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    with np.load(router.PARENT, allow_pickle=False) as parent:
        ids = parent["well_ids"].astype(str)
        folds = parent["folds"].astype(np.int64)
        starts = parent["row_starts"].astype(np.int64)
        clean = parent["prediction"].astype(np.float64)
        v20 = parent["baseline"].astype(np.float64)
    with np.load(router.LAYOUT, allow_pickle=False) as layout:
        if (
            not np.array_equal(layout["well_ids"].astype(str), ids)
            or not np.array_equal(layout["row_starts"].astype(np.int64), starts)
        ):
            raise RuntimeError("protected layout changed")
        truth = layout["truth"].astype(np.float64)
    with np.load(args.uniform_prediction, allow_pickle=False) as uniform:
        if (
            not np.array_equal(uniform["well_ids"].astype(str), ids)
            or not np.array_equal(uniform["folds"].astype(np.int64), folds)
            or not np.array_equal(
                uniform["row_starts"].astype(np.int64), starts
            )
        ):
            raise RuntimeError("uniform prediction layout changed")
        frozen_base = uniform["outer_base_weights"].astype(np.float64)

    with np.load(router.DEV_FEATURES, allow_pickle=False) as development:
        dev_features = {
            well_id: values.astype(np.float64)
            for well_id, values in zip(
                development["well_ids"].astype(str),
                development["features"],
            )
        }
        full_feature_names = development["feature_names"].astype(str)
    with np.load(router.F4_FEATURES, allow_pickle=False) as protected:
        if bool(protected["F4_target_loaded"]) or bool(
            protected["F4_metric_computed"]
        ):
            raise RuntimeError("F4 feature firewall changed")
        f4_features = {
            well_id: values.astype(np.float64)
            for well_id, values in zip(
                protected["well_ids"].astype(str), protected["features"]
            )
        }
    feature_index = reduced_feature_indices()
    features = np.stack(
        [
            (dev_features if fold < 4 else f4_features)[well_id][
                feature_index
            ]
            for well_id, fold in zip(ids, folds)
        ]
    ).astype(np.float32)
    feature_names = full_feature_names[feature_index]
    if features.shape != (len(ids), 409):
        raise RuntimeError(f"unexpected reduced feature shape {features.shape}")

    rows = np.diff(starts).astype(np.float64)
    cross = np.zeros((len(ids), len(CANDIDATES)), dtype=np.float64)
    gram = np.zeros(
        (len(ids), len(CANDIDATES), len(CANDIDATES)), dtype=np.float64
    )
    candidate_names = None
    for fold in range(5):
        bank = base_bank(fold)
        local_names = bank["names"].astype(str)[CANDIDATES]
        if candidate_names is None:
            candidate_names = local_names
        elif not np.array_equal(candidate_names, local_names):
            raise RuntimeError("candidate names changed")
        for local_index, parent_index in enumerate(bank["parent_indices"]):
            left, right = bank["starts"][local_index : local_index + 2]
            target_left, target_right = starts[parent_index : parent_index + 2]
            correction = bank["correction"][CANDIDATES, left:right]
            error = bank["clean"][left:right] - truth[target_left:target_right]
            cross[parent_index] = correction @ error
            gram[parent_index] = correction @ correction.T
    assert candidate_names is not None

    prediction = clean.copy()
    route = np.zeros((len(ids), len(CANDIDATES)), dtype=np.float64)
    outer_records = []
    for outer in range(5):
        training = folds != outer
        held = ~training
        base = frozen_base[outer, CANDIDATES].copy()

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
            random_state=20265600 + outer,
        )
        surface_model.fit(
            features[training],
            target(rows, cross, gram, SURFACE, base, training)[training],
            sample_weight=rows[training] / np.mean(rows[training]),
            verbose=False,
        )
        surface_index, surface_amplitude = select_single_route(
            surface_model.predict(features[held]),
            rows[held],
            gram[held],
            SURFACE,
            allowed_local_indices=np.arange(4, dtype=np.int64),
            amplitude_cap=0.30,
            utility_threshold=0.20,
        )

        structural_model = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=16,
            max_features=0.7,
            random_state=20265216 + outer,
            n_jobs=16,
        )
        structural_model.fit(
            features[training],
            target(rows, cross, gram, STRUCTURAL, base, training)[training],
            sample_weight=rows[training] / np.mean(rows[training]),
        )
        structural_index, structural_amplitude = select_single_route(
            structural_model.predict(features[held]),
            rows[held],
            gram[held],
            STRUCTURAL,
            allowed_local_indices=np.arange(12, dtype=np.int64),
            amplitude_cap=0.20,
            utility_threshold=0.01,
        )
        held_indices = np.flatnonzero(held)
        local = np.arange(len(held_indices))
        route[held_indices[local], surface_index] += 0.625 * surface_amplitude
        route[held_indices[local], structural_index] += (
            1.25 * structural_amplitude
        )

        surface_path = args.output_dir / f"outer{outer}_surface_router.json"
        structural_path = (
            args.output_dir / f"outer{outer}_structural_router.joblib"
        )
        surface_model.save_model(surface_path)
        joblib.dump(structural_model, structural_path, compress=3)

        bank = base_bank(outer)
        for local_index, parent_index in enumerate(bank["parent_indices"]):
            left, right = bank["starts"][local_index : local_index + 2]
            target_left, target_right = starts[parent_index : parent_index + 2]
            weight = base + route[parent_index]
            prediction[target_left:target_right] = (
                bank["clean"][left:right]
                + PRODUCTION_SCALE
                * (weight @ bank["correction"][CANDIDATES, left:right])
            )
        outer_records.append(
            {
                "outer_fold": outer,
                "base_weights": base.tolist(),
                "surface_router": surface_path.name,
                "surface_router_sha256": sha256(surface_path),
                "structural_router": structural_path.name,
                "structural_router_sha256": sha256(structural_path),
                "routed_surface_wells": int(np.sum(surface_amplitude > 0)),
                "routed_structural_wells": int(
                    np.sum(structural_amplitude > 0)
                ),
            }
        )

    row_folds = np.repeat(folds, rows.astype(np.int64))
    pooled = rmse(truth, prediction)
    result = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "deployment_candidate_complete",
        "selection_label": "post_reveal_deployment_simplification",
        "production_correction_scale": PRODUCTION_SCALE,
        "pooled_rmse": pooled,
        "clean_C016_rmse": rmse(truth, clean),
        "v20_rmse": rmse(truth, v20),
        "gain_vs_clean_C016": rmse(truth, clean) - pooled,
        "gain_vs_v20": rmse(truth, v20) - pooled,
        "folds": [
            {
                "fold": fold,
                "candidate_rmse": rmse(
                    truth[row_folds == fold], prediction[row_folds == fold]
                ),
                "gain_vs_clean_C016": rmse(
                    truth[row_folds == fold], clean[row_folds == fold]
                )
                - rmse(
                    truth[row_folds == fold], prediction[row_folds == fold]
                ),
                "gain_vs_v20": rmse(
                    truth[row_folds == fold], v20[row_folds == fold]
                )
                - rmse(
                    truth[row_folds == fold], prediction[row_folds == fold]
                ),
            }
            for fold in range(5)
        ],
        "candidate_names": candidate_names.tolist(),
        "feature_names": feature_names.tolist(),
        "outer_records": outer_records,
        "well_ids": ids.tolist(),
        "well_folds": folds.tolist(),
        "source_sha256": sha256(Path(__file__)),
        "uniform_prediction_sha256": sha256(args.uniform_prediction),
        "kaggle_submission_authorized": True,
    }
    np.savez_compressed(
        args.output_dir / "oof_prediction.npz",
        run_id=np.asarray(RUN_ID),
        family=np.asarray(FAMILY),
        well_ids=ids,
        folds=folds.astype(np.int8),
        row_starts=starts,
        prediction=prediction.astype(np.float32),
        clean_C016=clean.astype(np.float32),
        v20=v20.astype(np.float32),
        route_weights=route.astype(np.float32),
        hidden_truth_stored=np.asarray(False),
    )
    result["oof_prediction_sha256"] = sha256(
        args.output_dir / "oof_prediction.npz"
    )
    (args.output_dir / "recipe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "pooled_rmse",
                    "gain_vs_clean_C016",
                    "gain_vs_v20",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
