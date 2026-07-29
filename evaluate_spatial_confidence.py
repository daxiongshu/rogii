from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well


def pooled_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def row_mask(
    well_indices: np.ndarray, row_starts: np.ndarray, row_count: int
) -> np.ndarray:
    mask = np.zeros(row_count, dtype=bool)
    for index in well_indices:
        mask[row_starts[index] : row_starts[index + 1]] = True
    return mask


def cross_fitted_confidence(
    features: np.ndarray,
    family: np.ndarray,
    target: np.ndarray,
    correction_energy: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict a safe per-well multiplier for a fold-safe spatial correction."""
    categorical_family = family.astype(str)
    design = np.column_stack((features, categorical_family))
    cat_feature = features.shape[1]
    clipped_target = np.clip(target, -2.0, 4.0)
    energy_cap = float(np.quantile(correction_energy, 0.95))
    sample_weight = np.minimum(correction_energy, energy_cap)
    prediction = np.empty(len(folds), dtype=np.float64)

    for held_out in sorted(np.unique(folds)):
        train = folds != held_out
        valid = folds == held_out
        model = CatBoostRegressor(
            iterations=500,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=100.0,
            loss_function="RMSE",
            random_seed=seed + int(held_out),
            allow_writing_files=False,
            thread_count=16,
            verbose=False,
        )
        model.fit(
            design[train],
            clipped_target[train],
            sample_weight=sample_weight[train],
            cat_features=[cat_feature],
        )
        prediction[valid] = model.predict(design[valid])
    return np.clip(prediction, 0.0, 2.0)


def visible_row_median_path(well: Well, hidden_rows: int) -> np.ndarray:
    """Continue the median surface increment from the final eight visible rows."""
    anchor = well.anchor_index
    left = max(0, anchor - 7)
    visible_surface = (
        well.tvt_input[left : anchor + 1] + well.z[left : anchor + 1]
    )
    increment = float(np.median(np.diff(visible_surface)))
    hidden_surface = visible_surface[-1] + increment * np.arange(
        1, hidden_rows + 1, dtype=np.float64
    )
    return hidden_surface - well.z[anchor + 1 : anchor + 1 + hidden_rows]


def load_wells(ids: np.ndarray, data_root: Path, workers: int) -> list[Well]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(str(well_id), data_root), ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the cross-fitted spatial-confidence addition to v15."
    )
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--spatial-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1020)
    parser.add_argument("--geometry-weight", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.baseline_cache) as baseline_cache:
        ids = baseline_cache["well_ids"].astype(str)
        row_starts = baseline_cache["row_starts"].astype(np.int64)
        baseline = baseline_cache["prediction"].astype(np.float64)
        truth = baseline_cache["truth"].astype(np.float64)
        original_folds = baseline_cache["original"].astype(np.int64)

    with np.load(args.spatial_cache) as spatial_cache:
        spatial_ids = spatial_cache["ids"].astype(str)
        spatial_starts = spatial_cache["starts"].astype(np.int64)
        features = spatial_cache["X"].astype(np.float64)
        family = spatial_cache["family"].astype(np.int64)
        multiplier_target = spatial_cache["target"].astype(np.float64)
        correction_energy = spatial_cache["weight"].astype(np.float64)
        spatial_delta = spatial_cache["delta"].astype(np.float64)
        spatial_folds = spatial_cache["fold"].astype(np.int64)

    if not np.array_equal(ids, spatial_ids):
        raise ValueError("baseline and spatial cache well order differs")
    if not np.array_equal(row_starts, spatial_starts):
        raise ValueError("baseline and spatial cache row layout differs")
    if not np.array_equal(original_folds, spatial_folds):
        raise ValueError("baseline and spatial cache fold assignment differs")
    if len(baseline) != len(spatial_delta):
        raise ValueError("baseline and spatial correction row count differs")

    multiplier = cross_fitted_confidence(
        features,
        family,
        multiplier_target,
        correction_energy,
        original_folds,
        args.seed,
    )
    row_multiplier = np.repeat(multiplier, np.diff(row_starts))
    spatial_prediction = baseline + row_multiplier * spatial_delta

    wells = load_wells(ids, args.data_root, args.workers)
    geometric_prediction = np.empty_like(baseline)
    for index, well in enumerate(wells):
        left, right = row_starts[index : index + 2]
        geometric_prediction[left:right] = visible_row_median_path(
            well, int(right - left)
        )
    geometry_delta = geometric_prediction - baseline
    final_prediction = spatial_prediction + args.geometry_weight * geometry_delta

    print(
        f"pooled baseline={pooled_rmse(baseline, truth):.6f} "
        f"spatial={pooled_rmse(spatial_prediction, truth):.6f} "
        f"final={pooled_rmse(final_prediction, truth):.6f}"
    )
    for fold in sorted(np.unique(original_folds)):
        mask = row_mask(
            np.flatnonzero(original_folds == fold), row_starts, len(baseline)
        )
        spatial_error = spatial_prediction[mask] - truth[mask]
        fold_geometry = geometry_delta[mask]
        optimal_weight = -float(spatial_error @ fold_geometry) / max(
            float(fold_geometry @ fold_geometry), 1e-12
        )
        print(
            f"fold={fold} baseline={pooled_rmse(baseline[mask], truth[mask]):.6f} "
            f"spatial={pooled_rmse(spatial_prediction[mask], truth[mask]):.6f} "
            f"final={pooled_rmse(final_prediction[mask], truth[mask]):.6f} "
            f"geometry_optimum={optimal_weight:.5f}"
        )

    nested_prediction = np.empty_like(baseline)
    nested_weights: list[float] = []
    spatial_error = spatial_prediction - truth
    for held_out in sorted(np.unique(original_folds)):
        train_mask = row_mask(
            np.flatnonzero(original_folds != held_out), row_starts, len(baseline)
        )
        valid_mask = ~train_mask
        selected_weight = -float(
            spatial_error[train_mask] @ geometry_delta[train_mask]
        ) / max(float(geometry_delta[train_mask] @ geometry_delta[train_mask]), 1e-12)
        selected_weight = float(np.clip(selected_weight, 0.0, 0.02))
        nested_weights.append(selected_weight)
        nested_prediction[valid_mask] = (
            spatial_prediction[valid_mask]
            + selected_weight * geometry_delta[valid_mask]
        )
    print(
        "nested_geometry "
        f"weights={','.join(f'{weight:.5f}' for weight in nested_weights)} "
        f"rmse={pooled_rmse(nested_prediction, truth):.6f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            well_ids=ids,
            row_starts=row_starts,
            prediction=final_prediction.astype(np.float32),
            spatial_prediction=spatial_prediction.astype(np.float32),
            confidence=multiplier.astype(np.float32),
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
