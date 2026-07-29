from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def assemble_difference(
    candidate_caches: list[Path],
    component_caches: list[Path],
    well_ids: np.ndarray,
    row_starts: np.ndarray,
    folds: np.ndarray,
    candidate_key: str,
    component_name: str,
    reference_key: str | None,
) -> np.ndarray:
    position = {well_id: index for index, well_id in enumerate(well_ids)}
    correction = np.empty(row_starts[-1], dtype=np.float64)
    filled = np.zeros(len(well_ids), dtype=bool)
    for expected_fold, (candidate_path, component_path) in enumerate(
        zip(candidate_caches, component_caches)
    ):
        with np.load(candidate_path) as candidate:
            candidate_ids = candidate["well_ids"].astype(str)
            candidate_starts = candidate["row_starts"].astype(np.int64)
            candidate_prediction = candidate[candidate_key].astype(np.float64)
        with np.load(component_path, allow_pickle=True) as component:
            component_ids = component["well_ids"].astype(str)
            component_starts = component["row_starts"].astype(np.int64)
            if reference_key is not None:
                original_prediction = component[reference_key].astype(np.float64)
            else:
                names = component["component_names"].astype(str)
                selected = np.flatnonzero(names == component_name)
                if len(selected) != 1:
                    raise ValueError(f"{component_path}: missing {component_name}")
                original_prediction = component["components"][selected[0]].astype(
                    np.float64
                )
        if not np.array_equal(candidate_ids, component_ids):
            raise ValueError("candidate and component cache well order differs")
        if not np.array_equal(candidate_starts, component_starts):
            raise ValueError("candidate and component cache row starts differ")
        for cache_index, well_id in enumerate(candidate_ids):
            layout_index = position[well_id]
            if folds[layout_index] != expected_fold:
                raise ValueError(f"{well_id}: cache is in the wrong fold")
            left, right = row_starts[layout_index : layout_index + 2]
            cache_left, cache_right = candidate_starts[cache_index : cache_index + 2]
            if right - left != cache_right - cache_left:
                raise ValueError(f"{well_id}: cache and layout row counts differ")
            correction[left:right] = (
                candidate_prediction[cache_left:cache_right]
                - original_prediction[cache_left:cache_right]
            )
            filled[layout_index] = True
    if not np.all(filled):
        raise ValueError("candidate caches do not cover every well")
    return correction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a wider alignment model as a residual ensemble direction."
    )
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--candidate-caches", type=Path, nargs=5, required=True)
    parser.add_argument("--component-caches", type=Path, nargs=5, required=True)
    parser.add_argument("--candidate-key", default="prediction_t1")
    parser.add_argument("--component-name", default="wide_supervised")
    parser.add_argument(
        "--reference-key",
        help="Use this array directly from each component cache as the reference.",
    )
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.layout_cache) as layout:
        well_ids = layout["well_ids"].astype(str)
        row_starts = layout["row_starts"].astype(np.int64)
        truth = layout["truth"].astype(np.float64)
        groupings = {
            name: layout[name].astype(np.int64)
            for name in ("original", "stratified", "independent")
        }
    with np.load(args.baseline_cache) as baseline_cache:
        if not np.array_equal(well_ids, baseline_cache["well_ids"].astype(str)):
            raise ValueError("layout and baseline well order differs")
        if not np.array_equal(
            row_starts, baseline_cache["row_starts"].astype(np.int64)
        ):
            raise ValueError("layout and baseline row starts differ")
        baseline = baseline_cache["prediction"].astype(np.float64)

    correction = assemble_difference(
        args.candidate_caches,
        args.component_caches,
        well_ids,
        row_starts,
        groupings["original"],
        args.candidate_key,
        args.component_name,
        args.reference_key,
    )
    final = baseline + args.weight * correction
    print(
        f"baseline={rmse(baseline, truth):.9f} "
        f"candidate={rmse(final, truth):.9f} weight={args.weight:.4f}"
    )
    grid = np.arange(0.0, 0.2001, 0.025)
    for name, well_folds in groupings.items():
        row_folds = np.repeat(well_folds, np.diff(row_starts))
        gains = []
        continuous = np.empty_like(baseline)
        binary = np.empty_like(baseline)
        continuous_weights = []
        binary_weights = []
        for fold in range(5):
            train = row_folds != fold
            valid = ~train
            error = baseline[train] - truth[train]
            train_correction = correction[train]
            sse = (
                error @ error
                + 2.0 * grid * (error @ train_correction)
                + np.square(grid) * (train_correction @ train_correction)
            )
            weight = float(grid[int(np.argmin(sse))])
            fixed_error = error + args.weight * train_correction
            binary_weight = (
                args.weight if fixed_error @ fixed_error < error @ error else 0.0
            )
            continuous[valid] = baseline[valid] + weight * correction[valid]
            binary[valid] = baseline[valid] + binary_weight * correction[valid]
            continuous_weights.append(weight)
            binary_weights.append(binary_weight)
            gains.append(
                rmse(baseline[valid], truth[valid])
                - rmse(final[valid], truth[valid])
            )
        print(
            f"{name}: fixed_fold_gains={np.round(gains, 6).tolist()} "
            f"continuous={rmse(continuous, truth):.9f} "
            f"continuous_weights={continuous_weights} "
            f"binary={rmse(binary, truth):.9f} binary_weights={binary_weights}"
        )

    if args.output is not None:
        np.savez_compressed(
            args.output,
            well_ids=well_ids,
            row_starts=row_starts,
            correction=correction.astype(np.float32),
            prediction=final.astype(np.float32),
            weight=args.weight,
        )


if __name__ == "__main__":
    main()
