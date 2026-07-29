from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def assemble_correction(
    caches: list[Path],
    well_ids: np.ndarray,
    row_starts: np.ndarray,
    folds: np.ndarray,
    average_key: str,
) -> np.ndarray:
    position = {well_id: index for index, well_id in enumerate(well_ids)}
    correction = np.empty(row_starts[-1], dtype=np.float64)
    filled = np.zeros(len(well_ids), dtype=bool)
    for expected_fold, path in enumerate(caches):
        with np.load(path) as cache:
            cache_ids = cache["well_ids"].astype(str)
            cache_starts = cache["row_starts"].astype(np.int64)
            average = cache[average_key].astype(np.float64)
            normal = cache["prediction_normal"].astype(np.float64)
        for cache_index, well_id in enumerate(cache_ids):
            layout_index = position[well_id]
            if folds[layout_index] != expected_fold:
                raise ValueError(f"{well_id}: cache is in the wrong fold")
            left, right = row_starts[layout_index : layout_index + 2]
            cache_left, cache_right = cache_starts[cache_index : cache_index + 2]
            if right - left != cache_right - cache_left:
                raise ValueError(f"{well_id}: cache and layout row counts differ")
            correction[left:right] = (
                average[cache_left:cache_right] - normal[cache_left:cache_right]
            )
            filled[layout_index] = True
    if not np.all(filled):
        raise ValueError("mask caches do not cover every well")
    return correction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit deterministic missing-GR sensitivity extrapolation."
    )
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--mask-caches", type=Path, nargs=5, required=True)
    parser.add_argument("--average-key", default="prediction_avg4")
    parser.add_argument("--weight", type=float, default=-0.05)
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
    with np.load(args.baseline_cache) as cache:
        baseline_ids = cache["well_ids"].astype(str)
        baseline_starts = cache["row_starts"].astype(np.int64)
        baseline = cache["prediction"].astype(np.float64)
    if not np.array_equal(well_ids, baseline_ids):
        raise ValueError("layout and baseline well order differs")
    if not np.array_equal(row_starts, baseline_starts):
        raise ValueError("layout and baseline row starts differ")

    correction = assemble_correction(
        args.mask_caches,
        well_ids,
        row_starts,
        groupings["original"],
        args.average_key,
    )
    final = baseline + args.weight * correction
    print(
        f"baseline={rmse(baseline, truth):.9f} "
        f"candidate={rmse(final, truth):.9f} weight={args.weight:.4f}"
    )
    for name, well_folds in groupings.items():
        row_folds = np.repeat(well_folds, np.diff(row_starts))
        gains = []
        selected = []
        nested = np.empty_like(baseline)
        for fold in range(5):
            train = row_folds != fold
            valid = ~train
            baseline_train_sse = np.sum(np.square(baseline[train] - truth[train]))
            candidate_train_sse = np.sum(np.square(final[train] - truth[train]))
            weight = args.weight if candidate_train_sse < baseline_train_sse else 0.0
            nested[valid] = baseline[valid] + weight * correction[valid]
            selected.append(weight)
            gains.append(
                rmse(baseline[valid], truth[valid])
                - rmse(final[valid], truth[valid])
            )
        print(
            f"{name}: fixed_fold_gains={np.round(gains, 6).tolist()} "
            f"binary_weights={selected} nested={rmse(nested, truth):.9f}"
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
