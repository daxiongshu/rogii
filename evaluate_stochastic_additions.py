from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluate_gr_smoothing_tta import expanded_and_smoothed
from evaluate_surface_extrapolation import surface_extrapolation
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


WIDE_SUPERVISED_INDEX = 8
WIDE_SUPERVISED_WEIGHT = 0.325
FIXED_WEIGHT = 0.0537244898
WIDE_WEIGHT = 0.0716326531


def load_all(data_root: Path, workers: int) -> dict[str, Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
    return dict(zip(well_ids, wells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-root", type=Path, default=Path("cache/v6_oof"))
    parser.add_argument("--tta-root", type=Path, default=Path("cache/v6_tta"))
    parser.add_argument(
        "--candidate-root", type=Path, default=Path("cache/seed_ablation")
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--weights", type=float, nargs="+", default=(0.0, 0.05, 0.1, 0.2)
    )
    args = parser.parse_args()

    wells = load_all(args.data_root, args.workers)
    candidates = [
        (seed_weight, small_weight)
        for seed_weight in args.weights
        for small_weight in args.weights
        if seed_weight + small_weight <= 1.0
    ]
    fold_sse = np.zeros((5, len(candidates)), dtype=np.float64)
    fold_rows = np.zeros(5, dtype=np.int64)
    for fold in range(5):
        with np.load(args.base_root / f"fold{fold}.npz") as base_cache:
            well_ids = base_cache["well_ids"]
            starts = base_cache["row_starts"]
            components = base_cache["components"].astype(np.float64)
            raw = base_cache["coarse"].astype(np.float64)
            truth = base_cache["truth"].astype(np.float64)
        with np.load(args.tta_root / f"fold{fold}_gr25.npz") as tta_cache:
            tta_components = tta_cache["components"].astype(np.float64)
        with np.load(args.candidate_root / f"seed3_fold{fold}.npz") as cached:
            if not np.array_equal(well_ids, cached["well_ids"]):
                raise ValueError(f"fold {fold}: seed well order differs")
            seed_prediction = cached["prediction"].astype(np.float64)
        with np.load(args.candidate_root / f"base6_fold{fold}.npz") as cached:
            if not np.array_equal(well_ids, cached["well_ids"]):
                raise ValueError(f"fold {fold}: base6 well order differs")
            small_prediction = cached["prediction"].astype(np.float64)

        raw += FIXED_WEIGHT * (tta_components[0] - components[0])
        raw += WIDE_WEIGHT * (tta_components[2] - components[2])
        for index, well_id in enumerate(well_ids):
            well = wells[str(well_id)]
            left, right = int(starts[index]), int(starts[index + 1])
            base_pre = expanded_and_smoothed(well, raw[left:right], 0.05)
            original_pre = expanded_and_smoothed(
                well,
                components[WIDE_SUPERVISED_INDEX, left:right],
                0.05,
            )
            seed_pre = expanded_and_smoothed(
                well, seed_prediction[left:right], 0.05
            )
            small_pre = expanded_and_smoothed(
                well, small_prediction[left:right], 0.05
            )
            for candidate, (seed_weight, small_weight) in enumerate(candidates):
                prediction = base_pre + WIDE_SUPERVISED_WEIGHT * (
                    seed_weight * (seed_pre - original_pre)
                    + small_weight * (small_pre - original_pre)
                )
                prediction += 0.03 * surface_extrapolation(
                    well,
                    prediction,
                    visible_rows=1000,
                    correction_limit=24.0,
                )
                error = prediction - truth[left:right]
                fold_sse[fold, candidate] += float(error @ error)
            fold_rows[fold] += right - left

    total_rows = int(np.sum(fold_rows))
    pooled = np.sqrt(np.sum(fold_sse, axis=0) / total_rows)
    for candidate in np.argsort(pooled):
        print(
            f"fixed seed={candidates[candidate][0]:.3f} "
            f"base6={candidates[candidate][1]:.3f} "
            f"pooled={pooled[candidate]:.6f} folds="
            + ",".join(
                f"{np.sqrt(fold_sse[fold, candidate] / fold_rows[fold]):.6f}"
                for fold in range(5)
            )
        )

    nested_sse = 0.0
    print("Leave-one-fold-out selection:")
    for heldout in range(5):
        tuning_sse = np.sum(np.delete(fold_sse, heldout, axis=0), axis=0)
        tuning_rows = total_rows - int(fold_rows[heldout])
        selected = int(np.argmin(tuning_sse / tuning_rows))
        nested_sse += fold_sse[heldout, selected]
        print(
            f"heldout_fold={heldout} seed={candidates[selected][0]:.3f} "
            f"base6={candidates[selected][1]:.3f} "
            f"heldout_rmse="
            f"{np.sqrt(fold_sse[heldout, selected] / fold_rows[heldout]):.6f}"
        )
    print(f"nested_rmse={np.sqrt(nested_sse / total_rows):.6f}")


if __name__ == "__main__":
    main()
