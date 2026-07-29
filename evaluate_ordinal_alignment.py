from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def pooled_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def load_ordinal_prediction(
    caches: list[Path],
    well_ids: np.ndarray,
    row_starts: np.ndarray,
    folds: np.ndarray,
    prediction_key: str,
) -> np.ndarray:
    """Reassemble five independently held-out fold caches in layout order."""
    position = {well_id: index for index, well_id in enumerate(well_ids)}
    prediction = np.empty(row_starts[-1], dtype=np.float64)
    filled = np.zeros(len(well_ids), dtype=bool)
    for expected_fold, path in enumerate(caches):
        with np.load(path) as cache:
            cache_ids = cache["well_ids"].astype(str)
            cache_starts = cache["row_starts"].astype(np.int64)
            key = prediction_key if prediction_key in cache else "prediction"
            cache_prediction = cache[key].astype(np.float64)
        for cache_index, well_id in enumerate(cache_ids):
            layout_index = position[well_id]
            if folds[layout_index] != expected_fold:
                raise ValueError(
                    f"{well_id}: expected fold {expected_fold}, "
                    f"found {folds[layout_index]}"
                )
            left, right = row_starts[layout_index : layout_index + 2]
            cache_left, cache_right = cache_starts[cache_index : cache_index + 2]
            if right - left != cache_right - cache_left:
                raise ValueError(f"{well_id}: cache and layout row counts differ")
            prediction[left:right] = cache_prediction[cache_left:cache_right]
            filled[layout_index] = True
    if not np.all(filled):
        missing = ", ".join(well_ids[~filled][:5])
        raise ValueError(f"ordinal caches omit wells: {missing}")
    return prediction


def smooth_per_well(
    values: np.ndarray, row_starts: np.ndarray, sigma: float
) -> np.ndarray:
    if sigma <= 0.0:
        return values.copy()
    smoothed = np.empty_like(values)
    for left, right in zip(row_starts[:-1], row_starts[1:]):
        smoothed[left:right] = gaussian_filter1d(
            values[left:right], sigma=sigma, mode="nearest"
        )
    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an ordinal-loss alignment specialist against an OOF ensemble."
    )
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--ordinal-caches", type=Path, nargs=5, required=True)
    parser.add_argument("--prediction-key", default="p0.5")
    parser.add_argument("--weight", type=float, default=0.15)
    parser.add_argument("--smooth-sigma", type=float, default=32.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.layout_cache) as cache:
        well_ids = cache["well_ids"].astype(str)
        row_starts = cache["row_starts"].astype(np.int64)
        truth = cache["truth"].astype(np.float64)
        folds = cache["original"].astype(np.int64)
    with np.load(args.baseline_cache) as cache:
        baseline_ids = cache["well_ids"].astype(str)
        baseline_starts = cache["row_starts"].astype(np.int64)
        baseline = cache["prediction"].astype(np.float64)
    if not np.array_equal(well_ids, baseline_ids):
        raise ValueError("layout and baseline well order differs")
    if not np.array_equal(row_starts, baseline_starts):
        raise ValueError("layout and baseline row starts differ")

    ordinal = load_ordinal_prediction(
        args.ordinal_caches,
        well_ids,
        row_starts,
        folds,
        args.prediction_key,
    )
    correction = smooth_per_well(ordinal - baseline, row_starts, args.smooth_sigma)
    final = baseline + args.weight * correction
    row_folds = np.repeat(folds, np.diff(row_starts))
    print(
        f"pooled baseline={pooled_rmse(baseline, truth):.6f} "
        f"ordinal={pooled_rmse(ordinal, truth):.6f} "
        f"final={pooled_rmse(final, truth):.6f}"
    )
    for fold in sorted(np.unique(folds)):
        mask = row_folds == fold
        error = baseline[mask] - truth[mask]
        fold_correction = correction[mask]
        optimum = -float(error @ fold_correction) / max(
            float(fold_correction @ fold_correction), 1e-12
        )
        print(
            f"fold={fold} baseline={pooled_rmse(baseline[mask], truth[mask]):.6f} "
            f"ordinal={pooled_rmse(ordinal[mask], truth[mask]):.6f} "
            f"final={pooled_rmse(final[mask], truth[mask]):.6f} "
            f"weight_optimum={optimum:.5f}"
        )

    candidate_weights = np.arange(0.0, 0.3001, 0.025)
    nested = np.empty_like(baseline)
    selected_weights: list[float] = []
    for held_out in sorted(np.unique(folds)):
        train = row_folds != held_out
        valid = ~train
        selected = min(
            candidate_weights,
            key=lambda weight: np.mean(
                np.square(baseline[train] + weight * correction[train] - truth[train])
            ),
        )
        nested[valid] = baseline[valid] + selected * correction[valid]
        selected_weights.append(float(selected))
    print(
        "nested "
        f"weights={','.join(f'{weight:.3f}' for weight in selected_weights)} "
        f"rmse={pooled_rmse(nested, truth):.6f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            well_ids=well_ids,
            row_starts=row_starts,
            prediction=final.astype(np.float32),
            ordinal_prediction=ordinal.astype(np.float32),
            correction=correction.astype(np.float32),
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
