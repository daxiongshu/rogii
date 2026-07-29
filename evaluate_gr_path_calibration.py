from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well


def pooled_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(prediction.astype(np.float64) - truth.astype(np.float64))
            )
        )
    )


def load_wells(ids: np.ndarray, data_root: Path, workers: int) -> list[Well]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(str(well_id), data_root), ids))


def smooth_gr(values: np.ndarray, requested_window: int) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values, dtype=np.float64)
    rows = np.arange(len(values))
    filled = np.interp(rows, rows[finite], values[finite])
    window = min(
        requested_window,
        len(filled) if len(filled) % 2 else len(filled) - 1,
    )
    if window % 2 == 0:
        window -= 1
    return savgol_filter(filled, window, 2) if window >= 5 else filled


def gr_path_correction(
    well: Well,
    prediction: np.ndarray,
    *,
    bias_radius: float,
    slope_radius: float,
    grid_step: float,
    sample_stride: int,
    smooth_window: int,
    prior: float,
) -> tuple[np.ndarray, float, float]:
    """Select a constant/linear path shift using only inference-time GR."""
    tail = well.tail_indices
    if len(tail) != len(prediction):
        raise ValueError(f"{well.well_id}: prediction and hidden tail differ")
    sampled = np.unique(
        np.r_[np.arange(0, len(tail), sample_stride), len(tail) - 1]
    )
    observed = smooth_gr(well.gr[tail], smooth_window)[sampled]
    observed = (observed - np.mean(observed)) / max(float(np.std(observed)), 3.0)
    progress = np.linspace(-1.0, 1.0, len(tail))[sampled]

    biases = np.arange(-bias_radius, bias_radius + grid_step / 2, grid_step)
    slopes = np.arange(-slope_radius, slope_radius + grid_step / 2, grid_step)
    bias_grid, slope_grid = np.meshgrid(biases, slopes, indexing="ij")
    candidate_bias = bias_grid.ravel()
    candidate_slope = slope_grid.ravel()
    candidate_tvt = (
        prediction[sampled, None]
        + candidate_bias[None, :]
        + progress[:, None] * candidate_slope[None, :]
    )
    reference = np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    supported = np.isfinite(reference)
    count = np.sum(supported, axis=0)
    mean = np.nansum(reference, axis=0) / np.maximum(count, 1)
    centered = np.where(supported, reference - mean, 0.0)
    denominator = np.sqrt(
        np.sum(np.square(centered), axis=0) * np.sum(np.square(observed))
    )
    correlation = np.sum(centered * observed[:, None], axis=0) / np.maximum(
        denominator, 1e-9
    )
    correlation[count < 0.8 * len(sampled)] = -2.0
    utility = correlation - prior * (
        np.square(candidate_bias) + np.square(candidate_slope)
    )
    best = int(np.argmax(utility))
    bias = float(candidate_bias[best])
    slope = float(candidate_slope[best])
    correction = bias + slope * np.linspace(-1.0, 1.0, len(tail))
    return correction, bias, slope


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate ensemble paths by legal hidden-GR/typewell agreement."
    )
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--bias-radius", type=float, default=12.0)
    parser.add_argument("--slope-radius", type=float, default=18.0)
    parser.add_argument("--grid-step", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--smooth-window", type=int, default=51)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--weight", type=float, default=0.45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.layout_cache) as cache:
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        truth = cache["truth"].astype(np.float64)
        folds = cache["original"].astype(np.int64)
    with np.load(args.prediction_cache) as cache:
        prediction_ids = cache["well_ids"].astype(str)
        prediction_starts = cache["row_starts"].astype(np.int64)
        prediction = cache["prediction"].astype(np.float64)
    if not np.array_equal(ids, prediction_ids):
        raise ValueError("layout and prediction well order differs")
    if not np.array_equal(starts, prediction_starts):
        raise ValueError("layout and prediction row starts differ")

    wells = load_wells(ids, args.data_root, args.workers)
    correction = np.empty_like(prediction)
    coefficients = np.empty((len(ids), 2), dtype=np.float64)
    for index, well in enumerate(wells):
        left, right = starts[index : index + 2]
        well_correction, bias, slope = gr_path_correction(
            well,
            prediction[left:right],
            bias_radius=args.bias_radius,
            slope_radius=args.slope_radius,
            grid_step=args.grid_step,
            sample_stride=args.sample_stride,
            smooth_window=args.smooth_window,
            prior=args.prior,
        )
        correction[left:right] = well_correction
        coefficients[index] = (bias, slope)

    calibrated = prediction + args.weight * correction
    print(
        f"pooled baseline={pooled_rmse(prediction, truth):.6f} "
        f"calibrated={pooled_rmse(calibrated, truth):.6f}"
    )
    row_folds = np.repeat(folds, np.diff(starts))
    for fold in sorted(np.unique(folds)):
        mask = row_folds == fold
        print(
            f"fold={fold} baseline={pooled_rmse(prediction[mask], truth[mask]):.6f} "
            f"calibrated={pooled_rmse(calibrated[mask], truth[mask]):.6f}"
        )

    candidate_weights = np.arange(0.0, 0.5001, 0.05)
    nested = np.empty_like(prediction)
    selected_weights = []
    for held_out in sorted(np.unique(folds)):
        train = row_folds != held_out
        valid = ~train
        selected = min(
            candidate_weights,
            key=lambda weight: np.mean(
                np.square(
                    prediction[train]
                    + weight * correction[train]
                    - truth[train]
                )
            ),
        )
        nested[valid] = prediction[valid] + selected * correction[valid]
        selected_weights.append(float(selected))
    print(
        "nested "
        f"weights={','.join(f'{weight:.2f}' for weight in selected_weights)} "
        f"rmse={pooled_rmse(nested, truth):.6f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            well_ids=ids,
            row_starts=starts,
            prediction=calibrated.astype(np.float32),
            correction=correction.astype(np.float32),
            coefficients=coefficients.astype(np.float32),
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
