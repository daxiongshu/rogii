from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.signal import savgol_filter

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well
from rogii.dp import _robust_affine


@dataclass(frozen=True)
class Config:
    stride: int = 12
    radius: int = 24
    step: float = 1.0
    max_jump: int = 2
    transition: float = 0.5
    prior: float = 0.01
    emission_scale: float = 1.0
    emission_cap: float = 9.0
    smooth_window: int = 49


@njit(cache=True, nogil=True)
def viterbi(
    cost: np.ndarray, anchor: int, max_jump: int, transition: float
) -> np.ndarray:
    time_count, state_count = cost.shape
    previous = np.full(state_count, 1e30, dtype=np.float64)
    previous[anchor] = 0.0
    back = np.empty((time_count, state_count), dtype=np.int16)
    back[0] = anchor
    for time in range(1, time_count):
        current = np.empty(state_count, dtype=np.float64)
        for state in range(state_count):
            lower = max(0, state - max_jump)
            upper = min(state_count, state + max_jump + 1)
            best = 1e30
            argument = state
            for old_state in range(lower, upper):
                difference = state - old_state
                value = previous[old_state] + transition * difference * difference
                if value < best:
                    best = value
                    argument = old_state
            current[state] = best + cost[time, state]
            back[time, state] = argument
        previous = current
    path = np.empty(time_count, dtype=np.int16)
    path[-1] = np.argmin(previous)
    for time in range(time_count - 1, 0, -1):
        path[time - 1] = back[time, path[time]]
    path[0] = anchor
    return path


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    positions = np.arange(len(values))
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros(len(values), dtype=np.float64)
    result = np.interp(positions, positions[finite], values[finite])
    valid_window = min(window, len(result) if len(result) % 2 else len(result) - 1)
    return (
        savgol_filter(result, valid_window, 2)
        if valid_window >= 5
        else result
    )


def predict_correction(well: Well, baseline: np.ndarray, config: Config) -> np.ndarray:
    tail = well.tail_indices
    query = np.unique(
        np.r_[well.anchor_index, tail[:: config.stride], tail[-1]]
    ).astype(int)
    baseline_query = np.r_[
        well.tvt_input[well.anchor_index], np.interp(query[1:], tail, baseline)
    ]

    known = well.known_indices
    reference = np.interp(
        well.tvt_input[known],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, offset, residual_scale = _robust_affine(reference, well.gr[known])
    offsets = np.arange(
        -config.radius, config.radius + config.step / 2, config.step
    )
    candidate_tvt = baseline_query[:, None] + offsets[None]
    expected = gain * np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape) + offset
    observed = smooth(well.gr, config.smooth_window)[query, None]
    standardized = (observed - expected) / max(
        residual_scale * config.emission_scale, 1e-6
    )
    cost = np.minimum(np.square(standardized), config.emission_cap)
    cost[~np.isfinite(expected)] = config.emission_cap * 4
    cost += config.prior * np.square(offsets[None])

    # The last visible point is exactly known; its GR cannot move the anchor.
    cost[0] = config.emission_cap * 4
    anchor = int(np.argmin(np.abs(offsets)))
    cost[0, anchor] = 0.0
    path = viterbi(
        cost.astype(np.float64), anchor, config.max_jump, config.transition
    )
    return np.interp(tail, query, offsets[path])


def rmse(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None
) -> float:
    if mask is None:
        mask = np.ones(len(prediction), dtype=bool)
    return float(np.sqrt(np.mean(np.square(prediction[mask] - truth[mask]))))


def nested_audit(
    baseline: np.ndarray,
    correction: np.ndarray,
    truth: np.ndarray,
    row_fold: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, list[float]]:
    nested = np.empty_like(baseline)
    selected = []
    for fold in range(5):
        train = row_fold != fold
        valid = ~train
        errors = [
            np.mean(np.square(baseline[train] + weight * correction[train] - truth[train]))
            for weight in weights
        ]
        weight = float(weights[int(np.argmin(errors))])
        nested[valid] = baseline[valid] + weight * correction[valid]
        selected.append(weight)
    return rmse(nested, truth), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--max-jump", type=int, default=2)
    parser.add_argument("--transition", type=float, default=0.5)
    parser.add_argument("--prior", type=float, default=0.01)
    parser.add_argument("--emission-scale", type=float, default=1.0)
    parser.add_argument("--emission-cap", type=float, default=9.0)
    parser.add_argument("--smooth-window", type=int, default=49)
    parser.add_argument("--fixed-weight", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_cache = np.load(args.baseline_cache, allow_pickle=True)
    layout = np.load(args.layout_cache, allow_pickle=True)
    well_ids = baseline_cache["well_ids"].astype(str)
    row_starts = baseline_cache["row_starts"].astype(int)
    baseline = baseline_cache["prediction"].astype(np.float64)
    truth = layout["truth"].astype(np.float64)
    if not np.array_equal(well_ids, layout["well_ids"].astype(str)):
        raise ValueError("baseline and layout well IDs do not match")
    if not np.array_equal(row_starts, layout["row_starts"].astype(int)):
        raise ValueError("baseline and layout row starts do not match")

    config = Config(
        stride=args.stride,
        radius=args.radius,
        max_jump=args.max_jump,
        transition=args.transition,
        prior=args.prior,
        emission_scale=args.emission_scale,
        emission_cap=args.emission_cap,
        smooth_window=args.smooth_window,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        wells = list(pool.map(lambda well_id: load_well(well_id, args.data_root), well_ids))

    def one(index: int) -> np.ndarray:
        return predict_correction(
            wells[index], baseline[row_starts[index] : row_starts[index + 1]], config
        )

    one(0)  # Compile the Numba kernel outside the worker pool.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        correction = np.concatenate(list(pool.map(one, range(len(well_ids)))))

    fixed = baseline + args.fixed_weight * correction
    weights = np.arange(0.0, 0.30001, 0.025)
    print(config)
    print(
        f"baseline={rmse(baseline, truth):.9f} "
        f"fixed={rmse(fixed, truth):.9f} "
        f"correction_rms={np.sqrt(np.mean(np.square(correction))):.6f}"
    )
    for grouping in ("original", "stratified", "independent"):
        well_fold = layout[grouping].astype(int)
        row_fold = np.repeat(well_fold, np.diff(row_starts))
        nested_score, selected = nested_audit(
            baseline, correction, truth, row_fold, weights
        )
        fold_gains = []
        for fold in range(5):
            mask = row_fold == fold
            fold_gains.append(rmse(baseline, truth, mask) - rmse(fixed, truth, mask))
        print(
            f"{grouping}: nested={nested_score:.9f} weights={selected} "
            f"fixed_fold_gains={np.round(fold_gains, 6).tolist()}"
        )

    if args.output:
        np.savez_compressed(
            args.output,
            well_ids=well_ids,
            row_starts=row_starts,
            correction=correction.astype(np.float32),
            prediction=fixed.astype(np.float32),
            fixed_weight=args.fixed_weight,
        )


if __name__ == "__main__":
    main()
