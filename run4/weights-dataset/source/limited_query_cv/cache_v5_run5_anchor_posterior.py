from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well
from rogii.sequence import _robust_affine


V20_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_001_goal_050/v20_components"
)
PSEUDO_WEIGHTS = (0.0, 0.5, 0.8, 1.0)


@dataclass(frozen=True)
class Variant:
    prior: str
    pseudo_weight: float
    q_walk: float
    decoder: str

    @property
    def name(self) -> str:
        weight = str(self.pseudo_weight).replace(".", "p")
        q = str(self.q_walk).replace(".", "p")
        return (
            f"{self.prior}_pseudo{weight}_q{q}_{self.decoder}"
        )


VARIANTS = tuple(
    Variant(prior, weight, 0.35, "posterior_mean")
    for prior in ("linear", "v20")
    for weight in PSEUDO_WEIGHTS
) + (
    Variant("linear", 0.8, 0.2, "posterior_mean"),
    Variant("linear", 0.8, 0.6, "posterior_mean"),
    Variant("linear", 0.8, 0.35, "trimmed_posterior_mean"),
    Variant("linear", 0.8, 0.35, "map"),
    Variant("v20", 0.8, 0.35, "trimmed_posterior_mean"),
    Variant("v20", 0.8, 0.35, "map"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_wells(
    well_ids: np.ndarray, data_root: Path, workers: int
) -> list[Well]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
        )


def fill(values: np.ndarray, fallback: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), fallback, dtype=np.float64)
    positions = np.arange(len(values), dtype=np.float64)
    return np.interp(
        positions,
        positions[finite],
        values[finite],
    )


def surface_rate(well: Well, rows: int = 50) -> float:
    known = well.known_indices
    selected = known[-min(rows, len(known)) :]
    md = well.md[selected].astype(np.float64)
    surface = (
        well.tvt_input[selected].astype(np.float64)
        + well.z[selected].astype(np.float64)
    )
    difference_md = np.diff(md)
    valid = np.isfinite(difference_md) & (difference_md > 1e-6)
    if valid.sum() < 3:
        return 0.0
    rates = np.diff(surface)[valid] / difference_md[valid]
    return float(np.clip(np.median(rates), -0.25, 0.25))


def calibrated_typewell(
    well: Well,
    pseudo_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    tw_tvt = np.asarray(well.typewell_tvt, dtype=np.float64)
    tw_gr = np.asarray(well.typewell_gr, dtype=np.float64)
    finite = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[finite]
    tw_gr = tw_gr[finite]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    known = well.known_indices
    reference = np.interp(
        well.tvt_input[known],
        tw_tvt,
        tw_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, intercept, residual_scale = _robust_affine(
        reference, well.gr[known]
    )
    calibrated_gr = (
        fill(well.gr, float(gain * np.mean(tw_gr) + intercept)) - intercept
    ) / max(gain, 1e-6)
    calibrated_scale = float(
        np.clip(residual_scale / max(abs(gain), 1e-6), 3.0, 60.0)
    )
    step = 0.2
    grid = np.arange(tw_tvt[0], tw_tvt[-1] + 0.5 * step, step)
    curve = np.interp(grid, tw_tvt, tw_gr)
    if pseudo_weight > 0:
        known_tvt = well.tvt_input[known]
        known_gr = calibrated_gr[known]
        indices = np.rint((known_tvt - grid[0]) / step).astype(np.int64)
        inside = (
            np.isfinite(known_tvt)
            & np.isfinite(known_gr)
            & (indices >= 0)
            & (indices < len(grid))
        )
        total = np.bincount(
            indices[inside],
            weights=known_gr[inside],
            minlength=len(grid),
        ).astype(np.float64)
        count = np.bincount(
            indices[inside], minlength=len(grid)
        ).astype(np.float64)
        sigma = 0.5 / step
        smooth_total = gaussian_filter1d(
            total, sigma=sigma, mode="constant"
        )
        smooth_count = gaussian_filter1d(
            count, sigma=sigma, mode="constant"
        )
        pseudo = smooth_total / np.maximum(smooth_count, 1e-12)
        coverage = np.clip(smooth_count / 0.05, 0.0, 1.0)
        blend = pseudo_weight * coverage
        curve = (1.0 - blend) * curve + blend * pseudo
    return grid, curve, calibrated_gr, calibrated_scale


def prior_path(
    well: Well,
    baseline: np.ndarray,
    kind: str,
) -> np.ndarray:
    anchor = well.anchor_index
    tail = well.tail_indices
    if len(baseline) != len(tail):
        raise ValueError(f"{well.well_id}: baseline/tail changed")
    result = np.empty(len(well.md), dtype=np.float64)
    result[: anchor + 1] = well.tvt_input[: anchor + 1]
    if kind == "v20":
        result[tail] = baseline
    elif kind == "linear":
        rate = surface_rate(well)
        anchor_surface = float(
            well.tvt_input[anchor] + well.z[anchor]
        )
        surface = anchor_surface + rate * (
            well.md[tail] - well.md[anchor]
        )
        result[tail] = surface - well.z[tail]
    else:
        raise ValueError(f"unknown prior {kind}")
    return result


def posterior_decode(
    well: Well,
    prior: np.ndarray,
    typewell_grid: np.ndarray,
    typewell_gr: np.ndarray,
    observed_gr: np.ndarray,
    gr_scale: float,
    q_walk: float,
    decoder: str,
    stride: int,
    grid_half: float,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = well.anchor_index
    tail = well.tail_indices
    query = np.unique(
        np.r_[anchor, tail[::stride], tail[-1]]
    ).astype(np.int64)
    offsets = np.arange(
        -grid_half,
        grid_half + 0.5 * grid_step,
        grid_step,
        dtype=np.float64,
    )
    center = int(np.argmin(np.abs(offsets)))
    candidate_tvt = prior[query, None] + offsets[None, :]
    expected = np.interp(
        candidate_tvt.ravel(),
        typewell_grid,
        typewell_gr,
    ).reshape(candidate_tvt.shape)
    standardized = (
        observed_gr[query, None] - expected
    ) / max(gr_scale, 1e-6)
    log_likelihood = -0.5 * np.minimum(
        np.square(standardized), 80.0
    )
    outside = np.maximum(
        typewell_grid[0] - candidate_tvt,
        candidate_tvt - typewell_grid[-1],
    )
    outside = np.maximum(outside, 0.0)
    log_likelihood -= np.minimum(np.square(outside / 5.0), 80.0)
    log_likelihood[0] = -1e6
    log_likelihood[0, center] = 0.0
    emission = np.exp(
        log_likelihood
        - np.max(log_likelihood, axis=1, keepdims=True)
    )
    emission = np.maximum(emission, 1e-300)
    time_count, state_count = emission.shape
    forward = np.empty_like(emission)
    state = np.zeros(state_count, dtype=np.float64)
    state[center] = 1.0
    forward[0] = state
    md_step = np.r_[1.0, np.maximum(np.diff(well.md[query]), 1e-3)]
    sigma = np.clip(
        q_walk * np.sqrt(md_step) / grid_step,
        1e-3,
        state_count / 3.0,
    )
    for time_index in range(1, time_count):
        state = gaussian_filter1d(
            state, sigma[time_index], mode="constant"
        )
        state *= emission[time_index]
        total = float(np.sum(state))
        state = (
            state / total
            if total > 0 and np.isfinite(total)
            else np.full(state_count, 1.0 / state_count)
        )
        forward[time_index] = state
    backward = np.empty_like(emission)
    state = np.full(state_count, 1.0 / state_count)
    backward[-1] = state
    for time_index in range(time_count - 2, -1, -1):
        state = state * emission[time_index + 1]
        state = gaussian_filter1d(
            state, sigma[time_index + 1], mode="constant"
        )
        total = float(np.sum(state))
        state = (
            state / total
            if total > 0 and np.isfinite(total)
            else np.full(state_count, 1.0 / state_count)
        )
        backward[time_index] = state
    posterior = forward * backward
    total = np.sum(posterior, axis=1, keepdims=True)
    posterior = np.divide(
        posterior,
        total,
        out=np.full_like(posterior, 1.0 / state_count),
        where=total > 0,
    )
    if decoder == "posterior_mean":
        decoded_offset = posterior @ offsets
    elif decoder == "trimmed_posterior_mean":
        cumulative = np.cumsum(posterior, axis=1)
        supported = (cumulative >= 0.1) & (
            cumulative - posterior <= 0.9
        )
        trimmed = posterior * supported
        trimmed /= np.maximum(
            np.sum(trimmed, axis=1, keepdims=True), 1e-300
        )
        decoded_offset = trimmed @ offsets
    elif decoder == "map":
        decoded_offset = offsets[np.argmax(posterior, axis=1)]
    else:
        raise ValueError(f"unknown decoder {decoder}")
    posterior_mean = posterior @ offsets
    posterior_std = np.sqrt(
        np.maximum(
            posterior @ np.square(offsets)
            - np.square(posterior_mean),
            0.0,
        )
    )
    sampled = prior[query] + decoded_offset
    sampled[0] = float(well.tvt_input[anchor])
    return (
        np.interp(well.md[tail], well.md[query], sampled),
        np.interp(
            well.md[tail], well.md[query], posterior_std
        ),
    )


def predict_well(
    well: Well,
    baseline: np.ndarray,
    stride: int,
    grid_half: float,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    typewell_views = {
        weight: calibrated_typewell(well, weight)
        for weight in PSEUDO_WEIGHTS
    }
    priors = {
        kind: prior_path(well, baseline, kind)
        for kind in ("linear", "v20")
    }
    predictions = []
    uncertainties = []
    for variant in VARIANTS:
        grid, curve, observed, scale = typewell_views[
            variant.pseudo_weight
        ]
        prediction, uncertainty = posterior_decode(
            well,
            priors[variant.prior],
            grid,
            curve,
            observed,
            scale,
            variant.q_walk,
            variant.decoder,
            stride,
            grid_half,
            grid_step,
        )
        predictions.append(prediction.astype(np.float32))
        uncertainties.append(uncertainty.astype(np.float32))
    matrix = np.stack(predictions)
    uncertainty_matrix = np.stack(uncertainties)
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(uncertainty_matrix).all()
        or np.any(uncertainty_matrix < 0)
    ):
        raise RuntimeError(f"{well.well_id}: nonfinite posterior path")
    return matrix, uncertainty_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cache deterministic Run-5 heel-adapted forward-backward "
            "posterior paths for one F0-F3 fold without accessing truth."
        )
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--v20-root", type=Path, default=V20_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-wells", type=int, default=0)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--grid-half", type=float, default=50.0)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite posterior artifacts")
    if args.stride != 16 or args.grid_half != 50.0 or args.grid_step != 0.5:
        raise ValueError("eligible posterior geometry changed")

    started = time.time()
    v20_path = args.v20_root / f"fold{args.fold}.npz"
    with np.load(v20_path, allow_pickle=False) as cache:
        if int(cache["fold"]) != args.fold:
            raise ValueError("V20 fold changed")
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise ValueError("V20 cache firewall failed")
        well_ids = cache["well_ids"].astype(str)
        row_starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float64)
    if args.maximum_wells:
        well_ids = well_ids[: args.maximum_wells]
        row_starts = row_starts[: args.maximum_wells + 1]
        baseline = baseline[: row_starts[-1]]
    wells = load_wells(well_ids, args.data_root, args.workers)

    # Compile/cache the execution path and then use independent wells in
    # parallel. SciPy's one-dimensional filters release the GIL.
    predict_well(
        wells[0],
        baseline[row_starts[0] : row_starts[1]],
        args.stride,
        args.grid_half,
        args.grid_step,
    )

    def one(index: int) -> np.ndarray:
        left, right = row_starts[index : index + 2]
        return predict_well(
            wells[index],
            baseline[left:right],
            args.stride,
            args.grid_half,
            args.grid_step,
        )

    matrices = []
    uncertainty_matrices = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, (matrix, uncertainty) in enumerate(
            pool.map(one, range(len(wells))), start=1
        ):
            matrices.append(matrix)
            uncertainty_matrices.append(uncertainty)
            if count % 10 == 0 or count == len(wells):
                print(
                    f"fold={args.fold} wells={count}/{len(wells)} "
                    f"elapsed={time.time() - started:.1f}",
                    flush=True,
                )
    prediction = np.concatenate(matrices, axis=1)
    posterior_std = np.concatenate(uncertainty_matrices, axis=1)
    expected_rows = int(row_starts[-1])
    if (
        prediction.shape != (len(VARIANTS), expected_rows)
        or posterior_std.shape != prediction.shape
    ):
        raise RuntimeError("posterior cache shape changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fold=np.int8(args.fold),
        well_ids=well_ids,
        row_starts=row_starts,
        prediction=prediction,
        posterior_std=posterior_std,
        variant_names=np.asarray([variant.name for variant in VARIANTS]),
        stride=np.int32(args.stride),
        grid_half=np.float32(args.grid_half),
        grid_step=np.float32(args.grid_step),
        hidden_truth_loaded=np.bool_(False),
        hidden_metrics_computed=np.bool_(False),
        audit_fold_loaded=np.bool_(False),
    )
    record = {
        "schema_version": 1,
        "family": "heel_adapted_anchor_posterior_tracker",
        "status": "label_free_posterior_cache_complete",
        "fold": args.fold,
        "wells": len(well_ids),
        "rows": expected_rows,
        "variants": len(VARIANTS),
        "variant_names": [variant.name for variant in VARIANTS],
        "stride": args.stride,
        "grid_half": args.grid_half,
        "grid_step": args.grid_step,
        "hidden_truth_loaded": False,
        "hidden_metrics_computed": False,
        "audit_fold_loaded": False,
        "v20_source": str(v20_path),
        "v20_source_sha256": sha256(v20_path),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source": str(Path(__file__)),
        "source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
