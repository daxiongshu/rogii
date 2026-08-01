from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from limited_query_cv.cache_v5_run5_anchor_posterior import (
    calibrated_typewell,
    fill,
)
from limited_query_cv.v5_batch3_residual_path import (
    PathSearchConfig,
    _distinct_top,
    _post_smooth_gr_loss,
    _sample_paths,
    smooth_residual_paths,
)
from rogii.data import Well


@dataclass(frozen=True)
class PoolMap:
    grid: np.ndarray
    mean: np.ndarray
    coverage: np.ndarray
    contributing_wells: int
    contributing_points: int


def _pearson(curves: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite = np.isfinite(reference)
    if finite.sum() < 10:
        return np.zeros(len(curves), dtype=np.float64)
    values = curves[:, finite].astype(np.float64)
    ref = reference[finite].astype(np.float64)
    values -= values.mean(axis=1, keepdims=True)
    ref -= ref.mean()
    denominator = np.sqrt(
        np.sum(np.square(values), axis=1) * np.sum(np.square(ref))
    )
    denominator = np.maximum(denominator, 1e-12)
    return np.sum(values * ref[None], axis=1) / denominator


def z_shape_signal(
    well: Well,
    window: int = 200,
    shift: int = 250,
) -> np.ndarray:
    difference = np.diff(well.z.astype(np.float64), prepend=well.z[0])
    kernel = np.ones(int(window), dtype=np.float64) / int(window)
    smoothed = np.convolve(difference, kernel, mode="same")
    shifted = np.full_like(smoothed, np.nan)
    if shift:
        shifted[:-shift] = smoothed[shift:]
    else:
        shifted = smoothed
    return shifted[well.prediction_indices]


def enriched_map(
    base_grid: np.ndarray,
    base_curve: np.ndarray,
    pool: PoolMap,
    pool_weight: float,
) -> np.ndarray:
    if not np.array_equal(base_grid, pool.grid):
        pool_mean = np.interp(base_grid, pool.grid, pool.mean)
        coverage = np.interp(base_grid, pool.grid, pool.coverage)
    else:
        pool_mean = pool.mean
        coverage = pool.coverage
    blend = np.clip(float(pool_weight) * coverage, 0.0, 1.0)
    return (1.0 - blend) * base_curve + blend * pool_mean


def select_pool_corrections(
    residuals: np.ndarray,
    gr_losses: np.ndarray,
    baseline: np.ndarray,
    config_name: str,
    z_signal: np.ndarray | None,
    z_lam: float,
) -> tuple[np.ndarray, list[str]]:
    predictions = []
    names = []
    for top_m in (32, 64):
        chosen = _distinct_top(residuals, gr_losses, top_m, 0.5)
        curves = residuals[chosen].astype(np.float64)
        loss = gr_losses[chosen]
        divergence = np.sqrt(np.mean(np.square(curves), axis=1))
        correlation = (
            _pearson(baseline[None] + curves, z_signal)
            if z_signal is not None and z_lam
            else np.zeros(len(curves), dtype=np.float64)
        )
        for tau in (10.0, 30.0):
            for bandwidth in (2.0, 4.0, 8.0):
                log_weight = (
                    -np.square(loss) / tau
                    -np.square(divergence) / (2.0 * bandwidth**2)
                    + float(z_lam) * correlation
                )
                weight = np.exp(log_weight - np.max(log_weight))
                weight /= np.sum(weight)
                predictions.append(np.sum(curves * weight[:, None], axis=0))
                names.append(
                    f"{config_name}_m{top_m}_tau{tau:g}_"
                    f"sb{bandwidth:g}_zl{z_lam:g}"
                )
    return np.stack(predictions).astype(np.float32), names


def datum_pool_candidates(
    well: Well,
    baseline: np.ndarray,
    config: PathSearchConfig,
    pool: PoolMap,
    pool_weight: float,
    device: str = "cuda",
    z_lam: float = 0.0,
    observed_shift: int = 0,
) -> tuple[np.ndarray, list[str], dict]:
    rows = well.prediction_indices
    if len(baseline) != len(rows):
        raise ValueError(f"{well.well_id}: C016/tail row mismatch")
    grid, base_curve, calibrated_gr, gr_scale = calibrated_typewell(
        well, config.pseudo_weight
    )
    curve = enriched_map(grid, base_curve, pool, pool_weight)
    observed = fill(
        calibrated_gr[rows],
        float(np.nanmedian(calibrated_gr[well.known_indices])),
    )
    if observed_shift:
        observed = np.roll(observed, int(observed_shift))
    offsets = np.arange(
        -config.half_range,
        config.half_range + 0.5 * config.step,
        config.step,
        dtype=np.float64,
    )
    candidate = baseline[:, None].astype(np.float64) + offsets[None]
    expected = np.interp(candidate.ravel(), grid, curve).reshape(
        candidate.shape
    )
    standardized = (observed[:, None] - expected) / max(gr_scale, 3.0)
    emission = np.minimum(np.abs(standardized), 8.0).astype(np.float32)
    raw_square = np.square(observed[:, None] - expected).astype(np.float32)
    state_paths, raw_cost = _sample_paths(emission, config, device)
    residuals = offsets[state_paths].astype(np.float32)
    residuals = smooth_residual_paths(
        residuals, config.smooth_sigma, device
    )
    residuals -= residuals[:, :1]
    fade_rows = min(config.anchor_fade_rows, residuals.shape[1])
    if fade_rows:
        fade = np.linspace(0.0, 1.0, fade_rows, dtype=np.float32)
        residuals[:, :fade_rows] *= fade[None]
    residuals = np.clip(
        residuals, -config.half_range, config.half_range
    )
    gr_loss = _post_smooth_gr_loss(residuals, raw_square, config)
    z_signal = z_shape_signal(well) if z_lam else None
    map_name = (
        f"datum_pw{pool_weight:g}"
        if not z_lam
        else f"datum_z_pw{pool_weight:g}"
    )
    corrections, names = select_pool_corrections(
        residuals,
        gr_loss,
        baseline.astype(np.float64),
        f"{map_name}_{config.name}",
        z_signal,
        z_lam,
    )
    diagnostic = {
        "well": well.well_id,
        "config": config.name,
        "map": map_name,
        "pool_weight": float(pool_weight),
        "z_lam": float(z_lam),
        "contributing_wells": int(pool.contributing_wells),
        "contributing_points": int(pool.contributing_points),
        "particles": config.particles,
        "pool_keep": len(residuals),
        "tail_rows": len(baseline),
        "gr_scale": float(gr_scale),
        "raw_cost_min": float(np.min(raw_cost)),
        "post_gr_loss_min": float(np.min(gr_loss)),
        "post_gr_loss_median": float(np.median(gr_loss)),
        "observed_shift": int(observed_shift),
    }
    return baseline[None].astype(np.float32) + corrections, names, diagnostic
