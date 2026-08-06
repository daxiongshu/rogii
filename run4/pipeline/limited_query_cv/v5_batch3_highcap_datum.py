from __future__ import annotations

import numpy as np

from limited_query_cv.cache_v5_run5_anchor_posterior import (
    calibrated_typewell,
    fill,
)
from limited_query_cv.v5_batch3_datum_pool import PoolMap, enriched_map
from limited_query_cv.v5_batch3_highcap_path import highcap_select
from limited_query_cv.v5_batch3_residual_path import (
    PathSearchConfig,
    _post_smooth_gr_loss,
    _sample_paths,
    smooth_residual_paths,
)
from rogii.data import Well


def highcap_datum_candidates(
    well: Well,
    baseline: np.ndarray,
    config: PathSearchConfig,
    pool: PoolMap,
    pool_weight: float,
    device: str = "cuda",
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
    name = f"datum_pw{pool_weight:g}_{config.name}"
    corrections, names = highcap_select(residuals, gr_loss, name)
    diagnostic = {
        "well": well.well_id,
        "config": config.name,
        "map": f"datum_pw{pool_weight:g}",
        "pool_weight": float(pool_weight),
        "contributing_wells": int(pool.contributing_wells),
        "contributing_points": int(pool.contributing_points),
        "particles": config.particles,
        "pool_keep": len(residuals),
        "tail_rows": len(baseline),
        "gr_scale": float(gr_scale),
        "raw_cost_min": float(np.min(raw_cost)),
        "raw_cost_median": float(np.median(raw_cost)),
        "post_gr_loss_min": float(np.min(gr_loss)),
        "post_gr_loss_median": float(np.median(gr_loss)),
        "correction_rms_min": float(
            np.min(np.sqrt(np.mean(np.square(residuals), axis=1)))
        ),
        "correction_rms_median": float(
            np.median(np.sqrt(np.mean(np.square(residuals), axis=1)))
        ),
        "observed_shift": int(observed_shift),
    }
    return baseline[None].astype(np.float32) + corrections, names, diagnostic
