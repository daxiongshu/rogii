from __future__ import annotations

import numpy as np

from limited_query_cv.v5_batch3_residual_path import (
    PathSearchConfig,
    _distinct_top,
    _emission_map,
    _post_smooth_gr_loss,
    _sample_paths,
    smooth_residual_paths,
)
from rogii.data import Well


HIGHCAP_CONFIG = PathSearchConfig(
    name="self_prefix_highcap",
    half_range=16.0,
    step=0.25,
    particles=8192,
    temperature=1.5,
    momentum_weight=4.0,
    momentum_beta=0.99,
    smooth_sigma=150,
    pseudo_weight=0.8,
    seed=20263001,
    pool_keep=3000,
    anchor_fade_rows=128,
)


def highcap_select(
    residuals: np.ndarray,
    gr_losses: np.ndarray,
    config_name: str,
) -> tuple[np.ndarray, list[str]]:
    predictions = []
    names = []
    for top_m in (64, 128):
        chosen = _distinct_top(residuals, gr_losses, top_m, 1.0)
        curves = residuals[chosen].astype(np.float64)
        loss = gr_losses[chosen]
        divergence = np.sqrt(np.mean(np.square(curves), axis=1))
        for tau in (10.0, 30.0):
            for bandwidth in (4.0, 8.0, 12.0):
                log_weight = (
                    -np.square(loss) / tau
                    -np.square(divergence) / (2.0 * bandwidth**2)
                )
                weight = np.exp(log_weight - np.max(log_weight))
                weight /= np.sum(weight)
                predictions.append(np.sum(curves * weight[:, None], axis=0))
                names.append(
                    f"{config_name}_m{top_m}_tau{tau:g}_sb{bandwidth:g}"
                )
    return np.stack(predictions).astype(np.float32), names


def highcap_path_candidates(
    well: Well,
    baseline: np.ndarray,
    config: PathSearchConfig = HIGHCAP_CONFIG,
    device: str = "cuda",
    observed_shift: int = 0,
) -> tuple[np.ndarray, list[str], dict]:
    emission, raw_square, offsets, gr_scale = _emission_map(
        well, baseline, config, observed_shift=observed_shift
    )
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
    corrections, names = highcap_select(
        residuals, gr_loss, config.name
    )
    diagnostic = {
        "well": well.well_id,
        "config": config.name,
        "particles": config.particles,
        "pool_keep": len(residuals),
        "tail_rows": len(baseline),
        "gr_scale": gr_scale,
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
