from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from limited_query_cv.cache_v5_run5_anchor_posterior import (
    calibrated_typewell,
    fill,
)
from rogii.data import Well


@dataclass(frozen=True)
class PathSearchConfig:
    name: str = "self_prefix_medium"
    half_range: float = 16.0
    step: float = 0.25
    particles: int = 4096
    temperature: float = 1.5
    momentum_weight: float = 4.0
    momentum_beta: float = 0.99
    smooth_sigma: int = 128
    pseudo_weight: float = 0.8
    seed: int = 20263001
    pool_keep: int = 512
    anchor_fade_rows: int = 128


SEARCH_CONFIGS = (
    PathSearchConfig(
        name="self_prefix_tight",
        half_range=8.0,
        step=0.25,
        particles=2048,
        temperature=0.75,
        momentum_weight=2.0,
        smooth_sigma=64,
        seed=20263001,
    ),
    PathSearchConfig(),
    PathSearchConfig(
        name="self_prefix_wide",
        half_range=24.0,
        step=0.5,
        particles=4096,
        temperature=3.0,
        momentum_weight=4.0,
        smooth_sigma=192,
        seed=20263001,
    ),
)


def _gaussian_kernel(sigma: int, device: torch.device) -> torch.Tensor:
    radius = max(1, int(sigma))
    axis = torch.arange(
        -3 * radius,
        3 * radius + 1,
        dtype=torch.float32,
        device=device,
    )
    kernel = torch.exp(-0.5 * torch.square(axis / radius))
    return (kernel / kernel.sum())[None, None]


def smooth_residual_paths(
    residuals: np.ndarray,
    sigma: int,
    device: str,
    chunk: int = 128,
) -> np.ndarray:
    """Smooth only anchor-relative residuals, avoiding absolute-TVT TF32 loss."""
    target = torch.device(device)
    kernel = _gaussian_kernel(sigma, target)
    padding = kernel.shape[-1] // 2
    outputs = []
    for start in range(0, len(residuals), chunk):
        values = torch.as_tensor(
            residuals[start : start + chunk],
            dtype=torch.float32,
            device=target,
        )[:, None]
        padded = F.pad(values, (padding, padding), mode="replicate")
        outputs.append(
            F.conv1d(padded, kernel).squeeze(1).cpu().numpy()
        )
    return np.concatenate(outputs).astype(np.float32)


def _emission_map(
    well: Well,
    baseline: np.ndarray,
    config: PathSearchConfig,
    observed_shift: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = well.prediction_indices
    if len(baseline) != len(rows):
        raise ValueError(f"{well.well_id}: C016/tail row mismatch")
    grid, curve, calibrated_gr, gr_scale = calibrated_typewell(
        well, config.pseudo_weight
    )
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
    return emission, raw_square, offsets, float(gr_scale)


def _sample_paths(
    emission: np.ndarray,
    config: PathSearchConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = torch.device(device)
    energy = torch.as_tensor(emission, dtype=torch.float32, device=target)
    rows, states = energy.shape
    center = int(np.argmin(np.abs(
        np.linspace(-config.half_range, config.half_range, states)
    )))
    particles = int(config.particles)
    state = torch.full(
        (particles,), center, dtype=torch.long, device=target
    )
    velocity = torch.zeros(particles, dtype=torch.float32, device=target)
    generator = torch.Generator(device=target)
    generator.manual_seed(int(config.seed))
    momentum = (
        2.0
        * config.momentum_weight
        * torch.rand(particles, generator=generator, device=target)
    )
    moves = torch.tensor((-1, 0, 1), dtype=torch.long, device=target)
    paths = torch.empty(
        (particles, rows), dtype=torch.int16, device=target
    )
    cumulative = torch.zeros(particles, dtype=torch.float32, device=target)
    particle_index = torch.arange(particles, device=target)
    for row in range(rows):
        options = torch.clamp(state[:, None] + moves[None], 0, states - 1)
        unary = energy[row, options]
        motion = moves[None].float() - velocity[:, None]
        logits = (
            -unary / config.temperature
            - momentum[:, None] * torch.square(motion)
        )
        random = torch.rand(
            logits.shape, generator=generator, device=target
        ).clamp_(1e-7, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(random))
        choice = torch.argmax(logits + gumbel, dim=1)
        selected_move = moves[choice]
        state = options[particle_index, choice]
        selected_unary = unary[particle_index, choice]
        cumulative += selected_unary
        velocity = (
            config.momentum_beta * velocity
            + (1.0 - config.momentum_beta) * selected_move.float()
        )
        paths[:, row] = state.to(torch.int16)
    keep = min(int(config.pool_keep), particles)
    selected = torch.topk(cumulative, keep, largest=False).indices
    return (
        paths[selected].cpu().numpy().astype(np.int32),
        cumulative[selected].cpu().numpy().astype(np.float32),
    )


def _post_smooth_gr_loss(
    residuals: np.ndarray,
    raw_square: np.ndarray,
    config: PathSearchConfig,
) -> np.ndarray:
    position = (
        residuals.astype(np.float64) + config.half_range
    ) / config.step
    left = np.floor(position).astype(np.int64)
    fraction = position - left
    left = np.clip(left, 0, raw_square.shape[1] - 1)
    right = np.clip(left + 1, 0, raw_square.shape[1] - 1)
    row = np.arange(raw_square.shape[0], dtype=np.int64)[None]
    square = (
        raw_square[row, left] * (1.0 - fraction)
        + raw_square[row, right] * fraction
    )
    return np.sqrt(np.mean(square, axis=1)).astype(np.float64)


def _distinct_top(
    residuals: np.ndarray,
    losses: np.ndarray,
    top_m: int,
    tolerance: float,
) -> np.ndarray:
    order = np.argsort(losses)
    kept = [int(order[0])]
    for candidate in order[1:]:
        if len(kept) >= int(top_m):
            break
        separation = np.max(
            np.abs(residuals[int(candidate)] - residuals[kept]), axis=1
        )
        if np.all(separation > tolerance):
            kept.append(int(candidate))
    return np.asarray(kept, dtype=np.int64)


def select_corrections(
    residuals: np.ndarray,
    gr_losses: np.ndarray,
    config_name: str,
) -> tuple[np.ndarray, list[str]]:
    predictions = []
    names = []
    for top_m in (32, 64):
        chosen = _distinct_top(residuals, gr_losses, top_m, 0.5)
        curves = residuals[chosen].astype(np.float64)
        loss = gr_losses[chosen]
        divergence = np.sqrt(np.mean(np.square(curves), axis=1))
        for tau in (10.0, 30.0):
            for bandwidth in (2.0, 4.0, 8.0):
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
    predictions.append(
        residuals[int(np.argmin(gr_losses))].astype(np.float64)
    )
    names.append(f"{config_name}_best_gr")
    return np.stack(predictions).astype(np.float32), names


def residual_path_candidates(
    well: Well,
    baseline: np.ndarray,
    config: PathSearchConfig,
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
    corrections, names = select_corrections(
        residuals, gr_loss, config.name
    )
    diagnostics = {
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
    return baseline[None].astype(np.float32) + corrections, names, diagnostics


def hidden_target_variants(well: Well) -> tuple[Well, Well]:
    reverse = replace(well, tvt=well.tvt[::-1].copy())
    missing = replace(well, tvt=np.full_like(well.tvt, np.nan))
    return reverse, missing
