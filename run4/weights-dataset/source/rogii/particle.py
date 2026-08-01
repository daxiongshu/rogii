from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d

from .data import Well


@dataclass(frozen=True)
class ParticleConfig:
    particles: int = 384
    seeds: int = 16
    likelihood_temperature: float = 5.0
    rate_momentum: float = 0.998
    rate_noise: float = 0.002
    position_noise: float = 0.005
    initial_position_std: float = 4.5
    initial_rate_std: float = 0.01
    rough_position: float = 0.1
    rough_rate: float = 0.001
    resample_fraction: float = 0.5
    pseudo_typewell_weight: float = 0.0
    pseudo_bandwidth: float = 0.5


@njit(cache=True, nogil=True)
def _grid_interpolate(grid: np.ndarray, minimum: float, step: float, value: float) -> float:
    location = (value - minimum) / step
    left = int(np.floor(location))
    if left <= 0:
        return grid[0]
    if left >= len(grid) - 1:
        return grid[-1]
    fraction = location - left
    return grid[left] * (1.0 - fraction) + grid[left + 1] * fraction


@njit(cache=True, nogil=True)
def _systematic_resample(
    position: np.ndarray,
    rate: np.ndarray,
    weight: np.ndarray,
    rough_position: float,
    rough_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(weight)
    cumulative = np.cumsum(weight)
    start = np.random.random() / n
    output_position = np.empty(n, dtype=np.float64)
    output_rate = np.empty(n, dtype=np.float64)
    source = 0
    for j in range(n):
        threshold = start + j / n
        while source < n - 1 and cumulative[source] < threshold:
            source += 1
        output_position[j] = position[source] + rough_position * np.random.randn()
        output_rate[j] = rate[source] + rough_rate * np.random.randn()
    return output_position, output_rate


@njit(cache=True, nogil=True)
def _run_seed(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    typewell_grid: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    initial_surface: float,
    initial_rate: float,
    seed: int,
    particles: int,
    rate_momentum: float,
    rate_noise: float,
    position_noise: float,
    initial_position_std: float,
    initial_rate_std: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
) -> tuple[np.ndarray, float]:
    np.random.seed(seed)
    position = initial_surface + initial_position_std * np.random.randn(particles)
    rate = initial_rate + initial_rate_std * np.random.randn(particles)
    weight = np.full(particles, 1.0 / particles, dtype=np.float64)
    prediction = np.empty(len(md), dtype=np.float64)
    previous_md = md[0] - 1.0
    log_likelihood = 0.0

    for row in range(len(md)):
        md_step = max(md[row] - previous_md, 1.0)
        for j in range(particles):
            rate[j] = rate_momentum * rate[j] + rate_noise * np.random.randn()
            position[j] += rate[j] * md_step + position_noise * np.random.randn()

        if np.isfinite(gr[row]):
            weight_sum = 0.0
            likelihood_sum = 0.0
            for j in range(particles):
                candidate_tvt = position[j] - z[row]
                expected = _grid_interpolate(
                    typewell_grid, grid_minimum, grid_step, candidate_tvt
                )
                residual = (gr[row] - expected) / gr_scale
                likelihood = np.exp(-0.5 * min(residual * residual, 600.0))
                likelihood = max(likelihood, 1e-300)
                likelihood_sum += weight[j] * likelihood
                weight[j] *= likelihood
                weight_sum += weight[j]
            log_likelihood += np.log(max(likelihood_sum, 1e-300))
            if weight_sum > 0:
                weight /= weight_sum
            else:
                weight[:] = 1.0 / particles

        effective_inverse = np.sum(weight * weight)
        if 1.0 / effective_inverse < resample_fraction * particles:
            position, rate = _systematic_resample(
                position, rate, weight, rough_position, rough_rate
            )
            weight[:] = 1.0 / particles

        prediction[row] = np.sum(weight * (position - z[row]))
        previous_md = md[row]
    return prediction, log_likelihood


def particle_seed_candidates(
    well: Well, config: ParticleConfig
) -> tuple[np.ndarray, np.ndarray]:
    known = well.known_indices
    tail = well.tail_indices
    reference_known = np.interp(
        well.tvt_input[known], well.typewell_tvt, well.typewell_gr
    )
    residual = well.gr[known] - reference_known
    gr_scale = float(np.clip(np.nanstd(residual), 10.0, 60.0))

    recent = known[-30:]
    surface = well.tvt_input[recent] + well.z[recent]
    md_delta = np.diff(well.md[recent])
    valid = md_delta > 0
    if valid.sum() >= 3:
        initial_rate = float(np.median(np.diff(surface)[valid] / md_delta[valid]))
    else:
        initial_rate = 0.0
    initial_surface = float(well.tvt_input[known[-1]] + well.z[known[-1]])

    grid_step = 0.2
    grid_minimum = float(np.nanmin(well.typewell_tvt))
    grid_maximum = float(np.nanmax(well.typewell_tvt))
    grid_tvt = np.arange(grid_minimum, grid_maximum + grid_step, grid_step)
    typewell_grid = np.interp(grid_tvt, well.typewell_tvt, well.typewell_gr).astype(
        np.float64
    )
    if config.pseudo_typewell_weight > 0:
        valid = np.isfinite(well.tvt_input[known]) & np.isfinite(well.gr[known])
        known_tvt = well.tvt_input[known][valid]
        known_gr = well.gr[known][valid]
        indices = np.rint((known_tvt - grid_minimum) / grid_step).astype(np.int64)
        inside = (indices >= 0) & (indices < len(typewell_grid))
        total = np.bincount(
            indices[inside], weights=known_gr[inside], minlength=len(typewell_grid)
        ).astype(np.float64)
        count = np.bincount(indices[inside], minlength=len(typewell_grid)).astype(np.float64)
        sigma = max(config.pseudo_bandwidth / grid_step, 0.5)
        smooth_total = gaussian_filter1d(total, sigma=sigma, mode="constant")
        smooth_count = gaussian_filter1d(count, sigma=sigma, mode="constant")
        pseudo = smooth_total / np.maximum(smooth_count, 1e-12)
        coverage = np.clip(smooth_count / 0.05, 0.0, 1.0)
        blend = config.pseudo_typewell_weight * coverage
        typewell_grid = (1.0 - blend) * typewell_grid + blend * pseudo
    predictions = []
    likelihoods = []
    for seed in range(config.seeds):
        prediction, likelihood = _run_seed(
            well.md[tail].astype(np.float64),
            well.z[tail].astype(np.float64),
            well.gr[tail].astype(np.float64),
            typewell_grid,
            grid_minimum,
            grid_step,
            gr_scale,
            initial_surface,
            initial_rate,
            seed,
            config.particles,
            config.rate_momentum,
            config.rate_noise,
            config.position_noise,
            config.initial_position_std,
            config.initial_rate_std,
            config.rough_position,
            config.rough_rate,
            config.resample_fraction,
        )
        predictions.append(prediction)
        likelihoods.append(likelihood)
    return np.stack(predictions), np.asarray(likelihoods)


def predict_particle_tail(well: Well, config: ParticleConfig) -> np.ndarray:
    predictions, likelihoods_array = particle_seed_candidates(well, config)
    log_weight = (
        likelihoods_array - likelihoods_array.max()
    ) / config.likelihood_temperature
    weight = np.exp(log_weight)
    weight /= weight.sum()
    return np.sum(predictions * weight[:, None], axis=0)
