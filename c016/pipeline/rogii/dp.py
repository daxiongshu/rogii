from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .data import Well


@dataclass(frozen=True)
class DPConfig:
    stride: int = 10
    state_radius: float = 80.0
    state_step: float = 0.5
    max_jump: int = 4
    transition_weight: float = 0.10
    emission_scale: float = 2.0
    emission_cap: float = 9.0


@njit(cache=True, nogil=True)
def _viterbi(
    observations: np.ndarray,
    observation_valid: np.ndarray,
    state_signal: np.ndarray,
    anchor_state: int,
    max_jump: int,
    transition_weight: float,
    emission_scale: float,
    emission_cap: float,
) -> np.ndarray:
    n_time = observations.size
    n_state = state_signal.size
    inf = 1e30
    previous = np.full(n_state, inf, dtype=np.float64)
    previous[anchor_state] = 0.0
    back = np.zeros((n_time, n_state), dtype=np.int16)

    for t in range(1, n_time):
        current = np.full(n_state, inf, dtype=np.float64)
        for state in range(n_state):
            best_cost = inf
            best_previous = state
            lo = max(0, state - max_jump)
            hi = min(n_state, state + max_jump + 1)
            for old_state in range(lo, hi):
                jump = state - old_state
                candidate = previous[old_state] + transition_weight * jump * jump
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = old_state
            if observation_valid[t]:
                residual = (observations[t] - state_signal[state]) / emission_scale
                emission = residual * residual
                if emission > emission_cap:
                    emission = emission_cap
                best_cost += emission
            current[state] = best_cost
            back[t, state] = best_previous
        previous = current

    path = np.empty(n_time, dtype=np.int32)
    path[-1] = int(np.argmin(previous))
    for t in range(n_time - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    path[0] = anchor_state
    return path


def _robust_affine(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(reference) & np.isfinite(observed)
    x = reference[mask]
    y = observed[mask]
    if len(x) < 20 or np.std(x) < 1e-6:
        return 1.0, 0.0, 15.0

    design = np.column_stack([x, np.ones_like(x)])
    weights = np.ones(len(x), dtype=np.float64)
    coef = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(5):
        root_w = np.sqrt(weights)
        coef = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)[0]
        coef[0] = np.clip(coef[0], 0.2, 3.0)
        residual = y - design @ coef
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-6))
    residual = y - design @ coef
    scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
    return float(coef[0]), float(coef[1]), float(scale)


def _query_indices(well: Well, stride: int) -> np.ndarray:
    tail = well.tail_indices
    return np.unique(np.concatenate([[well.anchor_index], tail[::stride], [tail[-1]]])).astype(
        np.int64
    )


def predict_tail(well: Well, config: DPConfig) -> np.ndarray:
    anchor = well.anchor_index
    anchor_tvt = float(well.tvt_input[anchor])
    query = _query_indices(well, config.stride)

    known = well.known_indices
    known_reference = np.interp(
        well.tvt_input[known], well.typewell_tvt, well.typewell_gr, left=np.nan, right=np.nan
    )
    gain, offset, residual_scale = _robust_affine(known_reference, well.gr[known])

    offsets = np.arange(
        -config.state_radius,
        config.state_radius + 0.5 * config.state_step,
        config.state_step,
        dtype=np.float64,
    )
    states = anchor_tvt + offsets
    reference_signal = np.interp(
        states, well.typewell_tvt, well.typewell_gr, left=np.nan, right=np.nan
    )
    state_signal = gain * reference_signal + offset
    finite_state = np.isfinite(state_signal)
    if not np.all(finite_state):
        # A large sentinel makes states outside the typewell range unattractive.
        state_signal = np.where(finite_state, state_signal, 1e9)

    observations = well.gr[query].astype(np.float64)
    valid = np.isfinite(observations)
    observations = np.where(valid, observations, 0.0)
    anchor_state = int(np.argmin(np.abs(offsets)))
    path = _viterbi(
        observations,
        valid,
        state_signal.astype(np.float64),
        anchor_state,
        config.max_jump,
        config.transition_weight,
        max(residual_scale * config.emission_scale, 1e-6),
        config.emission_cap,
    )
    coarse_tvt = states[path]
    tail = well.tail_indices
    return np.interp(well.md[tail], well.md[query], coarse_tvt)
