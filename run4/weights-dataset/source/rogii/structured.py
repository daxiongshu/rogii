from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _logsumexp_slice(values: np.ndarray, start: int, stop: int) -> float:
    maximum = -1.0e30
    for index in range(start, stop):
        if values[index] > maximum:
            maximum = values[index]
    total = 0.0
    for index in range(start, stop):
        total += np.exp(values[index] - maximum)
    return maximum + np.log(max(total, 1.0e-30))


@njit(cache=True)
def structured_posterior_mean(
    logits: np.ndarray,
    offsets: np.ndarray,
    z_relative: np.ndarray,
    expected_first_step: float,
    expected_surface_step: float,
    transition_sigma: float,
    anchor_sigma: float,
    temperature: float = 1.5,
    truncation_sigma: float = 4.0,
) -> np.ndarray:
    """Posterior-mean offset under a smooth geological-surface path prior."""
    state_count, time_count = logits.shape
    alpha = np.empty((state_count, time_count), dtype=np.float64)
    beta = np.empty((state_count, time_count), dtype=np.float64)
    step = offsets[1] - offsets[0]
    radius = max(2, int(np.ceil(truncation_sigma * transition_sigma / step)) + 1)
    transition_scale = -0.5 / (transition_sigma * transition_sigma)
    anchor_scale = -0.5 / (anchor_sigma * anchor_sigma)

    maximum = -1.0e30
    for state in range(state_count):
        surface = offsets[state] + z_relative[0]
        value = (
            logits[state, 0] / temperature
            + anchor_scale * (surface - expected_first_step) ** 2
        )
        alpha[state, 0] = value
        if value > maximum:
            maximum = value
    for state in range(state_count):
        alpha[state, 0] -= maximum

    workspace = np.empty(state_count, dtype=np.float64)
    for time_index in range(1, time_count):
        delta_z = z_relative[time_index] - z_relative[time_index - 1]
        maximum = -1.0e30
        for current in range(state_count):
            desired_previous = offsets[current] + delta_z - expected_surface_step
            center = int(np.rint((desired_previous - offsets[0]) / step))
            start = max(0, center - radius)
            stop = min(state_count, center + radius + 1)
            for previous in range(start, stop):
                surface_step = offsets[current] - offsets[previous] + delta_z
                residual = surface_step - expected_surface_step
                workspace[previous] = (
                    alpha[previous, time_index - 1]
                    + transition_scale * residual * residual
                )
            value = (
                logits[current, time_index] / temperature
                + _logsumexp_slice(workspace, start, stop)
            )
            alpha[current, time_index] = value
            if value > maximum:
                maximum = value
        for state in range(state_count):
            alpha[state, time_index] -= maximum

    for state in range(state_count):
        beta[state, time_count - 1] = 0.0
    for time_index in range(time_count - 2, -1, -1):
        delta_z = z_relative[time_index + 1] - z_relative[time_index]
        maximum = -1.0e30
        for previous in range(state_count):
            desired_current = offsets[previous] - delta_z + expected_surface_step
            center = int(np.rint((desired_current - offsets[0]) / step))
            start = max(0, center - radius)
            stop = min(state_count, center + radius + 1)
            for current in range(start, stop):
                surface_step = offsets[current] - offsets[previous] + delta_z
                residual = surface_step - expected_surface_step
                workspace[current] = (
                    logits[current, time_index + 1] / temperature
                    + beta[current, time_index + 1]
                    + transition_scale * residual * residual
                )
            value = _logsumexp_slice(workspace, start, stop)
            beta[previous, time_index] = value
            if value > maximum:
                maximum = value
        for state in range(state_count):
            beta[state, time_index] -= maximum

    mean = np.empty(time_count, dtype=np.float64)
    for time_index in range(time_count):
        maximum = -1.0e30
        for state in range(state_count):
            workspace[state] = alpha[state, time_index] + beta[state, time_index]
            if workspace[state] > maximum:
                maximum = workspace[state]
        normalizer = 0.0
        weighted = 0.0
        for state in range(state_count):
            probability = np.exp(workspace[state] - maximum)
            normalizer += probability
            weighted += probability * offsets[state]
        mean[time_index] = weighted / max(normalizer, 1.0e-30)
    return mean
