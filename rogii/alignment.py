from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Well
from .sequence import SequenceConfig, make_sequence_example


SYNTHETIC_PROFILES = (
    "legacy",
    "distractor",
    "hard",
    "forward",
    "forward_distractor",
    "forward_extreme",
)


def _shift_state_columns(values: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift every state column by a possibly different fractional class count."""
    state_count, column_count = values.shape
    source = np.arange(state_count, dtype=np.float64)[:, None] - shifts[None, :]
    lower = np.floor(source).astype(np.int64)
    fraction = source - lower
    fraction[(source < 0.0) | (source > state_count - 1)] = 0.0
    lower = np.clip(lower, 0, state_count - 1)
    upper = np.clip(lower + 1, 0, state_count - 1)
    columns = np.broadcast_to(np.arange(column_count)[None, :], values.shape)
    return (1.0 - fraction) * values[lower, columns] + fraction * values[upper, columns]


def _smooth_random_curve(
    rng: np.random.Generator, length: int, width: int
) -> np.ndarray:
    width = max(3, min(width, length if length % 2 else length - 1))
    if width < 3:
        return np.zeros(length, dtype=np.float64)
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    padded = rng.normal(size=length + width - 1)
    curve = np.convolve(padded, kernel, mode="valid")
    curve -= curve.mean()
    curve /= max(curve.std(), 1e-6)
    return curve


def _forward_synthetic_alignment(
    well: Well,
    true_offset: np.ndarray,
    sampled_z: np.ndarray,
    valid_positions: np.ndarray,
    missing: np.ndarray,
    anchor_tvt: float,
    config: SequenceConfig,
    rng: np.random.Generator,
    hardness: float,
    add_distractor: bool,
    extreme_excursion: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    """Forward-generate horizontal GR along a randomized geological path."""
    anchor = config.prefix_points - 1
    z_delta = sampled_z - sampled_z[anchor]
    real_surface = true_offset + z_delta
    length = len(true_offset)
    columns = np.arange(length, dtype=np.float64)
    distance = (columns - anchor) / max(length - anchor - 1, 1)

    surface_scale = float(rng.uniform(0.72, 1.28))
    long_curve = _smooth_random_curve(rng, length, max(17, length // 5))
    long_curve -= long_curve[anchor]
    envelope = np.minimum(1.0, np.abs(columns - anchor) / max(length // 8, 1))
    deformation_scale = (4.0 + 24.0 * hardness) * float(rng.uniform(0.35, 1.0))
    deformation = deformation_scale * envelope * long_curve
    tilt = rng.uniform(-18.0, 18.0) * hardness * distance
    synthetic_surface = surface_scale * real_surface + deformation + tilt
    synthetic_surface -= synthetic_surface[anchor]
    synthetic_offset = synthetic_surface - z_delta

    reference_valid = np.isfinite(well.typewell_tvt) & np.isfinite(well.typewell_gr)
    reference_tvt = well.typewell_tvt[reference_valid]
    reference_gr = well.typewell_gr[reference_valid]
    if len(reference_tvt) < 2:
        raise ValueError(f"{well.well_id}: insufficient finite typewell GR")
    lower = max(
        -config.state_radius + config.state_step,
        float(reference_tvt[0]) - anchor_tvt + config.state_step,
    )
    upper = min(
        config.state_radius - config.state_step,
        float(reference_tvt[-1]) - anchor_tvt - config.state_step,
    )
    if lower >= upper:
        synthetic_offset = np.clip(
            true_offset,
            -config.state_radius + config.state_step,
            config.state_radius - config.state_step,
        )
    else:
        synthetic_offset = np.clip(synthetic_offset, lower, upper)
    synthetic_offset[anchor] = 0.0
    if extreme_excursion and rng.random() < 0.45 + 0.5 * hardness:
        synthetic_offset = _amplify_endpoint_excursion(
            synthetic_offset,
            valid_positions,
            config,
            rng,
            lower,
            upper,
        )
        synthetic_offset[anchor] = 0.0

    state_tvt = anchor_tvt + config.offsets
    state_gr = np.interp(
        state_tvt,
        reference_tvt,
        reference_gr,
        left=np.nan,
        right=np.nan,
    )
    path_gr = np.interp(
        anchor_tvt + synthetic_offset,
        reference_tvt,
        reference_gr,
        left=np.nan,
        right=np.nan,
    )
    finite_state = np.isfinite(state_gr)
    supported = finite_state.astype(np.float32)
    reference_center = float(np.nanmedian(state_gr[finite_state]))
    center = float(rng.uniform(45.0, 105.0))
    gain = float(rng.uniform(0.65, 1.45))
    expected_state = center + gain * (state_gr - reference_center)
    expected_path = center + gain * (path_gr - reference_center)

    correlated = _smooth_random_curve(rng, length, max(5, length // 80))
    white = rng.normal(size=length)
    noise_scale = float(rng.uniform(2.0, 5.0) + hardness * rng.uniform(2.0, 7.0))
    observed = expected_path + noise_scale * (0.82 * correlated + 0.35 * white)
    if rng.random() < 0.8 * hardness:
        gain_drift = np.exp(
            rng.uniform(0.03, 0.14)
            * _smooth_random_curve(rng, length, max(9, length // 12))
        )
        observed = center + gain_drift * (observed - center)

    scale = max(noise_scale, 4.0)
    mismatch = (observed[None, :] - expected_state[:, None]) / scale
    mismatch = np.clip(np.nan_to_num(mismatch, nan=0.0), -6.0, 6.0)
    pseudo_noise = 0.18 * _smooth_random_curve(rng, length, max(5, length // 100))
    pseudo_mismatch = np.clip(mismatch + pseudo_noise[None, :], -6.0, 6.0)
    labels = np.argmin(
        np.abs(config.offsets[:, None] - synthetic_offset[None, :]), axis=0
    ).astype(np.int64)

    cycle_shift = 0.0
    if add_distractor and rng.random() < hardness:
        mismatch, pseudo_mismatch, cycle_shift = _add_cycle_distractor(
            mismatch,
            pseudo_mismatch,
            labels,
            valid_positions,
            config,
            rng,
        )

    missing_columns = 0
    if rng.random() < 0.35 + 0.45 * hardness:
        for _ in range(int(rng.integers(1, 3))):
            start = int(rng.integers(config.prefix_points, length))
            width = int(rng.integers(4, max(5, length // 18)))
            stop = min(start + width, length)
            mismatch[:, start:stop] = 0.0
            pseudo_mismatch[:, start:stop] = 0.0
            missing[start:stop] = 1.0
            missing_columns += stop - start

    invalid = valid_positions < 0.5
    mismatch[:, invalid] = 0.0
    pseudo_mismatch[:, invalid] = 0.0
    missing[invalid] = 1.0
    return (
        mismatch,
        pseudo_mismatch,
        synthetic_offset,
        labels,
        supported,
        observed,
        cycle_shift,
        missing_columns,
    )


def _amplify_endpoint_excursion(
    offset: np.ndarray,
    valid_positions: np.ndarray,
    config: SequenceConfig,
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Increase a realizable path's signed tail excursion without changing its prefix."""
    tail = np.flatnonzero(
        (np.arange(len(offset)) >= config.prefix_points) & (valid_positions > 0.5)
    )
    if len(tail) < 16:
        return offset
    endpoint = float(offset[tail[-1]])
    if abs(endpoint) >= 8.0:
        sign = float(np.sign(endpoint))
    else:
        farthest = float(offset[tail[np.argmax(np.abs(offset[tail]))]])
        sign = (
            float(np.sign(farthest))
            if abs(farthest) >= 8.0
            else rng.choice((-1.0, 1.0))
        )
    limit = upper if sign > 0 else -lower
    minimum = max(32.0, abs(endpoint) + 8.0)
    maximum = min(limit, 0.95 * config.state_radius)
    if maximum <= minimum:
        return offset
    target = sign * float(rng.uniform(minimum, maximum))
    first, last = int(tail[0]), int(tail[-1])
    columns = np.arange(len(offset))
    progress = np.clip((columns - first + 1) / max(last - first + 1, 1), 0.0, 1.0)
    curve = (target - endpoint) * np.power(progress, rng.uniform(0.7, 1.35))
    curve[: config.prefix_points] = 0.0
    return np.clip(offset + curve, lower, upper)


def _synthetic_target_drift(
    true_offset: np.ndarray,
    valid_positions: np.ndarray,
    config: SequenceConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Generate a smooth signed tail excursion that remains inside the state grid."""
    tail = np.flatnonzero(
        (np.arange(len(true_offset)) >= config.prefix_points) & (valid_positions > 0.5)
    )
    drift = np.zeros_like(true_offset, dtype=np.float64)
    if len(tail) < 16:
        return drift, 0.0

    margin = max(2.0 * config.state_step, 1.0)
    observed = true_offset[tail]
    positive_room = config.state_radius - margin - float(np.max(observed))
    negative_room = float(np.min(observed)) + config.state_radius - margin
    minimum = max(12.0, 8.0 * config.state_step)
    choices = []
    if positive_room >= minimum:
        choices.append((1.0, positive_room))
    if negative_room >= minimum:
        choices.append((-1.0, negative_room))
    if not choices:
        return drift, 0.0

    sign, room = choices[int(rng.integers(len(choices)))]
    maximum = min(room, 0.72 * config.state_radius)
    amplitude = sign * float(rng.uniform(minimum, maximum))
    first, last = int(tail[0]), int(tail[-1])
    columns = np.arange(len(true_offset))
    progress = np.clip((columns - first + 1) / max(last - first + 1, 1), 0.0, 1.0)
    power = float(rng.uniform(0.7, 1.5))
    curve = amplitude * np.power(progress, power)
    bend = rng.uniform(-0.18, 0.18) * abs(amplitude) * np.sin(np.pi * progress)
    curve += bend
    curve[: config.prefix_points] = 0.0
    lower = -config.state_radius + margin
    upper = config.state_radius - margin
    shifted = np.clip(true_offset + curve, lower, upper)
    drift = shifted - true_offset
    drift[: config.prefix_points] = 0.0
    return drift, float(drift[last])


def _add_cycle_distractor(
    mismatch: np.ndarray,
    pseudo_mismatch: np.ndarray,
    labels: np.ndarray,
    valid_positions: np.ndarray,
    config: SequenceConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Copy the true correlation band to a long, wrong parallel alignment mode."""
    tail = np.flatnonzero(
        (np.arange(len(labels)) >= config.prefix_points) & (valid_positions > 0.5)
    )
    if len(tail) < 24:
        return mismatch, pseudo_mismatch, 0.0
    start_position = int(rng.integers(0, max(1, len(tail) // 3)))
    stop_position = int(
        rng.integers(
            max(start_position + 16, 2 * len(tail) // 3),
            len(tail) + 1,
        )
    )
    columns = tail[start_position:stop_position]
    minimum_classes = max(2, int(np.ceil(12.0 / config.state_step)))
    maximum_classes = max(
        minimum_classes,
        int(np.floor(min(48.0, 0.75 * config.state_radius) / config.state_step)),
    )
    magnitude = int(rng.integers(minimum_classes, maximum_classes + 1))
    candidate_shifts = [magnitude, -magnitude]
    rng.shuffle(candidate_shifts)
    viable = []
    for shift in candidate_shifts:
        destination = labels[columns] + shift
        fraction = np.mean((destination >= 0) & (destination < mismatch.shape[0]))
        if fraction >= 0.6:
            viable.append(shift)
    if not viable:
        return mismatch, pseudo_mismatch, 0.0
    shift = viable[int(rng.integers(len(viable)))]

    source_mismatch = mismatch.copy()
    source_pseudo = pseudo_mismatch.copy()
    strength = float(rng.uniform(0.72, 0.97))
    half_width = max(2, int(np.ceil(rng.uniform(2.0, 5.0) / config.state_step)))
    for delta in range(-half_width, half_width + 1):
        source = labels[columns] + delta
        destination = source + shift
        keep = (
            (source >= 0)
            & (source < mismatch.shape[0])
            & (destination >= 0)
            & (destination < mismatch.shape[0])
        )
        selected_columns = columns[keep]
        source = source[keep]
        destination = destination[keep]
        mismatch[destination, selected_columns] = (
            strength * source_mismatch[source, selected_columns]
            + (1.0 - strength) * mismatch[destination, selected_columns]
        )
        pseudo_mismatch[destination, selected_columns] = (
            strength * source_pseudo[source, selected_columns]
            + (1.0 - strength) * pseudo_mismatch[destination, selected_columns]
        )
    return mismatch, pseudo_mismatch, float(shift * config.state_step)


def expected_offset(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Decode categorical offset logits to their posterior mean in feet."""
    return torch.sum(torch.softmax(logits, dim=1) * offsets[None, :, None], dim=1)


def alignment_objective(
    logits: torch.Tensor,
    image: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    prefix_points: int,
    objective_kind: str = "categorical",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Production training objective, shared with the batch-overfit diagnostic."""
    tail = slice(prefix_points, None)
    tail_valid = valid[:, tail]
    if objective_kind not in ("categorical", "ordinal"):
        raise ValueError(f"unknown alignment objective: {objective_kind}")
    point_classification = F.cross_entropy(
        logits[:, :, tail], label[:, tail], reduction="none"
    )
    classification = (
        point_classification * tail_valid
    ).sum() / tail_valid.sum().clamp_min(1.0)
    prediction = expected_offset(logits[:, :, tail].float(), offsets)
    truth = offsets[label[:, tail]]
    point_regression = torch.square((prediction - truth) / 40.0)
    regression = (point_regression * tail_valid).sum() / tail_valid.sum().clamp_min(1.0)
    z_relative = image[:, 9, 0, tail].float() * 200.0
    surface = prediction + z_relative
    smooth_valid = tail_valid[:, 2:] * tail_valid[:, 1:-1] * tail_valid[:, :-2]
    point_smoothness = torch.abs(torch.diff(surface, n=2, dim=1))
    surface_smoothness = (
        point_smoothness * smooth_valid
    ).sum() / smooth_valid.sum().clamp_min(1.0)
    total = classification + regression + 1e-3 * surface_smoothness
    if objective_kind == "ordinal":
        probability = torch.softmax(logits[:, :, tail].float(), dim=1)
        predicted_cdf = torch.cumsum(probability, dim=1)
        state_index = torch.arange(
            logits.shape[1], device=logits.device
        )[None, :, None]
        target_cdf = (state_index >= label[:, None, tail]).float()
        point_ordinal = torch.mean(
            torch.square(predicted_cdf - target_cdf), dim=1
        )
        ordinal = (
            point_ordinal * tail_valid
        ).sum() / tail_valid.sum().clamp_min(1.0)
        total = total + 5.0 * ordinal
    components = {
        "classification": classification,
        "regression": regression,
        "surface_smoothness": surface_smoothness,
    }
    return total, components


def make_alignment_example(
    well: Well,
    cut: int,
    config: SequenceConfig,
    synthetic_rng: np.random.Generator | None = None,
    synthetic_profile: str = "legacy",
    synthetic_hardness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    if synthetic_profile not in SYNTHETIC_PROFILES:
        raise ValueError(
            f"unknown synthetic profile {synthetic_profile!r}; "
            f"expected one of {SYNTHETIC_PROFILES}"
        )
    if not 0.0 <= synthetic_hardness <= 1.0:
        raise ValueError("synthetic hardness must be between zero and one")
    sequence, target, metadata = make_sequence_example(well, cut, config)
    n_states = len(config.offsets)
    mismatch = sequence[:n_states]
    visible = sequence[n_states + 2]
    missing = sequence[n_states + 1]
    z_delta = sequence[n_states + 7]
    sampled_gr = np.asarray(metadata["gr"])
    sampled_md = np.asarray(metadata["md"])
    sampled_x = np.asarray(metadata["x"])
    sampled_y = np.asarray(metadata["y"])
    sampled_z = np.asarray(metadata["z"])
    valid_positions = np.asarray(metadata["valid_positions"])
    missing = np.maximum(missing, 1.0 - valid_positions)

    prefix_tvt = well.tvt[: cut + 1]
    prefix_gr = well.gr[: cut + 1]
    finite_prefix = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    prefix_tvt = prefix_tvt[finite_prefix]
    prefix_gr = prefix_gr[finite_prefix]
    distance_to_prefix = (
        (well.tvt[cut] + config.offsets)[:, None] - prefix_tvt[None, :]
    ) / 0.5
    pseudo_weight = np.exp(-0.5 * np.square(distance_to_prefix))
    pseudo_support = pseudo_weight.sum(axis=1)
    pseudo_signal = (pseudo_weight @ prefix_gr) / np.maximum(pseudo_support, 1e-12)
    pseudo_mismatch = (sampled_gr[None, :] - pseudo_signal[:, None]) / 10.0
    supported = (pseudo_support > 0.05).astype(np.float32)
    pseudo_mismatch = np.where(supported[:, None] > 0, pseudo_mismatch, mismatch)
    pseudo_mismatch = np.nan_to_num(pseudo_mismatch, nan=0.0)
    pseudo_mismatch = np.clip(pseudo_mismatch, -6.0, 6.0)

    true_offset = target * config.target_scale
    distance = config.offsets[:, None] - true_offset[None, :]
    labels = np.argmin(np.abs(distance), axis=0).astype(np.int64)
    synthetic_drift_amplitude = 0.0
    synthetic_cycle_shift = 0.0
    synthetic_missing_columns = 0
    if synthetic_rng is not None:
        forward_profile = synthetic_profile in (
            "forward",
            "forward_distractor",
            "forward_extreme",
        )
        if forward_profile:
            real_offset = true_offset.copy()
            (
                mismatch,
                pseudo_mismatch,
                true_offset,
                labels,
                supported,
                sampled_gr,
                synthetic_cycle_shift,
                synthetic_missing_columns,
            ) = _forward_synthetic_alignment(
                well,
                true_offset,
                sampled_z,
                valid_positions,
                missing,
                float(metadata["anchor_tvt"]),
                config,
                synthetic_rng,
                synthetic_hardness,
                synthetic_profile == "forward_distractor",
                synthetic_profile == "forward_extreme",
            )
            tail = np.flatnonzero(
                (np.arange(len(true_offset)) >= config.prefix_points)
                & (valid_positions > 0.5)
            )
            if len(tail):
                synthetic_drift_amplitude = float(
                    true_offset[tail[-1]] - real_offset[tail[-1]]
                )
            metadata["gr"] = sampled_gr.astype(np.float32)
        else:
            column = np.arange(mismatch.shape[1])
            mismatch = mismatch - mismatch[labels, column][None, :]
        if (
            synthetic_profile == "hard"
            and synthetic_rng.random() < 0.75 * synthetic_hardness
        ):
            drift, synthetic_drift_amplitude = _synthetic_target_drift(
                true_offset, valid_positions, config, synthetic_rng
            )
            state_shift = drift / config.state_step
            mismatch = _shift_state_columns(mismatch, state_shift)
            pseudo_mismatch = _shift_state_columns(pseudo_mismatch, state_shift)
            true_offset = true_offset + drift
            distance = config.offsets[:, None] - true_offset[None, :]
            labels = np.argmin(np.abs(distance), axis=0).astype(np.int64)
        if not forward_profile:
            mismatch += synthetic_rng.normal(0.0, 0.18, mismatch.shape[1])[None, :]
        if (
            synthetic_profile in ("distractor", "hard")
            and synthetic_rng.random() < synthetic_hardness
        ):
            mismatch, pseudo_mismatch, synthetic_cycle_shift = _add_cycle_distractor(
                mismatch,
                pseudo_mismatch,
                labels,
                valid_positions,
                config,
                synthetic_rng,
            )
        if (
            synthetic_profile == "hard"
            and synthetic_rng.random() < 0.8 * synthetic_hardness
        ):
            length = mismatch.shape[1]
            gain_curve = np.exp(
                synthetic_rng.uniform(0.08, 0.25)
                * _smooth_random_curve(synthetic_rng, length, max(9, length // 10))
            )
            baseline_curve = synthetic_rng.uniform(0.08, 0.45) * _smooth_random_curve(
                synthetic_rng, length, max(9, length // 8)
            )
            mismatch = mismatch * gain_curve[None, :] + baseline_curve[None, :]
            pseudo_mismatch = (
                pseudo_mismatch * gain_curve[None, :] + 0.7 * baseline_curve[None, :]
            )
        if not forward_profile and synthetic_rng.random() < 0.5:
            start = int(synthetic_rng.integers(config.prefix_points, mismatch.shape[1]))
            width = int(synthetic_rng.integers(8, max(9, mismatch.shape[1] // 8)))
            mismatch[:, start : start + width] = 0.0
            if synthetic_profile == "hard":
                pseudo_mismatch[:, start : start + width] = 0.0
                missing[start : start + width] = 1.0
                synthetic_missing_columns += min(width, mismatch.shape[1] - start)
        if (
            synthetic_profile == "hard"
            and synthetic_rng.random() < 0.65 * synthetic_hardness
        ):
            for _ in range(int(synthetic_rng.integers(1, 4))):
                start = int(
                    synthetic_rng.integers(config.prefix_points, mismatch.shape[1])
                )
                width = int(synthetic_rng.integers(6, max(7, mismatch.shape[1] // 12)))
                stop = min(start + width, mismatch.shape[1])
                mismatch[:, start:stop] = 0.0
                pseudo_mismatch[:, start:stop] = 0.0
                missing[start:stop] = 1.0
                synthetic_missing_columns += stop - start
    distance = config.offsets[:, None] - true_offset[None, :]
    visible_path = np.exp(-0.5 * np.square(distance / 3.0))
    visible_path[:, visible < 0.5] = 0.0

    state_coordinate = np.broadcast_to(
        (config.offsets / config.state_radius)[:, None], mismatch.shape
    )
    dmd = np.gradient(sampled_md)
    safe_dmd = np.where(np.abs(dmd) < 1e-6, 1.0, dmd)
    dx_dmd = np.gradient(sampled_x) / safe_dmd
    dy_dmd = np.gradient(sampled_y) / safe_dmd
    dz_dmd = np.gradient(sampled_z) / safe_dmd
    horizontal = np.sqrt(np.square(dx_dmd) + np.square(dy_dmd))
    azimuth_cos = dx_dmd / np.maximum(horizontal, 1e-4)
    azimuth_sin = dy_dmd / np.maximum(horizontal, 1e-4)
    curvature_z = np.gradient(dz_dmd) / safe_dmd

    def row_channel(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[None, :], mismatch.shape)

    anchor = config.prefix_points - 1
    image = np.stack(
        [
            mismatch,
            np.abs(mismatch),
            pseudo_mismatch,
            np.abs(pseudo_mismatch),
            np.broadcast_to(supported[:, None], mismatch.shape),
            state_coordinate,
            visible_path,
            np.broadcast_to(visible[None, :], mismatch.shape),
            np.broadcast_to(missing[None, :], mismatch.shape),
            np.broadcast_to(z_delta[None, :], mismatch.shape),
            row_channel((sampled_md - sampled_md[anchor]) / 5000.0),
            row_channel((sampled_x - sampled_x[anchor]) / 5000.0),
            row_channel((sampled_y - sampled_y[anchor]) / 5000.0),
            row_channel(np.clip(dz_dmd, -1.5, 1.5)),
            row_channel(np.clip(dx_dmd, -1.5, 1.5)),
            row_channel(np.clip(dy_dmd, -1.5, 1.5)),
            row_channel(np.nan_to_num(azimuth_cos)),
            row_channel(np.nan_to_num(azimuth_sin)),
            row_channel(np.clip(curvature_z * 100.0, -2.0, 2.0)),
        ],
        axis=0,
    ).astype(np.float32)
    metadata["synthetic_drift_amplitude"] = synthetic_drift_amplitude
    metadata["synthetic_cycle_shift"] = synthetic_cycle_shift
    metadata["synthetic_missing_columns"] = synthetic_missing_columns
    return image, labels, metadata


class AlignmentBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, (3, 7), padding=(1, 3))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 7), padding=(1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class AlignmentUNet(nn.Module):
    def __init__(self, input_channels: int = 19, base: int = 12) -> None:
        super().__init__()
        # Five reductions give every output column essentially full-well context
        # at the default 960-column representation.
        widths = [base, base * 2, base * 4, base * 6, base * 8, base * 8]
        self.stem = nn.Conv2d(input_channels, widths[0], (5, 9), padding=(2, 4))
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:]):
            self.encoder.append(
                nn.Sequential(AlignmentBlock(left), AlignmentBlock(left))
            )
            self.down.append(nn.Conv2d(left, right, 4, stride=2, padding=1))
        self.middle = nn.Sequential(
            AlignmentBlock(widths[-1]), AlignmentBlock(widths[-1])
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(nn.ConvTranspose2d(left, right, 4, stride=2, padding=1))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(right * 2, right, 3, padding=1),
                    AlignmentBlock(right),
                )
            )
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for block, down in zip(self.encoder, self.down):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)
