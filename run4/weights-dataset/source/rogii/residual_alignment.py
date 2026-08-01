from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Well
from .sequence import SequenceConfig, _robust_affine, make_sequence_example


RESIDUAL_INPUT_CHANNELS = 20


@dataclass(frozen=True)
class CoarsePrediction:
    tail: np.ndarray
    components: np.ndarray


def load_oof_cache(cache_root: Path) -> dict[str, CoarsePrediction]:
    predictions: dict[str, CoarsePrediction] = {}
    for path in sorted(cache_root.glob("fold*.npz")):
        with np.load(path) as cached:
            starts = cached["row_starts"]
            well_ids = cached["well_ids"]
            coarse = cached["coarse"].astype(np.float64)
            components = cached["components"].astype(np.float64)
        for index, well_id in enumerate(well_ids):
            left, right = int(starts[index]), int(starts[index + 1])
            key = str(well_id)
            if key in predictions:
                raise ValueError(f"duplicate cached prediction for {key}")
            predictions[key] = CoarsePrediction(
                tail=coarse[left:right],
                components=components[:, left:right],
            )
    return predictions


def full_coarse_arrays(
    well: Well, prediction: CoarsePrediction
) -> tuple[np.ndarray, np.ndarray]:
    tail = well.tail_indices
    if prediction.tail.shape != tail.shape:
        raise ValueError(
            f"{well.well_id}: cached rows {len(prediction.tail)} != tail {len(tail)}"
        )
    coarse = np.asarray(well.tvt_input, dtype=np.float64).copy()
    coarse[tail] = prediction.tail
    if not np.all(np.isfinite(coarse)):
        raise ValueError(f"{well.well_id}: incomplete coarse path")
    spread = np.zeros_like(coarse)
    spread[tail] = np.std(prediction.components, axis=0)
    return coarse, spread


def _smooth_noise(rng: np.random.Generator, length: int) -> np.ndarray:
    width = max(5, min(81, length // 8))
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return np.zeros(length, dtype=np.float64)
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    noise = np.convolve(rng.normal(size=length + width - 1), kernel, mode="valid")
    noise -= noise[0]
    scale = np.std(noise)
    return noise / max(scale, 1e-6)


def make_residual_alignment_example(
    well: Well,
    coarse_path: np.ndarray,
    component_spread: np.ndarray,
    config: SequenceConfig,
    rng: np.random.Generator | None = None,
    augmentation_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    cut = well.anchor_index
    _, _, metadata = make_sequence_example(well, cut, config)
    positions = np.asarray(metadata["positions"], dtype=np.float64)
    source = np.arange(len(well.md), dtype=np.float64)

    def sample(values: np.ndarray) -> np.ndarray:
        return np.interp(positions, source, np.asarray(values, dtype=np.float64))

    sampled_md = np.asarray(metadata["md"], dtype=np.float64)
    sampled_x = np.asarray(metadata["x"], dtype=np.float64)
    sampled_y = np.asarray(metadata["y"], dtype=np.float64)
    sampled_z = np.asarray(metadata["z"], dtype=np.float64)
    sampled_gr = np.asarray(metadata["gr"], dtype=np.float64)
    true_tvt = sample(well.tvt)
    coarse_tvt = sample(coarse_path)
    sampled_spread = sample(component_spread)
    visible = np.zeros(config.length, dtype=np.float32)
    visible[: config.prefix_points] = 1.0
    coarse_tvt[: config.prefix_points] = true_tvt[: config.prefix_points]
    sampled_spread[: config.prefix_points] = 0.0

    if rng is not None and augmentation_scale > 0.0:
        tail_length = config.tail_points
        progress = np.linspace(0.0, 1.0, tail_length)
        trend = rng.normal() * progress
        curve = 0.55 * trend + 0.45 * _smooth_noise(rng, tail_length) * progress
        perturbation = augmentation_scale * curve
        coarse_tvt[config.prefix_points :] += perturbation

    prefix_rows = np.arange(cut + 1)
    reference_prefix = np.interp(
        well.tvt[prefix_rows],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, offset, residual_scale = _robust_affine(
        reference_prefix, well.gr[prefix_rows]
    )
    candidate_tvt = coarse_tvt[None, :] + config.offsets[:, None]
    typewell_gr = np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    expected_gr = gain * typewell_gr + offset
    mismatch = (sampled_gr[None, :] - expected_gr) / max(residual_scale, 3.0)
    mismatch = np.clip(
        np.nan_to_num(mismatch, nan=0.0, posinf=0.0, neginf=0.0), -6.0, 6.0
    )

    prefix_tvt = np.asarray(well.tvt[: cut + 1], dtype=np.float64)
    prefix_gr = np.asarray(well.gr[: cut + 1], dtype=np.float64)
    finite = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    order = np.argsort(prefix_tvt[finite])
    prefix_tvt = prefix_tvt[finite][order]
    prefix_gr = prefix_gr[finite][order]
    prefix_tvt, unique = np.unique(prefix_tvt, return_index=True)
    prefix_gr = prefix_gr[unique]
    pseudo_gr = np.interp(
        candidate_tvt.ravel(),
        prefix_tvt,
        prefix_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    pseudo_mismatch = (sampled_gr[None, :] - pseudo_gr) / 10.0
    pseudo_supported = np.isfinite(pseudo_gr)
    pseudo_mismatch = np.where(pseudo_supported, pseudo_mismatch, mismatch)
    pseudo_mismatch = np.clip(
        np.nan_to_num(pseudo_mismatch, nan=0.0, posinf=0.0, neginf=0.0),
        -6.0,
        6.0,
    )

    supported = np.isfinite(typewell_gr).astype(np.float32)
    true_residual = true_tvt - coarse_tvt
    labels = np.argmin(
        np.abs(config.offsets[:, None] - true_residual[None, :]), axis=0
    ).astype(np.int64)
    distance = config.offsets[:, None] - true_residual[None, :]
    visible_path = np.exp(-0.5 * np.square(distance / 3.0))
    visible_path[:, visible < 0.5] = 0.0
    state_coordinate = np.broadcast_to(
        (config.offsets / config.state_radius)[:, None], mismatch.shape
    )

    valid_positions = np.asarray(metadata["valid_positions"], dtype=np.float32)
    missing = np.maximum((~np.isfinite(sampled_gr)).astype(np.float32), 1.0 - valid_positions)
    z_delta = (sampled_z - sampled_z[config.prefix_points - 1]) / 200.0
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
            supported,
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
            row_channel(np.clip(sampled_spread / 20.0, 0.0, 3.0)),
        ],
        axis=0,
    ).astype(np.float32)
    metadata["coarse_tvt"] = coarse_tvt.astype(np.float32)
    metadata["sampled_z"] = sampled_z.astype(np.float32)
    metadata["true_residual"] = true_residual.astype(np.float32)
    return image, labels, metadata


def residual_alignment_objective(
    logits: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    coarse_tvt: torch.Tensor,
    sampled_z: torch.Tensor,
    prefix_points: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tail = slice(prefix_points, None)
    tail_valid = valid[:, tail]
    point_classification = F.cross_entropy(
        logits[:, :, tail], label[:, tail], reduction="none"
    )
    classification = (point_classification * tail_valid).sum() / tail_valid.sum().clamp_min(1.0)
    prediction = torch.sum(
        torch.softmax(logits[:, :, tail].float(), dim=1) * offsets[None, :, None],
        dim=1,
    )
    truth = offsets[label[:, tail]]
    point_regression = torch.square((prediction - truth) / 16.0)
    regression = (point_regression * tail_valid).sum() / tail_valid.sum().clamp_min(1.0)
    corrected_surface = coarse_tvt[:, tail] + prediction + sampled_z[:, tail]
    smooth_valid = tail_valid[:, 2:] * tail_valid[:, 1:-1] * tail_valid[:, :-2]
    point_smoothness = torch.abs(torch.diff(corrected_surface, n=2, dim=1))
    surface_smoothness = (point_smoothness * smooth_valid).sum() / smooth_valid.sum().clamp_min(1.0)
    total = classification + regression + 1e-3 * surface_smoothness
    return total, {
        "classification": classification,
        "regression": regression,
        "surface_smoothness": surface_smoothness,
    }
