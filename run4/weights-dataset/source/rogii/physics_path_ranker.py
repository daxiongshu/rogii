from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from .data import Well


DATUM_COEFFICIENTS = (-12.0, -6.0, 0.0, 6.0, 12.0)
SLOPE_COEFFICIENTS = (-16.0, -8.0, 0.0, 8.0, 16.0)
BEND_COEFFICIENTS = (-8.0, 0.0, 8.0)
SMOOTH_WINDOWS = (1, 9, 31, 91)
SEQUENCE_CHANNELS = (
    "observed_gr",
    "simulated_gr",
    "calibrated_mismatch",
    "observed_gr_gradient",
    "simulated_gr_gradient",
    "candidate_surface_correction",
    "trajectory_z_gradient",
)


def candidate_coefficients() -> np.ndarray:
    return np.asarray(
        list(
            itertools.product(
                DATUM_COEFFICIENTS,
                SLOPE_COEFFICIENTS,
                BEND_COEFFICIENTS,
            )
        ),
        dtype=np.float32,
    )


def _corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - np.mean(left, axis=-1, keepdims=True)
    right = right - np.mean(right, axis=-1, keepdims=True)
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(left), axis=-1)
        * np.sum(np.square(right), axis=-1)
    )
    return numerator / np.maximum(denominator, 1e-6)


def _robust_affine(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(reference) & np.isfinite(observed)
    if valid.sum() < 20:
        center = float(np.nanmedian(observed)) if np.any(np.isfinite(observed)) else 0.0
        scale = float(np.nanstd(observed)) if np.any(np.isfinite(observed)) else 1.0
        return 1.0, center, max(scale, 3.0)
    x = reference[valid].astype(np.float64)
    y = observed[valid].astype(np.float64)
    design = np.column_stack((x, np.ones(len(x))))
    weights = np.ones(len(x), dtype=np.float64)
    coefficient = np.asarray((1.0, 0.0), dtype=np.float64)
    for _ in range(5):
        weighted = design * np.sqrt(weights[:, None])
        target = y * np.sqrt(weights)
        coefficient = np.linalg.lstsq(weighted, target, rcond=None)[0]
        residual = y - design @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weights = 1.0 / (1.0 + np.square(residual / (2.5 * scale)))
    residual = y - design @ coefficient
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
    return float(coefficient[0]), float(coefficient[1]), max(float(scale), 3.0)


def _finite_interp(
    query: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(source) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros_like(query, dtype=np.float64)
    return np.interp(query, source[valid], values[valid])


def _candidate_basis(md: np.ndarray, anchor_md: float) -> np.ndarray:
    distance = np.maximum(md - anchor_md, 0.0)
    span = max(float(distance[-1]), 1.0)
    progress = np.clip(distance / span, 0.0, 1.0)
    return np.stack(
        (
            1.0 - np.exp(-distance / 100.0),
            progress,
            4.0 * progress * (1.0 - progress),
        ),
        axis=0,
    )


def candidate_corrections(
    md: np.ndarray,
    anchor_md: float,
    coefficients: np.ndarray | None = None,
) -> np.ndarray:
    if coefficients is None:
        coefficients = candidate_coefficients()
    correction = coefficients.astype(np.float64) @ _candidate_basis(md, anchor_md)
    return np.clip(correction, -32.0, 32.0)


def _reverse_features(
    candidate_tvt: np.ndarray,
    observed: np.ndarray,
    well: Well,
    gain: float,
    intercept: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rmse = np.full(len(candidate_tvt), 6.0, dtype=np.float64)
    correlation = np.zeros(len(candidate_tvt), dtype=np.float64)
    typewell_tvt = well.typewell_tvt
    expected = gain * well.typewell_gr + intercept
    valid_typewell = np.isfinite(typewell_tvt) & np.isfinite(expected)
    for index, path in enumerate(candidate_tvt):
        valid = np.isfinite(path) & np.isfinite(observed)
        if valid.sum() < 20:
            continue
        order = np.argsort(path[valid])
        path_ordered = path[valid][order]
        observed_ordered = observed[valid][order]
        unique, unique_index = np.unique(path_ordered, return_index=True)
        observed_unique = observed_ordered[unique_index]
        overlap = (
            valid_typewell
            & (typewell_tvt >= unique[0])
            & (typewell_tvt <= unique[-1])
        )
        if overlap.sum() < 20:
            continue
        projected = np.interp(typewell_tvt[overlap], unique, observed_unique)
        target = expected[overlap]
        difference = (projected - target) / scale
        rmse[index] = np.sqrt(np.mean(np.square(difference)))
        correlation[index] = _corr(projected[None], target[None])[0]
    return np.clip(rmse, 0.0, 6.0), np.clip(correlation, -1.0, 1.0)


def engineered_feature_names() -> tuple[str, ...]:
    names = ["datum", "slope", "bend", "coefficient_energy"]
    for window in SMOOTH_WINDOWS:
        names.extend(
            (
                f"mismatch_rmse_w{window}",
                f"mismatch_mae_w{window}",
                f"mismatch_p90_w{window}",
                f"correlation_w{window}",
                f"gradient_rmse_w{window}",
            )
        )
    for segment in range(4):
        names.extend((f"segment{segment}_rmse", f"segment{segment}_correlation"))
    names.extend(
        (
            "reverse_rmse",
            "reverse_correlation",
            "maximum_abs_correction",
            "end_correction",
            "correction_curvature",
            "typewell_support_fraction",
        )
    )
    return tuple(names)


@dataclass(frozen=True)
class PathInputs:
    sequences: np.ndarray
    engineered: np.ndarray
    coefficients: np.ndarray
    sample_md: np.ndarray
    sample_baseline_tvt: np.ndarray
    sample_corrections: np.ndarray


def make_path_inputs(
    well: Well,
    baseline_tvt: np.ndarray,
    sequence_points: int = 256,
) -> PathInputs:
    prediction_rows = well.prediction_indices
    if len(prediction_rows) != len(baseline_tvt):
        raise ValueError(f"{well.well_id}: baseline row count mismatch")
    coefficients = candidate_coefficients()
    sample_rows = np.linspace(
        float(prediction_rows[0]),
        float(prediction_rows[-1]),
        sequence_points,
    )
    row_axis = np.arange(len(well.md), dtype=np.float64)
    sample_md = np.interp(sample_rows, row_axis, well.md)
    sample_z = np.interp(sample_rows, row_axis, well.z)
    sample_observed = _finite_interp(sample_rows, row_axis, well.gr)
    sample_baseline = np.interp(
        sample_rows,
        prediction_rows.astype(np.float64),
        baseline_tvt,
    )
    corrections = candidate_corrections(
        sample_md,
        float(well.md[well.anchor_index]),
        coefficients,
    )
    candidate_tvt = sample_baseline[None] + corrections

    known = well.known_indices[-min(640, len(well.known_indices)) :]
    reference = np.interp(
        well.tvt_input[known],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, intercept, residual_scale = _robust_affine(reference, well.gr[known])
    simulated = np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    simulated = gain * simulated + intercept
    observed_center = float(np.nanmedian(sample_observed))
    observed_scale = max(
        float(
            1.4826
            * np.nanmedian(np.abs(sample_observed - observed_center))
        ),
        residual_scale,
        3.0,
    )
    observed = np.nan_to_num(
        (sample_observed - observed_center) / observed_scale,
        nan=0.0,
    )
    simulated_normalized = np.nan_to_num(
        (simulated - observed_center) / observed_scale,
        nan=0.0,
    )
    mismatch = observed[None] - simulated_normalized
    observed_gradient = np.gradient(observed)
    simulated_gradient = np.gradient(simulated_normalized, axis=1)
    z_gradient = np.gradient(sample_z)
    z_gradient /= max(float(np.nanstd(z_gradient)), 0.25)
    sequences = np.stack(
        (
            np.broadcast_to(observed[None], simulated.shape),
            simulated_normalized,
            mismatch,
            np.broadcast_to(observed_gradient[None], simulated.shape),
            simulated_gradient,
            corrections / 32.0,
            np.broadcast_to(z_gradient[None], simulated.shape),
        ),
        axis=1,
    )
    sequences = np.clip(np.nan_to_num(sequences), -8.0, 8.0).astype(np.float32)

    feature_columns = [
        coefficients[:, 0] / 12.0,
        coefficients[:, 1] / 16.0,
        coefficients[:, 2] / 8.0,
        (
            np.square(coefficients[:, 0] / 12.0)
            + np.square(coefficients[:, 1] / 16.0)
            + np.square(coefficients[:, 2] / 8.0)
        ),
    ]
    for window in SMOOTH_WINDOWS:
        if window == 1:
            observed_smoothed = observed
            simulated_smoothed = simulated_normalized
        else:
            sigma = window / 6.0
            observed_smoothed = gaussian_filter1d(observed, sigma, mode="nearest")
            simulated_smoothed = gaussian_filter1d(
                simulated_normalized,
                sigma,
                axis=1,
                mode="nearest",
            )
        difference = observed_smoothed[None] - simulated_smoothed
        feature_columns.extend(
            (
                np.sqrt(np.mean(np.square(difference), axis=1)),
                np.mean(np.abs(difference), axis=1),
                np.quantile(np.abs(difference), 0.9, axis=1),
                _corr(
                    np.broadcast_to(observed_smoothed[None], simulated.shape),
                    simulated_smoothed,
                ),
                np.sqrt(
                    np.mean(
                        np.square(
                            np.gradient(observed_smoothed)[None]
                            - np.gradient(simulated_smoothed, axis=1)
                        ),
                        axis=1,
                    )
                ),
            )
        )
    for segment in range(4):
        left = segment * sequence_points // 4
        right = (segment + 1) * sequence_points // 4
        difference = mismatch[:, left:right]
        feature_columns.extend(
            (
                np.sqrt(np.mean(np.square(difference), axis=1)),
                _corr(
                    np.broadcast_to(
                        observed[None, left:right],
                        (len(coefficients), right - left),
                    ),
                    simulated_normalized[:, left:right],
                ),
            )
        )
    reverse_rmse, reverse_correlation = _reverse_features(
        candidate_tvt,
        sample_observed,
        well,
        gain,
        intercept,
        observed_scale,
    )
    support = (
        (candidate_tvt >= np.nanmin(well.typewell_tvt))
        & (candidate_tvt <= np.nanmax(well.typewell_tvt))
    )
    feature_columns.extend(
        (
            reverse_rmse,
            reverse_correlation,
            np.max(np.abs(corrections), axis=1) / 32.0,
            corrections[:, -1] / 32.0,
            np.mean(np.abs(np.diff(corrections, n=2, axis=1)), axis=1),
            np.mean(support, axis=1),
        )
    )
    engineered = np.stack(feature_columns, axis=1).astype(np.float32)
    if engineered.shape[1] != len(engineered_feature_names()):
        raise RuntimeError("engineered path feature contract changed")
    if not np.all(np.isfinite(engineered)):
        raise RuntimeError(f"{well.well_id}: nonfinite path features")
    return PathInputs(
        sequences=sequences,
        engineered=engineered,
        coefficients=coefficients,
        sample_md=sample_md,
        sample_baseline_tvt=sample_baseline,
        sample_corrections=corrections.astype(np.float32),
    )


def analytic_energy(engineered: np.ndarray, prior_strength: float) -> np.ndarray:
    names = engineered_feature_names()
    index = {name: offset for offset, name in enumerate(names)}
    forward = (
        0.15 * engineered[:, index["mismatch_rmse_w9"]]
        + 0.35 * engineered[:, index["mismatch_rmse_w31"]]
        + 0.25 * engineered[:, index["mismatch_rmse_w91"]]
        + 0.15 * engineered[:, index["reverse_rmse"]]
        + 0.10 * (1.0 - engineered[:, index["correlation_w31"]])
    )
    return forward + prior_strength * engineered[:, index["coefficient_energy"]]


def score_weights(
    score: np.ndarray,
    temperature: float,
    decoder: str,
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if decoder in {"argmin", "argmax"}:
        result = np.zeros(len(score), dtype=np.float64)
        result[int(np.argmin(score))] = 1.0
        return result
    center = np.median(score)
    scale = 1.4826 * np.median(np.abs(score - center)) + 1e-6
    log_weight = -(score - np.min(score)) / (scale * temperature)
    if decoder == "posterior_top3_mean":
        keep = np.argsort(score)[:3]
        mask = np.full(len(score), -np.inf, dtype=np.float64)
        mask[keep] = log_weight[keep]
        log_weight = mask
    elif decoder != "posterior_mean":
        raise ValueError(f"unknown path decoder: {decoder}")
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight)
    return weight / np.sum(weight)


class ConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                width,
                width,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=width,
            ),
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
            nn.Conv1d(width, width, 1),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class PhysicsPathConvRanker(nn.Module):
    def __init__(
        self,
        width: int,
        dilations: tuple[int, ...],
        dropout: float,
        engineered_features: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(len(SEQUENCE_CHANNELS), width, 7, padding=3)
        self.blocks = nn.Sequential(
            *(ConvBlock(width, dilation, dropout) for dilation in dilations)
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2 + engineered_features, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        engineered: torch.Tensor,
    ) -> torch.Tensor:
        shape = sequence.shape
        sequence = sequence.reshape(-1, shape[-2], shape[-1])
        encoded = self.blocks(self.stem(sequence))
        pooled = torch.cat(
            (torch.mean(encoded, dim=-1), torch.amax(encoded, dim=-1)),
            dim=1,
        )
        score = self.head(
            torch.cat((pooled, engineered.reshape(-1, engineered.shape[-1])), dim=1)
        )
        return score.reshape(shape[0], shape[1])
