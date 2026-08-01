from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .alignment import make_alignment_example
from .cost_volume_tcn import (
    AxisEncoder,
    DilatedTrajectoryBlock,
    _robust_normalize,
    cost_volume_objective,
)
from .data import Well
from .sequence import SequenceConfig


def make_calibrated_cost_volume_example(
    well: Well,
    cut: int,
    config: SequenceConfig,
    synthetic_rng: np.random.Generator | None = None,
    synthetic_profile: str = "forward_extreme",
    synthetic_hardness: float = 0.75,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Build learned trace features plus the prefix-calibrated v20 cost volume."""
    image, labels, metadata = make_alignment_example(
        well,
        cut,
        config,
        synthetic_rng=synthetic_rng,
        synthetic_profile=synthetic_profile,
        synthetic_hardness=synthetic_hardness,
    )
    observed = np.asarray(metadata["gr"], dtype=np.float64)
    observed_normalized, observed_support = _robust_normalize(observed)
    observed_gradient = np.gradient(observed_normalized).astype(np.float32)
    valid = np.asarray(metadata["valid_positions"], dtype=np.float32)
    observed_features = np.stack(
        [
            observed_normalized,
            observed_gradient,
            observed_support,
            valid,
        ],
        axis=0,
    )

    anchor_tvt = float(metadata["anchor_tvt"])
    reference = np.interp(
        anchor_tvt + config.offsets,
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    reference_normalized, reference_support = _robust_normalize(reference)
    reference_gradient = np.gradient(reference_normalized).astype(np.float32)
    reference_features = np.stack(
        [
            reference_normalized,
            reference_gradient,
            reference_support,
            (config.offsets / config.state_radius).astype(np.float32),
        ],
        axis=0,
    )

    # Unlike the original Siamese candidate, preserve the affine calibration
    # fitted entirely from the visible prefix. Channels 0 and 1 are the signed
    # and absolute horizontal/typewell mismatch for every state and position.
    calibrated_volume = image[:2].astype(np.float32)
    # Channels 7 onward are state-independent legal trajectory/missingness
    # context. Channel 6 is the visible TVT-path hint.
    context = image[7:, 0, :].astype(np.float32)
    visible_hint = image[6].astype(np.float32)
    z_relative = (
        np.asarray(metadata["z"], dtype=np.float64) - float(metadata["anchor_z"])
    ).astype(np.float32)
    return (
        observed_features,
        reference_features,
        calibrated_volume,
        context,
        visible_hint,
        labels.astype(np.int64),
        valid,
        z_relative,
        metadata,
    )


class _CalibratedCostFrontEnd(nn.Module):
    def __init__(
        self,
        state_count: int,
        context_channels: int,
        embedding_channels: int,
        trajectory_channels: int,
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.observed_encoder = AxisEncoder(4, embedding_channels)
        self.reference_encoder = AxisEncoder(4, embedding_channels)
        self.trajectory_stem = nn.Conv1d(
            4 * state_count + context_channels,
            trajectory_channels,
            1,
        )

    def forward(
        self,
        observed: torch.Tensor,
        reference: torch.Tensor,
        calibrated_volume: torch.Tensor,
        context: torch.Tensor,
        visible_hint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_embedding = F.normalize(
            self.observed_encoder(observed), dim=1, eps=1e-6
        )
        reference_embedding = F.normalize(
            self.reference_encoder(reference), dim=1, eps=1e-6
        )
        learned_cost = torch.einsum(
            "bcs,bcl->bsl",
            reference_embedding,
            observed_embedding,
        )
        batch, calibrated_channels, states, length = calibrated_volume.shape
        if calibrated_channels != 2 or states != self.state_count:
            raise ValueError(
                "calibrated volume must be [batch, 2, state_count, length]"
            )
        calibrated_flat = calibrated_volume.reshape(
            batch,
            calibrated_channels * states,
            length,
        )
        features = torch.cat(
            [learned_cost, calibrated_flat, visible_hint, context],
            dim=1,
        )
        return self.trajectory_stem(features), learned_cost, calibrated_volume[:, 1]


class CalibratedCostVolumeTCN(nn.Module):
    """Prefix-calibrated cost volume with a whole-well dilated TCN."""

    def __init__(
        self,
        state_count: int,
        context_channels: int = 12,
        embedding_channels: int = 32,
        trajectory_channels: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.front_end = _CalibratedCostFrontEnd(
            state_count,
            context_channels,
            embedding_channels,
            trajectory_channels,
        )
        self.trajectory_blocks = nn.Sequential(
            *[
                DilatedTrajectoryBlock(trajectory_channels, dilation)
                for dilation in dilations
            ]
        )
        self.trajectory_norm = nn.GroupNorm(8, trajectory_channels)
        self.head = nn.Conv1d(trajectory_channels, state_count, 1)
        self.learned_cost_scale = nn.Parameter(torch.tensor(1.0))
        self.calibrated_cost_scale = nn.Parameter(torch.tensor(1.0))
        self.hint_scale = nn.Parameter(torch.tensor(4.0))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        observed: torch.Tensor,
        reference: torch.Tensor,
        calibrated_volume: torch.Tensor,
        context: torch.Tensor,
        visible_hint: torch.Tensor,
    ) -> torch.Tensor:
        trajectory, learned_cost, absolute_mismatch = self.front_end(
            observed,
            reference,
            calibrated_volume,
            context,
            visible_hint,
        )
        trajectory = self.trajectory_blocks(trajectory)
        residual = self.head(F.silu(self.trajectory_norm(trajectory)))
        return (
            residual
            + self.learned_cost_scale * learned_cost
            - self.calibrated_cost_scale * absolute_mismatch
            + self.hint_scale * visible_hint
        )


class CalibratedCostVolumeGRU(nn.Module):
    """Prefix-calibrated cost volume with a bidirectional whole-well GRU."""

    def __init__(
        self,
        state_count: int,
        context_channels: int = 12,
        embedding_channels: int = 32,
        trajectory_channels: int = 128,
        hidden_channels: int = 96,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.front_end = _CalibratedCostFrontEnd(
            state_count,
            context_channels,
            embedding_channels,
            trajectory_channels,
        )
        self.input_norm = nn.GroupNorm(8, trajectory_channels)
        self.trajectory = nn.GRU(
            input_size=trajectory_channels,
            hidden_size=hidden_channels,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.10 if layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden_channels, state_count)
        self.learned_cost_scale = nn.Parameter(torch.tensor(1.0))
        self.calibrated_cost_scale = nn.Parameter(torch.tensor(1.0))
        self.hint_scale = nn.Parameter(torch.tensor(4.0))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        observed: torch.Tensor,
        reference: torch.Tensor,
        calibrated_volume: torch.Tensor,
        context: torch.Tensor,
        visible_hint: torch.Tensor,
    ) -> torch.Tensor:
        trajectory, learned_cost, absolute_mismatch = self.front_end(
            observed,
            reference,
            calibrated_volume,
            context,
            visible_hint,
        )
        trajectory = F.silu(self.input_norm(trajectory)).transpose(1, 2)
        trajectory, _ = self.trajectory(trajectory)
        residual = self.head(trajectory).transpose(1, 2)
        return (
            residual
            + self.learned_cost_scale * learned_cost
            - self.calibrated_cost_scale * absolute_mismatch
            + self.hint_scale * visible_hint
        )


__all__ = [
    "CalibratedCostVolumeGRU",
    "CalibratedCostVolumeTCN",
    "cost_volume_objective",
    "make_calibrated_cost_volume_example",
]
