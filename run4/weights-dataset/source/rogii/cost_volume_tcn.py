from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .alignment import make_alignment_example
from .data import Well
from .sequence import SequenceConfig


def _robust_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.zeros_like(values, dtype=np.float32), finite.astype(np.float32)
    center = float(np.median(values[finite]))
    scale = float(1.4826 * np.median(np.abs(values[finite] - center)))
    scale = max(scale, float(np.std(values[finite])) * 0.25, 3.0)
    normalized = np.where(finite, (values - center) / scale, 0.0)
    return np.clip(normalized, -6.0, 6.0).astype(np.float32), finite.astype(
        np.float32
    )


def make_cost_volume_example(
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
    dict,
]:
    """Build separate observed/reference traces and legal trajectory context."""
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

    # Channels 7 onward in the production image are state-independent and
    # contain only inference-safe visible/missing/trajectory information.
    context = image[7:, 0, :].astype(np.float32)
    visible_hint = image[6].astype(np.float32)
    z_relative = (
        np.asarray(metadata["z"], dtype=np.float64) - float(metadata["anchor_z"])
    ).astype(np.float32)
    return (
        observed_features,
        reference_features,
        context,
        visible_hint,
        labels.astype(np.int64),
        valid,
        z_relative,
        metadata,
    )


class AxisEncoderBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, 1)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(F.silu(self.norm(x)))
        left, right = self.pointwise(x).chunk(2, dim=1)
        x = self.projection(F.silu(left) * torch.sigmoid(right))
        return residual + x


class AxisEncoder(nn.Module):
    def __init__(self, input_channels: int, embedding_channels: int) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_channels, embedding_channels, 9, padding=4)
        self.blocks = nn.Sequential(
            AxisEncoderBlock(embedding_channels, 9),
            AxisEncoderBlock(embedding_channels, 7),
            AxisEncoderBlock(embedding_channels, 5),
        )
        self.output_norm = nn.GroupNorm(8, embedding_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.blocks(self.stem(x)))


class DilatedAxisEncoderBlock(nn.Module):
    """Depthwise residual block with an explicit long-range axis dilation."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("axis dilation must be positive")
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, 1)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(F.silu(self.norm(x)))
        left, right = self.pointwise(x).chunk(2, dim=1)
        x = self.projection(F.silu(left) * torch.sigmoid(right))
        return residual + x


class DilatedAxisEncoder(nn.Module):
    """Multiscale trace encoder used by the registered metric-CNN variant.

    The local encoder above sees fewer than twenty samples.  That is too short
    to distinguish many repeated GR motifs and, more importantly, gives very
    different physical receptive fields on the 0.5-ft typewell axis and the
    coarser horizontal axis.  This stack retains local resolution while adding
    explicit 1/2/4/8/16/32-sample contexts before the cosine cost volume.
    """

    def __init__(self, input_channels: int, embedding_channels: int) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_channels, embedding_channels, 9, padding=4)
        self.blocks = nn.Sequential(
            *[
                DilatedAxisEncoderBlock(embedding_channels, dilation)
                for dilation in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.output_norm = nn.GroupNorm(8, embedding_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.blocks(self.stem(x)))


class DilatedTrajectoryBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, 1)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(F.silu(self.norm(x)))
        left, right = self.pointwise(x).chunk(2, dim=1)
        x = self.projection(F.silu(left) * torch.sigmoid(right))
        return residual + x


class SiameseCostVolumeTCN(nn.Module):
    """Siamese trace encoders, explicit shift cost volume, and whole-well TCN."""

    def __init__(
        self,
        state_count: int,
        context_channels: int = 12,
        embedding_channels: int = 32,
        trajectory_channels: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.state_count = state_count
        self.observed_encoder = AxisEncoder(4, embedding_channels)
        self.reference_encoder = AxisEncoder(4, embedding_channels)
        self.trajectory_stem = nn.Conv1d(
            2 * state_count + context_channels,
            trajectory_channels,
            1,
        )
        self.trajectory_blocks = nn.Sequential(
            *[
                DilatedTrajectoryBlock(trajectory_channels, dilation)
                for dilation in dilations
            ]
        )
        self.trajectory_norm = nn.GroupNorm(8, trajectory_channels)
        self.head = nn.Conv1d(trajectory_channels, state_count, 1)
        self.cost_scale = nn.Parameter(torch.tensor(2.0))
        self.hint_scale = nn.Parameter(torch.tensor(4.0))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        observed: torch.Tensor,
        reference: torch.Tensor,
        context: torch.Tensor,
        visible_hint: torch.Tensor,
    ) -> torch.Tensor:
        observed_embedding = F.normalize(
            self.observed_encoder(observed), dim=1, eps=1e-6
        )
        reference_embedding = F.normalize(
            self.reference_encoder(reference), dim=1, eps=1e-6
        )
        cost = torch.einsum(
            "bcs,bcl->bsl", reference_embedding, observed_embedding
        )
        trajectory = self.trajectory_stem(
            torch.cat([cost, visible_hint, context], dim=1)
        )
        trajectory = self.trajectory_blocks(trajectory)
        residual_logits = self.head(F.silu(self.trajectory_norm(trajectory)))
        return (
            residual_logits
            + self.cost_scale * cost
            + self.hint_scale * visible_hint
        )


def cost_volume_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    z_relative: torch.Tensor,
    offsets: torch.Tensor,
    prefix_points: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tail = slice(prefix_points, None)
    tail_valid = valid[:, tail]
    point_classification = F.cross_entropy(
        logits[:, :, tail], labels[:, tail], reduction="none"
    )
    classification = (
        point_classification * tail_valid
    ).sum() / tail_valid.sum().clamp_min(1.0)
    probability = torch.softmax(logits[:, :, tail].float(), dim=1)
    prediction = torch.sum(probability * offsets[None, :, None], dim=1)
    truth = offsets[labels[:, tail]]
    point_regression = torch.square((prediction - truth) / 40.0)
    regression = (
        point_regression * tail_valid
    ).sum() / tail_valid.sum().clamp_min(1.0)
    surface = prediction + z_relative[:, tail].float()
    smooth_valid = tail_valid[:, 2:] * tail_valid[:, 1:-1] * tail_valid[:, :-2]
    point_smoothness = torch.abs(torch.diff(surface, n=2, dim=1))
    smoothness = (
        point_smoothness * smooth_valid
    ).sum() / smooth_valid.sum().clamp_min(1.0)
    total = classification + regression + 1e-3 * smoothness
    return total, {
        "classification": classification,
        "regression": regression,
        "surface_smoothness": smoothness,
    }
