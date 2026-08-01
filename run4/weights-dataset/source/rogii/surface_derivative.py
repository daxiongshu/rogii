from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class SurfaceDerivativeOutput(NamedTuple):
    state_logits: torch.Tensor
    derivative_logits: torch.Tensor
    sign_logits: torch.Tensor
    boundary_logits: torch.Tensor


def _surface_sequence(
    image: torch.Tensor,
    state_count: int,
    trajectory_features: str,
) -> torch.Tensor:
    if image.shape[2] != state_count or image.shape[1] < 19:
        raise ValueError("surface derivative image schema mismatch")
    parts = [
        image[:, 0],
        image[:, 2],
        image[:, 6],
        image[:, 7:19, 0],
    ]
    if trajectory_features == "tortuosity":
        # Q-3D-inspired steering features. These are deterministic functions
        # of legal XYZ trajectory derivatives already present in channels
        # 13:18; no target or formation column is involved.
        velocity = image[:, 13:16, 0]
        velocity_change = torch.diff(
            velocity,
            dim=2,
            prepend=velocity[:, :, :1],
        )
        turn = torch.sqrt(
            torch.sum(torch.square(velocity_change), dim=1) + 1e-8
        )
        azimuth = image[:, 16:18, 0]
        azimuth_change = torch.diff(
            azimuth,
            dim=2,
            prepend=azimuth[:, :, :1],
        )
        azimuth_turn = torch.sqrt(
            torch.sum(torch.square(azimuth_change), dim=1) + 1e-8
        )
        vertical_turn = torch.abs(velocity_change[:, 0])
        rolling = [
            F.avg_pool1d(
                turn[:, None],
                kernel_size=window,
                stride=1,
                padding=window // 2,
            ).squeeze(1)
            for window in (9, 33, 129)
        ]
        parts.append(
            torch.stack(
                [turn, azimuth_turn, vertical_turn, *rolling],
                dim=1,
            )
        )
    return torch.cat(parts, dim=1)


def _derivative_values(
    derivative_radius: float,
    derivative_step: float,
) -> torch.Tensor:
    return torch.arange(
        -derivative_radius,
        derivative_radius + 0.5 * derivative_step,
        derivative_step,
        dtype=torch.float32,
    )


def _initialize_derivative_heads(
    state_head: nn.Conv1d,
    derivative_head: nn.Conv1d,
    sign_head: nn.Conv1d,
    boundary_head: nn.Conv1d,
) -> None:
    for head in (state_head, derivative_head, sign_head):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    nn.init.zeros_(boundary_head.weight)
    nn.init.constant_(boundary_head.bias, -3.0)


class DerivativeTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.gate = nn.Conv1d(channels, 2 * channels, 1)
        self.projection = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(F.silu(self.norm(values)))
        left, right = self.gate(values).chunk(2, dim=1)
        values = self.projection(F.silu(left) * torch.sigmoid(right))
        return residual + self.dropout(values)


class SurfaceDerivativeTCN(nn.Module):
    """Complete-well derivative model over legal alignment likelihoods.

    Signed and pseudo-GR cost volumes, the visible path, and the legal
    trajectory channels are treated as one complete sequence.  A dilated TCN
    predicts the geological-surface increment distribution, while categorical
    state, dip-sign, and changepoint heads provide auxiliary supervision.
    """

    def __init__(
        self,
        state_count: int,
        width: int = 160,
        derivative_radius: float = 1.5,
        derivative_step: float = 0.05,
        dropout: float = 0.05,
        trajectory_features: str = "base",
    ) -> None:
        super().__init__()
        if state_count < 3:
            raise ValueError("surface derivative model requires at least 3 states")
        derivative_values = _derivative_values(
            derivative_radius,
            derivative_step,
        )
        self.state_count = int(state_count)
        if trajectory_features not in ("base", "tortuosity"):
            raise ValueError(
                "trajectory features must be 'base' or 'tortuosity'"
            )
        self.trajectory_features = trajectory_features
        self.register_buffer("derivative_values", derivative_values)
        # mismatch, pseudo mismatch, visible path, plus channels 7:19.
        input_channels = (
            3 * state_count
            + 12
            + (6 if trajectory_features == "tortuosity" else 0)
        )
        self.stem = nn.Conv1d(input_channels, width, 1)
        self.blocks = nn.Sequential(
            *[
                DerivativeTCNBlock(width, dilation, dropout)
                for dilation in (1, 2, 4, 8, 16, 32, 64, 128)
            ]
        )
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, width)
        self.state_head = nn.Conv1d(width, state_count, 1)
        self.derivative_head = nn.Conv1d(width, len(derivative_values), 1)
        self.sign_head = nn.Conv1d(width, 3, 1)
        self.boundary_head = nn.Conv1d(width, 1, 1)
        _initialize_derivative_heads(
            self.state_head,
            self.derivative_head,
            self.sign_head,
            self.boundary_head,
        )

    def forward(self, image: torch.Tensor) -> SurfaceDerivativeOutput:
        sequence = _surface_sequence(
            image,
            self.state_count,
            self.trajectory_features,
        )
        features = self.blocks(self.stem(sequence))
        features = F.silu(self.norm(features))
        return SurfaceDerivativeOutput(
            self.state_head(features),
            self.derivative_head(features),
            self.sign_head(features),
            self.boundary_head(features).squeeze(1),
        )


class SurfaceDerivativeTransformer(nn.Module):
    """Bidirectional complete-well Transformer derivative distribution."""

    def __init__(
        self,
        state_count: int,
        width: int = 160,
        derivative_radius: float = 1.5,
        derivative_step: float = 0.05,
        dropout: float = 0.05,
        trajectory_features: str = "base",
        layers: int = 4,
    ) -> None:
        super().__init__()
        if state_count < 3:
            raise ValueError("surface derivative model requires at least 3 states")
        if trajectory_features not in ("base", "tortuosity"):
            raise ValueError(
                "trajectory features must be 'base' or 'tortuosity'"
            )
        if width % 8:
            raise ValueError("Transformer width must be divisible by eight")
        derivative_values = _derivative_values(
            derivative_radius,
            derivative_step,
        )
        self.state_count = int(state_count)
        self.trajectory_features = trajectory_features
        self.register_buffer("derivative_values", derivative_values)
        input_channels = (
            3 * state_count
            + 12
            + (6 if trajectory_features == "tortuosity" else 0)
        )
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, groups=width),
            nn.Conv1d(width, width, 1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=8,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(width)
        self.state_head = nn.Conv1d(width, state_count, 1)
        self.derivative_head = nn.Conv1d(width, len(derivative_values), 1)
        self.sign_head = nn.Conv1d(width, 3, 1)
        self.boundary_head = nn.Conv1d(width, 1, 1)
        _initialize_derivative_heads(
            self.state_head,
            self.derivative_head,
            self.sign_head,
            self.boundary_head,
        )

    def forward(self, image: torch.Tensor) -> SurfaceDerivativeOutput:
        sequence = _surface_sequence(
            image,
            self.state_count,
            self.trajectory_features,
        )
        features = self.stem(sequence).transpose(1, 2)
        features = self.norm(self.encoder(features)).transpose(1, 2)
        return SurfaceDerivativeOutput(
            self.state_head(features),
            self.derivative_head(features),
            self.sign_head(features),
            self.boundary_head(features).squeeze(1),
        )


class SurfaceDerivativeGRU(nn.Module):
    """Bidirectional state-space derivative and changepoint model."""

    def __init__(
        self,
        state_count: int,
        width: int = 160,
        derivative_radius: float = 1.5,
        derivative_step: float = 0.05,
        dropout: float = 0.05,
        trajectory_features: str = "base",
        layers: int = 2,
    ) -> None:
        super().__init__()
        if state_count < 3:
            raise ValueError("surface derivative model requires at least 3 states")
        if trajectory_features not in ("base", "tortuosity"):
            raise ValueError(
                "trajectory features must be 'base' or 'tortuosity'"
            )
        if width % 2:
            raise ValueError("GRU width must be even")
        derivative_values = _derivative_values(
            derivative_radius,
            derivative_step,
        )
        self.state_count = int(state_count)
        self.trajectory_features = trajectory_features
        self.register_buffer("derivative_values", derivative_values)
        input_channels = (
            3 * state_count
            + 12
            + (6 if trajectory_features == "tortuosity" else 0)
        )
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2, groups=width),
            nn.Conv1d(width, width, 1),
        )
        self.gru = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(width)
        self.state_head = nn.Conv1d(width, state_count, 1)
        self.derivative_head = nn.Conv1d(width, len(derivative_values), 1)
        self.sign_head = nn.Conv1d(width, 3, 1)
        self.boundary_head = nn.Conv1d(width, 1, 1)
        _initialize_derivative_heads(
            self.state_head,
            self.derivative_head,
            self.sign_head,
            self.boundary_head,
        )

    def forward(self, image: torch.Tensor) -> SurfaceDerivativeOutput:
        sequence = _surface_sequence(
            image,
            self.state_count,
            self.trajectory_features,
        )
        features, _ = self.gru(self.stem(sequence).transpose(1, 2))
        features = self.norm(features).transpose(1, 2)
        return SurfaceDerivativeOutput(
            self.state_head(features),
            self.derivative_head(features),
            self.sign_head(features),
            self.boundary_head(features).squeeze(1),
        )


def make_surface_derivative_model(
    architecture: str,
    state_count: int,
    width: int,
    derivative_radius: float,
    derivative_step: float,
    trajectory_features: str,
) -> nn.Module:
    common = {
        "state_count": state_count,
        "width": width,
        "derivative_radius": derivative_radius,
        "derivative_step": derivative_step,
        "trajectory_features": trajectory_features,
    }
    if architecture == "tcn":
        return SurfaceDerivativeTCN(**common)
    if architecture == "transformer":
        return SurfaceDerivativeTransformer(**common)
    if architecture == "gru":
        return SurfaceDerivativeGRU(**common)
    raise ValueError(f"unknown surface derivative architecture: {architecture}")


def posterior_surface_derivative(
    output: SurfaceDerivativeOutput,
    derivative_values: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("derivative temperature must be positive")
    probability = torch.softmax(
        output.derivative_logits.float() / float(temperature),
        dim=1,
    )
    return torch.sum(
        probability * derivative_values[None, :, None],
        dim=1,
    )


def integrate_surface_derivative(
    output: SurfaceDerivativeOutput,
    derivative_values: torch.Tensor,
    z_relative: torch.Tensor,
    anchor: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    derivative = posterior_surface_derivative(
        output,
        derivative_values,
        temperature,
    )
    cumulative = torch.cat(
        [
            torch.zeros_like(derivative[:, :1]),
            torch.cumsum(derivative[:, 1:], dim=1),
        ],
        dim=1,
    )
    surface_delta = cumulative - cumulative[:, anchor : anchor + 1]
    return surface_delta - z_relative, derivative


def surface_derivative_objective(
    output: SurfaceDerivativeOutput,
    state_label: torch.Tensor,
    target_offset: torch.Tensor,
    z_relative: torch.Tensor,
    valid: torch.Tensor,
    prefix_points: int,
    derivative_values: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prefix_points < 2:
        raise ValueError("surface derivative objective requires visible history")
    anchor = prefix_points - 1
    predicted_offset, predicted_derivative = integrate_surface_derivative(
        output,
        derivative_values,
        z_relative,
        anchor,
    )
    surface = target_offset + z_relative
    true_derivative = torch.cat(
        [
            torch.zeros_like(surface[:, :1]),
            torch.diff(surface, dim=1),
        ],
        dim=1,
    )
    tail_valid = valid[:, prefix_points:].float()
    tail_prediction = predicted_offset[:, prefix_points:]
    tail_truth = target_offset[:, prefix_points:]
    path_point = F.smooth_l1_loss(
        tail_prediction / 16.0,
        tail_truth / 16.0,
        beta=0.25,
        reduction="none",
    )
    path = (path_point * tail_valid).sum() / tail_valid.sum().clamp_min(1.0)

    pair_valid = valid[:, 1:].float() * valid[:, :-1].float()
    derivative_truth = true_derivative[:, 1:]
    distance = torch.abs(
        derivative_truth[:, None]
        - derivative_values[None, :, None]
    )
    derivative_class = torch.argmin(distance, dim=1)
    derivative_ce_point = F.cross_entropy(
        output.derivative_logits[:, :, 1:],
        derivative_class,
        reduction="none",
    )
    derivative_ce = (
        derivative_ce_point * pair_valid
    ).sum() / pair_valid.sum().clamp_min(1.0)
    derivative_huber_point = F.smooth_l1_loss(
        predicted_derivative[:, 1:],
        derivative_truth,
        beta=0.10,
        reduction="none",
    )
    derivative_huber = (
        derivative_huber_point * pair_valid
    ).sum() / pair_valid.sum().clamp_min(1.0)

    sign_target = torch.where(
        derivative_truth < -0.075,
        torch.zeros_like(derivative_class),
        torch.where(
            derivative_truth > 0.075,
            torch.full_like(derivative_class, 2),
            torch.ones_like(derivative_class),
        ),
    )
    sign_point = F.cross_entropy(
        output.sign_logits[:, :, 1:],
        sign_target,
        reduction="none",
    )
    sign = (sign_point * pair_valid).sum() / pair_valid.sum().clamp_min(1.0)

    triple_valid = pair_valid[:, 1:] * pair_valid[:, :-1]
    curvature_truth = torch.diff(true_derivative[:, 1:], dim=1)
    boundary_target = (torch.abs(curvature_truth) > 0.15).float()
    boundary_point = F.binary_cross_entropy_with_logits(
        output.boundary_logits[:, 2:],
        boundary_target,
        pos_weight=torch.as_tensor(
            8.0,
            device=surface.device,
            dtype=surface.dtype,
        ),
        reduction="none",
    )
    boundary = (
        boundary_point * triple_valid
    ).sum() / triple_valid.sum().clamp_min(1.0)

    # Emission supervision stabilizes the cost-volume representation without
    # participating in the registered derivative decoder.
    state_point = F.cross_entropy(
        output.state_logits[:, :, prefix_points:],
        state_label[:, prefix_points:],
        reduction="none",
    )
    state = (state_point * tail_valid).sum() / tail_valid.sum().clamp_min(1.0)

    predicted_surface = predicted_offset + z_relative
    predicted_curvature = torch.diff(predicted_surface, n=2, dim=1)
    curvature = (
        torch.abs(predicted_curvature) * triple_valid
    ).sum() / triple_valid.sum().clamp_min(1.0)
    total = (
        8.0 * path
        + 0.5 * derivative_ce
        + 1.5 * derivative_huber
        + 0.15 * sign
        + 0.05 * boundary
        + 0.10 * state
        + 0.002 * curvature
    )
    return total, {
        "path": path,
        "derivative_ce": derivative_ce,
        "derivative_huber": derivative_huber,
        "sign": sign,
        "boundary": boundary,
        "state": state,
        "curvature": curvature,
        "predicted_offset": predicted_offset,
    }
