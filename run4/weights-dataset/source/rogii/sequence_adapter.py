from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_NAMES = (
    "base_mean",
    "base_std",
    "posterior_entropy",
    "posterior_max",
    "posterior_margin",
    "expected_gr_mismatch",
    "expected_abs_gr_mismatch",
    "expected_pseudo_mismatch",
    "expected_abs_pseudo_mismatch",
    "expected_prefix_support",
    "visible",
    "missing_gr",
    "z_relative",
    "md_relative",
    "x_relative",
    "y_relative",
    "dz_dmd",
    "dx_dmd",
    "dy_dmd",
    "azimuth_cos",
    "azimuth_sin",
    "curvature_z",
    "base_first_difference",
    "base_second_difference",
    "base_surface_first_difference",
)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(F.silu(self.norm(x)))
        left, gate = self.pointwise(x).chunk(2, dim=1)
        x = left * torch.sigmoid(gate)
        x = self.output(self.dropout(x))
        return residual + x


class TCNResidualAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = len(FEATURE_NAMES),
        width: int = 64,
        dropout: float = 0.10,
        correction_limit: float = 32.0,
    ) -> None:
        super().__init__()
        self.correction_limit = correction_limit
        self.input_norm = nn.LayerNorm(input_channels)
        self.stem = nn.Conv1d(input_channels, width, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(width, dilation, dropout)
                for dilation in (1, 2, 4, 8, 16, 32, 64, 128)
            ]
        )
        self.head = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(features).transpose(1, 2)
        raw = self.head(self.blocks(self.stem(x)))[:, 0]
        return self.correction_limit * torch.tanh(raw)


class BiGRUResidualAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int = len(FEATURE_NAMES),
        width: int = 48,
        layers: int = 2,
        dropout: float = 0.10,
        correction_limit: float = 32.0,
    ) -> None:
        super().__init__()
        self.correction_limit = correction_limit
        self.input_norm = nn.LayerNorm(input_channels)
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels, width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=width,
            hidden_size=width,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(self.input_norm(features))
        x, _ = self.gru(x)
        raw = self.head(x)[..., 0]
        return self.correction_limit * torch.tanh(raw)


def build_sequence_adapter(candidate: str) -> nn.Module:
    if candidate == "crossfit_residual_tcn":
        return TCNResidualAdapter()
    if candidate == "crossfit_residual_bigru":
        return BiGRUResidualAdapter()
    raise ValueError(f"unknown sequence adapter: {candidate}")


def sequence_adapter_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    prefix_points: int = 64,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tail_valid = valid[:, prefix_points:]
    pred_tail = prediction[:, prefix_points:]
    target_tail = target[:, prefix_points:]
    squared = torch.square((pred_tail - target_tail) / 16.0)
    regression = (squared * tail_valid).sum() / tail_valid.sum().clamp_min(1.0)
    first = torch.diff(pred_tail, dim=1)
    first_valid = tail_valid[:, 1:] * tail_valid[:, :-1]
    first_penalty = (
        torch.square(first / 8.0) * first_valid
    ).sum() / first_valid.sum().clamp_min(1.0)
    second = torch.diff(pred_tail, n=2, dim=1)
    second_valid = tail_valid[:, 2:] * tail_valid[:, 1:-1] * tail_valid[:, :-2]
    second_penalty = (
        torch.square(second / 4.0) * second_valid
    ).sum() / second_valid.sum().clamp_min(1.0)
    total = regression + 0.002 * first_penalty + 0.001 * second_penalty
    return total, {
        "regression": regression,
        "first_penalty": first_penalty,
        "second_penalty": second_penalty,
    }
