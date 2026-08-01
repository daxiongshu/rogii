from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import nn
from torch.nn import functional as F

from .data import Well


@dataclass(frozen=True)
class TrajectoryConfig:
    prefix_points: int = 128
    tail_points: int = 640
    prefix_context_rows: int = 2048
    target_scale: float = 40.0

    @property
    def length(self) -> int:
        return self.prefix_points + self.tail_points


def make_trajectory_example(
    well: Well, cut: int, config: TrajectoryConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    """Represent one full well using only legal trajectory/log information."""

    prefix_start = max(0, cut - config.prefix_context_rows + 1)
    positions = np.concatenate(
        [
            np.linspace(prefix_start, cut, config.prefix_points),
            np.linspace(cut + 1, len(well.md) - 1, config.tail_points),
        ]
    )
    source = np.arange(len(well.md), dtype=np.float64)

    def sample(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.zeros(config.length, dtype=np.float64)
        return np.interp(positions, source[valid], values[valid])

    md = sample(well.md)
    x = sample(well.x)
    y = sample(well.y)
    z = sample(well.z)
    tvt = sample(well.tvt)

    finite_gr = np.isfinite(well.gr)
    if finite_gr.sum() >= 2:
        filled = np.interp(source, source[finite_gr], well.gr[finite_gr])
        win = min(51, len(filled) if len(filled) % 2 else len(filled) - 1)
        smooth = savgol_filter(filled, win, 2) if win >= 5 else filled
        gr = np.interp(positions, source, smooth)
    else:
        gr = np.zeros(config.length, dtype=np.float64)
    observed_fraction = np.interp(positions, source, finite_gr.astype(float))

    anchor = config.prefix_points - 1
    dmd = np.gradient(md)
    safe_dmd = np.where(np.abs(dmd) < 1e-6, 1.0, dmd)
    dx = np.gradient(x) / safe_dmd
    dy = np.gradient(y) / safe_dmd
    dz = np.gradient(z) / safe_dmd
    horizontal = np.sqrt(np.square(dx) + np.square(dy))
    az_cos = dx / np.maximum(horizontal, 1e-4)
    az_sin = dy / np.maximum(horizontal, 1e-4)
    ddz = np.gradient(dz) / safe_dmd
    ddx = np.gradient(dx) / safe_dmd
    ddy = np.gradient(dy) / safe_dmd

    gr_center = float(np.median(gr[: config.prefix_points]))
    gr_scale = max(
        1.4826
        * float(
            np.median(
                np.abs(gr[: config.prefix_points] - gr_center)
            )
        ),
        10.0,
    )
    visible = np.zeros(config.length, dtype=np.float32)
    visible[: config.prefix_points] = 1.0
    anchor_tvt = float(well.tvt[cut])
    visible_tvt = np.where(visible > 0, (tvt - anchor_tvt) / 40.0, 0.0)

    features = np.stack(
        [
            np.clip((gr - gr_center) / gr_scale, -6.0, 6.0),
            1.0 - observed_fraction,
            visible,
            visible_tvt,
            (md - md[anchor]) / 5000.0,
            (x - x[anchor]) / 5000.0,
            (y - y[anchor]) / 5000.0,
            (z - z[anchor]) / 200.0,
            np.clip(dx, -1.5, 1.5),
            np.clip(dy, -1.5, 1.5),
            np.clip(dz, -1.5, 1.5),
            np.nan_to_num(az_cos),
            np.nan_to_num(az_sin),
            np.clip(horizontal, 0.0, 1.5),
            np.clip(ddz * 100.0, -2.0, 2.0),
            np.clip(ddx * 100.0, -2.0, 2.0),
            np.clip(ddy * 100.0, -2.0, 2.0),
        ],
        axis=0,
    ).astype(np.float32)
    target = ((tvt - anchor_tvt) / config.target_scale).astype(np.float32)
    metadata: dict[str, np.ndarray | float | int] = {
        "positions": positions,
        "anchor_tvt": anchor_tvt,
        "cut": cut,
    }
    return features, target, metadata


class TrajectoryBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class TrajectoryUNet(nn.Module):
    def __init__(self, input_channels: int = 17, base: int = 32, dropout: float = 0.08) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 6, base * 8, base * 8]
        self.stem = nn.Conv1d(input_channels, widths[0], 7, padding=3)
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:]):
            self.encoder.append(
                nn.Sequential(
                    TrajectoryBlock(left, dropout),
                    TrajectoryBlock(left, dropout),
                )
            )
            self.down.append(nn.Conv1d(left, right, 4, stride=2, padding=1))
        self.middle = nn.Sequential(
            TrajectoryBlock(widths[-1], dropout),
            TrajectoryBlock(widths[-1], dropout),
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(nn.ConvTranspose1d(left, right, 4, stride=2, padding=1))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(right * 2, right, 3, padding=1),
                    TrajectoryBlock(right, dropout),
                )
            )
        self.head = nn.Conv1d(widths[0], 1, 1)
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
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)
