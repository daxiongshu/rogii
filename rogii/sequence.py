from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import savgol_filter

from .data import Well


@dataclass(frozen=True)
class SequenceConfig:
    prefix_points: int = 384
    tail_points: int = 1536
    prefix_context_rows: int = 1536
    state_radius: float = 100.0
    state_step: float = 2.0
    target_scale: float = 40.0
    target_kind: str = "tvt_delta"
    row_stride: float | None = None
    gr_smooth_window: int = 51

    @property
    def offsets(self) -> np.ndarray:
        return np.arange(
            -self.state_radius,
            self.state_radius + self.state_step / 2,
            self.state_step,
            dtype=np.float64,
        )

    @property
    def length(self) -> int:
        return self.prefix_points + self.tail_points

    @property
    def input_channels(self) -> int:
        # One signed GR mismatch per candidate TVT plus eight context channels.
        return len(self.offsets) + 8


def _robust_affine(
    reference: np.ndarray, observed: np.ndarray
) -> tuple[float, float, float]:
    mask = np.isfinite(reference) & np.isfinite(observed)
    x = reference[mask]
    y = observed[mask]
    if len(x) < 20 or np.std(x) < 1e-6:
        return 1.0, 0.0, 15.0
    design = np.column_stack([x, np.ones_like(x)])
    weights = np.ones(len(x), dtype=np.float64)
    coefficient = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(4):
        root = np.sqrt(weights)
        coefficient = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        coefficient[0] = np.clip(coefficient[0], 0.2, 3.0)
        residual = y - design @ coefficient
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-6))
    residual = y - design @ coefficient
    scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
    return float(coefficient[0]), float(coefficient[1]), float(scale)


def sequence_positions(well: Well, cut: int, config: SequenceConfig) -> np.ndarray:
    if config.row_stride is not None:
        prefix = cut - config.row_stride * np.arange(
            config.prefix_points - 1, -1, -1, dtype=np.float64
        )
        tail = (
            cut
            + 1
            + config.row_stride * np.arange(config.tail_points, dtype=np.float64)
        )
        return np.concatenate(
            [np.clip(prefix, 0, len(well.md) - 1), np.clip(tail, 0, len(well.md) - 1)]
        )
    prefix_start = max(0, cut - config.prefix_context_rows + 1)
    prefix = np.linspace(prefix_start, cut, config.prefix_points)
    tail = np.linspace(cut + 1, len(well.md) - 1, config.tail_points)
    return np.concatenate([prefix, tail])


def make_sequence_example(
    well: Well,
    cut: int,
    config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    if cut < 20 or cut >= len(well.md) - 20:
        raise ValueError(f"{well.well_id}: invalid cut {cut}")
    floating_positions = sequence_positions(well, cut, config)
    valid_positions = np.ones(config.length, dtype=np.float32)
    if config.row_stride is not None:
        raw_prefix = cut - config.row_stride * np.arange(
            config.prefix_points - 1, -1, -1, dtype=np.float64
        )
        raw_tail = (
            cut
            + 1
            + config.row_stride * np.arange(config.tail_points, dtype=np.float64)
        )
        valid_positions = np.concatenate(
            [raw_prefix >= 0, raw_tail <= len(well.md) - 1]
        ).astype(np.float32)
    source_positions = np.arange(len(well.md), dtype=np.float64)

    def sample(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.full(config.length, np.nan, dtype=np.float64)
        return np.interp(floating_positions, source_positions[valid], values[valid])

    md = sample(well.md)
    x = sample(well.x)
    y = sample(well.y)
    z = sample(well.z)
    # Horizontal GR is much noisier than the typewell.  The successful image
    # formulations compare a block-scale horizontal trace with the 0.5-ft
    # typewell, rather than sparsely sampling raw one-foot measurements.
    finite_gr = np.isfinite(well.gr)
    if finite_gr.sum() >= 2:
        gr_filled = np.interp(
            source_positions, source_positions[finite_gr], well.gr[finite_gr]
        )
        requested_window = max(1, config.gr_smooth_window)
        if requested_window % 2 == 0:
            requested_window -= 1
        smooth_window = min(
            requested_window,
            len(gr_filled) if len(gr_filled) % 2 else len(gr_filled) - 1,
        )
        gr_smoothed = (
            savgol_filter(gr_filled, smooth_window, 2)
            if smooth_window >= 5
            else gr_filled
        )
        gr = np.interp(floating_positions, source_positions, gr_smoothed)
    else:
        gr = sample(well.gr)
    tvt = sample(well.tvt)
    anchor_tvt = float(well.tvt[cut])
    anchor_md = float(well.md[cut])
    anchor_x = float(well.x[cut])
    anchor_y = float(well.y[cut])
    anchor_z = float(well.z[cut])

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
    state_tvt = anchor_tvt + config.offsets
    state_gr = (
        gain
        * np.interp(
            state_tvt,
            well.typewell_tvt,
            well.typewell_gr,
            left=np.nan,
            right=np.nan,
        )
        + offset
    )

    mismatch = (gr[None, :] - state_gr[:, None]) / max(residual_scale, 3.0)
    mismatch = np.nan_to_num(mismatch, nan=0.0, posinf=0.0, neginf=0.0)
    mismatch = np.clip(mismatch, -6.0, 6.0).astype(np.float32)
    visible = np.zeros(config.length, dtype=np.float32)
    visible[: config.prefix_points] = 1.0
    missing_gr = (~np.isfinite(gr)).astype(np.float32)
    gr_normalized = np.nan_to_num((gr - offset) / max(residual_scale, 3.0)).astype(
        np.float32
    )
    gr_normalized = np.clip(gr_normalized, -6.0, 6.0)
    visible_tvt = np.where(visible > 0, (tvt - anchor_tvt) / 40.0, 0.0)

    context = np.stack(
        [
            gr_normalized,
            missing_gr,
            visible,
            visible_tvt.astype(np.float32),
            ((md - anchor_md) / 5000.0).astype(np.float32),
            ((x - anchor_x) / 5000.0).astype(np.float32),
            ((y - anchor_y) / 5000.0).astype(np.float32),
            ((z - anchor_z) / 200.0).astype(np.float32),
        ],
        axis=0,
    )
    inputs = np.concatenate([mismatch, context], axis=0).astype(np.float32)

    if config.target_kind == "surface_delta":
        raw_target = (tvt + z) - (anchor_tvt + anchor_z)
    elif config.target_kind == "tvt_delta":
        raw_target = tvt - anchor_tvt
    else:
        raise ValueError(f"unknown target kind: {config.target_kind}")
    target = (raw_target / config.target_scale).astype(np.float32)
    metadata: dict[str, np.ndarray | float | int] = {
        "positions": floating_positions,
        "valid_positions": valid_positions,
        "md": md,
        "x": x,
        "y": y,
        "z": z,
        "gr": gr,
        "anchor_tvt": anchor_tvt,
        "anchor_z": anchor_z,
        "gr_scale": residual_scale,
        "cut": cut,
    }
    return inputs, target, metadata


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class WellUNet(nn.Module):
    def __init__(
        self, input_channels: int, base: int = 48, zero_head: bool = False
    ) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 6]
        self.stem = nn.Conv1d(input_channels, widths[0], 7, padding=3)
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:]):
            self.encoder.append(nn.Sequential(ResidualBlock(left), ResidualBlock(left)))
            self.down.append(nn.Conv1d(left, right, 4, stride=2, padding=1))
        self.middle = nn.Sequential(
            ResidualBlock(widths[-1]), ResidualBlock(widths[-1])
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(nn.ConvTranspose1d(left, right, 4, stride=2, padding=1))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(right * 2, right, 3, padding=1),
                    ResidualBlock(right),
                    ResidualBlock(right),
                )
            )
        self.head = nn.Sequential(
            ResidualBlock(widths[0]),
            nn.GroupNorm(8, widths[0]),
            nn.SiLU(),
            nn.Conv1d(widths[0], 1, 1),
        )
        if zero_head:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

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
                x = F.interpolate(
                    x, size=skip.shape[-1], mode="linear", align_corners=False
                )
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)
