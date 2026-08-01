from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Well
from .sequence import _robust_affine


@dataclass(frozen=True)
class SurfaceAlignmentConfig:
    """Configuration for a GR/typewell image in structural-surface coordinates."""

    prefix_points: int = 192
    tail_points: int = 768
    prefix_context_rows: int = 1600
    state_radius: float = 400.0
    state_step: float = 2.0

    @property
    def offsets(self) -> np.ndarray:
        return np.arange(
            -self.state_radius,
            self.state_radius + self.state_step / 2,
            self.state_step,
            dtype=np.float32,
        )

    @property
    def length(self) -> int:
        return self.prefix_points + self.tail_points


def _positions(well: Well, cut: int, config: SurfaceAlignmentConfig) -> np.ndarray:
    prefix_start = max(0, cut - config.prefix_context_rows + 1)
    prefix = np.linspace(prefix_start, cut, config.prefix_points)
    tail = np.linspace(cut + 1, len(well.md) - 1, config.tail_points)
    return np.concatenate([prefix, tail]).astype(np.float64)


def make_surface_alignment_example(
    well: Well,
    cut: int,
    config: SurfaceAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    """Build a compact alignment image whose vertical axis is U = TVT + Z.

    At position j and candidate surface U_i, the corresponding typewell depth is
    TVT_ij = U_i - Z_j.  This removes the large, known Z shear from the learning
    problem and leaves the network to estimate only geological surface drift.
    """

    if cut < 20 or cut >= len(well.md) - 20:
        raise ValueError(f"{well.well_id}: invalid cut {cut}")

    positions = _positions(well, cut, config)
    source = np.arange(len(well.md), dtype=np.float64)

    def sample(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.full(config.length, np.nan, dtype=np.float64)
        return np.interp(positions, source[valid], values[valid])

    md = sample(well.md)
    z = sample(well.z)
    gr = sample(well.gr)
    tvt = sample(well.tvt)
    true_surface = tvt + z
    anchor_surface = float(well.tvt[cut] + well.z[cut])
    anchor_md = float(well.md[cut])

    prefix_rows = np.arange(cut + 1)
    prefix_reference = np.interp(
        well.tvt[prefix_rows],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, intercept, gr_scale = _robust_affine(
        prefix_reference, well.gr[prefix_rows]
    )

    offsets = config.offsets
    candidate_surface = anchor_surface + offsets[:, None]
    candidate_tvt = candidate_surface - z[None, :]
    reference_gr = np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    expected_gr = gain * reference_gr + intercept
    mismatch = (gr[None, :] - expected_gr) / max(gr_scale, 3.0)
    mismatch = np.clip(np.nan_to_num(mismatch, nan=0.0), -6.0, 6.0)

    # The derivative channel preserves thin-bed edges even when GR gain differs.
    reference_gradient = np.gradient(reference_gr, config.state_step, axis=0)
    gradient_scale = max(float(np.nanstd(reference_gradient)), 1.0)
    reference_gradient = np.clip(
        np.nan_to_num(reference_gradient / gradient_scale, nan=0.0), -6.0, 6.0
    )

    true_offset = true_surface - anchor_surface
    labels = np.rint((true_offset + config.state_radius) / config.state_step)
    labels = np.clip(labels, 0, len(offsets) - 1).astype(np.int64)

    visible = np.zeros(config.length, dtype=np.float32)
    visible[: config.prefix_points] = 1.0
    distance_to_path = offsets[:, None] - true_offset[None, :]
    visible_path = np.exp(-0.5 * np.square(distance_to_path / 3.0))
    visible_path[:, config.prefix_points :] = 0.0

    valid_reference = np.isfinite(reference_gr).astype(np.float32)
    missing_gr = (~np.isfinite(gr)).astype(np.float32)
    state_coordinate = np.broadcast_to(
        (offsets / config.state_radius)[:, None], candidate_tvt.shape
    )
    distance_coordinate = np.broadcast_to(
        ((md - anchor_md) / 5000.0)[None, :], candidate_tvt.shape
    )

    image = np.stack(
        [
            mismatch,
            np.abs(mismatch),
            reference_gradient,
            valid_reference,
            visible_path,
            np.broadcast_to(visible[None, :], candidate_tvt.shape),
            np.broadcast_to(missing_gr[None, :], candidate_tvt.shape),
            state_coordinate,
            distance_coordinate,
        ],
        axis=0,
    ).astype(np.float32)

    metadata: dict[str, np.ndarray | float | int] = {
        "positions": positions,
        "z": z.astype(np.float32),
        "true_surface": true_surface.astype(np.float32),
        "anchor_surface": anchor_surface,
        "cut": cut,
        "gr_scale": gr_scale,
    }
    return image, labels, metadata


class SurfaceBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, (5, 7), padding=(2, 3))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, (5, 7), padding=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class SurfaceAlignmentUNet(nn.Module):
    """Small 2-D U-Net that returns one logit per surface state and MD point."""

    def __init__(self, input_channels: int = 9, base: int = 12) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 6]
        self.stem = nn.Conv2d(input_channels, widths[0], (5, 9), padding=(2, 4))
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:]):
            self.encoder.append(nn.Sequential(SurfaceBlock(left), SurfaceBlock(left)))
            self.down.append(nn.Conv2d(left, right, 4, stride=2, padding=1))
        self.middle = nn.Sequential(SurfaceBlock(widths[-1]), SurfaceBlock(widths[-1]))
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(nn.ConvTranspose2d(left, right, 4, stride=2, padding=1))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(right * 2, right, 3, padding=1),
                    SurfaceBlock(right),
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
