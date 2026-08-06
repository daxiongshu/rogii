from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNeXtAlignmentBlock(nn.Module):
    """Channels-last ConvNeXt block with anisotropic alignment context."""

    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int] = (5, 11),
        expansion: int = 4,
        layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, expansion * channels)
        self.contract = nn.Linear(expansion * channels, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = F.gelu(x)
        x = self.contract(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return residual + x


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtAlignmentUNet(nn.Module):
    """Depthwise hierarchical alignment network with full-well receptive field."""

    def __init__(
        self,
        input_channels: int = 19,
        widths: tuple[int, ...] = (24, 48, 96, 144, 192),
        depths: tuple[int, ...] = (2, 2, 3, 3, 2),
    ) -> None:
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have equal length")
        self.stem = nn.Conv2d(
            input_channels,
            widths[0],
            kernel_size=(5, 9),
            padding=(2, 4),
        )
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvNeXtAlignmentBlock(channels)
                        for _ in range(depth)
                    ]
                )
                for channels, depth in zip(widths, depths)
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(left),
                    nn.Conv2d(left, right, kernel_size=2, stride=2),
                )
                for left, right in zip(widths[:-1], widths[1:])
            ]
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(
                nn.ConvTranspose2d(left, right, kernel_size=2, stride=2)
            )
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(2 * right, right, kernel_size=1),
                    ConvNeXtAlignmentBlock(right),
                    ConvNeXtAlignmentBlock(right),
                )
            )
        self.head = nn.Sequential(
            LayerNorm2d(widths[0]),
            nn.Conv2d(widths[0], 1, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        skips = []
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index == len(self.encoder) - 1:
                break
            skips.append(x)
            x = self.down[index](x)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)
