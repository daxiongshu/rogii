from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .alignment import AlignmentBlock


class GlobalAlignmentAttention(nn.Module):
    """Zero-initialized global residual over the coarsest state/time tokens."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, expansion * channels),
            nn.GELU(),
            nn.Linear(expansion * channels, channels),
        )
        self.attention_scale = nn.Parameter(torch.zeros(channels))
        self.feed_forward_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended * self.attention_scale
        tokens = tokens + self.feed_forward(self.norm2(tokens)) * (
            self.feed_forward_scale
        )
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class GlobalAttentionAlignmentUNet(nn.Module):
    """Production alignment U-Net plus global bottleneck self-attention."""

    def __init__(
        self,
        input_channels: int = 19,
        base: int = 8,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 6, base * 8, base * 8]
        self.stem = nn.Conv2d(
            input_channels, widths[0], (5, 9), padding=(2, 4)
        )
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:]):
            self.encoder.append(
                nn.Sequential(AlignmentBlock(left), AlignmentBlock(left))
            )
            self.down.append(nn.Conv2d(left, right, 4, stride=2, padding=1))
        self.middle = nn.Sequential(
            AlignmentBlock(widths[-1]), AlignmentBlock(widths[-1])
        )
        self.global_attention = GlobalAlignmentAttention(
            widths[-1], heads=attention_heads
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(
                nn.ConvTranspose2d(left, right, 4, stride=2, padding=1)
            )
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(right * 2, right, 3, padding=1),
                    AlignmentBlock(right),
                )
            )
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        skips = []
        for block, down in zip(self.encoder, self.down):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        x = self.global_attention(x)
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
