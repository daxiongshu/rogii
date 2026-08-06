from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, Swin_T_Weights, resnet18, swin_t
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)
from transformers import SegformerConfig, SegformerModel


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetAlignmentUNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.decode4 = DecoderBlock(512, 256, 256)
        self.decode3 = DecoderBlock(256, 128, 128)
        self.decode2 = DecoderBlock(128, 64, 96)
        self.decode1 = DecoderBlock(96, 64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stem = self.stem(image)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.middle(self.layer4(layer3))
        x = self.decode4(layer4, layer3)
        x = self.decode3(x, layer2)
        x = self.decode2(x, layer1)
        x = self.decode1(x, stem)
        x = self.head(x)
        return F.interpolate(
            x,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)


class SwinAlignmentUNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        backbone = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        self.middle = nn.Sequential(
            nn.Conv2d(768, 768, 3, padding=1),
            nn.GroupNorm(24, 768),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.decode4 = DecoderBlock(768, 384, 384)
        self.decode3 = DecoderBlock(384, 192, 192)
        self.decode2 = DecoderBlock(192, 96, 96)
        self.head = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    @staticmethod
    def _channels_first(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features[0](image)
        level1 = self.features[1](x)
        x = self.features[2](level1)
        level2 = self.features[3](x)
        x = self.features[4](level2)
        level3 = self.features[5](x)
        x = self.features[6](level3)
        level4 = self.features[7](x)
        x = self.middle(self._channels_first(level4))
        x = self.decode4(x, self._channels_first(level3))
        x = self.decode3(x, self._channels_first(level2))
        x = self.decode2(x, self._channels_first(level1))
        x = self.head(x)
        return F.interpolate(
            x,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)


class SegformerAlignmentNet(nn.Module):
    """License-safe randomly initialized version of the public SDF architecture."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        config = SegformerConfig(num_channels=3, drop_path_rate=dropout)
        self.backbone = SegformerModel(config)
        self.proj = nn.ModuleList(
            [nn.Conv2d(channels, 128, 1) for channels in (32, 64, 160, 256)]
        )
        self.fuse = nn.Conv2d(128 * 4, 128, 1)
        self.history_fuse = nn.Conv2d(1, 128, 1)
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=image, output_hidden_states=True)
        features = output.hidden_states
        height, width = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.proj):
            feature = projection(feature)
            if feature.shape[-2:] != (height, width):
                feature = F.interpolate(
                    feature,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)
        history = F.interpolate(
            image[:, 2:3],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fuse(torch.cat(projected, dim=1)) + self.history_fuse(history)
        return self.head(fused).squeeze(1)


class TorchvisionSegmentationSDF(nn.Module):
    def __init__(self, architecture: str, pretrained: bool = True) -> None:
        super().__init__()
        if architecture == "lraspp":
            self.model = lraspp_mobilenet_v3_large(
                weights=(
                    LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                    if pretrained
                    else None
                )
            )
            self.model.classifier.low_classifier = nn.Conv2d(40, 1, 1)
            self.model.classifier.high_classifier = nn.Conv2d(128, 1, 1)
        elif architecture == "deeplab":
            self.model = deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.model.classifier[-1] = nn.Conv2d(256, 1, 1)
            if self.model.aux_classifier is not None:
                self.model.aux_classifier[-1] = nn.Conv2d(256, 1, 1)
        elif architecture == "fcn":
            self.model = fcn_resnet50(
                weights=FCN_ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.model.classifier[-1] = nn.Conv2d(512, 1, 1)
            if self.model.aux_classifier is not None:
                self.model.aux_classifier[-1] = nn.Conv2d(256, 1, 1)
        else:
            raise ValueError(f"unknown segmentation architecture: {architecture}")

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"].squeeze(1)
