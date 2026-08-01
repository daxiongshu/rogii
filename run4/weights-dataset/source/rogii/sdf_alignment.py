from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .alignment import AlignmentUNet, alignment_objective, make_alignment_example
from .cost_volume_tcn import AxisEncoder, DilatedAxisEncoder
from .data import Well
from .sequence import SequenceConfig, _robust_affine


def make_sdf_alignment_example(
    well: Well,
    cut: int,
    config: SequenceConfig,
    synthetic_rng: np.random.Generator | None = None,
    synthetic_profile: str = "legacy",
    synthetic_hardness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | int]]:
    """Build an alignment image with separate calibrated GR planes.

    The categorical v20 image emphasizes pairwise mismatch.  The public SDF
    formulation instead exposes typewell and horizontal GR separately.  Keep
    the 19 legal v20 channels and append those two raw views so this family can
    learn its own local metric without using any train-only column.
    """
    image, label, metadata = make_alignment_example(
        well,
        cut,
        config,
        synthetic_rng=synthetic_rng,
        synthetic_profile=synthetic_profile,
        synthetic_hardness=synthetic_hardness,
        coordinate_kind="tvt_delta",
    )
    sampled_gr = np.asarray(metadata["gr"], dtype=np.float64)
    prefix_rows = np.arange(cut + 1)
    prefix_reference = np.interp(
        well.tvt[prefix_rows],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, intercept, residual_scale = _robust_affine(
        prefix_reference,
        well.gr[prefix_rows],
    )
    state_gr = gain * np.interp(
        float(metadata["anchor_tvt"]) + config.offsets,
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ) + intercept
    finite_state = np.isfinite(state_gr)
    finite_horizontal = np.isfinite(sampled_gr)
    combined = np.concatenate(
        [state_gr[finite_state], sampled_gr[finite_horizontal]]
    )
    center = float(np.median(combined)) if len(combined) else 0.0
    amplitude_scale = (
        float(1.4826 * np.median(np.abs(combined - center)))
        if len(combined)
        else 1.0
    )
    scale = max(amplitude_scale, residual_scale, 3.0)
    typewell_plane = np.broadcast_to(
        np.clip(
            np.nan_to_num((state_gr - center) / scale, nan=0.0),
            -6.0,
            6.0,
        )[:, None],
        image.shape[-2:],
    )
    horizontal_plane = np.broadcast_to(
        np.clip(
            np.nan_to_num((sampled_gr - center) / scale, nan=0.0),
            -6.0,
            6.0,
        )[None, :],
        image.shape[-2:],
    )
    raw_views = np.stack([typewell_plane, horizontal_plane], axis=0).astype(
        np.float32
    )
    metadata["sdf_gr_center"] = center
    metadata["sdf_gr_scale"] = scale
    return np.concatenate([image, raw_views], axis=0), label, metadata


class SDFAlignmentUNet(nn.Module):
    """Predict a dense signed-distance field on an alignment image."""

    def __init__(
        self,
        input_channels: int = 21,
        base: int = 12,
        dropout: float = 0.0,
        sdf_clip: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone = AlignmentUNet(
            input_channels=input_channels,
            base=base,
            dropout=dropout,
        )
        self.sdf_clip = float(sdf_clip)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.sdf_clip * torch.tanh(self.backbone(image))


class SDFAlignmentSegformer(nn.Module):
    """MiT-B0 dense SDF decoder with a legal 19-channel input adapter."""

    def __init__(
        self,
        input_channels: int = 21,
        pretrained: bool = False,
        decoder_channels: int = 64,
        sdf_clip: float = 3.0,
    ) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerModel

        self.input_adapter = nn.Conv2d(input_channels, 3, 1, bias=False)
        nn.init.zeros_(self.input_adapter.weight)
        with torch.no_grad():
            # Start from the public three-view representation: calibrated
            # typewell GR, calibrated horizontal GR, and visible history.
            self.input_adapter.weight[0, input_channels - 2, 0, 0] = 1.0
            self.input_adapter.weight[1, input_channels - 1, 0, 0] = 1.0
            self.input_adapter.weight[2, 6, 0, 0] = 1.0
        config = SegformerConfig(
            num_channels=3,
            hidden_sizes=[32, 64, 160, 256],
            depths=[2, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
        )
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(
                "nvidia/mit-b0",
                config=config,
            )
        else:
            self.encoder = SegformerModel(config)
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(channels, decoder_channels, 1)
                for channels in config.hidden_sizes
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * decoder_channels, decoder_channels, 1),
            nn.GroupNorm(8, decoder_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.SiLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_channels,
                decoder_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, decoder_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(
                decoder_channels,
                decoder_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, decoder_channels),
            nn.SiLU(),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels + 16, decoder_channels, 3, padding=1),
            nn.GroupNorm(8, decoder_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.sdf_clip = float(sdf_clip)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        adapted = self.input_adapter(image)
        output = self.encoder(
            pixel_values=adapted,
            output_hidden_states=True,
            return_dict=True,
        )
        features = output.hidden_states
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)
        decoded = self.upsample(self.fuse(torch.cat(projected, dim=1)))
        decoded = F.interpolate(
            decoded,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        field = self.head(
            torch.cat([decoded, self.input_skip(image)], dim=1)
        ).squeeze(1)
        return self.sdf_clip * torch.tanh(field)


class LearnedMetricAlignmentUNet(nn.Module):
    """Contextual GR embeddings followed by an explicit 2-D cost decoder.

    The v20 alignment image compares scalar GR values at each candidate row.
    This model first embeds neighborhoods on the typewell and horizontal axes,
    forms a cosine cost volume, and then lets a whole-well U-Net refine the
    candidate path using the legal trajectory/history channels.
    """

    def __init__(
        self,
        input_channels: int = 21,
        embedding_channels: int = 32,
        base: int = 12,
        dropout: float = 0.0,
        metric_encoder: str = "local",
        decoder_context: str = "compact",
        cost_scale_init: float = 2.0,
        hint_scale_init: float = 4.0,
    ) -> None:
        super().__init__()
        if input_channels < 21:
            raise ValueError("learned metric model requires the 21-channel image")
        encoder_types = {
            "local": AxisEncoder,
            "dilated": DilatedAxisEncoder,
        }
        if metric_encoder not in encoder_types:
            raise ValueError(
                f"unknown metric encoder {metric_encoder!r}; "
                f"expected one of {tuple(encoder_types)}"
            )
        encoder_type = encoder_types[metric_encoder]
        self.metric_encoder = metric_encoder
        self.reference_encoder = encoder_type(4, embedding_channels)
        self.observed_encoder = encoder_type(4, embedding_channels)
        # cost, signed/absolute calibrated mismatch, pseudo mismatch,
        # state coordinate, visible path/mask, missingness, Z delta, local dip,
        # and local curvature.
        decoder_channels = {
            "compact": (0, 1, 2, 5, 6, 7, 8, 9, 13, 18),
            # Preserve the complete protected alignment representation.  This
            # form can inherit a fold-local v20 decoder exactly while a new
            # metric branch is learned on the current development roles.
            "full": tuple(range(19)),
        }
        if decoder_context not in decoder_channels:
            raise ValueError(
                f"unknown decoder context {decoder_context!r}; "
                f"expected one of {tuple(decoder_channels)}"
            )
        self.decoder_context = decoder_context
        self.decoder_channels = decoder_channels[decoder_context]
        self.decoder = AlignmentUNet(
            input_channels=1 + len(self.decoder_channels),
            base=base,
            dropout=dropout,
        )
        auxiliary_channels = max(16, base)
        groups = min(8, auxiliary_channels)
        while auxiliary_channels % groups:
            groups -= 1
        self.sdf_auxiliary = nn.Sequential(
            nn.Conv2d(1 + len(self.decoder_channels), auxiliary_channels, 3, padding=1),
            nn.GroupNorm(groups, auxiliary_channels),
            nn.SiLU(),
            nn.Conv2d(auxiliary_channels, 1, 1),
        )
        if cost_scale_init <= 0.0:
            raise ValueError("cost scale initialization must be positive")
        self.log_cost_scale = nn.Parameter(
            torch.tensor(float(cost_scale_init)).log()
        )
        self.hint_scale = nn.Parameter(torch.tensor(float(hint_scale_init)))

    def forward(
        self,
        image: torch.Tensor,
        return_cost: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = image[:, -2, :, 0]
        observed = image[:, -1, 0, :]
        reference_features = torch.stack(
            [
                reference,
                torch.gradient(reference, dim=-1)[0],
                image[:, 5, :, 0],
                image[:, 4, :, 0],
            ],
            dim=1,
        )
        observed_features = torch.stack(
            [
                observed,
                torch.gradient(observed, dim=-1)[0],
                image[:, 7, 0, :],
                1.0 - image[:, 8, 0, :],
            ],
            dim=1,
        )
        reference_embedding = F.normalize(
            self.reference_encoder(reference_features),
            dim=1,
            eps=1e-6,
        )
        observed_embedding = F.normalize(
            self.observed_encoder(observed_features),
            dim=1,
            eps=1e-6,
        )
        cost = torch.einsum(
            "bcs,bcl->bsl",
            reference_embedding,
            observed_embedding,
        )
        decoder_image = torch.cat(
            [
                cost[:, None],
                image[:, self.decoder_channels],
            ],
            dim=1,
        )
        logits = (
            self.decoder(decoder_image)
            + self.log_cost_scale.exp().clamp(max=20.0) * cost
            + self.hint_scale * image[:, 6]
        )
        sdf_field = 3.0 * torch.tanh(
            self.sdf_auxiliary(decoder_image).squeeze(1)
        )
        if return_cost:
            return logits, cost, sdf_field
        return logits


def load_alignment_decoder(
    model: LearnedMetricAlignmentUNet,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Initialize the full-context metric decoder from a v20 U-Net.

    Decoder channel zero is the new learned cost.  Channels one through
    nineteen reproduce the protected alignment image, so the old 19-channel
    stem is copied there and the new cost weight starts at zero.  Every other
    shape-compatible decoder tensor is copied without modification.
    """

    if model.decoder_context != "full":
        raise ValueError("alignment decoder loading requires full context")
    target = model.decoder.state_dict()
    copied_tensors = 0
    copied_parameters = 0
    for name, source in state_dict.items():
        if name == "stem.weight":
            destination = target.get(name)
            if (
                destination is None
                or source.ndim != 4
                or destination.shape[0] != source.shape[0]
                or destination.shape[1] != source.shape[1] + 1
                or destination.shape[2:] != source.shape[2:]
            ):
                continue
            destination.zero_()
            destination[:, 1:].copy_(source)
            copied_tensors += 1
            copied_parameters += source.numel()
        elif name in target and target[name].shape == source.shape:
            target[name].copy_(source)
            copied_tensors += 1
            copied_parameters += source.numel()
    model.decoder.load_state_dict(target)
    return {
        "copied_tensors": copied_tensors,
        "copied_parameters": copied_parameters,
    }


def learned_metric_objective(
    logits: torch.Tensor,
    cost: torch.Tensor,
    image: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    prefix_points: int,
    state_step: float,
    metric_temperature: float = 0.1,
    metric_weight: float = 0.5,
    sdf_weight: float = 0.25,
    sdf_field: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Path loss plus direct all-column hard-negative metric supervision."""
    path_loss, components = alignment_objective(
        logits,
        image,
        label,
        valid,
        offsets,
        prefix_points,
        objective_kind="categorical",
        coordinate_kind="tvt_delta",
    )
    point_metric = F.cross_entropy(
        cost.float() / float(metric_temperature),
        label,
        reduction="none",
    )
    prefix_valid = valid[:, :prefix_points].float()
    tail_valid = valid[:, prefix_points:].float()
    prefix_metric = (
        point_metric[:, :prefix_points] * prefix_valid
    ).sum() / prefix_valid.sum().clamp_min(1.0)
    tail_metric = (
        point_metric[:, prefix_points:] * tail_valid
    ).sum() / tail_valid.sum().clamp_min(1.0)
    metric = 0.5 * (prefix_metric + tail_metric)
    if sdf_field is None:
        raise ValueError("registered metric objective requires its SDF auxiliary")
    sdf_auxiliary, _ = sdf_dense_mse_objective(
        sdf_field,
        label,
        valid,
        prefix_points,
        state_step,
    )
    total = (
        path_loss
        + float(metric_weight) * metric
        + float(sdf_weight) * sdf_auxiliary
    )
    return total, {
        **components,
        "metric": metric,
        "prefix_metric": prefix_metric,
        "tail_metric": tail_metric,
        "sdf_auxiliary": sdf_auxiliary,
    }


def sdf_target(
    labels: torch.Tensor,
    state_count: int,
    state_step: float,
    sdf_scale: float,
    sdf_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.arange(
        state_count,
        dtype=torch.float32,
        device=labels.device,
    )[None, :, None]
    raw_feet = (state - labels[:, None, :].float()) * float(state_step)
    return (
        torch.clamp(raw_feet / float(sdf_scale), -sdf_clip, sdf_clip),
        raw_feet,
    )


def sdf_path_probability(
    field: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("SDF decoder temperature must be positive")
    return torch.softmax(-torch.abs(field.float()) / temperature, dim=1)


def expected_sdf_offset(
    field: torch.Tensor,
    offsets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    probability = sdf_path_probability(field, temperature)
    return torch.sum(probability * offsets[None, :, None], dim=1)


def sdf_alignment_objective(
    field: torch.Tensor,
    image: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    prefix_points: int,
    state_step: float,
    sdf_scale: float = 40.0,
    sdf_clip: float = 3.0,
    decoder_temperature: float = 0.035,
    field_weight: float = 1.0,
    categorical_weight: float = 0.35,
    regression_weight: float = 1.0,
    smoothness_weight: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dense SDF loss plus path and geological-surface consistency."""
    tail = slice(prefix_points, None)
    tail_valid = valid[:, tail].float()
    target, raw_feet = sdf_target(
        label,
        field.shape[1],
        state_step,
        sdf_scale,
        sdf_clip,
    )
    target = target[:, :, tail]
    raw_feet = raw_feet[:, :, tail]
    predicted_field = field[:, :, tail].float()

    # Preserve dense ranking but focus most weight within 32 ft of the path.
    state_weight = 0.15 + torch.exp(-torch.abs(raw_feet) / 24.0)
    field_mask = state_weight * tail_valid[:, None, :]
    point_field = F.smooth_l1_loss(
        predicted_field,
        target,
        reduction="none",
        beta=0.08,
    )
    dense_field = (point_field * field_mask).sum() / field_mask.sum().clamp_min(1.0)

    path_logits = -torch.abs(predicted_field) / decoder_temperature
    point_categorical = F.cross_entropy(
        path_logits,
        label[:, tail],
        reduction="none",
    )
    categorical = (
        point_categorical * tail_valid
    ).sum() / tail_valid.sum().clamp_min(1.0)

    prediction = expected_sdf_offset(
        predicted_field,
        offsets,
        decoder_temperature,
    )
    truth = offsets[label[:, tail]]
    point_regression = torch.square((prediction - truth) / sdf_scale)
    regression = (
        point_regression * tail_valid
    ).sum() / tail_valid.sum().clamp_min(1.0)

    z_relative = image[:, 9, 0, tail].float() * 200.0
    surface = prediction + z_relative
    smooth_valid = tail_valid[:, 2:] * tail_valid[:, 1:-1] * tail_valid[:, :-2]
    point_smoothness = torch.abs(torch.diff(surface, n=2, dim=1))
    surface_smoothness = (
        point_smoothness * smooth_valid
    ).sum() / smooth_valid.sum().clamp_min(1.0)

    total = (
        field_weight * dense_field
        + categorical_weight * categorical
        + regression_weight * regression
        + smoothness_weight * surface_smoothness
    )
    return total, {
        "dense_field": dense_field,
        "categorical": categorical,
        "regression": regression,
        "surface_smoothness": surface_smoothness,
    }


def sdf_dense_mse_objective(
    field: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
    prefix_points: int,
    state_step: float,
    sdf_scale: float = 40.0,
    sdf_clip: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Faithful dense SDF regression control from the public formulation."""
    target, _ = sdf_target(
        label,
        field.shape[1],
        state_step,
        sdf_scale,
        sdf_clip,
    )
    mask = valid[:, None, prefix_points:].float()
    squared = torch.square(
        field[:, :, prefix_points:].float()
        - target[:, :, prefix_points:]
    )
    dense_field = (squared * mask).sum() / (
        mask.sum() * field.shape[1]
    ).clamp_min(1.0)
    return dense_field, {"dense_field": dense_field}


def decode_sdf_numpy(
    field: np.ndarray,
    offsets: np.ndarray,
    temperature: float,
) -> np.ndarray:
    score = -np.abs(field.astype(np.float64)) / float(temperature)
    score -= np.max(score, axis=0, keepdims=True)
    probability = np.exp(score)
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-300)
    return np.sum(probability * offsets[:, None], axis=0)
