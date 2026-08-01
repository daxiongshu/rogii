from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from limited_query_cv import train_v5_run6_pretrained_categorical_segformer as parent


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "trajectory_conditioned_pretrained_categorical_segmentation"
VARIANT = "mit_b0_three_plane_trajectory_categorical"
SEED = 20262301
PARENT_SOURCE_SHA256 = (
    "170c8747f8f451508a65d34233ef93826ff189bf2a94675a8f761ccd2b475c24"
)
TRAJECTORY_CHANNELS = (
    "visible",
    "missing_gr",
    "z_delta",
    "md_delta",
    "x_delta",
    "y_delta",
    "dz_dmd",
    "dx_dmd",
    "dy_dmd",
    "azimuth_cos",
    "azimuth_sin",
    "vertical_curvature",
)
_ORIGINAL_PARENT_FILE = Path(parent.__file__)


def configure_sources() -> None:
    if parent.local.parent.core.sha256(_ORIGINAL_PARENT_FILE) != PARENT_SOURCE_SHA256:
        raise RuntimeError("A830 frozen A820 source mismatch")
    if (
        not parent.PRETRAINED_WEIGHT.is_file()
        or parent.local.parent.core.sha256(parent.PRETRAINED_WEIGHT)
        != parent.PRETRAINED_WEIGHT_SHA256
    ):
        raise RuntimeError("A830 local MiT-B0 pretrained weight mismatch")
    parent.local.configure_parent()


def make_example(
    well,
    cut: int,
    grid,
    *,
    synthetic_rng: np.random.Generator | None = None,
    synthetic_hardness: float = 0.0,
):
    image, extra, labels, valid, metadata = (
        parent.local.parent.core.make_example(
            well,
            cut,
            grid,
            "scratch_calibrated_local_b12",
            synthetic_rng=synthetic_rng,
            synthetic_hardness=synthetic_hardness,
        )
    )
    three = parent.three_planes(image, extra)
    trajectory_volume = image[7:19]
    repeated = np.broadcast_to(
        trajectory_volume[:, :1, :], trajectory_volume.shape
    )
    if not np.array_equal(trajectory_volume, repeated):
        raise RuntimeError("A830 trajectory channels are not row-constant")
    combined = np.concatenate((three, trajectory_volume), axis=0).astype(
        np.float32
    )
    if (
        combined.shape
        != (3 + len(TRAJECTORY_CHANNELS), len(grid.offsets), grid.length)
        or not np.isfinite(combined).all()
    ):
        raise RuntimeError(f"invalid A830 input {combined.shape}")
    return combined, labels, valid, metadata


class TrajectoryBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.conv1(F.silu(self.norm1(values)))
        values = self.conv2(F.silu(self.norm2(values)))
        return values + residual


class TrajectoryCategoricalSegformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import SegformerModel

        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b0",
            local_files_only=True,
        )
        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, 64, 1) for channels in hidden_sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * len(hidden_sizes), 64, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.trajectory_stem = nn.Sequential(
            nn.Conv1d(len(TRAJECTORY_CHANNELS), 64, 7, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.trajectory_blocks = nn.Sequential(
            *[
                TrajectoryBlock(64, dilation)
                for dilation in (1, 4, 16, 64)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(64 + 16 + 64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        image = values[:, :3]
        trajectory = values[:, 3:, 0, :]
        output = self.encoder(
            pixel_values=image,
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
        decoded = self.fuse(torch.cat(projected, dim=1))
        decoded = F.interpolate(
            decoded,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        trajectory = self.trajectory_blocks(
            self.trajectory_stem(trajectory)
        )
        trajectory = trajectory[:, :, None, :].expand(
            -1, -1, image.shape[-2], -1
        )
        return self.head(
            torch.cat(
                (decoded, self.input_skip(image), trajectory), dim=1
            )
        ).squeeze(1)


def optimizer_for(
    model: TrajectoryCategoricalSegformer,
) -> torch.optim.Optimizer:
    decoder = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 5e-5},
            {"params": decoder, "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )


def trajectory_perturbation_contract(well, grid) -> dict[str, object]:
    original, _, _, _ = make_example(well, well.anchor_index, grid)
    hidden = np.arange(len(well.md)) > well.anchor_index
    progress = np.zeros(len(well.md), dtype=np.float64)
    progress[hidden] = np.linspace(0.0, 1.0, int(hidden.sum()))
    perturbed = dataclasses.replace(
        well,
        x=well.x + 300.0 * progress,
        y=well.y + 150.0 * np.square(progress),
        z=well.z + 20.0 * np.sin(np.pi * progress),
    )
    changed, _, _, _ = make_example(
        perturbed, perturbed.anchor_index, grid
    )
    return {
        "three_planes_unchanged": bool(
            np.array_equal(original[:3], changed[:3])
        ),
        "trajectory_stream_changed": bool(
            not np.array_equal(original[3:], changed[3:])
        ),
        "trajectory_changed_channels": int(
            np.sum(
                [
                    not np.array_equal(original[index], changed[index])
                    for index in range(3, original.shape[0])
                ]
            )
        ),
    }


def gate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    configure_sources()
    parent.seed_all(SEED)
    identifiers, original = parent.local.parent.core.development_layout(
        args.data_root
    )
    development = np.flatnonzero(original < 4)
    wells = parent.local.parent.core.load_ids(
        identifiers, development, args.data_root, args.workers
    )
    training = [
        well
        for well, fold in zip(wells, original[development])
        if int(fold) != args.fold
    ]
    grid = parent.local.local_config()
    device = torch.device(args.device)
    fixed = parent.stack_examples(
        training[:2], grid, device, synthetic=True
    )
    untouched = parent.stack_examples(
        training[2:4], grid, device, synthetic=False
    )
    model = TrajectoryCategoricalSegformer().to(device)
    zero_head = bool(
        torch.count_nonzero(model.head[-1].weight) == 0
        and torch.count_nonzero(model.head[-1].bias) == 0
    )
    model.eval()
    with torch.no_grad():
        initial_fixed = parent.sampled_rmse(
            model(fixed[0]), fixed[1], fixed[2], grid
        )
        initial_untouched = parent.sampled_rmse(
            model(untouched[0]), untouched[1], untouched[2], grid
        )
    optimizer = optimizer_for(model)
    nonfinite = 0
    for step in range(args.overfit_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed[0])
        loss = parent.local.parent.pure_ce(
            logits, fixed[1], fixed[2], grid.prefix_points
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite A830 gate loss")
        loss.backward()
        count = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += count
        if count:
            raise FloatingPointError("nonfinite A830 gate gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {"step": step + 1, "loss": float(loss.detach())},
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        final_fixed = parent.sampled_rmse(
            model(fixed[0]), fixed[1], fixed[2], grid
        )
        final_untouched = parent.sampled_rmse(
            model(untouched[0]), untouched[1], untouched[2], grid
        )
    input_contract = {
        "shape": list(untouched[0][0].shape),
        "finite": bool(torch.isfinite(untouched[0]).all()),
        "typewell_column_constant": bool(
            torch.equal(
                untouched[0][0, 0],
                untouched[0][0, 0, :, :1].expand_as(
                    untouched[0][0, 0]
                ),
            )
        ),
        "horizontal_state_constant": bool(
            torch.equal(
                untouched[0][0, 1],
                untouched[0][0, 1, :1, :].expand_as(
                    untouched[0][0, 1]
                ),
            )
        ),
        "history_bounded": bool(
            torch.all(
                (untouched[0][:, 2] >= 0.0)
                & (untouched[0][:, 2] <= 1.0)
            )
        ),
        "trajectory_state_constant": bool(
            torch.equal(
                untouched[0][:, 3:],
                untouched[0][:, 3:, :1, :].expand_as(
                    untouched[0][:, 3:]
                ),
            )
        ),
        "hidden_target_invariant": parent.hidden_invariant(
            training[2], grid
        ),
        "trajectory_perturbation": trajectory_perturbation_contract(
            training[2], grid
        ),
    }
    dummy = np.linspace(10000.0, 10100.0, 97)
    zero_share_exact = bool(
        np.array_equal(
            dummy + 0.0 * np.linspace(-2.0, 3.0, 97), dummy
        )
    )
    capacity = {
        "steps": args.overfit_steps,
        "initial_fixed_sampled_rmse": initial_fixed,
        "final_fixed_sampled_rmse": final_fixed,
        "initial_untouched_sampled_rmse": initial_untouched,
        "final_untouched_sampled_rmse": final_untouched,
        "nonfinite_gradient_elements": nonfinite,
        "zero_head": zero_head,
    }
    passed = (
        input_contract["shape"]
        == [15, len(grid.offsets), grid.length]
        and input_contract["finite"]
        and input_contract["typewell_column_constant"]
        and input_contract["horizontal_state_constant"]
        and input_contract["history_bounded"]
        and input_contract["trajectory_state_constant"]
        and input_contract["hidden_target_invariant"]
        and input_contract["trajectory_perturbation"]
        == {
            "three_planes_unchanged": True,
            "trajectory_stream_changed": True,
            "trajectory_changed_channels": 9,
        }
        and zero_head
        and final_fixed < 0.20 * initial_fixed
        and np.isfinite(final_untouched)
        and abs(final_untouched - initial_untouched) > 1e-6
        and nonfinite == 0
        and zero_share_exact
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": VARIANT,
        "status": "passed" if passed else "failed",
        "input_contract": input_contract,
        "trajectory_channels": list(TRAJECTORY_CHANNELS),
        "trajectory_dilations": [1, 4, 16, 64],
        "capacity": capacity,
        "pretrained_weight": str(parent.PRETRAINED_WEIGHT),
        "pretrained_weight_sha256": parent.PRETRAINED_WEIGHT_SHA256,
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "source_sha256": parent.local.parent.core.sha256(Path(__file__)),
        "d570_sha256": parent.D570_SHA256,
        "d570_zero_share_exact": zero_share_exact,
        "epochs": parent.EPOCHS,
        "pure_synthetic_epochs": parent.PURE_SYNTHETIC_EPOCHS,
        "mixed_synthetic_probability": parent.MIXED_SYNTHETIC_PROBABILITY,
        "snapshots": list(parent.SNAPSHOTS),
        "temperatures": list(parent.TEMPERATURES),
        "shares": list(parent.SHARES),
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    parent.local.parent.core.write_json(
        args.output_dir / "result.json", result
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("A830 implementation gate failed")


def verify_gate(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    if (
        result.get("status") != "passed"
        or result.get("source_sha256")
        != parent.local.parent.core.sha256(Path(__file__))
        or result.get("target_metric_computed") is not False
        or result.get("f4_loaded") is not False
        or result.get("pretrained_weight_sha256")
        != parent.PRETRAINED_WEIGHT_SHA256
        or result.get("parent_source_sha256") != PARENT_SOURCE_SHA256
    ):
        raise RuntimeError("A830 gate absent or invalid")
    return result


def configure_parent_module() -> None:
    if parent.local.parent.core.sha256(_ORIGINAL_PARENT_FILE) != PARENT_SOURCE_SHA256:
        raise RuntimeError("A830 frozen A820 parent mismatch")
    parent.RUN_ID = RUN_ID
    parent.FAMILY = FAMILY
    parent.VARIANT = VARIANT
    parent.SEED = SEED
    parent.make_example = make_example
    parent.CategoricalSegformer = TrajectoryCategoricalSegformer
    parent.optimizer_for = optimizer_for
    parent.configure_sources = configure_sources
    parent.gate = gate
    parent.verify_gate = verify_gate
    parent.__file__ = __file__


def main() -> None:
    configure_parent_module()
    parent.main()


if __name__ == "__main__":
    main()
