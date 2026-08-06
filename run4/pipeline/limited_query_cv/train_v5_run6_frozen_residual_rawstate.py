from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from torch import nn
from torch.utils.data import DataLoader, Dataset

from limited_query_cv.evaluate_v5_run6_protected_mode_bank import CHECKPOINT_ROOT
from limited_query_cv.train_v5_run6_absolute_state_continuation import (
    DEFAULT_D570,
    absolute_planes,
    load_d570,
)
from limited_query_cv.train_v5_run6_categorical_expansion import (
    config,
    development_layout,
    load_ids,
)
from limited_query_cv.train_v5_run6_pure_categorical import pair_planes, pure_ce
from limited_query_cv.train_v5_run6_purece_continuation import PARENTS
from rogii.alignment import AlignmentUNet, expected_offset, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "frozen_parent_raw_state_residual_unet"
VARIANTS = {
    "delta_pair_b8": {
        "extra_channels": 2,
        "base": 8,
        "seed": 20261823,
    },
    "delta_absolute_b8": {
        "extra_channels": 8,
        "base": 8,
        "seed": 20261829,
    },
    "delta_multiscale_absolute_b12": {
        "extra_channels": 14,
        "base": 12,
        "seed": 20261837,
    },
}
SYNTHETIC_EPOCHS = 4
EPOCHS = 20
SNAPSHOTS = (4, 8, 12, 16, 20)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
AMPLITUDES = (0.5, 1.0, 2.0)
SHARES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50)
DEFAULT_AGGREGATE_GATE = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050/"
    "frozen_residual_rawstate_aggregate_gate.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _multiscale_pair(pair: np.ndarray) -> np.ndarray:
    observed, expected = pair.astype(np.float32)
    observed_9 = uniform_filter1d(
        observed, size=9, axis=1, mode="nearest"
    )
    expected_9 = uniform_filter1d(
        expected, size=9, axis=0, mode="nearest"
    )
    observed_33 = uniform_filter1d(
        observed, size=33, axis=1, mode="nearest"
    )
    expected_33 = uniform_filter1d(
        expected, size=33, axis=0, mode="nearest"
    )
    observed_gradient = np.gradient(observed, axis=1)
    expected_gradient = np.gradient(expected, axis=0)
    return np.stack(
        [
            observed_9,
            expected_9,
            observed_33,
            expected_33,
            np.clip(observed_gradient, -3.0, 3.0),
            np.clip(expected_gradient, -3.0, 3.0),
        ],
        axis=0,
    ).astype(np.float32)


def residual_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
    variant: str,
) -> np.ndarray:
    pair = pair_planes(image, metadata)[19:].astype(np.float32)
    if variant == "delta_pair_b8":
        extra = pair
    else:
        absolute = absolute_planes(well, cut, image, metadata, grid)
        if variant == "delta_absolute_b8":
            extra = absolute
        elif variant == "delta_multiscale_absolute_b12":
            extra = np.concatenate(
                (absolute, _multiscale_pair(pair)), axis=0
            ).astype(np.float32)
        else:
            raise ValueError(variant)
    expected = int(VARIANTS[variant]["extra_channels"])
    if (
        extra.shape != (expected, image.shape[1], image.shape[2])
        or not np.isfinite(extra).all()
    ):
        raise RuntimeError(
            f"{well.well_id}: invalid {variant} residual planes {extra.shape}"
        )
    return extra


def make_example(
    well: Well,
    cut: int,
    grid,
    variant: str,
    synthetic_rng: np.random.Generator | None = None,
    synthetic_hardness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    image, labels, metadata = make_alignment_example(
        well,
        cut,
        grid,
        synthetic_rng=synthetic_rng,
        synthetic_profile="forward_extreme",
        synthetic_hardness=synthetic_hardness,
        coordinate_kind="tvt_delta",
    )
    extra = residual_planes(well, cut, image, metadata, grid, variant)
    valid = np.asarray(metadata["valid_positions"], dtype=np.float32)
    return (
        image.astype(np.float32),
        extra,
        labels.astype(np.int64),
        valid,
        metadata,
    )


class FrozenParentResidualUNet(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        specification = VARIANTS[variant]
        self.variant = variant
        self.parent = AlignmentUNet(
            input_channels=19, base=12, dropout=0.10
        )
        self.residual = AlignmentUNet(
            input_channels=19 + int(specification["extra_channels"]),
            base=int(specification["base"]),
            dropout=0.10,
        )
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)

    def residual_input(
        self, image: torch.Tensor, extra: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((image, extra), dim=1)

    def forward(
        self, image: torch.Tensor, extra: torch.Tensor
    ) -> torch.Tensor:
        self.parent.eval()
        with torch.no_grad():
            parent = self.parent(image)
        return parent + self.residual(self.residual_input(image, extra))


def initialized_model(
    fold: int,
    variant: str,
    device: torch.device,
) -> tuple[FrozenParentResidualUNet, Path]:
    checkpoint = CHECKPOINT_ROOT / PARENTS["base12"].format(fold=fold)
    model = FrozenParentResidualUNet(variant).to(device)
    parent_state = torch.load(
        checkpoint, map_location=device, weights_only=True
    )
    model.parent.load_state_dict(parent_state, strict=True)
    for parameter in model.parent.parameters():
        parameter.requires_grad_(False)
    model.residual.head.weight.data.zero_()
    model.residual.head.bias.data.zero_()
    return model, checkpoint


class ResidualDataset(Dataset):
    def __init__(
        self,
        wells: list[Well],
        grid,
        variant: str,
        repeats: int,
        seed: int,
    ) -> None:
        self.wells = wells
        self.grid = grid
        self.variant = variant
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.wells) * self.repeats

    def __getitem__(self, index: int):
        rng = np.random.default_rng(
            self.seed + self.epoch * len(self) + index
        )
        repeat = index // len(self.wells)
        well = self.wells[index % len(self.wells)]
        if repeat == 0:
            cut = well.anchor_index
        else:
            low = max(100, int(0.14 * len(well.md)))
            high = min(len(well.md) - 100, int(0.42 * len(well.md)))
            cut = int(rng.integers(low, high + 1))
        synthetic = self.epoch < SYNTHETIC_EPOCHS
        hardness = (
            0.70 * (self.epoch + 1) / SYNTHETIC_EPOCHS
            if synthetic
            else 0.0
        )
        image, extra, labels, valid, _ = make_example(
            well,
            cut,
            self.grid,
            self.variant,
            synthetic_rng=rng if synthetic else None,
            synthetic_hardness=hardness,
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(extra),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
        )


def loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        generator=generator,
    )


def stack_examples(
    wells: list[Well],
    grid,
    variant: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    examples = [
        make_example(well, well.anchor_index, grid, variant)
        for well in wells
    ]
    return (
        torch.from_numpy(np.stack([value[0] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[1] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[2] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[3] for value in examples])).to(device),
    )


@torch.no_grad()
def sampled_rmse_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    grid,
) -> float:
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=logits.device
    )
    prediction = expected_offset(logits, offsets)
    truth = offsets[labels]
    selected = valid > 0.5
    selected[:, : grid.prefix_points] = False
    return float(
        torch.sqrt(
            torch.mean(torch.square(prediction[selected] - truth[selected]))
        )
    )


def hidden_feature_invariance(
    well: Well,
    grid,
    variant: str,
) -> bool:
    records = []
    for mode in ("original", "reverse", "nan"):
        target = well.tvt.copy()
        hidden = np.arange(len(target)) > well.anchor_index
        if mode == "reverse":
            target[hidden] = target[hidden][::-1]
        elif mode == "nan":
            target[hidden] = np.nan
        modified = dataclasses.replace(well, tvt=target)
        image, extra, _, _, _ = make_example(
            modified, well.anchor_index, grid, variant
        )
        records.append((image, extra))
    return all(
        np.array_equal(records[0][0], value[0])
        and np.array_equal(records[0][1], value[1])
        for value in records[1:]
    )


def fixed_gate(
    wells: list[Well],
    grid,
    fold: int,
    variant: str,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    source_path = Path(__file__)
    cpu = torch.device("cpu")
    cpu_model, checkpoint = initialized_model(fold, variant, cpu)
    cpu_model.eval()
    parity_example = make_example(
        wells[0], wells[0].anchor_index, grid, variant
    )
    parity_image = torch.from_numpy(parity_example[0])[None]
    parity_extra = torch.from_numpy(parity_example[1])[None]
    with torch.no_grad():
        parent_logits = cpu_model.parent(parity_image)
        initialized_logits = cpu_model(parity_image, parity_extra)
    parity = float(torch.max(torch.abs(parent_logits - initialized_logits)))
    checkpoint_state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    inherited_exact = all(
        torch.equal(cpu_model.parent.state_dict()[name], value)
        for name, value in checkpoint_state.items()
    )
    residual_head_zero = bool(
        torch.count_nonzero(cpu_model.residual.head.weight) == 0
        and torch.count_nonzero(cpu_model.residual.head.bias) == 0
    )
    parent_gradient_free = all(
        not parameter.requires_grad
        for parameter in cpu_model.parent.parameters()
    )
    del cpu_model

    model, _ = initialized_model(fold, variant, device)
    fixed = stack_examples(wells[:2], grid, variant, device)
    untouched = stack_examples(wells[2:4], grid, variant, device)
    model.parent.eval()
    model.residual.eval()
    with torch.no_grad():
        fixed_parent = model.parent(fixed[0])
        untouched_parent = model.parent(untouched[0])
        fixed_initial_logits = fixed_parent + model.residual(
            model.residual_input(fixed[0], fixed[1])
        )
        untouched_initial_logits = untouched_parent + model.residual(
            model.residual_input(untouched[0], untouched[1])
        )
    initial = sampled_rmse_from_logits(
        fixed_initial_logits, fixed[2], fixed[3], grid
    )
    untouched_initial = sampled_rmse_from_logits(
        untouched_initial_logits, untouched[2], untouched[3], grid
    )

    optimizer = torch.optim.AdamW(
        model.residual.parameters(), lr=7e-4, weight_decay=1e-4
    )
    nonfinite = 0
    for _ in range(steps):
        model.residual.train()
        optimizer.zero_grad(set_to_none=True)
        logits = fixed_parent + model.residual(
            model.residual_input(fixed[0], fixed[1])
        )
        loss = pure_ce(logits, fixed[2], fixed[3], grid.prefix_points)
        loss.backward()
        count = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.residual.parameters()
            if parameter.grad is not None
        )
        nonfinite += count
        if count:
            raise FloatingPointError("nonfinite A680 gate gradient")
        nn.utils.clip_grad_norm_(model.residual.parameters(), 5.0)
        optimizer.step()
    model.residual.eval()
    with torch.no_grad():
        fixed_final_logits = fixed_parent + model.residual(
            model.residual_input(fixed[0], fixed[1])
        )
        untouched_final_logits = untouched_parent + model.residual(
            model.residual_input(untouched[0], untouched[1])
        )
    final = sampled_rmse_from_logits(
        fixed_final_logits, fixed[2], fixed[3], grid
    )
    untouched_final = sampled_rmse_from_logits(
        untouched_final_logits, untouched[2], untouched[3], grid
    )

    invariant = hidden_feature_invariance(wells[2], grid, variant)
    synthetic = make_example(
        wells[1],
        wells[1].anchor_index,
        grid,
        variant,
        synthetic_rng=np.random.default_rng(20261901),
        synthetic_hardness=0.70,
    )
    synthetic_target = np.asarray(
        synthetic[4]["target_offset"], dtype=np.float32
    )
    synthetic_labels = synthetic[2]
    synthetic_valid = synthetic[3] > 0.5
    discretization = float(
        np.max(
            np.abs(
                grid.offsets[synthetic_labels[synthetic_valid]]
                - synthetic_target[synthetic_valid]
            )
        )
    )
    with torch.no_grad():
        offsets = torch.as_tensor(
            grid.offsets, dtype=torch.float32, device=device
        )
        final_path = expected_offset(untouched_final_logits, offsets)
        parent_path = expected_offset(untouched_parent, offsets)
        correction = (
            final_path[:, grid.prefix_points :]
            - parent_path[:, grid.prefix_points :]
        )
        correction -= correction[:, :1]
        bounded = torch.clamp(correction, -64.0, 64.0)
    dummy = np.linspace(-1.0, 1.0, 31, dtype=np.float32)
    parent = np.linspace(10000.0, 10001.0, 31, dtype=np.float32)
    no_op = np.array_equal(parent + 0.0 * dummy, parent)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": variant,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_sha256": sha256(source_path),
        "feature_channels": 19
        + int(VARIANTS[variant]["extra_channels"]),
        "residual_base": int(VARIANTS[variant]["base"]),
        "residual_parameters": sum(
            parameter.numel() for parameter in model.residual.parameters()
        ),
        "maximum_initial_logit_difference": parity,
        "inherited_parent_tensors_bit_identical": inherited_exact,
        "residual_head_exact_zero": residual_head_zero,
        "parent_gradient_free": parent_gradient_free,
        "initial_fixed_rmse": initial,
        "final_fixed_rmse": final,
        "initial_untouched_rmse": untouched_initial,
        "final_untouched_rmse": untouched_final,
        "nonfinite_gradient_elements": nonfinite,
        "hidden_target_features_bit_identical": invariant,
        "synthetic_target_discretization_max_abs": discretization,
        "first_hidden_delta_anchor_exact": bool(
            torch.all(bounded[:, 0] == 0).item()
        ),
        "maximum_bounded_delta": float(torch.max(torch.abs(bounded))),
        "d570_zero_share_exact": no_op,
        "steps": steps,
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    result["status"] = (
        "passed"
        if parity <= 1e-6
        and inherited_exact
        and residual_head_zero
        and parent_gradient_free
        and final < 0.20 * initial
        and nonfinite == 0
        and np.isfinite(untouched_final)
        and abs(untouched_final - untouched_initial) > 1e-6
        and invariant
        and discretization <= 0.51
        and result["first_hidden_delta_anchor_exact"]
        and result["maximum_bounded_delta"] <= 64.0
        and no_op
        else "failed"
    )
    return result


@torch.no_grad()
def predict(
    model: FrozenParentResidualUNet,
    wells: list[Well],
    grid,
    variant: str,
    device: torch.device,
    d570_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[float, np.ndarray],
    dict[float, np.ndarray],
]:
    model.parent.eval()
    model.residual.eval()
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=device
    )
    candidate_parts = {temperature: [] for temperature in TEMPERATURES}
    parent_parts = {temperature: [] for temperature in TEMPERATURES}
    truth_parts, starts = [], [0]
    for well in wells:
        image, extra, _, valid, metadata = make_example(
            well, well.anchor_index, grid, variant
        )
        image_tensor = torch.from_numpy(image)[None].to(device)
        extra_tensor = torch.from_numpy(extra)[None].to(device)
        parent_logits = model.parent(image_tensor)
        candidate_logits = parent_logits + model.residual(
            model.residual_input(image_tensor, extra_tensor)
        )
        positions = np.asarray(metadata["positions"], dtype=np.float64)
        sampled_valid = valid > 0.5
        visible = sampled_valid & (positions <= well.anchor_index + 1e-6)
        visible_rows = np.flatnonzero(visible)[-32:]
        tail_rows = sampled_valid & (positions > well.anchor_index + 1e-6)
        visible_truth = np.interp(
            positions[visible_rows],
            np.arange(len(well.tvt_input), dtype=np.float64),
            np.nan_to_num(
                well.tvt_input, nan=well.tvt_input[well.anchor_index]
            ),
        )
        for temperature in TEMPERATURES:
            for logits, destination in (
                (candidate_logits, candidate_parts),
                (parent_logits, parent_parts),
            ):
                decoded = (
                    expected_offset(logits / temperature, offsets)[0]
                    .float()
                    .cpu()
                    .numpy()
                )
                sampled = float(metadata["anchor_tvt"]) + decoded
                sampled -= float(
                    np.median(sampled[visible_rows] - visible_truth)
                )
                native = np.interp(
                    well.tail_indices,
                    positions[tail_rows],
                    sampled[tail_rows],
                ).astype(np.float32)
                destination[temperature].append(native)
        truth = well.tvt[well.tail_indices].astype(np.float32)
        truth_parts.append(truth)
        starts.append(starts[-1] + len(truth))
    ids = np.asarray([well.well_id for well in wells])
    starts_array = np.asarray(starts, dtype=np.int64)
    truth = np.concatenate(truth_parts)
    d570 = load_d570(d570_path, ids, starts_array, truth)
    candidate, delta = {}, {}
    for temperature in TEMPERATURES:
        candidate_values = np.concatenate(candidate_parts[temperature])
        parent_values = np.concatenate(parent_parts[temperature])
        delta_values = candidate_values - parent_values
        for left, right in zip(starts_array[:-1], starts_array[1:]):
            candidate_values[left:right] -= (
                candidate_values[left] - d570[left]
            )
            delta_values[left:right] -= delta_values[left]
        candidate[temperature] = candidate_values.astype(np.float32)
        delta[temperature] = np.clip(
            delta_values, -64.0, 64.0
        ).astype(np.float32)
    return ids, starts_array, d570, truth, candidate, delta


def evaluate(
    d570: np.ndarray,
    truth: np.ndarray,
    candidate: dict[float, np.ndarray],
    delta: dict[float, np.ndarray],
) -> dict[str, object]:
    parent = float(
        np.sqrt(
            np.mean(
                np.square(
                    d570.astype(np.float64) - truth.astype(np.float64)
                )
            )
        )
    )
    records = []
    for temperature in TEMPERATURES:
        for share in SHARES:
            replacement = d570 + share * (
                candidate[temperature] - d570
            )
            score = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            replacement.astype(np.float64)
                            - truth.astype(np.float64)
                        )
                    )
                )
            )
            records.append(
                {
                    "treatment": "candidate_replacement",
                    "temperature": temperature,
                    "amplitude": 1.0,
                    "share": share,
                    "effective_scale": share,
                    "rmse": score,
                    "gain": parent - score,
                }
            )
        for amplitude in AMPLITUDES:
            for share in SHARES:
                prediction = d570 + amplitude * share * delta[temperature]
                score = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                prediction.astype(np.float64)
                                - truth.astype(np.float64)
                            )
                        )
                    )
                )
                records.append(
                    {
                        "treatment": "parent_delta",
                        "temperature": temperature,
                        "amplitude": amplitude,
                        "share": share,
                        "effective_scale": amplitude * share,
                        "rmse": score,
                        "gain": parent - score,
                    }
                )
    best = min(
        records,
        key=lambda item: (
            item["rmse"],
            item["share"],
            item["effective_scale"],
            item["treatment"],
            item["temperature"],
        ),
    )
    return {
        "parent_rmse": parent,
        "best_same_held_diagnostic_only": best,
        "grid": records,
    }


def verify_aggregate_gate(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "passed"
        or payload.get("source_sha256") != sha256(Path(__file__))
        or payload.get("target_metric_computed") is not False
        or payload.get("f4_loaded") is not False
    ):
        raise RuntimeError("A680 aggregate gate is absent or invalid")
    return payload


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    specification = VARIANTS[args.variant]
    seed = int(specification["seed"])
    seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    identifiers, original = development_layout(args.data_root)
    development = np.flatnonzero(original < 4)
    all_wells = load_ids(
        identifiers, development, args.data_root, args.workers
    )
    lookup = {well.well_id: well for well in all_wells}
    train_ids = identifiers[(original < 4) & (original != args.fold)]
    held_ids = identifiers[original == args.fold]
    train_wells = [lookup[str(well_id)] for well_id in train_ids]
    held_wells = [lookup[str(well_id)] for well_id in held_ids]
    grid = config()

    if args.overfit_only:
        result = fixed_gate(
            train_wells,
            grid,
            args.fold,
            args.variant,
            device,
            args.overfit_steps,
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError("A680 gate failed")
        return

    aggregate = verify_aggregate_gate(args.aggregate_gate)
    model, checkpoint = initialized_model(
        args.fold, args.variant, device
    )
    dataset = ResidualDataset(
        train_wells,
        grid,
        args.variant,
        args.repeats,
        seed,
    )
    optimizer = torch.optim.AdamW(
        model.residual.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    history, snapshots = [], []
    nonfinite = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        dataset.epoch = epoch - 1
        batches = loader(
            dataset,
            args.batch_size,
            args.loader_workers,
            seed + epoch,
        )
        model.parent.eval()
        model.residual.train()
        losses, gradients = [], []
        for image, extra, labels, valid in batches:
            image = image.to(device, non_blocking=True)
            extra = extra.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                parent_logits = model.parent(image)
            logits = parent_logits + model.residual(
                model.residual_input(image, extra)
            )
            loss = pure_ce(logits, labels, valid, grid.prefix_points)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite A680 loss")
            loss.backward()
            count = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.residual.parameters()
                if parameter.grad is not None
            )
            nonfinite += count
            if count:
                raise FloatingPointError("nonfinite A680 gradient")
            gradients.append(
                float(
                    nn.utils.clip_grad_norm_(
                        model.residual.parameters(), 5.0
                    )
                )
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "stage": (
                "forward_extreme_synthetic"
                if epoch <= SYNTHETIC_EPOCHS
                else "real_random_cut"
            ),
            "mean_loss": float(np.mean(losses)),
            "mean_gradient_norm": float(np.mean(gradients)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            checkpoint_path = args.output_dir / f"epoch{epoch:03d}.pt"
            torch.save(model.residual.state_dict(), checkpoint_path)
            (
                ids,
                starts,
                d570,
                truth,
                candidate,
                delta,
            ) = predict(
                model,
                held_wells,
                grid,
                args.variant,
                device,
                args.d570,
            )
            evaluation = evaluate(d570, truth, candidate, delta)
            prediction_path = (
                args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            )
            arrays = {
                "fold": np.asarray(args.fold, dtype=np.int8),
                "variant": np.asarray(args.variant),
                "well_ids": ids,
                "row_starts": starts,
                "d570": d570,
                "truth": truth,
                "source_sha256": np.asarray(sha256(Path(__file__))),
                "checkpoint_sha256": np.asarray(
                    sha256(checkpoint_path)
                ),
                "parent_checkpoint_sha256": np.asarray(
                    sha256(checkpoint)
                ),
                "aggregate_gate_sha256": np.asarray(
                    sha256(args.aggregate_gate)
                ),
                "audit_fold_loaded": np.asarray(False),
                "confirmation_regroupings_loaded": np.asarray(False),
                "f4_loaded": np.asarray(False),
            }
            for temperature in TEMPERATURES:
                suffix = int(round(temperature * 100))
                arrays[f"candidate_t{suffix:03d}"] = candidate[temperature]
                arrays[f"delta_t{suffix:03d}"] = delta[temperature]
            np.savez_compressed(prediction_path, **arrays)
            snapshot = {
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "evaluation": evaluation,
            }
            snapshots.append(snapshot)
            print(
                json.dumps(
                    {
                        "snapshot": epoch,
                        "best": evaluation[
                            "best_same_held_diagnostic_only"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": args.variant,
        "fold": args.fold,
        "training_folds": [
            int(value)
            for value in sorted(
                set(original[(original < 4) & (original != args.fold)])
            )
        ],
        "held_wells": len(held_wells),
        "training_wells": len(train_wells),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_sha256": sha256(Path(__file__)),
        "aggregate_gate": str(args.aggregate_gate),
        "aggregate_gate_sha256": sha256(args.aggregate_gate),
        "specification": specification,
        "history": history,
        "snapshots": snapshots,
        "nonfinite_gradient_elements": nonfinite,
        "elapsed_seconds": time.time() - started,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d570", type=Path, default=DEFAULT_D570)
    parser.add_argument(
        "--aggregate-gate", type=Path, default=DEFAULT_AGGREGATE_GATE
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=1200)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
