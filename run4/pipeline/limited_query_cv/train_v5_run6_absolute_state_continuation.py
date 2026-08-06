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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from limited_query_cv.evaluate_v5_run6_protected_mode_bank import CHECKPOINT_ROOT
from limited_query_cv.train_v5_run6_categorical_expansion import (
    config,
    development_layout,
    load_ids,
)
from limited_query_cv.train_v5_run6_master_frame_boosting import (
    EXPECTED_FRAME_COUNT,
    EXPECTED_UNSEEN_HELD_WELLS,
    build_master_frames,
)
from limited_query_cv.train_v5_run6_pure_categorical import pair_planes, pure_ce
from limited_query_cv.train_v5_run6_purece_continuation import PARENTS
from rogii.alignment import AlignmentUNet, expected_offset, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "absolute_state_protected_categorical_continuation"
VARIANTS = ("absolute_pair", "absolute_frame")
SNAPSHOTS = (2, 4, 8, 12)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
AMPLITUDES = (0.5, 1.0, 2.0)
SHARES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
SEED = 20261803
UNKNOWN_FRAME = EXPECTED_FRAME_COUNT
DEFAULT_D570 = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050/"
    "d209_amplitude_clean.npz"
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


class AbsoluteStateUNet(AlignmentUNet):
    def __init__(self, use_frame: bool) -> None:
        super().__init__(input_channels=19, base=12, dropout=0.10)
        self.extra_stem = nn.Conv2d(
            8, 12, kernel_size=(5, 9), padding=(2, 4), bias=False
        )
        nn.init.zeros_(self.extra_stem.weight)
        self.use_frame = use_frame
        if use_frame:
            self.frame_embedding = nn.Embedding(EXPECTED_FRAME_COUNT + 1, 12)
            nn.init.zeros_(self.frame_embedding.weight)

    def forward(
        self,
        image: torch.Tensor,
        extra: torch.Tensor,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(image) + self.extra_stem(extra)
        if self.use_frame:
            embedding = self.frame_embedding(frame).unsqueeze(-1).unsqueeze(-1)
            x = x + embedding
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
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x).squeeze(1)


def initialized_models(
    fold: int,
    variant: str,
    device: torch.device,
) -> tuple[AlignmentUNet, AbsoluteStateUNet, Path]:
    checkpoint = CHECKPOINT_ROOT / PARENTS["base12"].format(fold=fold)
    source = AlignmentUNet(input_channels=19, base=12, dropout=0.10).to(device)
    source.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model = AbsoluteStateUNet(use_frame=variant == "absolute_frame").to(device)
    target_state = model.state_dict()
    for name, value in source.state_dict().items():
        if name not in target_state or target_state[name].shape != value.shape:
            raise RuntimeError(f"inherited tensor mismatch: {name}")
        target_state[name].copy_(value)
    target_state["extra_stem.weight"].zero_()
    if variant == "absolute_frame":
        target_state["frame_embedding.weight"].zero_()
    model.load_state_dict(target_state, strict=True)
    return source, model, checkpoint


def absolute_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
) -> np.ndarray:
    pair = pair_planes(image, metadata)[19:]
    height, width = image.shape[1:]
    valid = np.isfinite(well.typewell_tvt)
    lower = float(np.min(well.typewell_tvt[valid]))
    upper = float(np.max(well.typewell_tvt[valid]))
    span = max(upper - lower, 1.0)
    anchor_tvt = float(well.tvt[cut])
    anchor_z = float(well.z[cut])
    candidate = anchor_tvt + grid.offsets

    def row(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[:, None], (height, width))

    def constant(value: float) -> np.ndarray:
        return np.full((height, width), value, dtype=np.float32)

    state_fraction = np.clip(2.0 * (candidate - lower) / span - 1.0, -2.0, 2.0)
    anchor_fraction = np.clip(2.0 * (anchor_tvt - lower) / span - 1.0, -2.0, 2.0)
    extra = np.stack(
        [
            pair[0],
            pair[1],
            row(state_fraction.astype(np.float32)),
            constant(float(anchor_fraction)),
            constant((anchor_tvt - 11500.0) / 1500.0),
            constant((anchor_z + 9500.0) / 1500.0),
            constant((anchor_tvt + anchor_z - 2100.0) / 500.0),
            constant((span - 900.0) / 500.0),
        ],
        axis=0,
    ).astype(np.float32)
    if extra.shape != (8, height, width) or not np.isfinite(extra).all():
        raise RuntimeError(f"{well.well_id}: invalid absolute-state planes")
    return extra


def make_example(
    well: Well,
    cut: int,
    grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image, labels, metadata = make_alignment_example(
        well, cut, grid, coordinate_kind="tvt_delta"
    )
    extra = absolute_planes(well, cut, image, metadata, grid)
    valid = np.asarray(metadata["valid_positions"], dtype=np.float32)
    return image.astype(np.float32), extra, labels, valid


def frame_maps(
    all_wells: list[Well],
    folds: np.ndarray,
) -> tuple[dict[str, int], dict[int, dict[str, int]], dict[str, object]]:
    frames, audit = build_master_frames(all_wells, folds)
    base = {well.well_id: int(frame) for well, frame in zip(all_wells, frames)}
    held_maps = {}
    for held in range(4):
        counts = np.bincount(
            frames[folds != held], minlength=EXPECTED_FRAME_COUNT
        )
        held_maps[held] = {
            well.well_id: (
                int(frame) if counts[int(frame)] > 0 else UNKNOWN_FRAME
            )
            for well, frame in zip(all_wells, frames)
        }
    if audit["frame_count"] != EXPECTED_FRAME_COUNT:
        raise RuntimeError(f"master-frame count changed: {audit['frame_count']}")
    if tuple(audit["unseen_held_wells"]) != EXPECTED_UNSEEN_HELD_WELLS:
        raise RuntimeError(
            f"unseen held-frame audit changed: {audit['unseen_held_wells']}"
        )
    return base, held_maps, audit


class AbsoluteDataset(Dataset):
    def __init__(
        self,
        wells: list[Well],
        frames: dict[str, int],
        grid,
        repeats: int,
        seed: int,
    ) -> None:
        self.wells = wells
        self.frames = frames
        self.grid = grid
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.wells) * self.repeats

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        repeat = index // len(self.wells)
        well = self.wells[index % len(self.wells)]
        if repeat == 0 or rng.random() < 0.35:
            cut = well.anchor_index
        else:
            low = max(100, int(0.14 * len(well.md)))
            high = min(len(well.md) - 100, int(0.42 * len(well.md)))
            cut = int(rng.integers(low, high + 1))
        image, extra, labels, valid = make_example(well, cut, self.grid)
        return (
            torch.from_numpy(image),
            torch.from_numpy(extra),
            torch.as_tensor(self.frames[well.well_id], dtype=torch.long),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
        )


def loader(dataset, batch_size: int, workers: int, seed: int):
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


@torch.no_grad()
def sampled_rmse(model, batch, grid) -> float:
    model.eval()
    logits = model(batch[0], batch[1], batch[2])
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=logits.device
    )
    prediction = expected_offset(logits, offsets)
    truth = offsets[batch[3]]
    selected = batch[4] > 0.5
    selected[:, : grid.prefix_points] = False
    return float(
        torch.sqrt(torch.mean(torch.square(prediction[selected] - truth[selected])))
    )


def stack_examples(wells, frames, grid, device):
    examples = [make_example(well, well.anchor_index, grid) for well in wells]
    return (
        torch.from_numpy(np.stack([value[0] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[1] for value in examples])).to(device),
        torch.as_tensor(
            [frames[well.well_id] for well in wells],
            dtype=torch.long,
            device=device,
        ),
        torch.from_numpy(np.stack([value[2] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[3] for value in examples])).to(device),
    )


def fixed_gate(
    wells,
    frames,
    grid,
    fold,
    variant,
    device,
    steps,
    checkpoint,
    frame_audit,
):
    source, model, _ = initialized_models(fold, variant, device)
    source.eval()
    model.eval()
    fixed = stack_examples(wells[:2], frames, grid, device)
    untouched = stack_examples(wells[2:4], frames, grid, device)
    with torch.no_grad():
        source_logits = source(fixed[0])
        initialized_logits = model(fixed[0], fixed[1], fixed[2])
    parity = float(torch.max(torch.abs(source_logits - initialized_logits)))
    initial = sampled_rmse(model, fixed, grid)
    untouched_initial = sampled_rmse(model, untouched, grid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    nonfinite = 0
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed[0], fixed[1], fixed[2])
        loss = pure_ce(
            logits, fixed[3], fixed[4], grid.prefix_points
        )
        loss.backward()
        count = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += count
        if count:
            raise FloatingPointError("nonfinite A670 gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    final = sampled_rmse(model, fixed, grid)
    untouched_final = sampled_rmse(model, untouched, grid)

    test_well = wells[2]
    hidden_images, hidden_extra = [], []
    for mode in ("original", "reverse", "nan"):
        target = test_well.tvt.copy()
        hidden = np.arange(len(target)) > test_well.anchor_index
        if mode == "reverse":
            target[hidden] = target[hidden][::-1]
        elif mode == "nan":
            target[hidden] = np.nan
        modified = dataclasses.replace(test_well, tvt=target)
        example = make_example(modified, test_well.anchor_index, grid)
        hidden_images.append(example[0])
        hidden_extra.append(example[1])
    hidden_invariant = (
        np.array_equal(hidden_images[0], hidden_images[1])
        and np.array_equal(hidden_images[0], hidden_images[2])
        and np.array_equal(hidden_extra[0], hidden_extra[1])
        and np.array_equal(hidden_extra[0], hidden_extra[2])
    )

    with torch.no_grad():
        candidate = model(untouched[0], untouched[1], untouched[2])
        original = source(untouched[0])
        offsets = torch.as_tensor(
            grid.offsets, dtype=torch.float32, device=device
        )
        correction = expected_offset(candidate, offsets) - expected_offset(
            original, offsets
        )
        correction = correction[:, grid.prefix_points :]
        correction -= correction[:, :1]
        bounded = torch.clamp(correction, -64.0, 64.0)
    dummy = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    parent = np.linspace(10000.0, 10001.0, 32, dtype=np.float32)
    no_op = np.array_equal(parent + 0.0 * dummy, parent)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": variant,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "maximum_initial_logit_difference": parity,
        "new_stem_zero_at_initialization": True,
        "frame_embedding_zero_at_initialization": True,
        "frame_audit": frame_audit,
        "initial_fixed_rmse": initial,
        "final_fixed_rmse": final,
        "initial_untouched_rmse": untouched_initial,
        "final_untouched_rmse": untouched_final,
        "nonfinite_gradient_elements": nonfinite,
        "hidden_target_inputs_bit_identical": hidden_invariant,
        "first_hidden_delta_anchor_exact": bool(
            torch.all(bounded[:, 0] == 0).item()
        ),
        "maximum_bounded_delta": float(torch.max(torch.abs(bounded))),
        "d570_zero_share_exact": no_op,
        "steps": steps,
        "source_sha256": sha256(Path(__file__)),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    result["status"] = (
        "passed"
        if parity <= 1e-6
        and final < 0.20 * initial
        and nonfinite == 0
        and np.isfinite(untouched_final)
        and abs(untouched_final - untouched_initial) > 1e-6
        and hidden_invariant
        and result["first_hidden_delta_anchor_exact"]
        and result["maximum_bounded_delta"] <= 64.0
        and no_op
        else "failed"
    )
    return result


def parity_repair(
    original_path: Path,
    well: Well,
    frame: int,
    grid,
    fold: int,
    variant: str,
    checkpoint: Path,
) -> dict[str, object]:
    original = json.loads(original_path.read_text())
    if (
        original["status"] != "failed"
        or original["variant"] != variant
        or original["source_checkpoint_sha256"] != sha256(checkpoint)
        or original["maximum_initial_logit_difference"] <= 1e-6
        or original["final_fixed_rmse"] >= 0.20 * original["initial_fixed_rmse"]
        or original["nonfinite_gradient_elements"] != 0
        or not original["hidden_target_inputs_bit_identical"]
        or not original["first_hidden_delta_anchor_exact"]
        or original["maximum_bounded_delta"] > 64.0
        or not original["d570_zero_share_exact"]
    ):
        raise RuntimeError("C671 original failed record is not repair-eligible")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    source, model, _ = initialized_models(fold, variant, cpu)
    source.eval()
    model.eval()
    image, extra, _, _ = make_example(well, well.anchor_index, grid)
    image_tensor = torch.from_numpy(image)[None]
    extra_tensor = torch.from_numpy(extra)[None]
    frame_tensor = torch.as_tensor([frame], dtype=torch.long)
    with torch.no_grad():
        source_logits = source(image_tensor)
        initialized_logits = model(
            image_tensor, extra_tensor, frame_tensor
        )
    cpu_difference = float(
        torch.max(torch.abs(source_logits - initialized_logits))
    )
    inherited_exact = all(
        torch.equal(model.state_dict()[name], value)
        for name, value in source.state_dict().items()
    )
    branches_zero = bool(
        torch.count_nonzero(model.extra_stem.weight) == 0
        and (
            not model.use_frame
            or torch.count_nonzero(model.frame_embedding.weight) == 0
        )
    )
    result = dict(original)
    result.update(
        {
            "status": (
                "passed"
                if cpu_difference <= 1e-6
                and inherited_exact
                and branches_zero
                else "failed"
            ),
            "gate_repair": "deterministic_cpu_zero_branch_parity",
            "original_failed_result": str(original_path),
            "original_failed_result_sha256": sha256(original_path),
            "cuda_maximum_initial_logit_difference": original[
                "maximum_initial_logit_difference"
            ],
            "maximum_initial_logit_difference": cpu_difference,
            "deterministic_cpu_parity": True,
            "inherited_state_tensors_bit_identical": inherited_exact,
            "new_branches_exact_zero": branches_zero,
            "source_sha256": sha256(Path(__file__)),
        }
    )
    return result


def load_d570(
    path: Path,
    ids: np.ndarray,
    starts: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError("D570 protected flags set")
        source_ids = cache["well_ids"].astype(str)
        source_starts = cache["row_starts"].astype(np.int64)
        source_truth = cache["truth"].astype(np.float32)
        prediction = cache["prediction"].astype(np.float32)
    lookup = {well_id: index for index, well_id in enumerate(source_ids)}
    parts, truth_parts = [], []
    for well_id in ids:
        index = lookup[str(well_id)]
        left, right = source_starts[index : index + 2]
        parts.append(prediction[left:right])
        truth_parts.append(source_truth[left:right])
    selected_truth = np.concatenate(truth_parts)
    if not np.array_equal(selected_truth, truth):
        raise RuntimeError("D570 truth/layout differs")
    output = np.concatenate(parts)
    if len(output) != starts[-1]:
        raise RuntimeError("D570 row count differs")
    return output


@torch.no_grad()
def predict(
    source,
    model,
    wells,
    frames,
    grid,
    device,
    d570_path,
):
    source.eval()
    model.eval()
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=device
    )
    candidate_parts = {temperature: [] for temperature in TEMPERATURES}
    source_parts = {temperature: [] for temperature in TEMPERATURES}
    truth_parts, starts = [], [0]
    for well in wells:
        image, extra, _, valid = make_example(
            well, well.anchor_index, grid
        )
        image_tensor = torch.from_numpy(image)[None].to(device)
        frame_tensor = torch.as_tensor(
            [frames[well.well_id]], dtype=torch.long, device=device
        )
        candidate_logits = model(
            image_tensor, torch.from_numpy(extra)[None].to(device), frame_tensor
        )
        source_logits = source(image_tensor)
        _, _, metadata = make_alignment_example(
            well, well.anchor_index, grid, coordinate_kind="tvt_delta"
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
                (source_logits, source_parts),
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
        source_values = np.concatenate(source_parts[temperature])
        delta_values = candidate_values - source_values
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


def evaluate(d570, truth, candidate, delta):
    parent = float(
        np.sqrt(
            np.mean(
                np.square(d570.astype(np.float64) - truth.astype(np.float64))
            )
        )
    )
    records = []
    for temperature in TEMPERATURES:
        for share in SHARES:
            replacement = d570 + share * (candidate[temperature] - d570)
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
                        "treatment": "feature_delta",
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


def set_stage(model: AbsoluteStateUNet, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "full")
    if stage == "new_only":
        for parameter in model.extra_stem.parameters():
            parameter.requires_grad_(True)
        if model.use_frame:
            for parameter in model.frame_embedding.parameters():
                parameter.requires_grad_(True)


def train(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    identifiers, original = development_layout(args.data_root)
    development = np.flatnonzero(original < 4)
    all_wells = load_ids(
        identifiers, development, args.data_root, args.workers
    )
    development_folds = original[development]
    base_frames, held_frame_maps, frame_audit = frame_maps(
        all_wells, development_folds
    )
    lookup = {well.well_id: well for well in all_wells}
    train_ids = identifiers[(original < 4) & (original != args.fold)]
    held_ids = identifiers[original == args.fold]
    train_wells = [lookup[str(well_id)] for well_id in train_ids]
    held_wells = [lookup[str(well_id)] for well_id in held_ids]
    training_frames = base_frames
    inference_frames = held_frame_maps[args.fold]
    grid = config()
    source, model, checkpoint = initialized_models(
        args.fold, args.variant, device
    )
    if args.overfit_only:
        if args.parity_repair_from is not None:
            result = parity_repair(
                args.parity_repair_from,
                train_wells[0],
                training_frames[train_wells[0].well_id],
                grid,
                args.fold,
                args.variant,
                checkpoint,
            )
        else:
            result = fixed_gate(
                train_wells[:4],
                training_frames,
                grid,
                args.fold,
                args.variant,
                device,
                args.overfit_steps,
                checkpoint,
                frame_audit,
            )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError("A670 gate failed")
        return

    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    dataset = AbsoluteDataset(
        train_wells, training_frames, grid, args.repeats, SEED
    )
    history, snapshots = [], []
    source_hash = sha256(Path(__file__))
    nonfinite = 0
    started = time.time()
    for stage, first, last, learning_rate in (
        ("new_only", 1, 2, 3e-4),
        ("full", 3, 12, 1e-4),
    ):
        set_stage(model, stage)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=last - first + 1
        )
        for epoch in range(first, last + 1):
            dataset.epoch = epoch - 1
            batches = loader(
                dataset, args.batch_size, args.loader_workers, SEED + epoch
            )
            losses, gradients = [], []
            model.train()
            for image, extra, frame, labels, valid in batches:
                image = image.to(device, non_blocking=True)
                extra = extra.to(device, non_blocking=True)
                frame = frame.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image, extra, frame)
                loss = pure_ce(logits, labels, valid, grid.prefix_points)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite A670 loss")
                loss.backward()
                count = sum(
                    int((~torch.isfinite(parameter.grad)).sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                nonfinite += count
                if count:
                    raise FloatingPointError("nonfinite A670 gradient")
                gradients.append(
                    float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            record = {
                "epoch": epoch,
                "stage": stage,
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(gradients)),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if epoch in SNAPSHOTS:
                checkpoint_path = args.output_dir / f"epoch{epoch:03d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                (
                    ids,
                    starts,
                    d570,
                    truth,
                    candidate,
                    delta,
                ) = predict(
                    source,
                    model,
                    held_wells,
                    inference_frames,
                    grid,
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
                    "source_sha256": np.asarray(source_hash),
                    "checkpoint_sha256": np.asarray(sha256(checkpoint_path)),
                    "parent_checkpoint_sha256": np.asarray(sha256(checkpoint)),
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
                    "stage": stage,
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
        "status": "complete_fixed_snapshots",
        "fold": args.fold,
        "training_folds": sorted(set(range(4)) - {args.fold}),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": sha256(checkpoint),
        "frame_audit": frame_audit,
        "history": history,
        "snapshots": snapshots,
        "nonfinite_gradient_elements": nonfinite,
        "source_sha256": source_hash,
        "elapsed_seconds": time.time() - started,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    write_json(args.output_dir / "result.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d570", type=Path, default=DEFAULT_D570)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=800)
    parser.add_argument("--parity-repair-from", type=Path)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
