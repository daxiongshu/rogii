from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from limited_query_cv.train_v5_run6_categorical_expansion import (
    BLEND_WEIGHTS,
    TEMPERATURES,
    config,
    development_layout,
    evaluate_grid,
    load_baseline,
    load_ids,
    phase_loader,
    sha256,
    write_json,
)
from rogii.alignment import (
    AlignmentUNet,
    expected_offset,
    make_alignment_example,
)
from rogii.data import DEFAULT_DATA_ROOT, Well


RUN_ID = "v5_batch2_run_006_goal_050"
VARIANTS = ("base12", "grpair_base8")
SEED = 20260837
SYNTHETIC_EPOCHS = 4
SUPERVISED_EPOCHS = 8
SNAPSHOTS = (2, 4, 6, 8, 10, 12)


def pair_planes(
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
) -> np.ndarray:
    gr = np.asarray(metadata["gr"], dtype=np.float64)
    valid = np.asarray(metadata["valid_positions"]) > 0.5
    finite = valid & np.isfinite(gr)
    if not np.any(finite):
        center, scale = 0.0, 4.0
    else:
        center = float(np.median(gr[finite]))
        mad = float(np.median(np.abs(gr[finite] - center)))
        scale = max(1.4826 * mad, float(np.std(gr[finite])), 4.0)
    observed_row = np.clip(
        np.nan_to_num((gr - center) / scale), -6.0, 6.0
    ).astype(np.float32)
    observed = np.broadcast_to(
        observed_row[None, :], image.shape[1:]
    )
    # The frozen likelihood plane is observed-minus-expected in the same
    # normalized units, including its synthetic perturbations.
    expected = np.clip(observed - image[0], -6.0, 6.0)
    return np.concatenate(
        (image, observed[None].astype(np.float32), expected[None]),
        axis=0,
    ).astype(np.float32)


def make_example(
    well: Well,
    cut: int,
    grid,
    variant: str,
    rng: np.random.Generator | None,
    hardness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, labels, metadata = make_alignment_example(
        well,
        cut,
        grid,
        synthetic_rng=rng,
        synthetic_profile="forward_extreme",
        synthetic_hardness=hardness,
        coordinate_kind="tvt_delta",
    )
    if variant == "grpair_base8":
        image = pair_planes(image, metadata)
    valid = np.asarray(metadata["valid_positions"], dtype=np.float32)
    return image, labels, valid


class PureDataset(Dataset):
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
        self.synthetic = False
        self.hardness = 0.0

    def __len__(self) -> int:
        return len(self.wells) * self.repeats

    def __getitem__(self, index: int):
        rng = np.random.default_rng(
            self.seed + self.epoch * len(self) + index
        )
        repeat = index // len(self.wells)
        well = self.wells[index % len(self.wells)]
        if repeat == 0 or rng.random() < 0.35:
            cut = well.anchor_index
        else:
            low = max(100, int(0.14 * len(well.md)))
            high = min(len(well.md) - 100, int(0.42 * len(well.md)))
            cut = int(rng.integers(low, high + 1))
        image, labels, valid = make_example(
            well,
            cut,
            self.grid,
            self.variant,
            rng if self.synthetic else None,
            self.hardness,
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
        )


def pure_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    prefix_points: int,
) -> torch.Tensor:
    tail = slice(prefix_points, None)
    weights = valid[:, tail]
    point = F.cross_entropy(
        logits[:, :, tail], labels[:, tail], reduction="none"
    )
    return (point * weights).sum() / weights.sum().clamp_min(1.0)


def make_model(variant: str) -> AlignmentUNet:
    if variant == "base12":
        return AlignmentUNet(
            input_channels=19, base=12, dropout=0.10
        )
    if variant == "grpair_base8":
        return AlignmentUNet(
            input_channels=21, base=8, dropout=0.10
        )
    raise ValueError(f"unknown pure categorical variant: {variant}")


def stack_gate_examples(
    wells: list[Well],
    grid,
    variant: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [
        make_example(
            well,
            well.anchor_index,
            grid,
            variant,
            None,
            0.0,
        )
        for well in wells
    ]
    return tuple(
        torch.from_numpy(
            np.stack([example[index] for example in examples])
        ).to(device)
        for index in range(3)
    )


@torch.no_grad()
def sampled_rmse(
    model: AlignmentUNet,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grid,
) -> float:
    model.eval()
    logits = model(batch[0])
    offsets = torch.as_tensor(
        grid.offsets, device=logits.device, dtype=torch.float32
    )
    prediction = expected_offset(logits, offsets)
    truth = offsets[batch[1]]
    selected = batch[2] > 0.5
    selected[:, : grid.prefix_points] = False
    return float(
        torch.sqrt(torch.mean(torch.square(prediction[selected] - truth[selected])))
    )


def fixed_batch_gate(
    wells: list[Well],
    grid,
    variant: str,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    fixed = stack_gate_examples(wells[:2], grid, variant, device)
    untouched = stack_gate_examples(wells[2:4], grid, variant, device)
    model = make_model(variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )
    initial_fixed = sampled_rmse(model, fixed, grid)
    initial_untouched = sampled_rmse(model, untouched, grid)
    nonfinite = 0
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed[0])
        loss = pure_ce(logits, fixed[1], fixed[2], grid.prefix_points)
        loss.backward()
        step_nonfinite = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += step_nonfinite
        if step_nonfinite:
            raise FloatingPointError("nonfinite pure-CE gate gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "fixed_rmse": sampled_rmse(model, fixed, grid),
                        "untouched_rmse": sampled_rmse(
                            model, untouched, grid
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    final_fixed = sampled_rmse(model, fixed, grid)
    final_untouched = sampled_rmse(model, untouched, grid)
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": f"pure_categorical_{variant}_fixed_batch",
        "fixed_well_ids": [well.well_id for well in wells[:2]],
        "untouched_well_ids": [well.well_id for well in wells[2:4]],
        "initial_fixed_sampled_rmse": initial_fixed,
        "final_fixed_sampled_rmse": final_fixed,
        "initial_untouched_sampled_rmse": initial_untouched,
        "final_untouched_sampled_rmse": final_untouched,
        "steps": steps,
        "nonfinite_gradient_elements": nonfinite,
        "status": (
            "passed"
            if final_fixed < 0.20 * initial_fixed and nonfinite == 0
            else "failed"
        ),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "seed": SEED,
    }


@torch.no_grad()
def predict_temperatures(
    model: AlignmentUNet,
    wells: list[Well],
    grid,
    variant: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray]:
    model.eval()
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=device
    )
    starts = [0]
    predictions: dict[float, list[np.ndarray]] = {
        temperature: [] for temperature in TEMPERATURES
    }
    truths = []
    for well in wells:
        image, _, valid = make_example(
            well,
            well.anchor_index,
            grid,
            variant,
            None,
            0.0,
        )
        logits = model(torch.from_numpy(image)[None].to(device))
        _, _, metadata = make_alignment_example(
            well,
            well.anchor_index,
            grid,
            coordinate_kind="tvt_delta",
        )
        positions = np.asarray(metadata["positions"], dtype=np.float64)
        sampled_valid = valid > 0.5
        visible = sampled_valid & (
            positions <= well.anchor_index + 1e-6
        )
        visible_rows = np.flatnonzero(visible)[-32:]
        tail_rows = sampled_valid & (
            positions > well.anchor_index + 1e-6
        )
        if not len(visible_rows) or not np.any(tail_rows):
            raise RuntimeError(f"{well.well_id}: incomplete sampled path")
        visible_truth = np.interp(
            positions[visible_rows],
            np.arange(len(well.tvt_input), dtype=np.float64),
            np.nan_to_num(
                well.tvt_input,
                nan=well.tvt_input[well.anchor_index],
            ),
        )
        for temperature in TEMPERATURES:
            offset = (
                expected_offset(logits / temperature, offsets)[0]
                .float()
                .cpu()
                .numpy()
            )
            sampled_tvt = float(metadata["anchor_tvt"]) + offset
            calibration = float(
                np.median(sampled_tvt[visible_rows] - visible_truth)
            )
            sampled_tvt -= calibration
            prediction = np.interp(
                well.tail_indices,
                positions[tail_rows],
                sampled_tvt[tail_rows],
            )
            predictions[temperature].append(
                prediction.astype(np.float32)
            )
        truth = well.tvt[well.tail_indices].astype(np.float32)
        truths.append(truth)
        starts.append(starts[-1] + len(truth))
    return (
        np.asarray(starts, dtype=np.int64),
        {
            temperature: np.concatenate(parts)
            for temperature, parts in predictions.items()
        },
        np.concatenate(truths),
    )


def save_snapshot(
    output_dir: Path,
    epoch: int,
    phase: str,
    model: AlignmentUNet,
    held_wells: list[Well],
    grid,
    variant: str,
    baseline_cache: Path,
    device: torch.device,
    fold: int,
    source_hash: str,
    seed: int = SEED,
) -> dict[str, object]:
    checkpoint = output_dir / f"epoch{epoch:03d}.pt"
    prediction_path = output_dir / f"epoch{epoch:03d}_prediction.npz"
    torch.save(model.state_dict(), checkpoint)
    starts, predictions, truth = predict_temperatures(
        model, held_wells, grid, variant, device
    )
    ids = np.asarray([well.well_id for well in held_wells])
    baseline, baseline_truth = load_baseline(
        baseline_cache, ids, starts, truth
    )
    evaluation = evaluate_grid(
        predictions, baseline, baseline_truth
    )
    arrays: dict[str, np.ndarray] = {
        "fold": np.asarray(fold, dtype=np.int8),
        "variant": np.asarray(variant),
        "seed": np.asarray(seed, dtype=np.int64),
        "epoch": np.asarray(epoch, dtype=np.int16),
        "phase": np.asarray(phase),
        "well_ids": ids,
        "row_starts": starts,
        "baseline": baseline,
        "truth": truth,
        "source_sha256": np.asarray(source_hash),
        "checkpoint_sha256": np.asarray(sha256(checkpoint)),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
    }
    for temperature, prediction in predictions.items():
        key = f"prediction_t{int(round(temperature * 100)):03d}"
        arrays[key] = prediction.astype(np.float32)
    np.savez_compressed(prediction_path, **arrays)
    return {
        "epoch": epoch,
        "phase": phase,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "evaluation": evaluation,
    }


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    identifiers, original = development_layout(args.data_root)
    training = np.flatnonzero(
        (original < 4) & (original != args.fold)
    )
    held = np.flatnonzero(original == args.fold)
    if np.any(original[training] == 4):
        raise RuntimeError("F4 entered pure-categorical development")
    train_wells = load_ids(
        identifiers, training, args.data_root, args.workers
    )
    held_wells = load_ids(
        identifiers, held, args.data_root, args.workers
    )
    grid = config()
    if args.overfit_only:
        gate_wells = train_wells[:4]
        result = fixed_batch_gate(
            gate_wells,
            grid,
            args.variant,
            device,
            args.overfit_steps,
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError("pure-categorical fixed-batch gate failed")
        return

    dataset = PureDataset(
        train_wells,
        grid,
        args.variant,
        args.repeats,
        SEED,
    )
    model = make_model(args.variant).to(device)
    source_hash = sha256(Path(__file__))
    snapshots = []
    epochs = []
    global_epoch = 0
    nonfinite = 0
    started = time.time()
    for phase, phase_epochs in (
        ("synthetic", SYNTHETIC_EPOCHS),
        ("supervised", SUPERVISED_EPOCHS),
    ):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase_epochs
        )
        for phase_epoch in range(phase_epochs):
            global_epoch += 1
            dataset.epoch = global_epoch - 1
            dataset.synthetic = phase == "synthetic"
            dataset.hardness = (
                0.7 * (phase_epoch + 1) / phase_epochs
                if dataset.synthetic
                else 0.0
            )
            loader = phase_loader(
                dataset,
                args.batch_size,
                args.loader_workers,
                SEED + global_epoch,
            )
            losses = []
            gradient_norms = []
            epoch_started = time.time()
            model.train()
            for image, labels, valid in loader:
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss = pure_ce(
                    logits, labels, valid, grid.prefix_points
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite pure-CE loss")
                loss.backward()
                step_nonfinite = sum(
                    int((~torch.isfinite(parameter.grad)).sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                nonfinite += step_nonfinite
                if step_nonfinite:
                    raise FloatingPointError(
                        f"nonfinite pure-CE gradients={step_nonfinite}"
                    )
                gradient_norms.append(
                    float(
                        nn.utils.clip_grad_norm_(
                            model.parameters(), 5.0
                        )
                    )
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            record = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch + 1,
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(
                    np.mean(gradient_norms)
                ),
                "elapsed_seconds": time.time() - epoch_started,
            }
            epochs.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if global_epoch in SNAPSHOTS:
                snapshot = save_snapshot(
                    args.output_dir,
                    global_epoch,
                    phase,
                    model,
                    held_wells,
                    grid,
                    args.variant,
                    args.baseline_cache,
                    device,
                    args.fold,
                    source_hash,
                )
                snapshots.append(snapshot)
                print(
                    json.dumps(
                        {
                            "snapshot": global_epoch,
                            "same_held_diagnostic_only": snapshot[
                                "evaluation"
                            ]["best_same_held_diagnostic_only"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": f"pure_categorical_{args.variant}",
        "status": "complete_fixed_snapshots",
        "fold": args.fold,
        "seed": SEED,
        "training_folds": [
            fold for fold in range(4) if fold != args.fold
        ],
        "training_wells": len(train_wells),
        "validation_wells": len(held_wells),
        "configuration": {
            "base": 12 if args.variant == "base12" else 8,
            "input_channels": (
                19 if args.variant == "base12" else 21
            ),
            "dropout": 0.10,
            "objective": "categorical_cross_entropy_only",
            "synthetic_epochs": SYNTHETIC_EPOCHS,
            "supervised_epochs": SUPERVISED_EPOCHS,
            "synthetic_profile": "forward_extreme",
            "maximum_synthetic_hardness": 0.7,
            "snapshots": list(SNAPSHOTS),
            "temperature_grid": list(TEMPERATURES),
            "blend_weight_grid": list(BLEND_WEIGHTS),
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "visible_calibration_columns": 32,
            "training_precision": "float32",
        },
        "epochs": epochs,
        "snapshots": snapshots,
        "selection_boundary": (
            "same-held metrics are diagnostic only; an eligible F0-F3 "
            "expansion must select variant, snapshot, temperature, and "
            "weight from the other three development folds"
        ),
        "nonfinite_gradient_elements": nonfinite,
        "source_sha256": source_hash,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=600)
    args = parser.parse_args()
    if not args.overfit_only and args.baseline_cache is None:
        parser.error("--baseline-cache is required outside --overfit-only")
    train(args)


if __name__ == "__main__":
    main()
