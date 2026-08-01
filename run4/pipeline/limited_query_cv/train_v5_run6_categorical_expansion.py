from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate_alignment import AlignmentDataset
from rogii.alignment import (
    AlignmentUNet,
    alignment_objective,
    expected_offset,
    make_alignment_example,
)
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.folds import fold_indices
from rogii.sequence import SequenceConfig


RUN_ID = "v5_batch2_run_006_goal_050"
REGISTERED_SEEDS = (20260803, 20260819)
SNAPSHOTS = (4, 8, 12, 16, 24)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
BLEND_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
SYNTHETIC_EPOCHS = 16
SUPERVISED_EPOCHS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def config() -> SequenceConfig:
    return SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=120.0,
        state_step=1.0,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=51,
    )


def development_layout(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.asarray(training_well_ids(data_root))
    original = np.empty(len(identifiers), dtype=np.int8)
    for fold in range(5):
        _, held = fold_indices(identifiers, fold, 5)
        original[held] = fold
    return identifiers, original


def load_ids(
    identifiers: np.ndarray,
    selected: np.ndarray,
    data_root: Path,
    workers: int,
) -> list[Well]:
    ids = identifiers[selected].tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda well_id: load_well(well_id, data_root), ids)
        )


def phase_loader(
    dataset: AlignmentDataset,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    # Recreate workers after setting epoch and phase. This is intentionally
    # non-persistent: earlier V5 pilots demonstrated that stale worker copies
    # can silently retain epoch-zero augmentation state.
    generator = torch.Generator()
    generator.manual_seed(seed)
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
def predict_temperatures(
    model: AlignmentUNet,
    wells: list[Well],
    grid: SequenceConfig,
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
        image, _, metadata = make_alignment_example(
            well,
            well.anchor_index,
            grid,
            coordinate_kind="tvt_delta",
        )
        logits = model(torch.from_numpy(image)[None].to(device))
        sampled_positions = np.asarray(
            metadata["positions"], dtype=np.float64
        )
        sampled_valid = np.asarray(
            metadata["valid_positions"], dtype=np.float64
        ) > 0.5
        visible = (
            sampled_valid
            & (sampled_positions <= well.anchor_index + 1e-6)
        )
        visible_rows = np.flatnonzero(visible)[-32:]
        if not len(visible_rows):
            raise RuntimeError(f"{well.well_id}: no visible calibration rows")
        visible_truth = np.interp(
            sampled_positions[visible_rows],
            np.arange(len(well.tvt_input), dtype=np.float64),
            np.nan_to_num(
                well.tvt_input,
                nan=well.tvt_input[well.anchor_index],
            ),
        )
        tail_rows = (
            sampled_valid
            & (sampled_positions > well.anchor_index + 1e-6)
        )
        positions = sampled_positions[tail_rows]
        if not len(positions):
            raise RuntimeError(f"{well.well_id}: no sampled hidden rows")
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
            sampled_tvt = sampled_tvt - calibration
            prediction = np.interp(
                well.tail_indices,
                positions,
                sampled_tvt[tail_rows],
            )
            predictions[temperature].append(prediction.astype(np.float32))
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


def load_baseline(
    path: Path,
    expected_ids: np.ndarray,
    expected_starts: np.ndarray,
    expected_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        truth = cache["truth"].astype(np.float32)
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise RuntimeError("baseline cache firewall failed")
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError("baseline and prediction well order differs")
    if not np.array_equal(starts, expected_starts):
        raise RuntimeError("baseline and prediction row layout differs")
    if not np.array_equal(truth, expected_truth):
        raise RuntimeError("baseline and prediction truth differs")
    return baseline, truth


def evaluate_grid(
    predictions: dict[float, np.ndarray],
    baseline: np.ndarray,
    truth: np.ndarray,
) -> dict[str, object]:
    baseline64 = baseline.astype(np.float64)
    truth64 = truth.astype(np.float64)
    baseline_rmse = float(
        np.sqrt(np.mean(np.square(baseline64 - truth64)))
    )
    records = []
    for temperature, candidate in predictions.items():
        candidate64 = candidate.astype(np.float64)
        standalone = float(
            np.sqrt(np.mean(np.square(candidate64 - truth64)))
        )
        for weight in BLEND_WEIGHTS:
            blended = baseline64 + weight * (candidate64 - baseline64)
            score = float(
                np.sqrt(np.mean(np.square(blended - truth64)))
            )
            records.append(
                {
                    "temperature": temperature,
                    "weight": weight,
                    "standalone_rmse": standalone,
                    "rmse": score,
                    "gain": baseline_rmse - score,
                }
            )
    best = min(
        records,
        key=lambda item: (
            item["rmse"],
            item["weight"],
            item["temperature"],
        ),
    )
    return {
        "baseline_rmse": baseline_rmse,
        "best_same_held_diagnostic_only": best,
        "grid": records,
    }


def save_snapshot(
    output_dir: Path,
    epoch: int,
    phase: str,
    model: AlignmentUNet,
    wells: list[Well],
    grid: SequenceConfig,
    baseline_cache: Path,
    device: torch.device,
    fold: int,
    seed: int,
    source_hash: str,
) -> dict[str, object]:
    checkpoint = output_dir / f"epoch{epoch:03d}.pt"
    prediction_path = output_dir / f"epoch{epoch:03d}_prediction.npz"
    if checkpoint.exists() or prediction_path.exists():
        raise FileExistsError(f"refusing to overwrite snapshot {epoch}")
    torch.save(model.state_dict(), checkpoint)
    starts, predictions, truth = predict_temperatures(
        model, wells, grid, device
    )
    ids = np.asarray([well.well_id for well in wells])
    baseline, baseline_truth = load_baseline(
        baseline_cache, ids, starts, truth
    )
    evaluation = evaluate_grid(
        predictions, baseline, baseline_truth
    )
    arrays: dict[str, np.ndarray] = {
        "fold": np.asarray(fold, dtype=np.int8),
        "seed": np.asarray(seed, dtype=np.int64),
        "epoch": np.asarray(epoch, dtype=np.int16),
        "phase": np.asarray(phase),
        "well_ids": ids,
        "row_starts": starts,
        "baseline": baseline,
        "truth": truth,
        "temperatures": np.asarray(TEMPERATURES, dtype=np.float32),
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
    if args.seed not in REGISTERED_SEEDS:
        raise ValueError(f"seed {args.seed} is not registered")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    identifiers, original = development_layout(args.data_root)
    training = np.flatnonzero(
        (original < 4) & (original != args.fold)
    )
    held = np.flatnonzero(original == args.fold)
    if np.any(original[training] == 4) or args.fold not in range(4):
        raise RuntimeError("F4 entered categorical development")
    train_wells = load_ids(
        identifiers, training, args.data_root, args.workers
    )
    held_wells = load_ids(
        identifiers, held, args.data_root, args.workers
    )
    grid = config()
    dataset = AlignmentDataset(
        train_wells,
        grid,
        repeats=2,
        seed=args.seed,
        synthetic_profile="forward_extreme",
        coordinate_kind="tvt_delta",
        excursion_sampling_factor=1.0,
    )
    model = AlignmentUNet(base=16).to(device)
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=device
    )
    source_hash = sha256(Path(__file__))
    snapshots = []
    epochs = []
    global_epoch = 0
    nonfinite_gradient_elements = 0
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
            dataset.synthetic_mix_probability = 0.0
            dataset.synthetic_hardness = (
                0.7 * (phase_epoch + 1) / phase_epochs
                if phase == "synthetic"
                else 0.0
            )
            dataset.excursion_sampling_active = False
            loader = phase_loader(
                dataset,
                args.batch_size,
                args.loader_workers,
                args.seed + global_epoch,
            )
            model.train()
            losses = []
            gradient_norms = []
            epoch_nonfinite = 0
            epoch_started = time.time()
            for image, label, valid in loader:
                image = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss, _ = alignment_objective(
                    logits,
                    image,
                    label,
                    valid,
                    offsets,
                    grid.prefix_points,
                    "categorical",
                    "tvt_delta",
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite categorical loss")
                loss.backward()
                step_nonfinite = sum(
                    int((~torch.isfinite(parameter.grad)).sum().item())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
                epoch_nonfinite += step_nonfinite
                if step_nonfinite:
                    raise FloatingPointError(
                        f"nonfinite categorical gradients={step_nonfinite}"
                    )
                gradient_norm = float(
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                )
                optimizer.step()
                losses.append(float(loss.detach()))
                gradient_norms.append(gradient_norm)
            scheduler.step()
            nonfinite_gradient_elements += epoch_nonfinite
            record = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch + 1,
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "nonfinite_gradient_elements": epoch_nonfinite,
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
                    args.baseline_cache,
                    device,
                    args.fold,
                    args.seed,
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

    if [record["epoch"] for record in snapshots] != list(SNAPSHOTS):
        raise RuntimeError("registered snapshot grid incomplete")
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": "base16_forward_extreme_fixed_phase_expansion",
        "status": "complete_fixed_snapshots",
        "fold": args.fold,
        "seed": args.seed,
        "training_folds": [
            fold for fold in range(4) if fold != args.fold
        ],
        "training_wells": len(train_wells),
        "validation_wells": len(held_wells),
        "configuration": {
            "base": 16,
            "prefix_points": grid.prefix_points,
            "tail_points": grid.tail_points,
            "row_stride": grid.row_stride,
            "state_radius": grid.state_radius,
            "state_step": grid.state_step,
            "synthetic_epochs": SYNTHETIC_EPOCHS,
            "supervised_epochs": SUPERVISED_EPOCHS,
            "synthetic_profile": "forward_extreme",
            "maximum_synthetic_hardness": 0.7,
            "repeats": 2,
            "batch_size": args.batch_size,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "training_precision": "float32",
            "temperature_grid": list(TEMPERATURES),
            "blend_weight_grid": list(BLEND_WEIGHTS),
            "visible_calibration_columns": 32,
        },
        "epochs": epochs,
        "snapshots": snapshots,
        "selection_boundary": (
            "same-held metrics are diagnostic only; the primary evaluator "
            "must select snapshot, temperature, seed bag, and weight using "
            "the other three development folds"
        ),
        "nonfinite_gradient_elements": nonfinite_gradient_elements,
        "source_sha256": source_hash,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, choices=REGISTERED_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
