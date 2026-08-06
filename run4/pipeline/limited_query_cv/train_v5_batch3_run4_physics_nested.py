from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from limited_query_cv import train_v5_run6_local_v20_residual as parent
from limited_query_cv import train_v5_run6_physics_prior_residual as physics
from rogii.alignment import AlignmentUNet
from rogii.data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    Well,
    load_well,
)


RUN_ID = "v5_batch3_run_004_goal_020"
FAMILY = "physics_prior_mixed_uniform_nested"
SNAPSHOTS = (2, 4, 8, 12, 16)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)
COMPONENT_PATHS = {
    0: Path(
        "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
        "v20_components/fold0.npz"
    ),
    1: Path(
        "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
        "v20_components/fold1.npz"
    ),
    2: Path(
        "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
        "v20_components/fold2.npz"
    ),
    3: Path(
        "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
        "v20_components/fold3.npz"
    ),
    4: Path(
        "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
        "promotion_audit/caches/v20_components_fold4.npz"
    ),
}


def blind_training_well(well_id: str, data_root: Path) -> Well:
    """Load train-side inference columns without reading hidden TVT."""
    train = data_root / "train"
    horizontal = pd.read_csv(
        train / f"{well_id}__horizontal_well.csv",
        usecols=list(HORIZONTAL_INFERENCE_COLUMNS),
    )
    typewell = pd.read_csv(
        train / f"{well_id}__typewell.csv",
        usecols=list(TYPEWELL_INFERENCE_COLUMNS),
    )
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise ValueError(f"{well_id}: no visible TVT_input prefix")
    placeholder = np.where(
        np.isfinite(tvt_input), tvt_input, tvt_input[known[-1]]
    )
    well = Well(
        well_id=well_id,
        md=horizontal["MD"].to_numpy(np.float64),
        x=horizontal["X"].to_numpy(np.float64),
        y=horizontal["Y"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=horizontal["GR"].to_numpy(np.float64),
        tvt=placeholder,
        tvt_input=tvt_input,
        typewell_tvt=typewell["TVT"].to_numpy(np.float64),
        typewell_gr=typewell["GR"].to_numpy(np.float64),
    )
    well.validate()
    return well


def load_role_records(
    data_root: Path,
    training_folds: tuple[int, ...],
    predicted_folds: tuple[int, ...],
) -> tuple[list[parent.ResidualRecord], list[parent.ResidualRecord]]:
    training: list[parent.ResidualRecord] = []
    predicted: list[parent.ResidualRecord] = []
    for fold in range(5):
        path = COMPONENT_PATHS[fold]
        with np.load(path, allow_pickle=False) as cache:
            if (
                int(cache["fold"]) != fold
                or bool(cache["audit_fold_loaded"])
                or bool(cache["layout_truth_array_accessed"])
            ):
                raise RuntimeError(f"{path}: component firewall failed")
            ids = cache["well_ids"].astype(str)
            starts = cache["row_starts"].astype(np.int64)
            baseline = cache["baseline"].astype(np.float32)
            components = cache["components"].astype(np.float32)
            cached_truth = (
                cache["truth"].astype(np.float32)
                if "truth" in cache.files and fold in training_folds
                else None
            )
        for index, well_id in enumerate(ids):
            left, right = starts[index : index + 2]
            if fold in training_folds:
                well = load_well(well_id, data_root)
                if cached_truth is not None:
                    expected = well.tvt[well.prediction_indices].astype(
                        np.float32
                    )
                    if not np.array_equal(cached_truth[left:right], expected):
                        raise RuntimeError(f"{well_id}: cached truth differs")
                destination = training
            elif fold in predicted_folds:
                well = blind_training_well(well_id, data_root)
                destination = predicted
            else:
                raise RuntimeError(f"fold {fold}: role partition incomplete")
            destination.append(
                parent.ResidualRecord(
                    well=well,
                    fold=fold,
                    baseline=baseline[left:right].copy(),
                    components=components[:, left:right].copy(),
                )
            )
    return training, predicted


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_temperatures(
    model: nn.Module,
    records: list[parent.ResidualRecord],
    config,
    device: torch.device,
) -> dict[float, np.ndarray]:
    model.eval()
    return {
        temperature: np.concatenate(
            [
                parent.predict_record(
                    model, record, config, device, temperature
                )
                for record in records
            ]
        ).astype(np.float32)
        for temperature in TEMPERATURES
    }


def train(args: argparse.Namespace) -> None:
    training_folds = tuple(sorted(set(args.training_folds)))
    predicted_folds = tuple(sorted(set(args.predicted_folds)))
    if (
        set(training_folds) & set(predicted_folds)
        or set(training_folds) | set(predicted_folds) != set(range(5))
        or len(training_folds) not in (3, 4)
        or len(predicted_folds) != 5 - len(training_folds)
    ):
        raise ValueError("role must partition five folds into 3/2 or 4/1")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    config = parent.grid()
    training, predicted = load_role_records(
        args.data_root, training_folds, predicted_folds
    )
    original_perturbation = parent.smooth_perturbation
    parent.smooth_perturbation = physics.profile_perturbation(
        "mixed", original_perturbation
    )
    dataset = parent.ResidualDataset(
        training,
        config,
        repeats=3,
        seed=args.seed,
        synthetic_probability=0.5,
    )
    model = AlignmentUNet(input_channels=23, base=12).to(device)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=16
    )
    history = []
    snapshots = []
    started = time.time()
    try:
        for epoch in range(1, 17):
            dataset.epoch = epoch - 1
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=args.loader_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=False,
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
            losses = []
            gradients = []
            model.train()
            epoch_started = time.time()
            for image, labels, valid in loader:
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss, _ = parent.residual_objective(
                    logits,
                    labels,
                    valid,
                    config.prefix_points,
                    0.0,
                    1.0,
                    "huber",
                    0.5,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite physics loss")
                loss.backward()
                if any(
                    not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ):
                    raise FloatingPointError("nonfinite physics gradient")
                gradients.append(
                    float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            record = {
                "epoch": epoch,
                "mean_training_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(gradients)),
                "elapsed_seconds": time.time() - epoch_started,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if epoch not in SNAPSHOTS:
                continue
            checkpoint = args.output_dir / f"epoch{epoch:03d}.pt"
            prediction_path = (
                args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            )
            torch.save(model.state_dict(), checkpoint)
            predictions = predict_temperatures(
                model, predicted, config, device
            )
            np.savez_compressed(
                prediction_path,
                schema_version=np.int64(1),
                protocol_revision=np.int64(11),
                run_id=np.asarray(RUN_ID),
                family=np.asarray(FAMILY),
                role=np.asarray(args.role),
                epoch=np.int16(epoch),
                training_folds=np.asarray(training_folds, dtype=np.int8),
                predicted_folds=np.asarray(predicted_folds, dtype=np.int8),
                well_ids=np.asarray(
                    [record.well.well_id for record in predicted]
                ),
                well_folds=np.asarray(
                    [record.fold for record in predicted], dtype=np.int8
                ),
                row_starts=np.concatenate(
                    (
                        [0],
                        np.cumsum(
                            [len(record.baseline) for record in predicted]
                        ),
                    )
                ).astype(np.int64),
                baseline=np.concatenate(
                    [record.baseline for record in predicted]
                ).astype(np.float32),
                **{
                    f"prediction_t{int(temperature * 100):03d}": prediction
                    for temperature, prediction in predictions.items()
                },
                hidden_truth_read_for_prediction=np.asarray(False),
                hidden_truth_stored=np.asarray(False),
                protected_metrics_computed=np.asarray(False),
            )
            snapshots.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": parent.sha256(checkpoint),
                    "prediction": str(prediction_path),
                    "prediction_sha256": parent.sha256(prediction_path),
                }
            )
    finally:
        parent.smooth_perturbation = original_perturbation

    result = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "role": args.role,
        "status": "complete_target_free_nested_role",
        "training_folds": list(training_folds),
        "predicted_folds": list(predicted_folds),
        "training_wells": len(training),
        "predicted_wells": len(predicted),
        "seed": args.seed,
        "snapshots": snapshots,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "source_sha256": parent.sha256(Path(__file__)),
        "parent_source_sha256": parent.sha256(Path(parent.__file__)),
        "physics_wrapper_source_sha256": parent.sha256(
            Path(physics.__file__)
        ),
        "hidden_truth_read_for_prediction": False,
        "hidden_truth_stored": False,
        "protected_metrics_computed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--training-folds", type=int, nargs="+", required=True)
    parser.add_argument("--predicted-folds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=parent.DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261183)
    parser.add_argument("--loader-workers", type=int, default=4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
