from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from limited_query_cv import (
    train_v5_run6_local_v20_residual as local,
)
from limited_query_cv import (
    train_v5_run6_protected_posterior_fusion as posterior,
)
from rogii.data import DEFAULT_DATA_ROOT, load_well


RUN_ID = "v5_batch2_run_025_goal_050"
ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_025_goal_050/promotion_audit"
)
ROLE_ROOT = RUN_ROOT / "roles"
RUN6_ROOT = (
    ROOT / "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
RUN3_ROOT = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_003_goal_050/v20_components"
)
CACHE_ROOT = RUN_ROOT / "caches"
LOCAL_SNAPSHOTS = (2, 4, 8, 12, 16)
POSTERIOR_SNAPSHOTS = (4, 8, 12, 16, 20, 24)
LOCAL_SEEDS = (20260831, 20261043)
POSTERIOR_VARIANTS = posterior.VARIANTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def component_path(fold: int) -> Path:
    if fold == 4:
        return CACHE_ROOT / "v20_components_fold4.npz"
    return RUN3_ROOT / f"fold{fold}.npz"


def posterior_path(fold: int) -> Path:
    if fold == 4:
        return CACHE_ROOT / "protected_posterior_volume_fold4.npz"
    return RUN6_ROOT / f"protected_posterior_volume_fold{fold}.npz"


def load_residual_fold(
    fold: int, data_root: Path
) -> list[local.ResidualRecord]:
    path = component_path(fold)
    with np.load(path, allow_pickle=False) as cache:
        if int(cache["fold"]) != fold:
            raise RuntimeError(f"{path}: fold mismatch")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        components = cache["components"].astype(np.float32)
        if fold == 4:
            if (
                "truth" in cache.files
                or not bool(cache["protected_f4_cache"])
                or bool(cache["hidden_truth_stored"])
            ):
                raise RuntimeError(f"{path}: F4 component firewall failed")
        elif bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise RuntimeError(f"{path}: source firewall failed")
    records = []
    for index, well_id in enumerate(ids):
        left, right = starts[index : index + 2]
        well = load_well(well_id, data_root)
        if len(well.prediction_indices) != right - left:
            raise RuntimeError(f"{well_id}: row layout changed")
        records.append(
            local.ResidualRecord(
                well=well,
                fold=fold,
                baseline=baseline[left:right].copy(),
                components=components[:, left:right].copy(),
            )
        )
    return records


def load_all_residual(
    data_root: Path,
) -> dict[int, list[local.ResidualRecord]]:
    return {fold: load_residual_fold(fold, data_root) for fold in range(5)}


def load_fusion_fold(
    fold: int,
    data_root: Path,
    residual_records: list[local.ResidualRecord],
) -> list[posterior.FusionRecord]:
    path = posterior_path(fold)
    with np.load(path, allow_pickle=False) as cache:
        if (
            int(cache["fold"]) != fold
            or not bool(cache["hidden_targets_masked_during_inference"])
            or bool(cache["stratified_checkpoint_loaded"])
            or any(
                "strat" in name
                for name in cache["checkpoint_names"].astype(str)
            )
            or "truth" in cache.files
        ):
            raise RuntimeError(f"{path}: posterior firewall failed")
        ids = cache["well_ids"].astype(str)
        probability = cache["posterior_probability"].astype(np.float32)
        source_hash = str(cache["source_sha256"])
        offsets = cache["residual_offsets"].astype(np.float32)
        parent_names = cache["parent_names"].astype(str)
    if (
        not np.array_equal(
            ids,
            np.asarray([record.well.well_id for record in residual_records]),
        )
        or probability.shape != (len(ids), 15, 65, 704)
        or not np.array_equal(offsets, local.RESIDUAL_OFFSETS)
        or len(np.unique(parent_names)) != 15
    ):
        raise RuntimeError(f"{path}: posterior layout changed")
    return [
        posterior.FusionRecord(
            residual=record,
            posterior_probability=probability[index],
            posterior_source_sha256=source_hash,
        )
        for index, record in enumerate(residual_records)
    ]


def role_name(kind: str, excluded: tuple[int, ...]) -> str:
    if kind == "pair":
        return f"pair_{excluded[0]}_{excluded[1]}"
    return f"outer_{excluded[0]}"


def role_sets(
    kind: str, first: int, second: int | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if kind == "pair":
        if second is None or first == second:
            raise ValueError("pair requires two distinct folds")
        excluded = tuple(sorted((first, second)))
    else:
        if second is not None:
            raise ValueError("outer role accepts one held fold")
        excluded = (first,)
    if any(fold not in range(5) for fold in excluded):
        raise ValueError("fold outside F0-F4")
    training = tuple(fold for fold in range(5) if fold not in excluded)
    return excluded, training


def combined_layout(records: list[local.ResidualRecord]) -> dict[str, np.ndarray]:
    starts = [0]
    for record in records:
        starts.append(starts[-1] + len(record.baseline))
    return {
        "well_ids": np.asarray([record.well.well_id for record in records]),
        "well_folds": np.asarray([record.fold for record in records], dtype=np.int8),
        "row_starts": np.asarray(starts, dtype=np.int64),
        "baseline": np.concatenate([record.baseline for record in records]).astype(
            np.float32
        ),
    }


@torch.no_grad()
def local_predictions(
    model: nn.Module,
    records: list[local.ResidualRecord],
    config,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for temperature in local.TEMPERATURES:
        values[f"prediction_t{int(temperature * 100):03d}"] = np.concatenate(
            [
                local.predict_record(
                    model, record, config, device, temperature
                )
                for record in records
            ]
        ).astype(np.float32)
    return values


def train_local(
    args: argparse.Namespace,
    excluded: tuple[int, ...],
    training_folds: tuple[int, ...],
) -> None:
    output = (
        ROLE_ROOT
        / role_name(args.role_kind, excluded)
        / f"local_seed{args.seed}"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    local.seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    config = local.grid()
    by_fold = load_all_residual(args.data_root)
    training = [
        record for fold in training_folds for record in by_fold[fold]
    ]
    predicted = [record for fold in excluded for record in by_fold[fold]]
    dataset = local.ResidualDataset(
        training,
        config,
        repeats=3,
        seed=args.seed,
        synthetic_probability=0.5,
    )
    model = local.AlignmentUNet(input_channels=23, base=12).to(device)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=16
    )
    layout = combined_layout(predicted)
    history = []
    snapshots = []
    started = time.time()
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
        for image, labels, valid in loader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, _ = local.residual_objective(
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
                raise FloatingPointError("nonfinite nested local loss")
            loss.backward()
            if any(
                not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError("nonfinite nested local gradient")
            gradients.append(
                float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(gradients)),
            }
        )
        if epoch in LOCAL_SNAPSHOTS:
            checkpoint = output / f"epoch{epoch:03d}.pt"
            prediction = output / f"epoch{epoch:03d}_prediction.npz"
            torch.save(model.state_dict(), checkpoint)
            model.eval()
            np.savez_compressed(
                prediction,
                **layout,
                **local_predictions(
                    model, predicted, config, device
                ),
                epoch=np.asarray(epoch, dtype=np.int16),
                seed=np.asarray(args.seed, dtype=np.int64),
                excluded_folds=np.asarray(excluded, dtype=np.int8),
                training_folds=np.asarray(training_folds, dtype=np.int8),
                checkpoint_sha256=np.asarray(sha256(checkpoint)),
                hidden_truth_stored=np.asarray(False),
                protected_metrics_computed=np.asarray(False),
                source_sha256=np.asarray(sha256(Path(__file__))),
            )
            snapshots.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "prediction": str(prediction.relative_to(ROOT)),
                    "prediction_sha256": sha256(prediction),
                }
            )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "sealed_nested_training",
        "family": "D570_local_residual",
        "role": role_name(args.role_kind, excluded),
        "seed": args.seed,
        "training_folds": list(training_folds),
        "predicted_folds": list(excluded),
        "training_wells": len(training),
        "predicted_wells": len(predicted),
        "snapshots": snapshots,
        "history": history,
        "hidden_truth_stored_in_predictions": False,
        "protected_metrics_computed": False,
        "source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        "status": "complete_frozen_candidates",
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def posterior_predictions(
    model: nn.Module,
    records: list[posterior.FusionRecord],
    config,
    variant: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    corrections, baseline, _, starts = posterior.predict_corrections(
        model, records, config, variant, device
    )
    layout = {
        "well_ids": np.asarray(
            [record.residual.well.well_id for record in records]
        ),
        "well_folds": np.asarray(
            [record.residual.fold for record in records], dtype=np.int8
        ),
        "row_starts": starts,
        "baseline": baseline.astype(np.float32),
    }
    values = {
        f"correction_t{int(round(temperature * 100)):03d}": correction.astype(
            np.float32
        )
        for temperature, correction in corrections.items()
    }
    return layout, values


def train_posterior(
    args: argparse.Namespace,
    excluded: tuple[int, ...],
    training_folds: tuple[int, ...],
) -> None:
    variant = str(args.variant)
    output = (
        ROLE_ROOT
        / role_name(args.role_kind, excluded)
        / f"posterior_{variant}"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    random.seed(posterior.SEED)
    np.random.seed(posterior.SEED)
    torch.manual_seed(posterior.SEED)
    torch.cuda.manual_seed_all(posterior.SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    config = local.grid()
    residual = load_all_residual(args.data_root)
    fusion = {
        fold: load_fusion_fold(
            fold, args.data_root, residual[fold]
        )
        for fold in range(5)
    }
    training = [
        record for fold in training_folds for record in fusion[fold]
    ]
    predicted = [record for fold in excluded for record in fusion[fold]]
    dataset = posterior.FusionDataset(
        training, config, variant, repeats=2
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=args.loader_workers,
        pin_memory=True,
        persistent_workers=args.loader_workers > 0,
        generator=torch.Generator().manual_seed(posterior.SEED),
    )
    model = posterior.make_model(variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=24
    )
    history = []
    snapshots = []
    started = time.time()
    for epoch in range(1, 25):
        model.train()
        losses = []
        gradients = []
        for image, labels, valid in loader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, _ = local.residual_objective(
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
                raise FloatingPointError("nonfinite nested posterior loss")
            loss.backward()
            if any(
                not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise FloatingPointError(
                    "nonfinite nested posterior gradient"
                )
            gradients.append(
                float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(np.mean(gradients)),
            }
        )
        if epoch in POSTERIOR_SNAPSHOTS:
            checkpoint = output / f"epoch{epoch:03d}.pt"
            prediction = output / f"epoch{epoch:03d}_prediction.npz"
            torch.save(model.state_dict(), checkpoint)
            model.eval()
            layout, values = posterior_predictions(
                model, predicted, config, variant, device
            )
            np.savez_compressed(
                prediction,
                **layout,
                **values,
                epoch=np.asarray(epoch, dtype=np.int16),
                variant=np.asarray(variant),
                excluded_folds=np.asarray(excluded, dtype=np.int8),
                training_folds=np.asarray(training_folds, dtype=np.int8),
                checkpoint_sha256=np.asarray(sha256(checkpoint)),
                hidden_truth_stored=np.asarray(False),
                protected_metrics_computed=np.asarray(False),
                source_sha256=np.asarray(sha256(Path(__file__))),
            )
            snapshots.append(
                {
                    "epoch": epoch,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "prediction": str(prediction.relative_to(ROOT)),
                    "prediction_sha256": sha256(prediction),
                }
            )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "sealed_nested_training",
        "family": "protected_posterior_fusion",
        "variant": variant,
        "role": role_name(args.role_kind, excluded),
        "training_folds": list(training_folds),
        "predicted_folds": list(excluded),
        "training_wells": len(training),
        "predicted_wells": len(predicted),
        "snapshots": snapshots,
        "history": history,
        "hidden_truth_stored_in_predictions": False,
        "protected_metrics_computed": False,
        "source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        "status": "complete_frozen_candidates",
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family", choices=("local", "posterior")
    )
    parser.add_argument(
        "--role-kind", choices=("pair", "outer"), required=True
    )
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--second", type=int)
    parser.add_argument("--seed", type=int, choices=LOCAL_SEEDS)
    parser.add_argument("--variant", choices=POSTERIOR_VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    excluded, training = role_sets(
        args.role_kind, args.first, args.second
    )
    if args.family == "local":
        if args.seed is None or args.variant is not None:
            parser.error("local requires --seed and forbids --variant")
        train_local(args, excluded, training)
    else:
        if args.variant is None or args.seed is not None:
            parser.error(
                "posterior requires --variant and forbids --seed"
            )
        train_posterior(args, excluded, training)


if __name__ == "__main__":
    main()
