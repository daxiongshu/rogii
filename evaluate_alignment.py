from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rogii.alignment import (
    AlignmentUNet,
    SYNTHETIC_PROFILES,
    alignment_objective,
    expected_offset,
    make_alignment_example,
)
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.folds import fold_indices
from rogii.sequence import SequenceConfig


class AlignmentDataset(Dataset):
    def __init__(
        self,
        wells: list[Well],
        config: SequenceConfig,
        repeats: int,
        seed: int,
        synthetic_profile: str,
        excursion_sampling_factor: float = 1.0,
        excursion_quantile: float = 0.8,
    ):
        self.wells = wells
        self.config = config
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0
        self.synthetic = False
        self.synthetic_mix_probability = 0.0
        self.synthetic_profile = synthetic_profile
        self.synthetic_hardness = 0.0
        self.excursion_sampling_factor = excursion_sampling_factor
        self.excursion_sampling_active = False
        excursions = np.asarray(
            [
                np.nanmax(
                    np.abs(well.tvt[well.tail_indices] - well.tvt[well.anchor_index])
                )
                for well in wells
            ],
            dtype=np.float64,
        )
        self.excursion_threshold = float(np.quantile(excursions, excursion_quantile))
        weights = np.where(
            excursions >= self.excursion_threshold,
            excursion_sampling_factor,
            1.0,
        )
        self.excursion_probabilities = weights / weights.sum()
        self.extreme_well_count = int(np.sum(excursions >= self.excursion_threshold))

    def __len__(self) -> int:
        return len(self.wells) * self.repeats

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        repeat = index // len(self.wells)
        well_index = index % len(self.wells)
        if self.excursion_sampling_active and repeat > 0:
            well_index = int(
                rng.choice(len(self.wells), p=self.excursion_probabilities)
            )
        well = self.wells[well_index]
        if repeat == 0 or rng.random() < 0.35:
            cut = well.anchor_index
        else:
            low = max(100, int(0.14 * len(well.md)))
            high = min(len(well.md) - 100, int(0.42 * len(well.md)))
            cut = int(rng.integers(low, high + 1))
        use_synthetic = self.synthetic or (
            self.synthetic_mix_probability > 0.0
            and rng.random() < self.synthetic_mix_probability
        )
        image, label, metadata = make_alignment_example(
            well,
            cut,
            self.config,
            synthetic_rng=rng if use_synthetic else None,
            synthetic_profile=self.synthetic_profile,
            synthetic_hardness=self.synthetic_hardness,
        )
        valid = np.asarray(metadata["valid_positions"], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(label), torch.from_numpy(valid)


def load_all(data_root: Path, workers: int) -> list[Well]:
    ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(well_id, data_root), ids))


@torch.no_grad()
def validate(
    model, wells, config, device, temperature: float = 1.0
) -> tuple[float, np.ndarray]:
    model.eval()
    offsets = torch.as_tensor(config.offsets, dtype=torch.float32, device=device)
    squared_error = 0.0
    count = 0
    scores = []
    for well in wells:
        image, _, metadata = make_alignment_example(well, well.anchor_index, config)
        logits = model(torch.from_numpy(image).unsqueeze(0).to(device))
        offset = expected_offset(logits / temperature, offsets)[0].float().cpu().numpy()
        sampled_tvt = float(metadata["anchor_tvt"]) + offset
        valid = np.asarray(metadata["valid_positions"])[config.prefix_points :] > 0.5
        tail_sampled = sampled_tvt[config.prefix_points :][valid]
        positions = np.asarray(metadata["positions"])[config.prefix_points :][valid]
        tail = well.tail_indices
        prediction = np.interp(tail, positions, tail_sampled)
        error = prediction - well.tvt[tail]
        squared_error += float(error @ error)
        count += len(error)
        scores.append(float(np.sqrt(np.mean(error * error))))
    return float(np.sqrt(squared_error / count)), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--last-checkpoint",
        type=Path,
        help="Optionally save the final epoch separately from validation-best weights.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prefix-points", type=int, default=192)
    parser.add_argument("--tail-points", type=int, default=768)
    parser.add_argument("--state-radius", type=float, default=120.0)
    parser.add_argument("--state-step", type=float, default=1.0)
    parser.add_argument("--synthetic-pretrain-epochs", type=int, default=0)
    parser.add_argument(
        "--synthetic-profile",
        choices=SYNTHETIC_PROFILES,
        default="legacy",
    )
    parser.add_argument("--synthetic-hardness", type=float, default=1.0)
    parser.add_argument("--synthetic-mix-probability", type=float, default=0.0)
    parser.add_argument("--row-stride", type=float)
    parser.add_argument("--gr-smooth-window", type=int, default=51)
    parser.add_argument(
        "--objective",
        choices=("categorical", "ordinal"),
        default="categorical",
    )
    parser.add_argument("--excursion-sampling-factor", type=float, default=1.0)
    parser.add_argument("--excursion-quantile", type=float, default=0.8)
    parser.add_argument(
        "--excursion-sampling-phase",
        choices=("synthetic", "supervised", "all"),
        default="synthetic",
    )
    args = parser.parse_args()
    if not 0.0 <= args.synthetic_hardness <= 1.0:
        raise ValueError("synthetic hardness must be between zero and one")
    if not 0.0 <= args.synthetic_mix_probability <= 1.0:
        raise ValueError("synthetic mix probability must be between zero and one")
    if args.excursion_sampling_factor < 1.0:
        raise ValueError("excursion sampling factor must be at least one")
    if not 0.0 < args.excursion_quantile < 1.0:
        raise ValueError("excursion quantile must be in (0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    wells = load_all(args.data_root, args.workers)
    identifiers = np.asarray([well.well_id for well in wells])
    train_idx, valid_idx = fold_indices(
        identifiers, args.fold, args.splits, args.fold_file
    )
    train_wells = [wells[i] for i in train_idx]
    valid_wells = [wells[i] for i in valid_idx]
    config = SequenceConfig(
        prefix_points=args.prefix_points,
        tail_points=args.tail_points,
        state_radius=args.state_radius,
        state_step=args.state_step,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=args.row_stride,
        gr_smooth_window=args.gr_smooth_window,
    )
    dataset = AlignmentDataset(
        train_wells,
        config,
        args.repeats,
        args.seed,
        args.synthetic_profile,
        args.excursion_sampling_factor,
        args.excursion_quantile,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=(
            min(args.workers, 12)
            if args.loader_workers is None
            else args.loader_workers
        ),
        pin_memory=True,
        drop_last=True,
    )
    model = AlignmentUNet(base=args.base).to(device)
    if args.resume:
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True)
        )
        print(f"resumed={args.resume}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    offsets = torch.as_tensor(config.offsets, dtype=torch.float32, device=device)
    print(
        f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
        f"train_wells={len(train_wells)} valid_wells={len(valid_wells)} "
        f"image=19x{len(config.offsets)}x{config.length} "
        f"excursion_threshold={dataset.excursion_threshold:.2f} "
        f"extreme_wells={dataset.extreme_well_count} "
        f"excursion_factor={dataset.excursion_sampling_factor:.2f} "
        f"excursion_phase={args.excursion_sampling_phase}",
        flush=True,
    )
    best = float("inf")
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        dataset.synthetic = epoch < args.synthetic_pretrain_epochs
        dataset.excursion_sampling_active = args.excursion_sampling_factor > 1.0 and (
            args.excursion_sampling_phase == "all"
            or (args.excursion_sampling_phase == "synthetic" and dataset.synthetic)
            or (args.excursion_sampling_phase == "supervised" and not dataset.synthetic)
        )
        dataset.synthetic_mix_probability = (
            0.0 if dataset.synthetic else args.synthetic_mix_probability
        )
        dataset.synthetic_hardness = (
            args.synthetic_hardness
            * (epoch + 1)
            / max(args.synthetic_pretrain_epochs, 1)
            if dataset.synthetic
            else (
                args.synthetic_hardness
                if dataset.synthetic_mix_probability > 0.0
                else 0.0
            )
        )
        model.train()
        losses = []
        for image, label, valid in loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(image)
                loss, _ = alignment_objective(
                    logits,
                    image,
                    label,
                    valid,
                    offsets,
                    config.prefix_points,
                    args.objective,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch == 0 or (epoch + 1) % args.validate_every == 0:
            score, per_well = validate(model, valid_wells, config, device)
            print(
                f"epoch={epoch + 1:03d} synthetic={dataset.synthetic} "
                f"profile={dataset.synthetic_profile} "
                f"hardness={dataset.synthetic_hardness:.2f} "
                f"mix={dataset.synthetic_mix_probability:.2f} "
                f"excursion_sampling={dataset.excursion_sampling_active} "
                f"loss={np.mean(losses):.5f} "
                f"valid_rmse={score:.6f} median={np.median(per_well):.4f} "
                f"p90={np.quantile(per_well, 0.9):.4f} elapsed={time.time() - started:.1f}",
                flush=True,
            )
            if score < best:
                best = score
                if args.checkpoint:
                    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), args.checkpoint)
        if args.last_checkpoint:
            args.last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.last_checkpoint)
    print(f"best_valid_rmse={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
