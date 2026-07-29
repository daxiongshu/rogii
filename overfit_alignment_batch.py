from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rogii.alignment import (
    AlignmentUNet,
    SYNTHETIC_PROFILES,
    alignment_objective,
    expected_offset,
    make_alignment_example,
)
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


def make_fixed_batch(
    wells: list[Well],
    config: SequenceConfig,
    device: torch.device,
    synthetic: bool,
    seed: int,
    synthetic_profile: str,
    synthetic_hardness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = []
    for index, well in enumerate(wells):
        rng = np.random.default_rng(seed + index) if synthetic else None
        examples.append(
            make_alignment_example(
                well,
                well.anchor_index,
                config,
                synthetic_rng=rng,
                synthetic_profile=synthetic_profile,
                synthetic_hardness=synthetic_hardness,
            )
        )
    images = torch.from_numpy(np.stack([example[0] for example in examples])).to(device)
    labels = torch.from_numpy(np.stack([example[1] for example in examples])).to(device)
    valid = torch.from_numpy(
        np.stack([np.asarray(example[2]["valid_positions"]) for example in examples])
    ).to(device)
    return images, labels, valid


@torch.no_grad()
def measure(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    offsets: torch.Tensor,
    prefix_points: int,
    objective_kind: str,
) -> dict[str, float]:
    model.eval()
    image, label, valid = batch
    logits = model(image)
    loss, parts = alignment_objective(
        logits, image, label, valid, offsets, prefix_points, objective_kind
    )
    logits = logits[:, :, prefix_points:].float()
    label = label[:, prefix_points:]
    mask = valid[:, prefix_points:] > 0.5
    truth = offsets[label]
    mean_prediction = expected_offset(logits, offsets)
    mode_prediction = offsets[torch.argmax(logits, dim=1)]
    mean_error = (mean_prediction - truth)[mask]
    mode_error = (mode_prediction - truth)[mask]
    accuracy = (torch.argmax(logits, dim=1) == label)[mask].float().mean()
    return {
        "loss": float(loss),
        "ce": float(parts["classification"]),
        "reg": float(parts["regression"]),
        "smooth": float(parts["surface_smoothness"]),
        "mean_rmse": float(torch.sqrt(torch.mean(torch.square(mean_error)))),
        "mode_rmse": float(torch.sqrt(torch.mean(torch.square(mode_error)))),
        "class_accuracy": float(accuracy),
    }


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"{name}_loss={metrics['loss']:.5f} {name}_ce={metrics['ce']:.5f} "
        f"{name}_mean_rmse={metrics['mean_rmse']:.4f}ft "
        f"{name}_mode_rmse={metrics['mode_rmse']:.4f}ft "
        f"{name}_accuracy={metrics['class_accuracy']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memorize one frozen alignment batch and measure generalization."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--synthetic-profile",
        choices=SYNTHETIC_PROFILES,
        default="legacy",
    )
    parser.add_argument("--synthetic-hardness", type=float, default=1.0)
    parser.add_argument("--prefix-points", type=int, default=64)
    parser.add_argument("--tail-points", type=int, default=640)
    parser.add_argument("--state-radius", type=float, default=64.0)
    parser.add_argument("--state-step", type=float, default=0.5)
    parser.add_argument("--row-stride", type=float, default=16.0)
    parser.add_argument("--gr-smooth-window", type=int, default=51)
    parser.add_argument(
        "--objective",
        choices=("categorical", "ordinal"),
        default="categorical",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.report_every < 1 or args.batch_size < 1:
        raise ValueError("steps, report-every, and batch-size must be positive")
    if not 0.0 <= args.synthetic_hardness <= 1.0:
        raise ValueError("synthetic hardness must be between zero and one")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ids = training_well_ids(args.data_root)
    needed = 2 * args.batch_size
    if len(ids) < needed:
        raise ValueError(f"need at least {needed} training wells")
    train_ids = ids[: args.batch_size]
    holdout_ids = ids[args.batch_size : needed]
    train_wells = [load_well(well_id, args.data_root) for well_id in train_ids]
    holdout_wells = [load_well(well_id, args.data_root) for well_id in holdout_ids]
    train_batch = make_fixed_batch(
        train_wells,
        config,
        device,
        args.synthetic,
        args.seed,
        args.synthetic_profile,
        args.synthetic_hardness,
    )
    holdout_batch = make_fixed_batch(
        holdout_wells,
        config,
        device,
        args.synthetic,
        args.seed + 10_000,
        args.synthetic_profile,
        args.synthetic_hardness,
    )
    offsets = torch.as_tensor(config.offsets, dtype=torch.float32, device=device)
    model = AlignmentUNet(base=args.base).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.time()
    mode = "synthetic" if args.synthetic else "real"
    if args.synthetic:
        mode += f"/{args.synthetic_profile}@{args.synthetic_hardness:.2f}"
    print(
        f"device={device} mode={mode} parameters={sum(p.numel() for p in model.parameters()):,} "
        f"train_ids={train_ids} holdout_ids={holdout_ids}",
        flush=True,
    )
    print(
        f"step=0000 {format_metrics('train', measure(model, train_batch, offsets, config.prefix_points, args.objective))} "
        f"{format_metrics('holdout', measure(model, holdout_batch, offsets, config.prefix_points, args.objective))}",
        flush=True,
    )

    image, label, valid = train_batch
    for step in range(1, args.steps + 1):
        model.train()
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
        if step == 1 or step % args.report_every == 0 or step == args.steps:
            train_metrics = measure(
                model,
                train_batch,
                offsets,
                config.prefix_points,
                args.objective,
            )
            holdout_metrics = measure(
                model,
                holdout_batch,
                offsets,
                config.prefix_points,
                args.objective,
            )
            print(
                f"step={step:04d} {format_metrics('train', train_metrics)} "
                f"{format_metrics('holdout', holdout_metrics)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
