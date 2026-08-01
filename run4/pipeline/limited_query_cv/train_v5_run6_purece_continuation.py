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

from limited_query_cv.evaluate_v5_run6_protected_mode_bank import (
    CHECKPOINT_ROOT,
    load_inference_well,
)
from limited_query_cv.train_v5_run6_categorical_expansion import (
    BLEND_WEIGHTS,
    TEMPERATURES,
    config,
    development_layout,
    load_ids,
    phase_loader,
    sha256,
    write_json,
)
from limited_query_cv.train_v5_run6_pure_categorical import (
    PureDataset,
    SEED,
    VARIANTS,
    make_example,
    sampled_rmse,
    save_snapshot,
    stack_gate_examples,
)
from rogii.alignment import AlignmentUNet
from rogii.data import DEFAULT_DATA_ROOT


RUN_ID = "v5_batch2_run_006_goal_050"
SNAPSHOTS = (1, 2, 4, 6)
EPOCHS = 6
PARENTS = {
    "base12": "alignment_wide_base12_exc2sup_fold{fold}.pt",
    "grpair_base8": "alignment_longsynthetic_exc2sup_fold{fold}.pt",
}


def hard_tail_objective(
    logits: torch.Tensor,
    source_logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    prefix_points: int,
    hard_alpha: float,
    direct_mse_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tail = slice(prefix_points, None)
    with torch.no_grad():
        source_probability = torch.softmax(
            source_logits[:, :, tail].float(), dim=1
        )
        source_prediction = torch.sum(
            source_probability * offsets[None, :, None], dim=1
        )
        truth = offsets[labels[:, tail]]
        multiplier = 1.0 + hard_alpha * torch.clamp(
            torch.abs(source_prediction - truth) / 8.0,
            min=0.0,
            max=4.0,
        )
    weights = valid[:, tail] * multiplier
    denominator = weights.sum().clamp_min(1.0)
    point_ce = F.cross_entropy(
        logits[:, :, tail], labels[:, tail], reduction="none"
    )
    classification = (point_ce * weights).sum() / denominator
    probability = torch.softmax(logits[:, :, tail].float(), dim=1)
    prediction = torch.sum(
        probability * offsets[None, :, None], dim=1
    )
    direct_mse = (
        torch.square((prediction - truth) / 8.0) * weights
    ).sum() / denominator
    total = classification + direct_mse_weight * direct_mse
    return total, {
        "classification": classification,
        "direct_mse": direct_mse,
    }


class ExpandedPairUNet(AlignmentUNet):
    """Preserve the 19-plane stem operation and add a zero pair branch."""

    def __init__(self, base: int = 8, dropout: float = 0.10) -> None:
        super().__init__(
            input_channels=19, base=base, dropout=dropout
        )
        self.pair_stem = nn.Conv2d(
            2,
            base,
            kernel_size=(5, 9),
            padding=(2, 4),
            bias=False,
        )
        nn.init.zeros_(self.pair_stem.weight)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image[:, :19]) + self.pair_stem(image[:, 19:])
        skips = []
        for block, down in zip(self.encoder, self.down):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, block, skip in zip(
            self.up, self.decoder, reversed(skips)
        ):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x).squeeze(1)


def initialized_models(
    variant: str,
    fold: int,
    device: torch.device,
) -> tuple[AlignmentUNet, AlignmentUNet, Path]:
    checkpoint = CHECKPOINT_ROOT / PARENTS[variant].format(fold=fold)
    if variant == "base12":
        source = AlignmentUNet(
            input_channels=19, base=12, dropout=0.10
        ).to(device)
        source.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True),
            strict=True,
        )
        target = AlignmentUNet(
            input_channels=19, base=12, dropout=0.10
        ).to(device)
        target.load_state_dict(source.state_dict(), strict=True)
        return source, target, checkpoint

    source = AlignmentUNet(
        input_channels=19, base=8, dropout=0.10
    ).to(device)
    source.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    target = ExpandedPairUNet(base=8, dropout=0.10).to(device)
    source_state = source.state_dict()
    target_state = target.state_dict()
    if source_state["stem.weight"].shape[1] != 19:
        raise RuntimeError("protected pair source is not 19-plane")
    for name, value in source_state.items():
        target_state[name].copy_(value)
    target_state["pair_stem.weight"].zero_()
    target.load_state_dict(target_state, strict=True)
    return source, target, checkpoint


@torch.no_grad()
def parity_gate(
    fold: int,
    data_root: Path,
    device: torch.device,
) -> dict[str, object]:
    identifiers, original = development_layout(data_root)
    well_id = str(identifiers[np.flatnonzero(original == fold)[0]])
    well = load_inference_well(well_id, data_root)
    grid = config()
    records = []
    for variant in VARIANTS:
        source, target, checkpoint = initialized_models(
            variant, fold, device
        )
        source.eval()
        target.eval()
        image, _, _ = make_example(
            well,
            well.anchor_index,
            grid,
            variant,
            None,
            0.0,
        )
        tensor = torch.from_numpy(image)[None].to(device)
        source_logits = source(tensor[:, :19])
        target_logits = target(tensor)
        maximum = float(
            torch.max(torch.abs(source_logits - target_logits))
        )
        records.append(
            {
                "variant": variant,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "maximum_logit_difference": maximum,
                "target_logits_finite": bool(
                    torch.all(torch.isfinite(target_logits))
                ),
                "source_input_channels": 19,
                "target_input_channels": (
                    19 if variant == "base12" else 21
                ),
                "new_pair_stem_maximum_absolute_weight": (
                    0.0
                    if variant == "base12"
                    else float(
                        torch.max(
                            torch.abs(
                                target.state_dict()["pair_stem.weight"]
                            )
                        )
                    )
                ),
            }
        )
    passed = all(
        record["maximum_logit_difference"] <= 1e-6
        and record["target_logits_finite"]
        and record["new_pair_stem_maximum_absolute_weight"] == 0.0
        for record in records
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": "purece_protected_parent_parity_gate",
        "well_id": well_id,
        "fold": fold,
        "records": records,
        "status": "passed" if passed else "failed",
        "held_horizontal_target_loaded": False,
        "held_formation_columns_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
    }


def hard_tail_fixed_gate(
    wells,
    grid,
    fold: int,
    device: torch.device,
    steps: int,
    seed: int,
    hard_alpha: float,
    direct_mse_weight: float,
) -> dict[str, object]:
    fixed = stack_gate_examples(
        wells[:2], grid, "base12", device
    )
    untouched = stack_gate_examples(
        wells[2:4], grid, "base12", device
    )
    source, model, checkpoint = initialized_models(
        "base12", fold, device
    )
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )
    offsets = torch.as_tensor(
        grid.offsets, device=device, dtype=torch.float32
    )
    initial_fixed = sampled_rmse(model, fixed, grid)
    initial_untouched = sampled_rmse(model, untouched, grid)
    nonfinite = 0
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            source_logits = source(fixed[0][:, :19])
        logits = model(fixed[0])
        loss, parts = hard_tail_objective(
            logits,
            source_logits,
            fixed[1],
            fixed[2],
            offsets,
            grid.prefix_points,
            hard_alpha,
            direct_mse_weight,
        )
        loss.backward()
        step_nonfinite = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += step_nonfinite
        if step_nonfinite:
            raise FloatingPointError(
                "nonfinite hard-tail continuation gradient"
            )
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "classification": float(
                            parts["classification"].detach()
                        ),
                        "direct_mse": float(
                            parts["direct_mse"].detach()
                        ),
                        "fixed_rmse": sampled_rmse(
                            model, fixed, grid
                        ),
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
    passed = bool(
        final_fixed < 0.20 * initial_fixed
        and nonfinite == 0
        and abs(final_untouched - final_fixed) > 1e-6
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": "protected_parent_hard_tail_fixed_gate",
        "status": "passed" if passed else "failed",
        "fold": fold,
        "fixed_well_ids": [well.well_id for well in wells[:2]],
        "untouched_well_ids": [
            well.well_id for well in wells[2:4]
        ],
        "initial_fixed_sampled_rmse": initial_fixed,
        "final_fixed_sampled_rmse": final_fixed,
        "initial_untouched_sampled_rmse": initial_untouched,
        "final_untouched_sampled_rmse": final_untouched,
        "hard_alpha": hard_alpha,
        "direct_mse_weight": direct_mse_weight,
        "steps": steps,
        "nonfinite_gradient_elements": nonfinite,
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": sha256(checkpoint),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "seed": seed,
    }


def train(args: argparse.Namespace) -> None:
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
    if np.any(original[training] == 4):
        raise RuntimeError("F4 entered protected pure-CE continuation")
    train_wells = load_ids(
        identifiers, training, args.data_root, args.workers
    )
    held_wells = load_ids(
        identifiers, held, args.data_root, args.workers
    )
    grid = config()
    if args.overfit_only:
        if args.variant != "base12":
            raise ValueError(
                "the registered hard-tail gate uses the base12 parent"
            )
        result = hard_tail_fixed_gate(
            train_wells[:4],
            grid,
            args.fold,
            device,
            args.overfit_steps,
            args.seed,
            args.hard_alpha,
            args.direct_mse_weight,
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError(
                "protected hard-tail continuation gate failed"
            )
        return
    dataset = PureDataset(
        train_wells,
        grid,
        args.variant,
        args.repeats,
        args.seed,
    )
    dataset.synthetic = False
    source, model, checkpoint = initialized_models(
        args.variant, args.fold, device
    )
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    source_hash = sha256(Path(__file__))
    offsets = torch.as_tensor(
        grid.offsets, device=device, dtype=torch.float32
    )
    history = []
    snapshots = []
    nonfinite = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        dataset.epoch = epoch - 1
        loader = phase_loader(
            dataset,
            args.batch_size,
            args.loader_workers,
            args.seed + epoch,
        )
        losses = []
        components = []
        gradient_norms = []
        epoch_started = time.time()
        model.train()
        for image, labels, valid in loader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                source_logits = source(image[:, :19])
            logits = model(image)
            loss, parts = hard_tail_objective(
                logits,
                source_logits,
                labels,
                valid,
                offsets,
                grid.prefix_points,
                args.hard_alpha,
                args.direct_mse_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "nonfinite protected continuation loss"
                )
            loss.backward()
            step_nonfinite = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            nonfinite += step_nonfinite
            if step_nonfinite:
                raise FloatingPointError(
                    f"nonfinite continuation gradients={step_nonfinite}"
                )
            gradient_norms.append(
                float(
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                )
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append(
                {
                    key: float(value.detach())
                    for key, value in parts.items()
                }
            )
        scheduler.step()
        record = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "mean_components": {
                key: float(
                    np.mean([item[key] for item in components])
                )
                for key in components[0]
            },
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            snapshot = save_snapshot(
                args.output_dir,
                epoch,
                "protected_real_purece",
                model,
                held_wells,
                grid,
                args.variant,
                args.baseline_cache,
                device,
                args.fold,
                source_hash,
                args.seed,
            )
            snapshots.append(snapshot)
            print(
                json.dumps(
                    {
                        "snapshot": epoch,
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
        "variant": f"purece_protected_continuation_{args.variant}",
        "status": "complete_fixed_snapshots",
        "fold": args.fold,
        "seed": args.seed,
        "training_folds": [
            fold for fold in range(4) if fold != args.fold
        ],
        "training_wells": len(train_wells),
        "validation_wells": len(held_wells),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": sha256(checkpoint),
        "configuration": {
            "epochs": EPOCHS,
            "snapshots": list(SNAPSHOTS),
            "objective": args.objective,
            "hard_alpha": args.hard_alpha,
            "direct_mse_weight": args.direct_mse_weight,
            "synthetic_examples": 0,
            "dropout": 0.10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "repeats": args.repeats,
            "temperature_grid": list(TEMPERATURES),
            "blend_weight_grid": list(BLEND_WEIGHTS),
            "new_pair_channels_zero_at_initialization": True,
        },
        "history": history,
        "snapshots": snapshots,
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
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--objective",
        choices=("pure_ce", "hard_ce", "hard_ce_mse"),
        default="pure_ce",
    )
    parser.add_argument("--hard-alpha", type=float, default=0.0)
    parser.add_argument("--direct-mse-weight", type=float, default=0.0)
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=500)
    args = parser.parse_args()
    expected_objective = (
        "hard_ce_mse"
        if args.direct_mse_weight > 0.0
        else ("hard_ce" if args.hard_alpha > 0.0 else "pure_ce")
    )
    if args.objective != expected_objective:
        parser.error(
            "--objective does not match --hard-alpha/"
            "--direct-mse-weight"
        )
    if args.hard_alpha < 0.0 or args.direct_mse_weight < 0.0:
        parser.error("hard-tail objective weights must be nonnegative")
    if args.parity_gate:
        if args.output_json is None:
            parser.error("--output-json is required for the parity gate")
        if args.output_json.exists():
            raise FileExistsError(
                f"refusing to overwrite {args.output_json}"
            )
        result = parity_gate(
            args.fold, args.data_root, torch.device(args.device)
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "passed":
            raise RuntimeError("protected continuation parity failed")
        return
    if (
        args.variant is None
        or args.output_dir is None
        or (args.baseline_cache is None and not args.overfit_only)
    ):
        parser.error(
            "--variant and --output-dir are required; --baseline-cache "
            "is also required outside --overfit-only"
        )
    train(args)


if __name__ == "__main__":
    main()
