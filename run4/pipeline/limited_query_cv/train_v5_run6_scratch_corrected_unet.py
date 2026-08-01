from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from limited_query_cv import train_v5_run6_frozen_residual_rawstate as core
from limited_query_cv import train_v5_run6_true_raw_absolute_residual as raw
from limited_query_cv import train_v5_run6_calibrated_separable_residual as calibrated
from limited_query_cv.train_v5_run6_pure_categorical import pure_ce
from rogii.alignment import AlignmentUNet, expected_offset, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "scratch_complete_well_corrected_input_unet"
RAW_SOURCE_SHA256 = (
    "860f335542a8f3f51cdc9fda11d9ed9c9147d1c6a9931bb4babb995333ac0ec5"
)
CALIBRATED_SOURCE_SHA256 = (
    "4750df1773ebd86361e3ee3f8de76a589a65d1cfec0390fd116a3d4e3912fc41"
)
VARIANTS = {
    "scratch_true_raw_b12": {
        "extra_channels": 8,
        "base": 12,
        "seed": 20262051,
    },
    "scratch_calibrated_b12": {
        "extra_channels": 8,
        "base": 12,
        "seed": 20262059,
    },
}
EPOCHS = 32
SYNTHETIC_EPOCHS = 4
SNAPSHOTS = (4, 8, 12, 16, 24, 32)
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SHARES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)

for _variant, _specification in VARIANTS.items():
    core.VARIANTS[_variant] = dict(_specification)
_PREVIOUS_RESIDUAL_PLANES = core.residual_planes


def corrected_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
    variant: str,
) -> np.ndarray:
    if variant == "scratch_true_raw_b12":
        return raw.true_raw_absolute_planes(well, cut, image, metadata, grid)
    if variant == "scratch_calibrated_b12":
        return calibrated.calibrated_separable_planes(
            well, cut, image, metadata, grid
        )[0]
    return _PREVIOUS_RESIDUAL_PLANES(
        well, cut, image, metadata, grid, variant
    )


core.residual_planes = corrected_planes
core.SYNTHETIC_EPOCHS = SYNTHETIC_EPOCHS


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(variant: str, device: torch.device) -> AlignmentUNet:
    specification = VARIANTS[variant]
    return AlignmentUNet(
        input_channels=19 + int(specification["extra_channels"]),
        base=int(specification["base"]),
        dropout=0.10,
    ).to(device)


def model_input(image: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    return torch.cat((image, extra), dim=1)


def source_feature_contract(
    well: Well,
    grid,
    variant: str,
    synthetic: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(20262073) if synthetic else None
    image, _, metadata = make_alignment_example(
        well,
        well.anchor_index,
        grid,
        synthetic_rng=rng,
        synthetic_profile="forward_extreme",
        synthetic_hardness=0.70,
        coordinate_kind="tvt_delta",
    )
    if variant == "scratch_true_raw_b12":
        direct = raw.true_raw_absolute_planes(
            well, well.anchor_index, image, metadata, grid
        )
        reconstruction_rmse = None
    else:
        direct, recovery = calibrated.calibrated_separable_planes(
            well, well.anchor_index, image, metadata, grid
        )
        reconstruction_rmse = float(recovery["reconstruction_rmse"])
    generated = core.make_example(
        well,
        well.anchor_index,
        grid,
        variant,
        synthetic_rng=np.random.default_rng(20262073) if synthetic else None,
        synthetic_hardness=0.70 if synthetic else 0.0,
    )
    return {
        "synthetic": synthetic,
        "base_image_bit_identical": bool(np.array_equal(image, generated[0])),
        "extra_features_bit_identical": bool(
            np.array_equal(direct, generated[1])
        ),
        "shape": list(generated[1].shape),
        "finite": bool(
            np.isfinite(generated[0]).all()
            and np.isfinite(generated[1]).all()
        ),
        "reconstruction_rmse": reconstruction_rmse,
    }


def scratch_fixed_gate(
    wells: list[Well],
    grid,
    variant: str,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    seed = int(VARIANTS[variant]["seed"])
    seed_all(seed)
    model = make_model(variant, device)
    head_zero = bool(
        torch.count_nonzero(model.head.weight) == 0
        and torch.count_nonzero(model.head.bias) == 0
    )
    fixed = core.stack_examples(wells[:2], grid, variant, device)
    untouched = core.stack_examples(wells[2:4], grid, variant, device)
    model.eval()
    with torch.no_grad():
        initial_fixed_logits = model(model_input(fixed[0], fixed[1]))
        initial_untouched_logits = model(
            model_input(untouched[0], untouched[1])
        )
    uniform_logits = bool(torch.count_nonzero(initial_fixed_logits) == 0)
    initial_fixed = core.sampled_rmse_from_logits(
        initial_fixed_logits, fixed[2], fixed[3], grid
    )
    initial_untouched = core.sampled_rmse_from_logits(
        initial_untouched_logits, untouched[2], untouched[3], grid
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )
    nonfinite = 0
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(model_input(fixed[0], fixed[1]))
        loss = pure_ce(logits, fixed[2], fixed[3], grid.prefix_points)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite A730 gate loss")
        loss.backward()
        count = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += count
        if count:
            raise FloatingPointError("nonfinite A730 gate gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 200 == 0:
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "step": step + 1,
                        "loss": float(loss.detach()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        final_fixed_logits = model(model_input(fixed[0], fixed[1]))
        final_untouched_logits = model(
            model_input(untouched[0], untouched[1])
        )
    final_fixed = core.sampled_rmse_from_logits(
        final_fixed_logits, fixed[2], fixed[3], grid
    )
    final_untouched = core.sampled_rmse_from_logits(
        final_untouched_logits, untouched[2], untouched[3], grid
    )
    starts, prediction, truth = predict_temperatures(
        model, wells[2:4], grid, variant, device
    )
    prediction_coverage = (
        len(starts) == 3
        and starts[0] == 0
        and starts[-1] == len(truth)
        and all(
            len(value) == len(truth) and np.isfinite(value).all()
            for value in prediction.values()
        )
    )
    dummy_a = np.linspace(-2.0, 1.0, 97, dtype=np.float64)
    dummy_b = np.linspace(3.0, -1.0, 97, dtype=np.float64)
    pair_direct = 0.5 * dummy_a + 0.5 * dummy_b
    pair_independent = np.mean(
        np.stack((dummy_a, dummy_b)).astype(np.float64), axis=0
    )
    pair_exact = bool(np.array_equal(pair_direct, pair_independent))
    d570 = np.linspace(10000.0, 10100.0, 97, dtype=np.float64)
    zero_share_exact = bool(
        np.array_equal(d570 + 0.0 * (pair_direct - d570), d570)
    )
    result = {
        "variant": variant,
        "seed": seed,
        "model_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "zero_head": head_zero,
        "uniform_initial_logits": uniform_logits,
        "initial_fixed_sampled_rmse": initial_fixed,
        "final_fixed_sampled_rmse": final_fixed,
        "initial_untouched_sampled_rmse": initial_untouched,
        "final_untouched_sampled_rmse": final_untouched,
        "nonfinite_gradient_elements": nonfinite,
        "hidden_target_features_bit_identical": core.hidden_feature_invariance(
            wells[2], grid, variant
        ),
        "prediction_coverage_and_finiteness": prediction_coverage,
        "fixed_pair_fp64_exact": pair_exact,
        "d570_zero_share_exact": zero_share_exact,
        "steps": steps,
    }
    result["status"] = (
        "passed"
        if head_zero
        and uniform_logits
        and final_fixed < 0.20 * initial_fixed
        and np.isfinite(final_untouched)
        and abs(final_untouched - initial_untouched) > 1e-6
        and nonfinite == 0
        and result["hidden_target_features_bit_identical"]
        and prediction_coverage
        and pair_exact
        and zero_share_exact
        else "failed"
    )
    return result


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
    truth_parts = []
    for well in wells:
        image, extra, _, valid, metadata = core.make_example(
            well, well.anchor_index, grid, variant
        )
        tensor = model_input(
            torch.from_numpy(image)[None].to(device),
            torch.from_numpy(extra)[None].to(device),
        )
        logits = model(tensor)
        positions = np.asarray(metadata["positions"], dtype=np.float64)
        sampled_valid = valid > 0.5
        visible = sampled_valid & (positions <= well.anchor_index + 1e-6)
        visible_rows = np.flatnonzero(visible)[-32:]
        tail_rows = sampled_valid & (positions > well.anchor_index + 1e-6)
        if len(visible_rows) == 0 or not np.any(tail_rows):
            raise RuntimeError(f"{well.well_id}: incomplete sampled path")
        visible_truth = np.interp(
            positions[visible_rows],
            np.arange(len(well.tvt_input), dtype=np.float64),
            np.nan_to_num(
                well.tvt_input, nan=well.tvt_input[well.anchor_index]
            ),
        )
        for temperature in TEMPERATURES:
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
            predictions[temperature].append(native)
        truth = well.tvt[well.tail_indices].astype(np.float32)
        truth_parts.append(truth)
        starts.append(starts[-1] + len(truth))
    return (
        np.asarray(starts, dtype=np.int64),
        {
            temperature: np.concatenate(parts)
            for temperature, parts in predictions.items()
        },
        np.concatenate(truth_parts),
    )


def evaluate(
    d570: np.ndarray,
    truth: np.ndarray,
    candidate: dict[float, np.ndarray],
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
        standalone = float(
            np.sqrt(
                np.mean(
                    np.square(
                        candidate[temperature].astype(np.float64)
                        - truth.astype(np.float64)
                    )
                )
            )
        )
        records.append(
            {
                "treatment": "standalone",
                "temperature": temperature,
                "share": 1.0,
                "rmse": standalone,
                "gain": parent - standalone,
            }
        )
        for share in SHARES:
            prediction = d570 + share * (candidate[temperature] - d570)
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
                    "treatment": "candidate_replacement",
                    "temperature": temperature,
                    "share": share,
                    "rmse": score,
                    "gain": parent - score,
                }
            )
    best = min(
        records,
        key=lambda item: (
            item["rmse"],
            item["share"],
            item["temperature"],
            item["treatment"],
        ),
    )
    return {
        "parent_rmse": parent,
        "best_same_held_diagnostic_only": best,
        "grid": records,
    }


def verify_sources(aggregate_gate: Path) -> dict[str, object]:
    if core.sha256(Path(raw.__file__)) != RAW_SOURCE_SHA256:
        raise RuntimeError("A730 A700 source hash mismatch")
    if core.sha256(Path(calibrated.__file__)) != CALIBRATED_SOURCE_SHA256:
        raise RuntimeError("A730 A710 source hash mismatch")
    return raw.verify_frozen_inputs(aggregate_gate)


def run_gate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    aggregate = verify_sources(args.aggregate_gate)
    identifiers, original = core.development_layout(args.data_root)
    development = np.flatnonzero(original < 4)
    wells = core.load_ids(
        identifiers, development, args.data_root, args.workers
    )
    training = [
        well
        for well, fold in zip(wells, original[development])
        if int(fold) != args.fold
    ]
    grid = core.config()
    contracts = [
        source_feature_contract(training[0], grid, args.variant, synthetic)
        for synthetic in (False, True)
    ]
    capacity = scratch_fixed_gate(
        training,
        grid,
        args.variant,
        torch.device(args.device),
        args.overfit_steps,
    )
    contracts_passed = all(
        record["base_image_bit_identical"]
        and record["extra_features_bit_identical"]
        and record["shape"] == [8, len(grid.offsets), grid.length]
        and record["finite"]
        and (
            args.variant != "scratch_calibrated_b12"
            or float(record["reconstruction_rmse"]) <= 0.15
        )
        for record in contracts
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": args.variant,
        "status": (
            "passed"
            if contracts_passed and capacity["status"] == "passed"
            else "failed"
        ),
        "feature_contracts": contracts,
        "capacity": capacity,
        "specification": VARIANTS[args.variant],
        "epochs": EPOCHS,
        "synthetic_epochs": SYNTHETIC_EPOCHS,
        "snapshots": list(SNAPSHOTS),
        "temperatures": list(TEMPERATURES),
        "shares": list(SHARES),
        "source_sha256": core.sha256(Path(__file__)),
        "raw_source_sha256": RAW_SOURCE_SHA256,
        "calibrated_source_sha256": CALIBRATED_SOURCE_SHA256,
        "aggregate_gate_sha256": core.sha256(args.aggregate_gate),
        "aggregate_gate_status": aggregate["status"],
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    core.write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise RuntimeError("A730 implementation gate failed")


def verify_gate(path: Path, variant: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "passed"
        or payload.get("variant") != variant
        or payload.get("source_sha256") != core.sha256(Path(__file__))
        or payload.get("target_metric_computed") is not False
        or payload.get("f4_loaded") is not False
    ):
        raise RuntimeError("A730 gate is absent or invalid")
    return payload


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    verify_sources(args.aggregate_gate)
    gate = verify_gate(args.gate, args.variant)
    specification = VARIANTS[args.variant]
    seed = int(specification["seed"])
    seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    identifiers, original = core.development_layout(args.data_root)
    development = np.flatnonzero(original < 4)
    all_wells = core.load_ids(
        identifiers, development, args.data_root, args.workers
    )
    lookup = {well.well_id: well for well in all_wells}
    train_ids = identifiers[(original < 4) & (original != args.fold)]
    held_ids = identifiers[original == args.fold]
    train_wells = [lookup[str(well_id)] for well_id in train_ids]
    held_wells = [lookup[str(well_id)] for well_id in held_ids]
    grid = core.config()
    dataset = core.ResidualDataset(
        train_wells, grid, args.variant, args.repeats, seed
    )
    model = make_model(args.variant, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    source_hash = core.sha256(Path(__file__))
    history, snapshots = [], []
    nonfinite = 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        dataset.epoch = epoch - 1
        batches = core.loader(
            dataset,
            args.batch_size,
            args.loader_workers,
            seed + epoch,
        )
        model.train()
        losses, gradients = [], []
        for image, extra, labels, valid in batches:
            image = image.to(device, non_blocking=True)
            extra = extra.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(model_input(image, extra))
            loss = pure_ce(logits, labels, valid, grid.prefix_points)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite A730 production loss")
            loss.backward()
            count = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            nonfinite += count
            if count:
                raise FloatingPointError("nonfinite A730 production gradient")
            gradients.append(
                float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "stage": (
                "forward_extreme_synthetic"
                if epoch <= SYNTHETIC_EPOCHS
                else "real_complete_well_random_cut"
            ),
            "mean_loss": float(np.mean(losses)),
            "mean_gradient_norm": float(np.mean(gradients)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            checkpoint = args.output_dir / f"epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint)
            starts, candidate, truth = predict_temperatures(
                model, held_wells, grid, args.variant, device
            )
            ids = np.asarray([well.well_id for well in held_wells])
            d570 = core.load_d570(args.d570, ids, starts, truth)
            evaluation = evaluate(d570, truth, candidate)
            prediction = args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            arrays: dict[str, np.ndarray] = {
                "fold": np.asarray(args.fold, dtype=np.int8),
                "variant": np.asarray(args.variant),
                "seed": np.asarray(seed, dtype=np.int64),
                "epoch": np.asarray(epoch, dtype=np.int16),
                "well_ids": ids,
                "row_starts": starts,
                "d570": d570,
                "truth": truth,
                "source_sha256": np.asarray(source_hash),
                "checkpoint_sha256": np.asarray(core.sha256(checkpoint)),
                "gate_sha256": np.asarray(core.sha256(args.gate)),
                "audit_fold_loaded": np.asarray(False),
                "confirmation_regroupings_loaded": np.asarray(False),
                "f4_loaded": np.asarray(False),
            }
            for temperature, values in candidate.items():
                arrays[
                    f"candidate_t{int(round(100 * temperature)):03d}"
                ] = values
            np.savez_compressed(prediction, **arrays)
            snapshot = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": core.sha256(checkpoint),
                "prediction": str(prediction),
                "prediction_sha256": core.sha256(prediction),
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
        "training_wells": len(train_wells),
        "held_wells": len(held_wells),
        "specification": specification,
        "epochs": EPOCHS,
        "synthetic_epochs": SYNTHETIC_EPOCHS,
        "snapshots": snapshots,
        "history": history,
        "nonfinite_gradient_elements": nonfinite,
        "elapsed_seconds": time.time() - started,
        "source_sha256": source_hash,
        "gate_sha256": core.sha256(args.gate),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    core.write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=range(4))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d570", type=Path, default=core.DEFAULT_D570)
    parser.add_argument(
        "--aggregate-gate",
        type=Path,
        default=core.DEFAULT_AGGREGATE_GATE,
    )
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overfit-steps", type=int, default=1200)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.gate_only:
        run_gate(args)
    else:
        if args.gate is None:
            raise ValueError("--gate is required for production")
        train(args)


if __name__ == "__main__":
    main()
