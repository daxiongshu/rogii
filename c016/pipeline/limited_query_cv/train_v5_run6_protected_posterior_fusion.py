from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from limited_query_cv import train_v5_run6_local_v20_residual as parent
from rogii.alignment import AlignmentUNet
from rogii.data import DEFAULT_DATA_ROOT, load_well


RUN_ID = "v5_batch2_run_006_goal_050"
RUN_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
POSTERIOR_PATTERN = RUN_ROOT / "protected_posterior_volume_fold{fold}.npz"
COMPONENT_PATTERN = Path(
    "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
    "v20_components/fold{fold}.npz"
)
D570 = RUN_ROOT / "d209_amplitude_clean.npz"
VARIANTS = ("posterior_only_b12", "posterior_physical_b12", "posterior_physical_b16")
SNAPSHOTS = (4, 8, 12, 16, 20, 24)
TEMPERATURES = (0.75, 1.0, 1.5, 2.0)
AMPLITUDES = (1.0, 1.5, 2.0, 3.0, 4.0)
SHARES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
SEED = 20261701


@dataclass
class FusionRecord:
    residual: parent.ResidualRecord
    posterior_probability: np.ndarray
    posterior_source_sha256: str


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


def load_fold_records(fold: int, data_root: Path) -> list[FusionRecord]:
    component_path = Path(str(COMPONENT_PATTERN).format(fold=fold))
    posterior_path = Path(str(POSTERIOR_PATTERN).format(fold=fold))
    with np.load(component_path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise RuntimeError(f"{component_path}: firewall failed")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        components = cache["components"].astype(np.float32)
    with np.load(posterior_path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError(f"{posterior_path}: protected flags set")
        if not bool(cache["hidden_targets_masked_during_inference"]):
            raise RuntimeError(f"{posterior_path}: hidden target not masked")
        if bool(cache["stratified_checkpoint_loaded"]):
            raise RuntimeError(f"{posterior_path}: stratified parent loaded")
        posterior_ids = cache["well_ids"].astype(str)
        posterior = cache["posterior_probability"].astype(np.float32)
        posterior_source = str(cache["source_sha256"])
        residual_offsets = cache["residual_offsets"].astype(np.float32)
        parent_names = cache["parent_names"].astype(str)
    if not np.array_equal(ids, posterior_ids):
        raise RuntimeError(f"fold {fold}: posterior/component IDs differ")
    if posterior.shape != (len(ids), 15, 65, 704):
        raise RuntimeError(f"fold {fold}: unexpected posterior shape {posterior.shape}")
    if not np.array_equal(residual_offsets, parent.RESIDUAL_OFFSETS):
        raise RuntimeError("posterior residual grid differs")
    if len(np.unique(parent_names)) != 15:
        raise RuntimeError("posterior parent names are not unique")
    records = []
    for index, well_id in enumerate(ids):
        left, right = starts[index : index + 2]
        well = load_well(well_id, data_root)
        if len(well.prediction_indices) != right - left:
            raise RuntimeError(f"{well_id}: native layout differs")
        records.append(
            FusionRecord(
                residual=parent.ResidualRecord(
                    well=well,
                    fold=fold,
                    baseline=baseline[left:right].copy(),
                    components=components[:, left:right].copy(),
                ),
                posterior_probability=posterior[index],
                posterior_source_sha256=posterior_source,
            )
        )
    return records


def posterior_planes(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    log_probability = np.log(np.maximum(probability, 1e-7))
    log_probability -= np.max(log_probability, axis=1, keepdims=True)
    log_probability = np.clip(log_probability, -16.0, 0.0) / 8.0
    arithmetic = np.mean(probability, axis=0, keepdims=True)
    aggregate = np.log(np.maximum(arithmetic, 1e-7))
    aggregate -= np.max(aggregate, axis=1, keepdims=True)
    aggregate = np.clip(aggregate, -16.0, 0.0) / 8.0
    dispersion = np.std(log_probability, axis=0, keepdims=True)
    return np.concatenate((log_probability, aggregate, dispersion), axis=0)


def make_example(
    record: FusionRecord,
    grid,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    physical, labels, valid, metadata = parent.make_residual_example(
        record.residual, grid, None
    )
    posterior = posterior_planes(record.posterior_probability)
    if posterior.shape[1:] != physical.shape[1:]:
        raise RuntimeError("posterior and physical volume shapes differ")
    if variant == "posterior_only_b12":
        image = np.concatenate((posterior, physical[-4:]), axis=0)
    elif variant in ("posterior_physical_b12", "posterior_physical_b16"):
        image = np.concatenate((posterior, physical), axis=0)
    else:
        raise ValueError(f"unknown A610 variant: {variant}")
    if not np.isfinite(image).all():
        raise RuntimeError(f"{record.residual.well.well_id}: nonfinite image")
    return image.astype(np.float32), labels, valid, metadata


def make_model(variant: str) -> AlignmentUNet:
    if variant == "posterior_only_b12":
        return AlignmentUNet(input_channels=21, base=12, dropout=0.10)
    if variant == "posterior_physical_b12":
        return AlignmentUNet(input_channels=40, base=12, dropout=0.10)
    if variant == "posterior_physical_b16":
        return AlignmentUNet(input_channels=40, base=16, dropout=0.10)
    raise ValueError(f"unknown A610 variant: {variant}")


class FusionDataset(Dataset):
    def __init__(self, records: list[FusionRecord], grid, variant: str, repeats: int):
        self.records = records
        self.grid = grid
        self.variant = variant
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def __getitem__(self, index: int):
        image, labels, valid, _ = make_example(
            self.records[index % len(self.records)], self.grid, self.variant
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
        )


def stack_examples(
    records: list[FusionRecord],
    grid,
    variant: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [make_example(record, grid, variant)[:3] for record in records]
    return tuple(
        torch.from_numpy(np.stack([item[index] for item in examples])).to(device)
        for index in range(3)
    )


@torch.no_grad()
def sampled_rmse(
    model: AlignmentUNet,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    model.eval()
    logits = model(batch[0])
    offsets = torch.as_tensor(
        parent.RESIDUAL_OFFSETS, device=logits.device, dtype=torch.float32
    )
    prediction = torch.sum(
        torch.softmax(logits.float(), dim=1) * offsets[None, :, None], dim=1
    )
    truth = offsets[batch[1]]
    selected = batch[2] > 0.5
    return float(
        torch.sqrt(torch.mean(torch.square(prediction[selected] - truth[selected])))
    )


def fixed_batch_gate(
    records: list[FusionRecord],
    grid,
    variant: str,
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    fixed = stack_examples(records[:2], grid, variant, device)
    untouched = stack_examples(records[2:4], grid, variant, device)
    model = make_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    initial_fixed = sampled_rmse(model, fixed)
    initial_untouched = sampled_rmse(model, untouched)
    nonfinite = 0
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed[0])
        loss, _ = parent.residual_objective(
            logits,
            fixed[1],
            fixed[2],
            grid.prefix_points,
            hard_alpha=0.0,
            classification_weight=1.0,
            regression_kind="huber",
            regression_weight=0.5,
        )
        loss.backward()
        step_nonfinite = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += step_nonfinite
        if step_nonfinite:
            raise FloatingPointError("A610 nonfinite gate gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "fixed_rmse": sampled_rmse(model, fixed),
                        "untouched_rmse": sampled_rmse(model, untouched),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    final_fixed = sampled_rmse(model, fixed)
    final_untouched = sampled_rmse(model, untouched)

    # The image path must remain invariant even when the hidden target is
    # replaced. Labels may change; they are deliberately excluded here.
    source = records[2]
    images = []
    for mode in ("original", "reverse", "nan"):
        well = source.residual.well
        hidden = np.arange(len(well.tvt)) > well.anchor_index
        target = well.tvt.copy()
        if mode == "reverse":
            target[hidden] = target[hidden][::-1]
        elif mode == "nan":
            target[hidden] = np.nan
        modified = replace(
            source,
            residual=replace(source.residual, well=replace(well, tvt=target)),
        )
        images.append(make_example(modified, grid, variant)[0])
    hidden_invariant = np.array_equal(images[0], images[1]) and np.array_equal(
        images[0], images[2]
    )

    # Exact parent no-ops and anchored correction are checked independently of
    # target metrics.
    baseline = source.residual.baseline.astype(np.float32)
    dummy_correction = np.linspace(0.0, 1.0, len(baseline), dtype=np.float32)
    dummy_correction -= dummy_correction[0]
    d570 = baseline + 0.25
    noops = np.array_equal(baseline + 0.0 * dummy_correction, baseline) and np.array_equal(
        d570 + 0.0 * dummy_correction, d570
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "fold_purged_protected_posterior_fusion",
        "variant": variant,
        "fixed_well_ids": [r.residual.well.well_id for r in records[:2]],
        "untouched_well_ids": [r.residual.well.well_id for r in records[2:4]],
        "initial_fixed_residual_rmse": initial_fixed,
        "final_fixed_residual_rmse": final_fixed,
        "initial_untouched_residual_rmse": initial_untouched,
        "final_untouched_residual_rmse": final_untouched,
        "steps": steps,
        "nonfinite_gradient_elements": nonfinite,
        "hidden_target_images_bit_identical": hidden_invariant,
        "first_hidden_anchor_exact": bool(dummy_correction[0] == 0.0),
        "v20_d570_zero_shares_exact": noops,
        "posterior_parent_count": 15,
        "training_folds": sorted({r.residual.fold for r in records}),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "status": (
            "passed"
            if final_fixed < 0.20 * initial_fixed
            and nonfinite == 0
            and hidden_invariant
            and noops
            else "failed"
        ),
    }
    return result


def load_d570(records: list[FusionRecord]) -> list[np.ndarray]:
    with np.load(D570, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError("D570 protected flags set")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        prediction = cache["prediction"].astype(np.float32)
    lookup = {well_id: index for index, well_id in enumerate(ids)}
    outputs = []
    for record in records:
        index = lookup[record.residual.well.well_id]
        left, right = starts[index : index + 2]
        outputs.append(prediction[left:right])
    return outputs


@torch.no_grad()
def predict_corrections(
    model: AlignmentUNet,
    records: list[FusionRecord],
    grid,
    variant: str,
    device: torch.device,
) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    offsets = torch.as_tensor(
        parent.RESIDUAL_OFFSETS, dtype=torch.float32, device=device
    )
    parts = {temperature: [] for temperature in TEMPERATURES}
    baseline_parts = []
    truth_parts = []
    starts = [0]
    for record in records:
        image, _, valid, metadata = make_example(record, grid, variant)
        logits = model(torch.from_numpy(image)[None].to(device))[0]
        positions = np.asarray(metadata["positions"], dtype=np.float64)
        prior_tvt = np.asarray(metadata["prior_tvt"], dtype=np.float64)
        # parent.make_residual_example returns the exact sampled validity mask
        # separately from metadata. Exclude padded duplicate tail columns.
        sampled_valid = np.asarray(valid) > 0.5
        sampled_valid[: grid.prefix_points] = False
        native = record.residual.well.prediction_indices.astype(np.float64)
        baseline = record.residual.baseline.astype(np.float32)
        truth = record.residual.well.tvt[
            record.residual.well.prediction_indices
        ].astype(np.float32)
        for temperature in TEMPERATURES:
            residual = torch.sum(
                torch.softmax(logits.float() / temperature, dim=0)
                * offsets[:, None],
                dim=0,
            ).cpu().numpy()
            sampled = prior_tvt + residual
            candidate = np.interp(
                native,
                positions[sampled_valid],
                sampled[sampled_valid],
            )
            correction = candidate - baseline
            correction -= correction[0]
            parts[temperature].append(
                np.clip(correction, -64.0, 64.0).astype(np.float32)
            )
        baseline_parts.append(baseline)
        truth_parts.append(truth)
        starts.append(starts[-1] + len(baseline))
    return (
        {key: np.concatenate(value) for key, value in parts.items()},
        np.concatenate(baseline_parts),
        np.concatenate(truth_parts),
        np.asarray(starts, dtype=np.int64),
    )


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64) - truth.astype(np.float64)
                )
            )
        )
    )


def evaluate_bank(
    corrections: dict[float, np.ndarray],
    baseline: np.ndarray,
    d570: np.ndarray,
    truth: np.ndarray,
) -> dict[str, object]:
    routes = {}
    for parent_name, base in (("v20", baseline), ("d570", d570)):
        parent_rmse = rmse(base, truth)
        grid = []
        for temperature in TEMPERATURES:
            correction = corrections[temperature]
            for amplitude in AMPLITUDES:
                for share in SHARES:
                    prediction = base + share * amplitude * correction
                    score = rmse(prediction, truth)
                    grid.append(
                        {
                            "temperature": temperature,
                            "amplitude": amplitude,
                            "share": share,
                            "effective_scale": amplitude * share,
                            "rmse": score,
                            "gain": parent_rmse - score,
                        }
                    )
        best = min(
            grid,
            key=lambda item: (
                item["rmse"],
                item["effective_scale"],
                item["share"],
                item["amplitude"],
                TEMPERATURES.index(item["temperature"]),
            ),
        )
        routes[parent_name] = {
            "parent_rmse": parent_rmse,
            "best_same_held_diagnostic_only": best,
            "grid": grid,
        }
    return routes


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    grid = parent.grid()

    training = []
    for fold in range(4):
        if fold != args.fold:
            training.extend(load_fold_records(fold, args.data_root))
    if any(record.residual.fold == args.fold for record in training):
        raise RuntimeError("held fold entered A610 training")
    if args.overfit_only:
        result = fixed_batch_gate(
            training[:4], grid, args.variant, device, args.overfit_steps
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError("A610 fixed-batch gate failed")
        return

    held = load_fold_records(args.fold, args.data_root)
    dataset = FusionDataset(training, grid, args.variant, args.repeats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        pin_memory=True,
        persistent_workers=args.loader_workers > 0,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = make_model(args.variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    d570_parts = load_d570(held)
    d570 = np.concatenate(d570_parts)
    history = []
    snapshots = []
    source_hash = sha256(Path(__file__))
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        gradients = []
        components = {"classification": [], "regression": [], "smoothness": []}
        epoch_started = time.time()
        for image, labels, valid in loader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, detail = parent.residual_objective(
                logits,
                labels,
                valid,
                grid.prefix_points,
                hard_alpha=0.0,
                classification_weight=1.0,
                regression_kind="huber",
                regression_weight=0.5,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("A610 nonfinite loss")
            loss.backward()
            nonfinite = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if nonfinite:
                raise FloatingPointError(f"A610 nonfinite gradients={nonfinite}")
            gradients.append(float(nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
            for key in components:
                components[key].append(float(detail[key].detach()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "mean_gradient_norm": float(np.mean(gradients)),
            "mean_components": {
                key: float(np.mean(values)) for key, values in components.items()
            },
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            checkpoint = args.output_dir / f"epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint)
            corrections, baseline, truth, starts = predict_corrections(
                model, held, grid, args.variant, device
            )
            if len(d570) != len(baseline):
                raise RuntimeError("D570 and held layout differ")
            evaluation = evaluate_bank(corrections, baseline, d570, truth)
            prediction_path = args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            np.savez_compressed(
                prediction_path,
                fold=np.asarray(args.fold, dtype=np.int8),
                variant=np.asarray(args.variant),
                well_ids=np.asarray([r.residual.well.well_id for r in held]),
                row_starts=starts,
                baseline=baseline,
                d570=d570,
                truth=truth,
                **{
                    f"correction_t{int(round(temperature * 100)):03d}": value
                    for temperature, value in corrections.items()
                },
                checkpoint_sha256=np.asarray(sha256(checkpoint)),
                source_sha256=np.asarray(source_hash),
                audit_fold_loaded=np.asarray(False),
                confirmation_regroupings_loaded=np.asarray(False),
            )
            snapshot = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "evaluation": evaluation,
            }
            snapshots.append(snapshot)
            print(
                json.dumps(
                    {
                        "snapshot": epoch,
                        "v20": evaluation["v20"][
                            "best_same_held_diagnostic_only"
                        ],
                        "d570": evaluation["d570"][
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
        "family": "fold_purged_protected_posterior_fusion",
        "variant": args.variant,
        "status": "complete_fixed_snapshots",
        "fold": args.fold,
        "training_folds": [fold for fold in range(4) if fold != args.fold],
        "training_wells": len(training),
        "held_wells": len(held),
        "parameters": vars(args) | {"output_dir": str(args.output_dir), "data_root": str(args.data_root)},
        "registered_snapshots": list(SNAPSHOTS),
        "registered_temperatures": list(TEMPERATURES),
        "registered_amplitudes": list(AMPLITUDES),
        "registered_shares": list(SHARES),
        "posterior_parent_count": 15,
        "objective": {
            "hard_alpha": 0.0,
            "classification_weight": 1.0,
            "regression_kind": "huber",
            "regression_weight": 0.5,
        },
        "history": history,
        "snapshots": snapshots,
        "source_sha256": source_hash,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "selection_boundary": (
            "same-held trajectories are capacity diagnostics only; no held "
            "recipe may select itself in a clean F0-F3 expansion"
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
