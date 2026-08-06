from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well
from rogii.sequence import SequenceConfig


RUN_ID = "v5_batch2_run_006_goal_050"
COMPONENT_PATTERN = Path(
    "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
    "v20_components/fold{fold}.npz"
)
RESIDUAL_OFFSETS = np.arange(-32.0, 33.0, 1.0, dtype=np.float32)
BLEND_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
SNAPSHOTS = (2, 4, 8, 12, 16)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass
class ResidualRecord:
    well: Well
    fold: int
    baseline: np.ndarray
    components: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grid() -> SequenceConfig:
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


def load_records(data_root: Path) -> list[ResidualRecord]:
    records = []
    for fold in range(4):
        path = Path(str(COMPONENT_PATTERN).format(fold=fold))
        with np.load(path, allow_pickle=False) as cache:
            if bool(cache["audit_fold_loaded"]) or bool(
                cache["layout_truth_array_accessed"]
            ):
                raise RuntimeError(f"{path}: protected cache firewall failed")
            ids = cache["well_ids"].astype(str)
            starts = cache["row_starts"].astype(np.int64)
            baseline = cache["baseline"].astype(np.float32)
            truth = cache["truth"].astype(np.float32)
            components = cache["components"].astype(np.float32)
        for index, well_id in enumerate(ids):
            left, right = starts[index : index + 2]
            well = load_well(well_id, data_root)
            expected = well.tvt[well.prediction_indices].astype(np.float32)
            if not np.array_equal(truth[left:right], expected):
                raise RuntimeError(f"{well_id}: protected truth differs")
            records.append(
                ResidualRecord(
                    well=well,
                    fold=fold,
                    baseline=baseline[left:right].copy(),
                    components=components[:, left:right].copy(),
                )
            )
    return records


def direction_matches(record: ResidualRecord, direction: str) -> bool:
    if direction == "all":
        return True
    y = np.asarray(record.well.y, dtype=np.float64)
    anchor = int(record.well.anchor_index)
    delta_y = float(y[-1] - y[anchor])
    if not np.isfinite(delta_y):
        raise ValueError(f"{record.well.well_id}: nonfinite drilling direction")
    if direction == "negative_y":
        return delta_y < 0.0
    if direction == "positive_y":
        return delta_y >= 0.0
    raise ValueError(f"unsupported drilling direction: {direction}")


def smooth_perturbation(
    length: int,
    prefix_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    width = int(rng.integers(33, 130))
    kernel = np.hanning(width)
    kernel /= max(float(kernel.sum()), 1e-12)
    values = np.convolve(
        rng.normal(size=length + width - 1),
        kernel,
        mode="valid",
    )
    anchor = prefix_points - 1
    values -= values[anchor]
    tail = values[prefix_points:]
    values /= max(float(np.std(tail)), 1e-6)
    progress = np.clip(
        (np.arange(length) - anchor) / max(length - 1 - anchor, 1),
        0.0,
        1.0,
    )
    values *= np.power(progress, float(rng.uniform(0.55, 1.4)))
    values *= float(rng.uniform(4.0, 24.0))
    values[:prefix_points] = 0.0
    return values.astype(np.float32)


def sampled_prior(
    record: ResidualRecord,
    positions: np.ndarray,
    valid: np.ndarray,
    truth_tvt: np.ndarray,
    config: SequenceConfig,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    well = record.well
    anchor = well.anchor_index
    source = np.arange(len(well.tvt_input), dtype=np.float64)
    visible_values = np.nan_to_num(
        well.tvt_input,
        nan=float(well.tvt_input[anchor]),
    )
    visible_prior = np.interp(
        np.minimum(positions, float(anchor)),
        source[: anchor + 1],
        visible_values[: anchor + 1],
    )
    tail_prior = np.interp(
        positions,
        well.prediction_indices.astype(np.float64),
        record.baseline.astype(np.float64),
    )
    prior = np.where(positions <= anchor + 1e-6, visible_prior, tail_prior)
    synthetic = np.zeros(len(positions), dtype=np.float32)
    if rng is not None:
        synthetic = smooth_perturbation(
            len(positions), config.prefix_points, rng
        )
        prior = truth_tvt + synthetic
        prior[: config.prefix_points] = truth_tvt[: config.prefix_points]
    prior = np.where(valid > 0.5, prior, float(well.tvt_input[anchor]))
    return prior.astype(np.float32), synthetic


def resample_state_volume(
    image: np.ndarray,
    prior_offset: np.ndarray,
    source_offsets: np.ndarray,
) -> np.ndarray:
    coordinates = (
        prior_offset[:, None] + RESIDUAL_OFFSETS[None]
        - float(source_offsets[0])
    ) / float(source_offsets[1] - source_offsets[0])
    lower = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - lower
    supported = (lower >= 0) & (lower + 1 < len(source_offsets))
    lower = np.clip(lower, 0, len(source_offsets) - 2)
    transposed = image.transpose(0, 2, 1)
    left = np.take_along_axis(
        transposed,
        lower[None],
        axis=2,
    )
    right = np.take_along_axis(
        transposed,
        (lower + 1)[None],
        axis=2,
    )
    output = left * (1.0 - fraction[None]) + right * fraction[None]
    output *= supported[None]
    return output.transpose(0, 2, 1).astype(np.float32)


def make_residual_example(
    record: ResidualRecord,
    config: SequenceConfig,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    well = record.well
    image, _, metadata = make_alignment_example(
        well,
        well.anchor_index,
        config,
        coordinate_kind="tvt_delta",
    )
    positions = np.asarray(metadata["positions"], dtype=np.float64)
    valid = np.asarray(metadata["valid_positions"], dtype=np.float32)
    sampled_z = np.asarray(metadata["z"], dtype=np.float32)
    anchor_tvt = float(metadata["anchor_tvt"])
    truth_tvt = anchor_tvt + np.asarray(
        metadata["target_offset"], dtype=np.float32
    )
    prior_tvt, synthetic = sampled_prior(
        record,
        positions,
        valid,
        truth_tvt,
        config,
        rng,
    )
    prior_offset = prior_tvt - anchor_tvt
    volume = resample_state_volume(image, prior_offset, config.offsets)
    height, width = volume.shape[1:]

    def plane(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[None], (height, width))

    prior_surface = prior_tvt + sampled_z
    prior_surface_slope = np.gradient(prior_surface) / 20.0
    disagreement_native = np.std(
        record.components.astype(np.float64), axis=0
    )
    disagreement = np.interp(
        positions,
        well.prediction_indices.astype(np.float64),
        disagreement_native,
        left=0.0,
        right=float(disagreement_native[-1]),
    )
    additions = np.stack(
        [
            np.broadcast_to(
                (
                    RESIDUAL_OFFSETS
                    / max(float(np.max(np.abs(RESIDUAL_OFFSETS))), 1.0)
                )[:, None],
                (height, width),
            ),
            plane(prior_offset / 120.0),
            plane(prior_surface_slope),
            plane(disagreement / 40.0),
        ]
    ).astype(np.float32)
    volume = np.concatenate((volume, additions), axis=0)
    residual = truth_tvt - prior_tvt
    labels = np.argmin(
        np.abs(RESIDUAL_OFFSETS[:, None] - residual[None]),
        axis=0,
    ).astype(np.int64)
    label_valid = valid.copy()
    label_valid[
        np.abs(residual) > float(np.max(np.abs(RESIDUAL_OFFSETS)))
    ] = 0.0
    label_valid[: config.prefix_points] *= 0.25
    if not np.isfinite(volume).all():
        raise RuntimeError(f"{well.well_id}: nonfinite residual volume")
    return (
        volume,
        labels,
        label_valid,
        {
            "positions": positions.astype(np.float32),
            "prior_tvt": prior_tvt,
            "truth_tvt": truth_tvt,
            "residual": residual.astype(np.float32),
            "synthetic_prior_error": synthetic,
        },
    )


class ResidualDataset(Dataset):
    def __init__(
        self,
        records: list[ResidualRecord],
        config: SequenceConfig,
        repeats: int,
        seed: int,
        synthetic_probability: float,
    ) -> None:
        self.records = records
        self.config = config
        self.repeats = repeats
        self.seed = seed
        self.synthetic_probability = synthetic_probability
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(
            self.seed + self.epoch * len(self) + index
        )
        repeat = index // len(self.records)
        use_synthetic = (
            repeat > 0 and rng.random() < self.synthetic_probability
        )
        image, labels, valid, _ = make_residual_example(
            self.records[index % len(self.records)],
            self.config,
            rng if use_synthetic else None,
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(labels),
            torch.from_numpy(valid),
        )


def residual_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    prefix_points: int,
    hard_alpha: float,
    classification_weight: float,
    regression_kind: str,
    regression_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    offsets = torch.as_tensor(
        RESIDUAL_OFFSETS,
        device=logits.device,
        dtype=torch.float32,
    )
    tail = slice(prefix_points, None)
    truth = offsets[labels[:, tail]]
    hard_multiplier = 1.0 + hard_alpha * torch.clamp(
        torch.abs(truth) / 8.0,
        min=0.0,
        max=4.0,
    )
    weights = valid[:, tail] * hard_multiplier
    point_ce = F.cross_entropy(
        logits[:, :, tail],
        labels[:, tail],
        reduction="none",
    )
    denominator = weights.sum().clamp_min(1.0)
    classification = (point_ce * weights).sum() / denominator
    probability = torch.softmax(logits[:, :, tail].float(), dim=1)
    prediction = torch.sum(
        probability * offsets[None, :, None], dim=1
    )
    if regression_kind == "huber":
        point_regression = F.smooth_l1_loss(
            prediction / 8.0,
            truth / 8.0,
            reduction="none",
            beta=0.5,
        )
    elif regression_kind == "mse":
        point_regression = torch.square((prediction - truth) / 8.0)
    else:
        raise ValueError(f"unsupported regression kind: {regression_kind}")
    regression = (point_regression * weights).sum() / denominator
    pair_valid = weights[:, 1:] * weights[:, :-1]
    smoothness = (
        torch.abs(
            torch.diff(prediction, dim=1)
            - torch.diff(truth, dim=1)
        )
        * pair_valid
    ).sum() / pair_valid.sum().clamp_min(1.0)
    total = (
        classification_weight * classification
        + regression_weight * regression
        + 0.01 * smoothness
    )
    return total, {
        "classification": classification,
        "regression": regression,
        "smoothness": smoothness,
    }


@torch.no_grad()
def sampled_residual_rmse(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    image, labels, valid = batch
    logits = model(image)
    offsets = torch.as_tensor(
        RESIDUAL_OFFSETS,
        device=logits.device,
        dtype=torch.float32,
    )
    prediction = torch.sum(
        torch.softmax(logits.float(), dim=1) * offsets[None, :, None],
        dim=1,
    )
    truth = offsets[labels]
    selected = valid > 0.5
    return float(
        torch.sqrt(torch.mean(torch.square(prediction[selected] - truth[selected])))
    )


def stack_examples(
    records: list[ResidualRecord],
    config: SequenceConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [
        make_residual_example(record, config, None)[:3]
        for record in records
    ]
    return (
        torch.from_numpy(np.stack([item[0] for item in examples])).to(device),
        torch.from_numpy(np.stack([item[1] for item in examples])).to(device),
        torch.from_numpy(np.stack([item[2] for item in examples])).to(device),
    )


def fixed_batch_gate(
    records: list[ResidualRecord],
    config: SequenceConfig,
    device: torch.device,
    steps: int,
    seed: int,
    base: int,
    hard_alpha: float,
    classification_weight: float,
    regression_kind: str,
    regression_weight: float,
) -> dict[str, object]:
    fixed = stack_examples(records[:2], config, device)
    untouched = stack_examples(records[2:4], config, device)
    model = AlignmentUNet(input_channels=23, base=base).to(device)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )
    initial_fixed = sampled_residual_rmse(model, fixed)
    initial_untouched = sampled_residual_rmse(model, untouched)
    nonfinite = 0
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed[0])
        loss, _ = residual_objective(
            logits,
            fixed[1],
            fixed[2],
            config.prefix_points,
            hard_alpha,
            classification_weight,
            regression_kind,
            regression_weight,
        )
        loss.backward()
        step_nonfinite = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += step_nonfinite
        if step_nonfinite:
            raise FloatingPointError("nonfinite local-residual gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "fixed_rmse": sampled_residual_rmse(model, fixed),
                        "untouched_rmse": sampled_residual_rmse(
                            model, untouched
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    final_fixed = sampled_residual_rmse(model, fixed)
    final_untouched = sampled_residual_rmse(model, untouched)
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_structural_latent_alignment",
        "variant": "local_v20_residual_fixed_batch",
        "base": base,
        "residual_state_radius_ft": float(
            max(abs(float(RESIDUAL_OFFSETS[0])), abs(float(RESIDUAL_OFFSETS[-1])))
        ),
        "residual_state_step_ft": float(
            RESIDUAL_OFFSETS[1] - RESIDUAL_OFFSETS[0]
        ),
        "residual_state_count": int(len(RESIDUAL_OFFSETS)),
        "objective": {
            "hard_alpha": hard_alpha,
            "classification_weight": classification_weight,
            "regression_kind": regression_kind,
            "regression_weight": regression_weight,
        },
        "fixed_well_ids": [record.well.well_id for record in records[:2]],
        "untouched_well_ids": [
            record.well.well_id for record in records[2:4]
        ],
        "initial_fixed_residual_rmse": initial_fixed,
        "final_fixed_residual_rmse": final_fixed,
        "initial_untouched_residual_rmse": initial_untouched,
        "final_untouched_residual_rmse": final_untouched,
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
        "seed": seed,
    }


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64)
                    - truth.astype(np.float64)
                )
            )
        )
    )


@torch.no_grad()
def predict_record(
    model: nn.Module,
    record: ResidualRecord,
    config: SequenceConfig,
    device: torch.device,
    temperature: float,
) -> np.ndarray:
    image, _, _, metadata = make_residual_example(record, config, None)
    logits = model(torch.from_numpy(image)[None].to(device))[0]
    offsets = torch.as_tensor(
        RESIDUAL_OFFSETS,
        device=device,
        dtype=torch.float32,
    )
    correction = torch.sum(
        torch.softmax(logits.float() / temperature, dim=0)
        * offsets[:, None],
        dim=0,
    ).cpu().numpy()
    positions = metadata["positions"].astype(np.float64)
    valid = np.isfinite(positions) & (
        np.arange(len(positions)) >= config.prefix_points
    )
    native_correction = np.interp(
        record.well.prediction_indices,
        positions[valid],
        correction[valid],
    )
    return (
        record.baseline.astype(np.float64) + native_correction
    ).astype(np.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    records: list[ResidualRecord],
    config: SequenceConfig,
    device: torch.device,
) -> tuple[dict[str, object], dict[float, np.ndarray]]:
    model.eval()
    truth = np.concatenate(
        [
            record.well.tvt[record.well.prediction_indices]
            for record in records
        ]
    )
    baseline = np.concatenate([record.baseline for record in records])
    baseline_score = rmse(baseline, truth)
    predictions = {}
    grid_records = []
    for temperature in TEMPERATURES:
        prediction = np.concatenate(
            [
                predict_record(
                    model,
                    record,
                    config,
                    device,
                    temperature,
                )
                for record in records
            ]
        )
        predictions[temperature] = prediction
        for weight in BLEND_WEIGHTS:
            blend = baseline + weight * (prediction - baseline)
            score = rmse(blend, truth)
            grid_records.append(
                {
                    "temperature": temperature,
                    "weight": weight,
                    "standalone_rmse": rmse(prediction, truth),
                    "rmse": score,
                    "gain": baseline_score - score,
                }
            )
    best = min(
        grid_records,
        key=lambda item: (
            item["rmse"],
            item["weight"],
            item["temperature"],
        ),
    )
    return (
        {
            "baseline_rmse": baseline_score,
            "best_same_held_diagnostic_only": best,
            "grid": grid_records,
        },
        predictions,
    )


def train(args: argparse.Namespace) -> None:
    global RESIDUAL_OFFSETS
    if args.residual_radius <= 0.0 or args.residual_step <= 0.0:
        raise ValueError("residual radius and step must be positive")
    if (
        args.hard_alpha < 0.0
        or args.classification_weight < 0.0
        or args.regression_weight < 0.0
    ):
        raise ValueError("objective weights must be nonnegative")
    if args.classification_weight == 0.0 and args.regression_weight == 0.0:
        raise ValueError("at least one data-fit objective must be active")
    RESIDUAL_OFFSETS = np.arange(
        -args.residual_radius,
        args.residual_radius + 0.5 * args.residual_step,
        args.residual_step,
        dtype=np.float32,
    )
    if len(RESIDUAL_OFFSETS) < 3:
        raise ValueError("residual state grid is too small")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    config = grid()
    records = load_records(args.data_root)
    if args.overfit_only:
        result = fixed_batch_gate(
            records,
            config,
            device,
            args.overfit_steps,
            args.seed,
            args.base,
            args.hard_alpha,
            args.classification_weight,
            args.regression_kind,
            args.regression_weight,
        )
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError("local v20 residual fixed-batch gate failed")
        return

    records = [
        record
        for record in records
        if direction_matches(record, args.direction)
    ]
    train_records = [
        record for record in records if record.fold != args.fold
    ]
    held_records = [
        record for record in records if record.fold == args.fold
    ]
    dataset = ResidualDataset(
        train_records,
        config,
        repeats=args.repeats,
        seed=args.seed,
        synthetic_probability=args.synthetic_probability,
    )
    model = AlignmentUNet(input_channels=23, base=args.base).to(device)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    history = []
    snapshots = []
    source_hash = sha256(Path(__file__))
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        dataset.epoch = epoch - 1
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        losses = []
        components = []
        gradients = []
        model.train()
        epoch_started = time.time()
        for image, labels, valid in loader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss, parts = residual_objective(
                logits,
                labels,
                valid,
                config.prefix_points,
                args.hard_alpha,
                args.classification_weight,
                args.regression_kind,
                args.regression_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite local-residual loss")
            loss.backward()
            nonfinite = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if nonfinite:
                raise FloatingPointError(
                    f"nonfinite local-residual gradients={nonfinite}"
                )
            gradients.append(
                float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
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
            "mean_gradient_norm": float(np.mean(gradients)),
            "mean_components": {
                key: float(np.mean([item[key] for item in components]))
                for key in components[0]
            },
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            evaluation, predictions = evaluate(
                model, held_records, config, device
            )
            checkpoint = args.output_dir / f"epoch{epoch:03d}.pt"
            prediction_path = (
                args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            )
            torch.save(model.state_dict(), checkpoint)
            np.savez_compressed(
                prediction_path,
                fold=np.asarray(args.fold, dtype=np.int8),
                epoch=np.asarray(epoch, dtype=np.int16),
                well_ids=np.asarray(
                    [record.well.well_id for record in held_records]
                ),
                row_starts=np.concatenate(
                    (
                        [0],
                        np.cumsum(
                            [len(record.baseline) for record in held_records]
                        ),
                    )
                ).astype(np.int64),
                baseline=np.concatenate(
                    [record.baseline for record in held_records]
                ),
                truth=np.concatenate(
                    [
                        record.well.tvt[record.well.prediction_indices]
                        for record in held_records
                    ]
                ).astype(np.float32),
                **{
                    f"prediction_t{int(temperature * 100):03d}": prediction
                    for temperature, prediction in predictions.items()
                },
                audit_fold_loaded=np.asarray(False),
                confirmation_regroupings_loaded=np.asarray(False),
            )
            item = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "evaluation": evaluation,
            }
            snapshots.append(item)
            print(
                json.dumps(
                    {
                        "snapshot": epoch,
                        "best_same_held_diagnostic_only": evaluation[
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
        "family": "complete_well_structural_latent_alignment",
        "variant": "local_v20_residual_categorical_unet",
        "direction": args.direction,
        "direction_routing_rule": (
            "sign of Y_toe minus Y_at_final_visible_anchor"
        ),
        "residual_state_grid": {
            "radius_ft": float(args.residual_radius),
            "step_ft": float(args.residual_step),
            "states": int(len(RESIDUAL_OFFSETS)),
        },
        "status": "complete_F0_capacity_diagnostic",
        "fold": args.fold,
        "training_folds": [
            fold for fold in range(4) if fold != args.fold
        ],
        "training_wells": len(train_records),
        "held_wells": len(held_records),
        "parameters": vars(args)
        | {
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
        },
        "history": history,
        "snapshots": snapshots,
        "source_sha256": source_hash,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "selection_clean_nested_selector_run": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--base", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--synthetic-probability", type=float, default=0.5)
    parser.add_argument("--residual-radius", type=float, default=32.0)
    parser.add_argument("--residual-step", type=float, default=1.0)
    parser.add_argument(
        "--direction",
        choices=("all", "negative_y", "positive_y"),
        default="all",
    )
    parser.add_argument("--hard-alpha", type=float, default=0.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument(
        "--regression-kind",
        choices=("huber", "mse"),
        default="huber",
    )
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
