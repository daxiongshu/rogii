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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from limited_query_cv import train_v5_run6_local_scratch_corrected_unet as local
from rogii.alignment import expected_offset
from rogii.data import DEFAULT_DATA_ROOT, Well


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "pretrained_complete_well_categorical_segmentation"
VARIANT = "mit_b0_three_plane_categorical"
SEED = 20262301
LOCAL_SOURCE_SHA256 = (
    "cfd3e2f783f061a6b45c7a69d877127659d25b6bce679459005354a3ad8b44a1"
)
PRETRAINED_WEIGHT = Path(
    "/geo/.cache/huggingface/hub/models--nvidia--mit-b0/"
    "snapshots/80983a413c30d36a39c20203974ae7807835e2b4/"
    "pytorch_model.bin"
)
PRETRAINED_WEIGHT_SHA256 = (
    "4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc"
)
D570_SHA256 = (
    "5e3b6ee41f8705cc6b0044b36255bf8e8b3b6c4752c2afa9f6049d83f77cd310"
)
EPOCHS = 24
PURE_SYNTHETIC_EPOCHS = 8
MIXED_SYNTHETIC_PROBABILITY = 0.50
SNAPSHOTS = (4, 8, 12, 16, 24)
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SHARES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
DEFAULT_D570 = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050/"
    "d209_amplitude_clean.npz"
)


def configure_sources() -> None:
    if local.parent.core.sha256(Path(local.__file__)) != LOCAL_SOURCE_SHA256:
        raise RuntimeError("A820 frozen A770 source mismatch")
    if (
        not PRETRAINED_WEIGHT.is_file()
        or local.parent.core.sha256(PRETRAINED_WEIGHT)
        != PRETRAINED_WEIGHT_SHA256
    ):
        raise RuntimeError("A820 local MiT-B0 pretrained weight mismatch")
    local.configure_parent()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def three_planes(image: np.ndarray, extra: np.ndarray) -> np.ndarray:
    # A770 calibrated planes are horizontal GR then typewell GR.  Keep the
    # public three-view order: typewell, horizontal, inference-visible history.
    result = np.stack((extra[1], extra[0], image[6]), axis=0).astype(np.float32)
    if result.shape[0] != 3 or not np.isfinite(result).all():
        raise RuntimeError(f"invalid A820 three-plane image {result.shape}")
    return result


def make_example(
    well: Well,
    cut: int,
    grid,
    *,
    synthetic_rng: np.random.Generator | None = None,
    synthetic_hardness: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    image, extra, labels, valid, metadata = local.parent.core.make_example(
        well,
        cut,
        grid,
        "scratch_calibrated_local_b12",
        synthetic_rng=synthetic_rng,
        synthetic_hardness=synthetic_hardness,
    )
    return three_planes(image, extra), labels, valid, metadata


class CategoricalSegformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import SegformerModel

        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b0",
            local_files_only=True,
        )
        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels, 64, 1) for channels in hidden_sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(64 * len(hidden_sizes), 64, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(80, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            pixel_values=image,
            output_hidden_states=True,
            return_dict=True,
        )
        features = output.hidden_states
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(feature)
        decoded = self.fuse(torch.cat(projected, dim=1))
        decoded = F.interpolate(
            decoded,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.head(
            torch.cat((decoded, self.input_skip(image)), dim=1)
        ).squeeze(1)


def optimizer_for(model: CategoricalSegformer) -> torch.optim.Optimizer:
    decoder = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 5e-5},
            {"params": decoder, "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )


class SegformerDataset(Dataset):
    def __init__(
        self,
        wells: list[Well],
        grid,
        repeats: int,
        seed: int,
    ) -> None:
        self.wells = wells
        self.grid = grid
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.wells) * self.repeats

    def __getitem__(self, index: int):
        rng = np.random.default_rng(
            self.seed + self.epoch * len(self) + index
        )
        repeat = index // len(self.wells)
        well = self.wells[index % len(self.wells)]
        if repeat == 0:
            cut = well.anchor_index
        else:
            low = max(100, int(0.14 * len(well.md)))
            high = min(len(well.md) - 100, int(0.42 * len(well.md)))
            cut = int(rng.integers(low, high + 1))
        if self.epoch < PURE_SYNTHETIC_EPOCHS:
            synthetic = True
            hardness = 0.70 * (self.epoch + 1) / PURE_SYNTHETIC_EPOCHS
        else:
            synthetic = bool(rng.random() < MIXED_SYNTHETIC_PROBABILITY)
            hardness = 0.70 if synthetic else 0.0
        image, labels, valid, _ = make_example(
            well,
            cut,
            self.grid,
            synthetic_rng=rng if synthetic else None,
            synthetic_hardness=hardness,
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(labels.astype(np.int64)),
            torch.from_numpy(valid.astype(np.float32)),
            torch.as_tensor(int(synthetic), dtype=torch.int8),
        )


def loader(dataset: Dataset, batch_size: int, workers: int, seed: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=1 if workers else None,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def predict_temperatures(
    model: nn.Module,
    wells: list[Well],
    grid,
    device: torch.device,
) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray]:
    model.eval()
    offsets = torch.as_tensor(
        grid.offsets, dtype=torch.float32, device=device
    )
    starts = [0]
    parts = {temperature: [] for temperature in TEMPERATURES}
    truths = []
    for well in wells:
        image, _, valid, metadata = make_example(
            well, well.anchor_index, grid
        )
        logits = model(torch.from_numpy(image)[None].to(device))
        positions = np.asarray(metadata["positions"], dtype=np.float64)
        sampled_valid = valid > 0.5
        visible = sampled_valid & (positions <= well.anchor_index + 1e-6)
        visible_rows = np.flatnonzero(visible)[-32:]
        tail_rows = sampled_valid & (positions > well.anchor_index + 1e-6)
        if not len(visible_rows) or not np.any(tail_rows):
            raise RuntimeError(f"{well.well_id}: incomplete A820 sampled path")
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
            parts[temperature].append(native)
        truth = well.tvt[well.tail_indices].astype(np.float32)
        truths.append(truth)
        starts.append(starts[-1] + len(truth))
    return (
        np.asarray(starts, dtype=np.int64),
        {
            temperature: np.concatenate(values)
            for temperature, values in parts.items()
        },
        np.concatenate(truths),
    )


def sampled_rmse(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    grid,
) -> float:
    return local.parent.core.sampled_rmse_from_logits(
        logits, labels, valid, grid
    )


def stack_examples(
    wells: list[Well],
    grid,
    device: torch.device,
    *,
    synthetic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [
        make_example(
            well,
            well.anchor_index,
            grid,
            synthetic_rng=(
                np.random.default_rng(SEED + index)
                if synthetic
                else None
            ),
            synthetic_hardness=0.70 if synthetic else 0.0,
        )
        for index, well in enumerate(wells)
    ]
    return (
        torch.from_numpy(np.stack([value[0] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[1] for value in examples])).to(device),
        torch.from_numpy(np.stack([value[2] for value in examples])).to(device),
    )


def hidden_invariant(well: Well, grid) -> bool:
    images = []
    hidden = np.arange(len(well.tvt)) > well.anchor_index
    for mode in ("original", "reversed", "nan"):
        tvt = well.tvt.copy()
        if mode == "reversed":
            tvt[hidden] = tvt[hidden][::-1]
        elif mode == "nan":
            tvt[hidden] = np.nan
        image, _, _, _ = make_example(
            dataclasses.replace(well, tvt=tvt),
            well.anchor_index,
            grid,
        )
        images.append(image)
    return all(np.array_equal(images[0], image) for image in images[1:])


def evaluate(
    d570: np.ndarray,
    truth: np.ndarray,
    candidates: dict[float, np.ndarray],
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
                        candidates[temperature].astype(np.float64)
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
            prediction = d570 + share * (
                candidates[temperature] - d570
            )
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
    return {
        "parent_rmse": parent,
        "best_same_held_diagnostic_only": min(
            records,
            key=lambda item: (
                item["rmse"],
                item["share"],
                item["temperature"],
                item["treatment"],
            ),
        ),
        "grid": records,
    }


def gate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    configure_sources()
    seed_all(SEED)
    identifiers, original = local.parent.core.development_layout(
        args.data_root
    )
    development = np.flatnonzero(original < 4)
    wells = local.parent.core.load_ids(
        identifiers, development, args.data_root, args.workers
    )
    training = [
        well
        for well, fold in zip(wells, original[development])
        if int(fold) != args.fold
    ]
    grid = local.local_config()
    device = torch.device(args.device)
    fixed = stack_examples(training[:2], grid, device, synthetic=True)
    untouched = stack_examples(training[2:4], grid, device, synthetic=False)
    model = CategoricalSegformer().to(device)
    zero_head = bool(
        torch.count_nonzero(model.head[-1].weight) == 0
        and torch.count_nonzero(model.head[-1].bias) == 0
    )
    model.eval()
    with torch.no_grad():
        initial_fixed = sampled_rmse(
            model(fixed[0]), fixed[1], fixed[2], grid
        )
        initial_untouched = sampled_rmse(
            model(untouched[0]), untouched[1], untouched[2], grid
        )
    optimizer = optimizer_for(model)
    nonfinite = 0
    for step in range(args.overfit_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixed[0])
        loss = local.parent.pure_ce(
            logits, fixed[1], fixed[2], grid.prefix_points
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite A820 gate loss")
        loss.backward()
        count = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += count
        if count:
            raise FloatingPointError("nonfinite A820 gate gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {"step": step + 1, "loss": float(loss.detach())},
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        final_fixed = sampled_rmse(
            model(fixed[0]), fixed[1], fixed[2], grid
        )
        final_untouched = sampled_rmse(
            model(untouched[0]), untouched[1], untouched[2], grid
        )
    plane_contract = {
        "shape": list(untouched[0][0].shape),
        "finite": bool(torch.isfinite(untouched[0]).all()),
        "typewell_column_constant": bool(
            torch.equal(
                untouched[0][0, 0],
                untouched[0][0, 0, :, :1].expand_as(
                    untouched[0][0, 0]
                ),
            )
        ),
        "horizontal_state_constant": bool(
            torch.equal(
                untouched[0][0, 1],
                untouched[0][0, 1, :1, :].expand_as(
                    untouched[0][0, 1]
                ),
            )
        ),
        "history_bounded": bool(
            torch.all(
                (untouched[0][:, 2] >= 0.0)
                & (untouched[0][:, 2] <= 1.0)
            )
        ),
        "hidden_target_invariant": hidden_invariant(training[2], grid),
    }
    dummy = np.linspace(10000.0, 10100.0, 97)
    zero_share_exact = bool(
        np.array_equal(dummy + 0.0 * np.linspace(-2.0, 3.0, 97), dummy)
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": VARIANT,
        "status": "passed",
        "plane_contract": plane_contract,
        "capacity": {
            "steps": args.overfit_steps,
            "initial_fixed_sampled_rmse": initial_fixed,
            "final_fixed_sampled_rmse": final_fixed,
            "initial_untouched_sampled_rmse": initial_untouched,
            "final_untouched_sampled_rmse": final_untouched,
            "nonfinite_gradient_elements": nonfinite,
            "zero_head": zero_head,
        },
        "pretrained_weight": str(PRETRAINED_WEIGHT),
        "pretrained_weight_sha256": PRETRAINED_WEIGHT_SHA256,
        "local_source_sha256": LOCAL_SOURCE_SHA256,
        "source_sha256": local.parent.core.sha256(Path(__file__)),
        "d570_sha256": D570_SHA256,
        "d570_zero_share_exact": zero_share_exact,
        "epochs": EPOCHS,
        "pure_synthetic_epochs": PURE_SYNTHETIC_EPOCHS,
        "mixed_synthetic_probability": MIXED_SYNTHETIC_PROBABILITY,
        "snapshots": list(SNAPSHOTS),
        "temperatures": list(TEMPERATURES),
        "shares": list(SHARES),
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    passed = (
        plane_contract
        == {
            "shape": [3, len(grid.offsets), grid.length],
            "finite": True,
            "typewell_column_constant": True,
            "horizontal_state_constant": True,
            "history_bounded": True,
            "hidden_target_invariant": True,
        }
        and zero_head
        and final_fixed < 0.20 * initial_fixed
        and np.isfinite(final_untouched)
        and abs(final_untouched - initial_untouched) > 1e-6
        and nonfinite == 0
        and zero_share_exact
    )
    result["status"] = "passed" if passed else "failed"
    local.parent.core.write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("A820 implementation gate failed")


def verify_gate(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text())
    if (
        result.get("status") != "passed"
        or result.get("source_sha256")
        != local.parent.core.sha256(Path(__file__))
        or result.get("target_metric_computed") is not False
        or result.get("f4_loaded") is not False
        or result.get("pretrained_weight_sha256")
        != PRETRAINED_WEIGHT_SHA256
    ):
        raise RuntimeError("A820 gate absent or invalid")
    return result


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    configure_sources()
    verify_gate(args.gate)
    if local.parent.core.sha256(args.d570) != D570_SHA256:
        raise RuntimeError("A820 D570 archive mismatch")
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    identifiers, original = local.parent.core.development_layout(
        args.data_root
    )
    development = np.flatnonzero(original < 4)
    wells = local.parent.core.load_ids(
        identifiers, development, args.data_root, args.workers
    )
    lookup = {well.well_id: well for well in wells}
    train_ids = identifiers[(original < 4) & (original != args.fold)]
    held_ids = identifiers[original == args.fold]
    train_wells = [lookup[str(well_id)] for well_id in train_ids]
    held_wells = [lookup[str(well_id)] for well_id in held_ids]
    grid = local.local_config()
    dataset = SegformerDataset(
        train_wells, grid, args.repeats, SEED
    )
    device = torch.device(args.device)
    model = CategoricalSegformer().to(device)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    history = []
    snapshots = []
    nonfinite = 0
    source_hash = local.parent.core.sha256(Path(__file__))
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        dataset.epoch = epoch - 1
        batches = loader(
            dataset,
            args.batch_size,
            args.loader_workers,
            SEED + epoch,
        )
        model.train()
        losses, gradients = [], []
        synthetic_examples, real_examples = 0, 0
        for image, labels, valid, synthetic in batches:
            synthetic_examples += int(torch.sum(synthetic).item())
            real_examples += int(torch.sum(synthetic == 0).item())
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = local.parent.pure_ce(
                logits, labels, valid, grid.prefix_points
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite A820 production loss")
            loss.backward()
            count = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            nonfinite += count
            if count:
                raise FloatingPointError(
                    "nonfinite A820 production gradient"
                )
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
                if epoch <= PURE_SYNTHETIC_EPOCHS
                else "fixed_half_synthetic_mixture"
            ),
            "synthetic_examples": synthetic_examples,
            "real_examples": real_examples,
            "mean_loss": float(np.mean(losses)),
            "mean_gradient_norm": float(np.mean(gradients)),
            "encoder_learning_rate": float(
                scheduler.get_last_lr()[0]
            ),
            "decoder_learning_rate": float(
                scheduler.get_last_lr()[1]
            ),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            checkpoint = args.output_dir / f"epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), checkpoint)
            starts, candidates, truth = predict_temperatures(
                model, held_wells, grid, device
            )
            ids = np.asarray([well.well_id for well in held_wells])
            d570 = local.parent.core.load_d570(
                args.d570, ids, starts, truth
            )
            evaluation = evaluate(d570, truth, candidates)
            prediction = (
                args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            )
            arrays: dict[str, np.ndarray] = {
                "fold": np.asarray(args.fold, dtype=np.int8),
                "variant": np.asarray(VARIANT),
                "seed": np.asarray(SEED, dtype=np.int64),
                "epoch": np.asarray(epoch, dtype=np.int16),
                "well_ids": ids,
                "row_starts": starts,
                "d570": d570,
                "truth": truth,
                "source_sha256": np.asarray(source_hash),
                "checkpoint_sha256": np.asarray(
                    local.parent.core.sha256(checkpoint)
                ),
                "gate_sha256": np.asarray(
                    local.parent.core.sha256(args.gate)
                ),
                "audit_fold_loaded": np.asarray(False),
                "confirmation_regroupings_loaded": np.asarray(False),
                "f4_loaded": np.asarray(False),
            }
            for temperature, values in candidates.items():
                arrays[
                    f"candidate_t{int(round(100 * temperature)):03d}"
                ] = values
            np.savez_compressed(prediction, **arrays)
            snapshot = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": local.parent.core.sha256(checkpoint),
                "prediction": str(prediction),
                "prediction_sha256": local.parent.core.sha256(prediction),
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
        "variant": VARIANT,
        "fold": args.fold,
        "training_folds": sorted(
            int(value)
            for value in set(
                original[(original < 4) & (original != args.fold)]
            )
        ),
        "training_wells": len(train_wells),
        "held_wells": len(held_wells),
        "seed": SEED,
        "history": history,
        "snapshots": snapshots,
        "nonfinite_gradient_elements": nonfinite,
        "elapsed_seconds": time.time() - started,
        "source_sha256": source_hash,
        "gate_sha256": local.parent.core.sha256(args.gate),
        "pretrained_weight_sha256": PRETRAINED_WEIGHT_SHA256,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    local.parent.core.write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=range(4))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d570", type=Path, default=DEFAULT_D570)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overfit-steps", type=int, default=600)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.gate_only:
        gate(args)
    else:
        if args.gate is None:
            raise ValueError("--gate is required for production")
        train(args)


if __name__ == "__main__":
    main()
