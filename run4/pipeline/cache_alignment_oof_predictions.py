from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


COMPONENTS = (
    ("fixed", "alignment_fixed_synth_fold{fold}.pt", "narrow", 1.5, 0.0537244898),
    ("distractor", "alignment_distractor_h05_fold{fold}.pt", "narrow", 1.5, 0.0358163265),
    ("wide", "alignment_wide_synth_fold{fold}.pt", "wide", 1.5, 0.0716326531),
    ("forward", "alignment_forward_std_fold{fold}.pt", "narrow", 1.0, 0.1164030612),
    ("forward_distractor", "alignment_forward_distractor_fold{fold}.pt", "narrow", 1.0, 0.0805867347),
    ("rare", "alignment_forward_exc4_fold{fold}.pt", "narrow", 1.25, 0.0551020408),
    ("supervised", "alignment_forward_exc2sup_fold{fold}.pt", "narrow", 1.0, 0.1239795918),
    ("wide_forward", "alignment_wide_forward_extreme_fold{fold}.pt", "wide", 1.0, 0.1377551020),
    ("wide_supervised", "alignment_wide_forward_extreme_exc2sup_fold{fold}.pt", "wide", 1.0, 0.325),
)


def load_all(data_root: Path, workers: int) -> list[Well]:
    ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(well_id, data_root), ids))


@torch.no_grad()
def indexed_prediction(
    model: AlignmentUNet,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    well: Well,
    config: SequenceConfig,
    temperature: float,
    device: torch.device,
    target_indices: np.ndarray,
) -> np.ndarray:
    logits = model(torch.from_numpy(image).unsqueeze(0).to(device))[
        0, :, config.prefix_points :
    ]
    valid = np.asarray(metadata["valid_positions"])[config.prefix_points :] > 0.5
    logits = logits[:, valid]
    offsets = torch.as_tensor(config.offsets, dtype=torch.float32, device=device)
    sampled_offset = torch.sum(
        torch.softmax(logits / temperature, dim=0) * offsets[:, None], dim=0
    )
    positions = np.asarray(metadata["positions"])[config.prefix_points :][valid]
    sampled_tvt = float(metadata["anchor_tvt"]) + sampled_offset.float().cpu().numpy()
    return np.interp(target_indices, positions, sampled_tvt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backtest-rows", type=int, default=0)
    parser.add_argument("--minimum-backtest-cut", type=int, default=100)
    parser.add_argument("--gr-smooth-window", type=int, default=51)
    args = parser.parse_args()

    started = time.time()
    wells = load_all(args.data_root, args.workers)
    identifiers = np.asarray([well.well_id for well in wells])
    _, valid_indices = list(GroupKFold(5).split(identifiers, groups=identifiers))[
        args.fold
    ]
    device = torch.device(args.device)
    narrow = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=64.0,
        state_step=0.5,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=args.gr_smooth_window,
    )
    wide = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=120.0,
        state_step=1.0,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=args.gr_smooth_window,
    )
    configs = {"narrow": narrow, "wide": wide}

    models = []
    for name, pattern, scale, temperature, weight in COMPONENTS:
        checkpoint = args.checkpoint_root / pattern.format(fold=args.fold)
        model = AlignmentUNet(base=args.base).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        models.append((name, model, scale, temperature, weight))

    well_ids: list[str] = []
    row_starts = [0]
    row_indices: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    component_predictions: list[np.ndarray] = []
    backtest_cuts: list[int] = []
    backtest_starts = [0]
    backtest_indices: list[np.ndarray] = []
    backtest_truths: list[np.ndarray] = []
    backtest_predictions: list[np.ndarray] = []
    for count, index in enumerate(valid_indices, start=1):
        well = wells[index]
        examples = {
            key: make_alignment_example(well, well.anchor_index, config)
            for key, config in configs.items()
        }
        predictions = []
        for _, model, scale, temperature, _ in models:
            image, _, metadata = examples[scale]
            predictions.append(
                indexed_prediction(
                    model,
                    image,
                    metadata,
                    well,
                    configs[scale],
                    temperature,
                    device,
                    well.tail_indices,
                )
            )
        matrix = np.stack(predictions).astype(np.float32)
        truth = well.tvt[well.tail_indices].astype(np.float32)
        well_ids.append(well.well_id)
        row_indices.append(well.tail_indices.astype(np.int32))
        truths.append(truth)
        component_predictions.append(matrix)
        row_starts.append(row_starts[-1] + len(truth))

        if args.backtest_rows:
            backtest_cut = max(
                args.minimum_backtest_cut,
                well.anchor_index - args.backtest_rows,
            )
            target_indices = np.arange(
                backtest_cut + 1, well.anchor_index + 1, dtype=np.int32
            )
            backtest_examples = {
                key: make_alignment_example(well, backtest_cut, config)
                for key, config in configs.items()
            }
            predicted = []
            for _, model, scale, temperature, _ in models:
                image, _, metadata = backtest_examples[scale]
                predicted.append(
                    indexed_prediction(
                        model,
                        image,
                        metadata,
                        well,
                        configs[scale],
                        temperature,
                        device,
                        target_indices,
                    )
                )
            backtest_cuts.append(backtest_cut)
            backtest_indices.append(target_indices)
            backtest_truths.append(well.tvt[target_indices].astype(np.float32))
            backtest_predictions.append(np.stack(predicted).astype(np.float32))
            backtest_starts.append(backtest_starts[-1] + len(target_indices))
        if count % 20 == 0 or count == len(valid_indices):
            print(
                f"fold={args.fold} wells={count}/{len(valid_indices)} "
                f"rows={row_starts[-1]} elapsed={time.time() - started:.1f}",
                flush=True,
            )

    component_matrix = np.concatenate(component_predictions, axis=1)
    truth_vector = np.concatenate(truths)
    weights = np.asarray([item[4] for item in COMPONENTS], dtype=np.float64)
    coarse = weights @ component_matrix.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(coarse - truth_vector))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "fold": np.int64(args.fold),
        "well_ids": np.asarray(well_ids),
        "row_starts": np.asarray(row_starts, dtype=np.int64),
        "row_indices": np.concatenate(row_indices),
        "truth": truth_vector,
        "components": component_matrix,
        "coarse": coarse.astype(np.float32),
        "component_names": np.asarray([item[0] for item in COMPONENTS]),
        "component_weights": weights,
        "component_temperatures": np.asarray([item[3] for item in COMPONENTS]),
    }
    if args.backtest_rows:
        output.update(
            backtest_rows=np.int64(args.backtest_rows),
            backtest_cuts=np.asarray(backtest_cuts, dtype=np.int32),
            backtest_starts=np.asarray(backtest_starts, dtype=np.int64),
            backtest_indices=np.concatenate(backtest_indices),
            backtest_truth=np.concatenate(backtest_truths),
            backtest_components=np.concatenate(backtest_predictions, axis=1),
        )
    np.savez_compressed(
        args.output,
        **output,
    )
    print(
        f"fold={args.fold} rmse={rmse:.6f} output={args.output} "
        f"elapsed={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
