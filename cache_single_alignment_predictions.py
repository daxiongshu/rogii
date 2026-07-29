from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from compare_alignment_scales import sampled_predictions
from rogii.alignment import AlignmentUNet
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.folds import fold_indices
from rogii.sequence import SequenceConfig


def load_all(data_root: Path, workers: int) -> list[Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(well_id, data_root), well_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fold-file", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument("--state-radius", type=float, default=120.0)
    parser.add_argument("--state-step", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gr-smooth-window", type=int, default=51)
    args = parser.parse_args()

    wells = load_all(args.data_root, args.workers)
    identifiers = np.asarray([well.well_id for well in wells])
    _, valid_indices = fold_indices(identifiers, args.fold, 5, args.fold_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=args.state_radius,
        state_step=args.state_step,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=args.gr_smooth_window,
    )
    offsets = torch.as_tensor(config.offsets, dtype=torch.float32, device=device)
    model = AlignmentUNet(base=args.base).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    well_ids = []
    starts = [0]
    predictions = []
    truths = []
    for count, index in enumerate(valid_indices, start=1):
        well = wells[index]
        prediction = sampled_predictions(
            model,
            well,
            config,
            offsets,
            device,
            (args.temperature,),
        )[args.temperature]
        truth = well.tvt[well.tail_indices]
        well_ids.append(well.well_id)
        predictions.append(prediction.astype(np.float32))
        truths.append(truth.astype(np.float32))
        starts.append(starts[-1] + len(truth))
        if count % 25 == 0 or count == len(valid_indices):
            print(f"wells={count}/{len(valid_indices)}", flush=True)
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fold=np.int64(args.fold),
        well_ids=np.asarray(well_ids),
        row_starts=np.asarray(starts, dtype=np.int64),
        prediction=prediction,
        truth=truth,
    )
    print(f"rmse={np.sqrt(np.mean(np.square(prediction - truth))):.6f}")


if __name__ == "__main__":
    main()
