from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


def load_all(data_root: Path, workers: int) -> list[Well]:
    ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(well_id, data_root), ids))


@torch.no_grad()
def sampled_predictions(
    model: AlignmentUNet,
    well: Well,
    config: SequenceConfig,
    offsets: torch.Tensor,
    device: torch.device,
    temperatures: tuple[float, ...],
) -> dict[float, np.ndarray]:
    image, _, metadata = make_alignment_example(well, well.anchor_index, config)
    logits = model(torch.from_numpy(image).unsqueeze(0).to(device))[
        0, :, config.prefix_points :
    ]
    valid = np.asarray(metadata["valid_positions"])[config.prefix_points :] > 0.5
    logits = logits[:, valid]
    positions = np.asarray(metadata["positions"])[config.prefix_points :][valid]
    tail = well.tail_indices
    predictions = {}
    for temperature in temperatures:
        probability = torch.softmax(logits / temperature, dim=0)
        sampled_offset = (
            torch.sum(probability * offsets[:, None], dim=0).float().cpu().numpy()
        )
        predictions[temperature] = np.interp(
            tail, positions, float(metadata["anchor_tvt"]) + sampled_offset
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--narrow", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--wide-state-radius", type=float, default=120.0)
    parser.add_argument("--wide-state-step", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wells = load_all(args.data_root, args.workers)
    identifiers = np.asarray([well.well_id for well in wells])
    _, valid_indices = list(GroupKFold(5).split(identifiers, groups=identifiers))[
        args.fold
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    narrow_config = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=64.0,
        state_step=0.5,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
    )
    wide_config = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=args.wide_state_radius,
        state_step=args.wide_state_step,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
    )
    narrow_offsets = torch.as_tensor(
        narrow_config.offsets, dtype=torch.float32, device=device
    )
    wide_offsets = torch.as_tensor(
        wide_config.offsets, dtype=torch.float32, device=device
    )
    narrow_model = AlignmentUNet(base=args.base).to(device)
    narrow_model.load_state_dict(
        torch.load(args.narrow, map_location=device, weights_only=True)
    )
    narrow_model.eval()
    wide_model = AlignmentUNet(base=args.base).to(device)
    wide_model.load_state_dict(
        torch.load(args.wide, map_location=device, weights_only=True)
    )
    wide_model.eval()
    wide_temperatures = (1.0, 1.5, 2.0, 2.5, 3.0)
    weights = np.linspace(0.0, 1.0, 21)

    records = []
    for index in valid_indices:
        well = wells[index]
        truth = well.tvt[well.tail_indices]
        narrow = sampled_predictions(
            narrow_model,
            well,
            narrow_config,
            narrow_offsets,
            device,
            (1.5,),
        )[1.5]
        wide_predictions = sampled_predictions(
            wide_model,
            well,
            wide_config,
            wide_offsets,
            device,
            wide_temperatures,
        )
        record: dict[str, str | int | float] = {
            "fold": args.fold,
            "well_id": well.well_id,
            "rows": len(truth),
        }
        for temperature, wide in wide_predictions.items():
            for weight in weights:
                prediction = (1.0 - weight) * narrow + weight * wide
                error = prediction - truth
                name = f"sse_t{int(temperature * 100):03d}_w{int(weight * 100):03d}"
                record[name] = float(error @ error)
        records.append(record)

    frame = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    candidates = [column for column in frame if column.startswith("sse_")]
    scores = {
        column: np.sqrt(frame[column].sum() / frame["rows"].sum())
        for column in candidates
    }
    for name, score in sorted(scores.items(), key=lambda item: item[1])[:10]:
        print(f"fold={args.fold} {name} rmse={score:.6f}")
    print(f"output={args.output} wells={len(frame)} rows={int(frame['rows'].sum())}")


if __name__ == "__main__":
    main()
