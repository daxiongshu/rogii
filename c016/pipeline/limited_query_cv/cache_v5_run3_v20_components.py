from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from cache_alignment_oof_predictions import COMPONENTS, indexed_prediction
from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
V20 = Path("/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz")
CHECKPOINT_ROOT = Path("/geo/geo_new/rogii_v20_checkpoint_vault")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected(
    identifiers: np.ndarray,
    selected: np.ndarray,
    data_root: Path,
    workers: int,
) -> list[Well]:
    selected_ids = identifiers[selected].tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda well_id: load_well(well_id, data_root), selected_ids)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate protected-v20 component paths for one F0-F3 "
            "development fold without loading any F4 well or target."
        )
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    # Layout supplies only row boundaries and fold identity. Its truth array is
    # deliberately not accessed.
    with np.load(LAYOUT, allow_pickle=False) as layout:
        layout_ids = layout["well_ids"].astype(str)
        layout_starts = layout["row_starts"].astype(np.int64)
        layout_folds = layout["original"].astype(np.int64)
    available_ids = set(training_well_ids(args.data_root))
    if available_ids != set(layout_ids):
        raise ValueError("training IDs and protected layout membership differ")
    selected = np.flatnonzero(layout_folds == args.fold)
    wells = load_selected(
        layout_ids,
        selected,
        args.data_root,
        args.workers,
    )
    with np.load(V20, allow_pickle=False) as cached:
        if not np.array_equal(layout_ids, cached["well_ids"].astype(str)):
            raise ValueError("layout and v20 well order differ")
        if not np.array_equal(
            layout_starts, cached["row_starts"].astype(np.int64)
        ):
            raise ValueError("layout and v20 row boundaries differ")
        protected_prediction = cached["prediction"].astype(np.float64)

    device = torch.device(args.device)
    narrow = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=64.0,
        state_step=0.5,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=51,
    )
    wide = SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=120.0,
        state_step=1.0,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=51,
    )
    configs = {"narrow": narrow, "wide": wide}
    models = []
    checkpoint_hashes = {}
    for name, pattern, scale, temperature, weight in COMPONENTS:
        checkpoint = CHECKPOINT_ROOT / pattern.format(fold=args.fold)
        model = AlignmentUNet(base=8).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        checkpoint_hashes[name] = sha256(checkpoint)
        models.append((name, model, scale, temperature, weight))

    well_ids: list[str] = []
    row_starts = [0]
    component_predictions = []
    baselines = []
    truths = []
    with torch.no_grad():
        for count, (global_index, well) in enumerate(
            zip(selected, wells), start=1
        ):
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
            truth = well.tvt[well.tail_indices].astype(np.float32)
            left, right = layout_starts[global_index : global_index + 2]
            if len(truth) != right - left:
                raise ValueError(f"{well.well_id}: layout tail length mismatch")
            component_predictions.append(
                np.stack(predictions).astype(np.float32)
            )
            baselines.append(
                protected_prediction[left:right].astype(np.float32)
            )
            truths.append(truth)
            well_ids.append(well.well_id)
            row_starts.append(row_starts[-1] + len(truth))
            if count % 20 == 0 or count == len(wells):
                print(
                    f"fold={args.fold} wells={count}/{len(wells)} "
                    f"rows={row_starts[-1]} elapsed={time.time() - started:.1f}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fold=np.int64(args.fold),
        well_ids=np.asarray(well_ids),
        row_starts=np.asarray(row_starts, dtype=np.int64),
        components=np.concatenate(component_predictions, axis=1),
        baseline=np.concatenate(baselines),
        truth=np.concatenate(truths),
        component_names=np.asarray([item[0] for item in COMPONENTS]),
        component_weights=np.asarray(
            [item[4] for item in COMPONENTS], dtype=np.float64
        ),
        checkpoint_names=np.asarray(
            [item[1].format(fold=args.fold) for item in COMPONENTS]
        ),
        checkpoint_sha256=np.asarray(
            [checkpoint_hashes[item[0]] for item in COMPONENTS]
        ),
        audit_fold_loaded=np.bool_(False),
        layout_truth_array_accessed=np.bool_(False),
    )
    print(
        f"wrote={args.output} components={len(COMPONENTS)} "
        f"rows={row_starts[-1]} audit_fold_loaded=false "
        f"elapsed={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
