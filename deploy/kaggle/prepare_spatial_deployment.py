from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from rogii.data import load_well


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package spatial reference arrays and confidence models."
    )
    parser.add_argument("--spatial-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1020)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.spatial_meta) as cache:
        well_ids = cache["ids"].astype(str)
        family = cache["family"].astype(np.int64)
        folds = cache["fold"].astype(np.int64)
        features = cache["X"].astype(np.float64)
        target = np.clip(cache["target"].astype(np.float64), -2.0, 4.0)
        energy = cache["weight"].astype(np.float64)

    design = np.column_stack((features, family.astype(str)))
    sample_weight = np.minimum(energy, np.quantile(energy, 0.95))
    for held_out in range(5):
        train = folds != held_out
        model = CatBoostRegressor(
            iterations=500,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=100.0,
            loss_function="RMSE",
            random_seed=args.seed + held_out,
            allow_writing_files=False,
            thread_count=16,
            verbose=False,
        )
        model.fit(
            design[train],
            target[train],
            sample_weight=sample_weight[train],
            cat_features=[features.shape[1]],
        )
        model.save_model(args.output / f"spatial_confidence_fold{held_out}.cbm")

    grid = np.arange(9200.0, 13001.0, 1.0)
    typewell_values = np.zeros((len(well_ids), len(grid)), dtype=np.float32)
    typewell_mask = np.zeros((len(well_ids), len(grid)), dtype=np.uint8)
    reference_xy: list[np.ndarray] = []
    reference_surface: list[np.ndarray] = []
    reference_starts = [0]
    for index, well_id in enumerate(well_ids):
        well = load_well(well_id)
        supported = (grid >= np.nanmin(well.typewell_tvt)) & (
            grid <= np.nanmax(well.typewell_tvt)
        )
        typewell_values[index, supported] = np.interp(
            grid[supported], well.typewell_tvt, well.typewell_gr
        ).astype(np.float32)
        typewell_mask[index, supported] = 1
        sampled = np.unique(
            np.r_[np.arange(0, len(well.md), 16), len(well.md) - 1, well.anchor_index]
        )
        sampled.sort()
        reference_xy.append(np.column_stack((well.x[sampled], well.y[sampled])))
        reference_surface.append(well.tvt[sampled] + well.z[sampled])
        reference_starts.append(reference_starts[-1] + len(sampled))
        if (index + 1) % 100 == 0:
            print(f"references={index + 1}/{len(well_ids)}", flush=True)

    np.savez_compressed(
        args.output / "spatial_reference.npz",
        well_ids=well_ids,
        family=family,
        folds=folds,
        grid=grid,
        typewell_values=typewell_values,
        typewell_mask=typewell_mask,
        reference_starts=np.asarray(reference_starts, dtype=np.int64),
        reference_xy=np.concatenate(reference_xy),
        reference_surface=np.concatenate(reference_surface),
    )
    print(
        f"wrote={args.output} wells={len(well_ids)} "
        f"reference_points={reference_starts[-1]}"
    )


if __name__ == "__main__":
    main()
