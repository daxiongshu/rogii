from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from limited_query_cv.v5_batch3_run2_surface import (
    LEGAL_HORIZONTAL_COLUMNS,
    SurfaceWell,
    role_purged_buda_prediction,
    typewell_clusters,
)


DATA_ROOT = Path(
    "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
)
PARENT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit/sealed/outer_predictions.npz"
)

CONFIGS = (
    {
        "name": "global_n10_bw1_d0_r138",
        "cluster_only": False,
        "neighbor_count": 10,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 0,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "global_n16_bw1_d500_r138",
        "cluster_only": False,
        "neighbor_count": 16,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 500,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "global_n24_bw2_d1000_r138",
        "cluster_only": False,
        "neighbor_count": 24,
        "bandwidth_multiplier": 2.0,
        "residual_decay_rows": 1000,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "global_n16_bw1_d0_r0",
        "cluster_only": False,
        "neighbor_count": 16,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 0,
        "azimuth_degrees": 0.0,
    },
    {
        "name": "cluster_n10_bw1_d0_r138",
        "cluster_only": True,
        "neighbor_count": 10,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 0,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "cluster_n16_bw1_d500_r138",
        "cluster_only": True,
        "neighbor_count": 16,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 500,
        "azimuth_degrees": 138.0,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_surface_well(well_id: str, data_root: Path) -> SurfaceWell:
    path = data_root / "train" / f"{well_id}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["X", "Y", "BUDA", "TVT_input"])
    finite = frame["BUDA"].notna().to_numpy()
    known = frame["TVT_input"].notna().to_numpy()
    unknown_index = np.flatnonzero(finite & ~known)
    known_index = np.flatnonzero(finite & known)
    keep = np.sort(
        np.concatenate((unknown_index, known_index[-min(800, len(known_index)) :]))
    )
    return SurfaceWell(
        well_id,
        frame["X"].to_numpy(np.float64)[keep],
        frame["Y"].to_numpy(np.float64)[keep],
        frame["BUDA"].to_numpy(np.float64)[keep],
    )


def load_typewell(well_id: str, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = data_root / "train" / f"{well_id}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"]).dropna()
    return (
        frame["TVT"].to_numpy(np.float64),
        frame["GR"].to_numpy(np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    with np.load(args.parent, allow_pickle=False) as parent:
        if bool(parent["hidden_truth_stored"]):
            raise ValueError("label-free C016 parent changed")
        all_ids = parent["well_ids"].astype(str)
        folds = parent["folds"].astype(np.int64)
        starts = parent["row_starts"].astype(np.int64)
        clean = parent["prediction"].astype(np.float64)
    development = np.flatnonzero(folds < 4)
    held = np.flatnonzero(folds == args.fold)
    training = np.flatnonzero((folds < 4) & (folds != args.fold))
    training_ids = [str(all_ids[index]) for index in training]
    pool = {
        well_id: load_surface_well(well_id, args.data_root)
        for well_id in training_ids
    }
    typewells = {
        str(all_ids[index]): load_typewell(str(all_ids[index]), args.data_root)
        for index in development
    }
    clusters = typewell_clusters(typewells)
    predictions = []
    held_ids = []
    row_starts = [0]
    cluster_pool_sizes = []
    for count, parent_index in enumerate(held, start=1):
        well_id = str(all_ids[parent_index])
        path = (
            args.data_root
            / "train"
            / f"{well_id}__horizontal_well.csv"
        )
        horizontal = pd.read_csv(path, usecols=list(LEGAL_HORIZONTAL_COLUMNS))
        target_cluster = clusters[well_id]
        cluster_pool = {
            other_id: surface_well
            for other_id, surface_well in pool.items()
            if clusters.get(other_id) == target_cluster
        }
        cluster_pool_sizes.append(len(cluster_pool))
        well_prediction = []
        for config in CONFIGS:
            selected_pool = (
                cluster_pool
                if config["cluster_only"] and len(cluster_pool) >= 3
                else pool
            )
            options = {
                key: value
                for key, value in config.items()
                if key not in {"name", "cluster_only"}
            }
            predicted = role_purged_buda_prediction(
                selected_pool,
                horizontal,
                **options,
            )
            expected_rows = int(starts[parent_index + 1] - starts[parent_index])
            if len(predicted) != expected_rows:
                raise ValueError(f"{well_id}: held tail layout changed")
            well_prediction.append(predicted)
        predictions.append(np.stack(well_prediction))
        held_ids.append(well_id)
        row_starts.append(row_starts[-1] + well_prediction[0].shape[0])
        print(
            f"fold={args.fold} well={count}/{len(held)} id={well_id} "
            f"cluster_pool={len(cluster_pool)}",
            flush=True,
        )
    bank = np.concatenate(predictions, axis=1).astype(np.float32)
    held_parent_rows = np.concatenate(
        [
            np.arange(starts[index], starts[index + 1], dtype=np.int64)
            for index in held
        ]
    )
    clean_tail = clean[held_parent_rows].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray("v5_batch3_run_002_goal_020"),
        family=np.asarray("c016_buda_delayed_dip_cluster_surface"),
        fold=np.int64(args.fold),
        development_folds=np.asarray([0, 1, 2, 3], dtype=np.int64),
        role_training_folds=np.asarray(
            [fold for fold in range(4) if fold != args.fold],
            dtype=np.int64,
        ),
        parent_indices=held.astype(np.int64),
        well_ids=np.asarray(held_ids),
        row_starts=np.asarray(row_starts, dtype=np.int64),
        variant_names=np.asarray([config["name"] for config in CONFIGS]),
        prediction=bank,
        clean_C016=clean_tail,
        cluster_pool_sizes=np.asarray(cluster_pool_sizes, dtype=np.int64),
        role_training_wells=np.int64(len(pool)),
        held_well_BUDA_loaded=np.bool_(False),
        held_well_ANCC_loaded=np.bool_(False),
        held_hidden_TVT_loaded=np.bool_(False),
        hidden_truth_loaded=np.bool_(False),
        hidden_metrics_computed=np.bool_(False),
        F4_loaded=np.bool_(False),
        F4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
        parent_sha256=np.asarray(sha256(args.parent)),
        source_sha256=np.asarray(sha256(Path(__file__))),
        implementation_source_sha256=np.asarray(
            sha256(Path("limited_query_cv/v5_batch3_run2_surface.py"))
        ),
        elapsed_seconds=np.float64(time.time() - started),
    )


if __name__ == "__main__":
    main()
