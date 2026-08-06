from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from limited_query_cv.cache_v5_batch3_residual_path import (
    DATA_ROOT,
    PARENT,
    inference_only_training_well,
)
from limited_query_cv.v5_batch3_datum_pool import (
    PoolMap,
    datum_pool_candidates,
)
from limited_query_cv.v5_batch3_highcap_datum import (
    highcap_datum_candidates,
)
from limited_query_cv.v5_batch3_highcap_path import HIGHCAP_CONFIG
from limited_query_cv.v5_batch3_residual_path import SEARCH_CONFIGS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--config",
        choices=[
            config.name for config in SEARCH_CONFIGS
        ] + [HIGHCAP_CONFIG.name],
        required=True,
    )
    parser.add_argument("--pool-role", type=Path, required=True)
    parser.add_argument(
        "--pool-weight",
        type=float,
        choices=(0.25, 0.5, 0.75, 1.0),
        required=True,
    )
    parser.add_argument(
        "--z-lam",
        type=float,
        choices=(0.0, 1.0, 2.0),
        default=0.0,
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--control",
        choices=("aligned", "circular_shift"),
        default="aligned",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    config = (
        HIGHCAP_CONFIG
        if args.config == HIGHCAP_CONFIG.name
        else next(item for item in SEARCH_CONFIGS if item.name == args.config)
    )
    started = time.time()
    with np.load(args.parent, allow_pickle=False) as parent:
        if bool(parent["hidden_truth_stored"]):
            raise ValueError("C016 parent unexpectedly stores hidden truth")
        all_ids = parent["well_ids"].astype(str)
        all_folds = parent["folds"].astype(np.int64)
        all_starts = parent["row_starts"].astype(np.int64)
        all_prediction = parent["prediction"].astype(np.float32)
    with np.load(args.pool_role, allow_pickle=False) as role:
        if (
            int(role["held_fold"]) != args.fold
            or bool(role["held_fold_hidden_target_loaded"])
            or bool(role["sealed_fold_hidden_target_loaded"])
            or bool(role["held_fold_metric_computed"])
            or bool(role["F4_metric_computed"])
            or bool(role["protected_reveal_spent"])
        ):
            raise ValueError("datum-pool role firewall failed")
        role_ids = role["well_ids"].astype(str)
        role_starts = role["row_starts"].astype(np.int64)
        role_grid = role["grid"].astype(np.float32)
        role_mean = role["pool_mean"].astype(np.float32)
        role_coverage = role["coverage"].astype(np.float32)
        role_wells = role["contributing_wells"].astype(np.int64)
        role_points = role["contributing_points"].astype(np.int64)
    role_index = {well_id: index for index, well_id in enumerate(role_ids)}
    fold_indices = np.flatnonzero(all_folds == args.fold)
    if not np.array_equal(all_ids[fold_indices], role_ids):
        raise ValueError("datum role and parent fold layout changed")
    parent_indices = fold_indices[
        np.arange(len(fold_indices)) % args.num_shards == args.shard_index
    ]
    predictions = []
    local_starts = [0]
    selected_ids = []
    variant_names: list[str] | None = None
    diagnostics = []
    for count, parent_index in enumerate(parent_indices, start=1):
        well_id = str(all_ids[parent_index])
        row_start = int(all_starts[parent_index])
        row_stop = int(all_starts[parent_index + 1])
        baseline = all_prediction[row_start:row_stop]
        well = inference_only_training_well(well_id, args.data_root)
        index = role_index[well_id]
        pool_start, pool_stop = role_starts[index : index + 2]
        pool = PoolMap(
            grid=role_grid[pool_start:pool_stop],
            mean=role_mean[pool_start:pool_stop],
            coverage=role_coverage[pool_start:pool_stop],
            contributing_wells=int(role_wells[index]),
            contributing_points=int(role_points[index]),
        )
        observed_shift = (
            len(baseline) // 3 if args.control == "circular_shift" else 0
        )
        if config.name == HIGHCAP_CONFIG.name:
            if args.z_lam:
                raise ValueError("high-capacity datum screen has no Z reward")
            prediction, names, diagnostic = highcap_datum_candidates(
                well,
                baseline,
                config,
                pool,
                pool_weight=args.pool_weight,
                device=args.device,
                observed_shift=observed_shift,
            )
        else:
            prediction, names, diagnostic = datum_pool_candidates(
                well,
                baseline,
                config,
                pool,
                pool_weight=args.pool_weight,
                device=args.device,
                z_lam=args.z_lam,
                observed_shift=observed_shift,
            )
        if variant_names is None:
            variant_names = names
        elif variant_names != names:
            raise ValueError("variant names changed between wells")
        predictions.append(prediction)
        local_starts.append(local_starts[-1] + len(baseline))
        selected_ids.append(well_id)
        diagnostics.append(diagnostic)
        print(
            f"fold={args.fold} config={args.config} "
            f"pw={args.pool_weight:g} zl={args.z_lam:g} "
            f"control={args.control} shard={args.shard_index}/"
            f"{args.num_shards} well={count}/{len(parent_indices)} "
            f"id={well_id}",
            flush=True,
        )
    if not predictions:
        raise ValueError("empty shard")
    bank = np.concatenate(predictions, axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        run_id=np.asarray("v5_batch3_run_001_goal_020"),
        family=np.asarray("c016_residual_gr_path_pool"),
        fold=np.int64(args.fold),
        config=np.asarray(
            f"datum_pw{args.pool_weight:g}_zl{args.z_lam:g}_"
            f"{args.config}"
        ),
        control=np.asarray(args.control),
        shard_index=np.int64(args.shard_index),
        num_shards=np.int64(args.num_shards),
        parent_indices=parent_indices.astype(np.int64),
        well_ids=np.asarray(selected_ids),
        row_starts=np.asarray(local_starts, dtype=np.int64),
        variant_names=np.asarray(variant_names),
        prediction=bank,
        diagnostic_json=np.asarray(
            [json.dumps(item, sort_keys=True) for item in diagnostics]
        ),
        parent_sha256=np.asarray(sha256(args.parent)),
        pool_role_sha256=np.asarray(sha256(args.pool_role)),
        source_sha256=np.asarray(sha256(Path(__file__))),
        datum_source_sha256=np.asarray(
            sha256(Path("limited_query_cv/v5_batch3_datum_pool.py"))
        ),
        highcap_datum_source_sha256=np.asarray(
            sha256(Path("limited_query_cv/v5_batch3_highcap_datum.py"))
        ),
        hidden_truth_loaded=np.bool_(False),
        hidden_metrics_computed=np.bool_(False),
        F4_loaded=np.bool_(False),
        F4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
        elapsed_seconds=np.float64(time.time() - started),
    )


if __name__ == "__main__":
    main()
