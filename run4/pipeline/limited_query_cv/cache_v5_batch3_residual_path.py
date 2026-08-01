from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from limited_query_cv.v5_batch3_residual_path import (
    SEARCH_CONFIGS,
    residual_path_candidates,
)
from rogii.data import (
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    Well,
)


PARENT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit/sealed/outer_predictions.npz"
)
DATA_ROOT = Path(
    "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_only_training_well(well_id: str, data_root: Path) -> Well:
    """Load the train-layout well without ever reading its hidden TVT column."""
    horizontal_path = data_root / "train" / f"{well_id}__horizontal_well.csv"
    typewell_path = data_root / "train" / f"{well_id}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=list(HORIZONTAL_INFERENCE_COLUMNS),
    )
    typewell = pd.read_csv(
        typewell_path,
        usecols=list(TYPEWELL_INFERENCE_COLUMNS),
    ).sort_values("TVT")
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    visible_only = np.where(np.isfinite(tvt_input), tvt_input, np.nan)
    return Well(
        well_id=well_id,
        md=horizontal["MD"].to_numpy(np.float64),
        x=horizontal["X"].to_numpy(np.float64),
        y=horizontal["Y"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=horizontal["GR"].to_numpy(np.float64),
        tvt=visible_only,
        tvt_input=tvt_input,
        typewell_tvt=typewell["TVT"].to_numpy(np.float64),
        typewell_gr=typewell["GR"].to_numpy(np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--config",
        choices=[config.name for config in SEARCH_CONFIGS],
        required=True,
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
    config = next(
        item for item in SEARCH_CONFIGS if item.name == args.config
    )
    started = time.time()
    with np.load(args.parent, allow_pickle=False) as parent:
        if bool(parent["hidden_truth_stored"]):
            raise ValueError("C016 parent unexpectedly stores hidden truth")
        if bool(parent["f4_metric_computed"]):
            raise ValueError("label-free parent metadata changed")
        all_ids = parent["well_ids"].astype(str)
        all_folds = parent["folds"].astype(np.int64)
        all_starts = parent["row_starts"].astype(np.int64)
        all_prediction = parent["prediction"].astype(np.float32)
    fold_indices = np.flatnonzero(all_folds == args.fold)
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
        if len(well.prediction_indices) != len(baseline):
            raise ValueError(f"{well_id}: parent/tail layout changed")
        observed_shift = (
            len(baseline) // 3 if args.control == "circular_shift" else 0
        )
        prediction, names, diagnostic = residual_path_candidates(
            well,
            baseline,
            config,
            device=args.device,
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
        config=np.asarray(args.config),
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
        source_sha256=np.asarray(sha256(Path(__file__))),
        residual_source_sha256=np.asarray(
            sha256(Path("limited_query_cv/v5_batch3_residual_path.py"))
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
