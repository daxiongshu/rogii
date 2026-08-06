from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structural_bank(
    prediction: np.ndarray,
    names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    names = names.astype(str)
    groups = [
        ("struct_uniform_mean", np.ones(len(names), dtype=bool), "mean"),
        ("struct_uniform_median", np.ones(len(names), dtype=bool), "median"),
        ("struct_m64_mean", np.char.find(names, "_m64_") >= 0, "mean"),
        ("struct_m128_mean", np.char.find(names, "_m128_") >= 0, "mean"),
        ("struct_tau10_mean", np.char.find(names, "_tau10_") >= 0, "mean"),
        ("struct_tau30_mean", np.char.find(names, "_tau30_") >= 0, "mean"),
        ("struct_sb4_mean", np.char.endswith(names, "_sb4"), "mean"),
        ("struct_sb8_mean", np.char.endswith(names, "_sb8"), "mean"),
        ("struct_sb12_mean", np.char.endswith(names, "_sb12"), "mean"),
        (
            "struct_m128_tau30_mean",
            (np.char.find(names, "_m128_") >= 0)
            & (np.char.find(names, "_tau30_") >= 0),
            "mean",
        ),
        (
            "struct_m128_sb12_mean",
            (np.char.find(names, "_m128_") >= 0)
            & np.char.endswith(names, "_sb12"),
            "mean",
        ),
        (
            "struct_tau30_sb12_mean",
            (np.char.find(names, "_tau30_") >= 0)
            & np.char.endswith(names, "_sb12"),
            "mean",
        ),
    ]
    output = []
    output_names = []
    for name, mask, reduction in groups:
        if not mask.any():
            raise ValueError(f"empty structural group {name}")
        values = prediction[mask].astype(np.float64)
        if reduction == "median":
            reduced = np.median(values, axis=0)
        else:
            reduced = np.mean(values, axis=0)
        output.append(reduced.astype(np.float32))
        output_names.append(name)
    return np.stack(output), np.asarray(output_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as cache:
        if (
            bool(cache["hidden_truth_loaded"])
            or bool(cache["hidden_metrics_computed"])
            or bool(cache["F4_loaded"])
            or bool(cache["F4_metric_computed"])
            or bool(cache["protected_reveal_spent"])
        ):
            raise ValueError("input cache firewall failed")
        fold = int(cache["fold"])
        control = str(cache["control"])
        shard_index = int(cache["shard_index"])
        num_shards = int(cache["num_shards"])
        parent_indices = cache["parent_indices"].astype(np.int64)
        well_ids = cache["well_ids"].astype(str)
        row_starts = cache["row_starts"].astype(np.int64)
        names = cache["variant_names"].astype(str)
        prediction = cache["prediction"].astype(np.float32)
        parent_sha256 = str(cache["parent_sha256"])
    transformed, transformed_names = structural_bank(prediction, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        run_id=np.asarray("v5_batch3_run_001_goal_020"),
        family=np.asarray("c016_residual_gr_path_pool"),
        fold=np.int64(fold),
        config=np.asarray("self_prefix_highcap_structural"),
        control=np.asarray(control),
        shard_index=np.int64(shard_index),
        num_shards=np.int64(num_shards),
        parent_indices=parent_indices,
        well_ids=well_ids,
        row_starts=row_starts,
        variant_names=transformed_names,
        prediction=transformed,
        parent_sha256=np.asarray(parent_sha256),
        input_cache_sha256=np.asarray(sha256(args.input)),
        source_sha256=np.asarray(sha256(Path(__file__))),
        hidden_truth_loaded=np.bool_(False),
        hidden_metrics_computed=np.bool_(False),
        F4_loaded=np.bool_(False),
        F4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
    )


if __name__ == "__main__":
    main()
