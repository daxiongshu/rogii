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


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as cache:
        if (
            bool(cache["hidden_truth_loaded"])
            or bool(cache["hidden_metrics_computed"])
            or bool(cache["F4_loaded"])
            or bool(cache["F4_metric_computed"])
            or bool(cache["protected_reveal_spent"])
        ):
            raise ValueError(f"{path}: structural cache firewall failed")
        return {name: cache[name] for name in cache.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-cache", type=Path, required=True)
    parser.add_argument("--datum-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    self_cache = load(args.self_cache)
    datum_cache = load(args.datum_cache)
    scalar_fields = ("fold", "control", "shard_index", "num_shards")
    for field in scalar_fields:
        if str(self_cache[field]) != str(datum_cache[field]):
            raise ValueError(f"cache mismatch: {field}")
    array_fields = ("parent_indices", "well_ids", "row_starts")
    for field in array_fields:
        if not np.array_equal(self_cache[field], datum_cache[field]):
            raise ValueError(f"cache mismatch: {field}")
    names = np.r_[
        np.char.add(
            "self__",
            self_cache["variant_names"].astype(str),
        ),
        np.char.add(
            "datum__",
            datum_cache["variant_names"].astype(str),
        ),
    ]
    prediction = np.concatenate(
        [
            self_cache["prediction"].astype(np.float32),
            datum_cache["prediction"].astype(np.float32),
        ],
        axis=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        run_id=np.asarray("v5_batch3_run_001_goal_020"),
        family=np.asarray("c016_residual_gr_path_pool"),
        fold=np.int64(self_cache["fold"]),
        config=np.asarray("self_and_datum_highcap_structural"),
        control=np.asarray(str(self_cache["control"])),
        shard_index=np.int64(self_cache["shard_index"]),
        num_shards=np.int64(self_cache["num_shards"]),
        parent_indices=self_cache["parent_indices"].astype(np.int64),
        well_ids=self_cache["well_ids"].astype(str),
        row_starts=self_cache["row_starts"].astype(np.int64),
        variant_names=names,
        prediction=prediction,
        parent_sha256=np.asarray(str(self_cache["parent_sha256"])),
        self_cache_sha256=np.asarray(sha256(args.self_cache)),
        datum_cache_sha256=np.asarray(sha256(args.datum_cache)),
        source_sha256=np.asarray(sha256(Path(__file__))),
        hidden_truth_loaded=np.bool_(False),
        hidden_metrics_computed=np.bool_(False),
        F4_loaded=np.bool_(False),
        F4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
    )


if __name__ == "__main__":
    main()
