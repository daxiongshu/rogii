from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_cache(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not NAME_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "cache must be a lowercase NAME=PATH pair"
        )
    return name, Path(raw_path)


def load(path: Path) -> dict[str, np.ndarray]:
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
    parser.add_argument(
        "--cache",
        action="append",
        type=parse_named_cache,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.cache) < 2:
        raise ValueError("at least two named caches are required")
    route_names = [name for name, _ in args.cache]
    if len(route_names) != len(set(route_names)):
        raise ValueError("duplicate route name")
    if args.output.exists():
        raise FileExistsError(args.output)

    loaded = [(name, path, load(path)) for name, path in args.cache]
    reference = loaded[0][2]
    scalar_fields = ("fold", "control", "shard_index", "num_shards")
    array_fields = ("parent_indices", "well_ids", "row_starts")
    for name, path, cache in loaded[1:]:
        for field in scalar_fields:
            if str(reference[field]) != str(cache[field]):
                raise ValueError(
                    f"{name} cache mismatch in {field}: {path}"
                )
        for field in array_fields:
            if not np.array_equal(reference[field], cache[field]):
                raise ValueError(
                    f"{name} cache mismatch in {field}: {path}"
                )
        if str(reference["parent_sha256"]) != str(cache["parent_sha256"]):
            raise ValueError(f"{name} cache parent mismatch: {path}")

    names = np.concatenate(
        [
            np.char.add(
                f"{route_name}__",
                cache["variant_names"].astype(str),
            )
            for route_name, _, cache in loaded
        ]
    )
    prediction = np.concatenate(
        [cache["prediction"].astype(np.float32) for _, _, cache in loaded],
        axis=0,
    )
    cache_records = [
        {
            "route": route_name,
            "path": str(path),
            "sha256": sha256(path),
        }
        for route_name, path, _ in loaded
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        run_id=np.asarray("v5_batch3_run_001_goal_020"),
        family=np.asarray("c016_residual_gr_path_pool"),
        fold=np.int64(reference["fold"]),
        config=np.asarray("named_highcap_structural_routes"),
        control=np.asarray(str(reference["control"])),
        shard_index=np.int64(reference["shard_index"]),
        num_shards=np.int64(reference["num_shards"]),
        parent_indices=reference["parent_indices"].astype(np.int64),
        well_ids=reference["well_ids"].astype(str),
        row_starts=reference["row_starts"].astype(np.int64),
        variant_names=names,
        prediction=prediction,
        parent_sha256=np.asarray(str(reference["parent_sha256"])),
        cache_records_json=np.asarray(
            json.dumps(cache_records, sort_keys=True)
        ),
        source_sha256=np.asarray(sha256(Path(__file__))),
        hidden_truth_loaded=np.bool_(False),
        hidden_metrics_computed=np.bool_(False),
        F4_loaded=np.bool_(False),
        F4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
    )


if __name__ == "__main__":
    main()
