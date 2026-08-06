from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RUN1_ROOT = Path(
    "limited_query_cv/runs/v5_batch3_run_001_goal_020"
)
RUN2_ROOT = Path(
    "limited_query_cv/runs/v5_batch3_run_002_goal_020"
)
PARENT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit/sealed/outer_predictions.npz"
)
V20_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_001_goal_050/v20_components"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent_fold(
    fold: int,
    parent_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(parent_path, allow_pickle=False) as archive:
        if bool(archive["hidden_truth_stored"]):
            raise ValueError("label-free C016 parent changed")
        ids = archive["well_ids"].astype(str)
        folds = archive["folds"].astype(np.int64)
        starts = archive["row_starts"].astype(np.int64)
        prediction = archive["prediction"].astype(np.float64)
    indices = np.flatnonzero(folds == fold)
    clean = np.concatenate(
        [
            prediction[starts[index] : starts[index + 1]]
            for index in indices
        ]
    )
    return ids[indices], indices, starts, clean


def load_run1_candidates(
    fold: int,
    parent_indices: np.ndarray,
    root: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    pieces: dict[int, np.ndarray] = {}
    names = None
    records = []
    for path in sorted(
        (root / f"f{fold}_highcap_structural").glob("shard*.npz")
    ):
        with np.load(path, allow_pickle=False) as cache:
            if (
                int(cache["fold"]) != fold
                or bool(cache["hidden_truth_loaded"])
                or bool(cache["hidden_metrics_computed"])
                or bool(cache["F4_loaded"])
                or bool(cache["F4_metric_computed"])
                or bool(cache["protected_reveal_spent"])
            ):
                raise ValueError(f"{path}: Run 1 cache firewall failed")
            local_names = cache["variant_names"].astype(str)
            indices = cache["parent_indices"].astype(np.int64)
            starts = cache["row_starts"].astype(np.int64)
            prediction = cache["prediction"].astype(np.float64)
        if names is None:
            names = local_names
        elif not np.array_equal(names, local_names):
            raise ValueError("Run 1 candidate schema changed")
        for local_index, parent_index in enumerate(indices):
            pieces[int(parent_index)] = prediction[
                :, starts[local_index] : starts[local_index + 1]
            ]
        records.append({"path": str(path), "sha256": sha256(path)})
    if names is None or set(pieces) != set(map(int, parent_indices)):
        raise ValueError(f"Run 1 shards do not cover fold {fold}")
    candidates = np.concatenate(
        [pieces[int(index)] for index in parent_indices],
        axis=1,
    )
    return names, candidates, records


def load_run2_cache(
    path: Path,
    fold: int,
    expected_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    with np.load(path, allow_pickle=False) as cache:
        cache_fold = int(
            cache["fold"] if "fold" in cache else cache["held_fold"]
        )
        firewall = [
            "held_well_BUDA_loaded",
            "held_well_ANCC_loaded",
            "held_hidden_TVT_loaded",
            "hidden_truth_loaded",
            "hidden_metrics_computed",
            "held_target_loaded",
            "F4_loaded",
            "F4_metric_computed",
            "protected_reveal_spent",
        ]
        if cache_fold != fold or any(
            name in cache and bool(cache[name])
            for name in firewall
        ):
            raise ValueError(f"{path}: Run 2 cache firewall failed")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        names = cache["variant_names"].astype(str)
        prediction = cache["prediction"].astype(np.float64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path}: held-well order changed")
    if prediction.shape[1] != int(starts[-1]):
        raise ValueError(f"{path}: row layout changed")
    return names, prediction, {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--run1-root", type=Path, default=RUN1_ROOT)
    parser.add_argument("--run2-root", type=Path, default=RUN2_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    ids, parent_indices, _, clean = load_parent_fold(
        args.fold,
        args.parent,
    )
    names1, candidates1, records1 = load_run1_candidates(
        args.fold,
        parent_indices,
        args.run1_root,
    )
    raw_path = args.run2_root / f"surface_f{args.fold}_cache.npz"
    selector_path = (
        args.run2_root / f"surface_selector_f{args.fold}_cache.npz"
    )
    raw_names, raw_candidates, raw_record = load_run2_cache(
        raw_path,
        args.fold,
        ids,
    )
    selector_names, selector_candidates, selector_record = load_run2_cache(
        selector_path,
        args.fold,
        ids,
    )
    names = np.concatenate(
        [
            np.char.add("run1_", names1),
            np.char.add("surface_raw_", raw_names),
            np.char.add("surface_selector_", selector_names),
        ]
    )
    candidates = np.concatenate(
        [candidates1, raw_candidates, selector_candidates],
        axis=0,
    )
    if candidates.shape != (len(names), len(clean)):
        raise ValueError("combined candidate layout changed")

    v20_path = V20_ROOT / f"fold{args.fold}.npz"
    with np.load(v20_path, allow_pickle=False) as v20:
        if (
            int(v20["fold"]) != args.fold
            or bool(v20["audit_fold_loaded"])
            or bool(v20["layout_truth_array_accessed"])
            or not np.array_equal(v20["well_ids"].astype(str), ids)
        ):
            raise ValueError("V20 fold firewall failed")
        truth = v20["truth"].astype(np.float64)
        v20_prediction = v20["baseline"].astype(np.float64)
    correction = candidates - clean[None]
    error = clean - truth
    result = {
        "schema_version": np.int64(1),
        "protocol_revision": np.int64(11),
        "run_id": np.asarray("v5_batch3_run_002_goal_020"),
        "family": np.asarray(
            "c016_buda_delayed_dip_cluster_surface"
        ),
        "stage": np.asarray(
            "run1_plus_role_purged_surface_stack_stats"
        ),
        "fold": np.int64(args.fold),
        "rows": np.int64(len(truth)),
        "wells": np.int64(len(ids)),
        "candidate_names": names,
        "clean_sse": np.float64(np.sum(np.square(error))),
        "v20_sse": np.float64(
            np.sum(np.square(v20_prediction - truth))
        ),
        "cross": (correction @ error).astype(np.float64),
        "gram": (correction @ correction.T).astype(np.float64),
        "parent_sha256": np.asarray(sha256(args.parent)),
        "v20_sha256": np.asarray(sha256(v20_path)),
        "cache_records_json": np.asarray(
            json.dumps(
                records1 + [raw_record, selector_record],
                sort_keys=True,
            )
        ),
        "source_sha256": np.asarray(sha256(Path(__file__))),
        "raw_hidden_target_stored": np.bool_(False),
        "F4_loaded": np.bool_(False),
        "F4_metric_computed": np.bool_(False),
        "protected_reveal_spent": np.bool_(False),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(
        f"fold={args.fold} rows={len(truth)} "
        f"candidates={len(names)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
