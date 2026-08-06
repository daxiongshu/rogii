from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from limited_query_cv.train_v5_batch3_run2_surface_selector import (
    DATA_ROOT,
    PARENT,
    all_development_clusters,
    load_fold,
)
from limited_query_cv.v5_batch3_run4_clean_bank import RUN_ID, RUN_ROOT


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
SURFACE_INDICES = np.arange(12, 18, dtype=np.int64)
NONRAMPED_INDICES = np.arange(39, dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--well-stats", type=Path, default=WELL_STATS)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "router_features.npz",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Read only label-free fields from the sufficient-statistics archive.
    with np.load(args.well_stats, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise ValueError("Run 4 well-stat firewall changed")
        well_ids = stats["well_ids"].astype(str)
        folds = stats["original_folds"].astype(np.int64)
        rows = stats["rows"].astype(np.float64)
        gram = stats["gram"].astype(np.float64)
        candidate_names = stats["candidate_names"].astype(str)
    clusters = all_development_clusters(args.parent, args.data_root)
    selector_by_id: dict[str, np.ndarray] = {}
    category_by_id: dict[str, str] = {}
    for fold in range(4):
        record = load_fold(
            fold,
            clusters,
            args.data_root,
            truth_allowed=False,
        )
        if (
            record["truth"] is not None
            or len(record["target_weight"])
            or len(record["cross"])
            or len(record["energy"])
        ):
            raise ValueError("selector feature loader accessed targets")
        for well_id, numeric, category in zip(
            record["ids"].astype(str),
            record["numeric"],
            record["category"].astype(str),
        ):
            selector_by_id[str(well_id)] = numeric.astype(np.float64)
            category_by_id[str(well_id)] = str(category)
    if set(selector_by_id) != set(well_ids):
        raise ValueError("router selector features do not cover F0-F3")
    selector = np.stack([selector_by_id[well_id] for well_id in well_ids])
    category = np.asarray([category_by_id[well_id] for well_id in well_ids])

    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    rms = np.sqrt(np.maximum(diagonal, 0.0) / rows[:, None])
    left_scale = np.sqrt(
        np.maximum(diagonal[:, NONRAMPED_INDICES], 1e-12)
    )
    right_scale = np.sqrt(
        np.maximum(diagonal[:, SURFACE_INDICES], 1e-12)
    )
    cosine = gram[
        :,
        NONRAMPED_INDICES[:, None],
        SURFACE_INDICES[None, :],
    ] / (left_scale[:, :, None] * right_scale[:, None, :])
    cosine = np.clip(cosine, -1.0, 1.0)
    disagreement = np.std(
        cosine[:, :12, :], axis=1
    )
    features = np.concatenate(
        (
            selector,
            np.log1p(rms),
            cosine.reshape(len(well_ids), -1),
            disagreement,
        ),
        axis=1,
    ).astype(np.float32)
    feature_names = [
        *(f"selector_numeric_{index}" for index in range(selector.shape[1])),
        *(
            f"log_rms_{name}"
            for name in candidate_names
        ),
        *(
            f"cosine_{candidate_names[left]}__{candidate_names[right]}"
            for left in NONRAMPED_INDICES
            for right in SURFACE_INDICES
        ),
        *(
            f"structural_cosine_std__{candidate_names[index]}"
            for index in SURFACE_INDICES
        ),
    ]
    if (
        features.shape != (619, len(feature_names))
        or not np.isfinite(features).all()
        or set(folds.tolist()) != set(range(4))
    ):
        raise ValueError("invalid Run 4 router feature matrix")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        well_ids=well_ids,
        original_folds=folds.astype(np.int8),
        features=features,
        feature_names=np.asarray(feature_names),
        category=category,
        well_stats_sha256=np.asarray(sha256(args.well_stats)),
        source_surface_indices=SURFACE_INDICES,
        target_fields_read=np.asarray(False),
        raw_hidden_target_stored=np.asarray(False),
        F4_loaded=np.asarray(False),
        F4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
    )
    print(
        f"wrote={args.output} sha256={sha256(args.output)} "
        f"shape={features.shape}"
    )


if __name__ == "__main__":
    main()
