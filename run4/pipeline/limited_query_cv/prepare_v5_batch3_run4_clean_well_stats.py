from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from limited_query_cv.v5_batch3_run4_clean_bank import (
    RUN_ID,
    RUN_ROOT,
    load_clean_fold_bank,
    verify_run4_lineage,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "clean_well_stats.npz",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    lineage = verify_run4_lineage()
    names = None
    well_ids: list[str] = []
    original_folds: list[int] = []
    rows: list[int] = []
    clean_sse: list[float] = []
    v20_sse: list[float] = []
    crosses: list[np.ndarray] = []
    grams: list[np.ndarray] = []
    source_files: dict[str, str] = {}
    for fold in range(4):
        bank = load_clean_fold_bank(fold, truth_allowed=True)
        if names is None:
            names = bank["names"].astype(str)
        elif not np.array_equal(names, bank["names"].astype(str)):
            raise ValueError("candidate schema changed between folds")
        truth = bank["truth"]
        if truth is None:
            raise ValueError(f"fold {fold}: development truth unavailable")
        for record in bank["records"]:
            source_files[record["path"]] = record["sha256"]
        for index, well_id in enumerate(bank["ids"].astype(str)):
            left, right = bank["starts"][index : index + 2]
            correction = bank["correction"][:, left:right]
            error = bank["clean"][left:right] - truth[left:right]
            v20_error = bank["baseline"][left:right] - truth[left:right]
            well_ids.append(str(well_id))
            original_folds.append(fold)
            rows.append(int(right - left))
            clean_sse.append(float(error @ error))
            v20_sse.append(float(v20_error @ v20_error))
            crosses.append(correction @ error)
            grams.append(correction @ correction.T)
    if (
        names is None
        or len(well_ids) != 619
        or len(set(well_ids)) != 619
        or set(original_folds) != set(range(4))
        or any(row <= 0 for row in rows)
    ):
        raise ValueError("Run 4 development-well layout changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        well_ids=np.asarray(well_ids),
        original_folds=np.asarray(original_folds, dtype=np.int8),
        rows=np.asarray(rows, dtype=np.int64),
        clean_sse=np.asarray(clean_sse, dtype=np.float64),
        v20_sse=np.asarray(v20_sse, dtype=np.float64),
        cross=np.stack(crosses).astype(np.float64),
        gram=np.stack(grams).astype(np.float64),
        candidate_names=names,
        source_paths=np.asarray(sorted(source_files)),
        source_sha256=np.asarray(
            [source_files[path] for path in sorted(source_files)]
        ),
        clean_primary_paths=np.asarray(
            [record["path"] for record in lineage]
        ),
        clean_primary_sha256=np.asarray(
            [record["sha256"] for record in lineage]
        ),
        raw_hidden_target_stored=np.asarray(False),
        F4_loaded=np.asarray(False),
        F4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
    )
    print(
        f"wrote={args.output} sha256={sha256(args.output)} "
        f"wells={len(well_ids)} candidates={len(names)}"
    )


if __name__ == "__main__":
    main()
