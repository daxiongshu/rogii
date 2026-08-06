from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from create_stratified_folds import assignment_score, well_statistics
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well


RUN_ID = "v5_batch3_run_004_goal_020"
RUN_ROOT = Path("limited_query_cv/runs") / RUN_ID
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
SEED_START = 20276000
SEED_CANDIDATES = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_wells(
    identifiers: np.ndarray,
    data_root: Path,
    workers: int,
) -> list[Well]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = list(
            pool.map(
                lambda well_id: load_well(str(well_id), data_root),
                identifiers,
            )
        )
    if len(wells) != len(identifiers):
        raise RuntimeError("development well load is incomplete")
    return wells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "promotion_confirmation_assignment.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    with np.load(LAYOUT, allow_pickle=False) as layout:
        all_ids = layout["well_ids"].astype(str)
        original = layout["original"].astype(np.int64)
    development = original != 4
    identifiers = all_ids[development]
    if (
        len(all_ids) != 773
        or len(identifiers) != 619
        or np.any(original[development] == 4)
    ):
        raise RuntimeError("protected development-well boundary changed")

    wells = load_development_wells(
        identifiers,
        args.data_root,
        args.workers,
    )
    statistics = well_statistics(wells)
    excursion = statistics[:, 1]
    endpoint = statistics[:, 2]
    edges = np.quantile(excursion, np.linspace(0.0, 1.0, 11))
    excursion_bin = np.searchsorted(edges[1:-1], excursion)
    strata = excursion_bin * 2 + (endpoint >= 0)

    best_score = float("inf")
    best_seed = -1
    best_assignment = None
    for seed in range(SEED_START, SEED_START + SEED_CANDIDATES):
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        assignment = np.empty(len(identifiers), dtype=np.int64)
        for fold, (_, valid) in enumerate(
            splitter.split(identifiers, strata)
        ):
            assignment[valid] = fold
        balance_score = assignment_score(statistics, assignment)
        if balance_score < best_score:
            best_score = balance_score
            best_seed = seed
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("fresh confirmation assignment was not generated")

    rows, magnitude, final_offset, outside = statistics.T
    group_summary = []
    for group in range(5):
        selected = best_assignment == group
        group_summary.append(
            {
                "group": group,
                "wells": int(np.sum(selected)),
                "hidden_rows": int(np.sum(rows[selected])),
                "mean_maximum_excursion": float(
                    np.mean(magnitude[selected])
                ),
                "mean_final_offset": float(
                    np.mean(final_offset[selected])
                ),
                "wells_with_outside_rows": int(
                    np.count_nonzero(outside[selected])
                ),
                "outside_rows": int(np.sum(outside[selected])),
            }
        )

    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "stage": "promotion_fresh_confirmation_assignment",
        "status": "generated_metrics_unopened",
        "strategy": (
            "excursion-sign stratification with rare-risk balance search "
            "restricted to original F0-F3 development wells"
        ),
        "folds": 5,
        "seed_search": {
            "seed_start": SEED_START,
            "candidates": SEED_CANDIDATES,
            "selected_seed": best_seed,
            "selection_uses_candidate_or_model_scores": False,
            "balance_score": best_score,
        },
        "firewall": {
            "source_outer_folds": [0, 1, 2, 3],
            "development_wells": len(identifiers),
            "F4_wells_loaded": False,
            "F4_target_loaded": False,
            "candidate_metrics_opened": False,
            "protected_reveal_spent": False,
        },
        "layout": str(LAYOUT),
        "layout_sha256": sha256(LAYOUT),
        "group_summary": group_summary,
        "assignments": {
            str(well_id): int(group)
            for well_id, group in zip(identifiers, best_assignment)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"selected_seed={best_seed} balance_score={best_score:.9f} "
        f"development_wells={len(identifiers)}"
    )
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
