from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


def load_all(data_root: Path, workers: int) -> list[Well]:
    identifiers = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda well_id: load_well(well_id, data_root), identifiers))


def well_statistics(wells: list[Well]) -> np.ndarray:
    values = []
    for well in wells:
        offset = well.tvt[well.tail_indices] - well.tvt[well.anchor_index]
        values.append(
            (
                len(offset),
                float(np.max(np.abs(offset))),
                float(offset[-1]),
                int(np.count_nonzero(np.abs(offset) > 64.0)),
            )
        )
    return np.asarray(values, dtype=np.float64)


def assignment_score(statistics: np.ndarray, assignments: np.ndarray) -> float:
    rows, excursion, endpoint, outside = statistics.T
    top_five = excursion >= np.quantile(excursion, 0.95)
    metrics = np.column_stack(
        (
            np.ones(len(rows)),
            rows,
            rows * excursion,
            rows * np.square(excursion),
            outside > 0,
            outside,
            endpoint > 25,
            endpoint < -25,
            rows * (endpoint > 25),
            rows * (endpoint < -25),
            top_five,
            rows * top_five,
        )
    )
    target = np.sum(metrics, axis=0) / 5.0
    valid = target > 0
    fold_sums = np.asarray(
        [np.sum(metrics[assignments == fold], axis=0) for fold in range(5)]
    )
    relative = fold_sums[:, valid] / target[valid] - 1.0
    return float(np.mean(np.square(relative)) + 0.25 * np.max(np.abs(relative)) ** 2)


def summarize(name: str, statistics: np.ndarray, assignments: np.ndarray) -> None:
    rows, excursion, endpoint, outside = statistics.T
    threshold = np.quantile(excursion, 0.95)
    print(name)
    for fold in range(5):
        selected = assignments == fold
        print(
            f"fold={fold} wells={int(np.sum(selected))} "
            f"rows={int(np.sum(rows[selected]))} "
            f"mean_excursion={np.mean(excursion[selected]):.3f} "
            f"mean_endpoint={np.mean(endpoint[selected]):.3f} "
            f"outside_wells={int(np.count_nonzero(outside[selected]))} "
            f"outside_rows={int(np.sum(outside[selected]))} "
            f"endpoint_gt25={int(np.sum(endpoint[selected] > 25))} "
            f"endpoint_lt-25={int(np.sum(endpoint[selected] < -25))} "
            f"top5pct={int(np.sum(excursion[selected] >= threshold))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("stratified_folds.json"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=20260000)
    parser.add_argument("--candidates", type=int, default=5000)
    args = parser.parse_args()

    identifiers = np.asarray(training_well_ids(args.data_root))
    wells = load_all(args.data_root, args.workers)
    statistics = well_statistics(wells)
    excursion = statistics[:, 1]
    endpoint = statistics[:, 2]
    edges = np.quantile(excursion, np.linspace(0.0, 1.0, 11))
    excursion_bin = np.searchsorted(edges[1:-1], excursion)
    strata = excursion_bin * 2 + (endpoint >= 0)

    best_score = float("inf")
    best_seed = -1
    best_assignment = None
    for seed in range(args.seed_start, args.seed_start + args.candidates):
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        assignment = np.empty(len(identifiers), dtype=np.int64)
        for fold, (_, valid) in enumerate(splitter.split(identifiers, strata)):
            assignment[valid] = fold
        score = assignment_score(statistics, assignment)
        if score < best_score:
            best_score = score
            best_seed = seed
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("no candidate fold assignment generated")

    original = np.empty(len(identifiers), dtype=np.int64)
    for fold, (_, valid) in enumerate(
        GroupKFold(5).split(identifiers, groups=identifiers)
    ):
        original[valid] = fold
    summarize("original", statistics, original)
    summarize("stratified", statistics, best_assignment)
    print(f"selected_seed={best_seed} score={best_score:.9f}")
    print("old_by_new_contingency")
    print(
        np.asarray(
            [
                [
                    np.sum((original == old) & (best_assignment == new))
                    for new in range(5)
                ]
                for old in range(5)
            ]
        )
    )

    payload = {
        "strategy": "excursion-sign stratification with rare-risk balance search",
        "folds": 5,
        "seed": best_seed,
        "score": best_score,
        "assignments": {
            str(identifier): int(fold)
            for identifier, fold in zip(identifiers, best_assignment)
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
