from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluate_surface_extrapolation import surface_extrapolation
from evaluate_surface_smoothing import smooth_surface
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


def load_all(data_root: Path, workers: int) -> dict[str, Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
    return dict(zip(well_ids, wells))


def expanded_and_smoothed(
    well: Well, raw: np.ndarray, tail_expansion: float
) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, len(raw))
    prediction = raw + tail_expansion * progress * (
        raw - well.tvt[well.anchor_index]
    )
    full_tvt = well.tvt_input.copy()
    full_tvt[well.tail_indices] = raw
    surface = full_tvt + well.z
    prediction += (smooth_surface(surface, 1601) - surface)[well.tail_indices]
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-root", type=Path, default=Path("cache/v6_oof"))
    parser.add_argument("--tta-root", type=Path, default=Path("cache/v6_tta"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--components", nargs="*")
    parser.add_argument("--tail-expansion", type=float, default=0.05)
    parser.add_argument("--surface-extrapolation-weight", type=float, default=0.03)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )
    args = parser.parse_args()

    wells = load_all(args.data_root, args.workers)
    fold_sse = np.zeros((5, len(args.weights)), dtype=np.float64)
    fold_rows = np.zeros(5, dtype=np.int64)
    for fold in range(5):
        with np.load(args.base_root / f"fold{fold}.npz") as base_cache:
            well_ids = base_cache["well_ids"]
            starts = base_cache["row_starts"]
            base_raw = base_cache["coarse"].astype(np.float64)
            truth = base_cache["truth"].astype(np.float64)
            base_components = base_cache["components"].astype(np.float64)
            component_names = [str(name) for name in base_cache["component_names"]]
            component_weights = base_cache["component_weights"].astype(np.float64)
        with np.load(args.tta_root / f"fold{fold}_gr25.npz") as tta_cache:
            if not np.array_equal(well_ids, tta_cache["well_ids"]):
                raise ValueError(f"fold {fold}: base and TTA well order differs")
            if args.components:
                tta_components = tta_cache["components"].astype(np.float64)
                tta_raw = base_raw.copy()
                for name in args.components:
                    component = component_names.index(name)
                    tta_raw += component_weights[component] * (
                        tta_components[component] - base_components[component]
                    )
            else:
                tta_raw = tta_cache["coarse"].astype(np.float64)
        for index, well_id in enumerate(well_ids):
            well = wells[str(well_id)]
            left, right = int(starts[index]), int(starts[index + 1])
            base_pre = expanded_and_smoothed(
                well, base_raw[left:right], args.tail_expansion
            )
            tta_pre = expanded_and_smoothed(
                well, tta_raw[left:right], args.tail_expansion
            )
            for candidate, weight in enumerate(args.weights):
                prediction = (1.0 - weight) * base_pre + weight * tta_pre
                prediction += args.surface_extrapolation_weight * surface_extrapolation(
                    well,
                    prediction,
                    visible_rows=1000,
                    correction_limit=24.0,
                )
                error = prediction - truth[left:right]
                fold_sse[fold, candidate] += float(error @ error)
            fold_rows[fold] += right - left
        print(
            f"fold={fold} "
            + " ".join(
                f"weight={weight:.3f}:rmse="
                f"{np.sqrt(fold_sse[fold, candidate] / fold_rows[fold]):.6f}"
                for candidate, weight in enumerate(args.weights)
            )
        )

    total_rows = int(np.sum(fold_rows))
    pooled = np.sqrt(np.sum(fold_sse, axis=0) / total_rows)
    for weight, score in zip(args.weights, pooled):
        print(f"pooled weight={weight:.3f} rmse={score:.6f}")
    print("Leave-one-fold-out selection:")
    nested_sse = 0.0
    for heldout in range(5):
        tuning_sse = np.sum(np.delete(fold_sse, heldout, axis=0), axis=0)
        tuning_rows = total_rows - int(fold_rows[heldout])
        selected = int(np.argmin(tuning_sse / tuning_rows))
        nested_sse += fold_sse[heldout, selected]
        print(
            f"heldout_fold={heldout} weight={args.weights[selected]:.3f} "
            f"tuning_rmse={np.sqrt(tuning_sse[selected] / tuning_rows):.6f} "
            f"heldout_rmse="
            f"{np.sqrt(fold_sse[heldout, selected] / fold_rows[heldout]):.6f}"
        )
    print(f"nested_rmse={np.sqrt(nested_sse / total_rows):.6f}")


if __name__ == "__main__":
    main()
