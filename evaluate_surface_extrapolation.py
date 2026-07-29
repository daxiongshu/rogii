from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluate_surface_smoothing import smooth_surface
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


def load_all(data_root: Path, workers: int) -> dict[str, Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
    return dict(zip(well_ids, wells))


def surface_extrapolation(
    well: Well,
    prediction: np.ndarray,
    visible_rows: int,
    correction_limit: float,
) -> np.ndarray:
    anchor = well.anchor_index
    left = max(0, anchor + 1 - visible_rows)
    prefix_md = well.md[left : anchor + 1]
    prefix_surface = well.tvt_input[left : anchor + 1] + well.z[left : anchor + 1]
    centered_md = prefix_md - np.mean(prefix_md)
    centered_surface = prefix_surface - np.mean(prefix_surface)
    slope = float(centered_md @ centered_surface) / max(
        float(centered_md @ centered_md), 1e-9
    )
    extrapolated = prefix_surface[-1] + slope * (
        well.md[well.tail_indices] - well.md[anchor]
    )
    correction = extrapolated - (prediction + well.z[well.tail_indices])
    return np.clip(correction, -correction_limit, correction_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("cache/v6_oof"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tail-expansion", type=float, default=0.02)
    parser.add_argument("--smooth-window", type=int, default=1601)
    parser.add_argument("--visible-rows", type=int, default=1000)
    parser.add_argument("--correction-limit", type=float, default=24.0)
    parser.add_argument("--weight", type=float, default=0.02)
    args = parser.parse_args()

    wells = load_all(args.data_root, args.workers)
    base_sse = 0.0
    corrected_sse = 0.0
    rows = 0
    for fold, path in enumerate(sorted(args.cache_root.glob("fold*.npz"))):
        fold_base_sse = 0.0
        fold_corrected_sse = 0.0
        fold_rows = 0
        with np.load(path) as cached:
            starts = cached["row_starts"]
            coarse = cached["coarse"].astype(np.float64)
            truth = cached["truth"].astype(np.float64)
            for index, well_id in enumerate(cached["well_ids"]):
                well = wells[str(well_id)]
                left, right = int(starts[index]), int(starts[index + 1])
                raw = coarse[left:right]
                progress = np.linspace(0.0, 1.0, right - left)
                base = raw + args.tail_expansion * progress * (
                    raw - well.tvt[well.anchor_index]
                )
                full_tvt = well.tvt_input.copy()
                full_tvt[well.tail_indices] = raw
                surface = full_tvt + well.z
                base += (smooth_surface(surface, args.smooth_window) - surface)[
                    well.tail_indices
                ]
                correction = surface_extrapolation(
                    well,
                    base,
                    args.visible_rows,
                    args.correction_limit,
                )
                base_error = base - truth[left:right]
                corrected_error = base + args.weight * correction - truth[left:right]
                fold_base_sse += float(base_error @ base_error)
                fold_corrected_sse += float(corrected_error @ corrected_error)
                fold_rows += len(base_error)
        print(
            f"fold={fold} base={np.sqrt(fold_base_sse / fold_rows):.6f} "
            f"corrected={np.sqrt(fold_corrected_sse / fold_rows):.6f}"
        )
        base_sse += fold_base_sse
        corrected_sse += fold_corrected_sse
        rows += fold_rows
    print(
        f"pooled_base={np.sqrt(base_sse / rows):.6f} "
        f"pooled_corrected={np.sqrt(corrected_sse / rows):.6f}"
    )


if __name__ == "__main__":
    main()
