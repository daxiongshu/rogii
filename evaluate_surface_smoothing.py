from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


def load_all(data_root: Path, workers: int) -> dict[str, Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
    return dict(zip(well_ids, wells))


def smooth_surface(surface: np.ndarray, requested_window: int) -> np.ndarray:
    window = min(
        requested_window,
        len(surface) if len(surface) % 2 else len(surface) - 1,
    )
    return savgol_filter(surface, window, 2) if window >= 5 else surface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=Path("cache/v6_oof"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--tail-expansion", type=float, default=0.02)
    parser.add_argument("--window", type=int, default=1601)
    args = parser.parse_args()
    if args.window % 2 != 1:
        raise ValueError("window must be odd")

    wells = load_all(args.data_root, args.workers)
    base_sse = 0.0
    smoothed_sse = 0.0
    rows = 0
    for fold, path in enumerate(sorted(args.cache_root.glob("fold*.npz"))):
        fold_base_sse = 0.0
        fold_smoothed_sse = 0.0
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
                expanded = raw + args.tail_expansion * progress * (
                    raw - well.tvt[well.anchor_index]
                )
                full_tvt = well.tvt_input.copy()
                full_tvt[well.tail_indices] = raw
                surface = full_tvt + well.z
                correction = (smooth_surface(surface, args.window) - surface)[
                    well.tail_indices
                ]
                base_error = expanded - truth[left:right]
                smoothed_error = expanded + correction - truth[left:right]
                fold_base_sse += float(base_error @ base_error)
                fold_smoothed_sse += float(smoothed_error @ smoothed_error)
                fold_rows += len(base_error)
        print(
            f"fold={fold} base={np.sqrt(fold_base_sse / fold_rows):.6f} "
            f"smoothed={np.sqrt(fold_smoothed_sse / fold_rows):.6f}"
        )
        base_sse += fold_base_sse
        smoothed_sse += fold_smoothed_sse
        rows += fold_rows
    print(
        f"pooled_base={np.sqrt(base_sse / rows):.6f} "
        f"pooled_smoothed={np.sqrt(smoothed_sse / rows):.6f}"
    )


if __name__ == "__main__":
    main()
