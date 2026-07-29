from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well


@dataclass(frozen=True)
class ReferencePath:
    xy: np.ndarray
    ancc: np.ndarray
    tree: cKDTree


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def load_reference_path(
    well_id: str, data_root: Path, reference_stride: int
) -> ReferencePath:
    path = data_root / "train" / f"{well_id}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["X", "Y", "ANCC"])
    values = frame[["X", "Y", "ANCC"]].to_numpy(np.float64)[::reference_stride]
    values = values[np.all(np.isfinite(values), axis=1)]
    xy = values[:, :2]
    return ReferencePath(xy=xy, ancc=values[:, 2], tree=cKDTree(xy))


def distinct_reference_wells(
    anchor_xy: np.ndarray,
    global_tree: cKDTree,
    point_well: np.ndarray,
    neighbor_count: int,
    candidate_points: int,
) -> list[int]:
    _, indices = global_tree.query(
        anchor_xy, k=min(candidate_points, len(point_well))
    )
    selected = []
    seen = set()
    for point_index in np.atleast_1d(indices):
        well_index = int(point_well[point_index])
        if well_index not in seen:
            seen.add(well_index)
            selected.append(well_index)
        if len(selected) == neighbor_count:
            break
    if not selected:
        raise RuntimeError("failed to find an ANCC reference well")
    return selected


def predict_ancc_correction(
    well: Well,
    baseline: np.ndarray,
    references: list[ReferencePath],
    selected: list[int],
    query_stride: int,
    distance_floor: float,
) -> np.ndarray:
    tail = well.tail_indices
    query_index = np.unique(
        np.r_[well.anchor_index, tail[::query_stride], tail[-1]]
    ).astype(int)
    query_xy = np.column_stack((well.x[query_index], well.y[query_index]))
    curves = []
    distances = []
    for reference_index in selected:
        reference = references[reference_index]
        distance, row = reference.tree.query(query_xy, k=1)
        curves.append(reference.ancc[row])
        distances.append(distance)
    curves = np.asarray(curves)
    distances = np.asarray(distances)
    weight = 1.0 / np.square(np.maximum(distances, distance_floor))
    ancc = np.sum(weight * curves, axis=0) / np.sum(weight, axis=0)

    anchor_surface = float(
        well.tvt_input[well.anchor_index] + well.z[well.anchor_index]
    )
    candidate_surface = anchor_surface + ancc - ancc[0]
    candidate_tvt = np.interp(tail, query_index, candidate_surface) - well.z[tail]
    return candidate_tvt - baseline


def nested_audit(
    baseline: np.ndarray,
    truth: np.ndarray,
    corrections: np.ndarray,
    well_fold: np.ndarray,
    row_starts: np.ndarray,
) -> tuple[float, list[np.ndarray]]:
    row_fold = np.repeat(well_fold, np.diff(row_starts))
    prediction = np.empty_like(baseline)
    selected = []
    for held_out in range(5):
        train = row_fold != held_out
        valid = ~train
        coefficient = np.linalg.lstsq(
            corrections[train], truth[train] - baseline[train], rcond=None
        )[0]
        coefficient = np.maximum(coefficient, 0.0)
        prediction[valid] = baseline[valid] + corrections[valid] @ coefficient
        selected.append(coefficient)
    return rmse(prediction, truth), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--reference-stride", type=int, default=8)
    parser.add_argument("--query-stride", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--candidate-points", type=int, default=10000)
    parser.add_argument("--distance-floor", type=float, default=50.0)
    parser.add_argument("--weight", type=float, default=0.01)
    parser.add_argument("--gr-cache", type=Path)
    parser.add_argument("--gr-weight", type=float, default=0.15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_cache = np.load(args.baseline_cache, allow_pickle=True)
    layout = np.load(args.layout_cache, allow_pickle=True)
    well_ids = baseline_cache["well_ids"].astype(str)
    row_starts = baseline_cache["row_starts"].astype(np.int64)
    baseline = baseline_cache["prediction"].astype(np.float64)
    truth = layout["truth"].astype(np.float64)
    if not np.array_equal(well_ids, layout["well_ids"].astype(str)):
        raise ValueError("baseline and layout well IDs do not match")
    if not np.array_equal(row_starts, layout["row_starts"].astype(np.int64)):
        raise ValueError("baseline and layout row starts do not match")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        wells = list(
            pool.map(lambda well_id: load_well(well_id, args.data_root), well_ids)
        )
        references = list(
            pool.map(
                lambda well_id: load_reference_path(
                    well_id, args.data_root, args.reference_stride
                ),
                well_ids,
            )
        )

    all_xy = np.concatenate([reference.xy for reference in references])
    all_well = np.concatenate(
        [
            np.full(len(reference.xy), index, dtype=np.int32)
            for index, reference in enumerate(references)
        ]
    )
    original_fold = layout["original"].astype(np.int64)
    correction = np.empty_like(baseline)
    for fold in range(5):
        keep = original_fold[all_well] != fold
        global_tree = cKDTree(all_xy[keep])
        point_well = all_well[keep]
        fold_indices = np.flatnonzero(original_fold == fold)
        for count, well_index in enumerate(fold_indices, start=1):
            well = wells[well_index]
            left, right = row_starts[well_index : well_index + 2]
            selected = distinct_reference_wells(
                np.asarray([well.x[well.anchor_index], well.y[well.anchor_index]]),
                global_tree,
                point_well,
                args.neighbors,
                args.candidate_points,
            )
            correction[left:right] = predict_ancc_correction(
                well,
                baseline[left:right],
                references,
                selected,
                args.query_stride,
                args.distance_floor,
            )
            if count % 40 == 0 or count == len(fold_indices):
                print(f"fold={fold} wells={count}/{len(fold_indices)}", flush=True)

    corrections = [correction]
    fixed_weights = [args.weight]
    if args.gr_cache:
        gr_cache = np.load(args.gr_cache, allow_pickle=True)
        if not np.array_equal(well_ids, gr_cache["well_ids"].astype(str)):
            raise ValueError("GR cache well IDs do not match")
        if not np.array_equal(row_starts, gr_cache["row_starts"].astype(np.int64)):
            raise ValueError("GR cache row starts do not match")
        corrections.append(gr_cache["correction"].astype(np.float64))
        fixed_weights.append(args.gr_weight)
    correction_matrix = np.column_stack(corrections)
    fixed_weights_array = np.asarray(fixed_weights)
    prediction = baseline + correction_matrix @ fixed_weights_array
    print(
        f"baseline={rmse(baseline, truth):.9f} "
        f"candidate={rmse(prediction, truth):.9f} "
        f"weights={fixed_weights}",
        flush=True,
    )

    for grouping in ("original", "stratified", "independent"):
        well_fold = layout[grouping].astype(np.int64)
        row_fold = np.repeat(well_fold, np.diff(row_starts))
        gains = []
        for fold in range(5):
            mask = row_fold == fold
            gains.append(rmse(baseline[mask], truth[mask]) - rmse(prediction[mask], truth[mask]))
        nested_score, nested_weights = nested_audit(
            baseline, truth, correction_matrix, well_fold, row_starts
        )
        print(
            f"{grouping}: fixed_fold_gains={np.round(gains, 6).tolist()} "
            f"nested={nested_score:.9f} "
            f"nested_weights={[np.round(weight, 5).tolist() for weight in nested_weights]}",
            flush=True,
        )

    if args.output:
        np.savez_compressed(
            args.output,
            well_ids=well_ids,
            row_starts=row_starts,
            correction=correction.astype(np.float32),
            prediction=prediction.astype(np.float32),
            ancc_weight=args.weight,
            gr_weight=args.gr_weight if args.gr_cache else 0.0,
        )


if __name__ == "__main__":
    main()
