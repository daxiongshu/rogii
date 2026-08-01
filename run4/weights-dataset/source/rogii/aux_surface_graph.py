from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .data import Well


@dataclass
class SurfaceReferenceStore:
    family_by_id: dict[str, int]
    reference_ids: list[str]
    reference_family: np.ndarray
    reference_xy: list[np.ndarray]
    reference_surface: list[np.ndarray]
    reference_trees: list[cKDTree]
    family_indices: dict[int, np.ndarray]
    family_trees: dict[int, tuple[cKDTree, np.ndarray]]


def typewell_families(
    wells: list[Well],
    grid: np.ndarray | None = None,
    minimum_overlap: int = 200,
    correlation_threshold: float = 0.99,
) -> tuple[np.ndarray, dict]:
    """Group legal typewell logs by their common-grid correlation."""
    if grid is None:
        grid = np.arange(9200.0, 13001.0, 1.0)
    values = np.zeros((len(wells), len(grid)), dtype=np.float32)
    support = np.zeros_like(values, dtype=bool)
    for index, well in enumerate(wells):
        valid = (
            (grid >= np.nanmin(well.typewell_tvt))
            & (grid <= np.nanmax(well.typewell_tvt))
        )
        values[index, valid] = np.interp(
            grid[valid],
            well.typewell_tvt,
            well.typewell_gr,
        )
        support[index, valid] = True

    parent = np.arange(len(wells), dtype=np.int64)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges = 0
    for left in range(len(wells)):
        for right in range(left):
            overlap = support[left] & support[right]
            if int(overlap.sum()) < minimum_overlap:
                continue
            first = values[left, overlap].astype(np.float64)
            second = values[right, overlap].astype(np.float64)
            first -= np.mean(first)
            second -= np.mean(second)
            correlation = float(first @ second) / np.sqrt(
                max(float(first @ first) * float(second @ second), 1e-12)
            )
            if correlation >= correlation_threshold:
                union(left, right)
                edges += 1

    roots = np.asarray([root(index) for index in range(len(wells))])
    unique_roots = sorted(
        np.unique(roots),
        key=lambda value: wells[int(np.flatnonzero(roots == value)[0])].well_id,
    )
    remap = {int(value): index for index, value in enumerate(unique_roots)}
    family = np.asarray([remap[int(value)] for value in roots], dtype=np.int32)
    sizes = np.bincount(family)
    return family, {
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "grid_step": float(grid[1] - grid[0]),
        "minimum_overlap": minimum_overlap,
        "correlation_threshold": correlation_threshold,
        "edges": edges,
        "families": len(sizes),
        "nontrivial_families": int(np.sum(sizes > 1)),
        "covered_by_nontrivial_family": int(np.sum(sizes[sizes > 1])),
        "maximum_family_size": int(np.max(sizes)),
    }


def build_surface_reference_store(
    all_wells: list[Well],
    family: np.ndarray,
    reference_mask: np.ndarray,
    stride: int = 16,
) -> SurfaceReferenceStore:
    if len(all_wells) != len(family) or len(all_wells) != len(reference_mask):
        raise ValueError("surface reference arrays differ in length")
    selected = np.flatnonzero(reference_mask)
    reference_ids = [all_wells[index].well_id for index in selected]
    reference_family = family[selected].astype(np.int32)
    reference_xy = []
    reference_surface = []
    for index in selected:
        well = all_wells[index]
        sampled = np.unique(
            np.r_[
                np.arange(0, len(well.md), stride),
                len(well.md) - 1,
                well.anchor_index,
            ]
        )
        sampled.sort()
        reference_xy.append(
            np.column_stack((well.x[sampled], well.y[sampled])).astype(
                np.float64
            )
        )
        reference_surface.append(
            (well.tvt[sampled] + well.z[sampled]).astype(np.float64)
        )
    reference_trees = [cKDTree(value) for value in reference_xy]
    family_indices = {
        int(value): np.flatnonzero(reference_family == value)
        for value in np.unique(reference_family)
    }
    family_trees = {}
    for value, indices in family_indices.items():
        xy = np.concatenate([reference_xy[index] for index in indices])
        surface = np.concatenate(
            [reference_surface[index] for index in indices]
        )
        family_trees[value] = (cKDTree(xy), surface)
    return SurfaceReferenceStore(
        family_by_id={
            well.well_id: int(code) for well, code in zip(all_wells, family)
        },
        reference_ids=reference_ids,
        reference_family=reference_family,
        reference_xy=reference_xy,
        reference_surface=reference_surface,
        reference_trees=reference_trees,
        family_indices=family_indices,
        family_trees=family_trees,
    )


def surface_transfer_paths(
    well: Well,
    baseline: np.ndarray,
    store: SurfaceReferenceStore,
    query_stride: int = 16,
    analog_wells: int = 3,
    local_points: int = 32,
) -> dict[str, np.ndarray | float | int]:
    """Return fold-purged, anchor-aligned analog and local surface paths."""
    family = store.family_by_id.get(well.well_id, -1)
    tail = well.prediction_indices
    anchor = well.anchor_index
    query_indices = np.unique(
        np.r_[
            np.arange(0, len(well.md), query_stride),
            len(well.md) - 1,
            anchor,
        ]
    )
    query_indices.sort()
    query_xy = np.column_stack(
        (well.x[query_indices], well.y[query_indices])
    )
    visible = query_indices <= anchor
    recent_visible = np.flatnonzero(visible)[
        -min(64, int(visible.sum())) :
    ]
    finite_visible = np.isfinite(
        well.tvt_input[query_indices[recent_visible]]
    )
    hidden_sample = query_indices > anchor
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for index in store.family_indices.get(
        family,
        np.empty(0, dtype=np.int64),
    ):
        distance, nearest = store.reference_trees[index].query(
            query_xy,
            k=1,
        )
        score = (
            float(np.median(distance[hidden_sample]))
            if np.any(hidden_sample)
            else float("inf")
        )
        candidates.append(
            (score, distance, store.reference_surface[index][nearest])
        )
    candidates.sort(key=lambda item: item[0])

    visible_surface = (
        well.tvt_input[query_indices[recent_visible]][finite_visible]
        + well.z[query_indices[recent_visible]][finite_visible]
    )
    if candidates and len(visible_surface):
        top = candidates[: min(analog_wells, len(candidates))]
        distance = np.stack([item[1] for item in top])
        surface = np.stack([item[2] for item in top])
        weight = 1.0 / np.square(np.maximum(distance, 25.0))
        analog_surface = np.sum(weight * surface, axis=0) / np.sum(
            weight,
            axis=0,
        )
        analog_bias = float(
            np.median(
                visible_surface
                - analog_surface[recent_visible][finite_visible]
            )
        )
        analog = np.interp(
            tail,
            query_indices,
            analog_surface + analog_bias - well.z[query_indices],
        )
        analog_distance = float(candidates[0][0])
    else:
        analog = baseline.copy()
        analog_distance = float("inf")

    selected = store.family_indices.get(
        family,
        np.empty(0, dtype=np.int64),
    )
    if len(selected) and len(visible_surface):
        tree, surface = store.family_trees[family]
        distance, nearest = tree.query(
            query_xy,
            k=min(local_points, len(surface)),
        )
        distance = np.asarray(distance)
        nearest = np.asarray(nearest)
        if distance.ndim == 1:
            distance = distance[:, None]
            nearest = nearest[:, None]
        weight = 1.0 / np.square(np.maximum(distance, 10.0))
        local_surface = np.sum(weight * surface[nearest], axis=1) / np.sum(
            weight,
            axis=1,
        )
        local_bias = float(
            np.median(
                visible_surface
                - local_surface[recent_visible][finite_visible]
            )
        )
        local = np.interp(
            tail,
            query_indices,
            local_surface + local_bias - well.z[query_indices],
        )
        local_distance = (
            float(np.median(distance[hidden_sample, 0]))
            if np.any(hidden_sample)
            else float("inf")
        )
    else:
        local = baseline.copy()
        local_distance = float("inf")

    return {
        "analog": analog,
        "local": local,
        "analog_distance": analog_distance,
        "local_distance": local_distance,
        "family": family,
        "family_size": len(selected),
    }
