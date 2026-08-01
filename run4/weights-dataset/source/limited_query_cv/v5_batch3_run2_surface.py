from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import BSpline


LEGAL_HORIZONTAL_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TRAIN_ONLY_HORIZONTAL_COLUMNS = ("TVT", "ANCC", "BUDA")


@dataclass(frozen=True)
class SurfaceWell:
    well_id: str
    x: np.ndarray
    y: np.ndarray
    buda: np.ndarray


def inference_frame(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Return the complete held-well inference view and nothing else."""
    missing = set(LEGAL_HORIZONTAL_COLUMNS) - set(horizontal.columns)
    if missing:
        raise ValueError(f"missing inference columns: {sorted(missing)}")
    return horizontal.loc[:, LEGAL_HORIZONTAL_COLUMNS].copy()


def _trajectory_distance_squared(
    x: np.ndarray,
    y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    anchors: int = 12,
) -> np.ndarray:
    index = np.linspace(
        0,
        len(target_x) - 1,
        min(anchors, len(target_x)),
    ).astype(np.int64)
    dx = x[:, None] - target_x[index][None]
    dy = y[:, None] - target_y[index][None]
    return np.min(dx * dx + dy * dy, axis=1)


def _structural_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    azimuth_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(float(azimuth_degrees))
    cosine = np.cos(angle)
    sine = np.sin(angle)
    dx = x - center_x
    dy = y - center_y
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _quadratic_design(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.ones(len(u)),
            u,
            v,
            np.square(u),
            u * v,
            np.square(v),
        )
    )


def predict_local_surface(
    pool: dict[str, SurfaceWell],
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    neighbor_count: int = 16,
    bandwidth_multiplier: float = 1.0,
    azimuth_degrees: float = 138.0,
    ridge: float = 1e-3,
    samples_per_well: int = 192,
) -> np.ndarray:
    """Predict BUDA from a role-purged pool using local weighted regression."""
    if not pool:
        raise ValueError("empty role-purged surface pool")
    target_x = np.asarray(target_x, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64)
    center_x = float(np.mean(target_x))
    center_y = float(np.mean(target_y))
    ranked = []
    for well_id, well in pool.items():
        distance = np.hypot(
            float(np.mean(well.x)) - center_x,
            float(np.mean(well.y)) - center_y,
        )
        ranked.append((distance, well_id))
    ranked.sort()
    chosen = [
        pool[well_id]
        for _, well_id in ranked[: min(int(neighbor_count), len(ranked))]
    ]
    x_parts = []
    y_parts = []
    surface_parts = []
    for well in chosen:
        finite = (
            np.isfinite(well.x)
            & np.isfinite(well.y)
            & np.isfinite(well.buda)
        )
        index = np.flatnonzero(finite)
        if len(index) > samples_per_well:
            index = index[
                np.linspace(0, len(index) - 1, samples_per_well).astype(int)
            ]
        x_parts.append(np.asarray(well.x[index], dtype=np.float64))
        y_parts.append(np.asarray(well.y[index], dtype=np.float64))
        surface_parts.append(np.asarray(well.buda[index], dtype=np.float64))
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    surface = np.concatenate(surface_parts)
    distance2 = _trajectory_distance_squared(x, y, target_x, target_y)
    positive_distances = np.sqrt(distance2[distance2 > 0])
    natural_bandwidth = (
        float(np.median(positive_distances))
        if len(positive_distances)
        else 1.0
    )
    bandwidth = max(50.0, bandwidth_multiplier * natural_bandwidth)
    weight = np.exp(-0.5 * distance2 / (bandwidth * bandwidth))
    u, v = _structural_coordinates(
        x, y, center_x, center_y, azimuth_degrees
    )
    target_u, target_v = _structural_coordinates(
        target_x, target_y, center_x, center_y, azimuth_degrees
    )
    scale_u = max(float(np.std(u)), 100.0)
    scale_v = max(float(np.std(v)), 100.0)
    design = _quadratic_design(u / scale_u, v / scale_v)
    target_design = _quadratic_design(
        target_u / scale_u, target_v / scale_v
    )
    surface_center = float(np.sum(weight * surface) / np.sum(weight))
    weighted_design = design * np.sqrt(weight[:, None])
    weighted_target = (surface - surface_center) * np.sqrt(weight)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 1e-10
    coefficient = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return target_design @ coefficient + surface_center


def surface_to_tvt(
    surface: np.ndarray,
    z: np.ndarray,
    tvt_input: np.ndarray,
    *,
    residual_decay_rows: int = 0,
    residual_fit_rows: int = 128,
) -> np.ndarray:
    """Use the visible anchor to cancel the unknown per-well surface datum."""
    surface = np.asarray(surface, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    tvt_input = np.asarray(tvt_input, dtype=np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise ValueError("surface reconstruction requires a visible anchor")
    anchor = int(known[-1])
    reconstructed = (
        tvt_input[anchor]
        + (surface - z)
        - (surface[anchor] - z[anchor])
    )
    if residual_decay_rows > 0 and len(known) >= 4:
        fit = known[-min(int(residual_fit_rows), len(known)) :]
        axis = fit.astype(np.float64) - float(anchor)
        residual = tvt_input[fit] - reconstructed[fit]
        slope = float(np.polyfit(axis, residual, 1)[0])
        future = np.maximum(
            np.arange(len(surface), dtype=np.float64) - anchor,
            0.0,
        )
        reconstructed += (
            slope
            * future
            * np.exp(-future / float(residual_decay_rows))
        )
    return reconstructed


def role_purged_buda_prediction(
    pool: dict[str, SurfaceWell],
    horizontal: pd.DataFrame,
    **surface_options: float | int,
) -> np.ndarray:
    """Predict a held well without reading BUDA, ANCC, or hidden TVT."""
    visible = inference_frame(horizontal)
    surface = predict_local_surface(
        pool,
        visible["X"].to_numpy(np.float64),
        visible["Y"].to_numpy(np.float64),
        neighbor_count=int(surface_options.get("neighbor_count", 16)),
        bandwidth_multiplier=float(
            surface_options.get("bandwidth_multiplier", 1.0)
        ),
        azimuth_degrees=float(
            surface_options.get("azimuth_degrees", 138.0)
        ),
        ridge=float(surface_options.get("ridge", 1e-3)),
        samples_per_well=int(
            surface_options.get("samples_per_well", 192)
        ),
    )
    tvt = surface_to_tvt(
        surface,
        visible["Z"].to_numpy(np.float64),
        visible["TVT_input"].to_numpy(np.float64),
        residual_decay_rows=int(
            surface_options.get("residual_decay_rows", 0)
        ),
        residual_fit_rows=int(
            surface_options.get("residual_fit_rows", 128)
        ),
    )
    return tvt[visible["TVT_input"].isna().to_numpy()]


def _bspline_design(
    values: np.ndarray,
    lower: float,
    upper: float,
    knots: int,
    degree: int = 3,
) -> tuple[np.ndarray, int]:
    values = np.clip(values, lower, upper)
    interior = np.linspace(lower, upper, int(knots) + 1)
    knot_vector = np.concatenate(
        ([lower] * degree, interior, [upper] * degree)
    )
    basis_count = len(knot_vector) - degree - 1
    design = BSpline.design_matrix(
        values,
        knot_vector,
        degree,
        extrapolate=True,
    ).toarray()
    return design, basis_count


def _difference_penalty(size: int, order: int = 2) -> np.ndarray:
    difference = np.eye(size)
    for _ in range(order):
        difference = np.diff(difference, axis=0)
    return difference.T @ difference


def _weighted_pspline(
    training_xy: np.ndarray,
    training_surface: np.ndarray,
    training_weight: np.ndarray,
    target_xy: np.ndarray,
    domain: tuple[float, float, float, float],
    knots_x: int,
    knots_y: int,
    penalty_x: float,
    penalty_y: float,
    device: str,
) -> np.ndarray:
    import torch

    target = torch.device(device)
    lower_x, upper_x, lower_y, upper_y = domain
    basis_x, size_x = _bspline_design(
        training_xy[:, 0], lower_x, upper_x, knots_x
    )
    basis_y, size_y = _bspline_design(
        training_xy[:, 1], lower_y, upper_y, knots_y
    )
    target_x, _ = _bspline_design(
        target_xy[:, 0], lower_x, upper_x, knots_x
    )
    target_y, _ = _bspline_design(
        target_xy[:, 1], lower_y, upper_y, knots_y
    )
    basis_x_t = torch.as_tensor(
        basis_x, dtype=torch.float64, device=target
    )
    basis_y_t = torch.as_tensor(
        basis_y, dtype=torch.float64, device=target
    )
    target_x_t = torch.as_tensor(
        target_x, dtype=torch.float64, device=target
    )
    target_y_t = torch.as_tensor(
        target_y, dtype=torch.float64, device=target
    )
    surface_t = torch.as_tensor(
        training_surface, dtype=torch.float64, device=target
    )
    weight_t = torch.as_tensor(
        training_weight, dtype=torch.float64, device=target
    )
    surface_center = torch.sum(weight_t * surface_t) / torch.sum(weight_t)
    surface_t = surface_t - surface_center

    def tensor_product(
        first: "torch.Tensor",
        second: "torch.Tensor",
    ) -> "torch.Tensor":
        return (
            first.unsqueeze(2) * second.unsqueeze(1)
        ).reshape(len(first), -1)

    design = tensor_product(basis_x_t, basis_y_t)
    weighted_transpose = design.T * weight_t.unsqueeze(0)
    px = torch.as_tensor(
        _difference_penalty(size_x),
        dtype=torch.float64,
        device=target,
    )
    py = torch.as_tensor(
        _difference_penalty(size_y),
        dtype=torch.float64,
        device=target,
    )
    identity_x = torch.eye(size_x, dtype=torch.float64, device=target)
    identity_y = torch.eye(size_y, dtype=torch.float64, device=target)
    penalty = (
        float(penalty_x) * torch.kron(px, identity_y)
        + float(penalty_y) * torch.kron(identity_x, py)
    )
    system = (
        weighted_transpose @ design
        + penalty
        + 1e-6
        * torch.eye(size_x * size_y, dtype=torch.float64, device=target)
    )
    coefficient = torch.linalg.solve(
        system,
        weighted_transpose @ surface_t,
    )
    prediction = (
        tensor_product(target_x_t, target_y_t) @ coefficient
        + surface_center
    )
    return prediction.cpu().numpy()


def _subsample_surface_well(
    well: SurfaceWell,
    ratio: float,
) -> np.ndarray:
    size = max(2, int(round(len(well.x) * float(ratio))))
    if size >= len(well.x):
        index = np.arange(len(well.x))
    else:
        index = np.linspace(0, len(well.x) - 1, size).astype(int)
    return np.column_stack(
        (well.x[index], well.y[index], well.buda[index])
    )


def pspline_surface_members(
    pool: dict[str, SurfaceWell],
    horizontal: pd.DataFrame,
    *,
    device: str = "cuda:0",
    subsample_ratio: float = 0.08,
    bandwidth: float = 1800.0,
    wide_bandwidth: float = 8000.0,
    gap_threshold_degrees: float = 200.0,
    gap_neighbors: int = 50,
    weight_cutoff: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis, structural-frame, and mean BUDA surfaces from a purged pool."""
    visible = inference_frame(horizontal)
    target_xy = visible[["X", "Y"]].to_numpy(np.float64)
    target_center = np.mean(target_xy, axis=0)
    points = np.vstack(
        [
            _subsample_surface_well(well, subsample_ratio)
            for well in pool.values()
        ]
    )
    centroids = np.asarray(
        [
            (float(np.mean(well.x)), float(np.mean(well.y)))
            for well in pool.values()
        ]
    )
    selected_bandwidth = float(bandwidth)
    if len(centroids) > int(gap_neighbors):
        distance2 = np.sum(
            np.square(centroids - target_center[None]),
            axis=1,
        )
        nearest = np.argsort(distance2)[: int(gap_neighbors)]
        angle = np.sort(
            np.arctan2(
                centroids[nearest, 1] - target_center[1],
                centroids[nearest, 0] - target_center[0],
            )
        )
        gaps = np.diff(np.concatenate((angle, [angle[0] + 2.0 * np.pi])))
        if np.degrees(np.max(gaps)) > float(gap_threshold_degrees):
            selected_bandwidth = float(wide_bandwidth)
    anchor_index = np.linspace(
        0,
        len(target_xy) - 1,
        min(8, len(target_xy)),
    ).astype(int)
    delta = (
        points[:, None, :2] - target_xy[anchor_index][None]
    )
    distance2 = np.min(np.sum(np.square(delta), axis=2), axis=1)
    weight = np.exp(
        -0.5 * distance2 / (selected_bandwidth * selected_bandwidth)
    )
    keep = weight > float(weight_cutoff)
    points = points[keep]
    weight = weight[keep]
    if len(points) < 100:
        raise ValueError("insufficient local points for P-spline surface")
    members = []
    recipes = (
        (0.0, 16, 20, 12.0, 6.0),
        (138.0, 20, 15, 18.0, 11.0),
    )
    for angle, knots_x, knots_y, penalty_x, penalty_y in recipes:
        training_u, training_v = _structural_coordinates(
            points[:, 0],
            points[:, 1],
            float(target_center[0]),
            float(target_center[1]),
            angle,
        )
        target_u, target_v = _structural_coordinates(
            target_xy[:, 0],
            target_xy[:, 1],
            float(target_center[0]),
            float(target_center[1]),
            angle,
        )
        training_frame = np.column_stack((training_u, training_v))
        target_frame = np.column_stack((target_u, target_v))
        domain = (
            min(float(np.min(training_u)), float(np.min(target_u))),
            max(float(np.max(training_u)), float(np.max(target_u))),
            min(float(np.min(training_v)), float(np.min(target_v))),
            max(float(np.max(training_v)), float(np.max(target_v))),
        )
        members.append(
            _weighted_pspline(
                training_frame,
                points[:, 2],
                weight,
                target_frame,
                domain,
                knots_x,
                knots_y,
                penalty_x,
                penalty_y,
                device,
            )
        )
    member_array = np.stack(members)
    return member_array, np.mean(member_array, axis=0)


def role_purged_pspline_predictions(
    pool: dict[str, SurfaceWell],
    horizontal: pd.DataFrame,
    *,
    device: str = "cuda:0",
) -> tuple[np.ndarray, list[str]]:
    visible = inference_frame(horizontal)
    members, ensemble = pspline_surface_members(
        pool, visible, device=device
    )
    surfaces = (members[0], members[1], ensemble)
    predictions = []
    names = []
    for surface_name, surface in zip(
        ("axis", "structural", "ensemble"),
        surfaces,
    ):
        for decay in (0, 500, 1000):
            tvt = surface_to_tvt(
                surface,
                visible["Z"].to_numpy(np.float64),
                visible["TVT_input"].to_numpy(np.float64),
                residual_decay_rows=decay,
                residual_fit_rows=200,
            )
            predictions.append(
                tvt[visible["TVT_input"].isna().to_numpy()]
            )
            names.append(f"pspline_{surface_name}_d{decay}")
    return np.stack(predictions), names


def smooth_dip(md: np.ndarray, z: np.ndarray, sigma_rows: float) -> np.ndarray:
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    derivative = np.gradient(z, md)
    return gaussian_filter1d(
        derivative,
        sigma=max(float(sigma_rows), 0.01),
        mode="nearest",
    )


def delayed_dip_channels(
    md: np.ndarray,
    z: np.ndarray,
    *,
    shifts: tuple[int, ...] = (200, 250, 300, 350, 400),
    smoothing_rows: tuple[int, ...] = (31, 63, 127, 255),
) -> tuple[np.ndarray, list[str]]:
    """Inference-only delayed steering channels; positive shift means i-shift."""
    channels = []
    names = []
    for smoothing in smoothing_rows:
        dip = smooth_dip(md, z, smoothing)
        for shift in shifts:
            index = np.maximum(
                np.arange(len(dip), dtype=np.int64) - int(shift),
                0,
            )
            channels.append(dip[index])
            names.append(f"dip_s{smoothing}_lag{shift}")
    return np.column_stack(channels), names


def resample_typewell(
    tvt: np.ndarray,
    gr: np.ndarray,
    step: float = 0.5,
) -> tuple[int, np.ndarray]:
    finite = np.isfinite(tvt) & np.isfinite(gr)
    tvt = np.asarray(tvt, dtype=np.float64)[finite]
    gr = np.asarray(gr, dtype=np.float64)[finite]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    grid = np.arange(
        np.floor(tvt.min() / step),
        np.ceil(tvt.max() / step) + 1,
    ) * step
    return int(round(grid[0] / step)), np.interp(grid, tvt, gr)


def typewell_match_count(
    first: tuple[int, np.ndarray],
    second: tuple[int, np.ndarray],
    *,
    minimum_overlap: int = 250,
    tolerance: float = 0.01,
) -> int:
    first_start, first_gr = first
    second_start, second_gr = second
    lower = max(first_start, second_start)
    upper = min(
        first_start + len(first_gr),
        second_start + len(second_gr),
    )
    if upper - lower < int(minimum_overlap):
        return 0
    first_slice = first_gr[lower - first_start : upper - first_start]
    second_slice = second_gr[lower - second_start : upper - second_start]
    return int(np.sum(np.abs(first_slice - second_slice) < tolerance))


def typewell_clusters(
    typewells: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    minimum_match: int = 50,
    minimum_overlap: int = 250,
) -> dict[str, int]:
    names = sorted(typewells)
    resampled = {
        name: resample_typewell(*typewells[name]) for name in names
    }
    parent = list(range(len(names)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if find(left) == find(right):
                continue
            count = typewell_match_count(
                resampled[names[left]],
                resampled[names[right]],
                minimum_overlap=minimum_overlap,
            )
            if count >= int(minimum_match):
                parent[find(right)] = find(left)
    roots: dict[int, int] = {}
    return {
        name: roots.setdefault(find(index), len(roots))
        for index, name in enumerate(names)
    }


def ridge_prediction(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Small deterministic residual learner used by delayed-dip controls."""
    train_features = np.asarray(train_features, dtype=np.float64)
    train_target = np.asarray(train_target, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)
    center = np.mean(train_features, axis=0)
    scale = np.maximum(np.std(train_features, axis=0), 1e-8)
    train = (train_features - center) / scale
    test = (test_features - center) / scale
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(train.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        train.T @ train + penalty,
        train.T @ train_target,
    )
    return test @ coefficient
