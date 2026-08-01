from __future__ import annotations

import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from catboost import CatBoostRegressor
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F
from xgboost import XGBRegressor


COMPETITION_SLUG = "rogii-wellbore-geology-prediction"
WEIGHTS_SLUG = "rogii-compact-alignment-cv5-weights"
C016_WEIGHTS_SLUG = "rogii-v5-promoted-c016-weights"
RUN4_WEIGHTS_SLUG = "rogii-v5-run4-structural-surface-router"
OWNER = "jiweiliu"
NARROW_TEMPERATURE = 1.5
DISTRACTOR_TEMPERATURE = 1.5
WIDE_TEMPERATURE = 1.5
FORWARD_TEMPERATURE = 1.0
FORWARD_DISTRACTOR_TEMPERATURE = 1.0
RARE_TEMPERATURE = 1.25
SUPERVISED_TEMPERATURE = 1.0
WIDE_FORWARD_TEMPERATURE = 1.0
WIDE_SUPERVISED_TEMPERATURE = 1.0
NARROW_WEIGHT = 0.0397959184
DISTRACTOR_WEIGHT = 0.0265306122
WIDE_WEIGHT = 0.0530612245
FORWARD_WEIGHT = 0.0862244898
FORWARD_DISTRACTOR_WEIGHT = 0.0596938776
RARE_WEIGHT = 0.0408163265
SUPERVISED_WEIGHT = 0.0918367347
WIDE_FORWARD_WEIGHT = 0.1020408163
WIDE_SUPERVISED_WEIGHT = 0.50
WIDE_SUPERVISED_SEED_FRACTION = 0.10
WIDE_SUPERVISED_SMALL_FRACTION = 0.10
WIDE_SUPERVISED_STRATIFIED_FRACTION = 0.45
WIDE_SUPERVISED_GR25_TTA_FRACTION = 0.50
TAIL_EXPANSION = 0.06
SURFACE_SMOOTH_WINDOW = 1601
SURFACE_EXTRAPOLATION_ROWS = 1000
SURFACE_EXTRAPOLATION_LIMIT = 24.0
SURFACE_EXTRAPOLATION_WEIGHT = 0.03
ADAPTIVE_FAMILY_BONUS = 0.55
ADAPTIVE_EXCURSION_THRESHOLD = 40.0
ADAPTIVE_EXCURSION_WIDTH = 10.0
DISAGREEMENT_EXTRA_BONUS = 0.10
DISAGREEMENT_THRESHOLD = 0.15
DISAGREEMENT_WIDTH = 0.20
SPATIAL_ANALOG_WEIGHT = 0.05
SPATIAL_ANALOG_LIMIT = 1500.0
SPATIAL_LOCAL_WEIGHT = 0.075
SPATIAL_LOCAL_LIMIT = 600.0
SPATIAL_GEOMETRY_WEIGHT = 0.01
GR_CALIBRATION_WEIGHT = 0.45
ORDINAL_TEMPERATURE = 0.5
ORDINAL_WEIGHT = 0.15
ORDINAL_SMOOTH_SIGMA = 32.0
ANCC_WEIGHT = 0.01
RESIDUAL_GR_WEIGHT = 0.15
MASK_SENSITIVITY_WEIGHT = -0.05
MASK_VIEWS = 4
BASE12_RESIDUAL_WEIGHT = 0.05
BASE16_RESIDUAL_WEIGHT = 0.05
LONG_SYNTHETIC_RESIDUAL_WEIGHT = 0.05
# Conservative final-selection hedge: stop at the 6.189485 accepted stack.
BASE12_MASK_SENSITIVITY_WEIGHT = 0.0
FEATURE_SENSITIVITY_WEIGHT = 0.0
FEATURE_SHIFT = 0.1
LONG_BASE12_RESIDUAL_WEIGHT = 0.0
C016_LOCAL_SEEDS = (20260831, 20261043)
C016_LOCAL_TEMPERATURES = (2.0, 2.0, 1.5, 1.5, 1.5)
C016_LOCAL_WEIGHT = 0.20
C016_LOCAL_AMPLITUDES = (1.0, 1.0, 1.0, 1.0, 2.0)
C016_POSTERIOR_OUTER = 3
C016_POSTERIOR_TEMPERATURE = 1.0
C016_POSTERIOR_EFFECTIVE_SCALE = 0.20
C016_POSTERIOR_RIDGE_WEIGHT = 0.50
C016_FINAL_AMPLITUDE = 0.50
C016_RESIDUAL_OFFSETS = np.arange(-32.0, 33.0, 1.0, dtype=np.float32)
RUN4_CORRECTION_SCALE = 0.60
RUN4_SURFACE_CONFIGS = (
    {
        "name": "global_n10_bw1_d0_r138",
        "cluster_only": False,
        "neighbor_count": 10,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 0,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "global_n16_bw1_d500_r138",
        "cluster_only": False,
        "neighbor_count": 16,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 500,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "global_n24_bw2_d1000_r138",
        "cluster_only": False,
        "neighbor_count": 24,
        "bandwidth_multiplier": 2.0,
        "residual_decay_rows": 1000,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "global_n16_bw1_d0_r0",
        "cluster_only": False,
        "neighbor_count": 16,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 0,
        "azimuth_degrees": 0.0,
    },
    {
        "name": "cluster_n10_bw1_d0_r138",
        "cluster_only": True,
        "neighbor_count": 10,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 0,
        "azimuth_degrees": 138.0,
    },
    {
        "name": "cluster_n16_bw1_d500_r138",
        "cluster_only": True,
        "neighbor_count": 16,
        "bandwidth_multiplier": 1.0,
        "residual_decay_rows": 500,
        "azimuth_degrees": 138.0,
    },
)

if not np.isclose(
    NARROW_WEIGHT
    + DISTRACTOR_WEIGHT
    + WIDE_WEIGHT
    + FORWARD_WEIGHT
    + FORWARD_DISTRACTOR_WEIGHT
    + RARE_WEIGHT
    + SUPERVISED_WEIGHT
    + WIDE_FORWARD_WEIGHT
    + WIDE_SUPERVISED_WEIGHT,
    1.0,
):
    raise RuntimeError("ensemble weights must sum to one")


def first_existing(candidates: list[Path], marker: str) -> Path:
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    for match in Path("/kaggle/input").rglob(marker):
        return match.parent
    raise FileNotFoundError(
        f"Could not locate {marker}; checked: {[str(path) for path in candidates]}"
    )


def discover_data_root() -> Path:
    override = os.environ.get("ROGII_DATA_ROOT")
    candidates = [
        Path(override) if override else Path("/__missing_override__"),
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
        Path("/kaggle/input") / COMPETITION_SLUG,
    ]
    return first_existing(candidates, "sample_submission.csv")


def discover_weights_root() -> Path:
    override = os.environ.get("ROGII_WEIGHTS_ROOT")
    candidates = [
        Path(override) if override else Path("/__missing_override__"),
        Path("/kaggle/input/datasets") / OWNER / WEIGHTS_SLUG,
        Path("/kaggle/input") / WEIGHTS_SLUG,
    ]
    return first_existing(candidates, "alignment_fixed_synth_fold0.pt")


def discover_c016_weights_root() -> Path:
    override = os.environ.get("ROGII_C016_WEIGHTS_ROOT")
    candidates = [
        Path(override) if override else Path("/__missing_override__"),
        Path("/kaggle/input/datasets") / OWNER / C016_WEIGHTS_SLUG,
        Path("/kaggle/input") / C016_WEIGHTS_SLUG,
    ]
    return first_existing(candidates, "c016_manifest.json")


def discover_run4_weights_root() -> Path:
    override = os.environ.get("ROGII_RUN4_WEIGHTS_ROOT")
    candidates = [
        Path(override) if override else Path("/__missing_override__"),
        Path("/kaggle/input/datasets") / OWNER / RUN4_WEIGHTS_SLUG,
        Path("/kaggle/input") / RUN4_WEIGHTS_SLUG,
    ]
    return first_existing(candidates, "recipe.json")


@dataclass(frozen=True)
class SequenceConfig:
    prefix_points: int = 64
    tail_points: int = 640
    state_radius: float = 64.0
    state_step: float = 0.5
    row_stride: float = 16.0
    gr_smooth_window: int = 51

    @property
    def offsets(self) -> np.ndarray:
        return np.arange(
            -self.state_radius,
            self.state_radius + self.state_step / 2,
            self.state_step,
            dtype=np.float64,
        )

    @property
    def length(self) -> int:
        return self.prefix_points + self.tail_points


@dataclass(frozen=True)
class Well:
    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray

    @property
    def known_indices(self) -> np.ndarray:
        return np.flatnonzero(np.isfinite(self.tvt_input))

    @property
    def prediction_indices(self) -> np.ndarray:
        return np.flatnonzero(~np.isfinite(self.tvt_input))

    @property
    def anchor_index(self) -> int:
        known = self.known_indices
        if not len(known):
            raise ValueError(f"{self.well_id}: no visible TVT_input prefix")
        return int(known[-1])


@dataclass
class SpatialReferenceStore:
    well_ids: np.ndarray
    family: np.ndarray
    folds: np.ndarray
    grid: np.ndarray
    typewell_values: np.ndarray
    typewell_mask: np.ndarray
    reference_xy: list[np.ndarray]
    reference_surface: list[np.ndarray]
    reference_trees: list[cKDTree]
    family_indices: dict[int, np.ndarray]
    family_trees: dict[int, tuple[cKDTree, np.ndarray]]
    id_to_index: dict[str, int]


@dataclass
class AnccReferenceStore:
    well_ids: np.ndarray
    xy: list[np.ndarray]
    ancc: list[np.ndarray]
    trees: list[cKDTree]
    global_tree: cKDTree
    point_well: np.ndarray


HORIZONTAL_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_COLUMNS = ("TVT", "GR")


def read_required_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return pd.read_csv(path, usecols=list(columns))


def test_well_ids(data_root: Path) -> list[str]:
    return sorted(
        path.name.split("__", 1)[0]
        for path in (data_root / "test").glob("*__horizontal_well.csv")
    )


def load_test_well(well_id: str, data_root: Path) -> Well:
    horizontal = read_required_columns(
        data_root / "test" / f"{well_id}__horizontal_well.csv",
        HORIZONTAL_COLUMNS,
    )
    typewell = read_required_columns(
        data_root / "test" / f"{well_id}__typewell.csv",
        TYPEWELL_COLUMNS,
    ).sort_values("TVT")
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise ValueError(f"{well_id}: no visible TVT_input prefix")
    tvt = np.where(np.isfinite(tvt_input), tvt_input, tvt_input[known[-1]])
    return Well(
        well_id=well_id,
        md=horizontal["MD"].to_numpy(np.float64),
        x=horizontal["X"].to_numpy(np.float64),
        y=horizontal["Y"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=horizontal["GR"].to_numpy(np.float64),
        tvt=tvt,
        tvt_input=tvt_input,
        typewell_tvt=typewell["TVT"].to_numpy(np.float64),
        typewell_gr=typewell["GR"].to_numpy(np.float64),
    )


def load_spatial_reference(weights_root: Path) -> SpatialReferenceStore:
    with np.load(weights_root / "spatial_reference.npz") as cache:
        well_ids = cache["well_ids"].astype(str)
        family = cache["family"].astype(np.int64)
        folds = cache["folds"].astype(np.int64)
        grid = cache["grid"].astype(np.float64)
        typewell_values = cache["typewell_values"].astype(np.float64)
        typewell_mask = cache["typewell_mask"].astype(np.float64)
        starts = cache["reference_starts"].astype(np.int64)
        flat_xy = cache["reference_xy"].astype(np.float64)
        flat_surface = cache["reference_surface"].astype(np.float64)
    reference_xy = [
        flat_xy[left:right] for left, right in zip(starts[:-1], starts[1:])
    ]
    reference_surface = [
        flat_surface[left:right] for left, right in zip(starts[:-1], starts[1:])
    ]
    reference_trees = [cKDTree(xy) for xy in reference_xy]
    family_indices = {
        int(value): np.flatnonzero(family == value) for value in np.unique(family)
    }
    family_trees = {}
    for value, indices in family_indices.items():
        xy = np.concatenate([reference_xy[index] for index in indices])
        surface = np.concatenate([reference_surface[index] for index in indices])
        family_trees[value] = (cKDTree(xy), surface)
    return SpatialReferenceStore(
        well_ids=well_ids,
        family=family,
        folds=folds,
        grid=grid,
        typewell_values=typewell_values,
        typewell_mask=typewell_mask,
        reference_xy=reference_xy,
        reference_surface=reference_surface,
        reference_trees=reference_trees,
        family_indices=family_indices,
        family_trees=family_trees,
        id_to_index={well_id: index for index, well_id in enumerate(well_ids)},
    )


def load_ancc_reference(weights_root: Path) -> AnccReferenceStore:
    with np.load(weights_root / "ancc_reference.npz") as cache:
        well_ids = cache["well_ids"].astype(str)
        starts = cache["starts"].astype(np.int64)
        flat_xy = cache["xy"].astype(np.float64)
        flat_ancc = cache["ancc"].astype(np.float64)
    xy = [flat_xy[left:right] for left, right in zip(starts[:-1], starts[1:])]
    ancc = [flat_ancc[left:right] for left, right in zip(starts[:-1], starts[1:])]
    point_well = np.concatenate(
        [np.full(len(value), index, dtype=np.int32) for index, value in enumerate(xy)]
    )
    return AnccReferenceStore(
        well_ids=well_ids,
        xy=xy,
        ancc=ancc,
        trees=[cKDTree(value) for value in xy],
        global_tree=cKDTree(flat_xy),
        point_well=point_well,
    )


def infer_typewell_family(well: Well, store: SpatialReferenceStore) -> int:
    reference_index = store.id_to_index.get(well.well_id)
    if reference_index is not None:
        return int(store.family[reference_index])
    supported = (store.grid >= np.nanmin(well.typewell_tvt)) & (
        store.grid <= np.nanmax(well.typewell_tvt)
    )
    observed = np.zeros(len(store.grid), dtype=np.float64)
    observed[supported] = np.interp(
        store.grid[supported], well.typewell_tvt, well.typewell_gr
    )
    count = store.typewell_mask @ supported.astype(np.float64)
    sum_x = store.typewell_values @ supported.astype(np.float64)
    sum_x2 = np.square(store.typewell_values) @ supported.astype(np.float64)
    sum_y = store.typewell_mask @ observed
    sum_y2 = store.typewell_mask @ np.square(observed)
    dot = store.typewell_values @ observed
    denominator_count = np.maximum(count, 1.0)
    covariance = dot - sum_x * sum_y / denominator_count
    variance_x = sum_x2 - np.square(sum_x) / denominator_count
    variance_y = sum_y2 - np.square(sum_y) / denominator_count
    correlation = covariance / np.sqrt(
        np.maximum(variance_x * variance_y, 1e-12)
    )
    correlation[count < 200] = -2.0
    best = int(np.argmax(correlation))
    return int(store.family[best]) if correlation[best] >= 0.99 else -1


def spatial_transfer_inputs(
    well: Well,
    baseline: np.ndarray,
    store: SpatialReferenceStore,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    family = infer_typewell_family(well, store)
    tail = well.prediction_indices
    anchor = well.anchor_index
    query_indices = np.unique(
        np.r_[np.arange(0, len(well.md), 16), len(well.md) - 1, anchor]
    )
    query_indices.sort()
    query_xy = np.column_stack((well.x[query_indices], well.y[query_indices]))
    visible = query_indices <= anchor
    recent_visible = np.flatnonzero(visible)[-min(64, int(visible.sum())) :]
    finite_visible = np.isfinite(well.tvt_input[query_indices[recent_visible]])
    hidden_sample = query_indices > anchor
    reference_index = store.id_to_index.get(well.well_id)
    excluded_fold = (
        int(store.folds[reference_index]) if reference_index is not None else None
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for index in store.family_indices.get(family, np.empty(0, dtype=np.int64)):
        if excluded_fold is not None and store.folds[index] == excluded_fold:
            continue
        distance, nearest = store.reference_trees[index].query(query_xy, k=1)
        score = (
            float(np.median(distance[hidden_sample]))
            if np.any(hidden_sample)
            else float("inf")
        )
        candidates.append((score, distance, store.reference_surface[index][nearest]))
    candidates.sort(key=lambda item: item[0])

    if candidates:
        top = candidates[: min(3, len(candidates))]
        distance = np.stack([item[1] for item in top])
        surface = np.stack([item[2] for item in top])
        weight = 1.0 / np.square(np.maximum(distance, 25.0))
        analog_surface = np.sum(weight * surface, axis=0) / np.sum(weight, axis=0)
        visible_surface = (
            well.tvt_input[query_indices[recent_visible]][finite_visible]
            + well.z[query_indices[recent_visible]][finite_visible]
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

    selected = store.family_indices.get(family, np.empty(0, dtype=np.int64))
    if excluded_fold is not None:
        selected = selected[store.folds[selected] != excluded_fold]
    if len(selected):
        if excluded_fold is None:
            tree, surface = store.family_trees[family]
        else:
            xy = np.concatenate([store.reference_xy[index] for index in selected])
            surface = np.concatenate(
                [store.reference_surface[index] for index in selected]
            )
            tree = cKDTree(xy)
        distance, nearest = tree.query(query_xy, k=min(32, len(surface)))
        distance = np.asarray(distance)
        nearest = np.asarray(nearest)
        if distance.ndim == 1:
            distance = distance[:, None]
            nearest = nearest[:, None]
        local_weight = 1.0 / np.square(np.maximum(distance, 10.0))
        local_surface = np.sum(local_weight * surface[nearest], axis=1) / np.sum(
            local_weight, axis=1
        )
        visible_surface = (
            well.tvt_input[query_indices[recent_visible]][finite_visible]
            + well.z[query_indices[recent_visible]][finite_visible]
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
    return analog, local, analog_distance, local_distance, family


def spatial_feature_vector(
    well: Well,
    baseline: np.ndarray,
    analog: np.ndarray,
    local: np.ndarray,
    delta: np.ndarray,
    analog_distance: float,
    local_distance: float,
    family_size: int,
) -> np.ndarray:
    tail = well.prediction_indices
    anchor = well.anchor_index
    anchor_tvt = float(well.tvt_input[anchor])
    visible_surface = well.tvt_input[: anchor + 1] + well.z[: anchor + 1]

    def slope(rows: int) -> float:
        x = well.md[: anchor + 1][-rows:]
        y = visible_surface[-rows:]
        centered_x = x - np.mean(x)
        return float(centered_x @ (y - np.mean(y))) / max(
            float(centered_x @ centered_x), 1e-9
        )

    finite_gr = np.isfinite(well.gr)
    if np.any(finite_gr):
        gr = np.interp(np.arange(len(well.gr)), np.flatnonzero(finite_gr), well.gr[finite_gr])
        prefix_gr = gr[: anchor + 1]
        gr_scale = max(
            1.4826 * float(np.median(np.abs(prefix_gr - np.median(prefix_gr)))),
            3.0,
        )
    else:
        gr_scale = 20.0
    dx = float(well.x[tail[-1]] - well.x[anchor])
    dy = float(well.y[tail[-1]] - well.y[anchor])
    horizontal_distance = max(float(np.hypot(dx, dy)), 1.0)
    energy = float(delta @ delta)
    return np.asarray(
        [
            np.log1p(min(analog_distance, 1e5)),
            np.log1p(min(local_distance, 1e5)),
            np.log1p(len(tail)),
            (well.md[tail[-1]] - well.md[anchor]) / 10000.0,
            anchor / (len(well.md) - 1),
            np.sqrt(np.mean(np.square(baseline - anchor_tvt))) / 50.0,
            (baseline[-1] - anchor_tvt) / 50.0,
            np.sqrt(np.mean(np.square(analog - baseline))) / 50.0,
            np.sqrt(np.mean(np.square(local - baseline))) / 50.0,
            np.sqrt(np.mean(np.square(analog - local))) / 50.0,
            np.sqrt(energy / max(len(tail), 1)) / 10.0,
            delta[-1] / 10.0,
            np.max(np.abs(delta)) / 10.0,
            (well.z[tail[-1]] - well.z[anchor]) / 100.0,
            dx / 10000.0,
            dy / 10000.0,
            slope(128) * 100.0,
            slope(500) * 100.0,
            slope(1000) * 100.0,
            gr_scale / 20.0,
            np.mean(np.abs(np.diff(delta))) / 2.0,
            (analog[-1] - baseline[-1]) / 50.0,
            (local[-1] - baseline[-1]) / 50.0,
            np.mean(np.abs(np.diff(analog + well.z[tail]))) / 5.0,
            np.mean(np.abs(np.diff(local + well.z[tail]))) / 5.0,
            well.x[anchor] / 1e6,
            well.y[anchor] / 1e6,
            well.z[anchor] / 1e4,
            dx / horizontal_distance,
            dy / horizontal_distance,
            np.log1p(family_size),
        ],
        dtype=np.float64,
    )


def spatially_correct_prediction(
    well: Well,
    baseline: np.ndarray,
    store: SpatialReferenceStore,
    confidence_models: list[CatBoostRegressor],
) -> tuple[np.ndarray, float, int]:
    analog, local, analog_distance, local_distance, family = spatial_transfer_inputs(
        well, baseline, store
    )
    delta = np.zeros_like(baseline)
    if analog_distance <= SPATIAL_ANALOG_LIMIT:
        delta += SPATIAL_ANALOG_WEIGHT * (analog - baseline)
    if local_distance <= SPATIAL_LOCAL_LIMIT:
        delta += SPATIAL_LOCAL_WEIGHT * (local - baseline)
    features = spatial_feature_vector(
        well,
        baseline,
        analog,
        local,
        delta,
        analog_distance,
        local_distance,
        len(store.family_indices.get(family, ())),
    )
    design = np.column_stack((features[None, :], np.asarray([str(family)])))
    reference_index = store.id_to_index.get(well.well_id)
    if reference_index is None:
        multiplier = float(
            np.mean([model.predict(design)[0] for model in confidence_models])
        )
    else:
        held_out = int(store.folds[reference_index])
        multiplier = float(confidence_models[held_out].predict(design)[0])
    multiplier = float(np.clip(multiplier, 0.0, 2.0))
    spatial = baseline + multiplier * delta

    left = max(0, well.anchor_index - 7)
    prefix_surface = (
        well.tvt_input[left : well.anchor_index + 1]
        + well.z[left : well.anchor_index + 1]
    )
    increment = float(np.median(np.diff(prefix_surface)))
    tail = well.prediction_indices
    geometric = (
        prefix_surface[-1]
        + increment * np.arange(1, len(tail) + 1, dtype=np.float64)
        - well.z[tail]
    )
    spatial += SPATIAL_GEOMETRY_WEIGHT * (geometric - baseline)
    return spatial, multiplier, family


def smooth_gr(values: np.ndarray, requested_window: int = 51) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values, dtype=np.float64)
    rows = np.arange(len(values))
    filled = np.interp(rows, rows[finite], values[finite])
    window = min(
        requested_window,
        len(filled) if len(filled) % 2 else len(filled) - 1,
    )
    if window % 2 == 0:
        window -= 1
    return savgol_filter(filled, window, 2) if window >= 5 else filled


def ancc_surface_correction(
    well: Well,
    baseline: np.ndarray,
    store: AnccReferenceStore,
    query_stride: int = 12,
    neighbor_count: int = 8,
    candidate_points: int = 10000,
    distance_floor: float = 50.0,
) -> np.ndarray:
    anchor_xy = np.asarray([well.x[well.anchor_index], well.y[well.anchor_index]])
    _, point_indices = store.global_tree.query(
        anchor_xy, k=min(candidate_points, len(store.point_well))
    )
    selected = []
    seen = set()
    for point_index in np.atleast_1d(point_indices):
        well_index = int(store.point_well[point_index])
        if well_index not in seen:
            seen.add(well_index)
            selected.append(well_index)
        if len(selected) == neighbor_count:
            break
    if not selected:
        raise RuntimeError(f"{well.well_id}: failed to find an ANCC reference")

    tail = well.prediction_indices
    query = np.unique(
        np.r_[well.anchor_index, tail[::query_stride], tail[-1]]
    ).astype(int)
    query_xy = np.column_stack((well.x[query], well.y[query]))
    curves = []
    distances = []
    for reference_index in selected:
        distance, row = store.trees[reference_index].query(query_xy, k=1)
        curves.append(store.ancc[reference_index][row])
        distances.append(distance)
    curves_array = np.asarray(curves)
    distances_array = np.asarray(distances)
    weight = 1.0 / np.square(np.maximum(distances_array, distance_floor))
    ancc = np.sum(weight * curves_array, axis=0) / np.sum(weight, axis=0)
    anchor_surface = float(
        well.tvt_input[well.anchor_index] + well.z[well.anchor_index]
    )
    candidate_surface = anchor_surface + ancc - ancc[0]
    candidate = np.interp(tail, query, candidate_surface) - well.z[tail]
    return candidate - baseline


def residual_robust_affine(
    reference: np.ndarray, observed: np.ndarray
) -> tuple[float, float, float]:
    mask = np.isfinite(reference) & np.isfinite(observed)
    x = reference[mask]
    y = observed[mask]
    if len(x) < 20 or np.std(x) < 1e-6:
        return 1.0, 0.0, 15.0
    design = np.column_stack([x, np.ones_like(x)])
    weights = np.ones(len(x), dtype=np.float64)
    coefficient = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(5):
        root = np.sqrt(weights)
        coefficient = np.linalg.lstsq(
            design * root[:, None], y * root, rcond=None
        )[0]
        coefficient[0] = np.clip(coefficient[0], 0.2, 3.0)
        residual = y - design @ coefficient
        scale = max(
            1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0
        )
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-6))
    residual = y - design @ coefficient
    scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
    return float(coefficient[0]), float(coefficient[1]), float(scale)


def residual_viterbi(
    cost: np.ndarray, anchor: int, max_jump: int = 2, transition: float = 0.5
) -> np.ndarray:
    time_count, state_count = cost.shape
    previous = np.full(state_count, 1e30, dtype=np.float64)
    previous[anchor] = 0.0
    back = np.empty((time_count, state_count), dtype=np.int16)
    back[0] = anchor
    for time_index in range(1, time_count):
        current = np.empty(state_count, dtype=np.float64)
        for state in range(state_count):
            lower = max(0, state - max_jump)
            upper = min(state_count, state + max_jump + 1)
            old_states = np.arange(lower, upper)
            difference = state - old_states
            values = previous[lower:upper] + transition * np.square(difference)
            best = int(np.argmin(values))
            current[state] = values[best] + cost[time_index, state]
            back[time_index, state] = old_states[best]
        previous = current
    path = np.empty(time_count, dtype=np.int16)
    path[-1] = np.argmin(previous)
    for time_index in range(time_count - 1, 0, -1):
        path[time_index - 1] = back[time_index, path[time_index]]
    path[0] = anchor
    return path


def residual_gr_correction(well: Well, baseline: np.ndarray) -> np.ndarray:
    tail = well.prediction_indices
    query = np.unique(np.r_[well.anchor_index, tail[::12], tail[-1]]).astype(int)
    baseline_query = np.r_[
        well.tvt_input[well.anchor_index], np.interp(query[1:], tail, baseline)
    ]
    known = well.known_indices
    reference = np.interp(
        well.tvt_input[known],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, offset, residual_scale = residual_robust_affine(reference, well.gr[known])
    offsets = np.arange(-24.0, 24.5, 1.0)
    candidate_tvt = baseline_query[:, None] + offsets[None]
    expected = gain * np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape) + offset
    observed = smooth_gr(well.gr, 49)[query, None]
    standardized = (observed - expected) / max(residual_scale, 1e-6)
    cost = np.minimum(np.square(standardized), 9.0)
    cost[~np.isfinite(expected)] = 36.0
    cost += 0.01 * np.square(offsets[None])
    cost[0] = 36.0
    anchor = int(np.argmin(np.abs(offsets)))
    cost[0, anchor] = 0.0
    path = residual_viterbi(cost.astype(np.float64), anchor)
    return np.interp(tail, query, offsets[path])


def gr_path_correction(well: Well, prediction: np.ndarray) -> np.ndarray:
    tail = well.prediction_indices
    sampled = np.unique(np.r_[np.arange(0, len(tail), 16), len(tail) - 1])
    observed = smooth_gr(well.gr[tail])[sampled]
    observed = (observed - np.mean(observed)) / max(float(np.std(observed)), 3.0)
    progress = np.linspace(-1.0, 1.0, len(tail))[sampled]
    biases = np.arange(-12.0, 12.5, 1.0)
    slopes = np.arange(-18.0, 18.5, 1.0)
    bias_grid, slope_grid = np.meshgrid(biases, slopes, indexing="ij")
    candidate_bias = bias_grid.ravel()
    candidate_slope = slope_grid.ravel()
    candidate_tvt = (
        prediction[sampled, None]
        + candidate_bias[None, :]
        + progress[:, None] * candidate_slope[None, :]
    )
    reference = np.interp(
        candidate_tvt.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(candidate_tvt.shape)
    supported = np.isfinite(reference)
    count = np.sum(supported, axis=0)
    mean = np.nansum(reference, axis=0) / np.maximum(count, 1)
    centered = np.where(supported, reference - mean, 0.0)
    denominator = np.sqrt(
        np.sum(np.square(centered), axis=0) * np.sum(np.square(observed))
    )
    correlation = np.sum(centered * observed[:, None], axis=0) / np.maximum(
        denominator, 1e-9
    )
    correlation[count < 0.8 * len(sampled)] = -2.0
    utility = correlation - 0.01 * (
        np.square(candidate_bias) + np.square(candidate_slope)
    )
    best = int(np.argmax(utility))
    return candidate_bias[best] + candidate_slope[best] * np.linspace(
        -1.0, 1.0, len(tail)
    )


def robust_affine(
    reference: np.ndarray, observed: np.ndarray
) -> tuple[float, float, float]:
    mask = np.isfinite(reference) & np.isfinite(observed)
    x = reference[mask]
    y = observed[mask]
    if len(x) < 20 or np.std(x) < 1e-6:
        return 1.0, 0.0, 15.0
    design = np.column_stack([x, np.ones_like(x)])
    weights = np.ones(len(x), dtype=np.float64)
    coefficient = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(4):
        root = np.sqrt(weights)
        coefficient = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        coefficient[0] = np.clip(coefficient[0], 0.2, 3.0)
        residual = y - design @ coefficient
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-6))
    residual = y - design @ coefficient
    scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 3.0)
    return float(coefficient[0]), float(coefficient[1]), float(scale)


def sequence_positions(well: Well, cut: int, config: SequenceConfig) -> np.ndarray:
    prefix = cut - config.row_stride * np.arange(
        config.prefix_points - 1, -1, -1, dtype=np.float64
    )
    tail = cut + 1 + config.row_stride * np.arange(config.tail_points, dtype=np.float64)
    return np.concatenate(
        [np.clip(prefix, 0, len(well.md) - 1), np.clip(tail, 0, len(well.md) - 1)]
    )


def make_sequence_example(
    well: Well, cut: int, config: SequenceConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    floating_positions = sequence_positions(well, cut, config)
    raw_prefix = cut - config.row_stride * np.arange(
        config.prefix_points - 1, -1, -1, dtype=np.float64
    )
    raw_tail = (
        cut + 1 + config.row_stride * np.arange(config.tail_points, dtype=np.float64)
    )
    valid_positions = np.concatenate(
        [raw_prefix >= 0, raw_tail <= len(well.md) - 1]
    ).astype(np.float32)
    source_positions = np.arange(len(well.md), dtype=np.float64)

    def sample(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return np.full(config.length, np.nan, dtype=np.float64)
        return np.interp(floating_positions, source_positions[finite], values[finite])

    md = sample(well.md)
    x = sample(well.x)
    y = sample(well.y)
    z = sample(well.z)
    finite_gr = np.isfinite(well.gr)
    if finite_gr.sum() >= 2:
        gr_filled = np.interp(
            source_positions, source_positions[finite_gr], well.gr[finite_gr]
        )
        requested_window = max(1, config.gr_smooth_window)
        if requested_window % 2 == 0:
            requested_window -= 1
        smooth_window = min(
            requested_window,
            len(gr_filled) if len(gr_filled) % 2 else len(gr_filled) - 1,
        )
        gr_smoothed = (
            savgol_filter(gr_filled, smooth_window, 2)
            if smooth_window >= 5
            else gr_filled
        )
        gr = np.interp(floating_positions, source_positions, gr_smoothed)
    else:
        gr = sample(well.gr)
    tvt = sample(well.tvt)
    anchor_tvt = float(well.tvt[cut])
    anchor_md = float(well.md[cut])
    anchor_x = float(well.x[cut])
    anchor_y = float(well.y[cut])
    anchor_z = float(well.z[cut])

    prefix_rows = np.arange(cut + 1)
    reference_prefix = np.interp(
        well.tvt[prefix_rows],
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, offset, residual_scale = robust_affine(reference_prefix, well.gr[prefix_rows])
    state_tvt = anchor_tvt + config.offsets
    state_gr = (
        gain
        * np.interp(
            state_tvt,
            well.typewell_tvt,
            well.typewell_gr,
            left=np.nan,
            right=np.nan,
        )
        + offset
    )
    mismatch = (gr[None, :] - state_gr[:, None]) / max(residual_scale, 3.0)
    mismatch = np.nan_to_num(mismatch, nan=0.0, posinf=0.0, neginf=0.0)
    mismatch = np.clip(mismatch, -6.0, 6.0).astype(np.float32)
    visible = np.zeros(config.length, dtype=np.float32)
    visible[: config.prefix_points] = 1.0
    missing_gr = (~np.isfinite(gr)).astype(np.float32)
    gr_normalized = np.nan_to_num((gr - offset) / max(residual_scale, 3.0))
    gr_normalized = np.clip(gr_normalized, -6.0, 6.0).astype(np.float32)
    visible_tvt = np.where(visible > 0, (tvt - anchor_tvt) / 40.0, 0.0)
    context = np.stack(
        [
            gr_normalized,
            missing_gr,
            visible,
            visible_tvt.astype(np.float32),
            ((md - anchor_md) / 5000.0).astype(np.float32),
            ((x - anchor_x) / 5000.0).astype(np.float32),
            ((y - anchor_y) / 5000.0).astype(np.float32),
            ((z - anchor_z) / 200.0).astype(np.float32),
        ],
        axis=0,
    )
    inputs = np.concatenate([mismatch, context], axis=0).astype(np.float32)
    target = ((tvt - anchor_tvt) / 40.0).astype(np.float32)
    metadata: dict[str, np.ndarray | float] = {
        "positions": floating_positions,
        "valid_positions": valid_positions,
        "md": md,
        "x": x,
        "y": y,
        "z": z,
        "gr": gr,
        "anchor_tvt": anchor_tvt,
    }
    return inputs, target, metadata


def make_alignment_example(
    well: Well, cut: int, config: SequenceConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    sequence, target, metadata = make_sequence_example(well, cut, config)
    n_states = len(config.offsets)
    mismatch = sequence[:n_states]
    visible = sequence[n_states + 2]
    missing = sequence[n_states + 1]
    z_delta = sequence[n_states + 7]
    sampled_gr = np.asarray(metadata["gr"])
    sampled_md = np.asarray(metadata["md"])
    sampled_x = np.asarray(metadata["x"])
    sampled_y = np.asarray(metadata["y"])
    sampled_z = np.asarray(metadata["z"])
    valid_positions = np.asarray(metadata["valid_positions"])
    missing = np.maximum(missing, 1.0 - valid_positions)

    prefix_tvt = well.tvt[: cut + 1]
    prefix_gr = well.gr[: cut + 1]
    finite_prefix = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    prefix_tvt = prefix_tvt[finite_prefix]
    prefix_gr = prefix_gr[finite_prefix]
    distance_to_prefix = (
        (well.tvt[cut] + config.offsets)[:, None] - prefix_tvt[None, :]
    ) / 0.5
    pseudo_weight = np.exp(-0.5 * np.square(distance_to_prefix))
    pseudo_support = pseudo_weight.sum(axis=1)
    pseudo_signal = (pseudo_weight @ prefix_gr) / np.maximum(pseudo_support, 1e-12)
    pseudo_mismatch = (sampled_gr[None, :] - pseudo_signal[:, None]) / 10.0
    supported = (pseudo_support > 0.05).astype(np.float32)
    pseudo_mismatch = np.where(supported[:, None] > 0, pseudo_mismatch, mismatch)
    pseudo_mismatch = np.nan_to_num(pseudo_mismatch, nan=0.0)
    pseudo_mismatch = np.clip(pseudo_mismatch, -6.0, 6.0)

    true_offset = target * 40.0
    distance = config.offsets[:, None] - true_offset[None, :]
    labels = np.argmin(np.abs(distance), axis=0).astype(np.int64)
    visible_path = np.exp(-0.5 * np.square(distance / 3.0))
    visible_path[:, visible < 0.5] = 0.0
    state_coordinate = np.broadcast_to(
        (config.offsets / config.state_radius)[:, None], mismatch.shape
    )
    dmd = np.gradient(sampled_md)
    safe_dmd = np.where(np.abs(dmd) < 1e-6, 1.0, dmd)
    dx_dmd = np.gradient(sampled_x) / safe_dmd
    dy_dmd = np.gradient(sampled_y) / safe_dmd
    dz_dmd = np.gradient(sampled_z) / safe_dmd
    horizontal = np.sqrt(np.square(dx_dmd) + np.square(dy_dmd))
    azimuth_cos = dx_dmd / np.maximum(horizontal, 1e-4)
    azimuth_sin = dy_dmd / np.maximum(horizontal, 1e-4)
    curvature_z = np.gradient(dz_dmd) / safe_dmd

    def row_channel(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[None, :], mismatch.shape)

    anchor = config.prefix_points - 1
    image = np.stack(
        [
            mismatch,
            np.abs(mismatch),
            pseudo_mismatch,
            np.abs(pseudo_mismatch),
            np.broadcast_to(supported[:, None], mismatch.shape),
            state_coordinate,
            visible_path,
            np.broadcast_to(visible[None, :], mismatch.shape),
            np.broadcast_to(missing[None, :], mismatch.shape),
            np.broadcast_to(z_delta[None, :], mismatch.shape),
            row_channel((sampled_md - sampled_md[anchor]) / 5000.0),
            row_channel((sampled_x - sampled_x[anchor]) / 5000.0),
            row_channel((sampled_y - sampled_y[anchor]) / 5000.0),
            row_channel(np.clip(dz_dmd, -1.5, 1.5)),
            row_channel(np.clip(dx_dmd, -1.5, 1.5)),
            row_channel(np.clip(dy_dmd, -1.5, 1.5)),
            row_channel(np.nan_to_num(azimuth_cos)),
            row_channel(np.nan_to_num(azimuth_sin)),
            row_channel(np.clip(curvature_z * 100.0, -2.0, 2.0)),
        ],
        axis=0,
    ).astype(np.float32)
    return image, labels, metadata


def deterministic_mask(
    well_id: str, view: int, length: int, prefix_points: int
) -> np.ndarray:
    seed = int(well_id, 16) ^ (0x9E3779B9 * (view + 1))
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    mask = np.zeros(length, dtype=bool)
    for _ in range(2):
        width = int(rng.integers(8, 33))
        start = int(rng.integers(prefix_points, max(prefix_points + 1, length - width)))
        mask[start : start + width] = True
    return mask


def masked_alignment_views(
    well_id: str, image: np.ndarray, prefix_points: int
) -> np.ndarray:
    images = [image]
    for view in range(MASK_VIEWS):
        augmented = image.copy()
        mask = deterministic_mask(well_id, view, image.shape[-1], prefix_points)
        augmented[:4, :, mask] = 0.0
        augmented[8, :, mask] = 1.0
        images.append(augmented)
    return np.stack(images)


def shifted_alignment_view(image: np.ndarray) -> np.ndarray:
    augmented = image.copy()
    usable = (augmented[8] < 0.5).astype(np.float32)
    for signed, absolute in ((0, 1), (2, 3)):
        values = (augmented[signed] + FEATURE_SHIFT) * usable
        augmented[signed] = values
        augmented[absolute] = np.abs(values)
    return augmented


class AlignmentBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, (3, 7), padding=(1, 3))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 7), padding=(1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class AlignmentUNet(nn.Module):
    def __init__(self, input_channels: int = 19, base: int = 8) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 6, base * 8, base * 8]
        self.stem = nn.Conv2d(input_channels, widths[0], (5, 9), padding=(2, 4))
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for left, right in zip(widths[:-1], widths[1:]):
            self.encoder.append(
                nn.Sequential(AlignmentBlock(left), AlignmentBlock(left))
            )
            self.down.append(nn.Conv2d(left, right, 4, stride=2, padding=1))
        self.middle = nn.Sequential(
            AlignmentBlock(widths[-1]), AlignmentBlock(widths[-1])
        )
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for left, right in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(nn.ConvTranspose2d(left, right, 4, stride=2, padding=1))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(right * 2, right, 3, padding=1),
                    AlignmentBlock(right),
                )
            )
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for block, down in zip(self.encoder, self.down):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)


def c016_resample_state_volume(
    image: np.ndarray,
    prior_offset: np.ndarray,
    source_offsets: np.ndarray,
) -> np.ndarray:
    coordinates = (
        prior_offset[:, None]
        + C016_RESIDUAL_OFFSETS[None]
        - float(source_offsets[0])
    ) / float(source_offsets[1] - source_offsets[0])
    lower = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - lower
    supported = (lower >= 0) & (lower + 1 < len(source_offsets))
    lower = np.clip(lower, 0, len(source_offsets) - 2)
    transposed = image.transpose(0, 2, 1)
    left = np.take_along_axis(transposed, lower[None], axis=2)
    right = np.take_along_axis(transposed, (lower + 1)[None], axis=2)
    output = left * (1.0 - fraction[None]) + right * fraction[None]
    output *= supported[None]
    return output.transpose(0, 2, 1).astype(np.float32)


def c016_residual_image(
    well: Well,
    baseline: np.ndarray,
    components: np.ndarray,
    wide_image: np.ndarray,
    wide_metadata: dict[str, np.ndarray | float],
    config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(wide_metadata["positions"], dtype=np.float64)
    valid = np.asarray(wide_metadata["valid_positions"], dtype=np.float32)
    sampled_z = np.asarray(wide_metadata["z"], dtype=np.float32)
    anchor_tvt = float(wide_metadata["anchor_tvt"])
    source = np.arange(len(well.tvt_input), dtype=np.float64)
    visible_values = np.nan_to_num(
        well.tvt_input,
        nan=float(well.tvt_input[well.anchor_index]),
    )
    visible_prior = np.interp(
        np.minimum(positions, float(well.anchor_index)),
        source[: well.anchor_index + 1],
        visible_values[: well.anchor_index + 1],
    )
    tail_prior = np.interp(
        positions,
        well.prediction_indices.astype(np.float64),
        baseline.astype(np.float64),
    )
    prior_tvt = np.where(
        positions <= well.anchor_index + 1e-6,
        visible_prior,
        tail_prior,
    )
    prior_tvt = np.where(
        valid > 0.5,
        prior_tvt,
        float(well.tvt_input[well.anchor_index]),
    ).astype(np.float32)
    prior_offset = prior_tvt - anchor_tvt
    volume = c016_resample_state_volume(
        wide_image, prior_offset, config.offsets
    )
    height, width = volume.shape[1:]

    def plane(values: np.ndarray) -> np.ndarray:
        return np.broadcast_to(values[None], (height, width))

    prior_surface = prior_tvt + sampled_z
    prior_surface_slope = np.gradient(prior_surface) / 20.0
    disagreement_native = np.std(
        components.astype(np.float64), axis=0
    )
    disagreement = np.interp(
        positions,
        well.prediction_indices.astype(np.float64),
        disagreement_native,
        left=0.0,
        right=float(disagreement_native[-1]),
    )
    additions = np.stack(
        [
            np.broadcast_to(
                (
                    C016_RESIDUAL_OFFSETS
                    / max(
                        float(np.max(np.abs(C016_RESIDUAL_OFFSETS))),
                        1.0,
                    )
                )[:, None],
                (height, width),
            ),
            plane(prior_offset / 120.0),
            plane(prior_surface_slope),
            plane(disagreement / 40.0),
        ]
    ).astype(np.float32)
    output = np.concatenate((volume, additions), axis=0)
    if not np.isfinite(output).all():
        raise RuntimeError(f"{well.well_id}: nonfinite C016 residual image")
    return output, positions, prior_tvt, valid


@torch.no_grad()
def c016_local_prediction(
    model: AlignmentUNet,
    image: np.ndarray,
    positions: np.ndarray,
    baseline: np.ndarray,
    well: Well,
    temperature: float,
    config: SequenceConfig,
    device: torch.device,
) -> np.ndarray:
    logits = model(torch.from_numpy(image)[None].to(device))[0]
    offsets = torch.as_tensor(
        C016_RESIDUAL_OFFSETS,
        device=device,
        dtype=torch.float32,
    )
    correction = torch.sum(
        torch.softmax(logits.float() / temperature, dim=0)
        * offsets[:, None],
        dim=0,
    ).cpu().numpy()
    selected = np.isfinite(positions) & (
        np.arange(len(positions)) >= config.prefix_points
    )
    native_correction = np.interp(
        well.prediction_indices,
        positions[selected],
        correction[selected],
    )
    return (
        baseline.astype(np.float64) + native_correction
    ).astype(np.float32)


def c016_residual_probability(
    probability: np.ndarray,
    source_offsets: np.ndarray,
    candidate_offsets: np.ndarray,
) -> np.ndarray:
    step = float(source_offsets[1] - source_offsets[0])
    coordinate = (candidate_offsets - float(source_offsets[0])) / step
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    supported = (lower >= 0) & (lower + 1 < len(source_offsets))
    lower = np.clip(lower, 0, len(source_offsets) - 2)
    transposed = probability.T
    left = np.take_along_axis(transposed, lower, axis=1)
    right = np.take_along_axis(transposed, lower + 1, axis=1)
    sampled = left * (1.0 - fraction) + right * fraction
    sampled *= supported
    return sampled.T.astype(np.float32)


def c016_posterior_planes(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    log_probability = np.log(np.maximum(probability, 1e-7))
    log_probability -= np.max(log_probability, axis=1, keepdims=True)
    log_probability = np.clip(log_probability, -16.0, 0.0) / 8.0
    arithmetic = np.mean(probability, axis=0, keepdims=True)
    aggregate = np.log(np.maximum(arithmetic, 1e-7))
    aggregate -= np.max(aggregate, axis=1, keepdims=True)
    aggregate = np.clip(aggregate, -16.0, 0.0) / 8.0
    dispersion = np.std(log_probability, axis=0, keepdims=True)
    return np.concatenate(
        (log_probability, aggregate, dispersion), axis=0
    )


@torch.no_grad()
def c016_posterior_probability(
    well: Well,
    baseline: np.ndarray,
    parent_models: list[
        tuple[AlignmentUNet, np.ndarray, SequenceConfig, float]
    ],
    positions: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    source = np.arange(len(well.tvt_input), dtype=np.float64)
    visible = np.interp(
        np.minimum(positions, float(well.anchor_index)),
        source[: well.anchor_index + 1],
        well.tvt_input[: well.anchor_index + 1],
    )
    hidden = np.interp(
        positions,
        well.prediction_indices.astype(np.float64),
        baseline.astype(np.float64),
        left=float(well.tvt_input[well.anchor_index]),
        right=float(baseline[-1]),
    )
    prior_tvt = np.where(
        positions <= well.anchor_index + 1e-6, visible, hidden
    )
    candidate_offsets = (
        prior_tvt[:, None]
        - float(well.tvt_input[well.anchor_index])
        + C016_RESIDUAL_OFFSETS[None]
    )
    planes = []
    for model, image, config, temperature in parent_models:
        logits = model(torch.from_numpy(image)[None].to(device))[0]
        probability = (
            torch.softmax(logits.float() / temperature, dim=0)
            .cpu()
            .numpy()
        )
        probability /= probability.sum(axis=0, keepdims=True)
        planes.append(
            c016_residual_probability(
                probability,
                config.offsets,
                candidate_offsets,
            )
        )
    output = np.stack(planes).astype(np.float16)
    if output.shape != (15, 65, config.length):
        raise RuntimeError(
            f"{well.well_id}: unexpected C016 posterior {output.shape}"
        )
    return output


@torch.no_grad()
def c016_posterior_correction(
    model: AlignmentUNet,
    physical: np.ndarray,
    posterior_probability: np.ndarray,
    positions: np.ndarray,
    prior_tvt: np.ndarray,
    valid_positions: np.ndarray,
    baseline: np.ndarray,
    well: Well,
    config: SequenceConfig,
    device: torch.device,
) -> np.ndarray:
    image = np.concatenate(
        (c016_posterior_planes(posterior_probability), physical),
        axis=0,
    ).astype(np.float32)
    logits = model(torch.from_numpy(image)[None].to(device))[0]
    offsets = torch.as_tensor(
        C016_RESIDUAL_OFFSETS,
        dtype=torch.float32,
        device=device,
    )
    residual = torch.sum(
        torch.softmax(
            logits.float() / C016_POSTERIOR_TEMPERATURE, dim=0
        )
        * offsets[:, None],
        dim=0,
    ).cpu().numpy()
    sampled_valid = np.asarray(valid_positions) > 0.5
    sampled_valid[: config.prefix_points] = False
    candidate = np.interp(
        well.prediction_indices.astype(np.float64),
        positions[sampled_valid],
        (prior_tvt + residual)[sampled_valid],
    )
    correction = candidate - baseline
    correction -= correction[0]
    return np.clip(correction, -64.0, 64.0).astype(np.float32)


def run4_finite_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0] * 7
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
        float(values[-1] - values[0]),
        float(np.sqrt(np.mean(np.square(values)))),
    ]


def run4_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    if np.sum(finite) < 10:
        return 0.0
    first = first[finite] - np.mean(first[finite])
    second = second[finite] - np.mean(second[finite])
    denominator = np.sqrt(
        np.sum(np.square(first)) * np.sum(np.square(second))
    )
    return float(np.sum(first * second) / max(denominator, 1e-12))


def load_run4_assets(
    run4_root: Path,
    data_root: Path,
    test_ids: list[str],
) -> dict:
    source_root = run4_root / "source"
    if not source_root.exists():
        source_root = run4_root / "source.zip"
    if not source_root.exists():
        raise FileNotFoundError("Run4 deployment support source is missing")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from limited_query_cv.transform_v5_batch3_highcap_structural import (
        structural_bank,
    )
    from limited_query_cv.v5_batch3_highcap_path import (
        HIGHCAP_CONFIG,
        highcap_path_candidates,
    )
    from limited_query_cv.v5_batch3_run2_surface import (
        SurfaceWell,
        delayed_dip_channels,
        role_purged_buda_prediction,
        typewell_clusters,
    )

    recipe = json.loads((run4_root / "recipe.json").read_text())
    if (
        recipe["status"] != "deployment_candidate_complete"
        or not np.isclose(
            recipe["production_correction_scale"], RUN4_CORRECTION_SCALE
        )
        or len(recipe["outer_records"]) != 5
        or len(recipe["candidate_names"]) != 18
        or len(recipe["feature_names"]) != 409
    ):
        raise RuntimeError("Run4 deployment recipe changed")
    fold_by_id = {
        well_id: int(fold)
        for well_id, fold in zip(recipe["well_ids"], recipe["well_folds"])
    }

    def typewell(well_id: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_csv(
            data_root / split / f"{well_id}__typewell.csv",
            usecols=["TVT", "GR"],
        ).dropna()
        return (
            frame["TVT"].to_numpy(np.float64),
            frame["GR"].to_numpy(np.float64),
        )

    surface_wells = {}
    for well_id in fold_by_id:
        frame = pd.read_csv(
            data_root / "train" / f"{well_id}__horizontal_well.csv",
            usecols=["X", "Y", "BUDA", "TVT_input"],
        )
        finite = frame["BUDA"].notna().to_numpy()
        known = frame["TVT_input"].notna().to_numpy()
        unknown_index = np.flatnonzero(finite & ~known)
        known_index = np.flatnonzero(finite & known)
        keep = np.sort(
            np.concatenate(
                (
                    unknown_index,
                    known_index[-min(800, len(known_index)) :],
                )
            )
        )
        surface_wells[well_id] = SurfaceWell(
            well_id=well_id,
            x=frame["X"].to_numpy(np.float64)[keep],
            y=frame["Y"].to_numpy(np.float64)[keep],
            buda=frame["BUDA"].to_numpy(np.float64)[keep],
        )
    outer_pools = [
        {
            well_id: surface_wells[well_id]
            for well_id, fold in fold_by_id.items()
            if fold != outer
        }
        for outer in range(5)
    ]
    typewells = {
        well_id: typewell(well_id, "train") for well_id in fold_by_id
    }
    typewells.update(
        {well_id: typewell(well_id, "test") for well_id in test_ids}
    )
    clusters = typewell_clusters(typewells)

    surface_models = []
    structural_models = []
    base_weights = []
    for outer_record in recipe["outer_records"]:
        outer = int(outer_record["outer_fold"])
        if outer != len(surface_models):
            raise RuntimeError("Run4 outer recipe order changed")
        surface_model = XGBRegressor()
        surface_model.load_model(
            run4_root / outer_record["surface_router"]
        )
        surface_models.append(surface_model)
        structural_models.append(
            joblib.load(run4_root / outer_record["structural_router"])
        )
        base_weights.append(
            np.asarray(outer_record["base_weights"], dtype=np.float64)
        )
    return {
        "recipe": recipe,
        "highcap_config": HIGHCAP_CONFIG,
        "highcap_path_candidates": highcap_path_candidates,
        "structural_bank": structural_bank,
        "delayed_dip_channels": delayed_dip_channels,
        "role_purged_buda_prediction": role_purged_buda_prediction,
        "outer_pools": outer_pools,
        "clusters": clusters,
        "surface_models": surface_models,
        "structural_models": structural_models,
        "base_weights": base_weights,
    }


def run4_numeric_features(
    well: Well,
    clean: np.ndarray,
    v20: np.ndarray,
    surfaces: np.ndarray,
    cluster_size: int,
    delayed_dip_channels,
) -> np.ndarray:
    hidden = ~np.isfinite(well.tvt_input)
    known = ~hidden
    md = well.md.astype(np.float64)
    x = well.x.astype(np.float64)
    y = well.y.astype(np.float64)
    z = well.z.astype(np.float64)
    gr = well.gr.astype(np.float64)
    tail_md = md[hidden]
    tail_x = x[hidden]
    tail_y = y[hidden]
    tail_z = z[hidden]
    tail_gr = gr[hidden]
    prefix_tvt = well.tvt_input[known]
    features = [
        np.log1p(len(clean)),
        np.log1p(np.sum(known)),
        np.log1p(cluster_size),
        float(cluster_size == 0),
        *run4_finite_stats(tail_md - tail_md[0]),
        *run4_finite_stats(tail_x - np.mean(tail_x)),
        *run4_finite_stats(tail_y - np.mean(tail_y)),
        *run4_finite_stats(tail_z - tail_z[0]),
        *run4_finite_stats(tail_gr),
        *run4_finite_stats(np.diff(tail_gr)),
        *run4_finite_stats(clean - clean[0]),
        *run4_finite_stats(np.diff(clean)),
        *run4_finite_stats(clean - v20),
        *run4_finite_stats(np.diff(prefix_tvt[-min(256, len(prefix_tvt)) :])),
        float(np.mean(tail_x)),
        float(np.mean(tail_y)),
        float(np.hypot(tail_x[-1] - tail_x[0], tail_y[-1] - tail_y[0])),
        float(np.arctan2(tail_y[-1] - tail_y[0], tail_x[-1] - tail_x[0])),
        *run4_finite_stats(well.typewell_tvt),
        *run4_finite_stats(well.typewell_gr),
    ]
    dip, _ = delayed_dip_channels(
        md,
        z,
        shifts=(200, 250, 300, 350, 400),
        smoothing_rows=(31, 63, 127),
    )
    dip = dip[hidden]
    for index in range(dip.shape[1]):
        features.extend(run4_finite_stats(dip[:, index]))
    corrections = surfaces - clean[None]
    for correction in corrections:
        features.extend(run4_finite_stats(correction))
        features.extend(run4_finite_stats(np.diff(correction)))
        features.extend(
            (
                run4_correlation(correction, tail_z),
                run4_correlation(correction, tail_gr),
                float(np.max(np.abs(correction))),
            )
        )
    features.extend(run4_finite_stats(np.mean(corrections, axis=0)))
    features.extend(run4_finite_stats(np.std(corrections, axis=0)))
    output = np.asarray(features, dtype=np.float32)
    if output.shape != (313,) or not np.isfinite(output).all():
        raise RuntimeError(
            f"{well.well_id}: invalid Run4 numeric features {output.shape}"
        )
    return output


def run4_select_route(
    predicted_utility: np.ndarray,
    gram: np.ndarray,
    candidate_indices: np.ndarray,
    allowed_local_indices: np.ndarray,
    row_count: int,
    amplitude_cap: float,
    utility_threshold: float,
) -> tuple[int, float]:
    diagonal = np.asarray(
        [gram[index, index] for index in candidate_indices],
        dtype=np.float64,
    )
    row_count = float(row_count)
    raw_amplitude = np.clip(
        predicted_utility
        * np.sqrt(row_count / np.maximum(diagonal, 1e-12)),
        0.0,
        amplitude_cap,
    )
    utility = (
        2.0
        * raw_amplitude
        * predicted_utility
        * np.sqrt(diagonal * row_count)
        - np.square(raw_amplitude) * diagonal
    ) / row_count
    allowed = np.zeros(len(candidate_indices), dtype=bool)
    allowed[allowed_local_indices] = True
    utility[~allowed] = -np.inf
    selected = int(np.argmax(utility))
    amplitude = float(raw_amplitude[selected])
    if utility[selected] <= utility_threshold:
        amplitude = 0.0
    return int(candidate_indices[selected]), amplitude


def run4_outer_prediction(
    well: Well,
    clean: np.ndarray,
    v20: np.ndarray,
    outer: int,
    assets: dict,
    device: torch.device,
) -> tuple[np.ndarray, int, int]:
    highcap, highcap_names, _ = assets["highcap_path_candidates"](
        well,
        clean,
        assets["highcap_config"],
        device=str(device),
        observed_shift=0,
    )
    structural, structural_names = assets["structural_bank"](
        highcap, np.asarray(highcap_names)
    )
    expected_structural = np.asarray(
        [name.removeprefix("run1_") for name in assets["recipe"]["candidate_names"][:12]]
    )
    if not np.array_equal(structural_names.astype(str), expected_structural):
        raise RuntimeError("Run4 structural candidate order changed")

    pool = assets["outer_pools"][outer]
    target_cluster = assets["clusters"][well.well_id]
    cluster_pool = {
        well_id: surface_well
        for well_id, surface_well in pool.items()
        if assets["clusters"][well_id] == target_cluster
    }
    horizontal = pd.DataFrame(
        {
            "MD": well.md,
            "X": well.x,
            "Y": well.y,
            "Z": well.z,
            "GR": well.gr,
            "TVT_input": well.tvt_input,
        }
    )
    surfaces = []
    for config in RUN4_SURFACE_CONFIGS:
        selected_pool = (
            cluster_pool
            if config["cluster_only"] and len(cluster_pool) >= 3
            else pool
        )
        options = {
            key: value
            for key, value in config.items()
            if key not in {"name", "cluster_only"}
        }
        surfaces.append(
            assets["role_purged_buda_prediction"](
                selected_pool, horizontal, **options
            )
        )
    surfaces = np.stack(surfaces).astype(np.float64)
    correction = np.concatenate((structural, surfaces), axis=0).astype(
        np.float64
    ) - clean[None]
    gram = correction @ correction.T
    diagonal = np.diag(gram)
    rms = np.sqrt(np.maximum(diagonal, 0.0) / len(clean))
    left_scale = np.sqrt(np.maximum(diagonal[:12], 1e-12))
    right_scale = np.sqrt(np.maximum(diagonal[12:18], 1e-12))
    cosine = gram[:12, 12:18] / (
        left_scale[:, None] * right_scale[None, :]
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    numeric = run4_numeric_features(
        well,
        clean,
        v20,
        surfaces,
        len(cluster_pool),
        assets["delayed_dip_channels"],
    )
    features = np.concatenate(
        (
            numeric,
            np.log1p(rms),
            cosine.reshape(-1),
            np.std(cosine, axis=0),
        )
    ).astype(np.float32)
    if features.shape != (409,) or not np.isfinite(features).all():
        raise RuntimeError(f"{well.well_id}: invalid Run4 route features")

    surface_utility = np.asarray(
        assets["surface_models"][outer].predict(features[None]),
        dtype=np.float64,
    ).reshape(-1)
    structural_utility = np.asarray(
        assets["structural_models"][outer].predict(features[None]),
        dtype=np.float64,
    ).reshape(-1)
    surface_index, surface_amplitude = run4_select_route(
        surface_utility,
        gram,
        np.arange(12, 18, dtype=np.int64),
        np.arange(4, dtype=np.int64),
        len(clean),
        0.30,
        0.20,
    )
    structural_index, structural_amplitude = run4_select_route(
        structural_utility,
        gram,
        np.arange(12, dtype=np.int64),
        np.arange(12, dtype=np.int64),
        len(clean),
        0.20,
        0.01,
    )
    route = np.zeros(18, dtype=np.float64)
    route[surface_index] += 0.625 * surface_amplitude
    route[structural_index] += 1.25 * structural_amplitude
    weight = assets["base_weights"][outer] + route
    prediction = clean + RUN4_CORRECTION_SCALE * (weight @ correction)
    return prediction, int(surface_amplitude > 0), int(structural_amplitude > 0)


def predict_shard(
    shard_index: int,
    device: torch.device,
    well_ids: list[str],
    data_root: Path,
    narrow_checkpoint_paths: list[Path],
    distractor_checkpoint_paths: list[Path],
    wide_checkpoint_paths: list[Path],
    forward_checkpoint_paths: list[Path],
    forward_distractor_checkpoint_paths: list[Path],
    rare_checkpoint_paths: list[Path],
    supervised_checkpoint_paths: list[Path],
    wide_forward_checkpoint_paths: list[Path],
    wide_supervised_checkpoint_paths: list[Path],
    wide_supervised_seed_checkpoint_paths: list[Path],
    wide_supervised_small_checkpoint_paths: list[Path],
    wide_supervised_stratified_checkpoint_paths: list[Path],
    ordinal_checkpoint_paths: list[Path],
    base12_checkpoint_paths: list[Path],
    base16_checkpoint_paths: list[Path],
    long_synthetic_checkpoint_paths: list[Path],
    long_base12_checkpoint_paths: list[Path],
    spatial_confidence_paths: list[Path],
    c016_local_checkpoint_paths: list[list[Path]],
    c016_posterior_checkpoint_path: Path,
    spatial_store: SpatialReferenceStore,
    ancc_store: AnccReferenceStore,
    run4_assets: dict,
    narrow_config: SequenceConfig,
    wide_config: SequenceConfig,
    narrow_legacy_config: SequenceConfig,
    wide_legacy_config: SequenceConfig,
) -> list[dict[str, str | float]]:
    started = time.time()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    narrow_offsets = torch.as_tensor(
        narrow_config.offsets, dtype=torch.float32, device=device
    )
    wide_offsets = torch.as_tensor(
        wide_config.offsets, dtype=torch.float32, device=device
    )

    def load_models(
        checkpoint_paths: list[Path],
        base: int = 8,
        input_channels: int = 19,
    ) -> list[AlignmentUNet]:
        models = []
        for checkpoint_path in checkpoint_paths:
            model = AlignmentUNet(
                input_channels=input_channels, base=base
            ).to(device)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=True)
            )
            model.eval()
            models.append(model)
        return models

    narrow_models = load_models(narrow_checkpoint_paths)
    distractor_models = load_models(distractor_checkpoint_paths)
    wide_models = load_models(wide_checkpoint_paths)
    forward_models = load_models(forward_checkpoint_paths)
    forward_distractor_models = load_models(forward_distractor_checkpoint_paths)
    rare_models = load_models(rare_checkpoint_paths)
    supervised_models = load_models(supervised_checkpoint_paths)
    wide_forward_models = load_models(wide_forward_checkpoint_paths)
    wide_supervised_models = load_models(wide_supervised_checkpoint_paths)
    wide_supervised_seed_models = load_models(wide_supervised_seed_checkpoint_paths)
    wide_supervised_small_models = load_models(
        wide_supervised_small_checkpoint_paths, base=6
    )
    wide_supervised_stratified_models = load_models(
        wide_supervised_stratified_checkpoint_paths
    )
    ordinal_models = load_models(ordinal_checkpoint_paths)
    base12_models = load_models(base12_checkpoint_paths, base=12)
    base16_models = load_models(base16_checkpoint_paths, base=16)
    long_synthetic_models = load_models(long_synthetic_checkpoint_paths)
    long_base12_models = load_models(long_base12_checkpoint_paths, base=12)
    c016_local_models = [
        load_models(paths, base=12, input_channels=23)
        for paths in c016_local_checkpoint_paths
    ]
    c016_posterior_model = AlignmentUNet(
        input_channels=40, base=16
    ).to(device)
    c016_posterior_model.load_state_dict(
        torch.load(
            c016_posterior_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    c016_posterior_model.eval()
    confidence_models = []
    for path in spatial_confidence_paths:
        model = CatBoostRegressor()
        model.load_model(path)
        confidence_models.append(model)
    print(
        f"shard={shard_index} device={device} "
        f"narrow_models={len(narrow_models)} "
        f"distractor_models={len(distractor_models)} "
        f"wide_models={len(wide_models)} "
        f"forward_models={len(forward_models)} "
        f"forward_distractor_models={len(forward_distractor_models)} "
        f"rare_models={len(rare_models)} "
        f"supervised_models={len(supervised_models)} "
        f"wide_forward_models={len(wide_forward_models)} "
        f"wide_supervised_models={len(wide_supervised_models)} "
        f"wide_supervised_seed_models={len(wide_supervised_seed_models)} "
        f"wide_supervised_small_models={len(wide_supervised_small_models)} "
        f"wide_supervised_stratified_models={len(wide_supervised_stratified_models)} "
        f"ordinal_models={len(ordinal_models)} "
        f"base12_models={len(base12_models)} "
        f"base16_models={len(base16_models)} "
        f"long_synthetic_models={len(long_synthetic_models)} "
        f"long_base12_models={len(long_base12_models)} "
        f"c016_local_models={sum(map(len, c016_local_models))} "
        "c016_posterior_models=1 "
        f"wells={len(well_ids)}",
        flush=True,
    )

    def model_offsets(
        models: list[AlignmentUNet],
        tensor: torch.Tensor,
        valid: np.ndarray,
        config: SequenceConfig,
        offsets: torch.Tensor,
        temperature: float,
    ) -> np.ndarray:
        values = []
        for model in models:
            logits = model(tensor)[:, :, config.prefix_points :][:, :, valid]
            probability = torch.softmax(logits / temperature, dim=1)
            values.append(
                torch.sum(probability * offsets[None, :, None], dim=1)
                .float()
                .cpu()
                .numpy()
            )
        return np.stack(values)

    def ensemble_offsets(
        models: list[AlignmentUNet],
        tensor: torch.Tensor,
        valid: np.ndarray,
        config: SequenceConfig,
        offsets: torch.Tensor,
        temperature: float,
    ) -> np.ndarray:
        return np.mean(
            model_offsets(
                models, tensor, valid, config, offsets, temperature
            ),
            axis=0,
        )

    def ensemble_offset(
        models: list[AlignmentUNet],
        tensor: torch.Tensor,
        valid: np.ndarray,
        config: SequenceConfig,
        offsets: torch.Tensor,
        temperature: float,
    ) -> np.ndarray:
        return ensemble_offsets(
            models, tensor, valid, config, offsets, temperature
        )[0]

    def expanded_and_smoothed_prediction(
        well: Well,
        positions: np.ndarray,
        sampled_tvt: np.ndarray,
        anchor_tvt: float,
    ) -> np.ndarray:
        prediction_indices = well.prediction_indices
        raw = np.interp(prediction_indices, positions, sampled_tvt)
        prediction = raw.copy()
        progress = np.linspace(0.0, 1.0, len(prediction))
        prediction += TAIL_EXPANSION * progress * (prediction - anchor_tvt)
        full_tvt = well.tvt_input.copy()
        full_tvt[prediction_indices] = raw
        surface = full_tvt + well.z
        smooth_window = min(
            SURFACE_SMOOTH_WINDOW,
            len(surface) if len(surface) % 2 else len(surface) - 1,
        )
        if smooth_window >= 5:
            smoothed_surface = savgol_filter(surface, smooth_window, 2)
            prediction += (
                smoothed_surface[prediction_indices] - surface[prediction_indices]
            )
        return prediction

    rows: list[dict[str, str | float]] = []
    with torch.no_grad():
        for well_id in well_ids:
            well = load_test_well(well_id, data_root)
            narrow_image, _, narrow_metadata = make_alignment_example(
                well, well.anchor_index, narrow_config
            )
            narrow_tensor = torch.from_numpy(narrow_image).unsqueeze(0).to(device)
            narrow_valid = (
                np.asarray(narrow_metadata["valid_positions"])[
                    narrow_config.prefix_points :
                ]
                > 0.5
            )
            positions = np.asarray(narrow_metadata["positions"])[
                narrow_config.prefix_points :
            ][narrow_valid]
            narrow_legacy_image, _, narrow_legacy_metadata = make_alignment_example(
                well, well.anchor_index, narrow_legacy_config
            )
            narrow_legacy_tensor = torch.from_numpy(narrow_legacy_image).unsqueeze(0).to(device)
            narrow_legacy_valid = (
                np.asarray(narrow_legacy_metadata["valid_positions"])[
                    narrow_legacy_config.prefix_points :
                ]
                > 0.5
            )
            if not np.array_equal(narrow_valid, narrow_legacy_valid):
                raise RuntimeError("legacy and standard narrow sampling grids differ")
            narrow_by_fold = model_offsets(
                narrow_models,
                narrow_legacy_tensor,
                narrow_legacy_valid,
                narrow_legacy_config,
                narrow_offsets,
                NARROW_TEMPERATURE,
            )[:, 0]
            narrow_offset = np.mean(narrow_by_fold, axis=0)
            c016_fixed_by_fold = model_offsets(
                narrow_models,
                narrow_tensor,
                narrow_valid,
                narrow_config,
                narrow_offsets,
                NARROW_TEMPERATURE,
            )[:, 0]
            distractor_by_fold = model_offsets(
                distractor_models,
                narrow_tensor,
                narrow_valid,
                narrow_config,
                narrow_offsets,
                DISTRACTOR_TEMPERATURE,
            )[:, 0]
            distractor_offset = np.mean(distractor_by_fold, axis=0)
            forward_by_fold = model_offsets(
                forward_models,
                narrow_tensor,
                narrow_valid,
                narrow_config,
                narrow_offsets,
                FORWARD_TEMPERATURE,
            )[:, 0]
            forward_offset = np.mean(forward_by_fold, axis=0)
            forward_distractor_by_fold = model_offsets(
                forward_distractor_models,
                narrow_tensor,
                narrow_valid,
                narrow_config,
                narrow_offsets,
                FORWARD_DISTRACTOR_TEMPERATURE,
            )[:, 0]
            forward_distractor_offset = np.mean(
                forward_distractor_by_fold, axis=0
            )
            rare_by_fold = model_offsets(
                rare_models,
                narrow_tensor,
                narrow_valid,
                narrow_config,
                narrow_offsets,
                RARE_TEMPERATURE,
            )[:, 0]
            rare_offset = np.mean(rare_by_fold, axis=0)
            supervised_by_fold = model_offsets(
                supervised_models,
                narrow_tensor,
                narrow_valid,
                narrow_config,
                narrow_offsets,
                SUPERVISED_TEMPERATURE,
            )[:, 0]
            supervised_offset = np.mean(supervised_by_fold, axis=0)

            wide_image, _, wide_metadata = make_alignment_example(
                well, well.anchor_index, wide_config
            )
            wide_tensor = torch.from_numpy(wide_image).unsqueeze(0).to(device)
            wide_valid = (
                np.asarray(wide_metadata["valid_positions"])[
                    wide_config.prefix_points :
                ]
                > 0.5
            )
            wide_positions = np.asarray(wide_metadata["positions"])[
                wide_config.prefix_points :
            ][wide_valid]
            if not np.array_equal(narrow_valid, wide_valid) or not np.allclose(
                positions, wide_positions
            ):
                raise RuntimeError("narrow and wide sampling grids differ")
            wide_legacy_image, _, wide_legacy_metadata = make_alignment_example(
                well, well.anchor_index, wide_legacy_config
            )
            wide_legacy_tensor = torch.from_numpy(wide_legacy_image).unsqueeze(0).to(device)
            wide_legacy_valid = (
                np.asarray(wide_legacy_metadata["valid_positions"])[
                    wide_legacy_config.prefix_points :
                ]
                > 0.5
            )
            if not np.array_equal(wide_valid, wide_legacy_valid):
                raise RuntimeError("legacy and standard wide sampling grids differ")
            wide_by_fold = model_offsets(
                wide_models,
                wide_legacy_tensor,
                wide_legacy_valid,
                wide_legacy_config,
                wide_offsets,
                WIDE_TEMPERATURE,
            )[:, 0]
            wide_offset = np.mean(wide_by_fold, axis=0)
            c016_wide_by_fold = model_offsets(
                wide_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_TEMPERATURE,
            )[:, 0]
            wide_forward_by_fold = model_offsets(
                wide_forward_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_FORWARD_TEMPERATURE,
            )[:, 0]
            wide_forward_offset = np.mean(wide_forward_by_fold, axis=0)
            wide_supervised_by_fold = model_offsets(
                wide_supervised_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )[:, 0]
            wide_supervised_offset = np.mean(
                wide_supervised_by_fold, axis=0
            )
            wide_supervised_gr25_offset = ensemble_offset(
                wide_supervised_models,
                wide_legacy_tensor,
                wide_legacy_valid,
                wide_legacy_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            wide_supervised_seed_offset = ensemble_offset(
                wide_supervised_seed_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            wide_supervised_small_offset = ensemble_offset(
                wide_supervised_small_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            wide_supervised_stratified_offset = ensemble_offset(
                wide_supervised_stratified_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            ordinal_offset = ensemble_offset(
                ordinal_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                ORDINAL_TEMPERATURE,
            )
            masked_tensor = torch.from_numpy(
                masked_alignment_views(
                    well.well_id, wide_image, wide_config.prefix_points
                )
            ).to(device)
            wide_supervised_mask_offsets = ensemble_offsets(
                wide_supervised_models,
                masked_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            base12_mask_offsets = ensemble_offsets(
                base12_models,
                masked_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            base16_mask_offsets = ensemble_offsets(
                base16_models,
                masked_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            long_synthetic_mask_offsets = ensemble_offsets(
                long_synthetic_models,
                masked_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            del masked_tensor
            long_base12_offset = ensemble_offset(
                long_base12_models,
                wide_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            feature_tensor = torch.from_numpy(
                np.stack((wide_image, shifted_alignment_view(wide_image)))
            ).to(device)
            wide_supervised_feature_offsets = ensemble_offsets(
                wide_supervised_models,
                feature_tensor,
                wide_valid,
                wide_config,
                wide_offsets,
                WIDE_SUPERVISED_TEMPERATURE,
            )
            del feature_tensor
            original_wide_supervised_offset = wide_supervised_offset
            wide_supervised_offset = (
                (
                    1.0
                    - WIDE_SUPERVISED_SEED_FRACTION
                    - WIDE_SUPERVISED_SMALL_FRACTION
                    - WIDE_SUPERVISED_STRATIFIED_FRACTION
                )
                * (
                    (1.0 - WIDE_SUPERVISED_GR25_TTA_FRACTION)
                    * wide_supervised_offset
                    + WIDE_SUPERVISED_GR25_TTA_FRACTION
                    * wide_supervised_gr25_offset
                )
                + WIDE_SUPERVISED_SEED_FRACTION * wide_supervised_seed_offset
                + WIDE_SUPERVISED_SMALL_FRACTION * wide_supervised_small_offset
                + WIDE_SUPERVISED_STRATIFIED_FRACTION
                * wide_supervised_stratified_offset
            )
            other_offset = (
                NARROW_WEIGHT * narrow_offset
                + DISTRACTOR_WEIGHT * distractor_offset
                + WIDE_WEIGHT * wide_offset
                + FORWARD_WEIGHT * forward_offset
                + FORWARD_DISTRACTOR_WEIGHT * forward_distractor_offset
                + RARE_WEIGHT * rare_offset
                + SUPERVISED_WEIGHT * supervised_offset
                + WIDE_FORWARD_WEIGHT * wide_forward_offset
            ) / (1.0 - WIDE_SUPERVISED_WEIGHT)
            anchor_tvt = float(narrow_metadata["anchor_tvt"])
            other_prediction = expanded_and_smoothed_prediction(
                well,
                positions,
                anchor_tvt + other_offset,
                anchor_tvt,
            )
            family_prediction = expanded_and_smoothed_prediction(
                well,
                positions,
                anchor_tvt + wide_supervised_offset,
                anchor_tvt,
            )
            baseline_prediction = (
                (1.0 - WIDE_SUPERVISED_WEIGHT) * other_prediction
                + WIDE_SUPERVISED_WEIGHT * family_prediction
            )
            predicted_excursion = float(
                np.max(np.abs(baseline_prediction - anchor_tvt))
            )
            other_excursion = float(
                np.max(np.abs(other_prediction - anchor_tvt))
            )
            family_excursion = float(
                np.max(np.abs(family_prediction - anchor_tvt))
            )
            confidence_gate = float(family_excursion > other_excursion)
            disagreement_ratio = float(
                np.sqrt(np.mean(np.square(family_prediction - other_prediction)))
                / max(predicted_excursion, 1e-6)
            )
            disagreement_gate = float(
                np.clip(
                    (disagreement_ratio - DISAGREEMENT_THRESHOLD)
                    / DISAGREEMENT_WIDTH,
                    0.0,
                    1.0,
                )
            )
            excursion_gate = float(
                np.clip(
                    (predicted_excursion - ADAPTIVE_EXCURSION_THRESHOLD)
                    / ADAPTIVE_EXCURSION_WIDTH,
                    0.0,
                    1.0,
                )
            )
            family_weight = (
                WIDE_SUPERVISED_WEIGHT
                + (
                    ADAPTIVE_FAMILY_BONUS
                    + DISAGREEMENT_EXTRA_BONUS * disagreement_gate
                )
                * excursion_gate
                * confidence_gate
            )
            prediction = (
                (1.0 - family_weight) * other_prediction
                + family_weight * family_prediction
            )
            prediction_indices = well.prediction_indices
            visible_left = max(
                0, well.anchor_index + 1 - SURFACE_EXTRAPOLATION_ROWS
            )
            visible_md = well.md[visible_left : well.anchor_index + 1]
            visible_surface = (
                well.tvt_input[visible_left : well.anchor_index + 1]
                + well.z[visible_left : well.anchor_index + 1]
            )
            centered_md = visible_md - np.mean(visible_md)
            centered_surface = visible_surface - np.mean(visible_surface)
            surface_slope = float(centered_md @ centered_surface) / max(
                float(centered_md @ centered_md), 1e-9
            )
            extrapolated_surface = visible_surface[-1] + surface_slope * (
                well.md[prediction_indices] - well.md[well.anchor_index]
            )
            surface_correction = np.clip(
                extrapolated_surface
                - (prediction + well.z[prediction_indices]),
                -SURFACE_EXTRAPOLATION_LIMIT,
                SURFACE_EXTRAPOLATION_LIMIT,
            )
            prediction += SURFACE_EXTRAPOLATION_WEIGHT * surface_correction
            prediction, spatial_multiplier, spatial_family = spatially_correct_prediction(
                well,
                prediction,
                spatial_store,
                confidence_models,
            )
            prediction += GR_CALIBRATION_WEIGHT * gr_path_correction(well, prediction)
            ordinal_prediction = np.interp(
                prediction_indices,
                positions,
                anchor_tvt + ordinal_offset,
            )
            ordinal_correction = gaussian_filter1d(
                ordinal_prediction - prediction,
                sigma=ORDINAL_SMOOTH_SIGMA,
                mode="nearest",
            )
            prediction += ORDINAL_WEIGHT * ordinal_correction

            def raw_prediction(offset: np.ndarray) -> np.ndarray:
                return np.interp(
                    prediction_indices,
                    positions,
                    anchor_tvt + offset,
                )

            original_raw = raw_prediction(original_wide_supervised_offset)
            base12_raw = raw_prediction(base12_mask_offsets[0])
            base16_raw = raw_prediction(base16_mask_offsets[0])
            long_synthetic_raw = raw_prediction(long_synthetic_mask_offsets[0])
            long_base12_raw = raw_prediction(long_base12_offset)
            v16_prediction = prediction.copy()
            prediction += ANCC_WEIGHT * ancc_surface_correction(
                well, v16_prediction, ancc_store
            )
            prediction += RESIDUAL_GR_WEIGHT * residual_gr_correction(
                well, v16_prediction
            )
            prediction += MASK_SENSITIVITY_WEIGHT * (
                raw_prediction(np.mean(wide_supervised_mask_offsets, axis=0))
                - raw_prediction(wide_supervised_mask_offsets[0])
            )
            prediction += BASE12_RESIDUAL_WEIGHT * (base12_raw - original_raw)
            prediction += BASE16_RESIDUAL_WEIGHT * (base16_raw - original_raw)
            prediction += LONG_SYNTHETIC_RESIDUAL_WEIGHT * (
                long_synthetic_raw - original_raw
            )
            prediction += MASK_SENSITIVITY_WEIGHT * (
                raw_prediction(np.mean(long_synthetic_mask_offsets, axis=0))
                - long_synthetic_raw
            )
            prediction += MASK_SENSITIVITY_WEIGHT * (
                raw_prediction(np.mean(base16_mask_offsets, axis=0)) - base16_raw
            )
            prediction += BASE12_MASK_SENSITIVITY_WEIGHT * (
                raw_prediction(np.mean(base12_mask_offsets, axis=0)) - base12_raw
            )
            prediction += FEATURE_SENSITIVITY_WEIGHT * (
                raw_prediction(wide_supervised_feature_offsets[1])
                - raw_prediction(wide_supervised_feature_offsets[0])
            )
            prediction += LONG_BASE12_RESIDUAL_WEIGHT * (
                long_base12_raw - original_raw
            )
            v20_prediction = prediction.copy()
            component_offsets = (
                c016_fixed_by_fold,
                distractor_by_fold,
                c016_wide_by_fold,
                forward_by_fold,
                forward_distractor_by_fold,
                rare_by_fold,
                supervised_by_fold,
                wide_forward_by_fold,
                wide_supervised_by_fold,
            )
            c016_components = np.stack(
                [
                    np.stack(
                        [
                            raw_prediction(family[outer])
                            for family in component_offsets
                        ]
                    )
                    for outer in range(5)
                ]
            ).astype(np.float32)
            c016_outer_predictions = []
            c016_residual_images = []
            c016_positions = None
            c016_prior = None
            c016_valid = None
            for outer in range(5):
                (
                    residual_image,
                    residual_positions,
                    residual_prior,
                    residual_valid,
                ) = c016_residual_image(
                    well,
                    v20_prediction,
                    c016_components[outer],
                    wide_image,
                    wide_metadata,
                    wide_config,
                )
                c016_residual_images.append(residual_image)
                if outer == C016_POSTERIOR_OUTER:
                    c016_positions = residual_positions
                    c016_prior = residual_prior
                    c016_valid = residual_valid
                local_candidates = [
                    c016_local_prediction(
                        model,
                        residual_image,
                        residual_positions,
                        v20_prediction,
                        well,
                        C016_LOCAL_TEMPERATURES[outer],
                        wide_config,
                        device,
                    )
                    for model in c016_local_models[outer]
                ]
                local_bag = np.mean(local_candidates, axis=0)
                d209 = (
                    v20_prediction.astype(np.float64)
                    + C016_LOCAL_WEIGHT
                    * (
                        local_bag.astype(np.float64)
                        - v20_prediction.astype(np.float64)
                    )
                )
                d570 = (
                    v20_prediction.astype(np.float64)
                    + C016_LOCAL_AMPLITUDES[outer]
                    * (
                        d209
                        - v20_prediction.astype(np.float64)
                    )
                ).astype(np.float32)
                c016_outer_predictions.append(d570)

            assert (
                c016_positions is not None
                and c016_prior is not None
                and c016_valid is not None
            )
            posterior_parent_models = [
                (
                    narrow_models[C016_POSTERIOR_OUTER],
                    narrow_image,
                    narrow_config,
                    NARROW_TEMPERATURE,
                ),
                (
                    distractor_models[C016_POSTERIOR_OUTER],
                    narrow_image,
                    narrow_config,
                    DISTRACTOR_TEMPERATURE,
                ),
                (
                    wide_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_TEMPERATURE,
                ),
                (
                    forward_models[C016_POSTERIOR_OUTER],
                    narrow_image,
                    narrow_config,
                    FORWARD_TEMPERATURE,
                ),
                (
                    forward_distractor_models[C016_POSTERIOR_OUTER],
                    narrow_image,
                    narrow_config,
                    FORWARD_DISTRACTOR_TEMPERATURE,
                ),
                (
                    rare_models[C016_POSTERIOR_OUTER],
                    narrow_image,
                    narrow_config,
                    RARE_TEMPERATURE,
                ),
                (
                    supervised_models[C016_POSTERIOR_OUTER],
                    narrow_image,
                    narrow_config,
                    SUPERVISED_TEMPERATURE,
                ),
                (
                    wide_forward_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_FORWARD_TEMPERATURE,
                ),
                (
                    wide_supervised_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_SUPERVISED_TEMPERATURE,
                ),
                (
                    long_synthetic_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_SUPERVISED_TEMPERATURE,
                ),
                (
                    ordinal_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    ORDINAL_TEMPERATURE,
                ),
                (
                    base12_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_SUPERVISED_TEMPERATURE,
                ),
                (
                    base16_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_SUPERVISED_TEMPERATURE,
                ),
                (
                    wide_supervised_small_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_SUPERVISED_TEMPERATURE,
                ),
                (
                    wide_supervised_seed_models[C016_POSTERIOR_OUTER],
                    wide_image,
                    wide_config,
                    WIDE_SUPERVISED_TEMPERATURE,
                ),
            ]
            posterior_probability = c016_posterior_probability(
                well,
                v20_prediction,
                posterior_parent_models,
                c016_positions,
                device,
            )
            posterior_correction = c016_posterior_correction(
                c016_posterior_model,
                c016_residual_images[C016_POSTERIOR_OUTER],
                posterior_probability,
                c016_positions,
                c016_prior,
                c016_valid,
                v20_prediction,
                well,
                wide_config,
                device,
            )
            c016_outer_predictions[C016_POSTERIOR_OUTER] = (
                c016_outer_predictions[
                    C016_POSTERIOR_OUTER
                ].astype(np.float64)
                + C016_FINAL_AMPLITUDE
                * C016_POSTERIOR_RIDGE_WEIGHT
                * C016_POSTERIOR_EFFECTIVE_SCALE
                * posterior_correction.astype(np.float64)
            ).astype(np.float32)
            run4_outer_predictions = []
            routed_surfaces = 0
            routed_structures = 0
            for outer, c016_outer in enumerate(c016_outer_predictions):
                run4_prediction, routed_surface, routed_structure = (
                    run4_outer_prediction(
                        well,
                        c016_outer.astype(np.float64),
                        v20_prediction.astype(np.float64),
                        outer,
                        run4_assets,
                        device,
                    )
                )
                run4_outer_predictions.append(run4_prediction)
                routed_surfaces += routed_surface
                routed_structures += routed_structure
            prediction = np.mean(np.stack(run4_outer_predictions), axis=0)
            rows.extend(
                {
                    "id": f"{well_id}_{int(row_index)}",
                    "tvt": float(value),
                }
                for row_index, value in zip(prediction_indices, prediction)
            )
            print(
                f"shard={shard_index} device={device} {well_id}: "
                f"{len(prediction_indices)} rows excursion={predicted_excursion:.2f} "
                f"family_weight={family_weight:.3f} confidence={confidence_gate:.0f} "
                f"disagreement={disagreement_ratio:.3f} "
                f"spatial_family={spatial_family} "
                f"spatial_multiplier={spatial_multiplier:.3f} "
                "c016_outer_models=5 "
                f"run4_surface_routes={routed_surfaces}/5 "
                f"run4_structural_routes={routed_structures}/5",
                flush=True,
            )
    print(
        f"shard={shard_index} device={device} complete "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return rows


def main() -> None:
    started = time.time()
    data_root = discover_data_root()
    weights_root = discover_weights_root()
    c016_weights_root = discover_c016_weights_root()
    run4_weights_root = discover_run4_weights_root()
    output_path = Path(os.environ.get("ROGII_OUTPUT", "/kaggle/working/submission.csv"))
    narrow_config = SequenceConfig()
    wide_config = SequenceConfig(state_radius=120.0, state_step=1.0)
    narrow_legacy_config = SequenceConfig(gr_smooth_window=25)
    wide_legacy_config = SequenceConfig(
        state_radius=120.0, state_step=1.0, gr_smooth_window=25
    )
    narrow_checkpoint_paths = [
        weights_root / f"alignment_fixed_synth_fold{fold}.pt" for fold in range(5)
    ]
    distractor_checkpoint_paths = [
        weights_root / f"alignment_distractor_h05_fold{fold}.pt" for fold in range(5)
    ]
    wide_checkpoint_paths = [
        weights_root / f"alignment_wide_synth_fold{fold}.pt" for fold in range(5)
    ]
    forward_checkpoint_paths = [
        weights_root / f"alignment_forward_std_fold{fold}.pt" for fold in range(5)
    ]
    forward_distractor_checkpoint_paths = [
        weights_root / f"alignment_forward_distractor_fold{fold}.pt"
        for fold in range(5)
    ]
    rare_checkpoint_paths = [
        weights_root / f"alignment_forward_exc4_fold{fold}.pt" for fold in range(5)
    ]
    supervised_checkpoint_paths = [
        weights_root / f"alignment_forward_exc2sup_fold{fold}.pt"
        for fold in range(5)
    ]
    wide_forward_checkpoint_paths = [
        weights_root / f"alignment_wide_forward_extreme_fold{fold}.pt"
        for fold in range(5)
    ]
    wide_supervised_checkpoint_paths = [
        weights_root / f"alignment_wide_forward_extreme_exc2sup_fold{fold}.pt"
        for fold in range(5)
    ]
    wide_supervised_seed_checkpoint_paths = [
        weights_root / f"alignment_wide_supervised_seed3_fold{fold}.pt"
        for fold in range(5)
    ]
    wide_supervised_small_checkpoint_paths = [
        weights_root / f"alignment_wide_supervised_base6_fold{fold}.pt"
        for fold in range(5)
    ]
    wide_supervised_stratified_checkpoint_paths = [
        weights_root / f"alignment_wide_supervised_strat_fold{fold}.pt"
        for fold in range(5)
    ]
    ordinal_checkpoint_paths = [
        weights_root / f"alignment_ordinal_wide_fold{fold}.pt" for fold in range(5)
    ]
    base12_checkpoint_paths = [
        weights_root / f"alignment_wide_base12_exc2sup_fold{fold}.pt"
        for fold in range(5)
    ]
    base16_checkpoint_paths = [
        weights_root / f"alignment_wide_base16_exc2sup_fold{fold}.pt"
        for fold in range(5)
    ]
    long_synthetic_checkpoint_paths = [
        weights_root / f"alignment_longsynthetic_exc2sup_fold{fold}.pt"
        for fold in range(5)
    ]
    long_base12_checkpoint_paths = [
        weights_root / f"alignment_longsynthetic_base12_exc2sup_fold{fold}.pt"
        for fold in range(5)
    ]
    spatial_confidence_paths = [
        weights_root / f"spatial_confidence_fold{fold}.cbm" for fold in range(5)
    ]
    c016_local_checkpoint_paths = [
        [
            c016_weights_root
            / (
                f"c016_outer{outer}_local_seed{seed}_"
                "epoch016.pt"
            )
            for seed in C016_LOCAL_SEEDS
        ]
        for outer in range(5)
    ]
    c016_posterior_checkpoint_path = (
        c016_weights_root
        / "c016_outer3_posterior_physical_b16_epoch020.pt"
    )
    missing = [
        str(path)
        for path in (
            narrow_checkpoint_paths
            + distractor_checkpoint_paths
            + wide_checkpoint_paths
            + forward_checkpoint_paths
            + forward_distractor_checkpoint_paths
            + rare_checkpoint_paths
            + supervised_checkpoint_paths
            + wide_forward_checkpoint_paths
            + wide_supervised_checkpoint_paths
            + wide_supervised_seed_checkpoint_paths
            + wide_supervised_small_checkpoint_paths
            + wide_supervised_stratified_checkpoint_paths
            + ordinal_checkpoint_paths
            + base12_checkpoint_paths
            + base16_checkpoint_paths
            + long_synthetic_checkpoint_paths
            + long_base12_checkpoint_paths
            + spatial_confidence_paths
            + [
                path
                for outer_paths in c016_local_checkpoint_paths
                for path in outer_paths
            ]
            + [
                weights_root / "spatial_reference.npz",
                weights_root / "ancc_reference.npz",
                c016_posterior_checkpoint_path,
                c016_weights_root / "c016_manifest.json",
            ]
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    spatial_store = load_spatial_reference(weights_root)
    ancc_store = load_ancc_reference(weights_root)
    well_ids = test_well_ids(data_root)
    run4_assets = load_run4_assets(run4_weights_root, data_root, well_ids)
    if torch.cuda.is_available():
        max_gpus = max(1, int(os.environ.get("ROGII_MAX_GPUS", "2")))
        devices = [
            torch.device(f"cuda:{index}")
            for index in range(min(max_gpus, torch.cuda.device_count()))
        ]
    else:
        devices = [torch.device("cpu")]
    shards = [well_ids[index :: len(devices)] for index in range(len(devices))]
    print(
        f"devices={[str(device) for device in devices]} "
        f"cuda_count={torch.cuda.device_count()} data={data_root} "
        f"weights={weights_root} c016_weights={c016_weights_root} "
        f"run4_weights={run4_weights_root} "
        f"wells={len(well_ids)}",
        flush=True,
    )

    rows: list[dict[str, str | float]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(
                predict_shard,
                shard_index,
                device,
                shard,
                data_root,
                narrow_checkpoint_paths,
                distractor_checkpoint_paths,
                wide_checkpoint_paths,
                forward_checkpoint_paths,
                forward_distractor_checkpoint_paths,
                rare_checkpoint_paths,
                supervised_checkpoint_paths,
                wide_forward_checkpoint_paths,
                wide_supervised_checkpoint_paths,
                wide_supervised_seed_checkpoint_paths,
                wide_supervised_small_checkpoint_paths,
                wide_supervised_stratified_checkpoint_paths,
                ordinal_checkpoint_paths,
                base12_checkpoint_paths,
                base16_checkpoint_paths,
                long_synthetic_checkpoint_paths,
                long_base12_checkpoint_paths,
                spatial_confidence_paths,
                c016_local_checkpoint_paths,
                c016_posterior_checkpoint_path,
                spatial_store,
                ancc_store,
                run4_assets,
                narrow_config,
                wide_config,
                narrow_legacy_config,
                wide_legacy_config,
            )
            for shard_index, (device, shard) in enumerate(zip(devices, shards))
        ]
        for future in futures:
            rows.extend(future.result())

    prediction = pd.DataFrame(rows)
    sample = pd.read_csv(data_root / "sample_submission.csv")[["id"]]
    submission = sample.merge(prediction, on="id", how="left", validate="one_to_one")
    if submission["tvt"].isna().any():
        raise RuntimeError(
            f"submission is missing {int(submission['tvt'].isna().sum())} predictions"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(
        f"wrote={output_path} rows={len(submission)} elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
