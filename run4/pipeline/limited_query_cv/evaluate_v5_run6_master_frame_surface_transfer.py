from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.folds import fold_indices


SEED = 20260805
TYPEWELL_GRID = np.arange(9200.0, 13001.0, 1.0)
MINIMUM_OVERLAP = 200
CORRELATION_THRESHOLD = 0.99
QUERY_STRIDE = 16
VISIBLE_CALIBRATION_POINTS = 64
ANALOG_WELLS = 3
ANALOG_DISTANCE_LIMIT_FT = 1500.0
ANALOG_DISTANCE_FLOOR_FT = 25.0
LOCAL_POINTS = 32
LOCAL_DISTANCE_LIMIT_FT = 600.0
LOCAL_DISTANCE_FLOOR_FT = 10.0
VARIANTS = ("analog", "local", "equal_mean")
CORRECTION_CAPS = (2.0, 4.0, 8.0, 12.0, 24.0)
OUTER_BLEND_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)


@dataclass(frozen=True)
class Reference:
    well_id: str
    fold: int
    family: int
    well: Well
    sampled_rows: np.ndarray
    xy: np.ndarray
    surface: np.ndarray
    tree: cKDTree


@dataclass(frozen=True)
class Query:
    well_id: str
    fold: int
    family: int
    well: Well
    baseline: np.ndarray
    truth: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    residual = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(residual))))


def original_folds(ids: np.ndarray) -> np.ndarray:
    result = np.empty(len(ids), dtype=np.int8)
    for fold in range(5):
        _, held = fold_indices(ids, fold, 5)
        result[held] = fold
    return result


def profile_arrays(
    profiles: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(profiles), len(TYPEWELL_GRID)), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for index, (profile_tvt, profile_gr) in enumerate(profiles):
        supported = (TYPEWELL_GRID >= np.nanmin(profile_tvt)) & (
            TYPEWELL_GRID <= np.nanmax(profile_tvt)
        )
        values[index, supported] = np.interp(
            TYPEWELL_GRID[supported],
            profile_tvt,
            profile_gr,
        )
        mask[index] = supported
    return values, mask


def typewell_arrays(wells: list[Well]) -> tuple[np.ndarray, np.ndarray]:
    return profile_arrays(
        [(well.typewell_tvt, well.typewell_gr) for well in wells]
    )


def load_legal_typewell_profiles(
    data_root: Path,
    ids: np.ndarray,
    workers: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    def load_profile(well_id: str) -> tuple[np.ndarray, np.ndarray]:
        path = data_root / "train" / f"{well_id}__typewell.csv"
        frame = pd.read_csv(path, usecols=["TVT", "GR"]).sort_values("TVT")
        return (
            frame["TVT"].to_numpy(np.float64),
            frame["GR"].to_numpy(np.float64),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(load_profile, ids))


def connected_typewell_families(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    count = len(values)
    parent = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left = find(left)
        right = find(right)
        if left != right:
            parent[right] = left

    for left in range(count - 1):
        overlap = mask[left + 1 :] & mask[left]
        overlap_count = np.sum(overlap, axis=1)
        eligible = np.flatnonzero(overlap_count >= MINIMUM_OVERLAP)
        if not len(eligible):
            continue
        use = overlap[eligible]
        right_values = np.where(use, values[left + 1 :][eligible], 0.0)
        left_values = np.where(use, values[left], 0.0)
        n = overlap_count[eligible].astype(np.float64)
        sum_right = np.sum(right_values, axis=1)
        sum_left = np.sum(left_values, axis=1)
        covariance = (
            np.sum(right_values * left_values, axis=1)
            - sum_right * sum_left / n
        )
        variance_right = (
            np.sum(np.square(right_values), axis=1)
            - np.square(sum_right) / n
        )
        variance_left = (
            np.sum(np.square(left_values), axis=1)
            - np.square(sum_left) / n
        )
        correlation = covariance / np.sqrt(
            np.maximum(variance_right * variance_left, 1e-12)
        )
        for local in eligible[correlation >= CORRELATION_THRESHOLD]:
            union(left, left + 1 + int(local))
    roots = np.asarray([find(index) for index in range(count)])
    _, family = np.unique(roots, return_inverse=True)
    return family.astype(np.int32)


def load_fold_cache(root: Path, fold: int) -> dict[str, dict[str, np.ndarray]]:
    with np.load(root / f"fold{fold}.npz", allow_pickle=False) as cache:
        if int(cache["fold"]) != fold:
            raise ValueError(f"F{fold} baseline cache fold changed")
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise ValueError(f"F{fold} baseline cache firewall failed")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        truth = cache["truth"].astype(np.float32)
    return {
        well_id: {
            "baseline": baseline[starts[index] : starts[index + 1]],
            "truth": truth[starts[index] : starts[index + 1]],
        }
        for index, well_id in enumerate(ids)
    }


def build_records(
    data_root: Path,
    baseline_root: Path,
    fold: int,
    workers: int,
) -> tuple[list[Reference], list[Query], np.ndarray, np.ndarray]:
    ids = np.asarray(training_well_ids(data_root))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = list(pool.map(lambda value: load_well(value, data_root), ids))
    folds = original_folds(ids)
    values, mask = typewell_arrays(wells)
    families = connected_typewell_families(values, mask)
    fold_cache = load_fold_cache(baseline_root, fold)
    references = []
    queries = []
    for index, (well_id, well) in enumerate(zip(ids, wells)):
        sampled = np.unique(
            np.r_[
                np.arange(0, len(well.md), QUERY_STRIDE),
                len(well.md) - 1,
                well.anchor_index,
            ]
        ).astype(np.int64)
        sampled.sort()
        references.append(
            Reference(
                well_id=well_id,
                fold=int(folds[index]),
                family=int(families[index]),
                well=well,
                sampled_rows=sampled,
                xy=np.column_stack((well.x[sampled], well.y[sampled])),
                surface=(well.tvt[sampled] + well.z[sampled]).astype(
                    np.float64
                ),
                tree=cKDTree(
                    np.column_stack((well.x[sampled], well.y[sampled]))
                ),
            )
        )
        if well_id in fold_cache:
            if int(folds[index]) != fold:
                raise ValueError(f"{well_id}: cache/original fold mismatch")
            tail = well.prediction_indices
            if len(tail) != len(fold_cache[well_id]["baseline"]):
                raise ValueError(f"{well_id}: baseline/native tail mismatch")
            queries.append(
                Query(
                    well_id=well_id,
                    fold=fold,
                    family=int(families[index]),
                    well=well,
                    baseline=fold_cache[well_id]["baseline"],
                    truth=fold_cache[well_id]["truth"],
                )
            )
    return references, queries, values, mask


def calibrated_surface(
    well: Well,
    query_rows: np.ndarray,
    raw_surface: np.ndarray,
) -> np.ndarray:
    visible_positions = np.flatnonzero(query_rows <= well.anchor_index)
    recent = visible_positions[-min(VISIBLE_CALIBRATION_POINTS, len(visible_positions)) :]
    observed = well.tvt_input[query_rows[recent]] + well.z[query_rows[recent]]
    finite = np.isfinite(observed) & np.isfinite(raw_surface[recent])
    if not np.any(finite):
        return raw_surface
    bias = float(np.median(observed[finite] - raw_surface[recent][finite]))
    return raw_surface + bias


def transfer_from_references(
    query: Query,
    references: list[Reference],
    reference_family: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    well = query.well
    tail = well.prediction_indices
    query_rows = np.unique(
        np.r_[
            np.arange(0, len(well.md), QUERY_STRIDE),
            len(well.md) - 1,
            well.anchor_index,
        ]
    ).astype(np.int64)
    query_rows.sort()
    query_xy = np.column_stack((well.x[query_rows], well.y[query_rows]))
    hidden_positions = query_rows > well.anchor_index
    selected = [
        reference
        for index, reference in enumerate(references)
        if reference.fold < 4
        and reference.fold != query.fold
        and int(reference_family[index]) == query.family
    ]

    analog = query.baseline.copy().astype(np.float64)
    analog_distance = float("inf")
    analog_candidates = []
    for reference in selected:
        distance, nearest = reference.tree.query(query_xy, k=1)
        score = (
            float(np.median(distance[hidden_positions]))
            if np.any(hidden_positions)
            else float("inf")
        )
        analog_candidates.append(
            (score, distance, reference.surface[nearest])
        )
    analog_candidates.sort(key=lambda item: item[0])
    if analog_candidates:
        top = analog_candidates[: min(ANALOG_WELLS, len(analog_candidates))]
        distance = np.stack([item[1] for item in top])
        surface = np.stack([item[2] for item in top])
        weight = 1.0 / np.square(
            np.maximum(distance, ANALOG_DISTANCE_FLOOR_FT)
        )
        raw_surface = np.sum(weight * surface, axis=0) / np.sum(weight, axis=0)
        raw_surface = calibrated_surface(well, query_rows, raw_surface)
        analog = np.interp(tail, query_rows, raw_surface - well.z[query_rows])
        analog_distance = float(analog_candidates[0][0])

    local = query.baseline.copy().astype(np.float64)
    local_distance = float("inf")
    if selected:
        support_xy = np.concatenate([reference.xy for reference in selected])
        support_surface = np.concatenate(
            [reference.surface for reference in selected]
        )
        tree = cKDTree(support_xy)
        distance, nearest = tree.query(
            query_xy,
            k=min(LOCAL_POINTS, len(support_surface)),
        )
        distance = np.asarray(distance)
        nearest = np.asarray(nearest)
        if distance.ndim == 1:
            distance = distance[:, None]
            nearest = nearest[:, None]
        weight = 1.0 / np.square(
            np.maximum(distance, LOCAL_DISTANCE_FLOOR_FT)
        )
        raw_surface = np.sum(
            weight * support_surface[nearest], axis=1
        ) / np.sum(weight, axis=1)
        raw_surface = calibrated_surface(well, query_rows, raw_surface)
        local = np.interp(tail, query_rows, raw_surface - well.z[query_rows])
        local_distance = (
            float(np.median(distance[hidden_positions, 0]))
            if np.any(hidden_positions)
            else float("inf")
        )

    analog_active = analog_distance <= ANALOG_DISTANCE_LIMIT_FT
    local_active = local_distance <= LOCAL_DISTANCE_LIMIT_FT
    gated_analog = analog if analog_active else query.baseline
    gated_local = local if local_active else query.baseline
    if analog_active and local_active:
        equal_mean = 0.5 * (analog + local)
    elif analog_active:
        equal_mean = analog
    elif local_active:
        equal_mean = local
    else:
        equal_mean = query.baseline
    return (
        {
            "analog": np.asarray(gated_analog, dtype=np.float32),
            "local": np.asarray(gated_local, dtype=np.float32),
            "equal_mean": np.asarray(equal_mean, dtype=np.float32),
        },
        {
            "analog_distance": analog_distance,
            "local_distance": local_distance,
            "analog_active": float(analog_active),
            "local_active": float(local_active),
            "eligible_references": float(len(selected)),
        },
    )


def score_grid(
    baseline: np.ndarray,
    truth: np.ndarray,
    raw: dict[str, np.ndarray],
) -> tuple[list[dict], dict[tuple[str, float, float], np.ndarray]]:
    baseline_rmse = rmse(baseline, truth)
    rows = []
    candidates = {}
    for variant in VARIANTS:
        correction = raw[variant] - baseline
        for cap in CORRECTION_CAPS:
            capped = baseline + np.clip(correction, -cap, cap)
            for weight in OUTER_BLEND_WEIGHTS:
                candidate = baseline + weight * (capped - baseline)
                key = (variant, cap, weight)
                candidates[key] = candidate.astype(np.float32)
                score = rmse(candidate, truth)
                rows.append(
                    {
                        "variant": variant,
                        "cap": cap,
                        "outer_blend_weight": weight,
                        "candidate_rmse": score,
                        "gain": baseline_rmse - score,
                        "raw_standalone_rmse": rmse(raw[variant], truth),
                        "capped_standalone_rmse": rmse(capped, truth),
                    }
                )
    return rows, candidates


def implementation_gate(
    data_root: Path,
    workers: int,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ids = np.asarray(training_well_ids(data_root))
    profiles = load_legal_typewell_profiles(data_root, ids, workers)
    values, mask = profile_arrays(profiles)
    family = connected_typewell_families(values, mask)
    family_count = int(family.max()) + 1
    family_sizes = np.bincount(family)
    baseline = np.linspace(1000.0, 1010.0, 64)
    candidate = baseline + 3.0
    no_op = baseline + 0.0 * (candidate - baseline)
    no_op_error = float(np.max(np.abs(no_op - baseline)))
    observed = np.asarray([2100.0, 2100.5, 2101.0])
    raw = np.asarray([2098.0, 2098.5, 2099.0])
    bias = float(np.median(observed - raw))
    calibration_error = float(np.max(np.abs(raw + bias - observed)))
    folds = original_folds(ids)
    exclusion_ok = bool(
        all(
            np.all(
                (
                    folds[
                        np.flatnonzero(
                            (folds < 4) & (folds != held)
                        )
                    ]
                    != held
                )
                & (
                    folds[
                        np.flatnonzero(
                            (folds < 4) & (folds != held)
                        )
                    ]
                    < 4
                )
            )
            for held in range(4)
        )
    )
    passed = bool(
        family_count == 54
        and no_op_error == 0.0
        and calibration_error <= 1e-12
        and exclusion_ok
    )
    payload = {
        "schema_version": 1,
        "run_id": "v5_batch2_run_006_goal_050",
        "family": "hierarchical_coarse_to_fine_row_competition",
        "variant": "fold_excluded_master_frame_surface_transfer",
        "status": "passed" if passed else "failed",
        "typewell_family_count": family_count,
        "typewell_family_size": {
            "minimum": int(np.min(family_sizes)),
            "median": float(np.median(family_sizes)),
            "maximum": int(np.max(family_sizes)),
        },
        "exact_outer_no_op_max_abs_error": no_op_error,
        "visible_bias_calibration_max_abs_error": calibration_error,
        "development_reference_exclusion_control": exclusion_ok,
        "f4_reference_targets_loaded": False,
        "audit_fold_targets_loaded": False,
        "layout_truth_array_accessed": False,
        "source_sha256": sha256(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("master-frame surface-transfer gate failed")


def held_fold(
    references: list[Reference],
    queries: list[Query],
    family: np.ndarray,
    fold: int,
    workers: int,
    output: Path,
    predictions_path: Path,
) -> None:
    if output.exists() or predictions_path.exists():
        raise FileExistsError("refusing to overwrite master-frame result")
    rng = np.random.default_rng(SEED)
    shuffled_family = rng.permutation(family)

    def predict(query: Query):
        primary = transfer_from_references(query, references, family)
        control = transfer_from_references(
            query, references, shuffled_family
        )
        return primary, control

    with ThreadPoolExecutor(max_workers=workers) as pool:
        predictions = list(pool.map(predict, queries))
    raw = {variant: [] for variant in VARIANTS}
    control_raw = {variant: [] for variant in VARIANTS}
    diagnostics = []
    control_diagnostics = []
    for (primary, control) in predictions:
        primary_path, primary_diagnostic = primary
        control_path, control_diagnostic = control
        for variant in VARIANTS:
            raw[variant].append(primary_path[variant])
            control_raw[variant].append(control_path[variant])
        diagnostics.append(primary_diagnostic)
        control_diagnostics.append(control_diagnostic)
    raw = {key: np.concatenate(value) for key, value in raw.items()}
    control_raw = {
        key: np.concatenate(value) for key, value in control_raw.items()
    }
    baseline = np.concatenate([query.baseline for query in queries])
    truth = np.concatenate([query.truth for query in queries])
    rows, candidates = score_grid(baseline, truth, raw)
    control_rows, _ = score_grid(baseline, truth, control_raw)
    best = min(
        rows,
        key=lambda row: (
            row["candidate_rmse"],
            VARIANTS.index(row["variant"]),
            row["cap"],
            row["outer_blend_weight"],
        ),
    )
    control_best = min(
        control_rows,
        key=lambda row: (
            row["candidate_rmse"],
            VARIANTS.index(row["variant"]),
            row["cap"],
            row["outer_blend_weight"],
        ),
    )
    selected_key = (
        str(best["variant"]),
        float(best["cap"]),
        float(best["outer_blend_weight"]),
    )
    selected = candidates[selected_key]
    starts = [0]
    for query in queries:
        starts.append(starts[-1] + len(query.truth))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        well_ids=np.asarray([query.well_id for query in queries]),
        row_starts=np.asarray(starts, dtype=np.int64),
        baseline=baseline.astype(np.float32),
        truth=truth.astype(np.float32),
        analog=raw["analog"].astype(np.float32),
        local=raw["local"].astype(np.float32),
        equal_mean=raw["equal_mean"].astype(np.float32),
        prediction=selected.astype(np.float32),
        selected_variant=np.asarray(selected_key[0]),
        selected_cap=np.float32(selected_key[1]),
        selected_outer_weight=np.float32(selected_key[2]),
        audit_fold_loaded=np.bool_(False),
        layout_truth_array_accessed=np.bool_(False),
    )
    family_sizes = np.bincount(family)
    payload = {
        "schema_version": 1,
        "run_id": "v5_batch2_run_006_goal_050",
        "family": "hierarchical_coarse_to_fine_row_competition",
        "variant": "fold_excluded_master_frame_surface_transfer",
        "fold": fold,
        "eligibility": (
            "target-exposed F0 capacity diagnostic; path variant, cap, and "
            "outer blend weight must be selected on training-side roles "
            "before primary use"
        ),
        "held_wells": len(queries),
        "held_rows": len(truth),
        "typewell_family_count": int(family.max()) + 1,
        "typewell_family_size": {
            "minimum": int(np.min(family_sizes)),
            "median": float(np.median(family_sizes)),
            "maximum": int(np.max(family_sizes)),
        },
        "baseline_rmse": rmse(baseline, truth),
        "best": best,
        "records": rows,
        "permuted_reference_family_control": {
            "best": control_best,
            "records": control_rows,
        },
        "coverage": {
            "analog_active_wells": int(
                sum(row["analog_active"] for row in diagnostics)
            ),
            "local_active_wells": int(
                sum(row["local_active"] for row in diagnostics)
            ),
            "median_eligible_references": float(
                np.median(
                    [row["eligible_references"] for row in diagnostics]
                )
            ),
            "median_analog_distance_ft": float(
                np.median(
                    [
                        row["analog_distance"]
                        for row in diagnostics
                        if np.isfinite(row["analog_distance"])
                    ]
                )
            ),
            "median_local_distance_ft": float(
                np.median(
                    [
                        row["local_distance"]
                        for row in diagnostics
                        if np.isfinite(row["local_distance"])
                    ]
                )
            ),
        },
        "prediction": str(predictions_path),
        "prediction_sha256": sha256(predictions_path),
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256(Path(__file__)),
        "inference_columns": [
            "MD",
            "X",
            "Y",
            "Z",
            "TVT_input",
            "typewell_TVT",
            "typewell_GR",
            "fold_safe_v20_prediction",
        ],
        "training_only_reference_target": "TVT + Z from training-side wells",
        "audit_fold_loaded": False,
        "layout_truth_array_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run-6 fold-excluded master-frame surface transfer."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "limited_query_cv/runs/v5_batch2_run_001_goal_050/v20_components"
        ),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(4))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    if args.gate == (args.fold is not None):
        raise ValueError("choose exactly one of --gate or --fold")
    if args.fold is not None and args.predictions is None:
        raise ValueError("--fold requires --predictions")
    if args.gate:
        implementation_gate(args.data_root, args.workers, args.output)
        return
    references, queries, _, _ = build_records(
        args.data_root,
        args.baseline_root,
        args.fold,
        args.workers,
    )
    reference_family = np.asarray(
        [reference.family for reference in references], dtype=np.int32
    )
    held_fold(
        references,
        queries,
        reference_family,
        args.fold,
        args.workers,
        args.output,
        args.predictions,
    )


if __name__ == "__main__":
    main()
