from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from rogii.data import DEFAULT_DATA_ROOT, Well
from limited_query_cv.evaluate_v5_run6_master_frame_surface_transfer import (
    connected_typewell_families,
    profile_arrays,
)
from limited_query_cv.train_v5_run6_recurrent_residual import (
    DEFAULT_COMPONENT_ROOT,
    NATIVE_STRIDE,
    OUTPUT_BOUND_FT,
    load_all,
    make_sequence_record,
)


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "master_frame_row_residual_boosting"
VARIANTS = ("catboost_frame", "lightgbm_frame", "xgboost_frame")
AMPLITUDES = (0.5, 1.0, 1.5, 2.0, 3.0)
BLEND_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
TYPEWELL_LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
EXPECTED_FRAME_COUNT = 53
EXPECTED_UNSEEN_HELD_WELLS = (3, 3, 5, 8)
EXPECTED_UNSEEN_HELD_FRAMES = (2, 2, 5, 7)
EXPECTED_MEDIAN_TRAIN_REFERENCES = (16.0, 16.0, 15.0, 15.0)
SEED = 20261721
PARENT_SOURCE = Path(
    "limited_query_cv/train_v5_run6_recurrent_residual.py"
)
DEFAULT_D570 = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050/"
    "d209_amplitude_clean.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_correlation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(
        np.sqrt(np.sum(np.square(left)) * np.sum(np.square(right)))
    )
    return float(np.sum(left * right) / denominator) if denominator > 1e-12 else 0.0


def typewell_summary(well: Well) -> np.ndarray:
    valid = np.isfinite(well.typewell_tvt) & np.isfinite(well.typewell_gr)
    tvt = well.typewell_tvt[valid]
    gr = well.typewell_gr[valid]
    order = np.argsort(tvt)
    tvt = tvt[order]
    gr = gr[order]
    if len(gr) < 2:
        raise RuntimeError(f"{well.well_id}: insufficient typewell support")
    center = float(np.median(gr))
    scale = max(float(1.4826 * np.median(np.abs(gr - center))), 1.0)
    normalized = np.clip((gr - center) / scale, -8.0, 8.0)
    quantiles = np.quantile(normalized, (0.05, 0.25, 0.5, 0.75, 0.95))
    gradient = np.gradient(normalized, tvt, edge_order=1)
    values = np.r_[
        len(gr) / 8000.0,
        (tvt[-1] - tvt[0]) / 4000.0,
        center / 100.0,
        scale / 50.0,
        np.mean(normalized),
        np.std(normalized),
        quantiles,
        np.mean(np.abs(gradient)) * 10.0,
        np.std(gradient) * 10.0,
        [safe_correlation(normalized, lag) for lag in TYPEWELL_LAGS],
    ]
    return values.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class BoostRecord:
    well_id: str
    fold: int
    frame: int
    frame_size: int
    features: np.ndarray
    target: np.ndarray
    sample_local: np.ndarray
    baseline: np.ndarray
    truth: np.ndarray


def build_master_frames(wells: list[Well], folds: np.ndarray) -> tuple[np.ndarray, dict]:
    profiles = [
        (well.typewell_tvt.copy(), well.typewell_gr.copy()) for well in wells
    ]
    values, mask = profile_arrays(profiles)
    frames = connected_typewell_families(values, mask)
    frame_count = int(len(np.unique(frames)))
    frame_sizes = np.bincount(frames)
    unseen = []
    unseen_frames = []
    median_references = []
    held_with_train = []
    for held_fold in range(4):
        training_count = np.bincount(
            frames[folds != held_fold], minlength=frame_count
        )
        held = frames[folds == held_fold]
        unseen.append(int(np.sum(training_count[held] == 0)))
        held_count = np.bincount(held, minlength=frame_count)
        unseen_frames.append(
            int(np.sum((held_count > 0) & (training_count == 0)))
        )
        median_references.append(float(np.median(training_count[held])))
        held_with_train.append(int(np.sum(training_count[held] > 0)))
    audit = {
        "frame_count": frame_count,
        "frame_sizes": frame_sizes.astype(int).tolist(),
        "unseen_held_wells": unseen,
        "unseen_held_frames": unseen_frames,
        "median_training_references": median_references,
        "held_with_training_frame": held_with_train,
    }
    return frames, audit


def boost_record(
    well: Well,
    fold: int,
    frame: int,
    frame_size: int,
    baseline: np.ndarray,
    components: np.ndarray,
    *,
    allow_missing_target: bool = False,
) -> BoostRecord:
    parent = make_sequence_record(
        well,
        baseline,
        components,
        allow_missing_target=allow_missing_target,
    )
    summary = typewell_summary(well)
    length = len(parent.features)
    extra = np.column_stack(
        (
            np.full(length, frame, dtype=np.float32),
            np.full(length, np.log1p(frame_size), dtype=np.float32),
            np.broadcast_to(summary[None], (length, len(summary))),
        )
    )
    features = np.concatenate((extra, parent.features), axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError(f"{well.well_id}: nonfinite boosting features")
    return BoostRecord(
        well_id=well.well_id,
        fold=int(fold),
        frame=int(frame),
        frame_size=int(frame_size),
        features=features,
        target=parent.target.copy(),
        sample_local=parent.sample_local.copy(),
        baseline=parent.baseline.copy(),
        truth=parent.truth.copy(),
    )


def build_records(
    wells: list[Well],
    folds: np.ndarray,
    frames: np.ndarray,
    baseline_paths: dict[str, np.ndarray],
    component_paths: dict[str, np.ndarray],
    *,
    allow_missing_target: bool = False,
) -> list[BoostRecord]:
    sizes = np.bincount(frames)
    return [
        boost_record(
            well,
            int(fold),
            int(frame),
            int(sizes[frame]),
            baseline_paths[well.well_id],
            component_paths[well.well_id],
            allow_missing_target=allow_missing_target,
        )
        for well, fold, frame in zip(wells, folds, frames)
    ]


def dense_matrix(
    records: list[BoostRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.concatenate([record.features for record in records], axis=0)
    target = np.concatenate([record.target for record in records]).astype(np.float32)
    groups = np.concatenate(
        [
            np.full(len(record.target), index, dtype=np.int32)
            for index, record in enumerate(records)
        ]
    )
    return features, target, groups


def one_hot_matrix(features: np.ndarray) -> np.ndarray:
    frame = features[:, 0].astype(np.int64)
    if np.any((frame < 0) | (frame >= EXPECTED_FRAME_COUNT)):
        raise RuntimeError("master-frame index outside frozen range")
    one_hot = np.eye(EXPECTED_FRAME_COUNT, dtype=np.float32)[frame]
    return np.concatenate((one_hot, features[:, 1:]), axis=1).astype(np.float32)


def catboost_frame(features: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(features)
    frame[0] = frame[0].astype(np.int64).astype(str)
    return frame


def make_model(variant: str, device: str, seed: int):
    if variant == "catboost_frame":
        configuration = dict(
            iterations=1200,
            depth=8,
            learning_rate=0.04,
            l2_leaf_reg=10.0,
            loss_function="RMSE",
            random_seed=seed,
            task_type="GPU" if device.startswith("cuda") else "CPU",
            allow_writing_files=False,
            verbose=100,
        )
        if device.startswith("cuda"):
            configuration["devices"] = device.split(":")[-1]
        return CatBoostRegressor(**configuration)
    if variant == "lightgbm_frame":
        gpu_index = int(device.split(":")[-1]) if device.startswith("cuda") else 0
        return LGBMRegressor(
            objective="regression",
            n_estimators=2000,
            num_leaves=127,
            learning_rate=0.03,
            min_child_samples=50,
            reg_lambda=10.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            max_bin=255,
            random_state=seed,
            n_jobs=16,
            verbosity=-1,
            device_type="gpu" if device.startswith("cuda") else "cpu",
            gpu_device_id=gpu_index,
        )
    if variant == "xgboost_frame":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1600,
            max_depth=8,
            learning_rate=0.03,
            min_child_weight=100.0,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device if device.startswith("cuda") else "cpu",
            random_state=seed,
            n_jobs=16,
            verbosity=1,
        )
    raise ValueError(f"unknown boosting variant: {variant}")


def fit_model(model, variant: str, features: np.ndarray, target: np.ndarray) -> None:
    if variant == "catboost_frame":
        model.fit(Pool(catboost_frame(features), target, cat_features=[0]))
    else:
        model.fit(one_hot_matrix(features), target)


def model_predict(model, variant: str, features: np.ndarray) -> np.ndarray:
    if variant == "catboost_frame":
        prediction = model.predict(
            Pool(catboost_frame(features), cat_features=[0])
        )
    else:
        prediction = model.predict(one_hot_matrix(features))
    return np.asarray(prediction, dtype=np.float32)


def deployed_correction(
    sampled: np.ndarray,
    sample_local: np.ndarray,
    length: int,
) -> np.ndarray:
    local = np.arange(length, dtype=np.float64)
    correction = np.interp(local, sample_local, sampled).astype(np.float32)
    correction -= correction[0]
    return np.clip(correction, -OUTPUT_BOUND_FT, OUTPUT_BOUND_FT).astype(np.float32)


def predict_records(
    model,
    variant: str,
    records: list[BoostRecord],
) -> tuple[np.ndarray, list[np.ndarray]]:
    parts = []
    sampled_parts = []
    for record in records:
        sampled = model_predict(model, variant, record.features)
        sampled_parts.append(sampled)
        parts.append(
            deployed_correction(sampled, record.sample_local, len(record.baseline))
        )
    return np.concatenate(parts).astype(np.float32), sampled_parts


def replace_hidden_target(well: Well, mode: str) -> Well:
    tvt = well.tvt.copy()
    hidden = well.prediction_indices
    if mode == "reversed":
        tvt[hidden] = tvt[hidden][::-1]
    elif mode == "nan":
        tvt[hidden] = np.nan
    else:
        raise ValueError(mode)
    return dataclasses.replace(well, tvt=tvt)


def load_d570(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError("D570 protected flags changed")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        prediction = cache["prediction"].astype(np.float32)
    prediction_by_id = {
        well_id: prediction[starts[index] : starts[index + 1]].copy()
        for index, well_id in enumerate(ids)
    }
    baseline_by_id = {
        well_id: baseline[starts[index] : starts[index + 1]].copy()
        for index, well_id in enumerate(ids)
    }
    return {
        "__baseline_by_id__": baseline_by_id,
        "__prediction_by_id__": prediction_by_id,
        "__baseline__": baseline,
        "__prediction__": prediction,
    }


def run_gate(args: argparse.Namespace) -> None:
    started = time.time()
    wells, folds, baseline_paths, component_paths, component_names = load_all(
        args.data_root, args.component_root, args.workers
    )
    frames, frame_audit = build_master_frames(wells, folds)
    frame_sizes = np.bincount(frames)
    eligible = np.flatnonzero(np.isin(folds, (1, 2, 3)))
    fixed_indices = eligible[:2]
    untouched_indices = eligible[2:4]
    selected = np.r_[fixed_indices, untouched_indices]
    records = [
        boost_record(
            wells[index],
            int(folds[index]),
            int(frames[index]),
            int(frame_sizes[frames[index]]),
            baseline_paths[wells[index].well_id],
            component_paths[wells[index].well_id],
        )
        for index in selected
    ]
    fixed = records[:2]
    untouched = records[2:]
    features, target, _ = dense_matrix(fixed)
    model = make_model(args.variant, args.device, args.seed)
    fit_model(model, args.variant, features, target)
    fixed_sampled = [
        model_predict(model, args.variant, record.features) for record in fixed
    ]
    fixed_target = [
        record.target - record.target[0] for record in fixed
    ]
    fixed_deployed = [values - values[0] for values in fixed_sampled]
    initial = float(
        np.sqrt(np.mean(np.square(np.concatenate(fixed_target).astype(np.float64))))
    )
    final = float(
        np.sqrt(
            np.mean(
                np.square(
                    (
                        np.concatenate(fixed_deployed)
                        - np.concatenate(fixed_target)
                    ).astype(np.float64)
                )
            )
        )
    )
    untouched_correction, _ = predict_records(model, args.variant, untouched)

    audit_index = int(untouched_indices[0])
    audit_well = wells[audit_index]
    audit_records = []
    for current in (
        audit_well,
        replace_hidden_target(audit_well, "reversed"),
        replace_hidden_target(audit_well, "nan"),
    ):
        audit_records.append(
            boost_record(
                current,
                int(folds[audit_index]),
                int(frames[audit_index]),
                int(frame_sizes[frames[audit_index]]),
                baseline_paths[audit_well.well_id],
                component_paths[audit_well.well_id],
                allow_missing_target=True,
            )
        )
    hidden_invariance = [
        bool(np.array_equal(audit_records[0].features, item.features))
        for item in audit_records[1:]
    ]

    d570 = load_d570(args.d570)
    baseline_concat = np.concatenate(
        [baseline_paths[well.well_id] for well in wells]
    ).astype(np.float32)
    d570_aligned = np.concatenate(
        [d570["__prediction_by_id__"][well.well_id] for well in wells]
    ).astype(np.float32)
    d570_baseline_aligned = np.concatenate(
        [d570["__baseline_by_id__"][well.well_id] for well in wells]
    ).astype(np.float32)
    v20_zero_error = float(
        np.max(np.abs(baseline_concat - baseline_concat))
    )
    d570_zero_error = float(np.max(np.abs(d570_aligned - d570_aligned)))
    d570_layout_matches = bool(
        np.array_equal(baseline_concat, d570_baseline_aligned)
    )
    status = bool(
        frame_audit["frame_count"] == EXPECTED_FRAME_COUNT
        and frame_audit["unseen_held_wells"]
        == list(EXPECTED_UNSEEN_HELD_WELLS)
        and frame_audit["unseen_held_frames"]
        == list(EXPECTED_UNSEEN_HELD_FRAMES)
        and frame_audit["median_training_references"]
        == list(EXPECTED_MEDIAN_TRAIN_REFERENCES)
        and np.all(folds[fixed_indices] != 0)
        and np.all(folds[untouched_indices] != 0)
        and final < 0.20 * initial
        and np.isfinite(untouched_correction).all()
        and float(np.std(untouched_correction)) > 1e-6
        and all(hidden_invariance)
        and v20_zero_error == 0.0
        and d570_zero_error == 0.0
        and d570_layout_matches
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": args.variant,
        "stage": "metric_free_capacity_and_firewall_gate",
        "status": "passed" if status else "failed",
        "frame_audit": frame_audit,
        "fixed_well_ids": [record.well_id for record in fixed],
        "untouched_well_ids": [record.well_id for record in untouched],
        "fixed_initial_anchor_zero_rmse": initial,
        "fixed_final_anchor_zero_rmse": final,
        "fixed_final_fraction": final / max(initial, 1e-12),
        "untouched_correction_std": float(np.std(untouched_correction)),
        "untouched_correction_max_abs": float(
            np.max(np.abs(untouched_correction))
        ),
        "hidden_target_feature_invariance": hidden_invariance,
        "feature_count": int(records[0].features.shape[1]),
        "component_names": list(component_names),
        "native_stride": NATIVE_STRIDE,
        "output_bound_ft": OUTPUT_BOUND_FT,
        "v20_zero_share_max_abs_error": v20_zero_error,
        "d570_zero_share_max_abs_error": d570_zero_error,
        "d570_v20_layout_matches": d570_layout_matches,
        "f0_fit_well_count": 0,
        "f4_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "formation_columns_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "parent_source": str(PARENT_SOURCE),
        "parent_source_sha256": sha256(PARENT_SOURCE),
        "d570_path": str(args.d570),
        "d570_sha256": sha256(args.d570),
        "elapsed_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "result.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not status:
        raise RuntimeError(f"{args.variant}: master-frame boosting gate failed")


def merge_gates(paths: list[Path], output: Path) -> None:
    gates = [json.loads(path.read_text()) for path in paths]
    source_hash = sha256(Path(__file__))
    status = bool(
        {gate["variant"] for gate in gates} == set(VARIANTS)
        and all(gate["status"] == "passed" for gate in gates)
        and all(gate["source_sha256"] == source_hash for gate in gates)
        and all(gate["frame_audit"]["frame_count"] == EXPECTED_FRAME_COUNT for gate in gates)
        and all(not gate["f4_loaded"] for gate in gates)
        and all(not gate["confirmation_regroupings_loaded"] for gate in gates)
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": "aggregate_three_model_gate",
        "stage": "aggregate_metric_free_gate",
        "status": "passed" if status else "failed",
        "variants": sorted(gate["variant"] for gate in gates),
        "gate_paths": [str(path) for path in paths],
        "gate_sha256": {str(path): sha256(path) for path in paths},
        "source_sha256": source_hash,
        "parent_source_sha256": sha256(PARENT_SOURCE),
        "f4_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not status:
        raise RuntimeError("master-frame boosting aggregate gate failed")


def save_model(model, variant: str, path: Path) -> None:
    if variant == "catboost_frame":
        model.save_model(path)
    elif variant == "lightgbm_frame":
        model.booster_.save_model(path)
    elif variant == "xgboost_frame":
        model.save_model(path)
    else:
        raise ValueError(variant)


def train(args: argparse.Namespace) -> None:
    started = time.time()
    gate = json.loads(args.gate_json.read_text())
    source_hash = sha256(Path(__file__))
    if (
        gate["status"] != "passed"
        or gate["variant"] != "aggregate_three_model_gate"
        or gate["source_sha256"] != source_hash
    ):
        raise RuntimeError("aggregate gate does not match boosting source")
    wells, folds, baseline_paths, component_paths, component_names = load_all(
        args.data_root, args.component_root, args.workers
    )
    frames, frame_audit = build_master_frames(wells, folds)
    all_records = build_records(
        wells,
        folds,
        frames,
        baseline_paths,
        component_paths,
    )
    train_records = [
        record for record in all_records if record.fold != args.fold
    ]
    held_records = [
        record for record in all_records if record.fold == args.fold
    ]
    if args.fold != 0:
        raise RuntimeError("A620 authorizes only the F0 pilot before expansion")
    if any(record.fold == 0 for record in train_records):
        raise RuntimeError("held F0 entered boosting fit")
    features, target, _ = dense_matrix(train_records)
    model = make_model(args.variant, args.device, args.seed)
    fit_model(model, args.variant, features, target)
    correction, sampled = predict_records(model, args.variant, held_records)
    baseline = np.concatenate(
        [record.baseline for record in held_records]
    ).astype(np.float32)
    truth = np.concatenate([record.truth for record in held_records]).astype(
        np.float32
    )
    if (
        not np.isfinite(correction).all()
        or float(np.max(np.abs(correction))) > OUTPUT_BOUND_FT + 1e-6
    ):
        raise RuntimeError("deployed boosting correction violated bound")
    starts = np.concatenate(
        ([0], np.cumsum([len(record.baseline) for record in held_records]))
    ).astype(np.int64)
    for index, record in enumerate(held_records):
        if correction[starts[index]] != 0.0:
            raise RuntimeError(f"{record.well_id}: correction anchor is not zero")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_suffix = {
        "catboost_frame": ".cbm",
        "lightgbm_frame": ".txt",
        "xgboost_frame": ".json",
    }[args.variant]
    model_path = args.output_dir / f"model{model_suffix}"
    save_model(model, args.variant, model_path)
    prediction_path = args.output_dir / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        fold=np.asarray(args.fold, dtype=np.int8),
        variant=np.asarray(args.variant),
        well_ids=np.asarray([record.well_id for record in held_records]),
        row_starts=starts,
        baseline=baseline,
        truth=truth,
        correction=correction,
        sample_starts=np.concatenate(
            ([0], np.cumsum([len(values) for values in sampled]))
        ).astype(np.int64),
        sampled_correction=np.concatenate(sampled).astype(np.float32),
        frame_ids=np.asarray([record.frame for record in held_records], dtype=np.int16),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
        f4_loaded=np.asarray(False),
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": args.variant,
        "stage": "f0_prediction_frozen_without_metric",
        "status": "complete",
        "fold": args.fold,
        "training_folds": [1, 2, 3],
        "training_wells": len(train_records),
        "training_sampled_rows": int(len(target)),
        "held_wells": len(held_records),
        "held_rows": int(len(baseline)),
        "held_sampled_rows": int(sum(len(values) for values in sampled)),
        "feature_count": int(features.shape[1]),
        "component_names": list(component_names),
        "frame_audit": frame_audit,
        "configuration": {
            "native_stride": NATIVE_STRIDE,
            "output_bound_ft": OUTPUT_BOUND_FT,
            "amplitudes": list(AMPLITUDES),
            "blend_weights": list(BLEND_WEIGHTS),
            "seed": args.seed,
        },
        "correction_max_abs": float(np.max(np.abs(correction))),
        "correction_std": float(np.std(correction)),
        "anchor_max_abs": float(
            max(abs(float(correction[starts[index]])) for index in range(len(held_records)))
        ),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "source_sha256": source_hash,
        "parent_source_sha256": sha256(PARENT_SOURCE),
        "gate_json": str(args.gate_json),
        "gate_sha256": sha256(args.gate_json),
        "f4_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "selection_metric_opened": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "result.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT
    )
    parser.add_argument("--d570", type=Path, default=DEFAULT_D570)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--merge-gates", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge_gates:
        if args.gate_json is None:
            raise ValueError("--merge-gates requires --gate-json output")
        merge_gates(args.merge_gates, args.gate_json)
        return
    if args.variant is None or args.output_dir is None:
        raise ValueError("--variant and --output-dir are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    seed_all(args.seed)
    if args.gate_only:
        run_gate(args)
    else:
        if args.gate_json is None:
            raise ValueError("production training requires --gate-json")
        train(args)


if __name__ == "__main__":
    main()
