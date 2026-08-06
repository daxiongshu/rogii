from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from limited_query_cv.v5_batch3_run2_surface import (
    delayed_dip_channels,
    typewell_clusters,
)


DATA_ROOT = Path(
    "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
)
RUN_ROOT = Path(
    "limited_query_cv/runs/v5_batch3_run_002_goal_020"
)
V20_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_001_goal_050/v20_components"
)
PARENT = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit/sealed/outer_predictions.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(values: np.ndarray) -> list[float]:
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


def correlation(first: np.ndarray, second: np.ndarray) -> float:
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


def load_typewell(
    well_id: str,
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        data_root / "train" / f"{well_id}__typewell.csv",
        usecols=["TVT", "GR"],
    ).dropna()
    return (
        frame["TVT"].to_numpy(np.float64),
        frame["GR"].to_numpy(np.float64),
    )


def all_development_clusters(
    parent: Path,
    data_root: Path,
) -> dict[str, int]:
    with np.load(parent, allow_pickle=False) as archive:
        ids = archive["well_ids"].astype(str)
        folds = archive["folds"].astype(np.int64)
    typewells = {
        str(well_id): load_typewell(str(well_id), data_root)
        for well_id, fold in zip(ids, folds)
        if fold < 4
    }
    return typewell_clusters(typewells)


def well_features(
    well_id: str,
    clean: np.ndarray,
    v20: np.ndarray,
    surfaces: np.ndarray,
    cluster_size: int,
    cluster_id: int,
    data_root: Path,
) -> tuple[np.ndarray, str]:
    horizontal = pd.read_csv(
        data_root / "train" / f"{well_id}__horizontal_well.csv",
        usecols=["MD", "X", "Y", "Z", "GR", "TVT_input"],
    )
    typewell = pd.read_csv(
        data_root / "train" / f"{well_id}__typewell.csv",
        usecols=["TVT", "GR"],
    ).dropna()
    hidden = horizontal["TVT_input"].isna().to_numpy()
    known = ~hidden
    if int(np.sum(hidden)) != len(clean):
        raise ValueError(f"{well_id}: surface/visible layout changed")
    md = horizontal["MD"].to_numpy(np.float64)
    x = horizontal["X"].to_numpy(np.float64)
    y = horizontal["Y"].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    gr = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    tail_md = md[hidden]
    tail_x = x[hidden]
    tail_y = y[hidden]
    tail_z = z[hidden]
    tail_gr = gr[hidden]
    prefix_tvt = tvt_input[known]
    features = [
        np.log1p(len(clean)),
        np.log1p(np.sum(known)),
        np.log1p(cluster_size),
        float(cluster_size == 0),
        *finite_stats(tail_md - tail_md[0]),
        *finite_stats(tail_x - np.mean(tail_x)),
        *finite_stats(tail_y - np.mean(tail_y)),
        *finite_stats(tail_z - tail_z[0]),
        *finite_stats(tail_gr),
        *finite_stats(np.diff(tail_gr)),
        *finite_stats(clean - clean[0]),
        *finite_stats(np.diff(clean)),
        *finite_stats(clean - v20),
        *finite_stats(np.diff(prefix_tvt[-min(256, len(prefix_tvt)) :])),
        float(np.mean(tail_x)),
        float(np.mean(tail_y)),
        float(np.hypot(tail_x[-1] - tail_x[0], tail_y[-1] - tail_y[0])),
        float(np.arctan2(tail_y[-1] - tail_y[0], tail_x[-1] - tail_x[0])),
        *finite_stats(typewell["TVT"].to_numpy(np.float64)),
        *finite_stats(typewell["GR"].to_numpy(np.float64)),
    ]
    dip, _ = delayed_dip_channels(
        md,
        z,
        shifts=(200, 250, 300, 350, 400),
        smoothing_rows=(31, 63, 127),
    )
    dip = dip[hidden]
    for index in range(dip.shape[1]):
        features.extend(finite_stats(dip[:, index]))
    corrections = surfaces - clean[None]
    for correction in corrections:
        features.extend(finite_stats(correction))
        features.extend(finite_stats(np.diff(correction)))
        features.extend(
            (
                correlation(correction, tail_z),
                correlation(correction, tail_gr),
                float(np.max(np.abs(correction))),
            )
        )
    ensemble_mean = np.mean(corrections, axis=0)
    ensemble_std = np.std(corrections, axis=0)
    features.extend(finite_stats(ensemble_mean))
    features.extend(finite_stats(ensemble_std))
    values = np.asarray(features, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{well_id}: non-finite selector feature")
    return values, f"cluster_{cluster_id}"


def load_fold(
    fold: int,
    clusters: dict[str, int],
    data_root: Path,
    *,
    truth_allowed: bool,
) -> dict:
    cache_path = RUN_ROOT / f"surface_f{fold}_cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        if (
            int(cache["fold"]) != fold
            or bool(cache["held_well_BUDA_loaded"])
            or bool(cache["held_well_ANCC_loaded"])
            or bool(cache["held_hidden_TVT_loaded"])
            or bool(cache["hidden_truth_loaded"])
            or bool(cache["F4_loaded"])
            or bool(cache["F4_metric_computed"])
        ):
            raise ValueError(f"fold {fold} surface firewall failed")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        names = cache["variant_names"].astype(str)
        surfaces = cache["prediction"].astype(np.float64)
        clean = cache["clean_C016"].astype(np.float64)
        cluster_sizes = cache["cluster_pool_sizes"].astype(np.int64)
    v20_path = V20_ROOT / f"fold{fold}.npz"
    with np.load(v20_path, allow_pickle=False) as v20_archive:
        if (
            int(v20_archive["fold"]) != fold
            or bool(v20_archive["audit_fold_loaded"])
            or bool(v20_archive["layout_truth_array_accessed"])
        ):
            raise ValueError(f"fold {fold} V20 firewall failed")
        if not np.array_equal(v20_archive["well_ids"].astype(str), ids):
            raise ValueError(f"fold {fold} well layout changed")
        v20 = v20_archive["baseline"].astype(np.float64)
        truth = (
            v20_archive["truth"].astype(np.float64)
            if truth_allowed
            else None
        )
    numeric = []
    category = []
    target_weight = []
    cross = []
    energy = []
    rows = np.diff(starts).astype(np.int64)
    for index, well_id in enumerate(ids):
        start, stop = starts[index : index + 2]
        feature, cluster = well_features(
            str(well_id),
            clean[start:stop],
            v20[start:stop],
            surfaces[:, start:stop],
            int(cluster_sizes[index]),
            int(clusters[str(well_id)]),
            data_root,
        )
        numeric.append(feature)
        category.append(cluster)
        if truth_allowed:
            error = clean[start:stop] - truth[start:stop]
            correction = surfaces[0, start:stop] - clean[start:stop]
            local_cross = -float(np.sum(error * correction))
            local_energy = float(np.sum(np.square(correction)))
            cross.append(local_cross)
            energy.append(local_energy)
            target_weight.append(
                np.clip(local_cross / max(local_energy, 1e-12), 0.0, 0.15)
            )
    return {
        "fold": fold,
        "ids": ids,
        "starts": starts,
        "variant_names": names,
        "surfaces": surfaces,
        "clean": clean,
        "v20": v20,
        "truth": truth,
        "numeric": np.stack(numeric),
        "category": np.asarray(category),
        "target_weight": np.asarray(target_weight, dtype=np.float64),
        "cross": np.asarray(cross, dtype=np.float64),
        "energy": np.asarray(energy, dtype=np.float64),
        "rows": rows,
        "cache_path": cache_path,
        "v20_path": v20_path,
    }


def catboost_matrix(folds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.concatenate([fold["numeric"] for fold in folds], axis=0)
    category = np.concatenate([fold["category"] for fold in folds])
    return (
        np.column_stack((numeric, category.astype(object))),
        np.asarray([numeric.shape[1]], dtype=np.int64),
    )


def fit_regressor(
    train: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    categorical: np.ndarray,
    depth: int,
    iterations: int,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=0.035,
        l2_leaf_reg=5.0,
        random_seed=20263031 + int(depth),
        random_strength=0.15,
        verbose=False,
        allow_writing_files=False,
        thread_count=16,
    )
    model.fit(
        train,
        target,
        sample_weight=weights,
        cat_features=categorical.tolist(),
    )
    return model


def overfit_gate(
    training_folds: list[dict],
    output: Path,
) -> None:
    matrix, categorical = catboost_matrix(training_folds)
    target = np.concatenate(
        [fold["target_weight"] for fold in training_folds]
    )
    rows = np.concatenate([fold["rows"] for fold in training_folds])
    subset = np.arange(min(64, len(target)))
    before = float(np.sqrt(np.average(
        np.square(target[subset] - np.average(target[subset], weights=rows[subset])),
        weights=rows[subset],
    )))
    diagnostics = []
    for depth in (4, 8):
        model = fit_regressor(
            matrix[subset],
            target[subset],
            rows[subset],
            categorical,
            depth,
            2500,
        )
        prediction = np.clip(model.predict(matrix[subset]), 0.0, 0.15)
        after = float(np.sqrt(np.average(
            np.square(target[subset] - prediction),
            weights=rows[subset],
        )))
        diagnostics.append(
            {
                "depth": depth,
                "initial_constant_rmse": before,
                "final_training_rmse": after,
                "fraction_of_initial": after / max(before, 1e-12),
                "diagnosis": (
                    "capacity_can_overfit_fixed_batch"
                    if after < 0.25 * before
                    else "underfit_or_feature_collision"
                ),
            }
        )
    result = {
        "schema_version": 1,
        "protocol_revision": 11,
        "run_id": "v5_batch3_run_002_goal_020",
        "stage": "fixed_batch_overfit_gate_surface_selector",
        "status": (
            "passed"
            if any(
                item["fraction_of_initial"] < 0.25
                for item in diagnostics
            )
            else "failed"
        ),
        "batch_wells": len(subset),
        "diagnostics": diagnostics,
        "F4_loaded": False,
        "F4_metric_computed": False,
        "protected_reveal_spent": False,
        "source": str(Path(__file__)),
        "source_sha256": sha256(Path(__file__)),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(2)


def build_selector_cache(
    held: dict,
    training_folds: list[dict],
    output: Path,
) -> None:
    train_matrix, categorical = catboost_matrix(training_folds)
    held_matrix, held_categorical = catboost_matrix([held])
    if not np.array_equal(categorical, held_categorical):
        raise ValueError("selector categorical index changed")
    target = np.concatenate(
        [fold["target_weight"] for fold in training_folds]
    )
    row_weights = np.concatenate([fold["rows"] for fold in training_folds])
    cross = np.concatenate([fold["cross"] for fold in training_folds])
    energy = np.concatenate([fold["energy"] for fold in training_folds])
    global_weight = float(np.clip(
        np.sum(cross) / max(np.sum(energy), 1e-12),
        0.0,
        0.15,
    ))
    cluster_cross: dict[str, float] = {}
    cluster_energy: dict[str, float] = {}
    categories = np.concatenate(
        [fold["category"] for fold in training_folds]
    )
    for category, local_cross, local_energy in zip(
        categories,
        cross,
        energy,
    ):
        cluster_cross[str(category)] = (
            cluster_cross.get(str(category), 0.0) + float(local_cross)
        )
        cluster_energy[str(category)] = (
            cluster_energy.get(str(category), 0.0) + float(local_energy)
        )
    positive_energy = energy[energy > 0]
    typical_energy = (
        float(np.median(positive_energy)) if len(positive_energy) else 1.0
    )
    candidates = []
    names = []
    selected_weights = []

    for pseudo_wells in (2.0, 8.0, 32.0):
        weight = []
        prior_cross = global_weight * pseudo_wells * typical_energy
        prior_energy = pseudo_wells * typical_energy
        for category in held["category"]:
            numerator = cluster_cross.get(str(category), 0.0) + prior_cross
            denominator = cluster_energy.get(str(category), 0.0) + prior_energy
            weight.append(np.clip(numerator / denominator, 0.0, 0.15))
        weight = np.asarray(weight, dtype=np.float64)
        selected_weights.append(weight)
        names.append(f"cluster_surface_pseudo{pseudo_wells:g}")

    selected_weights.append(
        np.full(len(held["ids"]), global_weight, dtype=np.float64)
    )
    names.append("role_global_surface")
    for depth in (4, 6, 8):
        model = fit_regressor(
            train_matrix,
            target,
            row_weights,
            categorical,
            depth,
            1600,
        )
        predicted = np.clip(model.predict(held_matrix), 0.0, 0.15)
        selected_weights.append(0.5 * predicted)
        names.append(f"catboost_depth{depth}_half")
        selected_weights.append(predicted)
        names.append(f"catboost_depth{depth}_full")

    for weight in selected_weights:
        parts = []
        for index in range(len(held["ids"])):
            start, stop = held["starts"][index : index + 2]
            clean = held["clean"][start:stop]
            surface = held["surfaces"][0, start:stop]
            parts.append(clean + weight[index] * (surface - clean))
        candidates.append(np.concatenate(parts))
    prediction = np.stack(candidates).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray("v5_batch3_run_002_goal_020"),
        family=np.asarray("c016_buda_delayed_dip_cluster_surface"),
        held_fold=np.int64(held["fold"]),
        training_folds=np.asarray(
            [fold["fold"] for fold in training_folds],
            dtype=np.int64,
        ),
        well_ids=held["ids"],
        row_starts=held["starts"],
        variant_names=np.asarray(names),
        prediction=prediction,
        selected_weights=np.stack(selected_weights).astype(np.float32),
        global_training_weight=np.float64(global_weight),
        training_target_loaded=np.bool_(True),
        held_target_loaded=np.bool_(False),
        held_well_BUDA_loaded=np.bool_(False),
        held_well_ANCC_loaded=np.bool_(False),
        held_hidden_TVT_loaded=np.bool_(False),
        F4_loaded=np.bool_(False),
        F4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
        source_sha256=np.asarray(sha256(Path(__file__))),
        input_cache_sha256=np.asarray(
            [
                sha256(fold["cache_path"])
                for fold in training_folds + [held]
            ]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-fold", type=int, choices=range(4), default=0)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--overfit-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.overfit_output is None) == (args.output is None):
        raise ValueError("choose exactly one output mode")
    output = args.overfit_output or args.output
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    clusters = all_development_clusters(args.parent, args.data_root)
    training = [
        load_fold(
            fold,
            clusters,
            args.data_root,
            truth_allowed=True,
        )
        for fold in range(4)
        if fold != args.held_fold
    ]
    if args.overfit_output is not None:
        overfit_gate(training, args.overfit_output)
    else:
        held = load_fold(
            args.held_fold,
            clusters,
            args.data_root,
            truth_allowed=False,
        )
        build_selector_cache(held, training, args.output)
        print(
            f"held_fold={args.held_fold} training_folds="
            f"{[fold['fold'] for fold in training]} "
            f"elapsed_seconds={time.time() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
