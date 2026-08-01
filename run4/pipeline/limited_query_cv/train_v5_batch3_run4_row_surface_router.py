from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np

from limited_query_cv.train_v5_batch3_run2_reverse_bank import (
    conditional_weight,
)
from limited_query_cv.v5_batch3_run4_clean_bank import (
    RUN_ID,
    RUN_ROOT,
    load_clean_fold_bank,
)


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
ROW_FEATURE_ROOT = Path(
    "limited_query_cv/runs/v5_batch3_run_002_goal_020"
)
SURFACE_INDEX = 12
TRAINING_STRIDE = 4
AMPLITUDE_CAP = 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_weight(
    folds: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    held_fold: int,
) -> np.ndarray:
    training = folds != held_fold
    total_cross = np.sum(cross[training], axis=0)
    total_gram = np.sum(gram[training], axis=0)
    weight = np.zeros(49, dtype=np.float64)
    weight[:39] = conditional_weight(
        total_cross[:39],
        total_gram[:39, :39],
        28,
        np.arange(29, 39, dtype=np.int64),
        0.75,
        1e-4,
        0.75,
        0.75,
    )
    weight *= 0.875
    weight[17] = 0.0
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--held-fold", type=int, choices=range(4), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--well-stats", type=Path, default=WELL_STATS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError(args.output if args.output.exists() else args.model_output)

    with np.load(args.well_stats, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise ValueError("Run 4 well-stat firewall changed")
        folds = stats["original_folds"].astype(np.int64)
        cross = stats["cross"].astype(np.float64)
        gram = stats["gram"].astype(np.float64)
    weight = base_weight(folds, cross, gram, args.held_fold)
    train_features: list[np.ndarray] = []
    train_target: list[np.ndarray] = []
    train_weight: list[np.ndarray] = []
    feature_names = None
    feature_records: list[tuple[str, str]] = []
    training_folds = [
        fold for fold in range(4) if fold != args.held_fold
    ]
    for fold in training_folds:
        bank = load_clean_fold_bank(fold, truth_allowed=True)
        feature_path = ROW_FEATURE_ROOT / f"row_features_f{fold}.npz"
        with np.load(feature_path, allow_pickle=False) as cache:
            if (
                bool(cache["hidden_truth_loaded"])
                or bool(cache["F4_loaded"])
                or bool(cache["F4_metric_computed"])
                or not np.array_equal(
                    cache["well_ids"].astype(str), bank["ids"]
                )
            ):
                raise ValueError(f"fold {fold}: row-feature firewall changed")
            local_names = cache["feature_names"].astype(str)
            features = cache["features"][::TRAINING_STRIDE].astype(
                np.float32
            )
        if feature_names is None:
            feature_names = local_names
        elif not np.array_equal(feature_names, local_names):
            raise ValueError("row-feature schema changed between folds")
        base = bank["clean"] + weight @ bank["correction"]
        error = base - bank["truth"]
        correction = bank["correction"][SURFACE_INDEX]
        amplitude = np.clip(
            -error * correction / np.maximum(np.square(correction), 1e-4),
            0.0,
            AMPLITUDE_CAP,
        )
        energy = np.square(correction)
        energy = np.minimum(energy, np.quantile(energy, 0.99))
        train_features.append(features)
        train_target.append(
            amplitude[::TRAINING_STRIDE].astype(np.float32)
        )
        train_weight.append(energy[::TRAINING_STRIDE].astype(np.float32))
        feature_records.append((str(feature_path), sha256(feature_path)))
        del bank, base, error, correction, amplitude, energy, features
        gc.collect()

    features = np.concatenate(train_features)
    target = np.concatenate(train_target)
    sample_weight = np.concatenate(train_weight)
    sample_weight /= max(float(np.mean(sample_weight)), 1e-12)
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=900,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=12.0,
        random_seed=20265512 + args.held_fold,
        random_strength=0.3,
        task_type="GPU",
        devices=str(args.device),
        verbose=200,
        allow_writing_files=False,
    )
    model.fit(
        features,
        target,
        sample_weight=sample_weight,
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_output))
    del features, target, sample_weight, train_features
    del train_target, train_weight
    gc.collect()

    held = load_clean_fold_bank(args.held_fold, truth_allowed=False)
    held_feature_path = (
        ROW_FEATURE_ROOT / f"row_features_f{args.held_fold}.npz"
    )
    with np.load(held_feature_path, allow_pickle=False) as cache:
        if (
            bool(cache["hidden_truth_loaded"])
            or bool(cache["F4_loaded"])
            or bool(cache["F4_metric_computed"])
            or not np.array_equal(
                cache["well_ids"].astype(str), held["ids"]
            )
            or not np.array_equal(
                cache["feature_names"].astype(str), feature_names
            )
        ):
            raise ValueError("held row-feature firewall changed")
        held_features = cache["features"].astype(np.float32)
        starts = cache["row_starts"].astype(np.int64)
    raw_amplitude = np.clip(
        model.predict(held_features),
        0.0,
        AMPLITUDE_CAP,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray("clean_bounded_row_surface_router"),
        held_fold=np.int64(args.held_fold),
        training_folds=np.asarray(training_folds, dtype=np.int8),
        training_stride=np.int64(TRAINING_STRIDE),
        surface_index=np.int64(SURFACE_INDEX),
        amplitude_cap=np.float64(AMPLITUDE_CAP),
        well_ids=held["ids"],
        row_starts=starts,
        raw_amplitude=raw_amplitude.astype(np.float32),
        model_path=np.asarray(str(args.model_output)),
        model_sha256=np.asarray(sha256(args.model_output)),
        well_stats_sha256=np.asarray(sha256(args.well_stats)),
        feature_paths=np.asarray(
            [path for path, _ in feature_records]
            + [str(held_feature_path)]
        ),
        feature_sha256=np.asarray(
            [digest for _, digest in feature_records]
            + [sha256(held_feature_path)]
        ),
        training_target_loaded=np.asarray(True),
        held_target_loaded=np.asarray(False),
        raw_hidden_target_stored=np.asarray(False),
        F4_loaded=np.asarray(False),
        F4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
    )
    print(
        f"wrote={args.output} model={args.model_output} "
        f"rows={len(raw_amplitude)}"
    )


if __name__ == "__main__":
    main()
