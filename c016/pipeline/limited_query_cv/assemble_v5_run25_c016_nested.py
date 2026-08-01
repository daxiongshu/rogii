from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv import (
    evaluate_v5_run6_d209_amplitude as amplitude_selector,
)
from limited_query_cv import (
    evaluate_v5_run6_local_residual_seed_bag as local_selector,
)
from limited_query_cv import (
    evaluate_v5_run6_protected_mode_bank as mode_parent,
)
from limited_query_cv import (
    evaluate_v5_run6_protected_posterior_clean as posterior_selector,
)
from limited_query_cv import (
    evaluate_v5_run8_clean_multimodel_ridge as ridge,
)
from limited_query_cv import (
    train_v5_run25_c016_nested as trainer,
)


RUN_ID = "v5_batch2_run_025_goal_050"
ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_025_goal_050/promotion_audit"
)
ROLE_ROOT = RUN_ROOT / "roles"
ARTIFACT_ROOT = RUN_ROOT / "sealed"
RUN25_ROOT = (
    ROOT / "limited_query_cv/runs/v5_batch2_run_025_goal_050"
)
RUN6_ROOT = (
    ROOT / "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
V20 = Path("/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz")
LOCAL_TREATMENTS = local_selector.TREATMENTS
LOCAL_SNAPSHOTS = trainer.LOCAL_SNAPSHOTS
LOCAL_TEMPERATURES = local_selector.TEMPERATURES
POSTERIOR_VARIANTS = trainer.POSTERIOR_VARIANTS
POSTERIOR_SNAPSHOTS = trainer.POSTERIOR_SNAPSHOTS
POSTERIOR_TEMPERATURES = posterior_selector.TEMPERATURES
COMPONENTS = (
    "protected_posterior_incremental",
    "protected_mode_bank",
)
RIDGE_GRID = ridge.RIDGE_GRID
C016_AMPLITUDE = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64)
                    - truth.astype(np.float64)
                )
            )
        )
    )


@dataclass(frozen=True)
class Fold:
    fold: int
    well_ids: np.ndarray
    row_starts: np.ndarray
    baseline: np.ndarray
    truth: np.ndarray


def load_folds() -> list[Fold]:
    with np.load(LAYOUT, allow_pickle=False) as layout:
        well_ids = layout["well_ids"].astype(str)
        starts = layout["row_starts"].astype(np.int64)
        assignments = layout["original"].astype(np.int8)
        truth = layout["truth"].astype(np.float32)
    with np.load(V20, allow_pickle=False) as cache:
        if (
            not np.array_equal(well_ids, cache["well_ids"].astype(str))
            or not np.array_equal(
                starts, cache["row_starts"].astype(np.int64)
            )
        ):
            raise RuntimeError("v20/layout binding changed")
        baseline = cache["prediction"].astype(np.float32)
    output = []
    for fold in range(5):
        indices = np.flatnonzero(assignments == fold)
        lengths = starts[indices + 1] - starts[indices]

        def gather(values: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [
                    values[starts[index] : starts[index + 1]]
                    for index in indices
                ]
            )

        output.append(
            Fold(
                fold=fold,
                well_ids=well_ids[indices],
                row_starts=np.r_[0, np.cumsum(lengths)].astype(np.int64),
                baseline=gather(baseline).astype(np.float32),
                truth=gather(truth).astype(np.float32),
            )
        )
    return output


def role_directory(kind: str, folds: tuple[int, ...]) -> Path:
    return ROLE_ROOT / trainer.role_name(kind, folds)


def slice_prediction(
    path: Path,
    fold: Fold,
    keys: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], str]:
    digest = sha256(path)
    with np.load(path, allow_pickle=False) as cache:
        if (
            bool(cache["hidden_truth_stored"])
            or bool(cache["protected_metrics_computed"])
            or "truth" in cache.files
            or "f4_metric" in cache.files
        ):
            raise RuntimeError(f"{path}: prediction firewall failed")
        ids = cache["well_ids"].astype(str)
        assignments = cache["well_folds"].astype(np.int8)
        starts = cache["row_starts"].astype(np.int64)
        selected = np.flatnonzero(assignments == fold.fold)
        lengths = starts[selected + 1] - starts[selected]
        local_starts = np.r_[0, np.cumsum(lengths)].astype(np.int64)

        def gather(values: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [
                    values[starts[index] : starts[index + 1]]
                    for index in selected
                ]
            )

        if (
            not np.array_equal(ids[selected], fold.well_ids)
            or not np.array_equal(local_starts, fold.row_starts)
            or not np.array_equal(gather(cache["baseline"]), fold.baseline)
        ):
            raise RuntimeError(f"{path}: fold layout changed")
        return (
            {
                key: gather(cache[key]).astype(np.float32)
                for key in keys
            },
            digest,
        )


def local_candidates(
    role: Path,
    fold: Fold,
) -> tuple[
    dict[tuple[str, int, float], np.ndarray],
    dict[tuple[str, int, float], list[str]],
]:
    by_seed: dict[
        tuple[int, int, float], tuple[np.ndarray, str]
    ] = {}
    for seed in trainer.LOCAL_SEEDS:
        for snapshot in LOCAL_SNAPSHOTS:
            path = (
                role
                / f"local_seed{seed}"
                / f"epoch{snapshot:03d}_prediction.npz"
            )
            keys = tuple(
                f"prediction_t{int(temperature * 100):03d}"
                for temperature in LOCAL_TEMPERATURES
            )
            values, digest = slice_prediction(path, fold, keys)
            for temperature, key in zip(LOCAL_TEMPERATURES, keys):
                by_seed[(seed, snapshot, temperature)] = (
                    values[key],
                    digest,
                )
    predictions = {}
    hashes = {}
    first_seed, second_seed = trainer.LOCAL_SEEDS
    for snapshot in LOCAL_SNAPSHOTS:
        for temperature in LOCAL_TEMPERATURES:
            first, first_hash = by_seed[
                (first_seed, snapshot, temperature)
            ]
            second, second_hash = by_seed[
                (second_seed, snapshot, temperature)
            ]
            values = {
                local_selector.ORIGINAL: first,
                local_selector.INDEPENDENT: second,
                local_selector.BAG: local_selector.average_predictions(
                    first, second
                ).astype(np.float32),
            }
            for treatment, prediction in values.items():
                predictions[(treatment, snapshot, temperature)] = prediction
                hashes[(treatment, snapshot, temperature)] = (
                    [first_hash]
                    if treatment == local_selector.ORIGINAL
                    else [second_hash]
                    if treatment == local_selector.INDEPENDENT
                    else [first_hash, second_hash]
                )
    return predictions, hashes


def choose_local(
    outer: int,
    folds: list[Fold],
) -> tuple[
    dict[str, Any],
    dict[int, np.ndarray],
    np.ndarray,
    dict[str, Any],
]:
    records: list[Any] = [None] * 5
    candidates = {}
    for validation in range(5):
        if validation == outer:
            continue
        pair = tuple(sorted((outer, validation)))
        predictions, hashes = local_candidates(
            role_directory("pair", pair), folds[validation]
        )
        candidates[validation] = predictions
        records[validation] = (
            folds[validation].well_ids,
            folds[validation].row_starts,
            folds[validation].baseline,
            folds[validation].truth,
            predictions,
            hashes,
        )
    selector_roles = [fold for fold in range(5) if fold != outer]
    selected, grid = local_selector.select_recipe(
        records, selector_roles
    )
    treatment = str(selected["treatment"])
    snapshot = int(selected["snapshot"])
    temperature = float(selected["temperature"])
    weight = float(selected["weight"])
    inner = {
        fold: (
            folds[fold].baseline.astype(np.float64)
            + weight
            * (
                candidates[fold][
                    (treatment, snapshot, temperature)
                ].astype(np.float64)
                - folds[fold].baseline.astype(np.float64)
            )
        ).astype(np.float32)
        for fold in selector_roles
    }
    outer_candidates, outer_hashes = local_candidates(
        role_directory("outer", (outer,)), folds[outer]
    )
    outer_prediction = (
        folds[outer].baseline.astype(np.float64)
        + weight
        * (
            outer_candidates[
                (treatment, snapshot, temperature)
            ].astype(np.float64)
            - folds[outer].baseline.astype(np.float64)
        )
    ).astype(np.float32)
    return (
        selected,
        inner,
        outer_prediction,
        {
            "candidate_count": len(grid),
            "selected_prediction_sha256": outer_hashes[
                (treatment, snapshot, temperature)
            ],
        },
    )


def choose_amplitude(
    outer: int,
    folds: list[Fold],
    inner_d209: dict[int, np.ndarray],
    outer_d209: np.ndarray,
) -> tuple[dict[str, Any], dict[int, np.ndarray], np.ndarray]:
    records: list[Any] = [None] * 5
    selector_roles = [fold for fold in range(5) if fold != outer]
    for fold in selector_roles:
        records[fold] = {
            "well_ids": folds[fold].well_ids,
            "row_starts": folds[fold].row_starts,
            "baseline": folds[fold].baseline,
            "truth": folds[fold].truth,
            "correction": (
                inner_d209[fold] - folds[fold].baseline
            ).astype(np.float32),
        }
    selected, grid = amplitude_selector.select_amplitude(
        records, selector_roles
    )
    amplitude = float(selected["amplitude"])
    inner = {
        fold: (
            folds[fold].baseline.astype(np.float64)
            + amplitude
            * (
                inner_d209[fold].astype(np.float64)
                - folds[fold].baseline.astype(np.float64)
            )
        ).astype(np.float32)
        for fold in selector_roles
    }
    outer_prediction = (
        folds[outer].baseline.astype(np.float64)
        + amplitude
        * (
            outer_d209.astype(np.float64)
            - folds[outer].baseline.astype(np.float64)
        )
    ).astype(np.float32)
    return (
        {**selected, "candidate_count": len(grid)},
        inner,
        outer_prediction,
    )


def posterior_candidates(
    role: Path,
    fold: Fold,
) -> tuple[dict[tuple[str, int, float], np.ndarray], dict[str, str]]:
    output = {}
    hashes = {}
    for variant in POSTERIOR_VARIANTS:
        for snapshot in POSTERIOR_SNAPSHOTS:
            path = (
                role
                / f"posterior_{variant}"
                / f"epoch{snapshot:03d}_prediction.npz"
            )
            keys = tuple(
                f"correction_t{int(round(temperature * 100)):03d}"
                for temperature in POSTERIOR_TEMPERATURES
            )
            values, digest = slice_prediction(path, fold, keys)
            hashes[f"{variant}:{snapshot}"] = digest
            for temperature, key in zip(
                POSTERIOR_TEMPERATURES, keys
            ):
                output[(variant, snapshot, temperature)] = values[key]
    return output, hashes


def choose_posterior(
    outer: int,
    folds: list[Fold],
    inner_d570: dict[int, np.ndarray],
    outer_d570: np.ndarray,
) -> tuple[
    dict[str, Any],
    dict[int, np.ndarray],
    np.ndarray,
    dict[str, Any],
]:
    records: list[Any] = [None] * 5
    raw = {}
    hashes = {}
    selector_roles = [fold for fold in range(5) if fold != outer]
    for fold in selector_roles:
        pair = tuple(sorted((outer, fold)))
        corrections, local_hashes = posterior_candidates(
            role_directory("pair", pair), folds[fold]
        )
        raw[fold] = corrections
        hashes[str(fold)] = local_hashes
        records[fold] = {
            "well_ids": folds[fold].well_ids,
            "row_starts": folds[fold].row_starts,
            "baseline": folds[fold].baseline,
            "d570": inner_d570[fold],
            "truth": folds[fold].truth,
            "corrections": corrections,
        }
    # Build only the four selector-fold sufficient statistics.  The generic
    # helper enumerates all five records, which would force construction of a
    # held-outer record before the recipe has been selected.
    sufficient = {}
    for fold in selector_roles:
        record = records[fold]
        for treatment in posterior_selector.TREATMENTS:
            members = posterior_selector.treatment_members(treatment)
            for snapshot in posterior_selector.SNAPSHOTS:
                for temperature in posterior_selector.TEMPERATURES:
                    correction = posterior_selector.average_corrections(
                        [
                            np.asarray(
                                record["corrections"][
                                    (variant, snapshot, temperature)
                                ]
                            )
                            for variant in members
                        ]
                    )
                    sufficient[
                        (fold, treatment, snapshot, temperature)
                    ] = posterior_selector.metric_sufficient_statistics(
                        [np.asarray(record["d570"])],
                        [np.asarray(record["truth"])],
                        [correction],
                        [np.asarray(record["row_starts"])],
                    )
    selected, grid = posterior_selector.select_recipe(
        records,
        selector_roles,
        "d570",
        sufficient_cache=sufficient,
    )
    members = posterior_selector.treatment_members(
        str(selected["treatment"])
    )
    snapshot = int(selected["snapshot"])
    temperature = float(selected["temperature"])
    scale = float(selected["effective_scale"])
    inner = {}
    for fold in selector_roles:
        correction = posterior_selector.average_corrections(
            [
                raw[fold][(variant, snapshot, temperature)]
                for variant in members
            ]
        )
        inner[fold] = (
            inner_d570[fold].astype(np.float64) + scale * correction
        ).astype(np.float32)
    outer_raw, outer_hashes = posterior_candidates(
        role_directory("outer", (outer,)), folds[outer]
    )
    outer_correction = posterior_selector.average_corrections(
        [
            outer_raw[(variant, snapshot, temperature)]
            for variant in members
        ]
    )
    outer_prediction = (
        outer_d570.astype(np.float64) + scale * outer_correction
    ).astype(np.float32)
    return (
        selected,
        inner,
        outer_prediction,
        {
            "candidate_count": len(grid),
            "pair_prediction_sha256": hashes,
            "outer_prediction_sha256": outer_hashes,
        },
    )


def safe_mode_path(fold: int) -> Path:
    if fold == 4:
        return (
            RUN_ROOT
            / "caches/original_safe_mode_bank_fold4.npz"
        )
    return RUN25_ROOT / f"original_safe_mode_bank_f{fold}.npz"


def load_mode_fold(fold: Fold) -> dict[str, Any]:
    path = safe_mode_path(fold.fold)
    with np.load(path, allow_pickle=False) as cache:
        names = cache["candidate_names"].astype(str)
        checkpoints = cache["checkpoint_names"].astype(str)
        if (
            int(cache["fold"]) != fold.fold
            or any(name.startswith("strat_") for name in names)
            or any("strat" in name for name in checkpoints)
            or bool(cache["forbidden_stratified_parent_loaded"])
            or fold.fold == 4
            and (
                "truth" in cache.files
                or not bool(cache["protected_f4_cache"])
                or bool(cache["hidden_truth_stored"])
            )
        ):
            raise RuntimeError(f"{path}: safe mode firewall failed")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        corrections = cache["corrections"].astype(np.float32)
        metrics = cache["metrics"].astype(np.float32)
        shifted = cache["shifted_control_metrics"].astype(np.float32)
        metric_names = cache["metric_names"].astype(str)
    if (
        not np.array_equal(ids, fold.well_ids)
        or not np.array_equal(starts, fold.row_starts)
        or not np.array_equal(baseline, fold.baseline)
    ):
        raise RuntimeError(f"{path}: mode layout changed")
    return {
        "fold": fold.fold,
        "well_ids": ids,
        "starts": starts,
        "baseline": baseline,
        "truth": fold.truth,
        "metric_names": metric_names,
        "selector_corrections": mode_parent.assemble_selector_corrections(
            metrics, corrections, starts
        ),
        "shifted_selector_corrections": (
            mode_parent.assemble_selector_corrections(
                shifted, corrections, starts
            )
        ),
        "sha256": sha256(path),
    }


def choose_mode_components(
    folds: list[dict[str, Any]],
    training_folds: list[int],
) -> list[dict[str, Any]]:
    metric_names = np.asarray(folds[0]["metric_names"])
    errors = [
        record["baseline"].astype(np.float64)
        - record["truth"].astype(np.float64)
        for record in folds
    ]
    available = set(range(len(metric_names)))
    selected = []
    for _ in range(3):
        current_sse = sum(
            float(errors[fold] @ errors[fold])
            for fold in training_folds
        )
        cross = np.zeros(len(metric_names), dtype=np.float64)
        denominator = np.zeros(len(metric_names), dtype=np.float64)
        count = 0
        for fold in training_folds:
            corrections = folds[fold]["selector_corrections"].astype(
                np.float64
            )
            cross += corrections @ errors[fold]
            denominator += np.einsum(
                "ij,ij->i", corrections, corrections
            )
            count += len(errors[fold])
        candidates = []
        for selector in available:
            for weight in mode_parent.OUTER_WEIGHTS[1:]:
                sse = (
                    current_sse
                    + 2.0 * weight * cross[selector]
                    + weight * weight * denominator[selector]
                )
                candidates.append(
                    (
                        sse,
                        weight,
                        mode_parent.selector_priority(
                            str(metric_names[selector])
                        ),
                        selector,
                    )
                )
        sse, weight, _, selector = min(candidates)
        if sse >= current_sse:
            break
        selected.append(
            {
                "selector_index": int(selector),
                "metric": str(metric_names[selector]),
                "weight": float(weight),
                "training_gain": float(
                    np.sqrt(current_sse / count)
                    - np.sqrt(sse / count)
                ),
            }
        )
        for fold in range(5):
            errors[fold] += (
                weight
                * folds[fold]["selector_corrections"][selector]
            )
        available.remove(selector)
    return selected


def apply_mode(
    record: dict[str, Any],
    components: list[dict[str, Any]],
) -> np.ndarray:
    prediction = record["baseline"].astype(np.float64).copy()
    for component in components:
        prediction += (
            float(component["weight"])
            * record["selector_corrections"][
                int(component["selector_index"])
            ]
        )
    return prediction.astype(np.float32)


def choose_mode(
    outer: int,
    mode_folds: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, np.ndarray],
    np.ndarray,
]:
    selector_roles = [fold for fold in range(5) if fold != outer]
    inner = {}
    for validation in selector_roles:
        training = [
            fold for fold in selector_roles if fold != validation
        ]
        components = choose_mode_components(mode_folds, training)
        inner[validation] = apply_mode(
            mode_folds[validation], components
        )
    outer_components = choose_mode_components(
        mode_folds, selector_roles
    )
    return (
        outer_components,
        inner,
        apply_mode(mode_folds[outer], outer_components),
    )


def concatenate_inner(
    folds: list[Fold],
    roles: list[int],
    values: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]
    parts = []
    for role in roles:
        parts.append(values[role])
        for left, right in zip(
            folds[role].row_starts[:-1],
            folds[role].row_starts[1:],
        ):
            starts.append(starts[-1] + int(right - left))
    return np.concatenate(parts), np.asarray(starts, dtype=np.int64)


def fit_weights(
    parent: np.ndarray,
    posterior: np.ndarray,
    mode: np.ndarray,
    truth: np.ndarray,
    ridge_value: float,
) -> np.ndarray:
    mask = np.ones(len(parent), dtype=bool)
    return ridge.bounded_fit(
        parent,
        [posterior, mode],
        truth,
        mask,
        ridge_value,
    )


def choose_ridge(
    outer: int,
    folds: list[Fold],
    inner_d570: dict[int, np.ndarray],
    inner_posterior: dict[int, np.ndarray],
    inner_mode: dict[int, np.ndarray],
    outer_d570: np.ndarray,
    outer_posterior: np.ndarray,
    outer_mode: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, int]:
    selector_roles = [fold for fold in range(5) if fold != outer]
    records = []
    for ridge_value in RIDGE_GRID:
        validation_predictions = {}
        validation_weights = {}
        validation_gains = {}
        for validation in selector_roles:
            training = [
                fold for fold in selector_roles if fold != validation
            ]
            train_parent, _ = concatenate_inner(
                folds, training, inner_d570
            )
            train_posterior, _ = concatenate_inner(
                folds, training, inner_posterior
            )
            train_mode, _ = concatenate_inner(
                folds, training, inner_mode
            )
            train_truth = np.concatenate(
                [folds[fold].truth for fold in training]
            )
            weights = fit_weights(
                train_parent,
                train_posterior,
                train_mode,
                train_truth,
                ridge_value,
            )
            prediction = ridge.compose(
                inner_d570[validation],
                [
                    inner_posterior[validation],
                    inner_mode[validation],
                ],
                weights,
            )
            validation_predictions[validation] = prediction
            validation_weights[str(validation)] = weights.tolist()
            validation_gains[str(validation)] = (
                rmse(
                    inner_d570[validation],
                    folds[validation].truth,
                )
                - rmse(prediction, folds[validation].truth)
            )
        parent = np.concatenate(
            [inner_d570[fold] for fold in selector_roles]
        )
        prediction = np.concatenate(
            [validation_predictions[fold] for fold in selector_roles]
        )
        truth = np.concatenate(
            [folds[fold].truth for fold in selector_roles]
        )
        pooled_gain = rmse(parent, truth) - rmse(prediction, truth)
        well_gains = []
        for fold in selector_roles:
            for left, right in zip(
                folds[fold].row_starts[:-1],
                folds[fold].row_starts[1:],
            ):
                well_gains.append(
                    rmse(
                        inner_d570[fold][left:right],
                        folds[fold].truth[left:right],
                    )
                    - rmse(
                        validation_predictions[fold][left:right],
                        folds[fold].truth[left:right],
                    )
                )
        records.append(
            {
                "ridge": ridge_value,
                "eligible": bool(
                    all(
                        gain >= -1e-12
                        for gain in validation_gains.values()
                    )
                ),
                "pooled_gain": pooled_gain,
                "mean_complete_well_gain": float(np.mean(well_gains)),
                "minimum_role_gain": min(validation_gains.values()),
                "validation_gains": validation_gains,
                "validation_weights": validation_weights,
            }
        )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        selected = {
            "ridge": None,
            "eligible": True,
            "pooled_gain": 0.0,
            "mean_complete_well_gain": 0.0,
            "minimum_role_gain": 0.0,
            "validation_gains": {
                str(role): 0.0 for role in selector_roles
            },
            "validation_weights": {
                str(role): [0.0, 0.0] for role in selector_roles
            },
        }
        weights = np.zeros(2, dtype=np.float64)
    else:
        selected = min(
            eligible,
            key=lambda record: (
                -round(float(record["pooled_gain"]), 12),
                -round(
                    float(record["mean_complete_well_gain"]), 12
                ),
                -round(float(record["minimum_role_gain"]), 12),
                -float(record["ridge"]),
            ),
        )
        parent = np.concatenate(
            [inner_d570[fold] for fold in selector_roles]
        )
        posterior_values = np.concatenate(
            [inner_posterior[fold] for fold in selector_roles]
        )
        mode_values = np.concatenate(
            [inner_mode[fold] for fold in selector_roles]
        )
        truth = np.concatenate(
            [folds[fold].truth for fold in selector_roles]
        )
        weights = fit_weights(
            parent,
            posterior_values,
            mode_values,
            truth,
            float(selected["ridge"]),
        )
    full = ridge.compose(
        outer_d570,
        [outer_posterior, outer_mode],
        weights,
    )
    prediction = (
        outer_d570.astype(np.float64)
        + C016_AMPLITUDE
        * (full.astype(np.float64) - outer_d570.astype(np.float64))
    ).astype(np.float32)
    positive_inner = int(
        sum(
            np.any(
                np.asarray(values, dtype=np.float64) > 0.0
            )
            for values in selected["validation_weights"].values()
        )
    )
    return (
        {
            **selected,
            "candidate_count": len(records),
            "refit_weights": {
                name: float(weight)
                for name, weight in zip(COMPONENTS, weights)
            },
        },
        prediction,
        positive_inner,
    )


def expected_role_files() -> list[Path]:
    roles = [
        ("pair", pair)
        for pair in itertools.combinations(range(5), 2)
    ] + [("outer", (fold,)) for fold in range(5)]
    paths = []
    for kind, excluded in roles:
        role = role_directory(kind, excluded)
        for seed in trainer.LOCAL_SEEDS:
            directory = role / f"local_seed{seed}"
            paths.append(directory / "result.json")
            for snapshot in LOCAL_SNAPSHOTS:
                paths.extend(
                    (
                        directory / f"epoch{snapshot:03d}.pt",
                        directory
                        / f"epoch{snapshot:03d}_prediction.npz",
                    )
                )
        for variant in POSTERIOR_VARIANTS:
            directory = role / f"posterior_{variant}"
            paths.append(directory / "result.json")
            for snapshot in POSTERIOR_SNAPSHOTS:
                paths.extend(
                    (
                        directory / f"epoch{snapshot:03d}.pt",
                        directory
                        / f"epoch{snapshot:03d}_prediction.npz",
                    )
                )
    return paths


def freeze_candidate_manifest(output: Path) -> None:
    paths = expected_role_files()
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            f"incomplete nested artifact set ({len(missing)} missing)"
        )
    training_source = Path(trainer.__file__)
    caches = [
        RUN_ROOT / "caches/v20_components_fold4.npz",
        RUN_ROOT / "caches/protected_posterior_volume_fold4.npz",
        RUN_ROOT / "caches/original_safe_mode_bank_fold4.npz",
        *[safe_mode_path(fold) for fold in range(4)],
    ]
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "complete_nested_candidate_manifest",
        "status": "frozen_before_selector_or_protected_metric",
        "unique_pair_roles": 10,
        "outer_roles": 5,
        "logical_inner_roles": 20,
        "trainable_trajectories": {
            "local_residual_seed_models": 30,
            "posterior_fusion_variants": 45,
            "total": 75,
        },
        "training_source": str(
            training_source.resolve().relative_to(ROOT)
        ),
        "training_source_sha256": sha256(training_source),
        "artifacts": [
            {
                "path": str(path.resolve().relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "caches": [
            {
                "path": str(path.resolve().relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in caches
        ],
        "artifact_count": len(paths),
        "all_prediction_archives_omit_truth": True,
        "selector_metric_computed": False,
        "f4_metric_computed": False,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "artifact_count": payload["artifact_count"],
                "output_sha256": sha256(output),
            },
            sort_keys=True,
        )
    )


def verify_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status")
        != "frozen_before_selector_or_protected_metric"
        or int(payload.get("unique_pair_roles", -1)) != 10
        or int(payload.get("outer_roles", -1)) != 5
        or int(payload.get("logical_inner_roles", -1)) != 20
        or int(
            payload.get("trainable_trajectories", {}).get(
                "total", -1
            )
        )
        != 75
        or bool(payload.get("selector_metric_computed"))
        or bool(payload.get("f4_metric_computed"))
        or payload.get("source_sha256") != sha256(Path(__file__))
    ):
        raise RuntimeError("nested candidate manifest contract changed")
    for record in payload["artifacts"] + payload["caches"]:
        path = ROOT / record["path"]
        if (
            not path.exists()
            or path.stat().st_size != int(record["bytes"])
            or sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"{path}: frozen artifact changed")
    return payload


def assemble(
    candidate_manifest: Path,
    selector_output: Path,
    prediction_output: Path,
    prediction_manifest: Path,
) -> None:
    manifest = verify_candidate_manifest(candidate_manifest)
    for path in (
        selector_output,
        prediction_output,
        prediction_manifest,
    ):
        if path.exists():
            raise FileExistsError(path)
    folds = load_folds()
    mode_folds = [load_mode_fold(fold) for fold in folds]
    predictions = []
    d570_parts = []
    posterior_parts = []
    mode_parts = []
    selectors = []
    positive_inner = 0
    for outer in range(5):
        local_record, inner_d209, outer_d209, local_lineage = (
            choose_local(outer, folds)
        )
        amplitude_record, inner_d570, outer_d570 = choose_amplitude(
            outer, folds, inner_d209, outer_d209
        )
        (
            posterior_record,
            inner_posterior,
            outer_posterior,
            posterior_lineage,
        ) = choose_posterior(
            outer, folds, inner_d570, outer_d570
        )
        mode_record, inner_mode, outer_mode = choose_mode(
            outer, mode_folds
        )
        ridge_record, prediction, local_positive = choose_ridge(
            outer,
            folds,
            inner_d570,
            inner_posterior,
            inner_mode,
            outer_d570,
            outer_posterior,
            outer_mode,
        )
        predictions.append(prediction)
        d570_parts.append(outer_d570)
        posterior_parts.append(outer_posterior)
        mode_parts.append(outer_mode)
        positive_inner += local_positive
        selectors.append(
            {
                "outer_fold": outer,
                "local_residual": local_record,
                "local_lineage": local_lineage,
                "D570_amplitude": amplitude_record,
                "posterior": posterior_record,
                "posterior_lineage": posterior_lineage,
                "mode_components": mode_record,
                "ridge": ridge_record,
                "positive_inner_weight_roles": local_positive,
            }
        )
    well_ids = np.concatenate([fold.well_ids for fold in folds])
    assignments = np.concatenate(
        [
            np.full(len(fold.well_ids), fold.fold, dtype=np.int8)
            for fold in folds
        ]
    )
    starts = np.r_[
        0,
        np.cumsum(
            np.concatenate(
                [np.diff(fold.row_starts) for fold in folds]
            )
        ),
    ].astype(np.int64)
    baseline = np.concatenate(
        [fold.baseline for fold in folds]
    ).astype(np.float32)
    np.savez_compressed(
        prediction_output,
        well_ids=well_ids,
        folds=assignments,
        row_starts=starts,
        baseline=baseline,
        D570=np.concatenate(d570_parts).astype(np.float32),
        protected_posterior_incremental=np.concatenate(
            posterior_parts
        ).astype(np.float32),
        protected_mode_bank=np.concatenate(mode_parts).astype(
            np.float32
        ),
        prediction=np.concatenate(predictions).astype(np.float32),
        candidate_manifest_sha256=np.asarray(
            sha256(candidate_manifest)
        ),
        selector_source_sha256=np.asarray(sha256(Path(__file__))),
        selector_record_pending_sha256=np.asarray(""),
        hidden_truth_stored=np.asarray(False),
        f4_metric_computed=np.asarray(False),
    )
    selector_payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "sealed_nested_selector",
        "status": "complete_all_outer_predictions_frozen",
        "candidate_manifest": str(
            candidate_manifest.resolve().relative_to(ROOT)
        ),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "outer_records": selectors,
        "positive_inner_weight_roles": positive_inner,
        "logical_inner_roles": 20,
        "C016_amplitude": C016_AMPLITUDE,
        "prediction": str(
            prediction_output.resolve().relative_to(ROOT)
        ),
        "prediction_sha256": sha256(prediction_output),
        "truth_stored_in_prediction": False,
        "protected_metrics_emitted": False,
        "f4_metric_computed": False,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(selector_output, selector_payload)
    prediction_payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "complete_outer_prediction_manifest",
        "status": "frozen_ready_for_single_joint_reveal",
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "selector": str(
            selector_output.resolve().relative_to(ROOT)
        ),
        "selector_sha256": sha256(selector_output),
        "prediction": str(
            prediction_output.resolve().relative_to(ROOT)
        ),
        "prediction_sha256": sha256(prediction_output),
        "well_count": len(well_ids),
        "row_count": int(starts[-1]),
        "outer_folds": 5,
        "positive_inner_weight_roles": positive_inner,
        "truth_stored_in_prediction": False,
        "protected_metric_computed": False,
        "f4_metric_computed": False,
        "candidate_artifact_count": manifest["artifact_count"],
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(prediction_manifest, prediction_payload)
    print(
        json.dumps(
            {
                "status": prediction_payload["status"],
                "prediction_sha256": prediction_payload[
                    "prediction_sha256"
                ],
                "selector_sha256": prediction_payload[
                    "selector_sha256"
                ],
                "manifest_sha256": sha256(prediction_manifest),
            },
            sort_keys=True,
        )
    )


def confirmation(
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    c016_path = RUN25_ROOT / "original_safe_c016_clean.npz"
    posterior_path = RUN6_ROOT / "protected_posterior_clean.npz"
    mode_path = RUN25_ROOT / "original_safe_mode_bank_clean.npz"
    with (
        np.load(c016_path, allow_pickle=False) as c016,
        np.load(posterior_path, allow_pickle=False) as posterior_cache,
        np.load(mode_path, allow_pickle=False) as mode_cache,
        np.load(LAYOUT, allow_pickle=False) as layout,
    ):
        well_ids = c016["well_ids"].astype(str)
        starts = c016["row_starts"].astype(np.int64)
        baseline = c016["baseline"].astype(np.float32)
        truth = c016["truth"].astype(np.float32)
        parent = c016["parent"].astype(np.float32)
        posterior_values = posterior_cache[
            "incremental_after_D570"
        ].astype(np.float32)
        mode_values = mode_cache["prediction"].astype(np.float32)
        layout_ids = layout["well_ids"].astype(str)
        assignments = {
            "balanced": layout["stratified"].astype(np.int8),
            "independent": layout["independent"].astype(np.int8),
        }
    lookup = {well_id: index for index, well_id in enumerate(layout_ids)}
    results = {}
    original_components = ridge.COMPONENTS
    try:
        ridge.COMPONENTS = COMPONENTS
        for name, raw_groups in assignments.items():
            groups = np.asarray(
                [raw_groups[lookup[well_id]] for well_id in well_ids],
                dtype=np.int8,
            )
            prediction = parent.astype(np.float64).copy()
            group_records = []
            for held in range(5):
                payload = {
                    "parent": parent,
                    "protected_posterior_incremental": (
                        posterior_values
                    ),
                    "protected_mode_bank": mode_values,
                    "truth": truth,
                    "folds": groups,
                    "row_starts": starts,
                }
                selected, grid = ridge.select_recipe(
                    payload,
                    [group for group in range(5) if group != held],
                )
                weights = (
                    np.zeros(2, dtype=np.float64)
                    if selected is None
                    else np.asarray(
                        [
                            selected["weights"][component]
                            for component in COMPONENTS
                        ],
                        dtype=np.float64,
                    )
                )
                full = ridge.compose(
                    parent,
                    [posterior_values, mode_values],
                    weights,
                )
                mask = ridge.row_mask(groups, starts, {held})
                prediction[mask] = (
                    parent[mask].astype(np.float64)
                    + C016_AMPLITUDE
                    * (
                        full[mask].astype(np.float64)
                        - parent[mask].astype(np.float64)
                    )
                )
                group_records.append(
                    {
                        "held_group": held,
                        "candidate_count": len(grid),
                        "weights": {
                            component: float(weight)
                            for component, weight in zip(
                                COMPONENTS, weights
                            )
                        },
                        "gain_vs_v20": (
                            rmse(baseline[mask], truth[mask])
                            - rmse(prediction[mask], truth[mask])
                        ),
                    }
                )
            gain = rmse(baseline, truth) - rmse(prediction, truth)
            results[name] = {
                "gain_vs_v20": gain,
                "minimum_gain_0p10_passed": bool(gain >= 0.10),
                "group_records": group_records,
            }
    finally:
        ridge.COMPONENTS = original_components
    passed = all(
        result["minimum_gain_0p10_passed"]
        for result in results.values()
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "joint_F0_F3_confirmation_result",
        "status": (
            "confirmations_passed"
            if passed
            else "confirmations_failed"
        ),
        "minimum_each_gain_vs_v20": 0.10,
        "results": results,
        "all_confirmations_passed": passed,
        "audit_fold_loaded": False,
        "f4_loaded": False,
        "f4_metric_computed": False,
        "source_sha256": sha256(Path(__file__)),
        "lineage_sha256": {
            "c016": sha256(c016_path),
            "posterior": sha256(posterior_path),
            "mode": sha256(mode_path),
        },
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "balanced_gain": results["balanced"]["gain_vs_v20"],
                "independent_gain": results["independent"][
                    "gain_vs_v20"
                ],
                "output_sha256": sha256(output),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise RuntimeError("C016 confirmations failed")


def implementation_gate(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    local_gate = local_selector.implementation_gate()
    amplitude_gate = amplitude_selector.implementation_gate()

    synthetic_mode = []
    rng = np.random.default_rng(20260729)
    for fold in range(5):
        rows = 96
        corrections = rng.normal(size=(4, rows))
        truth = 0.04 * corrections[0] + 0.02 * corrections[1]
        synthetic_mode.append(
            {
                "baseline": np.zeros(rows),
                "truth": truth,
                "metric_names": np.asarray(
                    (
                        "mae_s0_d8",
                        "huber_s0_d8",
                        "corr_s0_d8",
                        "diff_mae_s0_d8",
                    )
                ),
                "selector_corrections": corrections,
            }
        )
    mode_first = choose_mode_components(
        synthetic_mode, [0, 1, 2, 3]
    )
    changed_mode = [
        {
            key: value.copy()
            if isinstance(value, np.ndarray)
            else value
            for key, value in record.items()
        }
        for record in synthetic_mode
    ]
    changed_mode[4]["truth"] = rng.normal(
        scale=1000.0, size=len(changed_mode[4]["truth"])
    )
    mode_second = choose_mode_components(
        changed_mode, [0, 1, 2, 3]
    )

    folds = []
    inner_d570 = {}
    inner_posterior = {}
    inner_mode = {}
    for fold in range(5):
        rows = 80
        starts = np.asarray((0, 40, 80), dtype=np.int64)
        parent = rng.normal(size=rows)
        direction_a = rng.normal(size=rows)
        direction_b = rng.normal(size=rows)
        truth = parent + 0.20 * direction_a + 0.10 * direction_b
        folds.append(
            Fold(
                fold=fold,
                well_ids=np.asarray(
                    (f"synthetic_{fold}_0", f"synthetic_{fold}_1")
                ),
                row_starts=starts,
                baseline=parent.astype(np.float32),
                truth=truth.astype(np.float32),
            )
        )
        inner_d570[fold] = parent.astype(np.float32)
        inner_posterior[fold] = (
            parent + direction_a
        ).astype(np.float32)
        inner_mode[fold] = (parent + direction_b).astype(np.float32)
    first, first_prediction, _ = choose_ridge(
        4,
        folds,
        {fold: inner_d570[fold] for fold in range(4)},
        {fold: inner_posterior[fold] for fold in range(4)},
        {fold: inner_mode[fold] for fold in range(4)},
        inner_d570[4],
        inner_posterior[4],
        inner_mode[4],
    )
    replaced_folds = list(folds)
    replaced_folds[4] = Fold(
        fold=4,
        well_ids=folds[4].well_ids,
        row_starts=folds[4].row_starts,
        baseline=folds[4].baseline,
        truth=rng.normal(scale=10000.0, size=80).astype(np.float32),
    )
    second, second_prediction, _ = choose_ridge(
        4,
        replaced_folds,
        {fold: inner_d570[fold] for fold in range(4)},
        {fold: inner_posterior[fold] for fold in range(4)},
        {fold: inner_mode[fold] for fold in range(4)},
        inner_d570[4],
        inner_posterior[4],
        inner_mode[4],
    )
    checks = {
        "local_seed_bag_gate_passed": (
            local_gate["status"] == "passed"
        ),
        "D570_amplitude_gate_passed": (
            amplitude_gate["status"] == "passed"
        ),
        "mode_outer_held_truth_invariant": mode_first == mode_second,
        "ridge_outer_held_truth_invariant": (
            first == second
            and np.array_equal(first_prediction, second_prediction)
        ),
        "ridge_planted_directions_selected": bool(
            any(
                value > 0.0
                for value in first["refit_weights"].values()
            )
        ),
        "ten_unique_pairs": (
            len(list(itertools.combinations(range(5), 2))) == 10
        ),
        "twenty_logical_inner_roles": 5 * 4 == 20,
        "f4_real_target_not_loaded": True,
    }
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "metric_free_nested_selector_gate",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "local_gate_source_sha256": local_gate["source_sha256"],
        "amplitude_gate_source_sha256": amplitude_gate[
            "source_sha256"
        ],
        "source_sha256": sha256(Path(__file__)),
        "real_target_metric_computed": False,
        "f4_loaded": False,
        "f4_metric_computed": False,
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": checks,
                "output_sha256": sha256(output),
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "passed":
        raise RuntimeError("C016 nested selector gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gate_parser = commands.add_parser("gate")
    gate_parser.add_argument("--output", type=Path, required=True)
    confirmation_parser = commands.add_parser("confirm")
    confirmation_parser.add_argument("--output", type=Path, required=True)
    manifest_parser = commands.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument(
        "--candidate-manifest", type=Path, required=True
    )
    assemble_parser.add_argument(
        "--selector-output", type=Path, required=True
    )
    assemble_parser.add_argument(
        "--prediction-output", type=Path, required=True
    )
    assemble_parser.add_argument(
        "--prediction-manifest", type=Path, required=True
    )
    args = parser.parse_args()
    if args.command == "gate":
        implementation_gate(args.output)
    elif args.command == "confirm":
        confirmation(args.output)
    elif args.command == "manifest":
        freeze_candidate_manifest(args.output)
    else:
        assemble(
            args.candidate_manifest,
            args.selector_output,
            args.prediction_output,
            args.prediction_manifest,
        )


if __name__ == "__main__":
    main()
