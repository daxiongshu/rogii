from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from limited_query_cv.evaluate_v5_run6_local_residual_clean import (
    RUN_ID,
    SNAPSHOTS,
    TEMPERATURES,
    WEIGHTS,
    load_fold,
    prediction_key,
    rmse,
    sha256,
    subset_metrics,
)


ORIGINAL = "seed20260831"
INDEPENDENT = "seed20261043"
BAG = "equal_seed_bag"
TREATMENTS = (ORIGINAL, INDEPENDENT, BAG)
TREATMENT_TIE_ORDER = {
    ORIGINAL: 0,
    BAG: 1,
    INDEPENDENT: 2,
}


def average_predictions(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    return np.mean(
        np.stack((first, second)).astype(np.float64),
        axis=0,
    )


def load_independent_fold(
    root: Path,
    fold: int,
    reference: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[tuple[int, float], np.ndarray],
        dict[tuple[int, float], str],
    ],
) -> tuple[
    dict[tuple[int, float], np.ndarray],
    dict[tuple[int, float], str],
]:
    predictions: dict[tuple[int, float], np.ndarray] = {}
    hashes: dict[tuple[int, float], str] = {}
    for snapshot in SNAPSHOTS:
        path = (
            root
            / f"local_v20_residual_base12_seed20261043_f{fold}"
            / f"epoch{snapshot:03d}_prediction.npz"
        )
        with np.load(path, allow_pickle=False) as cache:
            local_ids = cache["well_ids"].astype(str)
            local_starts = cache["row_starts"].astype(np.int64)
            local_baseline = cache["baseline"].astype(np.float32)
            local_truth = cache["truth"].astype(np.float32)
            if bool(cache["audit_fold_loaded"]) or bool(
                cache["confirmation_regroupings_loaded"]
            ):
                raise ValueError(f"{path}: protected firewall failed")
            if (
                not np.array_equal(reference[0], local_ids)
                or not np.array_equal(reference[1], local_starts)
                or not np.array_equal(reference[2], local_baseline)
                or not np.array_equal(reference[3], local_truth)
            ):
                raise ValueError(f"{path}: fold layout changed")
            for temperature in TEMPERATURES:
                predictions[(snapshot, temperature)] = cache[
                    prediction_key(temperature)
                ].astype(np.float32)
        digest = sha256(path)
        for temperature in TEMPERATURES:
            hashes[(snapshot, temperature)] = digest
    return predictions, hashes


def load_seed_fold(
    root: Path,
    fold: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[str, int, float], np.ndarray],
    dict[tuple[str, int, float], list[str]],
]:
    original = load_fold(root, fold)
    independent_predictions, independent_hashes = load_independent_fold(
        root,
        fold,
        original,
    )
    predictions: dict[tuple[str, int, float], np.ndarray] = {}
    hashes: dict[tuple[str, int, float], list[str]] = {}
    for snapshot in SNAPSHOTS:
        for temperature in TEMPERATURES:
            key = (snapshot, temperature)
            predictions[(ORIGINAL, snapshot, temperature)] = original[4][key]
            predictions[(INDEPENDENT, snapshot, temperature)] = (
                independent_predictions[key]
            )
            predictions[(BAG, snapshot, temperature)] = average_predictions(
                original[4][key],
                independent_predictions[key],
            )
            hashes[(ORIGINAL, snapshot, temperature)] = [original[5][key]]
            hashes[(INDEPENDENT, snapshot, temperature)] = [
                independent_hashes[key]
            ]
            hashes[(BAG, snapshot, temperature)] = [
                original[5][key],
                independent_hashes[key],
            ]
    return (
        original[0],
        original[1],
        original[2],
        original[3],
        predictions,
        hashes,
    )


def select_recipe(
    folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[tuple[str, int, float], np.ndarray],
            dict[tuple[str, int, float], list[str]],
        ]
    ],
    training_folds: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_parts = [folds[fold][2] for fold in training_folds]
    truth_parts = [folds[fold][3] for fold in training_folds]
    starts_parts = [folds[fold][1] for fold in training_folds]
    records: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        for snapshot in SNAPSHOTS:
            for temperature in TEMPERATURES:
                prediction_parts = [
                    folds[fold][4][(treatment, snapshot, temperature)]
                    for fold in training_folds
                ]
                weight_scores = []
                for weight in WEIGHTS:
                    metrics = subset_metrics(
                        baseline_parts,
                        truth_parts,
                        prediction_parts,
                        starts_parts,
                        weight,
                    )
                    weight_scores.append((metrics["blend_rmse"], weight, metrics))
                _, weight, metrics = min(weight_scores)
                records.append(
                    {
                        "treatment": treatment,
                        "snapshot": snapshot,
                        "temperature": temperature,
                        "weight": weight,
                        **metrics,
                    }
                )
    metric_names = (
        "blend_rmse",
        "standalone_rmse",
        "toe_quartile_rmse",
        "endpoint_rmse",
    )
    rank_matrix = np.column_stack(
        [
            rankdata(
                [record[metric] for record in records],
                method="average",
            )
            for metric in metric_names
        ]
    )
    for record, ranks in zip(records, rank_matrix):
        record["metric_ranks"] = {
            metric: float(rank)
            for metric, rank in zip(metric_names, ranks.tolist())
        }
        record["mean_rank"] = float(np.mean(ranks))
    selected = min(
        records,
        key=lambda item: (
            item["mean_rank"],
            item["blend_rmse"],
            item["standalone_rmse"],
            item["toe_quartile_rmse"],
            item["endpoint_rmse"],
            item["weight"],
            TREATMENT_TIE_ORDER[item["treatment"]],
            item["snapshot"],
            abs(item["temperature"] - 1.0),
            item["temperature"],
        ),
    )
    return selected, records


def implementation_gate() -> dict[str, object]:
    original = np.asarray((1.0, 3.0, 5.0, 7.0), dtype=np.float32)
    independent = np.asarray((2.0, 5.0, 8.0, 11.0), dtype=np.float32)
    bag = average_predictions(original, independent)
    independent_mean = (
        original.astype(np.float64) + independent.astype(np.float64)
    ) / 2.0
    parent = np.asarray((11200.0, 11201.0, 11202.0, 11203.0))
    zero_share = parent + 0.0 * (bag - parent)
    ids = np.asarray(("a", "b"))
    starts = np.asarray((0, 2, 4), dtype=np.int64)
    checks = {
        "equal_bag_exact": np.array_equal(bag, independent_mean),
        "original_fallback_exact": np.array_equal(original, original.copy()),
        "zero_share_exact_parent": np.array_equal(zero_share, parent),
        "all_finite": bool(np.isfinite(bag).all()),
        "bag_distinct_from_members": (
            not np.array_equal(bag, original)
            and not np.array_equal(bag, independent)
        ),
        "row_layout_stable": bool(
            len(ids) + 1 == len(starts)
            and starts[0] == 0
            and starts[-1] == len(bag)
            and np.all(np.diff(starts) > 0)
        ),
    }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_structural_latent_alignment",
        "variant": "local_residual_two_seed_bag_implementation_gate",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "bag": bag.tolist(),
        "source_sha256": sha256(Path(__file__)),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }


def evaluate(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds = [load_seed_fold(args.root, fold) for fold in range(4)]
    selected_predictions = []
    selected_records = []
    selected_hashes = []
    for held in range(4):
        training = [fold for fold in range(4) if fold != held]
        selected, grid = select_recipe(folds, training)
        treatment = str(selected["treatment"])
        snapshot = int(selected["snapshot"])
        temperature = float(selected["temperature"])
        weight = float(selected["weight"])
        baseline = folds[held][2]
        candidate = folds[held][4][(treatment, snapshot, temperature)]
        prediction = baseline + weight * (candidate - baseline)
        selected_predictions.append(prediction.astype(np.float32))
        selected_hashes.append(
            folds[held][5][(treatment, snapshot, temperature)]
        )
        selected_records.append(
            {
                "held_fold": held,
                "selection_folds": training,
                "selected": selected,
                "training_grid": grid,
                "held_baseline_rmse": rmse(baseline, folds[held][3]),
                "held_selected_rmse": rmse(prediction, folds[held][3]),
                "held_gain": rmse(baseline, folds[held][3])
                - rmse(prediction, folds[held][3]),
            }
        )
    baseline = np.concatenate([fold[2] for fold in folds])
    truth = np.concatenate([fold[3] for fold in folds])
    prediction = np.concatenate(selected_predictions)
    baseline_score = rmse(baseline, truth)
    selected_score = rmse(prediction, truth)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_structural_latent_alignment",
        "variant": "local_v20_residual_two_seed_other_three_fold_selector",
        "status": "complete_selection_clean_F0_F3",
        "baseline_rmse": baseline_score,
        "selected_rmse": selected_score,
        "gain": baseline_score - selected_score,
        "fold_records": selected_records,
        "selected_prediction_sha256": selected_hashes,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "selection_clean_nested_selector_run": True,
        "trajectory_treatments": list(TREATMENTS),
        "independent_seed": 20261043,
    }
    payload = {
        "well_ids": np.concatenate([fold[0] for fold in folds]),
        "folds": np.concatenate(
            [
                np.full(len(fold[0]), fold_index, dtype=np.int8)
                for fold_index, fold in enumerate(folds)
            ]
        ),
        "row_starts": np.concatenate(
            (
                [0],
                np.cumsum(
                    [
                        right - left
                        for fold in folds
                        for left, right in zip(fold[1][:-1], fold[1][1:])
                    ]
                ),
            )
        ).astype(np.int64),
        "baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
        "source_sha256": np.asarray(sha256(Path(__file__))),
    }
    return result, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-gate", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("limited_query_cv/runs/v5_batch2_run_006_goal_050"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    if args.output_json.exists() or (
        args.output_npz is not None and args.output_npz.exists()
    ):
        raise FileExistsError("refusing to overwrite residual seed bag")
    if args.implementation_gate:
        result = implementation_gate()
        payload = None
    else:
        if args.output_npz is None:
            parser.error("--output-npz is required for clean evaluation")
        result, payload = evaluate(args)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if payload is not None:
        np.savez_compressed(args.output_npz, **payload)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "baseline_rmse",
                    "selected_rmse",
                    "gain",
                    "checks",
                )
                if key in result
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] == "failed":
        raise RuntimeError("residual seed bag implementation gate failed")


if __name__ == "__main__":
    main()
