from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


RUN_ID = "v5_batch2_run_006_goal_050"
SNAPSHOTS = (2, 4, 8, 12, 16)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)
WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64) - truth.astype(np.float64)
                )
            )
        )
    )


def prediction_key(temperature: float) -> str:
    return f"prediction_t{int(temperature * 100):03d}"


def load_fold(
    root: Path, fold: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, float], np.ndarray],
    dict[tuple[int, float], str],
]:
    well_ids = None
    starts = None
    baseline = None
    truth = None
    predictions = {}
    hashes = {}
    for snapshot in SNAPSHOTS:
        path = (
            root
            / f"local_v20_residual_base12_f{fold}"
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
            if well_ids is None:
                well_ids = local_ids
                starts = local_starts
                baseline = local_baseline
                truth = local_truth
            elif (
                not np.array_equal(well_ids, local_ids)
                or not np.array_equal(starts, local_starts)
                or not np.array_equal(baseline, local_baseline)
                or not np.array_equal(truth, local_truth)
            ):
                raise ValueError(f"{path}: fold layout changed")
            for temperature in TEMPERATURES:
                predictions[(snapshot, temperature)] = cache[
                    prediction_key(temperature)
                ].astype(np.float32)
        digest = sha256(path)
        for temperature in TEMPERATURES:
            hashes[(snapshot, temperature)] = digest
    assert well_ids is not None
    assert starts is not None
    assert baseline is not None
    assert truth is not None
    return well_ids, starts, baseline, truth, predictions, hashes


def subset_metrics(
    baseline_parts: list[np.ndarray],
    truth_parts: list[np.ndarray],
    prediction_parts: list[np.ndarray],
    starts_parts: list[np.ndarray],
    weight: float,
) -> dict[str, float]:
    baseline = np.concatenate(baseline_parts)
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    blend = baseline + weight * (prediction - baseline)
    toe_errors = []
    endpoint_errors = []
    for local_baseline, local_truth, local_prediction, starts in zip(
        baseline_parts, truth_parts, prediction_parts, starts_parts
    ):
        local_blend = local_baseline + weight * (
            local_prediction - local_baseline
        )
        for left, right in zip(starts[:-1], starts[1:]):
            length = right - left
            toe_left = right - max(length // 4, 1)
            toe_errors.append(
                local_blend[toe_left:right] - local_truth[toe_left:right]
            )
            endpoint_errors.append(local_blend[right - 1] - local_truth[right - 1])
    toe_error = np.concatenate(toe_errors)
    endpoint_error = np.asarray(endpoint_errors)
    return {
        "blend_rmse": rmse(blend, truth),
        "standalone_rmse": rmse(prediction, truth),
        "toe_quartile_rmse": float(np.sqrt(np.mean(np.square(toe_error)))),
        "endpoint_rmse": float(np.sqrt(np.mean(np.square(endpoint_error)))),
    }


def select_recipe(
    folds: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[tuple[int, float], np.ndarray],
            dict[tuple[int, float], str],
        ]
    ],
    training_folds: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = []
    for snapshot in SNAPSHOTS:
        for temperature in TEMPERATURES:
            baseline_parts = [folds[fold][2] for fold in training_folds]
            truth_parts = [folds[fold][3] for fold in training_folds]
            prediction_parts = [
                folds[fold][4][(snapshot, temperature)]
                for fold in training_folds
            ]
            starts_parts = [folds[fold][1] for fold in training_folds]
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
            item["snapshot"],
            abs(item["temperature"] - 1.0),
            item["temperature"],
        ),
    )
    return selected, records


def evaluate(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds = [load_fold(args.root, fold) for fold in range(4)]
    selected_predictions = []
    selected_records = []
    selected_hashes = []
    for held in range(4):
        training = [fold for fold in range(4) if fold != held]
        selected, grid = select_recipe(folds, training)
        snapshot = int(selected["snapshot"])
        temperature = float(selected["temperature"])
        weight = float(selected["weight"])
        baseline = folds[held][2]
        candidate = folds[held][4][(snapshot, temperature)]
        prediction = baseline + weight * (candidate - baseline)
        selected_predictions.append(prediction.astype(np.float32))
        selected_hashes.append(
            folds[held][5][(snapshot, temperature)]
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
        "variant": "local_v20_residual_other_three_fold_selector",
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
                        for left, right in zip(
                            fold[1][:-1], fold[1][1:]
                        )
                    ]
                ),
            )
        ).astype(np.int64),
        "baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
    }
    return result, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "limited_query_cv/runs/v5_batch2_run_006_goal_050"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_npz.exists():
        raise FileExistsError("refusing to overwrite clean residual output")
    result, payload = evaluate(args)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(args.output_npz, **payload)
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline_rmse": result["baseline_rmse"],
                "selected_rmse": result["selected_rmse"],
                "gain": result["gain"],
                "fold_records": [
                    {
                        "held_fold": item["held_fold"],
                        "selected": item["selected"],
                        "held_gain": item["held_gain"],
                    }
                    for item in result["fold_records"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
