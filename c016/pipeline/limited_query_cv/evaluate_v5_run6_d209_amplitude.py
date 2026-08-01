from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


RUN_ID = "v5_batch2_run_006_goal_050"
AMPLITUDES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
METRICS = (
    "pooled_rmse",
    "mean_well_rmse",
    "toe_quartile_rmse",
    "endpoint_rmse",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def fold_metrics(
    baseline: np.ndarray,
    truth: np.ndarray,
    correction: np.ndarray,
    starts: np.ndarray,
    amplitude: float,
) -> dict[str, float]:
    prediction = baseline.astype(np.float64) + amplitude * correction.astype(
        np.float64
    )
    error = prediction - truth.astype(np.float64)
    well_scores = []
    toe_parts = []
    endpoints = []
    for left, right in zip(starts[:-1], starts[1:]):
        local = error[left:right]
        well_scores.append(float(np.sqrt(np.mean(np.square(local)))))
        toe_left = int(right - max((right - left) // 4, 1))
        toe_parts.append(error[toe_left:right])
        endpoints.append(float(error[right - 1]))
    return {
        "pooled_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_well_rmse": float(np.mean(well_scores)),
        "toe_quartile_rmse": float(
            np.sqrt(np.mean(np.square(np.concatenate(toe_parts))))
        ),
        "endpoint_rmse": float(np.sqrt(np.mean(np.square(endpoints)))),
    }


def combined_metrics(
    folds: list[dict[str, np.ndarray]],
    selected_folds: list[int],
    amplitude: float,
) -> dict[str, float]:
    squared = []
    well_scores = []
    toe_parts = []
    endpoints = []
    for fold in selected_folds:
        record = folds[fold]
        prediction = record["baseline"].astype(np.float64) + amplitude * record[
            "correction"
        ].astype(np.float64)
        error = prediction - record["truth"].astype(np.float64)
        squared.append(np.square(error))
        for left, right in zip(record["row_starts"][:-1], record["row_starts"][1:]):
            local = error[left:right]
            well_scores.append(float(np.sqrt(np.mean(np.square(local)))))
            toe_left = int(right - max((right - left) // 4, 1))
            toe_parts.append(error[toe_left:right])
            endpoints.append(float(error[right - 1]))
    return {
        "pooled_rmse": float(np.sqrt(np.mean(np.concatenate(squared)))),
        "mean_well_rmse": float(np.mean(well_scores)),
        "toe_quartile_rmse": float(
            np.sqrt(np.mean(np.square(np.concatenate(toe_parts))))
        ),
        "endpoint_rmse": float(np.sqrt(np.mean(np.square(endpoints)))),
    }


def select_amplitude(
    folds: list[dict[str, np.ndarray]],
    training_folds: list[int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    parent_fold_scores = {
        fold: fold_metrics(
            folds[fold]["baseline"],
            folds[fold]["truth"],
            folds[fold]["correction"],
            folds[fold]["row_starts"],
            1.0,
        )["pooled_rmse"]
        for fold in training_folds
    }
    records: list[dict[str, object]] = []
    for amplitude in AMPLITUDES:
        metrics = combined_metrics(folds, training_folds, amplitude)
        local_scores = {
            fold: fold_metrics(
                folds[fold]["baseline"],
                folds[fold]["truth"],
                folds[fold]["correction"],
                folds[fold]["row_starts"],
                amplitude,
            )["pooled_rmse"]
            for fold in training_folds
        }
        eligible = all(
            local_scores[fold] <= parent_fold_scores[fold] + 1e-12
            for fold in training_folds
        )
        records.append(
            {
                "amplitude": amplitude,
                "eligible": eligible,
                "metrics": metrics,
                "fold_rmse": local_scores,
                "fold_gain_vs_amplitude_one": {
                    fold: parent_fold_scores[fold] - local_scores[fold]
                    for fold in training_folds
                },
            }
        )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise RuntimeError("amplitude one must be an eligible fallback")
    ranks = np.column_stack(
        [
            rankdata(
                [record["metrics"][metric] for record in eligible],
                method="average",
            )
            for metric in METRICS
        ]
    )
    for record, values in zip(eligible, ranks):
        record["metric_ranks"] = {
            metric: float(rank)
            for metric, rank in zip(METRICS, values.tolist())
        }
        record["mean_rank"] = float(np.mean(values))
    selected = min(
        eligible,
        key=lambda record: (
            record["mean_rank"],
            record["metrics"]["pooled_rmse"],
            record["metrics"]["mean_well_rmse"],
            record["metrics"]["toe_quartile_rmse"],
            record["metrics"]["endpoint_rmse"],
            abs(float(record["amplitude"]) - 1.0),
            float(record["amplitude"]),
        ),
    )
    return selected, records


def synthetic_fold(
    fold: int,
    coefficient: float,
    rows: int = 64,
) -> dict[str, np.ndarray]:
    correction = np.sin(np.linspace(0.0, 3.0 * np.pi, rows)) + np.linspace(
        0.0, 1.0, rows
    )
    correction[0] = 0.0
    baseline = np.zeros(rows, dtype=np.float64)
    truth = coefficient * correction
    return {
        "well_ids": np.asarray((f"f{fold}a", f"f{fold}b")),
        "row_starts": np.asarray((0, rows // 2, rows), dtype=np.int64),
        "baseline": baseline,
        "truth": truth,
        "correction": correction,
    }


def implementation_gate() -> dict[str, object]:
    stable = [synthetic_fold(fold, 1.5) for fold in range(4)]
    held_exclusion = True
    selected = []
    for held in range(4):
        training = [fold for fold in range(4) if fold != held]
        choice, _ = select_amplitude(stable, training)
        altered = [
            {key: value.copy() for key, value in record.items()}
            for record in stable
        ]
        altered[held]["truth"][:] = 10_000.0 + held
        choice_after, _ = select_amplitude(altered, training)
        selected.append(float(choice["amplitude"]))
        held_exclusion &= choice["amplitude"] == choice_after["amplitude"]

    distractor = [
        synthetic_fold(0, 2.5, 1000),
        synthetic_fold(1, 2.5, 1000),
        synthetic_fold(2, 0.0, 10),
    ]
    _, grid = select_amplitude(distractor, [0, 1, 2])
    pooled_best = min(
        grid, key=lambda record: record["metrics"]["pooled_rmse"]
    )
    amplitude_25 = next(
        record for record in grid if record["amplitude"] == 2.5
    )
    parent = np.arange(32, dtype=np.float32)
    correction = np.linspace(0.0, 3.0, 32, dtype=np.float32)
    checks = {
        "stable_amplitude_recovered": selected == [1.5] * 4,
        "held_exclusion": held_exclusion,
        "pooled_distractor_is_amplitude_2_5": (
            float(pooled_best["amplitude"]) == 2.5
        ),
        "fold_negative_distractor_rejected": not bool(amplitude_25["eligible"]),
        "amplitude_one_parent_exact": np.array_equal(
            parent + 1.0 * correction,
            parent + correction,
        ),
        "stable_layout": all(
            len(record["well_ids"]) + 1 == len(record["row_starts"])
            and record["row_starts"][0] == 0
            and record["row_starts"][-1] == len(record["truth"])
            for record in stable
        ),
        "finite": all(
            np.isfinite(record["truth"]).all()
            and np.isfinite(record["correction"]).all()
            for record in stable + distractor
        ),
    }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "accepted_residual_parent_calibration",
        "variant": "D209_clean_amplitude_selector_gate",
        "stage": "metric_free_gate",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "amplitudes": list(AMPLITUDES),
        "metrics": list(METRICS),
        "source_sha256": sha256(Path(__file__)),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }


def load_d209(path: Path) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError("D209 protected flags changed")
        all_ids = cache["well_ids"].astype(str)
        all_folds = cache["folds"].astype(np.int8)
        all_starts = cache["row_starts"].astype(np.int64)
        all_baseline = cache["baseline"].astype(np.float32)
        all_truth = cache["truth"].astype(np.float32)
        all_prediction = cache["prediction"].astype(np.float32)
    folds = []
    for fold in range(4):
        indices = np.flatnonzero(all_folds == fold)
        lengths = all_starts[indices + 1] - all_starts[indices]

        def gather(values: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [
                    values[all_starts[index] : all_starts[index + 1]]
                    for index in indices
                ]
            )

        baseline = gather(all_baseline)
        folds.append(
            {
                "well_ids": all_ids[indices],
                "row_starts": np.r_[0, np.cumsum(lengths)].astype(np.int64),
                "baseline": baseline,
                "truth": gather(all_truth),
                "correction": gather(all_prediction) - baseline,
            }
        )
    return folds, {
        "well_ids": all_ids,
        "folds": all_folds,
        "row_starts": all_starts,
        "baseline": all_baseline,
        "truth": all_truth,
        "d209_prediction": all_prediction,
    }


def evaluate(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds, combined = load_d209(path)
    predictions = []
    fold_records = []
    for held in range(4):
        training = [fold for fold in range(4) if fold != held]
        selected, grid = select_amplitude(folds, training)
        amplitude = float(selected["amplitude"])
        prediction = (
            folds[held]["baseline"].astype(np.float64)
            + amplitude * folds[held]["correction"].astype(np.float64)
        )
        parent = (
            folds[held]["baseline"].astype(np.float64)
            + folds[held]["correction"].astype(np.float64)
        )
        truth = folds[held]["truth"]
        predictions.append(prediction.astype(np.float32))
        fold_records.append(
            {
                "held_fold": held,
                "selection_folds": training,
                "selected": selected,
                "training_grid": grid,
                "held_v20_rmse": rmse(folds[held]["baseline"], truth),
                "held_d209_rmse": rmse(parent, truth),
                "held_selected_rmse": rmse(prediction, truth),
                "held_gain_vs_v20": rmse(folds[held]["baseline"], truth)
                - rmse(prediction, truth),
                "held_gain_vs_d209": rmse(parent, truth) - rmse(prediction, truth),
            }
        )
    prediction = np.concatenate(predictions)
    baseline = combined["baseline"]
    truth = combined["truth"]
    d209 = combined["d209_prediction"]
    v20_rmse = rmse(baseline, truth)
    d209_rmse = rmse(d209, truth)
    selected_rmse = rmse(prediction, truth)
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "accepted_residual_parent_calibration",
        "variant": "D209_clean_amplitude_response",
        "stage": "selection_clean_F0_F3",
        "status": "complete",
        "amplitudes": list(AMPLITUDES),
        "metrics": list(METRICS),
        "v20_rmse": v20_rmse,
        "d209_rmse": d209_rmse,
        "selected_rmse": selected_rmse,
        "gain_vs_v20": v20_rmse - selected_rmse,
        "gain_vs_d209": d209_rmse - selected_rmse,
        "fold_records": fold_records,
        "d209_path": str(path),
        "d209_sha256": sha256(path),
        "source_sha256": sha256(Path(__file__)),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    combined["prediction"] = prediction.astype(np.float32)
    combined["audit_fold_loaded"] = np.asarray(False)
    combined["confirmation_regroupings_loaded"] = np.asarray(False)
    return payload, combined


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--d209", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    if args.gate:
        if args.d209 is not None or args.output_npz is not None:
            raise ValueError("gate accepts only --output-json")
        payload = implementation_gate()
        write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] != "passed":
            raise RuntimeError("D209 amplitude selector gate failed")
        return
    if args.d209 is None or args.output_npz is None:
        raise ValueError("evaluation requires --d209 and --output-npz")
    if args.output_npz.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_npz}")
    payload, cache = evaluate(args.d209)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **cache)
    payload["prediction"] = str(args.output_npz)
    payload["prediction_sha256"] = sha256(args.output_npz)
    write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "v20_rmse": payload["v20_rmse"],
                "d209_rmse": payload["d209_rmse"],
                "selected_rmse": payload["selected_rmse"],
                "gain_vs_v20": payload["gain_vs_v20"],
                "gain_vs_d209": payload["gain_vs_d209"],
                "folds": [
                    {
                        "held_fold": record["held_fold"],
                        "amplitude": record["selected"]["amplitude"],
                        "gain_vs_v20": record["held_gain_vs_v20"],
                        "gain_vs_d209": record["held_gain_vs_d209"],
                    }
                    for record in payload["fold_records"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
