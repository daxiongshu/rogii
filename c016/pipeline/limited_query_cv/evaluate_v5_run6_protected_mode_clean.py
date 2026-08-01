from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from limited_query_cv.evaluate_v5_run6_protected_mode_bank import (
    OUTER_WEIGHTS,
    assemble_selector_corrections,
    rmse,
    selector_priority,
)


RUN_ID = "v5_batch2_run_006_goal_050"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold(
    path: Path,
) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as cache:
        if int(cache["fold"]) not in range(4):
            raise ValueError(f"{path}: unexpected fold")
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise ValueError(f"{path}: protected firewall failed")
        candidate_names = cache["candidate_names"].astype(str)
        checkpoint_names = cache["checkpoint_names"].astype(str)
        if (
            np.any(np.char.startswith(candidate_names, "strat_"))
            or any("strat" in name for name in checkpoint_names)
        ):
            raise ValueError(
                f"{path}: quarantined stratified-fold parent entered "
                "an original-fold mode bank"
            )
        starts = cache["row_starts"].astype(np.int64)
        corrections = cache["corrections"].astype(np.float32)
        metrics = cache["metrics"].astype(np.float32)
        shifted_metrics = cache["shifted_control_metrics"].astype(
            np.float32
        )
        return {
            "fold": cache["fold"].astype(np.int8),
            "well_ids": cache["well_ids"].astype(str),
            "starts": starts,
            "baseline": cache["baseline"].astype(np.float32),
            "truth": cache["truth"].astype(np.float32),
            "metric_names": cache["metric_names"].astype(str),
            "selector_corrections": assemble_selector_corrections(
                metrics, corrections, starts
            ),
            "shifted_selector_corrections": (
                assemble_selector_corrections(
                    shifted_metrics, corrections, starts
                )
            ),
            "sha256": sha256(path),
        }


def choose_components(
    folds: list[dict[str, np.ndarray | str]],
    training_folds: list[int],
    *,
    shifted_control: bool,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    metric_names = folds[0]["metric_names"]
    assert isinstance(metric_names, np.ndarray)
    correction_key = (
        "shifted_selector_corrections"
        if shifted_control
        else "selector_corrections"
    )
    errors = []
    for fold in range(4):
        baseline = folds[fold]["baseline"]
        truth = folds[fold]["truth"]
        assert isinstance(baseline, np.ndarray)
        assert isinstance(truth, np.ndarray)
        errors.append(baseline.astype(np.float64) - truth.astype(np.float64))
    available = set(range(len(metric_names)))
    selected = []
    for _ in range(3):
        current_sse = sum(
            float(errors[fold] @ errors[fold])
            for fold in training_folds
        )
        cross = np.zeros(len(metric_names), dtype=np.float64)
        denominator = np.zeros(len(metric_names), dtype=np.float64)
        row_count = 0
        for fold in training_folds:
            corrections = folds[fold][correction_key]
            assert isinstance(corrections, np.ndarray)
            cross += corrections.astype(np.float64) @ errors[fold]
            denominator += np.einsum(
                "ij,ij->i",
                corrections.astype(np.float64),
                corrections.astype(np.float64),
            )
            row_count += len(errors[fold])
        candidates = []
        for selector in available:
            for weight in OUTER_WEIGHTS[1:]:
                sse = (
                    current_sse
                    + 2.0 * weight * cross[selector]
                    + weight * weight * denominator[selector]
                )
                candidates.append(
                    (
                        sse,
                        weight,
                        selector_priority(str(metric_names[selector])),
                        selector,
                    )
                )
        sse, weight, _, selector = min(candidates)
        if sse >= current_sse:
            break
        before_rmse = float(np.sqrt(current_sse / row_count))
        after_rmse = float(np.sqrt(sse / row_count))
        selected.append(
            {
                "selector_index": selector,
                "metric": str(metric_names[selector]),
                "weight": weight,
                "training_rmse_before": before_rmse,
                "training_rmse_after": after_rmse,
                "training_incremental_gain": before_rmse - after_rmse,
            }
        )
        for fold in range(4):
            corrections = folds[fold][correction_key]
            assert isinstance(corrections, np.ndarray)
            errors[fold] += weight * corrections[selector]
        available.remove(selector)
    return selected, errors


def evaluate_variant(
    folds: list[dict[str, np.ndarray | str]],
    *,
    shifted_control: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    fold_predictions = []
    records = []
    correction_key = (
        "shifted_selector_corrections"
        if shifted_control
        else "selector_corrections"
    )
    for held in range(4):
        training_folds = [fold for fold in range(4) if fold != held]
        selected, _ = choose_components(
            folds,
            training_folds,
            shifted_control=shifted_control,
        )
        baseline = folds[held]["baseline"]
        truth = folds[held]["truth"]
        corrections = folds[held][correction_key]
        assert isinstance(baseline, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(corrections, np.ndarray)
        prediction = baseline.astype(np.float64).copy()
        for component in selected:
            prediction += (
                float(component["weight"])
                * corrections[int(component["selector_index"])]
            )
        before = rmse(baseline, truth)
        after = rmse(prediction, truth)
        records.append(
            {
                "held_fold": held,
                "selection_folds": training_folds,
                "components": selected,
                "held_baseline_rmse": before,
                "held_selected_rmse": after,
                "held_gain": before - after,
            }
        )
        fold_predictions.append(prediction.astype(np.float32))
    return np.concatenate(fold_predictions), records


def evaluate(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds = [
        load_fold(args.root / f"protected_mode_bank_f{fold}.npz")
        for fold in range(4)
    ]
    names = folds[0]["metric_names"]
    assert isinstance(names, np.ndarray)
    for fold, record in enumerate(folds):
        if int(record["fold"]) != fold:
            raise ValueError("mode-bank fold order differs")
        local_names = record["metric_names"]
        assert isinstance(local_names, np.ndarray)
        if not np.array_equal(names, local_names):
            raise ValueError("metric schema differs between folds")
    prediction, records = evaluate_variant(
        folds, shifted_control=False
    )
    shifted_prediction, shifted_records = evaluate_variant(
        folds, shifted_control=True
    )
    baseline = np.concatenate(
        [record["baseline"] for record in folds]
    )
    truth = np.concatenate([record["truth"] for record in folds])
    baseline_score = rmse(baseline, truth)
    selected_score = rmse(prediction, truth)
    shifted_score = rmse(shifted_prediction, truth)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": "protected_v20_posterior_mode_bank_clean_selector",
        "status": "complete_selection_clean_F0_F3",
        "baseline_rmse": baseline_score,
        "selected_rmse": selected_score,
        "gain": baseline_score - selected_score,
        "fold_records": records,
        "shifted_hidden_gr_negative_control": {
            "rmse": shifted_score,
            "gain": baseline_score - shifted_score,
            "fold_records": shifted_records,
        },
        "bank_sha256": [str(record["sha256"]) for record in folds],
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "selection_clean_nested_selector_run": True,
    }
    row_starts = [0]
    well_ids = []
    well_folds = []
    for fold, record in enumerate(folds):
        ids = record["well_ids"]
        starts = record["starts"]
        assert isinstance(ids, np.ndarray)
        assert isinstance(starts, np.ndarray)
        well_ids.extend(ids.tolist())
        well_folds.extend([fold] * len(ids))
        for left, right in zip(starts[:-1], starts[1:]):
            row_starts.append(row_starts[-1] + right - left)
    payload = {
        "well_ids": np.asarray(well_ids),
        "folds": np.asarray(well_folds, dtype=np.int8),
        "row_starts": np.asarray(row_starts, dtype=np.int64),
        "baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "shifted_control_prediction": shifted_prediction.astype(np.float32),
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
        raise FileExistsError("refusing to overwrite mode-clean output")
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
                "fold_records": result["fold_records"],
                "shifted_hidden_gr_negative_control": result[
                    "shifted_hidden_gr_negative_control"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
