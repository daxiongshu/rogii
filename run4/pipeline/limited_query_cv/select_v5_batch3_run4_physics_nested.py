from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv import evaluate_v5_run6_local_residual_clean as selector
from limited_query_cv import train_v5_run6_local_v20_residual as parent
from rogii.data import DEFAULT_DATA_ROOT, load_well


RUN_ID = "v5_batch3_run_004_goal_020"
FAMILY = "physics_prior_mixed_uniform_nested"
SNAPSHOTS = (2, 4, 8, 12, 16)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)


def extract_fold(
    path: Path,
    fold: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[float, np.ndarray],
]:
    with np.load(path, allow_pickle=False) as cache:
        if (
            str(cache["run_id"]) != RUN_ID
            or str(cache["family"]) != FAMILY
            or bool(cache["hidden_truth_read_for_prediction"])
            or bool(cache["hidden_truth_stored"])
            or bool(cache["protected_metrics_computed"])
            or "truth" in cache.files
        ):
            raise RuntimeError(f"{path}: nested prediction firewall failed")
        ids = cache["well_ids"].astype(str)
        folds = cache["well_folds"].astype(np.int64)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        raw_predictions = {
            temperature: cache[
                selector.prediction_key(temperature)
            ].astype(np.float32)
            for temperature in TEMPERATURES
        }
    selected = np.flatnonzero(folds == fold)
    if not len(selected):
        raise RuntimeError(f"{path}: fold {fold} missing")
    selected_ids = ids[selected]
    lengths = np.diff(starts)[selected]
    selected_starts = np.concatenate(
        ([0], np.cumsum(lengths, dtype=np.int64))
    )

    def take(values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [values[starts[index] : starts[index + 1]] for index in selected]
        )

    return (
        selected_ids,
        selected_starts,
        take(baseline),
        {temperature: take(values) for temperature, values in raw_predictions.items()},
    )


def role_name(left: int, right: int) -> str:
    return f"pair_{min(left, right)}_{max(left, right)}"


def role_fold_tuple(
    root: Path,
    outer: int,
    inner: int,
    data_root: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, float], np.ndarray],
    dict[tuple[int, float], str],
]:
    role = role_name(outer, inner)
    ids = None
    starts = None
    baseline = None
    predictions: dict[tuple[int, float], np.ndarray] = {}
    hashes: dict[tuple[int, float], str] = {}
    for snapshot in SNAPSHOTS:
        path = root / role / f"epoch{snapshot:03d}_prediction.npz"
        local_ids, local_starts, local_baseline, local_predictions = (
            extract_fold(path, inner)
        )
        if ids is None:
            ids = local_ids
            starts = local_starts
            baseline = local_baseline
        elif (
            not np.array_equal(ids, local_ids)
            or not np.array_equal(starts, local_starts)
            or not np.array_equal(baseline, local_baseline)
        ):
            raise RuntimeError(f"{role}: snapshot layout changed")
        digest = parent.sha256(path)
        for temperature, prediction in local_predictions.items():
            predictions[(snapshot, temperature)] = prediction
            hashes[(snapshot, temperature)] = digest
    assert ids is not None
    assert starts is not None
    assert baseline is not None
    truth_parts = []
    for well_id in ids:
        well = load_well(well_id, data_root)
        truth_parts.append(well.tvt[well.prediction_indices])
    truth = np.concatenate(truth_parts).astype(np.float32)
    if len(truth) != len(baseline):
        raise RuntimeError(f"{role}: selector truth layout changed")
    return ids, starts, baseline, truth, predictions, hashes


def verify_roles(root: Path) -> list[dict[str, Any]]:
    expected = [
        *(role_name(left, right) for left, right in itertools.combinations(range(5), 2)),
        *(f"outer_{fold}" for fold in range(5)),
    ]
    records = []
    for role in expected:
        role_root = root / role
        result_path = role_root / "result.json"
        result = json.loads(result_path.read_text())
        if (
            result["run_id"] != RUN_ID
            or result["family"] != FAMILY
            or result["role"] != role
            or result["status"] != "complete_target_free_nested_role"
            or result["hidden_truth_read_for_prediction"]
            or result["hidden_truth_stored"]
            or result["protected_metrics_computed"]
            or len(result["snapshots"]) != len(SNAPSHOTS)
        ):
            raise RuntimeError(f"{result_path}: role contract failed")
        files = [result_path]
        for item in result["snapshots"]:
            checkpoint = Path(item["checkpoint"])
            prediction = Path(item["prediction"])
            if (
                parent.sha256(checkpoint) != item["checkpoint_sha256"]
                or parent.sha256(prediction) != item["prediction_sha256"]
            ):
                raise RuntimeError(f"{role}: snapshot checksum changed")
            files.extend((checkpoint, prediction))
        records.extend(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": parent.sha256(path),
            }
            for path in files
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--selector-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    if any(
        path.exists()
        for path in (
            args.prediction_output,
            args.selector_output,
            args.manifest_output,
        )
    ):
        raise FileExistsError("refusing to overwrite nested selection output")

    artifact_records = verify_roles(args.root)
    selected_records = []
    output_ids = []
    output_folds = []
    output_lengths = []
    output_baseline = []
    output_prediction = []
    for outer in range(5):
        role_folds: list[Any] = [None] * 5
        training_folds = [fold for fold in range(5) if fold != outer]
        for inner in training_folds:
            role_folds[inner] = role_fold_tuple(
                args.root, outer, inner, args.data_root
            )
        selected, _ = selector.select_recipe(role_folds, training_folds)
        snapshot = int(selected["snapshot"])
        temperature = float(selected["temperature"])
        weight = float(selected["weight"])
        outer_role = f"outer_{outer}"
        prediction_path = (
            args.root
            / outer_role
            / f"epoch{snapshot:03d}_prediction.npz"
        )
        ids, starts, baseline, predictions = extract_fold(
            prediction_path, outer
        )
        prediction = baseline + weight * (
            predictions[temperature] - baseline
        )
        result = json.loads((args.root / outer_role / "result.json").read_text())
        checkpoint_item = next(
            item for item in result["snapshots"] if item["epoch"] == snapshot
        )
        selected_records.append(
            {
                "outer_fold": outer,
                "selection_folds": training_folds,
                "snapshot": snapshot,
                "temperature": temperature,
                "physics_blend_weight": weight,
                "checkpoint": checkpoint_item["checkpoint"],
                "checkpoint_sha256": checkpoint_item["checkpoint_sha256"],
                "held_prediction": str(prediction_path),
                "held_prediction_sha256": parent.sha256(prediction_path),
            }
        )
        output_ids.extend(ids.tolist())
        output_folds.extend([outer] * len(ids))
        output_lengths.extend(np.diff(starts).tolist())
        output_baseline.append(baseline)
        output_prediction.append(prediction.astype(np.float32))

    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.prediction_output,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray(FAMILY),
        well_ids=np.asarray(output_ids),
        folds=np.asarray(output_folds, dtype=np.int8),
        row_starts=np.concatenate(
            ([0], np.cumsum(output_lengths, dtype=np.int64))
        ),
        baseline=np.concatenate(output_baseline).astype(np.float32),
        prediction=np.concatenate(output_prediction).astype(np.float32),
        hidden_truth_stored=np.asarray(False),
        protected_metric_computed=np.asarray(False),
        F4_metric_recomputed=np.asarray(False),
    )
    selector_payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "complete_nested_selection_without_outer_metric",
        "roles": selected_records,
        "prediction": str(args.prediction_output),
        "prediction_sha256": parent.sha256(args.prediction_output),
        "F4_metric_recomputed": False,
        "kaggle_submission_authorized": False,
        "source_sha256": parent.sha256(Path(__file__)),
    }
    args.selector_output.write_text(
        json.dumps(selector_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "all_15_nested_roles_and_selected_outer_predictions_frozen",
        "records": artifact_records,
        "record_count": len(artifact_records),
        "selector": str(args.selector_output),
        "selector_sha256": parent.sha256(args.selector_output),
        "prediction": str(args.prediction_output),
        "prediction_sha256": parent.sha256(args.prediction_output),
        "hidden_truth_stored": False,
        "F4_metric_recomputed": False,
    }
    args.manifest_output.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest_payload["status"],
                "record_count": len(artifact_records),
                "prediction_sha256": manifest_payload["prediction_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
