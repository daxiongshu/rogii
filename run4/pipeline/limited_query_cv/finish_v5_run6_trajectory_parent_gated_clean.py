from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from limited_query_cv import (
    evaluate_v5_run6_trajectory_segformer_clean as plain,
)
from limited_query_cv import (
    evaluate_v5_run6_trajectory_segformer_gated_clean as gated,
)
from limited_query_cv import (
    finish_v5_run6_trajectory_segformer_gated_clean as finish,
)


RUN_ID = finish.RUN_ID
FAMILY = finish.FAMILY
ROOT = finish.ROOT
D895_PATH = ROOT / "trajectory_parent_uncertainty_f0_result.json"
D895_SHA256 = (
    "e8323b4b649b7ebd8865f0e2fd55d4af4cc8d4e41dcd7e4a44bf07e4377c8f7e"
)
FINISH_SOURCE_SHA256 = (
    "0358a69f84aef2ff55ef00c8120de91e3cd9c16789bff2f9bfc48796c2717269"
)
FINISH_GATE_PATH = (
    ROOT / "trajectory_categorical_gated_clean_finish_gate.json"
)
FINISH_GATE_SHA256 = (
    "f76cc6b427bb875aa6a1b7882392f98ed9a1898602b5c5c0cfbf93d85b61aa3e"
)
TREATMENTS = (*gated.TREATMENTS, "parent_top20")


def source_sha256() -> str:
    return plain.sha256(Path(__file__))


def verify_prerequisites() -> None:
    finish.verify_selector_lineage()
    if (
        plain.sha256(Path(finish.__file__)) != FINISH_SOURCE_SHA256
        or plain.sha256(FINISH_GATE_PATH) != FINISH_GATE_SHA256
        or plain.sha256(D895_PATH) != D895_SHA256
        or plain.sha256(finish.D570_PATH) != finish.D570_SHA256
    ):
        raise RuntimeError("C898 prerequisite lineage changed")
    finisher_gate = json.loads(FINISH_GATE_PATH.read_text())
    d895 = json.loads(D895_PATH.read_text())
    if (
        finisher_gate.get("status") != "passed"
        or finisher_gate.get("source_sha256") != FINISH_SOURCE_SHA256
        or bool(finisher_gate.get("target_metric_computed"))
        or bool(finisher_gate.get("audit_fold_loaded"))
        or bool(finisher_gate.get("confirmation_regroupings_loaded"))
        or bool(finisher_gate.get("f4_loaded"))
        or d895.get("status") != "expanded"
        or d895.get("selected", {}).get("treatment") != "parent_top20"
        or float(d895.get("gain_after_d570", -np.inf)) < 0.125
        or bool(d895.get("audit_fold_loaded"))
        or bool(d895.get("confirmation_regroupings_loaded"))
        or bool(d895.get("f4_loaded"))
    ):
        raise RuntimeError("C898 prerequisite record changed")


def parent_top20_mask(fold: dict[str, object]) -> np.ndarray:
    parent = np.asarray(fold["d570"], dtype=np.float64)
    baseline = np.asarray(fold["baseline"], dtype=np.float64)
    starts = np.asarray(fold["row_starts"], dtype=np.int64)
    well_ids = np.asarray(fold["well_ids"]).astype(str)
    if (
        len(parent) != len(baseline)
        or len(well_ids) != len(starts) - 1
        or starts[0] != 0
        or starts[-1] != len(parent)
    ):
        raise RuntimeError("invalid parent-top20 layout")
    movement = parent - baseline
    scores = np.asarray(
        [
            np.sqrt(np.mean(np.square(movement[left:right])))
            for left, right in zip(starts[:-1], starts[1:])
        ],
        dtype=np.float64,
    )
    order = sorted(
        range(len(well_ids)),
        key=lambda index: (-float(scores[index]), well_ids[index]),
    )
    count = max(1, int(math.ceil(0.20 * len(well_ids))))
    selected = set(order[:count])
    mask = np.zeros(len(parent), dtype=bool)
    for index, (left, right) in enumerate(
        zip(starts[:-1], starts[1:])
    ):
        if index in selected:
            mask[left:right] = True
    return mask


def treatment_mask(
    fold: dict[str, object],
    snapshot: int,
    temperature: float,
    treatment: str,
) -> np.ndarray:
    if treatment == "parent_top20":
        return parent_top20_mask(fold)
    return gated.treatment_mask(
        fold, snapshot, temperature, treatment
    )


def blend(
    fold: dict[str, object],
    treatment: str,
    snapshot: int,
    temperature: float,
    share: float,
) -> np.ndarray:
    parent = np.asarray(fold["d570"], dtype=np.float64)
    candidate = np.asarray(
        fold["candidates"][plain.candidate_key(snapshot, temperature)],
        dtype=np.float64,
    )
    correction = candidate - parent
    return parent + share * correction * treatment_mask(
        fold, snapshot, temperature, treatment
    )


def coefficient_statistics(
    fold: dict[str, object],
    treatment: str,
    snapshot: int,
    temperature: float,
) -> dict[str, np.ndarray | int]:
    truth = np.asarray(fold["truth"], dtype=np.float64)
    parent = np.asarray(fold["d570"], dtype=np.float64)
    candidate = np.asarray(
        fold["candidates"][plain.candidate_key(snapshot, temperature)],
        dtype=np.float64,
    )
    mask = treatment_mask(
        fold, snapshot, temperature, treatment
    )
    error = parent - truth
    correction = (candidate - parent) * mask
    starts = np.asarray(fold["row_starts"], dtype=np.int64)
    n_wells = len(starts) - 1
    values = {
        name: np.empty(n_wells, dtype=np.float64)
        for name in (
            "count",
            "a",
            "b",
            "c",
            "toe_count",
            "toe_a",
            "toe_b",
            "toe_c",
            "endpoint_error",
            "endpoint_correction",
        )
    }
    for index, (left, right) in enumerate(
        zip(starts[:-1], starts[1:])
    ):
        local_error = error[left:right]
        local_correction = correction[left:right]
        width = max(len(local_error) // 4, 1)
        toe_error = local_error[-width:]
        toe_correction = local_correction[-width:]
        values["count"][index] = len(local_error)
        values["a"][index] = np.dot(local_error, local_error)
        values["b"][index] = np.dot(local_error, local_correction)
        values["c"][index] = np.dot(local_correction, local_correction)
        values["toe_count"][index] = width
        values["toe_a"][index] = np.dot(toe_error, toe_error)
        values["toe_b"][index] = np.dot(toe_error, toe_correction)
        values["toe_c"][index] = np.dot(
            toe_correction, toe_correction
        )
        values["endpoint_error"][index] = local_error[-1]
        values["endpoint_correction"][index] = local_correction[-1]
    return {**values, "corrected_rows": int(mask.sum())}


def add_ranks(records: list[dict[str, object]]) -> None:
    eligible = [record for record in records if record["eligible"]]
    ranks = np.column_stack(
        [
            rankdata(
                [record[name] for record in eligible],
                method="average",
            )
            for name in plain.METRICS
        ]
    )
    for record, values in zip(eligible, ranks):
        record["metric_ranks"] = {
            name: float(value)
            for name, value in zip(plain.METRICS, values)
        }
        record["mean_rank"] = float(np.mean(values))


def fast_records(
    folds: list[dict[str, object]], training_roles: list[int]
) -> list[dict[str, object]]:
    parent_scores = {
        role: plain.rmse(
            np.asarray(folds[role]["d570"]),
            np.asarray(folds[role]["truth"]),
        )
        for role in training_roles
    }
    records: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        for snapshot in plain.SNAPSHOTS:
            for temperature in plain.TEMPERATURES:
                statistics = {
                    role: coefficient_statistics(
                        folds[role],
                        treatment,
                        snapshot,
                        temperature,
                    )
                    for role in training_roles
                }
                for share in plain.SHARES:
                    role_scores = {
                        role: finish.metrics_from_statistics(
                            [statistics[role]], share
                        )["pooled_rmse"]
                        for role in training_roles
                    }
                    aggregate = finish.metrics_from_statistics(
                        [statistics[role] for role in training_roles],
                        share,
                    )
                    records.append(
                        {
                            "treatment": treatment,
                            "snapshot": snapshot,
                            "temperature": temperature,
                            "share": share,
                            "eligible": all(
                                role_scores[role]
                                <= parent_scores[role] + 1e-12
                                for role in training_roles
                            ),
                            "corrected_rows": sum(
                                int(statistics[role]["corrected_rows"])
                                for role in training_roles
                            ),
                            "role_rmse": {
                                str(role): value
                                for role, value in role_scores.items()
                            },
                            **aggregate,
                        }
                    )
    add_ranks(records)
    return records


def direct_records(
    folds: list[dict[str, object]], training_roles: list[int]
) -> list[dict[str, object]]:
    parent_scores = {
        role: plain.rmse(
            np.asarray(folds[role]["d570"]),
            np.asarray(folds[role]["truth"]),
        )
        for role in training_roles
    }
    records: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        for snapshot in plain.SNAPSHOTS:
            for temperature in plain.TEMPERATURES:
                for share in plain.SHARES:
                    predictions = []
                    truths = []
                    starts = [0]
                    role_scores = {}
                    corrected_rows = 0
                    for role in training_roles:
                        prediction = blend(
                            folds[role],
                            treatment,
                            snapshot,
                            temperature,
                            share,
                        )
                        truth = np.asarray(folds[role]["truth"])
                        local_starts = np.asarray(
                            folds[role]["row_starts"], dtype=np.int64
                        )
                        predictions.append(prediction)
                        truths.append(truth)
                        starts.extend(
                            (local_starts[1:] + starts[-1]).tolist()
                        )
                        role_scores[role] = plain.rmse(
                            prediction, truth
                        )
                        corrected_rows += int(
                            treatment_mask(
                                folds[role],
                                snapshot,
                                temperature,
                                treatment,
                            ).sum()
                        )
                    aggregate = plain.metrics(
                        np.concatenate(predictions),
                        np.concatenate(truths),
                        np.asarray(starts, dtype=np.int64),
                    )
                    records.append(
                        {
                            "treatment": treatment,
                            "snapshot": snapshot,
                            "temperature": temperature,
                            "share": share,
                            "eligible": all(
                                role_scores[role]
                                <= parent_scores[role] + 1e-12
                                for role in training_roles
                            ),
                            "corrected_rows": corrected_rows,
                            "role_rmse": {
                                str(role): value
                                for role, value in role_scores.items()
                            },
                            **aggregate,
                        }
                    )
    add_ranks(records)
    return records


def selected_record(
    records: list[dict[str, object]]
) -> dict[str, object]:
    return min(
        (record for record in records if record["eligible"]),
        key=lambda record: (
            record["mean_rank"],
            *(record[name] for name in plain.METRICS),
            record["share"],
            record["corrected_rows"],
            TREATMENTS.index(str(record["treatment"])),
            record["snapshot"],
            plain.TEMPERATURES.index(record["temperature"]),
        ),
    )


def fast_select(
    folds: list[dict[str, object]], held_role: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = fast_records(
        folds, [role for role in range(4) if role != held_role]
    )
    return selected_record(records), records


def synthetic_folds() -> list[dict[str, object]]:
    folds = []
    movement_scores = np.asarray([6.0, 5.0, 5.0, 3.0, 2.0])
    for role in range(4):
        fold = gated.synthetic_fold(role)
        starts = np.asarray(fold["row_starts"], dtype=np.int64)
        movement = np.concatenate(
            [
                np.full(
                    right - left,
                    movement_scores[index] + 0.01 * role,
                )
                for index, (left, right) in enumerate(
                    zip(starts[:-1], starts[1:])
                )
            ]
        )
        folds.append(
            {
                **fold,
                "baseline": np.asarray(fold["d570"]) - movement,
            }
        )
    return folds


def gate(output: Path) -> None:
    verify_prerequisites()
    folds = synthetic_folds()
    parent_masks = [
        parent_top20_mask(fold) for fold in folds
    ]
    parent_counts = [int(mask.sum()) for mask in parent_masks]
    parent_ids = [
        np.asarray(fold["well_ids"])[
            [
                bool(mask[left:right].any())
                for left, right in zip(
                    fold["row_starts"][:-1],
                    fold["row_starts"][1:],
                )
            ]
        ].astype(str).tolist()
        for fold, mask in zip(folds, parent_masks)
    ]
    original_parity = True
    for fold in folds:
        for treatment in gated.TREATMENTS:
            for snapshot in plain.SNAPSHOTS:
                for temperature in plain.TEMPERATURES:
                    original_parity &= np.array_equal(
                        treatment_mask(
                            fold, snapshot, temperature, treatment
                        ),
                        gated.treatment_mask(
                            fold, snapshot, temperature, treatment
                        ),
                    )
                    for share in plain.SHARES:
                        original_parity &= np.array_equal(
                            blend(
                                fold,
                                treatment,
                                snapshot,
                                temperature,
                                share,
                            ),
                            gated.blend(
                                fold,
                                treatment,
                                snapshot,
                                temperature,
                                share,
                            ),
                        )
    zero_exact = all(
        np.array_equal(
            blend(fold, "parent_top20", 12, 0.75, 0.0),
            np.asarray(fold["d570"], dtype=np.float64),
        )
        for fold in folds
    )
    maximum_difference = 0.0
    selection_parity = True
    held_truth_invariant = True
    selections = {}
    candidate_count = 0
    for held in range(4):
        training_roles = [
            role for role in range(4) if role != held
        ]
        direct = direct_records(folds, training_roles)
        optimized = fast_records(folds, training_roles)
        candidate_count = len(optimized)
        direct_by_key = {
            (
                record["treatment"],
                record["snapshot"],
                record["temperature"],
                record["share"],
            ): record
            for record in direct
        }
        for record in optimized:
            key = (
                record["treatment"],
                record["snapshot"],
                record["temperature"],
                record["share"],
            )
            reference = direct_by_key[key]
            if (
                bool(record["eligible"])
                != bool(reference["eligible"])
                or int(record["corrected_rows"])
                != int(reference["corrected_rows"])
            ):
                selection_parity = False
            for name in plain.METRICS:
                difference = abs(
                    float(record[name]) - float(reference[name])
                )
                maximum_difference = max(
                    maximum_difference, difference
                )
                if difference > 1e-7:
                    selection_parity = False
        direct_selected = selected_record(direct)
        fast_selected = selected_record(optimized)
        recipe_names = (
            "treatment",
            "snapshot",
            "temperature",
            "share",
        )
        direct_recipe = {
            name: direct_selected[name] for name in recipe_names
        }
        fast_recipe = {
            name: fast_selected[name] for name in recipe_names
        }
        selection_parity &= direct_recipe == fast_recipe
        selections[str(held)] = fast_recipe
        changed = [
            {
                **fold,
                "truth": (
                    np.asarray(fold["truth"]) + 1000.0
                    if role == held
                    else np.asarray(fold["truth"])
                ),
            }
            for role, fold in enumerate(folds)
        ]
        changed_selected, _ = fast_select(changed, held)
        held_truth_invariant &= all(
            changed_selected[name] == fast_selected[name]
            for name in recipe_names
        )
    expected_count = (
        len(TREATMENTS)
        * len(plain.SNAPSHOTS)
        * len(plain.TEMPERATURES)
        * len(plain.SHARES)
    )
    passed = (
        parent_counts == [4, 4, 4, 4]
        and parent_ids
        == [["00000000"], ["00000000"], ["00000000"], ["00000000"]]
        and original_parity
        and zero_exact
        and selection_parity
        and held_truth_invariant
        and candidate_count == expected_count
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "passed" if passed else "failed",
        "source_sha256": source_sha256(),
        "d895_sha256": D895_SHA256,
        "finisher_source_sha256": FINISH_SOURCE_SHA256,
        "finisher_gate_sha256": FINISH_GATE_SHA256,
        "treatments": list(TREATMENTS),
        "candidate_count_per_held_role": candidate_count,
        "parent_top20_selected_ids": parent_ids,
        "parent_top20_row_counts": parent_counts,
        "original_five_exact": bool(original_parity),
        "zero_fallback_exact": bool(zero_exact),
        "optimized_direct_selection_parity": bool(selection_parity),
        "maximum_metric_difference": maximum_difference,
        "held_truth_replacement_invariant": bool(
            held_truth_invariant
        ),
        "selections": selections,
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    plain.write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("C898 expanded-selector gate failed")


def verify_gate(path: Path) -> None:
    verify_prerequisites()
    record = json.loads(path.read_text())
    if (
        record.get("status") != "passed"
        or record.get("source_sha256") != source_sha256()
        or record.get("d895_sha256") != D895_SHA256
        or record.get("finisher_source_sha256")
        != FINISH_SOURCE_SHA256
        or record.get("finisher_gate_sha256") != FINISH_GATE_SHA256
        or record.get("treatments") != list(TREATMENTS)
        or int(record.get("candidate_count_per_held_role", -1))
        != (
            len(TREATMENTS)
            * len(plain.SNAPSHOTS)
            * len(plain.TEMPERATURES)
            * len(plain.SHARES)
        )
        or bool(record.get("target_metric_computed"))
        or bool(record.get("audit_fold_loaded"))
        or bool(record.get("confirmation_regroupings_loaded"))
        or bool(record.get("f4_loaded"))
    ):
        raise RuntimeError("invalid C898 expanded-selector gate")


def evaluate(
    gate_path: Path,
    manifest_path: Path,
    output_json: Path,
    output_npz: Path,
) -> None:
    verify_gate(gate_path)
    manifest = finish.load_manifest(manifest_path)
    with np.load(finish.D570_PATH, allow_pickle=False) as parent:
        folds = [
            finish.load_fold(fold, parent) for fold in range(4)
        ]
    selections = []
    predictions = []
    fold_gains = []
    for held in range(4):
        selected, records = fast_select(folds, held)
        prediction = blend(
            folds[held],
            str(selected["treatment"]),
            int(selected["snapshot"]),
            float(selected["temperature"]),
            float(selected["share"]),
        )
        truth = np.asarray(folds[held]["truth"])
        d570 = np.asarray(folds[held]["d570"])
        parent_rmse = plain.rmse(d570, truth)
        selected_rmse = plain.rmse(prediction, truth)
        gain = parent_rmse - selected_rmse
        fold_gains.append(gain)
        predictions.append(prediction)
        selections.append(
            {
                "held_fold": held,
                "selection_folds": [
                    role for role in range(4) if role != held
                ],
                "selected": {
                    name: selected[name]
                    for name in (
                        "treatment",
                        "snapshot",
                        "temperature",
                        "share",
                        "mean_rank",
                    )
                },
                "eligible_recipes": int(
                    sum(record["eligible"] for record in records)
                ),
                "held_d570_rmse": parent_rmse,
                "held_selected_rmse": selected_rmse,
                "held_gain_after_d570": gain,
            }
        )
    truth = np.concatenate(
        [np.asarray(fold["truth"]) for fold in folds]
    )
    baseline = np.concatenate(
        [np.asarray(fold["baseline"]) for fold in folds]
    )
    d570 = np.concatenate(
        [np.asarray(fold["d570"]) for fold in folds]
    )
    prediction = np.concatenate(predictions)
    baseline_rmse = plain.rmse(baseline, truth)
    d570_rmse = plain.rmse(d570, truth)
    selected_rmse = plain.rmse(prediction, truth)
    gain_after_d570 = d570_rmse - selected_rmse
    passed = (
        gain_after_d570 > 0.0
        and all(gain > 0.0 for gain in fold_gains)
    )
    finish.save_clean_archive(output_npz, folds, predictions)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": (
            "accepted_selection_clean_F0_F3"
            if passed
            else "rejected_selection_clean_F0_F3"
        ),
        "source_sha256": source_sha256(),
        "gate_sha256": plain.sha256(gate_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": plain.sha256(manifest_path),
        "manifest_status": manifest["status"],
        "d895_sha256": D895_SHA256,
        "d570_sha256": finish.D570_SHA256,
        "baseline_v20_rmse": baseline_rmse,
        "d570_rmse": d570_rmse,
        "selected_rmse": selected_rmse,
        "gain_after_d570": gain_after_d570,
        "gain_vs_v20": baseline_rmse - selected_rmse,
        "every_fold_positive_after_d570": bool(
            all(gain > 0.0 for gain in fold_gains)
        ),
        "fold_records": selections,
        "clean_archive_path": str(output_npz),
        "clean_archive_sha256": plain.sha256(output_npz),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    plain.write_json(output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--gate-only", action="store_true")
    modes.add_argument("--manifest-only", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gate-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-npz", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gate_only:
        if args.gate_path is None:
            raise ValueError("--gate-path is required for --gate-only")
        gate(args.gate_path)
    elif args.manifest_only:
        if args.gate_path is None or args.manifest is None:
            raise ValueError(
                "--gate-path and --manifest are required"
            )
        verify_gate(args.gate_path)
        finish.artifact_manifest(args.manifest)
    else:
        if (
            args.gate_path is None
            or args.manifest is None
            or args.output_json is None
            or args.output_npz is None
        ):
            raise ValueError(
                "--gate-path, --manifest, --output-json, and "
                "--output-npz are required"
            )
        evaluate(
            args.gate_path,
            args.manifest,
            args.output_json,
            args.output_npz,
        )


if __name__ == "__main__":
    main()
