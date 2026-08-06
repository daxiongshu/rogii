from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from limited_query_cv import (
    evaluate_v5_run6_trajectory_segformer_clean as base,
)


RUN_ID = base.RUN_ID
FAMILY = base.FAMILY
TREATMENTS = (
    "all",
    "well_qrms_top60",
    "well_qrms_top40",
    "well_qrms_top20",
    "row_abs12",
)
TOP_FRACTIONS = {
    "well_qrms_top60": 0.60,
    "well_qrms_top40": 0.40,
    "well_qrms_top20": 0.20,
}


def treatment_mask(
    fold: dict[str, object],
    snapshot: int,
    temperature: float,
    treatment: str,
) -> np.ndarray:
    parent = np.asarray(fold["d570"], dtype=np.float64)
    candidate = np.asarray(
        fold["candidates"][base.candidate_key(snapshot, temperature)],
        dtype=np.float64,
    )
    correction = candidate - parent
    if treatment == "all":
        return np.ones(len(parent), dtype=bool)
    if treatment == "row_abs12":
        return np.abs(correction) >= 12.0
    fraction = TOP_FRACTIONS[treatment]
    starts = np.asarray(fold["row_starts"])
    well_ids = np.asarray(fold["well_ids"]).astype(str)
    rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(correction[left:right])))
            for left, right in zip(starts[:-1], starts[1:])
        ]
    )
    order = sorted(
        range(len(well_ids)),
        key=lambda index: (-float(rms[index]), well_ids[index]),
    )
    count = max(1, int(math.ceil(fraction * len(well_ids))))
    selected = set(order[:count])
    mask = np.zeros(len(parent), dtype=bool)
    for index, (left, right) in enumerate(
        zip(starts[:-1], starts[1:])
    ):
        if index in selected:
            mask[left:right] = True
    return mask


def blend(
    fold: dict[str, object],
    treatment: str,
    snapshot: int,
    temperature: float,
    share: float,
) -> np.ndarray:
    parent = np.asarray(fold["d570"], dtype=np.float64)
    candidate = np.asarray(
        fold["candidates"][base.candidate_key(snapshot, temperature)],
        dtype=np.float64,
    )
    correction = candidate - parent
    mask = treatment_mask(
        fold, snapshot, temperature, treatment
    )
    return parent + share * correction * mask


def enumerate_records(
    folds: list[dict[str, object]],
    training_roles: list[int],
) -> list[dict[str, object]]:
    records = []
    parent_scores = {
        role: base.rmse(
            np.asarray(folds[role]["d570"]),
            np.asarray(folds[role]["truth"]),
        )
        for role in training_roles
    }
    for treatment in TREATMENTS:
        for snapshot in base.SNAPSHOTS:
            for temperature in base.TEMPERATURES:
                for share in base.SHARES:
                    role_scores = {}
                    predictions = []
                    truths = []
                    starts = [0]
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
                            folds[role]["row_starts"]
                        )
                        role_scores[role] = base.rmse(
                            prediction, truth
                        )
                        predictions.append(prediction)
                        truths.append(truth)
                        starts.extend(
                            (
                                local_starts[1:]
                                + starts[-1]
                            ).tolist()
                        )
                        corrected_rows += int(
                            treatment_mask(
                                folds[role],
                                snapshot,
                                temperature,
                                treatment,
                            ).sum()
                        )
                    eligible = all(
                        role_scores[role]
                        <= parent_scores[role] + 1e-12
                        for role in training_roles
                    )
                    aggregate = base.metrics(
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
                            "eligible": eligible,
                            "corrected_rows": corrected_rows,
                            "role_rmse": {
                                str(key): value
                                for key, value in role_scores.items()
                            },
                            **aggregate,
                        }
                    )
    eligible_records = [record for record in records if record["eligible"]]
    ranks = np.column_stack(
        [
            rankdata(
                [record[name] for record in eligible_records],
                method="average",
            )
            for name in base.METRICS
        ]
    )
    for record, values in zip(eligible_records, ranks):
        record["metric_ranks"] = {
            name: float(value)
            for name, value in zip(base.METRICS, values)
        }
        record["mean_rank"] = float(np.mean(values))
    return records


def select_recipe(
    folds: list[dict[str, object]], held_role: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = enumerate_records(
        folds, [role for role in range(4) if role != held_role]
    )
    selected = min(
        (record for record in records if record["eligible"]),
        key=lambda record: (
            record["mean_rank"],
            *(record[name] for name in base.METRICS),
            record["share"],
            record["corrected_rows"],
            TREATMENTS.index(record["treatment"]),
            record["snapshot"],
            base.TEMPERATURES.index(record["temperature"]),
        ),
    )
    return selected, records


def synthetic_fold(role: int, distractor_role: int = 2):
    well_ids = np.asarray(
        [f"{index:08d}" for index in range(5)]
    )
    starts = np.arange(0, 21, 4, dtype=np.int64)
    d570 = np.zeros(20, dtype=np.float64)
    truth = np.zeros(20, dtype=np.float64)
    truth[:4] = 1.0 + 0.1 * role
    candidates = {}
    for snapshot in base.SNAPSHOTS:
        for temperature in base.TEMPERATURES:
            correction = np.ones(20, dtype=np.float64)
            if (snapshot, temperature) == (12, 0.75):
                correction[:4] = truth[:4] / 0.10
            if (snapshot, temperature) == (16, 1.0):
                correction[:4] = (
                    truth[:4] / 0.20
                    if role != distractor_role
                    else -truth[:4] / 0.20
                )
            candidates[base.candidate_key(snapshot, temperature)] = (
                d570 + correction
            )
    return {
        "fold": role,
        "source_sha256": base.SOURCE_SHA256,
        "gate_sha256": base.GATE_SHA256,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
        "well_ids": well_ids,
        "row_starts": starts,
        "truth": truth,
        "d570": d570,
        "candidates": candidates,
    }


def validate_fold(fold: dict[str, object], role: int) -> None:
    base.validate_fold(fold, role)
    if len(np.asarray(fold["well_ids"])) != len(
        np.asarray(fold["row_starts"])
    ) - 1:
        raise RuntimeError("invalid gated well layout")


def gate(output: Path) -> None:
    folds = [synthetic_fold(role) for role in range(4)]
    for role, fold in enumerate(folds):
        validate_fold(fold, role)
    mask_counts = {
        treatment: int(
            treatment_mask(folds[0], 12, 0.75, treatment).sum()
        )
        for treatment in TREATMENTS
    }
    expected_counts = {
        "all": 20,
        "well_qrms_top60": 12,
        "well_qrms_top40": 8,
        "well_qrms_top20": 4,
        "row_abs12": 0,
    }
    selections = {}
    invariant = True
    distractor_rejected = True
    for held in range(4):
        selected, records = select_recipe(folds, held)
        selections[str(held)] = {
            key: selected[key]
            for key in (
                "treatment",
                "snapshot",
                "temperature",
                "share",
            )
        }
        distractor = next(
            record
            for record in records
            if (
                record["treatment"],
                record["snapshot"],
                record["temperature"],
                record["share"],
            )
            == ("well_qrms_top20", 16, 1.0, 0.20)
        )
        if held != 2:
            distractor_rejected &= not distractor["eligible"]
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
        changed_selected, _ = select_recipe(changed, held)
        invariant &= all(
            changed_selected[key] == selected[key]
            for key in (
                "treatment",
                "snapshot",
                "temperature",
                "share",
            )
        )
    plain_exact = np.array_equal(
        blend(folds[0], "all", 12, 0.75, 0.10),
        base.blend(folds[0], 12, 0.75, 0.10),
    )
    zero_exact = np.array_equal(
        blend(folds[0], "well_qrms_top20", 12, 0.75, 0.0),
        np.asarray(folds[0]["d570"], dtype=np.float64),
    )
    records = enumerate_records(folds, [0, 1, 2])
    expected_recipe = {
        "treatment": "well_qrms_top20",
        "snapshot": 12,
        "temperature": 0.75,
        "share": 0.10,
    }
    passed = (
        mask_counts == expected_counts
        and all(
            selection == expected_recipe
            for selection in selections.values()
        )
        and distractor_rejected
        and invariant
        and plain_exact
        and zero_exact
        and len(records)
        == len(TREATMENTS)
        * len(base.SNAPSHOTS)
        * len(base.TEMPERATURES)
        * len(base.SHARES)
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "passed" if passed else "failed",
        "source_sha256": base.sha256(Path(__file__)),
        "base_selector_source_sha256": base.sha256(Path(base.__file__)),
        "model_source_sha256": base.SOURCE_SHA256,
        "model_gate_sha256": base.GATE_SHA256,
        "treatments": list(TREATMENTS),
        "candidate_count": len(records),
        "mask_counts": mask_counts,
        "selections": selections,
        "distractor_rejected": distractor_rejected,
        "held_truth_replacement_invariant": invariant,
        "plain_treatment_exact": plain_exact,
        "zero_fallback_exact": zero_exact,
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    base.write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("A832 gated selector gate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gate_only:
        raise RuntimeError("only A832 metric-free gate is enabled yet")
    gate(args.output)


if __name__ == "__main__":
    main()
