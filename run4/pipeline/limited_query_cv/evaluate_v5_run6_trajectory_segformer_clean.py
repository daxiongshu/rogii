from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "trajectory_conditioned_pretrained_categorical_segmentation"
SOURCE_SHA256 = (
    "ff89e1ccee7ac7eb01bcdd8a3ca9f9a3b45c48f445f7577eac17806b6627fac4"
)
GATE_SHA256 = (
    "ca613128f76a271aca7a203cfd8cd2e45dc3a381fa74a5892369f3b4b588add0"
)
SNAPSHOTS = (4, 8, 12, 16, 24)
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SHARES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
METRICS = (
    "pooled_rmse",
    "mean_well_rmse",
    "toe_quartile_rmse",
    "endpoint_rmse",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    starts: np.ndarray,
) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    well_scores = []
    toe_parts = []
    endpoints = []
    for left, right in zip(starts[:-1], starts[1:]):
        local = error[left:right]
        well_scores.append(float(np.sqrt(np.mean(np.square(local)))))
        width = max(len(local) // 4, 1)
        toe_parts.append(local[-width:])
        endpoints.append(float(local[-1]))
    toe = np.concatenate(toe_parts)
    return {
        "pooled_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_well_rmse": float(np.mean(well_scores)),
        "toe_quartile_rmse": float(np.sqrt(np.mean(np.square(toe)))),
        "endpoint_rmse": float(
            np.sqrt(np.mean(np.square(endpoints)))
        ),
    }


def candidate_key(snapshot: int, temperature: float) -> tuple[int, float]:
    return snapshot, temperature


def validate_fold(fold: dict[str, object], expected_fold: int) -> None:
    if (
        int(fold["fold"]) != expected_fold
        or str(fold["source_sha256"]) != SOURCE_SHA256
        or str(fold["gate_sha256"]) != GATE_SHA256
        or bool(fold["audit_fold_loaded"])
        or bool(fold["confirmation_regroupings_loaded"])
        or bool(fold["f4_loaded"])
    ):
        raise RuntimeError(f"invalid A831 fold provenance F{expected_fold}")
    truth = np.asarray(fold["truth"])
    parent = np.asarray(fold["d570"])
    starts = np.asarray(fold["row_starts"])
    if (
        truth.shape != parent.shape
        or starts[0] != 0
        or starts[-1] != len(truth)
        or not np.isfinite(truth).all()
        or not np.isfinite(parent).all()
    ):
        raise RuntimeError(f"invalid A831 fold layout F{expected_fold}")
    for key in (
        candidate_key(snapshot, temperature)
        for snapshot in SNAPSHOTS
        for temperature in TEMPERATURES
    ):
        candidate = np.asarray(fold["candidates"][key])
        if candidate.shape != truth.shape or not np.isfinite(candidate).all():
            raise RuntimeError(f"invalid A831 candidate {key}")


def blend(
    fold: dict[str, object],
    snapshot: int,
    temperature: float,
    share: float,
) -> np.ndarray:
    parent = np.asarray(fold["d570"], dtype=np.float64)
    candidate = np.asarray(
        fold["candidates"][candidate_key(snapshot, temperature)],
        dtype=np.float64,
    )
    return parent + share * (candidate - parent)


def enumerate_records(
    folds: list[dict[str, object]],
    training_roles: list[int],
) -> list[dict[str, object]]:
    records = []
    parent_scores = {
        role: rmse(
            np.asarray(folds[role]["d570"]),
            np.asarray(folds[role]["truth"]),
        )
        for role in training_roles
    }
    for snapshot in SNAPSHOTS:
        for temperature in TEMPERATURES:
            for share in SHARES:
                role_scores = {}
                predictions = []
                truths = []
                local_starts = [0]
                for role in training_roles:
                    prediction = blend(
                        folds[role], snapshot, temperature, share
                    )
                    truth = np.asarray(folds[role]["truth"])
                    starts = np.asarray(folds[role]["row_starts"])
                    role_scores[role] = rmse(prediction, truth)
                    predictions.append(prediction)
                    truths.append(truth)
                    local_starts.extend(
                        (
                            starts[1:]
                            - starts[0]
                            + local_starts[-1]
                        ).tolist()
                    )
                eligible = all(
                    role_scores[role] <= parent_scores[role] + 1e-12
                    for role in training_roles
                )
                aggregate = metrics(
                    np.concatenate(predictions),
                    np.concatenate(truths),
                    np.asarray(local_starts, dtype=np.int64),
                )
                records.append(
                    {
                        "snapshot": snapshot,
                        "temperature": temperature,
                        "share": share,
                        "eligible": eligible,
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
            for name in METRICS
        ]
    )
    for record, values in zip(eligible_records, ranks):
        record["metric_ranks"] = {
            name: float(value)
            for name, value in zip(METRICS, values)
        }
        record["mean_rank"] = float(np.mean(values))
    return records


def select_recipe(
    folds: list[dict[str, object]], held_role: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    training_roles = [role for role in range(4) if role != held_role]
    records = enumerate_records(folds, training_roles)
    eligible = [record for record in records if record["eligible"]]
    selected = min(
        eligible,
        key=lambda record: (
            record["mean_rank"],
            *(record[name] for name in METRICS),
            record["share"],
            record["snapshot"],
            TEMPERATURES.index(record["temperature"]),
        ),
    )
    return selected, records


def synthetic_fold(
    role: int,
    planted: tuple[int, float, float] | None,
    distractor_role: int | None = None,
) -> dict[str, object]:
    truth = np.asarray(
        [1.0 + 0.1 * role, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
        dtype=np.float64,
    )
    d570 = np.zeros_like(truth)
    candidates = {}
    for snapshot in SNAPSHOTS:
        for temperature in TEMPERATURES:
            values = np.full_like(truth, 20.0)
            if planted and (snapshot, temperature) == planted[:2]:
                values = truth / planted[2]
            if (snapshot, temperature) == (16, 1.0):
                values = (
                    truth / 0.20
                    if role != distractor_role
                    else -truth / 0.20
                )
            candidates[candidate_key(snapshot, temperature)] = values
    return {
        "fold": role,
        "source_sha256": SOURCE_SHA256,
        "gate_sha256": GATE_SHA256,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
        "truth": truth,
        "d570": d570,
        "row_starts": np.asarray([0, 4, 8], dtype=np.int64),
        "candidates": candidates,
    }


def gate(output: Path) -> None:
    planted = (12, 0.75, 0.10)
    folds = [
        synthetic_fold(role, planted, distractor_role=2)
        for role in range(4)
    ]
    for role, fold in enumerate(folds):
        validate_fold(fold, role)
    selections = {}
    invariant = True
    distractor_rejected = True
    for held in range(4):
        selected, records = select_recipe(folds, held)
        selections[str(held)] = {
            key: selected[key]
            for key in ("snapshot", "temperature", "share")
        }
        distractor = next(
            record
            for record in records
            if (
                record["snapshot"],
                record["temperature"],
                record["share"],
            )
            == (16, 1.0, 0.20)
        )
        if held != 2:
            distractor_rejected &= not distractor["eligible"]
        changed = [
            {
                **fold,
                "truth": (
                    np.asarray(fold["truth"]) * 1000.0
                    if role == held
                    else np.asarray(fold["truth"])
                ),
            }
            for role, fold in enumerate(folds)
        ]
        changed_selected, _ = select_recipe(changed, held)
        invariant &= all(
            changed_selected[key] == selected[key]
            for key in ("snapshot", "temperature", "share")
        )
    zero_folds = [
        synthetic_fold(role, None, distractor_role=role)
        for role in range(4)
    ]
    zero_selected, zero_records = select_recipe(zero_folds, 0)
    wrong_source_rejected = False
    wrong = dict(folds[0])
    wrong["source_sha256"] = "0" * 64
    try:
        validate_fold(wrong, 0)
    except RuntimeError:
        wrong_source_rejected = True
    passed = (
        all(
            tuple(selection.values()) == planted
            for selection in selections.values()
        )
        and distractor_rejected
        and invariant
        and zero_selected["share"] == 0.0
        and len(zero_records)
        == len(SNAPSHOTS) * len(TEMPERATURES) * len(SHARES)
        and wrong_source_rejected
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "passed" if passed else "failed",
        "source_sha256": sha256(Path(__file__)),
        "model_source_sha256": SOURCE_SHA256,
        "model_gate_sha256": GATE_SHA256,
        "candidate_count": len(zero_records),
        "planted_recipe": {
            "snapshot": planted[0],
            "temperature": planted[1],
            "share": planted[2],
        },
        "selections": selections,
        "distractor_rejected": distractor_rejected,
        "held_truth_replacement_invariant": invariant,
        "zero_fallback_exact": zero_selected["share"] == 0.0,
        "wrong_source_rejected": wrong_source_rejected,
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("A831 selector gate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gate_only:
        raise RuntimeError("only the A831 metric-free gate is enabled yet")
    gate(args.output)


if __name__ == "__main__":
    main()
