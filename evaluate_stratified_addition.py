from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluate_gr_smoothing_tta import expanded_and_smoothed
from evaluate_surface_extrapolation import surface_extrapolation
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


WIDE_SUPERVISED_INDEX = 8
WIDE_SUPERVISED_WEIGHT = 0.325
FIXED_WEIGHT = 0.0537244898
WIDE_WEIGHT = 0.0716326531
SEED_FRACTION = 0.10
SMALL_FRACTION = 0.10


def load_all(data_root: Path, workers: int) -> dict[str, Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
    return dict(zip(well_ids, wells))


def load_candidate(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predictions = {}
    for fold in range(5):
        with np.load(root / f"fold{fold}.npz") as cached:
            well_ids = cached["well_ids"]
            starts = cached["row_starts"]
            prediction = cached["prediction"].astype(np.float64)
            truth = cached["truth"].astype(np.float64)
            for index, well_id in enumerate(well_ids):
                left, right = int(starts[index]), int(starts[index + 1])
                key = str(well_id)
                if key in predictions:
                    raise ValueError(f"duplicate candidate well {key}")
                predictions[key] = (prediction[left:right], truth[left:right])
    return predictions


def nested_score(
    name: str,
    fold_sse: np.ndarray,
    fold_rows: np.ndarray,
    weights: np.ndarray,
) -> None:
    nested_sse = 0.0
    print(f"{name}_leave_one_fold_out")
    for heldout in range(5):
        tuning_sse = np.sum(np.delete(fold_sse, heldout, axis=0), axis=0)
        selected = int(np.argmin(tuning_sse))
        nested_sse += fold_sse[heldout, selected]
        print(
            f"heldout_fold={heldout} selected_weight={weights[selected]:.3f} "
            f"heldout_rmse="
            f"{np.sqrt(fold_sse[heldout, selected] / fold_rows[heldout]):.6f}"
        )
    print(
        f"{name}_nested_rmse="
        f"{np.sqrt(nested_sse / np.sum(fold_rows)):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-root", type=Path, default=Path("cache/v6_oof"))
    parser.add_argument("--tta-root", type=Path, default=Path("cache/v6_tta"))
    parser.add_argument(
        "--diversity-root", type=Path, default=Path("cache/seed_ablation")
    )
    parser.add_argument(
        "--candidate-root", type=Path, default=Path("cache/stratified_ablation")
    )
    parser.add_argument("--second-candidate-root", type=Path)
    parser.add_argument(
        "--fold-file", type=Path, default=Path("stratified_folds.json")
    )
    parser.add_argument("--second-fold-file", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--preserve-diversity",
        action="store_true",
        help="keep seed/small at 10%% each and move weight only from original to stratified",
    )
    parser.add_argument("--audit-weight", type=float, default=0.40)
    parser.add_argument("--per-well-output", type=Path)
    parser.add_argument(
        "--reference-v8",
        action="store_true",
        help="use the deployed 40/40/10/10 policy as per-well baseline",
    )
    parser.add_argument(
        "--three-partition",
        action="store_true",
        help="test a third partition by moving weight equally from original and first repeat",
    )
    parser.add_argument(
        "--balanced-seed",
        action="store_true",
        help="move weight from the first balanced model to its independent seed",
    )
    parser.add_argument(
        "--family-total",
        action="store_true",
        help="vary the total wide-supervised weight at a locked family composition",
    )
    parser.add_argument(
        "--composition-at-total",
        type=float,
        help="vary the balanced-partition fraction at this fixed family weight",
    )
    parser.add_argument("--locked-stratified-fraction", type=float, default=0.45)
    args = parser.parse_args()

    weights = np.asarray(args.weights, dtype=np.float64)
    audit_matches = np.flatnonzero(np.isclose(weights, args.audit_weight))
    if args.per_well_output is not None and len(audit_matches) != 1:
        raise ValueError("audit weight must occur exactly once in --weights")
    audit_index = int(audit_matches[0]) if len(audit_matches) == 1 else -1
    wells = load_all(args.data_root, args.workers)
    stratified_assignment = {
        key: int(value)
        for key, value in json.loads(args.fold_file.read_text())["assignments"].items()
    }
    candidate = load_candidate(args.candidate_root)
    if set(candidate) != set(wells):
        raise ValueError("candidate OOF cache does not cover every training well")
    second_candidate = None
    second_assignment = None
    if sum(
        (
            args.three_partition,
            args.balanced_seed,
            args.family_total,
            args.composition_at_total is not None,
        )
    ) > 1:
        raise ValueError("choose only one candidate audit mode")
    if args.second_fold_file is not None:
        second_assignment = {
            key: int(value)
            for key, value in json.loads(args.second_fold_file.read_text())[
                "assignments"
            ].items()
        }
    if args.three_partition or args.balanced_seed:
        if args.second_candidate_root is None or second_assignment is None:
            raise ValueError("three-partition audit requires second candidate/fold paths")
        second_candidate = load_candidate(args.second_candidate_root)
        if set(second_candidate) != set(wells):
            raise ValueError("second candidate cache does not cover every training well")

    original_sse = np.zeros((5, len(weights)), dtype=np.float64)
    original_rows = np.zeros(5, dtype=np.int64)
    stratified_sse = np.zeros((5, len(weights)), dtype=np.float64)
    stratified_rows = np.zeros(5, dtype=np.int64)
    second_sse = np.zeros((5, len(weights)), dtype=np.float64)
    second_rows = np.zeros(5, dtype=np.int64)
    per_well_rows: list[dict[str, str | int | float]] = []
    for original_fold in range(5):
        with np.load(args.base_root / f"fold{original_fold}.npz") as base_cache:
            well_ids = base_cache["well_ids"]
            starts = base_cache["row_starts"]
            components = base_cache["components"].astype(np.float64)
            raw = base_cache["coarse"].astype(np.float64)
            truth = base_cache["truth"].astype(np.float64)
        with np.load(args.tta_root / f"fold{original_fold}_gr25.npz") as tta_cache:
            tta_components = tta_cache["components"].astype(np.float64)
        with np.load(
            args.diversity_root / f"seed3_fold{original_fold}.npz"
        ) as cached:
            if not np.array_equal(well_ids, cached["well_ids"]):
                raise ValueError(f"fold {original_fold}: seed well order differs")
            seed_prediction = cached["prediction"].astype(np.float64)
        with np.load(
            args.diversity_root / f"base6_fold{original_fold}.npz"
        ) as cached:
            if not np.array_equal(well_ids, cached["well_ids"]):
                raise ValueError(f"fold {original_fold}: small well order differs")
            small_prediction = cached["prediction"].astype(np.float64)

        raw += FIXED_WEIGHT * (tta_components[0] - components[0])
        raw += WIDE_WEIGHT * (tta_components[2] - components[2])
        for index, well_id_value in enumerate(well_ids):
            well_id = str(well_id_value)
            well = wells[well_id]
            left, right = int(starts[index]), int(starts[index + 1])
            candidate_prediction, candidate_truth = candidate[well_id]
            if not np.allclose(candidate_truth, truth[left:right], atol=1e-5):
                raise ValueError(f"{well_id}: candidate truth differs")

            base_pre = expanded_and_smoothed(well, raw[left:right], 0.05)
            original_pre = expanded_and_smoothed(
                well,
                components[WIDE_SUPERVISED_INDEX, left:right],
                0.05,
            )
            seed_pre = expanded_and_smoothed(
                well, seed_prediction[left:right], 0.05
            )
            small_pre = expanded_and_smoothed(
                well, small_prediction[left:right], 0.05
            )
            candidate_pre = expanded_and_smoothed(
                well, candidate_prediction, 0.05
            )
            if args.three_partition or args.balanced_seed:
                assert second_candidate is not None
                second_prediction, second_truth = second_candidate[well_id]
                if not np.allclose(second_truth, truth[left:right], atol=1e-5):
                    raise ValueError(f"{well_id}: second candidate truth differs")
                second_pre = expanded_and_smoothed(
                    well, second_prediction, 0.05
                )
                current_family_pre = (
                    0.40 * original_pre
                    + SEED_FRACTION * seed_pre
                    + SMALL_FRACTION * small_pre
                    + 0.40 * candidate_pre
                )
            elif args.family_total:
                original_fraction = (
                    1.0
                    - SEED_FRACTION
                    - SMALL_FRACTION
                    - args.locked_stratified_fraction
                )
                if original_fraction < 0.0:
                    raise ValueError("locked family fractions exceed one")
                current_family_pre = (
                    original_fraction * original_pre
                    + SEED_FRACTION * seed_pre
                    + SMALL_FRACTION * small_pre
                    + args.locked_stratified_fraction * candidate_pre
                )
            elif args.composition_at_total is not None:
                current_family_pre = original_pre
            else:
                current_family_pre = (
                    (1.0 - SEED_FRACTION - SMALL_FRACTION) * original_pre
                    + SEED_FRACTION * seed_pre
                    + SMALL_FRACTION * small_pre
                )
            current_pre = base_pre + WIDE_SUPERVISED_WEIGHT * (
                current_family_pre - original_pre
            )
            new_fold = stratified_assignment[well_id]
            second_fold = second_assignment[well_id] if second_assignment else -1
            rows = right - left
            original_rows[original_fold] += rows
            stratified_rows[new_fold] += rows
            if second_assignment is not None:
                second_rows[second_fold] += rows
            well_sse = np.zeros(len(weights), dtype=np.float64)
            for weight_index, weight in enumerate(weights):
                if args.three_partition:
                    prediction = current_pre + WIDE_SUPERVISED_WEIGHT * weight * (
                        second_pre - 0.5 * (original_pre + candidate_pre)
                    )
                elif args.balanced_seed:
                    prediction = current_pre + WIDE_SUPERVISED_WEIGHT * weight * (
                        second_pre - candidate_pre
                    )
                elif args.family_total:
                    other_pre = (
                        base_pre - WIDE_SUPERVISED_WEIGHT * original_pre
                    ) / (1.0 - WIDE_SUPERVISED_WEIGHT)
                    prediction = (1.0 - weight) * other_pre + weight * current_family_pre
                elif args.composition_at_total is not None:
                    if weight > 1.0 - SEED_FRACTION - SMALL_FRACTION:
                        raise ValueError("weight leaves a negative original-family share")
                    other_pre = (
                        base_pre - WIDE_SUPERVISED_WEIGHT * original_pre
                    ) / (1.0 - WIDE_SUPERVISED_WEIGHT)
                    family_pre = (
                        (1.0 - SEED_FRACTION - SMALL_FRACTION - weight)
                        * original_pre
                        + SEED_FRACTION * seed_pre
                        + SMALL_FRACTION * small_pre
                        + weight * candidate_pre
                    )
                    prediction = (
                        (1.0 - args.composition_at_total) * other_pre
                        + args.composition_at_total * family_pre
                    )
                elif args.preserve_diversity:
                    if weight > 1.0 - SEED_FRACTION - SMALL_FRACTION:
                        raise ValueError("weight leaves a negative original-family share")
                    prediction = current_pre + WIDE_SUPERVISED_WEIGHT * weight * (
                        candidate_pre - original_pre
                    )
                else:
                    prediction = current_pre + WIDE_SUPERVISED_WEIGHT * weight * (
                        candidate_pre - current_family_pre
                    )
                prediction += 0.03 * surface_extrapolation(
                    well,
                    prediction,
                    visible_rows=1000,
                    correction_limit=24.0,
                )
                error = prediction - truth[left:right]
                sse = float(error @ error)
                well_sse[weight_index] = sse
                original_sse[original_fold, weight_index] += sse
                stratified_sse[new_fold, weight_index] += sse
                if second_assignment is not None:
                    second_sse[second_fold, weight_index] += sse
            if args.per_well_output is not None:
                if args.reference_v8:
                    v8_family_pre = (
                        0.40 * original_pre
                        + SEED_FRACTION * seed_pre
                        + SMALL_FRACTION * small_pre
                        + 0.40 * candidate_pre
                    )
                    baseline_prediction = base_pre + WIDE_SUPERVISED_WEIGHT * (
                        v8_family_pre - original_pre
                    )
                    baseline_prediction += 0.03 * surface_extrapolation(
                        well,
                        baseline_prediction,
                        visible_rows=1000,
                        correction_limit=24.0,
                    )
                    baseline_error = baseline_prediction - truth[left:right]
                    baseline_sse = float(baseline_error @ baseline_error)
                else:
                    baseline_matches = np.flatnonzero(np.isclose(weights, 0.0))
                    if len(baseline_matches) != 1:
                        raise ValueError(
                            "per-well audit requires weight 0 exactly once"
                        )
                    baseline_sse = well_sse[int(baseline_matches[0])]
                per_well_rows.append(
                    {
                        "well_id": well_id,
                        "original_fold": original_fold,
                        "stratified_fold": new_fold,
                        "second_stratified_fold": second_fold,
                        "rows": rows,
                        "baseline_sse": baseline_sse,
                        "candidate_sse": well_sse[audit_index],
                    }
                )

    total_rows = int(np.sum(original_rows))
    if total_rows != int(np.sum(stratified_rows)):
        raise RuntimeError("fold regrouping changed row count")
    if second_assignment is not None and total_rows != int(np.sum(second_rows)):
        raise RuntimeError("second fold regrouping changed row count")
    pooled = np.sqrt(np.sum(original_sse, axis=0) / total_rows)
    for weight_index in np.argsort(pooled):
        print(
            f"fixed weight={weights[weight_index]:.3f} "
            f"pooled={pooled[weight_index]:.6f} original_folds="
            + ",".join(
                f"{np.sqrt(original_sse[fold, weight_index] / original_rows[fold]):.6f}"
                for fold in range(5)
            )
            + " stratified_folds="
            + ",".join(
                f"{np.sqrt(stratified_sse[fold, weight_index] / stratified_rows[fold]):.6f}"
                for fold in range(5)
            )
        )
    nested_score("original", original_sse, original_rows, weights)
    nested_score("stratified", stratified_sse, stratified_rows, weights)
    if second_assignment is not None:
        nested_score("second_stratified", second_sse, second_rows, weights)
    if args.per_well_output is not None:
        args.per_well_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_well_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_well_rows[0]))
            writer.writeheader()
            writer.writerows(per_well_rows)
        print(f"wrote={args.per_well_output} wells={len(per_well_rows)}")


if __name__ == "__main__":
    main()
