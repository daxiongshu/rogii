from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from evaluate_gr_smoothing_tta import expanded_and_smoothed
from evaluate_stratified_addition import load_candidate
from evaluate_surface_extrapolation import surface_extrapolation
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids


WIDE_SUPERVISED_INDEX = 8
OLD_WIDE_SUPERVISED_WEIGHT = 0.325
FIXED_WEIGHT = 0.0537244898
WIDE_WEIGHT = 0.0716326531
FINAL_WIDE_SUPERVISED_WEIGHT = 0.50
ORIGINAL_FRACTION = 0.35
SEED_FRACTION = 0.10
SMALL_FRACTION = 0.10
STRATIFIED_FRACTION = 0.45
CURRENT_TAIL_EXPANSION = 0.06
CURRENT_SURFACE_WEIGHT = 0.03


def load_all(data_root: Path, workers: int) -> dict[str, Well]:
    well_ids = training_well_ids(data_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = pool.map(lambda well_id: load_well(well_id, data_root), well_ids)
    return dict(zip(well_ids, wells))


def load_assignment(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in json.loads(path.read_text())["assignments"].items()
    }


def nested_score(
    name: str,
    fold_sse: np.ndarray,
    fold_rows: np.ndarray,
    candidates: list[tuple[float, float]],
    allowed: np.ndarray | None = None,
    adaptive: bool = False,
    internal: bool = False,
    confidence: bool = False,
    confidence_smooth: bool = False,
    consensus_bonus: bool = False,
    disagreement_bonus: bool = False,
    specialist_tta: bool = False,
) -> float:
    if allowed is None:
        allowed = np.arange(len(candidates))
    nested_sse = 0.0
    print(f"{name}_leave_one_fold_out")
    for heldout in range(5):
        tuning_sse = np.sum(np.delete(fold_sse, heldout, axis=0), axis=0)
        selected = int(allowed[np.argmin(tuning_sse[allowed])])
        first, second = candidates[selected]
        if specialist_tta:
            label = f"specialist_gr25_fraction={first:.4f}"
        elif disagreement_bonus:
            label = f"extra={first:.4f} threshold={second:.3f}"
        elif consensus_bonus:
            label = f"consensus_bonus={first:.4f}"
        elif confidence:
            label = (
                "confidence_margin=none"
                if first < -999.0
                else (
                    f"confidence_width={first:.2f}"
                    if confidence_smooth
                    else f"confidence_margin={first:.2f}"
                )
            )
        elif internal:
            label = f"internal_stratified_bonus={first:.4f}"
        elif adaptive:
            label = f"bonus={first:.4f} threshold={second:.1f}"
        else:
            label = f"expansion={first:.4f} surface={second:.4f}"
        nested_sse += fold_sse[heldout, selected]
        print(
            f"heldout_fold={heldout} {label} heldout_rmse="
            f"{np.sqrt(fold_sse[heldout, selected] / fold_rows[heldout]):.6f}"
        )
    score = float(np.sqrt(nested_sse / np.sum(fold_rows)))
    print(f"{name}_nested_rmse={score:.6f}")
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-root", type=Path, default=Path("cache/v6_oof"))
    parser.add_argument("--tta-root", type=Path, default=Path("cache/v6_tta"))
    parser.add_argument(
        "--diversity-root", type=Path, default=Path("cache/seed_ablation")
    )
    parser.add_argument(
        "--stratified-root", type=Path, default=Path("cache/stratified_ablation")
    )
    parser.add_argument(
        "--stratified-fold-file", type=Path, default=Path("stratified_folds.json")
    )
    parser.add_argument(
        "--audit-fold-file", type=Path, default=Path("stratified_folds_b.json")
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--expansions",
        type=float,
        nargs="+",
        default=(0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
    )
    parser.add_argument(
        "--surface-weights",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--audit-expansion", type=float)
    parser.add_argument("--audit-surface", type=float)
    parser.add_argument(
        "--adaptive-family-weight",
        action="store_true",
        help="increase strongest-family weight smoothly on predicted large excursions",
    )
    parser.add_argument(
        "--adaptive-internal-stratified",
        action="store_true",
        help="move gated strongest-family weight from original to balanced models",
    )
    parser.add_argument(
        "--adaptive-confidence",
        action="store_true",
        help="require strongest family to predict a larger excursion before gating",
    )
    parser.add_argument(
        "--current-adaptive-corrections",
        action="store_true",
        help="recalibrate expansion/surface corrections on the complete v12 gate",
    )
    parser.add_argument(
        "--adaptive-consensus-bonus",
        action="store_true",
        help="tune strongest-family bonus while retaining the v12 consensus gate",
    )
    parser.add_argument(
        "--adaptive-disagreement-bonus",
        action="store_true",
        help="increase the v13 bonus when normalized family disagreement is high",
    )
    parser.add_argument(
        "--disagreement-extras",
        type=float,
        nargs="+",
        default=(0.025, 0.05, 0.075, 0.10, 0.15, 0.20),
    )
    parser.add_argument(
        "--disagreement-thresholds",
        type=float,
        nargs="+",
        default=(0.10, 0.15, 0.20, 0.25, 0.30),
    )
    parser.add_argument("--disagreement-width", type=float, default=0.20)
    parser.add_argument(
        "--specialist-gr25-tta",
        action="store_true",
        help="blend GR-25 inference into the original strongest-family models",
    )
    parser.add_argument(
        "--tta-fractions",
        type=float,
        nargs="+",
        default=(0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0),
    )
    parser.add_argument(
        "--consensus-bonuses",
        type=float,
        nargs="+",
        default=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    )
    parser.add_argument(
        "--confidence-margins",
        type=float,
        nargs="+",
        default=(-5.0, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0),
    )
    parser.add_argument(
        "--smooth-confidence",
        action="store_true",
        help="interpret confidence margins as positive ramp widths from zero",
    )
    parser.add_argument(
        "--internal-bonuses",
        type=float,
        nargs="+",
        default=(-0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.35),
    )
    parser.add_argument(
        "--adaptive-bonuses",
        type=float,
        nargs="+",
        default=(0.025, 0.05, 0.075, 0.10, 0.15, 0.20),
    )
    parser.add_argument(
        "--adaptive-thresholds",
        type=float,
        nargs="+",
        default=(20.0, 30.0, 40.0, 50.0, 60.0),
    )
    parser.add_argument("--adaptive-width", type=float, default=40.0)
    parser.add_argument("--audit-bonus", type=float)
    parser.add_argument("--audit-threshold", type=float)
    parser.add_argument("--audit-tta-fraction", type=float)
    parser.add_argument("--per-well-output", type=Path)
    args = parser.parse_args()

    if sum(
        (
            args.adaptive_family_weight,
            args.adaptive_internal_stratified,
            args.adaptive_confidence,
            args.current_adaptive_corrections,
            args.adaptive_consensus_bonus,
            args.adaptive_disagreement_bonus,
            args.specialist_gr25_tta,
        )
    ) > 1:
        raise ValueError("choose only one adaptive audit mode")
    if args.specialist_gr25_tta:
        candidates = [(fraction, 0.0) for fraction in args.tta_fractions]
        current_matches = [
            index
            for index, candidate in enumerate(candidates)
            if np.allclose(candidate, (0.0, 0.0))
        ]
    elif args.adaptive_disagreement_bonus:
        candidates = [(0.0, 0.0)] + [
            (extra, threshold)
            for extra in args.disagreement_extras
            for threshold in args.disagreement_thresholds
        ]
        current_matches = [0]
    elif args.adaptive_consensus_bonus:
        candidates = [(bonus, 0.0) for bonus in args.consensus_bonuses]
        current_matches = [
            index
            for index, candidate in enumerate(candidates)
            if np.allclose(candidate, (0.50, 0.0))
        ]
    elif args.adaptive_confidence:
        candidates = [(-1000.0, 0.0)] + [
            (margin, 0.0) for margin in args.confidence_margins
        ]
        current_matches = [0]
    elif args.adaptive_internal_stratified:
        candidates = [(0.0, 0.0)] + [
            (bonus, 0.0)
            for bonus in args.internal_bonuses
            if not np.isclose(bonus, 0.0)
        ]
        current_matches = [0]
    elif args.adaptive_family_weight:
        candidates = [(0.0, 0.0)] + [
            (bonus, threshold)
            for bonus in args.adaptive_bonuses
            for threshold in args.adaptive_thresholds
        ]
        current_matches = [0]
    else:
        candidates = [
            (expansion, surface_weight)
            for expansion in args.expansions
            for surface_weight in args.surface_weights
        ]
        current_matches = [
            index
            for index, candidate in enumerate(candidates)
            if np.allclose(candidate, (CURRENT_TAIL_EXPANSION, CURRENT_SURFACE_WEIGHT))
        ]
    if len(current_matches) != 1:
        raise ValueError("grid must contain the current policy exactly once")
    current_index = current_matches[0]
    audit_index = -1
    if args.per_well_output is not None:
        if args.specialist_gr25_tta:
            audit_candidate = (args.audit_tta_fraction, 0.0)
            if args.audit_tta_fraction is None:
                raise ValueError(
                    "specialist TTA per-well output requires an audit fraction"
                )
        elif args.adaptive_disagreement_bonus:
            audit_candidate = (args.audit_bonus, args.audit_threshold)
            if None in audit_candidate:
                raise ValueError(
                    "disagreement per-well output requires extra and threshold"
                )
        elif args.adaptive_consensus_bonus:
            audit_candidate = (args.audit_bonus, 0.0)
            if args.audit_bonus is None:
                raise ValueError("consensus per-well output requires audit bonus")
        elif args.adaptive_confidence:
            audit_candidate = (args.audit_bonus, 0.0)
            if args.audit_bonus is None:
                raise ValueError("confidence per-well output requires audit margin")
        elif args.adaptive_internal_stratified:
            audit_candidate = (args.audit_bonus, 0.0)
            if args.audit_bonus is None:
                raise ValueError("internal per-well output requires audit bonus")
        elif args.adaptive_family_weight:
            audit_candidate = (args.audit_bonus, args.audit_threshold)
            if None in audit_candidate:
                raise ValueError("adaptive per-well output requires bonus and threshold")
        else:
            audit_candidate = (args.audit_expansion, args.audit_surface)
            if None in audit_candidate:
                raise ValueError("per-well output requires both correction parameters")
        audit_matches = [
            index
            for index, candidate in enumerate(candidates)
            if np.allclose(candidate, audit_candidate)
        ]
        if len(audit_matches) != 1:
            raise ValueError("audit policy must occur exactly once in the grid")
        audit_index = audit_matches[0]

    wells = load_all(args.data_root, args.workers)
    stratified_assignment = load_assignment(args.stratified_fold_file)
    audit_assignment = load_assignment(args.audit_fold_file)
    stratified = load_candidate(args.stratified_root)
    if set(stratified) != set(wells):
        raise ValueError("stratified cache does not cover every training well")

    group_sse = {
        "original": np.zeros((5, len(candidates)), dtype=np.float64),
        "stratified": np.zeros((5, len(candidates)), dtype=np.float64),
        "independent": np.zeros((5, len(candidates)), dtype=np.float64),
    }
    group_rows = {
        name: np.zeros(5, dtype=np.int64) for name in group_sse
    }
    per_well_rows: list[dict[str, str | int | float]] = []

    for original_fold in range(5):
        with np.load(args.base_root / f"fold{original_fold}.npz") as cached:
            well_ids = cached["well_ids"]
            starts = cached["row_starts"]
            components = cached["components"].astype(np.float64)
            raw = cached["coarse"].astype(np.float64)
            truth = cached["truth"].astype(np.float64)
        with np.load(args.tta_root / f"fold{original_fold}_gr25.npz") as cached:
            tta_components = cached["components"].astype(np.float64)
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
            stratified_prediction, stratified_truth = stratified[well_id]
            if not np.allclose(stratified_truth, truth[left:right], atol=1e-5):
                raise ValueError(f"{well_id}: stratified truth differs")

            raws = (
                raw[left:right],
                components[WIDE_SUPERVISED_INDEX, left:right],
                seed_prediction[left:right],
                small_prediction[left:right],
                stratified_prediction,
            )
            zero = [expanded_and_smoothed(well, value, 0.0) for value in raws]
            progress = np.linspace(0.0, 1.0, right - left)
            anchor = float(well.tvt[well.anchor_index])
            direction = [progress * (value - anchor) for value in raws]
            base_zero, original_zero, seed_zero, small_zero, stratified_zero = zero
            (
                base_direction,
                original_direction,
                seed_direction,
                small_direction,
                stratified_direction,
            ) = direction
            other_zero = (
                base_zero - OLD_WIDE_SUPERVISED_WEIGHT * original_zero
            ) / (1.0 - OLD_WIDE_SUPERVISED_WEIGHT)
            other_direction = (
                base_direction - OLD_WIDE_SUPERVISED_WEIGHT * original_direction
            ) / (1.0 - OLD_WIDE_SUPERVISED_WEIGHT)
            family_zero = (
                ORIGINAL_FRACTION * original_zero
                + SEED_FRACTION * seed_zero
                + SMALL_FRACTION * small_zero
                + STRATIFIED_FRACTION * stratified_zero
            )
            family_direction = (
                ORIGINAL_FRACTION * original_direction
                + SEED_FRACTION * seed_direction
                + SMALL_FRACTION * small_direction
                + STRATIFIED_FRACTION * stratified_direction
            )
            final_zero = (
                (1.0 - FINAL_WIDE_SUPERVISED_WEIGHT) * other_zero
                + FINAL_WIDE_SUPERVISED_WEIGHT * family_zero
            )
            final_direction = (
                (1.0 - FINAL_WIDE_SUPERVISED_WEIGHT) * other_direction
                + FINAL_WIDE_SUPERVISED_WEIGHT * family_direction
            )

            folds = {
                "original": original_fold,
                "stratified": stratified_assignment[well_id],
                "independent": audit_assignment[well_id],
            }
            rows = right - left
            for name, fold in folds.items():
                group_rows[name][fold] += rows
            well_sse = np.zeros(len(candidates), dtype=np.float64)
            if (
                args.adaptive_family_weight
                or args.adaptive_internal_stratified
                or args.adaptive_confidence
                or args.current_adaptive_corrections
                or args.adaptive_consensus_bonus
                or args.adaptive_disagreement_bonus
                or args.specialist_gr25_tta
            ):
                other_pre = other_zero + CURRENT_TAIL_EXPANSION * other_direction
                family_pre = family_zero + CURRENT_TAIL_EXPANSION * family_direction
                baseline_pre = 0.50 * other_pre + 0.50 * family_pre
                predicted_excursion = float(np.max(np.abs(baseline_pre - anchor)))
            if args.specialist_gr25_tta:
                original_tta_raw = tta_components[
                    WIDE_SUPERVISED_INDEX, left:right
                ]
                original_tta_zero = expanded_and_smoothed(
                    well, original_tta_raw, 0.0
                )
                original_tta_direction = progress * (original_tta_raw - anchor)
            for candidate_index, (first, second) in enumerate(candidates):
                if args.specialist_gr25_tta:
                    tta_fraction = first
                    candidate_original_zero = original_zero + tta_fraction * (
                        original_tta_zero - original_zero
                    )
                    candidate_original_direction = (
                        original_direction
                        + tta_fraction
                        * (original_tta_direction - original_direction)
                    )
                    candidate_family_zero = (
                        ORIGINAL_FRACTION * candidate_original_zero
                        + SEED_FRACTION * seed_zero
                        + SMALL_FRACTION * small_zero
                        + STRATIFIED_FRACTION * stratified_zero
                    )
                    candidate_family_direction = (
                        ORIGINAL_FRACTION * candidate_original_direction
                        + SEED_FRACTION * seed_direction
                        + SMALL_FRACTION * small_direction
                        + STRATIFIED_FRACTION * stratified_direction
                    )
                    other_pre = other_zero + CURRENT_TAIL_EXPANSION * other_direction
                    family_pre = (
                        candidate_family_zero
                        + CURRENT_TAIL_EXPANSION * candidate_family_direction
                    )
                    baseline_pre = 0.50 * other_pre + 0.50 * family_pre
                    predicted_excursion = float(
                        np.max(np.abs(baseline_pre - anchor))
                    )
                    other_excursion = float(np.max(np.abs(other_pre - anchor)))
                    family_excursion = float(np.max(np.abs(family_pre - anchor)))
                    disagreement_ratio = float(
                        np.sqrt(np.mean(np.square(family_pre - other_pre)))
                        / max(predicted_excursion, 1e-6)
                    )
                    disagreement_gate = float(
                        np.clip((disagreement_ratio - 0.15) / 0.20, 0.0, 1.0)
                    )
                    excursion_gate = float(
                        np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    )
                    confidence_gate = float(family_excursion > other_excursion)
                    consensus_bonus = 0.55 + 0.10 * disagreement_gate
                    family_weight = (
                        0.50
                        + consensus_bonus * excursion_gate * confidence_gate
                    )
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * family_pre
                    )
                    surface_weight = CURRENT_SURFACE_WEIGHT
                elif args.adaptive_disagreement_bonus:
                    extra_bonus, disagreement_threshold = first, second
                    other_pre = other_zero + CURRENT_TAIL_EXPANSION * other_direction
                    family_pre = (
                        family_zero + CURRENT_TAIL_EXPANSION * family_direction
                    )
                    baseline_pre = 0.50 * other_pre + 0.50 * family_pre
                    predicted_excursion = float(
                        np.max(np.abs(baseline_pre - anchor))
                    )
                    other_excursion = float(np.max(np.abs(other_pre - anchor)))
                    family_excursion = float(np.max(np.abs(family_pre - anchor)))
                    disagreement_ratio = float(
                        np.sqrt(np.mean(np.square(family_pre - other_pre)))
                        / max(predicted_excursion, 1e-6)
                    )
                    disagreement_gate = float(
                        np.clip(
                            (disagreement_ratio - disagreement_threshold)
                            / args.disagreement_width,
                            0.0,
                            1.0,
                        )
                    )
                    excursion_gate = float(
                        np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    )
                    confidence_gate = float(family_excursion > other_excursion)
                    consensus_bonus = 0.55 + extra_bonus * disagreement_gate
                    family_weight = (
                        0.50
                        + consensus_bonus * excursion_gate * confidence_gate
                    )
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * family_pre
                    )
                    surface_weight = CURRENT_SURFACE_WEIGHT
                elif args.adaptive_consensus_bonus:
                    consensus_bonus = first
                    other_pre = other_zero + CURRENT_TAIL_EXPANSION * other_direction
                    family_pre = (
                        family_zero + CURRENT_TAIL_EXPANSION * family_direction
                    )
                    baseline_pre = 0.50 * other_pre + 0.50 * family_pre
                    predicted_excursion = float(
                        np.max(np.abs(baseline_pre - anchor))
                    )
                    other_excursion = float(np.max(np.abs(other_pre - anchor)))
                    family_excursion = float(np.max(np.abs(family_pre - anchor)))
                    excursion_gate = float(
                        np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    )
                    confidence_gate = float(family_excursion > other_excursion)
                    family_weight = (
                        0.50
                        + consensus_bonus * excursion_gate * confidence_gate
                    )
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * family_pre
                    )
                    surface_weight = CURRENT_SURFACE_WEIGHT
                elif args.current_adaptive_corrections:
                    expansion, surface_weight = first, second
                    other_pre = other_zero + expansion * other_direction
                    family_pre = family_zero + expansion * family_direction
                    baseline_pre = 0.50 * other_pre + 0.50 * family_pre
                    predicted_excursion = float(
                        np.max(np.abs(baseline_pre - anchor))
                    )
                    other_excursion = float(np.max(np.abs(other_pre - anchor)))
                    family_excursion = float(np.max(np.abs(family_pre - anchor)))
                    excursion_gate = float(
                        np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    )
                    confidence_gate = float(family_excursion > other_excursion)
                    family_weight = (
                        0.50 + 0.50 * excursion_gate * confidence_gate
                    )
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * family_pre
                    )
                elif args.adaptive_confidence:
                    confidence_margin = first
                    gate = np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    other_excursion = float(np.max(np.abs(other_pre - anchor)))
                    family_excursion = float(np.max(np.abs(family_pre - anchor)))
                    excursion_advantage = family_excursion - other_excursion
                    if confidence_margin < -999.0:
                        confidence_gate = 1.0
                    elif args.smooth_confidence:
                        if confidence_margin <= 0.0:
                            raise ValueError("confidence width must be positive")
                        confidence_gate = float(
                            np.clip(excursion_advantage / confidence_margin, 0.0, 1.0)
                        )
                    else:
                        confidence_gate = float(
                            excursion_advantage > confidence_margin
                        )
                    family_weight = 0.50 + 0.50 * gate * confidence_gate
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * family_pre
                    )
                    surface_weight = CURRENT_SURFACE_WEIGHT
                elif args.adaptive_internal_stratified:
                    internal_bonus = first
                    gate = np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    family_weight = 0.50 + 0.50 * gate
                    original_pre = original_zero + 0.06 * original_direction
                    stratified_pre = stratified_zero + 0.06 * stratified_direction
                    adjusted_family_pre = family_pre + internal_bonus * gate * (
                        stratified_pre - original_pre
                    )
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * adjusted_family_pre
                    )
                    surface_weight = CURRENT_SURFACE_WEIGHT
                elif args.adaptive_family_weight:
                    bonus, threshold = first, second
                    gate = np.clip(
                        (predicted_excursion - threshold) / args.adaptive_width,
                        0.0,
                        1.0,
                    )
                    family_weight = 0.50 + bonus * gate
                    prediction = (
                        (1.0 - family_weight) * other_pre
                        + family_weight * family_pre
                    )
                    surface_weight = CURRENT_SURFACE_WEIGHT
                else:
                    expansion, surface_weight = first, second
                    prediction = final_zero + expansion * final_direction
                prediction += surface_weight * surface_extrapolation(
                    well,
                    prediction,
                    visible_rows=1000,
                    correction_limit=24.0,
                )
                error = prediction - truth[left:right]
                sse = float(error @ error)
                well_sse[candidate_index] = sse
                for name, fold in folds.items():
                    group_sse[name][fold, candidate_index] += sse
            if args.per_well_output is not None:
                audit_row: dict[str, str | int | float] = {
                    "well_id": well_id,
                    "original_fold": folds["original"],
                    "stratified_fold": folds["stratified"],
                    "independent_fold": folds["independent"],
                    "rows": rows,
                    "baseline_sse": well_sse[current_index],
                    "candidate_sse": well_sse[audit_index],
                }
                if args.specialist_gr25_tta:
                    audit_row.update(
                        {
                            "specialist_gr25_fraction": candidates[audit_index][0],
                        }
                    )
                elif args.adaptive_disagreement_bonus:
                    audit_extra, audit_threshold = candidates[audit_index]
                    audit_disagreement_gate = float(
                        np.clip(
                            (disagreement_ratio - audit_threshold)
                            / args.disagreement_width,
                            0.0,
                            1.0,
                        )
                    )
                    audit_consensus_bonus = (
                        0.55 + audit_extra * audit_disagreement_gate
                    )
                    audit_family_weight = (
                        0.50
                        + audit_consensus_bonus
                        * excursion_gate
                        * confidence_gate
                    )
                    audit_row.update(
                        {
                            "predicted_excursion": predicted_excursion,
                            "gate": excursion_gate,
                            "family_weight": audit_family_weight,
                            "family_minus_other_excursion": family_excursion
                            - other_excursion,
                            "disagreement_ratio": disagreement_ratio,
                            "disagreement_gate": audit_disagreement_gate,
                            "extra_bonus": audit_extra,
                            "disagreement_threshold": audit_threshold,
                        }
                    )
                elif args.adaptive_consensus_bonus:
                    audit_bonus = candidates[audit_index][0]
                    audit_family_weight = (
                        0.50
                        + audit_bonus * excursion_gate * confidence_gate
                    )
                    audit_row.update(
                        {
                            "predicted_excursion": predicted_excursion,
                            "gate": excursion_gate,
                            "family_weight": audit_family_weight,
                            "family_minus_other_excursion": family_excursion
                            - other_excursion,
                            "consensus_bonus": audit_bonus,
                        }
                    )
                elif args.adaptive_confidence:
                    audit_gate = float(
                        np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    )
                    other_excursion = float(np.max(np.abs(other_pre - anchor)))
                    family_excursion = float(np.max(np.abs(family_pre - anchor)))
                    audit_margin = candidates[audit_index][0]
                    excursion_advantage = family_excursion - other_excursion
                    if args.smooth_confidence:
                        confidence_gate = float(
                            np.clip(excursion_advantage / audit_margin, 0.0, 1.0)
                        )
                    else:
                        confidence_gate = float(excursion_advantage > audit_margin)
                    audit_row.update(
                        {
                            "predicted_excursion": predicted_excursion,
                            "gate": audit_gate,
                            "family_weight": 0.50
                            + 0.50 * audit_gate * confidence_gate,
                            "family_minus_other_excursion": family_excursion
                            - other_excursion,
                            "confidence_gate": confidence_gate,
                        }
                    )
                elif args.adaptive_internal_stratified:
                    audit_gate = float(
                        np.clip((predicted_excursion - 40.0) / 10.0, 0.0, 1.0)
                    )
                    audit_row.update(
                        {
                            "predicted_excursion": predicted_excursion,
                            "gate": audit_gate,
                            "family_weight": 0.50 + 0.50 * audit_gate,
                            "internal_stratified_bonus": candidates[audit_index][0],
                        }
                    )
                elif args.adaptive_family_weight:
                    audit_bonus, audit_threshold = candidates[audit_index]
                    audit_gate = float(
                        np.clip(
                            (predicted_excursion - audit_threshold)
                            / args.adaptive_width,
                            0.0,
                            1.0,
                        )
                    )
                    audit_row.update(
                        {
                            "predicted_excursion": predicted_excursion,
                            "gate": audit_gate,
                            "family_weight": 0.50 + audit_bonus * audit_gate,
                            "other_excursion": float(
                                np.max(np.abs(other_pre - anchor))
                            ),
                            "family_excursion": float(
                                np.max(np.abs(family_pre - anchor))
                            ),
                            "family_minus_other_excursion": float(
                                np.max(np.abs(family_pre - anchor))
                                - np.max(np.abs(other_pre - anchor))
                            ),
                            "family_other_disagreement": float(
                                np.sqrt(np.mean(np.square(family_pre - other_pre)))
                            ),
                            "baseline_signed_extreme": float(
                                baseline_pre[
                                    np.argmax(np.abs(baseline_pre - anchor))
                                ]
                                - anchor
                            ),
                            "other_endpoint": float(other_pre[-1] - anchor),
                            "family_endpoint": float(family_pre[-1] - anchor),
                        }
                    )
                per_well_rows.append(audit_row)

    total_rows = int(np.sum(group_rows["original"]))
    for name in group_sse:
        if int(np.sum(group_rows[name])) != total_rows:
            raise RuntimeError(f"{name}: regrouping changed row count")
    pooled = np.sqrt(np.sum(group_sse["original"], axis=0) / total_rows)
    print("fixed_top")
    shown = set(np.argsort(pooled)[: args.top]) | {current_index}
    for candidate_index in sorted(shown, key=lambda index: pooled[index]):
        first, second = candidates[candidate_index]
        if args.specialist_gr25_tta:
            label = f"specialist_gr25_fraction={first:.4f}"
        elif args.adaptive_disagreement_bonus:
            label = f"extra={first:.4f} threshold={second:.3f}"
        elif args.adaptive_consensus_bonus:
            label = f"consensus_bonus={first:.4f}"
        elif args.adaptive_confidence:
            label = (
                "confidence_margin=none"
                if first < -999.0
                else (
                    f"confidence_width={first:.2f}"
                    if args.smooth_confidence
                    else f"confidence_margin={first:.2f}"
                )
            )
        elif args.adaptive_internal_stratified:
            label = f"internal_stratified_bonus={first:.4f}"
        elif args.adaptive_family_weight:
            label = f"bonus={first:.4f} threshold={second:.1f}"
        else:
            label = f"expansion={first:.4f} surface={second:.4f}"
        print(
            f"{label} pooled={pooled[candidate_index]:.6f} original_folds="
            + ",".join(
                f"{np.sqrt(group_sse['original'][fold, candidate_index] / group_rows['original'][fold]):.6f}"
                for fold in range(5)
            )
        )

    if (
        args.adaptive_consensus_bonus
        or args.adaptive_disagreement_bonus
        or args.specialist_gr25_tta
    ):
        print("binary_adaptive_audit")
        for candidate_index, (bonus, _) in enumerate(candidates):
            if candidate_index == current_index:
                continue
            for group_name in ("original", "stratified", "independent"):
                selections = []
                nested_sse = 0.0
                for heldout in range(5):
                    tuning = np.sum(
                        np.delete(group_sse[group_name], heldout, axis=0), axis=0
                    )
                    selected = (
                        candidate_index
                        if tuning[candidate_index] < tuning[current_index]
                        else current_index
                    )
                    selections.append(int(selected == candidate_index))
                    nested_sse += group_sse[group_name][heldout, selected]
                print(
                    f"candidate={candidates[candidate_index]} group={group_name} "
                    f"selected={sum(selections)}/5 pattern="
                    + "".join(str(value) for value in selections)
                    + f" nested={np.sqrt(nested_sse / total_rows):.6f}"
                )

    if (
        args.adaptive_family_weight
        or args.adaptive_internal_stratified
        or args.adaptive_confidence
        or args.adaptive_consensus_bonus
        or args.adaptive_disagreement_bonus
        or args.specialist_gr25_tta
    ):
        for group_name in ("original", "stratified", "independent"):
            nested_score(
                group_name,
                group_sse[group_name],
                group_rows[group_name],
                candidates,
                adaptive=args.adaptive_family_weight,
                internal=args.adaptive_internal_stratified,
                confidence=args.adaptive_confidence,
                confidence_smooth=args.smooth_confidence,
                consensus_bonus=args.adaptive_consensus_bonus,
                disagreement_bonus=args.adaptive_disagreement_bonus,
                specialist_tta=args.specialist_gr25_tta,
            )
        if args.per_well_output is not None:
            args.per_well_output.parent.mkdir(parents=True, exist_ok=True)
            with args.per_well_output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(per_well_rows[0]))
                writer.writeheader()
                writer.writerows(per_well_rows)
            print(f"wrote={args.per_well_output} wells={len(per_well_rows)}")
        return

    axis_masks = {
        "tail_only": np.asarray(
            [
                index
                for index, (_, surface_weight) in enumerate(candidates)
                if np.isclose(surface_weight, CURRENT_SURFACE_WEIGHT)
            ]
        ),
        "surface_only": np.asarray(
            [
                index
                for index, (expansion, _) in enumerate(candidates)
                if np.isclose(expansion, CURRENT_TAIL_EXPANSION)
            ]
        ),
        "joint": np.arange(len(candidates)),
    }
    for audit_name, allowed in axis_masks.items():
        print(f"audit={audit_name}")
        for group_name in ("original", "stratified", "independent"):
            nested_score(
                group_name,
                group_sse[group_name],
                group_rows[group_name],
                candidates,
                allowed,
            )

    if args.per_well_output is not None:
        args.per_well_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_well_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_well_rows[0]))
            writer.writeheader()
            writer.writerows(per_well_rows)
        print(f"wrote={args.per_well_output} wells={len(per_well_rows)}")


if __name__ == "__main__":
    main()
