from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import (
    DEFAULT_DATA_ROOT,
    HORIZONTAL_INFERENCE_COLUMNS,
    TYPEWELL_INFERENCE_COLUMNS,
    Well,
)
from rogii.sequence import SequenceConfig, _robust_affine


RUN_ID = "v5_batch2_run_006_goal_050"
CHECKPOINT_ROOT = Path("/geo/geo_new/rogii_v20_checkpoint_vault")
PARENTS = (
    ("fixed", "alignment_fixed_synth_fold{fold}.pt", "narrow", 8, 1.5),
    (
        "distractor",
        "alignment_distractor_h05_fold{fold}.pt",
        "narrow",
        8,
        1.5,
    ),
    ("wide", "alignment_wide_synth_fold{fold}.pt", "wide", 8, 1.5),
    ("forward", "alignment_forward_std_fold{fold}.pt", "narrow", 8, 1.0),
    (
        "forward_distractor",
        "alignment_forward_distractor_fold{fold}.pt",
        "narrow",
        8,
        1.0,
    ),
    ("rare", "alignment_forward_exc4_fold{fold}.pt", "narrow", 8, 1.25),
    (
        "supervised",
        "alignment_forward_exc2sup_fold{fold}.pt",
        "narrow",
        8,
        1.0,
    ),
    (
        "wide_forward",
        "alignment_wide_forward_extreme_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    (
        "wide_supervised",
        "alignment_wide_forward_extreme_exc2sup_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    ("ordinal", "alignment_ordinal_wide_fold{fold}.pt", "wide", 8, 0.5),
    (
        "long",
        "alignment_longsynthetic_exc2sup_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    (
        "base12",
        "alignment_wide_base12_exc2sup_fold{fold}.pt",
        "wide",
        12,
        1.0,
    ),
    (
        "base16",
        "alignment_wide_base16_exc2sup_fold{fold}.pt",
        "wide",
        16,
        1.0,
    ),
    (
        "base6",
        "alignment_wide_supervised_base6_fold{fold}.pt",
        "wide",
        6,
        1.0,
    ),
    (
        "seed3",
        "alignment_wide_supervised_seed3_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    (
        "strat",
        "alignment_wide_supervised_strat_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
CORRECTION_SIGMAS = (32.0, 128.0)
GR_SIGMAS = (0.0, 4.0, 12.0, 32.0, 64.0)
GR_STRIDES = (8, 16, 32)
OUTER_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
CORRECTION_CAP = 64.0


@dataclass(frozen=True)
class Parent:
    name: str
    model: AlignmentUNet
    scale: str
    temperature: float
    checkpoint: Path
    checkpoint_sha256: str


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


def finite_fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return np.zeros_like(values)
    return np.interp(np.arange(len(values)), finite, values[finite])


def load_inference_well(well_id: str, data_root: Path) -> Well:
    root = data_root / "train"
    horizontal = pd.read_csv(
        root / f"{well_id}__horizontal_well.csv",
        usecols=list(HORIZONTAL_INFERENCE_COLUMNS),
    )
    typewell = pd.read_csv(
        root / f"{well_id}__typewell.csv",
        usecols=list(TYPEWELL_INFERENCE_COLUMNS),
    ).sort_values("TVT")
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise ValueError(f"{well_id}: no legal visible TVT prefix")
    placeholder = np.where(
        np.isfinite(tvt_input), tvt_input, tvt_input[known[-1]]
    )
    well = Well(
        well_id=well_id,
        md=horizontal["MD"].to_numpy(np.float64),
        x=horizontal["X"].to_numpy(np.float64),
        y=horizontal["Y"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=horizontal["GR"].to_numpy(np.float64),
        tvt=placeholder,
        tvt_input=tvt_input,
        typewell_tvt=typewell["TVT"].to_numpy(np.float64),
        typewell_gr=typewell["GR"].to_numpy(np.float64),
    )
    well.validate()
    return well


def load_baseline(
    path: Path, fold: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        if int(cache["fold"]) != fold:
            raise ValueError("baseline cache fold differs")
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise ValueError("baseline cache firewall failed")
        return (
            cache["well_ids"].astype(str),
            cache["row_starts"].astype(np.int64),
            cache["baseline"].astype(np.float32),
            cache["truth"].astype(np.float32),
        )


def configs() -> dict[str, SequenceConfig]:
    return {
        "narrow": SequenceConfig(
            prefix_points=64,
            tail_points=640,
            state_radius=64.0,
            state_step=0.5,
            target_kind="tvt_delta",
            target_scale=40.0,
            row_stride=16.0,
            gr_smooth_window=51,
        ),
        "wide": SequenceConfig(
            prefix_points=64,
            tail_points=640,
            state_radius=120.0,
            state_step=1.0,
            target_kind="tvt_delta",
            target_scale=40.0,
            row_stride=16.0,
            gr_smooth_window=51,
        ),
    }


def load_parents(fold: int, device: torch.device) -> list[Parent]:
    parents = []
    for name, pattern, scale, base, temperature in PARENTS:
        checkpoint = CHECKPOINT_ROOT / pattern.format(fold=fold)
        model = AlignmentUNet(input_channels=19, base=base).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        parents.append(
            Parent(
                name=name,
                model=model,
                scale=scale,
                temperature=temperature,
                checkpoint=checkpoint,
                checkpoint_sha256=sha256(checkpoint),
            )
        )
    return parents


def sampled_path(
    offset: np.ndarray,
    positions: np.ndarray,
    anchor_tvt: float,
    target_indices: np.ndarray,
) -> np.ndarray:
    return np.interp(target_indices, positions, anchor_tvt + offset)


def candidate_schema(parents: list[Parent]) -> np.ndarray:
    names = []
    for parent in parents:
        views = [
            *(f"q{quantile:g}" for quantile in QUANTILES),
            "mode",
            "temp0.5",
            "temp2",
        ]
        for view in views:
            for sigma in CORRECTION_SIGMAS:
                names.append(f"{parent.name}_{view}_s{sigma:g}")
    return np.asarray(names)


@torch.no_grad()
def candidate_corrections(
    well: Well,
    parents: list[Parent],
    configurations: dict[str, SequenceConfig],
    device: torch.device,
) -> np.ndarray:
    examples = {
        name: make_alignment_example(
            well, well.anchor_index, configuration
        )
        for name, configuration in configurations.items()
    }
    corrections = []
    for parent in parents:
        configuration = configurations[parent.scale]
        image, _, metadata = examples[parent.scale]
        logits = parent.model(torch.from_numpy(image)[None].to(device))[
            0, :, configuration.prefix_points :
        ].float()
        valid = (
            np.asarray(metadata["valid_positions"])[
                configuration.prefix_points :
            ]
            > 0.5
        )
        logits = logits[:, valid]
        positions = np.asarray(metadata["positions"])[
            configuration.prefix_points :
        ][valid]
        offsets = configuration.offsets.astype(np.float64)
        probability = torch.softmax(
            logits / parent.temperature, dim=0
        ).cpu().numpy()
        reference_offset = np.sum(probability * offsets[:, None], axis=0)
        reference = sampled_path(
            reference_offset,
            positions,
            float(metadata["anchor_tvt"]),
            well.prediction_indices,
        )
        cdf = np.cumsum(probability, axis=0)
        variants = [
            offsets[np.argmax(cdf >= quantile, axis=0)]
            for quantile in QUANTILES
        ]
        variants.append(offsets[np.argmax(probability, axis=0)])
        for multiplier in (0.5, 2.0):
            view = torch.softmax(
                logits / (parent.temperature * multiplier), dim=0
            ).cpu().numpy()
            variants.append(np.sum(view * offsets[:, None], axis=0))
        for variant_offset in variants:
            variant = sampled_path(
                variant_offset,
                positions,
                float(metadata["anchor_tvt"]),
                well.prediction_indices,
            )
            raw = variant - reference
            for sigma in CORRECTION_SIGMAS:
                correction = gaussian_filter1d(
                    raw, sigma=sigma, mode="nearest"
                )
                correction -= correction[0]
                corrections.append(
                    np.clip(
                        correction, -CORRECTION_CAP, CORRECTION_CAP
                    ).astype(np.float32)
                )
    return np.stack(corrections)


def row_correlation(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    centered_target = target - np.mean(target)
    denominator = np.linalg.norm(centered, axis=1) * max(
        float(np.linalg.norm(centered_target)), 1e-12
    )
    return (centered @ centered_target) / np.maximum(denominator, 1e-12)


def calibrated_metrics(
    well: Well,
    paths: np.ndarray,
    *,
    shifted_control: bool,
) -> tuple[np.ndarray, list[str]]:
    anchor = well.anchor_index
    visible_tvt = well.tvt_input[: anchor + 1]
    visible_gr = well.gr[: anchor + 1]
    reference_visible = np.interp(
        visible_tvt,
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, offset, residual_scale = _robust_affine(
        reference_visible, visible_gr
    )
    observed = finite_fill(well.gr[well.prediction_indices])
    if shifted_control and len(observed) > 1:
        observed = np.roll(observed, len(observed) // 2)
    raw_reference = np.interp(
        paths.ravel(),
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    ).reshape(paths.shape)
    supported = np.isfinite(raw_reference)
    support_fraction = np.mean(supported, axis=1)
    expected = gain * np.stack(
        [finite_fill(candidate) for candidate in raw_reference]
    ) + offset
    columns = []
    names = []
    for sigma in GR_SIGMAS:
        observed_view = (
            observed
            if sigma == 0.0
            else gaussian_filter1d(observed, sigma, mode="nearest")
        )
        expected_view = (
            expected
            if sigma == 0.0
            else gaussian_filter1d(
                expected, sigma, axis=1, mode="nearest"
            )
        )
        for stride in GR_STRIDES:
            sample = np.unique(
                np.r_[np.arange(0, len(observed), stride), len(observed) - 1]
            )
            residual = (
                expected_view[:, sample] - observed_view[sample]
            ) / max(residual_scale, 3.0)
            absolute = np.abs(residual)
            huber = np.where(
                absolute <= 1.5,
                0.5 * np.square(residual),
                1.5 * (absolute - 0.75),
            )
            unsupported = 2.0 * (1.0 - support_fraction)
            columns.extend(
                (
                    np.mean(absolute, axis=1) + unsupported,
                    np.mean(huber, axis=1) + unsupported,
                    -row_correlation(
                        expected_view[:, sample], observed_view[sample]
                    )
                    + unsupported,
                )
            )
            names.extend(
                (
                    f"mae_s{sigma:g}_d{stride}",
                    f"huber_s{sigma:g}_d{stride}",
                    f"corr_s{sigma:g}_d{stride}",
                )
            )
            observed_difference = np.diff(observed_view[sample])
            expected_difference = np.diff(
                expected_view[:, sample], axis=1
            )
            difference_scale = max(
                float(np.std(observed_difference)), 3.0
            )
            difference_residual = (
                expected_difference - observed_difference[None]
            ) / difference_scale
            columns.extend(
                (
                    np.mean(np.abs(difference_residual), axis=1)
                    + unsupported,
                    -row_correlation(
                        expected_difference, observed_difference
                    )
                    + unsupported,
                )
            )
            names.extend(
                (
                    f"diff_mae_s{sigma:g}_d{stride}",
                    f"diff_corr_s{sigma:g}_d{stride}",
                )
            )
    return np.column_stack(columns).astype(np.float32), names


def assemble_selector_corrections(
    metrics: np.ndarray,
    corrections: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    selector_count = metrics.shape[2]
    output = np.empty(
        (selector_count, corrections.shape[1]), dtype=np.float32
    )
    for selector in range(selector_count):
        chosen = np.argmin(metrics[:, :, selector], axis=1)
        for well_index, candidate_index in enumerate(chosen):
            left, right = starts[well_index : well_index + 2]
            output[selector, left:right] = corrections[
                candidate_index, left:right
            ]
    return output


def selector_priority(name: str) -> tuple[int, float, int, str]:
    metric = name.split("_", 1)[0]
    metric_priority = {
        "mae": 0,
        "huber": 1,
        "diff": 2,
        "corr": 3,
    }.get(metric, 4)
    sigma = float(name.split("_s", 1)[1].split("_", 1)[0])
    stride = int(name.rsplit("_d", 1)[1])
    return metric_priority, sigma, -stride, name


def capacity_screen(
    baseline: np.ndarray,
    truth: np.ndarray,
    selector_corrections: np.ndarray,
    metric_names: list[str],
) -> dict[str, object]:
    baseline_score = rmse(baseline, truth)
    individual = []
    for selector, name in enumerate(metric_names):
        delta = selector_corrections[selector]
        standalone = rmse(baseline + delta, truth)
        for weight in OUTER_WEIGHTS:
            score = rmse(baseline + weight * delta, truth)
            individual.append(
                {
                    "selector_index": selector,
                    "metric": name,
                    "weight": weight,
                    "rmse": score,
                    "gain": baseline_score - score,
                    "standalone_rmse": standalone,
                }
            )
    best_individual = min(
        individual,
        key=lambda item: (
            item["rmse"],
            item["weight"],
            selector_priority(item["metric"]),
        ),
    )
    current = baseline.astype(np.float64).copy()
    available = set(range(len(metric_names)))
    selected = []
    for _ in range(3):
        candidates = []
        for selector in available:
            delta = selector_corrections[selector]
            for weight in OUTER_WEIGHTS[1:]:
                score = rmse(current + weight * delta, truth)
                candidates.append(
                    (
                        score,
                        weight,
                        selector_priority(metric_names[selector]),
                        selector,
                    )
                )
        score, weight, _, selector = min(candidates)
        before = rmse(current, truth)
        if score >= before:
            break
        current += weight * selector_corrections[selector]
        selected.append(
            {
                "selector_index": selector,
                "metric": metric_names[selector],
                "weight": weight,
                "rmse_after": score,
                "incremental_gain": before - score,
            }
        )
        available.remove(selector)
    return {
        "baseline_rmse": baseline_score,
        "best_individual": best_individual,
        "individual_grid": individual,
        "greedy_up_to_three_diagnostic_only": {
            "components": selected,
            "rmse": rmse(current, truth),
            "gain": baseline_score - rmse(current, truth),
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    well_ids, starts, baseline, truth = load_baseline(
        args.baseline_cache, args.fold
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        wells = list(
            pool.map(
                lambda well_id: load_inference_well(
                    well_id, args.data_root
                ),
                well_ids.tolist(),
            )
        )
    device = torch.device(args.device)
    configurations = configs()
    parents = load_parents(args.fold, device)
    candidate_names = candidate_schema(parents)
    correction_parts = []
    metric_parts = []
    shifted_metric_parts = []
    metric_names = None
    for well_index, well in enumerate(wells):
        left, right = starts[well_index : well_index + 2]
        correction = candidate_corrections(
            well, parents, configurations, device
        )
        if correction.shape != (len(candidate_names), right - left):
            raise RuntimeError(f"{well.well_id}: candidate shape changed")
        paths = baseline[left:right][None].astype(np.float64) + correction
        metrics, names = calibrated_metrics(
            well, paths, shifted_control=False
        )
        shifted_metrics, shifted_names = calibrated_metrics(
            well, paths, shifted_control=True
        )
        if names != shifted_names:
            raise RuntimeError("real/control metric schema differs")
        if metric_names is None:
            metric_names = names
        elif metric_names != names:
            raise RuntimeError("metric schema changed between wells")
        correction_parts.append(correction.astype(np.float16))
        metric_parts.append(metrics)
        shifted_metric_parts.append(shifted_metrics)
        if (well_index + 1) % 10 == 0 or well_index + 1 == len(wells):
            print(
                json.dumps(
                    {
                        "fold": args.fold,
                        "wells": well_index + 1,
                        "total_wells": len(wells),
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    assert metric_names is not None
    corrections = np.concatenate(correction_parts, axis=1)
    metrics = np.stack(metric_parts)
    shifted_metrics = np.stack(shifted_metric_parts)
    selector_corrections = assemble_selector_corrections(
        metrics, corrections.astype(np.float32), starts
    )
    shifted_selector_corrections = assemble_selector_corrections(
        shifted_metrics, corrections.astype(np.float32), starts
    )
    capacity = capacity_screen(
        baseline, truth, selector_corrections, metric_names
    )
    shifted_capacity = capacity_screen(
        baseline, truth, shifted_selector_corrections, metric_names
    )
    args.bank_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.bank_output,
        fold=np.asarray(args.fold, dtype=np.int8),
        well_ids=well_ids,
        row_starts=starts,
        baseline=baseline,
        truth=truth,
        corrections=corrections,
        candidate_names=candidate_names,
        metrics=metrics,
        shifted_control_metrics=shifted_metrics,
        metric_names=np.asarray(metric_names),
        checkpoint_names=np.asarray(
            [str(parent.checkpoint) for parent in parents]
        ),
        checkpoint_sha256=np.asarray(
            [parent.checkpoint_sha256 for parent in parents]
        ),
        source_sha256=np.asarray(sha256(Path(__file__))),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
        held_horizontal_target_loaded_for_candidates=np.asarray(False),
        held_formation_columns_loaded=np.asarray(False),
        held_typewell_geology_loaded=np.asarray(False),
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": "protected_v20_posterior_mode_bank",
        "status": "complete_F0_capacity_diagnostic",
        "fold": args.fold,
        "held_wells": len(wells),
        "held_rows": len(baseline),
        "parent_count": len(parents),
        "candidate_count": len(candidate_names),
        "selector_count": len(metric_names),
        "capacity_same_held_diagnostic_only": capacity,
        "shifted_hidden_gr_negative_control_same_held_diagnostic_only": (
            shifted_capacity
        ),
        "parameters": {
            "parents": [
                {
                    "name": parent.name,
                    "checkpoint": str(parent.checkpoint),
                    "checkpoint_sha256": parent.checkpoint_sha256,
                    "scale": parent.scale,
                    "temperature": parent.temperature,
                }
                for parent in parents
            ],
            "quantiles": list(QUANTILES),
            "correction_sigmas": list(CORRECTION_SIGMAS),
            "gr_sigmas": list(GR_SIGMAS),
            "gr_strides": list(GR_STRIDES),
            "outer_weights": list(OUTER_WEIGHTS),
            "correction_cap": CORRECTION_CAP,
        },
        "controls": {
            "candidate_generation_columns": [
                "MD",
                "X",
                "Y",
                "Z",
                "GR",
                "TVT_input",
                "typewell_TVT",
                "typewell_GR",
            ],
            "held_horizontal_target_loaded_for_candidates": False,
            "held_formation_columns_loaded": False,
            "held_typewell_geology_loaded": False,
            "audit_fold_loaded": False,
            "confirmation_regroupings_loaded": False,
            "zero_weight_reproduces_v20_exactly": all(
                item["rmse"] == capacity["baseline_rmse"]
                for item in capacity["individual_grid"]
                if item["weight"] == 0.0
            ),
        },
        "bank_output": str(args.bank_output),
        "bank_sha256": sha256(args.bank_output),
        "source_sha256": sha256(Path(__file__)),
        "selection_clean_nested_selector_run": False,
        "elapsed_seconds": time.time() - started,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        default=Path(
            "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
            "v20_components/fold0.npz"
        ),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bank-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    if args.bank_output.exists() or args.result_output.exists():
        raise FileExistsError("refusing to overwrite mode-bank output")
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "best_individual": result[
                    "capacity_same_held_diagnostic_only"
                ]["best_individual"],
                "greedy_up_to_three_diagnostic_only": result[
                    "capacity_same_held_diagnostic_only"
                ]["greedy_up_to_three_diagnostic_only"],
                "shifted_hidden_gr_negative_control": result[
                    "shifted_hidden_gr_negative_control_same_held_diagnostic_only"
                ]["best_individual"],
                "controls": result["controls"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
