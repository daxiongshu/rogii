from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from limited_query_cv import train_v5_run6_frozen_residual_rawstate as core
from limited_query_cv import train_v5_run6_true_raw_absolute_residual as raw
from limited_query_cv.train_v5_run6_absolute_state_continuation import (
    absolute_planes,
)
from rogii.data import DEFAULT_DATA_ROOT, Well


VARIANT = "delta_calibrated_separable_absolute_b8"
SEED = 20262017
core.VARIANTS[VARIANT] = {
    "extra_channels": 8,
    "base": 8,
    "seed": SEED,
}
_PREVIOUS_RESIDUAL_PLANES = core.residual_planes


def calibrated_separable_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
) -> tuple[np.ndarray, dict[str, float | int]]:
    mismatch = np.asarray(image[0], dtype=np.float64)
    observed = np.asarray(metadata["gr"], dtype=np.float64)
    valid = (
        (np.asarray(metadata["valid_positions"]) > 0.5)
        & np.isfinite(observed)
        & (np.asarray(image[8, 0]) < 0.5)
    )
    slopes = []
    for row in mismatch:
        selected = valid & np.isfinite(row) & (np.abs(row) < 5.5)
        if int(np.sum(selected)) < 32:
            continue
        horizontal = observed[selected]
        likelihood = row[selected]
        horizontal = horizontal - np.mean(horizontal)
        likelihood = likelihood - np.mean(likelihood)
        denominator = float(np.dot(likelihood, likelihood))
        if denominator <= 1e-8:
            continue
        slope = float(np.dot(likelihood, horizontal) / denominator)
        if 0.5 <= slope <= 100.0 and np.isfinite(slope):
            slopes.append(slope)
    if slopes:
        scale = float(np.median(slopes))
    else:
        scale = max(float(metadata["gr_scale"]), 3.0)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"{well.well_id}: invalid A710 recovered scale")
    center = (
        float(np.median(observed[valid])) if np.any(valid) else 0.0
    )
    horizontal = np.zeros_like(observed, dtype=np.float64)
    horizontal[valid] = (observed[valid] - center) / scale
    horizontal = np.clip(horizontal, -6.0, 6.0)

    expected = np.full(len(mismatch), np.nan, dtype=np.float64)
    residuals = []
    active_entries = 0
    for index, row in enumerate(mismatch):
        selected = valid & np.isfinite(row) & (np.abs(row) < 5.5)
        if int(np.sum(selected)) < 8:
            continue
        state_value = float(np.median(horizontal[selected] - row[selected]))
        expected[index] = state_value
        residual = horizontal[selected] - state_value - row[selected]
        residuals.append(residual)
        active_entries += len(residual)
    finite_states = np.flatnonzero(np.isfinite(expected))
    if len(finite_states) < 8:
        raise RuntimeError(f"{well.well_id}: insufficient A710 states")
    missing_states = np.flatnonzero(~np.isfinite(expected))
    expected[missing_states] = np.interp(
        missing_states,
        finite_states,
        expected[finite_states],
        left=expected[finite_states[0]],
        right=expected[finite_states[-1]],
    )
    expected = np.clip(expected, -6.0, 6.0)
    reconstruction_rmse = float(
        np.sqrt(
            np.mean(
                np.square(np.concatenate(residuals))
            )
        )
    )
    height, width = image.shape[1:]
    horizontal_plane = np.broadcast_to(
        horizontal[None, :], (height, width)
    )
    expected_plane = np.broadcast_to(
        expected[:, None], (height, width)
    )
    fixed_absolute = absolute_planes(
        well, cut, image, metadata, grid
    )[2:]
    planes = np.concatenate(
        (
            horizontal_plane[None].astype(np.float32),
            expected_plane[None].astype(np.float32),
            fixed_absolute,
        ),
        axis=0,
    ).astype(np.float32)
    if planes.shape != (8, height, width) or not np.isfinite(planes).all():
        raise RuntimeError(f"{well.well_id}: invalid A710 planes")
    return planes, {
        "recovered_scale": scale,
        "reconstruction_rmse": reconstruction_rmse,
        "active_entries": active_entries,
        "state_slopes": len(slopes),
    }


def patched_residual_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
    variant: str,
) -> np.ndarray:
    if variant == VARIANT:
        return calibrated_separable_planes(
            well, cut, image, metadata, grid
        )[0]
    return _PREVIOUS_RESIDUAL_PLANES(
        well, cut, image, metadata, grid, variant
    )


core.residual_planes = patched_residual_planes


def example_contract(
    well: Well,
    grid,
    synthetic: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(710) if synthetic else None
    image, _, metadata = core.make_alignment_example(
        well,
        well.anchor_index,
        grid,
        synthetic_rng=rng,
        synthetic_profile="forward_extreme",
        synthetic_hardness=0.7,
        coordinate_kind="tvt_delta",
    )
    planes, recovery = calibrated_separable_planes(
        well, well.anchor_index, image, metadata, grid
    )
    horizontal = planes[0]
    expected = planes[1]
    return {
        "synthetic": synthetic,
        "shape": list(planes.shape),
        "finite": bool(np.isfinite(planes).all()),
        "horizontal_state_constant": bool(
            np.array_equal(
                horizontal,
                np.broadcast_to(horizontal[:1], horizontal.shape),
            )
        ),
        "expected_column_constant": bool(
            np.array_equal(
                expected,
                np.broadcast_to(expected[:, :1], expected.shape),
            )
        ),
        "horizontal_nonconstant": float(np.ptp(horizontal[0])) > 0.1,
        "expected_nonconstant": float(np.ptp(expected[:, 0])) > 0.1,
        **recovery,
    }


def hidden_invariance(well: Well, grid) -> bool:
    records = []
    hidden = np.arange(len(well.tvt)) > well.anchor_index
    for mode in ("original", "reverse", "nan"):
        target = well.tvt.copy()
        if mode == "reverse":
            target[hidden] = target[hidden][::-1]
        elif mode == "nan":
            target[hidden] = np.nan
        modified = dataclasses.replace(well, tvt=target)
        image, _, metadata = core.make_alignment_example(
            modified,
            well.anchor_index,
            grid,
            coordinate_kind="tvt_delta",
        )
        records.append(
            calibrated_separable_planes(
                modified,
                well.anchor_index,
                image,
                metadata,
                grid,
            )[0]
        )
    return all(np.array_equal(records[0], value) for value in records[1:])


def run_gate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    raw.verify_frozen_inputs(args.aggregate_gate)
    identifiers, original = core.development_layout(args.data_root)
    development = np.flatnonzero(original < 4)
    wells = core.load_ids(
        identifiers, development, args.data_root, args.workers
    )
    training = [
        well
        for well, fold in zip(wells, original[development])
        if int(fold) != args.fold
    ]
    grid = core.config()
    contracts = [
        example_contract(training[0], grid, False),
        example_contract(training[0], grid, True),
    ]
    invariance = hidden_invariance(training[0], grid)
    capacity = core.fixed_gate(
        training,
        grid,
        args.fold,
        VARIANT,
        torch.device(args.device),
        args.overfit_steps,
    )
    contract_passed = invariance and all(
        record["shape"] == [8, len(grid.offsets), grid.length]
        and record["finite"]
        and record["horizontal_state_constant"]
        and record["expected_column_constant"]
        and record["horizontal_nonconstant"]
        and record["expected_nonconstant"]
        and float(record["recovered_scale"]) > 0.0
        and float(record["reconstruction_rmse"]) <= 0.15
        for record in contracts
    )
    result = {
        "schema_version": 1,
        "run_id": core.RUN_ID,
        "family": "calibrated_separable_gr_residual",
        "variant": VARIANT,
        "status": (
            "passed"
            if contract_passed and capacity["status"] == "passed"
            else "failed"
        ),
        "contracts": contracts,
        "hidden_target_invariant": invariance,
        "capacity": capacity,
        "seed": SEED,
        "core_sha256": raw.CORE_SHA256,
        "aggregate_gate_sha256": raw.AGGREGATE_GATE_SHA256,
        "wrapper_sha256": core.sha256(Path(__file__)),
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    core.write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise RuntimeError("A710 implementation gate failed")


def train(args: argparse.Namespace) -> None:
    raw.verify_frozen_inputs(args.aggregate_gate)
    core_args = argparse.Namespace(
        fold=args.fold,
        variant=VARIANT,
        output_dir=args.output_dir,
        data_root=args.data_root,
        d570=args.d570,
        aggregate_gate=args.aggregate_gate,
        device=args.device,
        workers=args.workers,
        loader_workers=args.loader_workers,
        batch_size=args.batch_size,
        repeats=args.repeats,
        overfit_only=False,
        overfit_steps=args.overfit_steps,
    )
    core.train(core_args)
    result_path = args.output_dir / "result.json"
    result = json.loads(result_path.read_text())
    if (
        result.get("variant") != VARIANT
        or result.get("specification", {}).get("seed") != SEED
        or result.get("source_sha256") != raw.CORE_SHA256
        or result.get("f4_loaded") is not False
        or result.get("confirmation_regroupings_loaded") is not False
    ):
        raise RuntimeError("A710 result provenance failed")
    core.write_json(
        args.output_dir / "wrapper_attestation.json",
        {
            "schema_version": 1,
            "run_id": core.RUN_ID,
            "family": "calibrated_separable_gr_residual",
            "variant": VARIANT,
            "seed": SEED,
            "wrapper_sha256": core.sha256(Path(__file__)),
            "core_sha256": raw.CORE_SHA256,
            "aggregate_gate_sha256": raw.AGGREGATE_GATE_SHA256,
            "result_sha256": core.sha256(result_path),
            "audit_fold_loaded": False,
            "confirmation_regroupings_loaded": False,
            "f4_loaded": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=range(4))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--d570", type=Path, default=core.DEFAULT_D570)
    parser.add_argument(
        "--aggregate-gate",
        type=Path,
        default=core.DEFAULT_AGGREGATE_GATE,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overfit-steps", type=int, default=1200)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.gate_only:
        run_gate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
