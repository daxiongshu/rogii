from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from limited_query_cv import train_v5_run6_frozen_residual_rawstate as core
from limited_query_cv.train_v5_run6_absolute_state_continuation import (
    absolute_planes,
)
from rogii.data import DEFAULT_DATA_ROOT, Well


VARIANT = "delta_true_raw_absolute_b8"
SEED = 20262003
CORE_SHA256 = (
    "b4e117b6a027cce3879c384c2bdf79cfcf84dd27f3946a9aa034f32fdf6f21c1"
)
AGGREGATE_GATE_SHA256 = (
    "35bfeb2eb3c9deb48099b74c8fe7820eaafcff387eaa588202e0fe6e6edc0eb0"
)
core.VARIANTS[VARIANT] = {
    "extra_channels": 8,
    "base": 8,
    "seed": SEED,
}
_ORIGINAL_RESIDUAL_PLANES = core.residual_planes


def robust_normalize(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool) & np.isfinite(source)
    if not np.any(selected):
        return np.zeros_like(source, dtype=np.float32)
    center = float(np.median(source[selected]))
    mad = float(np.median(np.abs(source[selected] - center)))
    scale = max(
        1.4826 * mad,
        float(np.std(source[selected])),
        4.0,
    )
    normalized = np.zeros_like(source, dtype=np.float64)
    normalized[selected] = (source[selected] - center) / scale
    return np.clip(normalized, -6.0, 6.0).astype(np.float32)


def true_raw_absolute_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
) -> np.ndarray:
    height, width = image.shape[1:]
    valid_columns = np.asarray(metadata["valid_positions"]) > 0.5
    horizontal = robust_normalize(
        np.asarray(metadata["gr"], dtype=np.float64),
        valid_columns,
    )
    candidate_tvt = float(well.tvt[cut]) + grid.offsets
    typewell = np.interp(
        candidate_tvt,
        well.typewell_tvt,
        well.typewell_gr,
        left=np.nan,
        right=np.nan,
    )
    typewell = robust_normalize(typewell, np.isfinite(typewell))
    horizontal_plane = np.broadcast_to(
        horizontal[None, :], (height, width)
    )
    typewell_plane = np.broadcast_to(
        typewell[:, None], (height, width)
    )
    # A680's remaining six planes are already prospectively frozen and legal.
    fixed_absolute = absolute_planes(
        well, cut, image, metadata, grid
    )[2:]
    result = np.concatenate(
        (
            horizontal_plane[None].astype(np.float32),
            typewell_plane[None].astype(np.float32),
            fixed_absolute,
        ),
        axis=0,
    ).astype(np.float32)
    if result.shape != (8, height, width) or not np.isfinite(result).all():
        raise RuntimeError(f"{well.well_id}: invalid A700 raw planes")
    return result


def patched_residual_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
    variant: str,
) -> np.ndarray:
    if variant == VARIANT:
        return true_raw_absolute_planes(well, cut, image, metadata, grid)
    return _ORIGINAL_RESIDUAL_PLANES(
        well, cut, image, metadata, grid, variant
    )


core.residual_planes = patched_residual_planes


def verify_frozen_inputs(aggregate_gate: Path) -> dict[str, object]:
    core_path = Path(core.__file__)
    if core.sha256(core_path) != CORE_SHA256:
        raise RuntimeError("A700 frozen A680 core hash mismatch")
    if core.sha256(aggregate_gate) != AGGREGATE_GATE_SHA256:
        raise RuntimeError("A700 frozen aggregate-gate hash mismatch")
    return core.verify_aggregate_gate(aggregate_gate)


def axis_contract(
    well: Well,
    grid,
) -> dict[str, object]:
    image, _, metadata = core.make_alignment_example(
        well,
        well.anchor_index,
        grid,
        coordinate_kind="tvt_delta",
    )
    planes = true_raw_absolute_planes(
        well, well.anchor_index, image, metadata, grid
    )
    horizontal = planes[0]
    typewell = planes[1]
    horizontal_state_constant = bool(
        np.array_equal(
            horizontal,
            np.broadcast_to(horizontal[:1], horizontal.shape),
        )
    )
    typewell_column_constant = bool(
        np.array_equal(
            typewell,
            np.broadcast_to(typewell[:, :1], typewell.shape),
        )
    )
    horizontal_nonconstant = float(np.ptp(horizontal[0])) > 0.1
    typewell_nonconstant = float(np.ptp(typewell[:, 0])) > 0.1
    variants = []
    hidden = np.arange(len(well.tvt)) > well.anchor_index
    for mode in ("original", "reverse", "nan"):
        target = well.tvt.copy()
        if mode == "reverse":
            target[hidden] = target[hidden][::-1]
        elif mode == "nan":
            target[hidden] = np.nan
        modified = dataclasses.replace(well, tvt=target)
        changed_image, _, changed_metadata = core.make_alignment_example(
            modified,
            well.anchor_index,
            grid,
            coordinate_kind="tvt_delta",
        )
        variants.append(
            true_raw_absolute_planes(
                modified,
                well.anchor_index,
                changed_image,
                changed_metadata,
                grid,
            )
        )
    hidden_target_invariant = all(
        np.array_equal(variants[0], value) for value in variants[1:]
    )
    return {
        "shape": list(planes.shape),
        "horizontal_state_constant": horizontal_state_constant,
        "typewell_column_constant": typewell_column_constant,
        "horizontal_nonconstant": horizontal_nonconstant,
        "typewell_nonconstant": typewell_nonconstant,
        "hidden_target_invariant": hidden_target_invariant,
        "finite": bool(np.isfinite(planes).all()),
    }


def run_gate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    verify_frozen_inputs(args.aggregate_gate)
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
    contract = axis_contract(training[0], grid)
    capacity = core.fixed_gate(
        training,
        grid,
        args.fold,
        VARIANT,
        torch.device(args.device),
        args.overfit_steps,
    )
    contract_passed = (
        contract["shape"] == [8, len(grid.offsets), grid.length]
        and all(
            contract[name]
            for name in (
                "horizontal_state_constant",
                "typewell_column_constant",
                "horizontal_nonconstant",
                "typewell_nonconstant",
                "hidden_target_invariant",
                "finite",
            )
        )
    )
    result = {
        "schema_version": 1,
        "run_id": core.RUN_ID,
        "family": "true_separable_raw_gr_residual",
        "variant": VARIANT,
        "status": (
            "passed"
            if contract_passed and capacity["status"] == "passed"
            else "failed"
        ),
        "axis_contract": contract,
        "capacity": capacity,
        "seed": SEED,
        "core_sha256": CORE_SHA256,
        "aggregate_gate_sha256": AGGREGATE_GATE_SHA256,
        "wrapper_sha256": core.sha256(Path(__file__)),
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    core.write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise RuntimeError("A700 implementation gate failed")


def train(args: argparse.Namespace) -> None:
    verify_frozen_inputs(args.aggregate_gate)
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
        or result.get("source_sha256") != CORE_SHA256
        or result.get("f4_loaded") is not False
        or result.get("confirmation_regroupings_loaded") is not False
    ):
        raise RuntimeError("A700 result provenance failed")
    attestation = {
        "schema_version": 1,
        "run_id": core.RUN_ID,
        "family": "true_separable_raw_gr_residual",
        "variant": VARIANT,
        "seed": SEED,
        "wrapper_sha256": core.sha256(Path(__file__)),
        "core_sha256": CORE_SHA256,
        "aggregate_gate_sha256": AGGREGATE_GATE_SHA256,
        "result_sha256": core.sha256(result_path),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    core.write_json(args.output_dir / "wrapper_attestation.json", attestation)


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
