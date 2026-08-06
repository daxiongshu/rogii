from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from limited_query_cv import (
    evaluate_v5_run6_protected_mode_bank as parent,
)
from rogii.alignment import AlignmentUNet
from rogii.data import DEFAULT_DATA_ROOT


RUN_ID = "v5_batch2_run_025_goal_050"
FAMILY = "original_fold_safe_protected_mode_bank"
CHECKPOINT_ROOT = Path("/geo/geo_new/rogii_v20_checkpoint_vault")
SAFE_PARENT_SPECS = tuple(
    spec for spec in parent.PARENTS if spec[0] != "strat"
)
EXPECTED_PARENT_COUNT = 15


def safe_parents(
    fold: int, device: torch.device
) -> list[parent.Parent]:
    if (
        len(SAFE_PARENT_SPECS) != EXPECTED_PARENT_COUNT
        or any(
            name == "strat" or "strat" in pattern
            for name, pattern, *_ in SAFE_PARENT_SPECS
        )
    ):
        raise RuntimeError("stratified parent entered safe mode-bank schema")
    models = []
    for name, pattern, scale, base, temperature in SAFE_PARENT_SPECS:
        checkpoint = CHECKPOINT_ROOT / pattern.format(fold=fold)
        if "strat" in checkpoint.name:
            raise RuntimeError("stratified checkpoint entered safe mode bank")
        model = AlignmentUNet(input_channels=19, base=base).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(
            parent.Parent(
                name=name,
                model=model,
                scale=scale,
                temperature=temperature,
                checkpoint=checkpoint,
                checkpoint_sha256=parent.sha256(checkpoint),
            )
        )
    return models


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    well_ids, starts, baseline, truth = parent.load_baseline(
        args.baseline_cache, args.fold
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        wells = list(
            pool.map(
                lambda well_id: parent.load_inference_well(
                    well_id, args.data_root
                ),
                well_ids.tolist(),
            )
        )
    device = torch.device(args.device)
    configurations = parent.configs()
    parents = safe_parents(args.fold, device)
    candidate_names = parent.candidate_schema(parents)
    if (
        len(candidate_names) != 240
        or np.any(np.char.startswith(candidate_names.astype(str), "strat_"))
    ):
        raise RuntimeError("safe candidate schema changed")

    correction_parts = []
    metric_parts = []
    shifted_metric_parts = []
    metric_names = None
    for well_index, well in enumerate(wells):
        left, right = starts[well_index : well_index + 2]
        correction = parent.candidate_corrections(
            well, parents, configurations, device
        )
        if correction.shape != (len(candidate_names), right - left):
            raise RuntimeError(f"{well.well_id}: candidate shape changed")
        paths = baseline[left:right][None].astype(np.float64) + correction
        metrics, names = parent.calibrated_metrics(
            well, paths, shifted_control=False
        )
        shifted_metrics, shifted_names = parent.calibrated_metrics(
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
    selector_corrections = parent.assemble_selector_corrections(
        metrics, corrections.astype(np.float32), starts
    )
    shifted_selector_corrections = parent.assemble_selector_corrections(
        shifted_metrics, corrections.astype(np.float32), starts
    )
    capacity = parent.capacity_screen(
        baseline, truth, selector_corrections, metric_names
    )
    shifted_capacity = parent.capacity_screen(
        baseline, truth, shifted_selector_corrections, metric_names
    )

    if args.bank_output.exists() or args.result_output.exists():
        raise FileExistsError("refusing to overwrite safe mode-bank output")
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
        parent_names=np.asarray([item.name for item in parents]),
        checkpoint_names=np.asarray(
            [str(item.checkpoint) for item in parents]
        ),
        checkpoint_sha256=np.asarray(
            [item.checkpoint_sha256 for item in parents]
        ),
        source_sha256=np.asarray(parent.sha256(Path(__file__))),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
        held_horizontal_target_loaded_for_candidates=np.asarray(False),
        held_formation_columns_loaded=np.asarray(False),
        held_typewell_geology_loaded=np.asarray(False),
        forbidden_stratified_parent_loaded=np.asarray(False),
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": "fifteen_original_fold_safe_parents",
        "status": "complete_F0_F3_capacity_diagnostic",
        "fold": args.fold,
        "held_wells": len(wells),
        "held_rows": len(baseline),
        "parent_count": len(parents),
        "parent_names": [item.name for item in parents],
        "candidate_count": len(candidate_names),
        "selector_count": len(metric_names),
        "removed_parent": "strat",
        "capacity_same_held_diagnostic_only": capacity,
        "shifted_hidden_gr_negative_control_same_held_diagnostic_only": (
            shifted_capacity
        ),
        "bank_output": str(args.bank_output),
        "bank_sha256": parent.sha256(args.bank_output),
        "source_sha256": parent.sha256(Path(__file__)),
        "forbidden_stratified_parent_loaded": False,
        "held_horizontal_target_loaded_for_candidates": False,
        "held_formation_columns_loaded": False,
        "held_typewell_geology_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "selection_clean_nested_selector_run": False,
        "elapsed_seconds": time.time() - started,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bank-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold": result["fold"],
                "parent_count": result["parent_count"],
                "candidate_count": result["candidate_count"],
                "bank_sha256": result["bank_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
