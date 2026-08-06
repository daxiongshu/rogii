from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from limited_query_cv.evaluate_v5_run25_c016_promotion import (
    bootstrap_gain,
    grouping_audit,
    rmse,
)
from rogii.data import load_well


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit"
)
ORIGINAL_OOF = RUN_ROOT / "promotion_oof.npz"
COMPONENTS = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
    "v20_components/fold3.npz"
)
POSTERIOR_VOLUME = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_006_goal_050/"
    "protected_posterior_volume_fold3.npz"
)
POSTERIOR_CHECKPOINT = (
    RUN_ROOT
    / "roles/outer_3/posterior_posterior_physical_b16/epoch020.pt"
)
POSTERIOR_PREDICTION = POSTERIOR_CHECKPOINT.with_name(
    "epoch020_prediction.npz"
)
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
KERNEL_SOURCE = (
    ROOT / "deploy/kaggle/c016-kernel/infer.py"
)
EXPECTED = {
    "original_oof": (
        "e319f320defbf0a042d79170adcbb72e87d2a4853d6c5370c43d3d08ba2c87d3"
    ),
    "posterior_checkpoint": (
        "9f76986c29203837aa6e88b63097fa29a6e52c22bcda2c406bf9c05aa1eaa983"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_kernel_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "c016_production_infer", KERNEL_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load C016 production inference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_npz.exists() or args.output_json.exists():
        raise FileExistsError("refusing to overwrite legal repair outputs")
    if (
        sha256(ORIGINAL_OOF) != EXPECTED["original_oof"]
        or sha256(POSTERIOR_CHECKPOINT)
        != EXPECTED["posterior_checkpoint"]
    ):
        raise RuntimeError("immutable C016 source hash changed")

    infer = load_kernel_module()
    device = torch.device(args.device)
    config = infer.SequenceConfig(state_radius=120.0, state_step=1.0)
    model = infer.AlignmentUNet(input_channels=40, base=16).to(device)
    model.load_state_dict(
        torch.load(
            POSTERIOR_CHECKPOINT,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    with np.load(COMPONENTS, allow_pickle=False) as cache:
        component_ids = cache["well_ids"].astype(str)
        component_starts = cache["row_starts"].astype(np.int64)
        component_baseline = cache["baseline"].astype(np.float32)
        component_values = cache["components"].astype(np.float32)
    with np.load(POSTERIOR_VOLUME, allow_pickle=False) as cache:
        posterior_ids = cache["well_ids"].astype(str)
        posterior_probability = cache["posterior_probability"].astype(
            np.float16
        )
    with np.load(POSTERIOR_PREDICTION, allow_pickle=False) as cache:
        prediction_ids = cache["well_ids"].astype(str)
        prediction_starts = cache["row_starts"].astype(np.int64)
        saved_correction = cache["correction_t100"].astype(np.float32)
    if not (
        np.array_equal(component_ids, posterior_ids)
        and np.array_equal(component_ids, prediction_ids)
        and np.array_equal(component_starts, prediction_starts)
    ):
        raise RuntimeError("outer-3 posterior artifact layout changed")

    legal_parts = []
    correction_delta_sse = 0.0
    correction_delta_max = 0.0
    for index, well_id in enumerate(component_ids):
        left, right = component_starts[index : index + 2]
        well = load_well(str(well_id), args.data_root)
        wide_image, _, wide_metadata = infer.make_alignment_example(
            well, well.anchor_index, config
        )
        physical, positions, prior, valid = infer.c016_residual_image(
            well,
            component_baseline[left:right],
            component_values[:, left:right],
            wide_image,
            wide_metadata,
            config,
        )
        correction = infer.c016_posterior_correction(
            model,
            physical,
            posterior_probability[index],
            positions,
            prior,
            valid,
            component_baseline[left:right],
            well,
            config,
            device,
        )
        delta = (
            correction.astype(np.float64)
            - saved_correction[left:right].astype(np.float64)
        )
        correction_delta_sse += float(delta @ delta)
        correction_delta_max = max(
            correction_delta_max, float(np.max(np.abs(delta)))
        )
        legal_parts.append(correction)
        if (index + 1) % 20 == 0 or index + 1 == len(component_ids):
            print(
                json.dumps(
                    {
                        "stage": "legal_posterior_mask_repair",
                        "wells": index + 1,
                        "total_wells": len(component_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    legal_correction = np.concatenate(legal_parts).astype(np.float32)

    with np.load(ORIGINAL_OOF, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    well_ids = arrays["well_ids"].astype(str)
    folds = arrays["folds"].astype(np.int8)
    starts = arrays["row_starts"].astype(np.int64)
    baseline = arrays["baseline"].astype(np.float32)
    truth = arrays["truth"].astype(np.float32)
    repaired_prediction = arrays["prediction"].astype(np.float64).copy()
    repaired_posterior = arrays[
        "protected_posterior_incremental"
    ].astype(np.float64).copy()
    lookup = {well_id: index for index, well_id in enumerate(well_ids)}
    for local_index, well_id in enumerate(component_ids):
        local_left, local_right = component_starts[
            local_index : local_index + 2
        ]
        global_index = lookup[str(well_id)]
        global_left, global_right = starts[
            global_index : global_index + 2
        ]
        if (
            int(folds[global_index]) != 3
            or global_right - global_left != local_right - local_left
        ):
            raise RuntimeError(f"{well_id}: global outer-3 layout changed")
        legal = legal_correction[local_left:local_right].astype(
            np.float64
        )
        saved = saved_correction[local_left:local_right].astype(
            np.float64
        )
        repaired_prediction[global_left:global_right] += (
            0.05 * (legal - saved)
        )
        repaired_posterior[global_left:global_right] += (
            0.20 * (legal - saved)
        )
    repaired_prediction = repaired_prediction.astype(np.float32)
    repaired_posterior = repaired_posterior.astype(np.float32)

    with np.load(LAYOUT, allow_pickle=False) as layout:
        grouping_values = {
            "original": layout["original"].astype(np.int8),
            "balanced": layout["stratified"].astype(np.int8),
            "independent": layout["independent"].astype(np.int8),
        }
    fold_records = []
    for fold in range(5):
        mask = np.repeat(folds == fold, np.diff(starts))
        before = rmse(baseline[mask], truth[mask])
        after = rmse(repaired_prediction[mask], truth[mask])
        fold_records.append(
            {
                "fold": fold,
                "before_rmse": before,
                "after_rmse": after,
                "gain": before - after,
                "rows": int(mask.sum()),
                "wells": int(np.sum(folds == fold)),
            }
        )
    grouping = {
        name: grouping_audit(
            repaired_prediction,
            baseline,
            truth,
            starts,
            groups,
        )
        for name, groups in grouping_values.items()
    }
    bootstrap = bootstrap_gain(
        repaired_prediction, baseline, truth, starts
    )
    score = rmse(repaired_prediction, truth)
    baseline_score = rmse(baseline, truth)
    gain = baseline_score - score
    maximum_regression = max(
        audit["maximum_held_group_regression"]
        for audit in grouping.values()
    )
    strong_clauses = {
        "pooled_gain_at_least_0p03": gain >= 0.03,
        "at_least_four_original_folds_improve": (
            sum(record["gain"] > 0.0 for record in fold_records) >= 4
        ),
        "all_grouping_gains_nonnegative": all(
            audit["gain"] >= 0.0 for audit in grouping.values()
        ),
        "maximum_held_group_regression_at_most_0p01": (
            maximum_regression <= 0.01
        ),
    }
    f4_veto = fold_records[4]["gain"] >= 0.0
    accepted = bool(all(strong_clauses.values()) and f4_veto)
    np.savez_compressed(
        args.output_npz,
        well_ids=well_ids,
        folds=folds,
        row_starts=starts,
        baseline=baseline,
        truth=truth,
        D570=arrays["D570"],
        protected_posterior_incremental=repaired_posterior,
        protected_mode_bank=arrays["protected_mode_bank"],
        prediction=repaired_prediction,
        legal_geometric_posterior_mask=np.asarray(True),
        accepted=np.asarray(accepted),
    )
    payload = {
        "schema_version": 1,
        "run_id": "v5_batch2_run_025_goal_050",
        "stage": "post_promotion_production_parity_repair",
        "status": (
            "legal_repair_complete_promotion_retained"
            if accepted
            else "legal_repair_complete_promotion_invalidated"
        ),
        "repair": {
            "issue": (
                "Historical posterior interpolation reused the target-derived "
                "classification-validity mask on outer-held predictions."
            ),
            "production_rule": (
                "Use only the label-free sampled geometry validity mask."
            ),
            "recipe_or_weight_changed": False,
            "post_reveal_tuning_performed": False,
            "all_outer3_wells_regenerated": True,
            "posterior_correction_delta_rmse": float(
                np.sqrt(correction_delta_sse / len(legal_correction))
            ),
            "posterior_correction_delta_max_abs": correction_delta_max,
        },
        "original_oof_rmse": rmse(arrays["prediction"], truth),
        "legal_oof_rmse": score,
        "baseline_rmse": baseline_score,
        "legal_oof_gain": gain,
        "folds": fold_records,
        "groupings": grouping,
        "maximum_held_group_regression": maximum_regression,
        "bootstrap": bootstrap,
        "strong_gate": {
            "clauses": strong_clauses,
            "passed": all(strong_clauses.values()),
        },
        "f4_nonnegative_gain_veto_passed": f4_veto,
        "accepted": accepted,
        "original_oof": str(ORIGINAL_OOF.relative_to(ROOT)),
        "original_oof_sha256": EXPECTED["original_oof"],
        "legal_oof": str(args.output_npz.resolve().relative_to(ROOT)),
        "legal_oof_sha256": sha256(args.output_npz),
        "posterior_checkpoint_sha256": EXPECTED[
            "posterior_checkpoint"
        ],
        "kernel_source_sha256": sha256(KERNEL_SOURCE),
        "competition_submission_authorized": False,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not accepted:
        raise RuntimeError("legal posterior repair invalidated promotion")


if __name__ == "__main__":
    main()
