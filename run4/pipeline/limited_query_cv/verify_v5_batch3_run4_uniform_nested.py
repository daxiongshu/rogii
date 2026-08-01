from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from limited_query_cv import promote_v5_batch3_run4_boosted_router as router
from limited_query_cv import train_v5_run6_local_v20_residual as parent
from limited_query_cv.assemble_v5_batch3_run4_uniform_nested import (
    DISABLED_INDICES,
)
from limited_query_cv.select_v5_batch3_run4_physics_nested import verify_roles


RUN_ID = "v5_batch3_run_004_goal_020"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--physics-prediction", type=Path, required=True)
    parser.add_argument("--physics-selector", type=Path, required=True)
    parser.add_argument("--physics-manifest", type=Path, required=True)
    parser.add_argument("--outer-prediction", type=Path, required=True)
    parser.add_argument("--outer-manifest", type=Path, required=True)
    parser.add_argument("--record-output", type=Path)
    args = parser.parse_args()

    role_records = verify_roles(args.physics_root)
    prediction_archives = 0
    for result_path in sorted(args.physics_root.glob("*/result.json")):
        result = load_json(result_path)
        for snapshot in result["snapshots"]:
            with np.load(snapshot["prediction"], allow_pickle=False) as archive:
                if (
                    "truth" in archive.files
                    or bool(archive["hidden_truth_read_for_prediction"])
                    or bool(archive["hidden_truth_stored"])
                    or bool(archive["protected_metrics_computed"])
                ):
                    raise RuntimeError(
                        f"{snapshot['prediction']}: prediction firewall failed"
                    )
            prediction_archives += 1
    if prediction_archives != 75:
        raise RuntimeError(
            f"expected 75 prediction archives, found {prediction_archives}"
        )
    physics_selector = load_json(args.physics_selector)
    physics_manifest = load_json(args.physics_manifest)
    outer_manifest = load_json(args.outer_manifest)
    if (
        physics_selector["run_id"] != RUN_ID
        or physics_selector["status"]
        != "complete_nested_selection_without_outer_metric"
        or physics_selector["F4_metric_recomputed"]
        or physics_manifest["run_id"] != RUN_ID
        or physics_manifest["status"]
        != "all_15_nested_roles_and_selected_outer_predictions_frozen"
        or physics_manifest["hidden_truth_stored"]
        or physics_manifest["F4_metric_recomputed"]
        or outer_manifest["run_id"] != RUN_ID
        or outer_manifest["status"]
        != "uniform_nested_outer_predictions_frozen_without_metric"
        or outer_manifest["hidden_truth_stored"]
        or outer_manifest["protected_metric_computed"]
        or outer_manifest["F4_metric_recomputed"]
    ):
        raise RuntimeError("uniform nested metadata contract failed")
    if (
        physics_manifest["record_count"] != len(role_records)
        or physics_manifest["selector_sha256"]
        != parent.sha256(args.physics_selector)
        or physics_manifest["prediction_sha256"]
        != parent.sha256(args.physics_prediction)
        or outer_manifest["prediction_sha256"]
        != parent.sha256(args.outer_prediction)
        or outer_manifest["physics_prediction_sha256"]
        != parent.sha256(args.physics_prediction)
        or outer_manifest["physics_selector_sha256"]
        != parent.sha256(args.physics_selector)
    ):
        raise RuntimeError("uniform nested checksum contract failed")

    with np.load(args.physics_prediction, allow_pickle=False) as physics:
        if (
            "truth" in physics.files
            or bool(physics["hidden_truth_stored"])
            or bool(physics["protected_metric_computed"])
            or bool(physics["F4_metric_recomputed"])
            or set(physics["folds"].astype(int).tolist()) != set(range(5))
            or not np.isfinite(physics["prediction"]).all()
        ):
            raise RuntimeError("selected physics prediction firewall failed")

    with np.load(router.PARENT, allow_pickle=False) as protected_parent:
        expected_ids = protected_parent["well_ids"].astype(str)
        expected_folds = protected_parent["folds"].astype(np.int64)
        expected_starts = protected_parent["row_starts"].astype(np.int64)
    with np.load(args.outer_prediction, allow_pickle=False) as outer:
        if (
            "truth" in outer.files
            or bool(outer["hidden_truth_stored"])
            or bool(outer["protected_metric_computed"])
            or bool(outer["F4_metric_recomputed"])
            or not np.array_equal(outer["well_ids"].astype(str), expected_ids)
            or not np.array_equal(outer["folds"].astype(np.int64), expected_folds)
            or not np.array_equal(
                outer["row_starts"].astype(np.int64), expected_starts
            )
            or not np.isfinite(outer["prediction"]).all()
        ):
            raise RuntimeError("uniform outer prediction firewall failed")
        base = outer["outer_base_weights"].astype(np.float64)
        route = outer["outer_route_weights"].astype(np.float64)
        # The nonnegative ridge solver can return a shared numerical residue
        # for an all-zero column. The corresponding corrections are installed
        # as exact zero by the frozen assembler, so tolerate only solver-scale
        # base residue and require routed additions to remain exact zero.
        disabled_base_residue = float(
            np.max(np.abs(base[:, DISABLED_INDICES]))
        )
        if (
            disabled_base_residue > 1e-5
            or np.any(route[:, DISABLED_INDICES] != 0.0)
        ):
            raise RuntimeError(
                "a uniformly disabled direction received material weight"
            )

    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "status": "uniform_nested_artifacts_verified_without_metric",
        "roles": 15,
        "records": len(role_records),
        "prediction_archives": prediction_archives,
        "wells": len(expected_ids),
        "rows": int(expected_starts[-1]),
        "disabled_base_solver_residue_max": disabled_base_residue,
        "disabled_route_weights_nonzero": 0,
        "physics_manifest_sha256": parent.sha256(args.physics_manifest),
        "outer_manifest_sha256": parent.sha256(args.outer_manifest),
        "validator_source_sha256": parent.sha256(Path(__file__)),
        "hidden_truth_stored": False,
        "protected_metric_computed": False,
        "F4_metric_recomputed": False,
        "kaggle_submission_authorized": False,
    }
    if args.record_output is not None:
        if args.record_output.exists():
            raise FileExistsError(args.record_output)
        args.record_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
