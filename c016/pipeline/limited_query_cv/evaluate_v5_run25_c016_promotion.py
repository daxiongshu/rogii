from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RUN_ID = "v5_batch2_run_025_goal_050"
ROOT = Path(__file__).resolve().parents[1]
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
V20 = Path("/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz")
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_RESAMPLES = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64)
                    - truth.astype(np.float64)
                )
            )
        )
    )


def grouping_audit(
    prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    starts: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    records = []
    for held in range(5):
        mask = np.repeat(groups == held, np.diff(starts))
        before = rmse(baseline[mask], truth[mask])
        after = rmse(prediction[mask], truth[mask])
        records.append(
            {
                "held_group": held,
                "before_rmse": before,
                "after_rmse": after,
                "gain": before - after,
                "rows": int(mask.sum()),
            }
        )
    return {
        "gain": rmse(baseline, truth) - rmse(prediction, truth),
        "maximum_held_group_regression": max(
            0.0, max(-record["gain"] for record in records)
        ),
        "held_groups": records,
    }


def bootstrap_gain(
    prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    starts: np.ndarray,
    selected_wells: np.ndarray | None = None,
) -> dict[str, Any]:
    indices = (
        np.arange(len(starts) - 1)
        if selected_wells is None
        else np.asarray(selected_wells, dtype=np.int64)
    )
    before_sse = []
    after_sse = []
    rows = []
    for index in indices:
        left, right = starts[index : index + 2]
        before_error = (
            baseline[left:right].astype(np.float64)
            - truth[left:right].astype(np.float64)
        )
        after_error = (
            prediction[left:right].astype(np.float64)
            - truth[left:right].astype(np.float64)
        )
        before_sse.append(float(before_error @ before_error))
        after_sse.append(float(after_error @ after_error))
        rows.append(int(right - left))
    before_sse = np.asarray(before_sse)
    after_sse = np.asarray(after_sse)
    rows = np.asarray(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(
        0, len(rows), size=(BOOTSTRAP_RESAMPLES, len(rows))
    )
    sampled_rows = np.sum(rows[samples], axis=1)
    before = np.sqrt(
        np.sum(before_sse[samples], axis=1) / sampled_rows
    )
    after = np.sqrt(
        np.sum(after_sse[samples], axis=1) / sampled_rows
    )
    gain = before - after
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "well_count": len(indices),
        "probability_gain_positive": float(np.mean(gain > 0.0)),
        "gain_quantile_0_10": float(np.quantile(gain, 0.10)),
        "gain_quantile_0_50": float(np.quantile(gain, 0.50)),
        "gain_quantile_0_90": float(np.quantile(gain, 0.90)),
    }


def verify_contract(
    preregistration_path: Path,
    confirmation_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    if (
        preregistration.get("status")
        != "preregistered_clean_c016_exception_audit"
        or preregistration.get("run_id") != RUN_ID
        or preregistration.get("evaluator_source_sha256")
        != sha256(Path(__file__))
        or not bool(
            preregistration.get("exception_scope", {}).get(
                "clean_c016_treated_as_ready_for_this_audit"
            )
        )
        or not bool(
            preregistration.get("promotion_gates", {}).get(
                "f4_nonnegative_gain_veto"
            )
        )
        or int(
            preregistration.get("nested_rerun", {}).get(
                "unique_pair_roles", -1
            )
        )
        != 10
        or int(
            preregistration.get("nested_rerun", {}).get(
                "outer_roles", -1
            )
        )
        != 5
    ):
        raise RuntimeError("C016 preregistration contract changed")
    for relative, expected in preregistration[
        "frozen_source_sha256"
    ].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"{relative}: frozen source changed")
    confirmation = json.loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    if (
        confirmation.get("status") != "confirmations_passed"
        or not bool(confirmation.get("all_confirmations_passed"))
        or float(
            confirmation["results"]["balanced"]["gain_vs_v20"]
        )
        < 0.10
        or float(
            confirmation["results"]["independent"]["gain_vs_v20"]
        )
        < 0.10
        or bool(confirmation.get("f4_loaded"))
        or bool(confirmation.get("f4_metric_computed"))
        or preregistration.get("confirmation_result")
        != str(confirmation_path.resolve().relative_to(ROOT))
        or confirmation.get("source_sha256")
        != preregistration.get("frozen_source_sha256", {}).get(
            "limited_query_cv/assemble_v5_run25_c016_nested.py"
        )
    ):
        raise RuntimeError("C016 confirmation contract changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status")
        != "frozen_ready_for_single_joint_reveal"
        or int(manifest.get("outer_folds", -1)) != 5
        or bool(manifest.get("truth_stored_in_prediction"))
        or bool(manifest.get("protected_metric_computed"))
        or bool(manifest.get("f4_metric_computed"))
    ):
        raise RuntimeError("C016 prediction manifest not reveal-ready")
    prediction_path = ROOT / manifest["prediction"]
    selector_path = ROOT / manifest["selector"]
    if (
        sha256(prediction_path) != manifest["prediction_sha256"]
        or sha256(selector_path) != manifest["selector_sha256"]
    ):
        raise RuntimeError("C016 sealed prediction changed")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    if (
        selector.get("status")
        != "complete_all_outer_predictions_frozen"
        or bool(selector.get("truth_stored_in_prediction"))
        or bool(selector.get("protected_metrics_emitted"))
        or bool(selector.get("f4_metric_computed"))
    ):
        raise RuntimeError("C016 sealed selector contract changed")
    return preregistration, manifest


def evaluate(
    preregistration_path: Path,
    confirmation_path: Path,
    manifest_path: Path,
    output_json: Path,
    output_npz: Path,
) -> None:
    if output_json.exists() or output_npz.exists():
        raise FileExistsError(
            "refusing a second C016 protected promotion reveal"
        )
    preregistration, manifest = verify_contract(
        preregistration_path, confirmation_path, manifest_path
    )
    prediction_path = ROOT / manifest["prediction"]
    with np.load(prediction_path, allow_pickle=False) as cache:
        if (
            bool(cache["hidden_truth_stored"])
            or bool(cache["f4_metric_computed"])
            or "truth" in cache.files
        ):
            raise RuntimeError("sealed C016 prediction firewall failed")
        well_ids = cache["well_ids"].astype(str)
        folds = cache["folds"].astype(np.int8)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        prediction = cache["prediction"].astype(np.float32)
        d570 = cache["D570"].astype(np.float32)
        posterior = cache[
            "protected_posterior_incremental"
        ].astype(np.float32)
        mode = cache["protected_mode_bank"].astype(np.float32)
    with np.load(LAYOUT, allow_pickle=False) as layout:
        layout_ids = layout["well_ids"].astype(str)
        layout_starts = layout["row_starts"].astype(np.int64)
        truth = layout["truth"].astype(np.float32)
        grouping_values = {
            "original": layout["original"].astype(np.int8),
            "balanced": layout["stratified"].astype(np.int8),
            "independent": layout["independent"].astype(np.int8),
        }
    with np.load(V20, allow_pickle=False) as cache:
        v20_ids = cache["well_ids"].astype(str)
        v20_starts = cache["row_starts"].astype(np.int64)
        v20 = cache["prediction"].astype(np.float32)
    if not (
        np.array_equal(well_ids, layout_ids)
        and np.array_equal(well_ids, v20_ids)
        and np.array_equal(starts, layout_starts)
        and np.array_equal(starts, v20_starts)
        and np.array_equal(folds, grouping_values["original"])
        and np.array_equal(baseline, v20)
        and prediction.shape == truth.shape
        and np.isfinite(prediction).all()
    ):
        raise RuntimeError("C016 protected layout changed")

    baseline_score = rmse(baseline, truth)
    score = rmse(prediction, truth)
    gain = baseline_score - score
    fold_records = []
    for fold in range(5):
        mask = np.repeat(folds == fold, np.diff(starts))
        before = rmse(baseline[mask], truth[mask])
        after = rmse(prediction[mask], truth[mask])
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
            prediction,
            baseline,
            truth,
            starts,
            groups,
        )
        for name, groups in grouping_values.items()
    }
    bootstrap = bootstrap_gain(
        prediction, baseline, truth, starts
    )
    f4_wells = np.flatnonzero(folds == 4)
    f4_bootstrap = bootstrap_gain(
        prediction,
        baseline,
        truth,
        starts,
        selected_wells=f4_wells,
    )
    f4_gain = float(fold_records[4]["gain"])
    maximum_regression = max(
        audit["maximum_held_group_regression"]
        for audit in grouping.values()
    )
    positive_inner = int(manifest["positive_inner_weight_roles"])
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
    small_clauses = {
        "pooled_gain_positive": gain > 0.0,
        "all_grouping_gains_positive": all(
            audit["gain"] > 0.0 for audit in grouping.values()
        ),
        "at_least_ten_positive_inner_weights": positive_inner >= 10,
        "maximum_held_group_regression_at_most_0p01": (
            maximum_regression <= 0.01
        ),
        "bootstrap_probability_at_least_0p80": (
            bootstrap["probability_gain_positive"] >= 0.80
        ),
        "bootstrap_q10_at_least_negative_0p005": (
            bootstrap["gain_quantile_0_10"] >= -0.005
        ),
    }
    strong = all(strong_clauses.values())
    small = all(small_clauses.values())
    f4_veto_passed = f4_gain >= 0.0
    accepted = bool((strong or small) and f4_veto_passed)
    np.savez_compressed(
        output_npz,
        well_ids=well_ids,
        folds=folds,
        row_starts=starts,
        baseline=baseline,
        truth=truth,
        D570=d570,
        protected_posterior_incremental=posterior,
        protected_mode_bank=mode,
        prediction=prediction,
        protected_reveal_complete=np.asarray(True),
        f4_metric_computed=np.asarray(True),
        accepted=np.asarray(accepted),
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_name": "cv-protocol-v5",
        "batch": 2,
        "stage": "joint_protected_outer_promotion_result",
        "status": (
            "complete_with_promotion"
            if accepted
            else "complete_no_promotion"
        ),
        "disclosure_label": (
            "selection-clean, development-exposed; F4 prospective"
        ),
        "candidate": "original_fold_safe_C016",
        "baseline_rmse": baseline_score,
        "oof_rmse": score,
        "oof_gain": gain,
        "folds": fold_records,
        "f4_audit": {
            "baseline_rmse": fold_records[4]["before_rmse"],
            "candidate_rmse": fold_records[4]["after_rmse"],
            "gain": f4_gain,
            "nonnegative_gain_veto_passed": f4_veto_passed,
            "bootstrap": f4_bootstrap,
        },
        "groupings": grouping,
        "maximum_held_group_regression": maximum_regression,
        "positive_fold_local_inner_weight_estimates": positive_inner,
        "bootstrap": bootstrap,
        "strong_gate": {
            "clauses": strong_clauses,
            "passed": strong,
        },
        "small_diverse_gate": {
            "clauses": small_clauses,
            "passed": small,
        },
        "f4_nonnegative_gain_veto_passed": f4_veto_passed,
        "accepted": accepted,
        "protected_outer_reveal_spent": True,
        "protected_outer_reveal_count": 1,
        "output_npz": str(output_npz.resolve().relative_to(ROOT)),
        "output_npz_sha256": sha256(output_npz),
        "prediction_manifest": str(
            manifest_path.resolve().relative_to(ROOT)
        ),
        "prediction_manifest_sha256": sha256(manifest_path),
        "preregistration": str(
            preregistration_path.resolve().relative_to(ROOT)
        ),
        "preregistration_sha256": sha256(preregistration_path),
        "confirmation": str(
            confirmation_path.resolve().relative_to(ROOT)
        ),
        "confirmation_sha256": sha256(confirmation_path),
        "exception_authorization": preregistration[
            "exception_authorization"
        ],
        "kaggle_dataset_authorized": False,
        "kaggle_kernel_authorized": False,
        "kaggle_submission_authorized": False,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "oof_rmse": score,
                "oof_gain": gain,
                "f4_gain": f4_gain,
                "strong_gate_passed": strong,
                "small_diverse_gate_passed": small,
                "f4_veto_passed": f4_veto_passed,
                "accepted": accepted,
                "result_sha256": sha256(output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration", type=Path, required=True
    )
    parser.add_argument(
        "--confirmation", type=Path, required=True
    )
    parser.add_argument(
        "--prediction-manifest", type=Path, required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()
    evaluate(
        args.preregistration,
        args.confirmation,
        args.prediction_manifest,
        args.output_json,
        args.output_npz,
    )


if __name__ == "__main__":
    main()
