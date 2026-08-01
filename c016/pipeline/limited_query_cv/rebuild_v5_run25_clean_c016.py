from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv import (
    evaluate_v5_run8_clean_multimodel_ridge as ridge,
)


RUN_ID = "v5_batch2_run_025_goal_050"
FAMILY = "clean_c016_without_quarantined_d072"
RUN_ROOT = Path("limited_query_cv/runs") / RUN_ID
SOURCE_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
D570 = SOURCE_ROOT / "d209_amplitude_clean.npz"
D570_SHA256 = (
    "5e3b6ee41f8705cc6b0044b36255bf8e8b3b6c4752c2afa9f6049d83f77cd310"
)
POSTERIOR = SOURCE_ROOT / "protected_posterior_clean.npz"
POSTERIOR_SHA256 = (
    "999123a3392b7af9d10e2a02ef26bd94b8a732f95b109a2a073a05b2b722c43c"
)
MODE_BANK = SOURCE_ROOT / "protected_mode_bank_clean.npz"
MODE_BANK_SHA256 = (
    "6240d9077188be3efcc2299f97fd59a879cac0a095201168e5250d7dbf341aee"
)
MODE_BANK_INVALIDATION = Path(
    "limited_query_cv/v5_batch2_mode_bank_strat_lineage_invalidation.json"
)
QUARANTINE = Path(
    "limited_query_cv/v5_batch2_purece_strat_lineage_invalidation.json"
)
QUARANTINE_SHA256 = (
    "85cccc9982398b69f0210dc60c9177ac1d0fbabb5578ede60a967e0e345be2ec"
)
SAFE_D072_RESULT = RUN_ROOT / "original_safe_d072_result.json"
SAFE_D072_RESULT_SHA256 = (
    "a0ac4bb11119b8a9146255acecbb42e391720cbba803220b0498ca82d3e146b0"
)
COMPONENTS = (
    "protected_posterior_incremental",
    "protected_mode_bank",
)
AMPLITUDE = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"{path}: expected {expected}, observed {observed}"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def load_sources(
    posterior_path: Path = POSTERIOR,
    posterior_sha256: str = POSTERIOR_SHA256,
) -> dict[str, np.ndarray]:
    if MODE_BANK.exists() and sha256(MODE_BANK) == MODE_BANK_SHA256:
        raise RuntimeError(
            "The historical protected mode bank contains the mismatched "
            "stratified-fold parent and is quarantined. Use "
            "finalize_v5_run25_original_safe_c016.py; see "
            f"{MODE_BANK_INVALIDATION}."
        )
    expected = (
        (D570, D570_SHA256),
        (posterior_path, posterior_sha256),
        (MODE_BANK, MODE_BANK_SHA256),
        (QUARANTINE, QUARANTINE_SHA256),
        (SAFE_D072_RESULT, SAFE_D072_RESULT_SHA256),
    )
    for path, expected_hash in expected:
        require_hash(path, expected_hash)
    source_specs = (
        ("D570", D570, "prediction"),
        (
            "protected_posterior_incremental",
            posterior_path,
            "incremental_after_D570",
        ),
        ("protected_mode_bank", MODE_BANK, "prediction"),
    )
    reference = None
    loaded: dict[str, np.ndarray] = {}
    for name, path, prediction_key in source_specs:
        with np.load(path, allow_pickle=False) as archive:
            for firewall in (
                "audit_fold_loaded",
                "confirmation_regroupings_loaded",
                "f4_loaded",
            ):
                if (
                    firewall in archive.files
                    and bool(archive[firewall])
                ):
                    raise ValueError(
                        f"{name}: protected firewall failed"
                    )
            layout = {
                "well_ids": archive["well_ids"].astype(str),
                "folds": archive["folds"].astype(np.int8),
                "row_starts": archive["row_starts"].astype(
                    np.int64
                ),
                "baseline": archive["baseline"].astype(np.float32),
                "truth": archive["truth"].astype(np.float32),
            }
            if reference is None:
                reference = layout
            elif any(
                not np.array_equal(reference[key], layout[key])
                for key in reference
            ):
                raise ValueError(f"{name}: common layout changed")
            prediction = archive[prediction_key].astype(np.float32)
            if (
                prediction.shape != layout["truth"].shape
                or not np.isfinite(prediction).all()
            ):
                raise ValueError(f"{name}: invalid prediction")
            loaded[name] = prediction
    if reference is None:
        raise RuntimeError("no clean C016 sources")
    if set(reference["folds"].astype(int).tolist()) != {
        0,
        1,
        2,
        3,
    }:
        raise ValueError("expected F0-F3 only")
    safe_d072 = json.loads(
        SAFE_D072_RESULT.read_text(encoding="utf-8")
    )
    if (
        safe_d072["status"]
        != "complete_selection_clean_F0_F3"
        or float(safe_d072["gain_versus_v20"]) >= 0.0
        or bool(safe_d072["audit_fold_loaded"])
        or bool(safe_d072["confirmation_regroupings_loaded"])
    ):
        raise ValueError("safe D072 rejection record changed")
    loaded.update(reference)
    loaded["parent"] = loaded["D570"].copy()
    del loaded["D570"]
    return loaded


def configure_ridge() -> None:
    ridge.RUN_ID = RUN_ID
    ridge.COMPONENTS = COMPONENTS


def role_mask(
    folds: np.ndarray,
    row_starts: np.ndarray,
    role: int,
) -> np.ndarray:
    return ridge.row_mask(folds, row_starts, {role})


def dual_result(
    payload: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    baseline = payload["baseline"].astype(np.float64)
    truth = payload["truth"].astype(np.float64)
    parent = payload["parent"].astype(np.float64)
    folds = payload["folds"].astype(np.int8)
    row_starts = payload["row_starts"].astype(np.int64)
    records = []
    for held in range(4):
        mask = role_mask(folds, row_starts, held)
        records.append(
            {
                "held_fold": held,
                "selector_folds": [
                    fold for fold in range(4) if fold != held
                ],
                "selected_share": 0.0,
                "held_d570_rmse": rmse(
                    parent[mask], truth[mask]
                ),
                "held_selected_rmse": rmse(
                    parent[mask], truth[mask]
                ),
                "held_incremental_gain": 0.0,
                "reason": (
                    "D072 removed after every original-fold-safe "
                    "replacement was negative"
                ),
            }
        )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "clean_dual_parent_selector",
        "variant": "D072_removed_exact_D570_fallback",
        "status": "complete_selection_clean_F0_F3",
        "source_sha256": sha256(Path(__file__)),
        "immutable_inputs": {"D570_sha256": D570_SHA256},
        "removed_input": {
            "name": "D072",
            "quarantine_record": str(QUARANTINE),
            "quarantine_record_sha256": QUARANTINE_SHA256,
            "safe_replacement_result": str(SAFE_D072_RESULT),
            "safe_replacement_result_sha256": (
                SAFE_D072_RESULT_SHA256
            ),
        },
        "v20_rmse": rmse(baseline, truth),
        "d570_rmse": rmse(parent, truth),
        "selected_rmse": rmse(parent, truth),
        "gain_versus_v20": rmse(baseline, truth)
        - rmse(parent, truth),
        "incremental_gain_versus_d570": 0.0,
        "positive_incremental_folds": 0,
        "fold_records": records,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "selection_clean_nested_selector_run": True,
        "forbidden_stratified_parent_loaded": False,
    }
    output = {
        "well_ids": payload["well_ids"].astype(str),
        "folds": folds,
        "row_starts": row_starts,
        "baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "d570": parent.astype(np.float32),
        "prediction": parent.astype(np.float32),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
        "selection_clean_nested_selector_run": np.asarray(True),
        "forbidden_stratified_parent_loaded": np.asarray(False),
    }
    return result, output


def half_result(
    ridge_result: dict[str, Any],
    ridge_output: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parent = ridge_output["parent"].astype(np.float64)
    full_ridge = ridge_output["prediction"].astype(np.float64)
    baseline = ridge_output["baseline"].astype(np.float64)
    truth = ridge_output["truth"].astype(np.float64)
    prediction = parent + AMPLITUDE * (full_ridge - parent)
    folds = ridge_output["folds"].astype(np.int8)
    row_starts = ridge_output["row_starts"].astype(np.int64)
    records = []
    for held in range(4):
        mask = role_mask(folds, row_starts, held)
        records.append(
            {
                "held_fold": held,
                "parent_rmse": rmse(parent[mask], truth[mask]),
                "selected_rmse": rmse(
                    prediction[mask], truth[mask]
                ),
                "incremental_gain": (
                    rmse(parent[mask], truth[mask])
                    - rmse(prediction[mask], truth[mask])
                ),
            }
        )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "clean_multimodel_bounded_ridge",
        "variant": "fixed_half_amplitude_D072_removed",
        "status": "complete_selection_clean_F0_F3",
        "amplitude": AMPLITUDE,
        "source_sha256": sha256(Path(__file__)),
        "v20_rmse": rmse(baseline, truth),
        "parent_rmse": rmse(parent, truth),
        "full_ridge_rmse": float(ridge_result["selected_rmse"]),
        "selected_rmse": rmse(prediction, truth),
        "gain_versus_v20": rmse(baseline, truth)
        - rmse(prediction, truth),
        "incremental_gain_versus_parent": rmse(parent, truth)
        - rmse(prediction, truth),
        "positive_incremental_folds": int(
            sum(item["incremental_gain"] > 0.0 for item in records)
        ),
        "fold_records": records,
        "selection_clean_nested_selector_run": True,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "forbidden_stratified_parent_loaded": False,
    }
    output = {
        "well_ids": ridge_output["well_ids"].astype(str),
        "folds": folds,
        "row_starts": row_starts,
        "baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "parent": parent.astype(np.float32),
        "full_ridge": full_ridge.astype(np.float32),
        "prediction": prediction.astype(np.float32),
        "amplitude": np.asarray(AMPLITUDE, dtype=np.float32),
        "selection_clean_nested_selector_run": np.asarray(True),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
        "forbidden_stratified_parent_loaded": np.asarray(False),
    }
    return result, output


def run_gate() -> dict[str, Any]:
    configure_ridge()
    real = load_sources()
    zero = ridge.compose(
        real["parent"],
        [real[name] for name in COMPONENTS],
        np.zeros(len(COMPONENTS), dtype=np.float64),
    )
    planted = ridge.synthetic_payload()
    selected, _ = ridge.select_recipe(planted, [0, 1, 2])
    if selected is None:
        raise AssertionError("planted clean C016 selector returned zero")
    weights = selected["weights"]
    changed = ridge.synthetic_payload()
    first, _ = ridge.select_recipe(changed, [1, 2, 3])
    held_mask = role_mask(
        changed["folds"], changed["row_starts"], 0
    )
    changed["truth"] = changed["truth"].copy()
    changed["truth"][held_mask] = np.linspace(
        -1e6, 1e6, int(held_mask.sum())
    )
    second, _ = ridge.select_recipe(changed, [1, 2, 3])
    corrupted_rejected = False
    with tempfile.TemporaryDirectory(
        prefix="run25_clean_c016_gate_"
    ) as temporary:
        corrupt = Path(temporary) / "corrupt.npz"
        shutil.copyfile(POSTERIOR, corrupt)
        with corrupt.open("r+b") as handle:
            handle.seek(-17, 2)
            byte = handle.read(1)
            handle.seek(-1, 1)
            handle.write(bytes([byte[0] ^ 1]))
        try:
            load_sources(corrupt, POSTERIOR_SHA256)
        except ValueError:
            corrupted_rejected = True
    checks = {
        "all_sources_rehashed": True,
        "quarantine_record_rehashed": True,
        "negative_safe_replacement_record_rehashed": True,
        "D072_absent_from_components": "D072" not in COMPONENTS,
        "forbidden_stratified_parent_absent": True,
        "common_layout_exact": True,
        "protected_flags_false": True,
        "exact_parent_zero_noop": np.array_equal(
            zero, real["parent"].astype(np.float64)
        ),
        "planted_weights_recovered": (
            abs(weights[COMPONENTS[0]] - 0.20) <= 0.01
            and abs(weights[COMPONENTS[1]] - 0.10) <= 0.01
        ),
        "excluded_held_truth_invariant": first == second,
        "corrupted_archive_rejected": corrupted_rejected,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "metric_free_gate_passed",
        "components": list(COMPONENTS),
        "ridge_grid": list(ridge.RIDGE_GRID),
        "maximum_component_weight": ridge.MAX_WEIGHT,
        "half_amplitude": AMPLITUDE,
        "checks": checks,
        "planted_selection": selected,
        "source_sha256": sha256(Path(__file__)),
        "held_target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }


def evaluate() -> dict[str, Any]:
    configure_ridge()
    payload = load_sources()
    dual_json, dual_output = dual_result(payload)
    ridge_json, ridge_output = ridge.evaluate(payload)
    ridge_json.update(
        {
            "run_id": RUN_ID,
            "family": FAMILY,
            "variant": "bounded_ridge_D072_removed",
            "source_sha256": sha256(Path(__file__)),
            "forbidden_stratified_parent_loaded": False,
            "removed_component": "D072",
        }
    )
    ridge_output[
        "forbidden_stratified_parent_loaded"
    ] = np.asarray(False)
    half_json, half_output = half_result(
        ridge_json, ridge_output
    )
    paths = {
        "dual_json": RUN_ROOT / "clean_dual_parent_result.json",
        "dual_npz": RUN_ROOT / "clean_dual_parent_clean.npz",
        "ridge_json": RUN_ROOT / "clean_multimodel_ridge_result.json",
        "ridge_npz": RUN_ROOT / "clean_multimodel_ridge_clean.npz",
        "half_json": RUN_ROOT / "clean_ridge_half_result.json",
        "half_npz": RUN_ROOT / "clean_ridge_half_clean.npz",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite clean C016 outputs")
    write_json(paths["dual_json"], dual_json)
    np.savez_compressed(paths["dual_npz"], **dual_output)
    write_json(paths["ridge_json"], ridge_json)
    np.savez_compressed(paths["ridge_npz"], **ridge_output)
    write_json(paths["half_json"], half_json)
    np.savez_compressed(paths["half_npz"], **half_output)
    return {
        "dual": {
            "rmse": dual_json["selected_rmse"],
            "gain_versus_v20": dual_json["gain_versus_v20"],
        },
        "full_ridge": {
            "rmse": ridge_json["selected_rmse"],
            "gain_versus_v20": ridge_json["gain_versus_v20"],
            "incremental_gain": ridge_json[
                "incremental_gain_versus_parent"
            ],
        },
        "clean_C016": {
            "rmse": half_json["selected_rmse"],
            "gain_versus_v20": half_json["gain_versus_v20"],
            "incremental_gain": half_json[
                "incremental_gain_versus_parent"
            ],
            "fold_incremental_gains": [
                item["incremental_gain"]
                for item in half_json["fold_records"]
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.gate:
        if args.output_json is None:
            parser.error("--output-json is required for --gate")
        result = run_gate()
        write_json(args.output_json, result)
    else:
        if args.output_json is not None:
            parser.error("--output-json is only used by --gate")
        result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
