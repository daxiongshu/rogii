from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv import (
    build_v5_run25_original_safe_mode_bank as safe_builder,
)
from limited_query_cv import (
    evaluate_v5_run6_protected_mode_clean as mode_clean,
)
from limited_query_cv import (
    evaluate_v5_run8_clean_multimodel_ridge as ridge,
)
from limited_query_cv import rebuild_v5_run25_clean_c016 as legacy_c016


RUN_ID = "v5_batch2_run_025_goal_050"
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
SAFE_BANK_PATTERN = RUN_ROOT / "original_safe_mode_bank_f{fold}.npz"
SAFE_RESULT_PATTERN = RUN_ROOT / "original_safe_mode_bank_f{fold}.json"
COMPONENTS = (
    "protected_posterior_incremental",
    "protected_mode_bank",
)


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


def load_safe_fold(fold: int) -> dict[str, np.ndarray | str]:
    bank_path = Path(str(SAFE_BANK_PATTERN).format(fold=fold))
    result_path = Path(str(SAFE_RESULT_PATTERN).format(fold=fold))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    builder_hash = sha256(Path(safe_builder.__file__))
    with np.load(bank_path, allow_pickle=False) as cache:
        parent_names = cache["parent_names"].astype(str)
        candidate_names = cache["candidate_names"].astype(str)
        if (
            int(cache["fold"]) != fold
            or result.get("fold") != fold
            or result.get("family") != safe_builder.FAMILY
            or result.get("parent_count") != 15
            or result.get("candidate_count") != 240
            or result.get("source_sha256") != builder_hash
            or str(cache["source_sha256"]) != builder_hash
            or result.get("bank_sha256") != sha256(bank_path)
            or result.get("forbidden_stratified_parent_loaded") is not False
            or bool(cache["forbidden_stratified_parent_loaded"])
            or bool(cache["audit_fold_loaded"])
            or bool(cache["confirmation_regroupings_loaded"])
            or bool(cache["held_horizontal_target_loaded_for_candidates"])
            or bool(cache["held_formation_columns_loaded"])
            or bool(cache["held_typewell_geology_loaded"])
            or len(parent_names) != 15
            or any("strat" in name for name in parent_names)
            or any(
                "strat" in name
                for name in cache["checkpoint_names"].astype(str)
            )
            or len(candidate_names) != 240
            or any(
                name.startswith("strat_") for name in candidate_names
            )
        ):
            raise RuntimeError(f"{bank_path}: safe lineage failed")
        starts = cache["row_starts"].astype(np.int64)
        corrections = cache["corrections"].astype(np.float32)
        metrics = cache["metrics"].astype(np.float32)
        shifted = cache["shifted_control_metrics"].astype(np.float32)
        return {
            "fold": np.asarray(fold, dtype=np.int8),
            "well_ids": cache["well_ids"].astype(str),
            "starts": starts,
            "baseline": cache["baseline"].astype(np.float32),
            "truth": cache["truth"].astype(np.float32),
            "metric_names": cache["metric_names"].astype(str),
            "selector_corrections": (
                mode_clean.assemble_selector_corrections(
                    metrics, corrections, starts
                )
            ),
            "shifted_selector_corrections": (
                mode_clean.assemble_selector_corrections(
                    shifted, corrections, starts
                )
            ),
            "bank_sha256": sha256(bank_path),
            "result_sha256": sha256(result_path),
        }


def combined_layout(
    folds: list[dict[str, np.ndarray | str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    well_ids: list[str] = []
    assignments: list[int] = []
    starts = [0]
    for fold, record in enumerate(folds):
        ids = np.asarray(record["well_ids"]).astype(str)
        local_starts = np.asarray(record["starts"]).astype(np.int64)
        well_ids.extend(ids.tolist())
        assignments.extend([fold] * len(ids))
        for left, right in zip(
            local_starts[:-1], local_starts[1:]
        ):
            starts.append(starts[-1] + int(right - left))
    return (
        np.asarray(well_ids),
        np.asarray(assignments, dtype=np.int8),
        np.asarray(starts, dtype=np.int64),
    )


def load_clean_components(
    well_ids: np.ndarray,
    folds: np.ndarray,
    row_starts: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    mode_prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    if sha256(D570) != D570_SHA256 or sha256(POSTERIOR) != POSTERIOR_SHA256:
        raise RuntimeError("clean C016 parent component changed")
    reference = {
        "well_ids": well_ids,
        "folds": folds,
        "row_starts": row_starts,
        "baseline": baseline,
        "truth": truth,
    }
    loaded: dict[str, np.ndarray] = {}
    for name, path, key in (
        ("parent", D570, "prediction"),
        (
            "protected_posterior_incremental",
            POSTERIOR,
            "incremental_after_D570",
        ),
    ):
        with np.load(path, allow_pickle=False) as cache:
            if (
                bool(cache["audit_fold_loaded"])
                or bool(cache["confirmation_regroupings_loaded"])
                or "f4_loaded" in cache.files
                and bool(cache["f4_loaded"])
            ):
                raise RuntimeError(f"{path}: protected flags changed")
            candidate_layout = {
                "well_ids": cache["well_ids"].astype(str),
                "folds": cache["folds"].astype(np.int8),
                "row_starts": cache["row_starts"].astype(np.int64),
                "baseline": cache["baseline"].astype(np.float32),
                "truth": cache["truth"].astype(np.float32),
            }
            if any(
                not np.array_equal(reference[field], candidate_layout[field])
                for field in reference
            ):
                raise RuntimeError(f"{path}: common layout changed")
            loaded[name] = cache[key].astype(np.float32)
    loaded.update(reference)
    loaded["protected_mode_bank"] = mode_prediction.astype(np.float32)
    return loaded


def main() -> None:
    outputs = {
        "mode_json": RUN_ROOT / "original_safe_mode_bank_result.json",
        "mode_npz": RUN_ROOT / "original_safe_mode_bank_clean.npz",
        "c016_json": RUN_ROOT / "original_safe_c016_result.json",
        "c016_npz": RUN_ROOT / "original_safe_c016_clean.npz",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite original-safe outputs")

    folds = [load_safe_fold(fold) for fold in range(4)]
    mode_prediction, mode_records = mode_clean.evaluate_variant(
        folds, shifted_control=False
    )
    baseline = np.concatenate(
        [np.asarray(record["baseline"]) for record in folds]
    ).astype(np.float32)
    truth = np.concatenate(
        [np.asarray(record["truth"]) for record in folds]
    ).astype(np.float32)
    well_ids, assignments, row_starts = combined_layout(folds)
    baseline_score = mode_clean.rmse(baseline, truth)
    mode_score = mode_clean.rmse(mode_prediction, truth)
    bank_hashes = [str(record["bank_sha256"]) for record in folds]
    result_hashes = [str(record["result_sha256"]) for record in folds]
    np.savez_compressed(
        outputs["mode_npz"],
        well_ids=well_ids,
        folds=assignments,
        row_starts=row_starts,
        baseline=baseline,
        truth=truth,
        prediction=mode_prediction.astype(np.float32),
        source_bank_sha256=np.asarray(bank_hashes),
        source_result_sha256=np.asarray(result_hashes),
        forbidden_stratified_parent_loaded=np.asarray(False),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
    )
    mode_result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": safe_builder.FAMILY,
        "variant": "selection_clean_fifteen_parent_mode_bank",
        "status": "complete_selection_clean_F0_F3",
        "baseline_rmse": baseline_score,
        "selected_rmse": mode_score,
        "gain": baseline_score - mode_score,
        "fold_records": mode_records,
        "source_bank_sha256": bank_hashes,
        "source_result_sha256": result_hashes,
        "output": str(outputs["mode_npz"]),
        "output_sha256": sha256(outputs["mode_npz"]),
        "source_sha256": sha256(Path(__file__)),
        "parent_count": 15,
        "candidate_count": 240,
        "removed_parent": "strat",
        "forbidden_stratified_parent_loaded": False,
        "selection_clean_nested_selector_run": True,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    write_json(outputs["mode_json"], mode_result)

    ridge.RUN_ID = RUN_ID
    ridge.COMPONENTS = COMPONENTS
    payload = load_clean_components(
        well_ids,
        assignments,
        row_starts,
        baseline,
        truth,
        mode_prediction,
    )
    ridge_result, ridge_output = ridge.evaluate(payload)
    half_result, half_output = legacy_c016.half_result(
        ridge_result, ridge_output
    )
    total_fold_records = []
    for held in range(4):
        mask = ridge.row_mask(assignments, row_starts, {held})
        v20_score = legacy_c016.rmse(
            baseline[mask], truth[mask]
        )
        selected_score = legacy_c016.rmse(
            half_output["prediction"][mask], truth[mask]
        )
        parent_score = legacy_c016.rmse(
            half_output["parent"][mask], truth[mask]
        )
        total_fold_records.append(
            {
                "held_fold": held,
                "v20_rmse": v20_score,
                "selected_rmse": selected_score,
                "gain_versus_v20": v20_score - selected_score,
                "incremental_gain_versus_D570": (
                    parent_score - selected_score
                ),
            }
        )
    half_output["forbidden_stratified_parent_loaded"] = np.asarray(
        False
    )
    np.savez_compressed(outputs["c016_npz"], **half_output)
    c016_result = {
        **half_result,
        "family": "original_fold_safe_c016",
        "variant": "fixed_half_amplitude_fifteen_parent_mode_bank",
        "status": "complete_selection_clean_F0_F3",
        "source_sha256": sha256(Path(__file__)),
        "mode_bank_result": str(outputs["mode_json"]),
        "mode_bank_result_sha256": sha256(outputs["mode_json"]),
        "mode_bank_archive": str(outputs["mode_npz"]),
        "mode_bank_archive_sha256": sha256(outputs["mode_npz"]),
        "output": str(outputs["c016_npz"]),
        "output_sha256": sha256(outputs["c016_npz"]),
        "total_fold_records": total_fold_records,
        "positive_total_gain_folds": int(
            sum(
                record["gain_versus_v20"] > 0.0
                for record in total_fold_records
            )
        ),
        "forbidden_stratified_parent_loaded": False,
    }
    write_json(outputs["c016_json"], c016_result)
    print(
        json.dumps(
            {
                "mode_bank_rmse": mode_score,
                "mode_bank_gain": baseline_score - mode_score,
                "c016_rmse": c016_result["selected_rmse"],
                "c016_gain": c016_result["gain_versus_v20"],
                "positive_total_gain_folds": c016_result[
                    "positive_total_gain_folds"
                ],
                "incremental_fold_gains": [
                    record["incremental_gain"]
                    for record in c016_result["fold_records"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
