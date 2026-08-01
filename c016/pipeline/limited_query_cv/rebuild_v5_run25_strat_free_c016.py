from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv import (
    evaluate_v5_run6_protected_mode_clean as mode_clean,
)
from limited_query_cv import (
    evaluate_v5_run8_clean_multimodel_ridge as ridge,
)
from limited_query_cv import rebuild_v5_run25_clean_c016 as c016


RUN_ID = "v5_batch2_run_025_goal_050"
RUN_ROOT = Path("limited_query_cv/runs") / RUN_ID
SOURCE_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
MODE_BANK_PATTERN = SOURCE_ROOT / "protected_mode_bank_f{fold}.npz"
QUARANTINE = Path(
    "limited_query_cv/v5_batch2_purece_strat_lineage_invalidation.json"
)
QUARANTINE_SHA256 = (
    "85cccc9982398b69f0210dc60c9177ac1d0fbabb5578ede60a967e0e345be2ec"
)
STRAT_PREFIX = "strat_"


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


def load_strat_free_fold(fold: int) -> dict[str, np.ndarray | str]:
    path = Path(str(MODE_BANK_PATTERN).format(fold=fold))
    with np.load(path, allow_pickle=False) as cache:
        if (
            int(cache["fold"]) != fold
            or bool(cache["audit_fold_loaded"])
            or bool(cache["confirmation_regroupings_loaded"])
            or bool(cache["held_horizontal_target_loaded_for_candidates"])
            or bool(cache["held_formation_columns_loaded"])
            or bool(cache["held_typewell_geology_loaded"])
        ):
            raise RuntimeError(f"{path}: protected mode-bank firewall failed")
        names = cache["candidate_names"].astype(str)
        keep = ~np.char.startswith(names, STRAT_PREFIX)
        if int(np.sum(~keep)) != 16:
            raise RuntimeError(
                f"{path}: expected 16 stratified-parent candidates"
            )
        safe_names = names[keep]
        if np.any(np.char.startswith(safe_names, STRAT_PREFIX)):
            raise RuntimeError(f"{path}: stratified candidate survived")
        starts = cache["row_starts"].astype(np.int64)
        corrections = cache["corrections"][keep].astype(np.float32)
        metrics = cache["metrics"][:, keep, :].astype(np.float32)
        shifted = cache["shifted_control_metrics"][
            :, keep, :
        ].astype(np.float32)
        return {
            "fold": np.asarray(fold, dtype=np.int8),
            "well_ids": cache["well_ids"].astype(str),
            "starts": starts,
            "baseline": cache["baseline"].astype(np.float32),
            "truth": cache["truth"].astype(np.float32),
            "metric_names": cache["metric_names"].astype(str),
            "candidate_names": safe_names,
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
            "sha256": sha256(path),
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


def main() -> None:
    if sha256(QUARANTINE) != QUARANTINE_SHA256:
        raise RuntimeError("lineage quarantine record changed")
    mode_json_path = RUN_ROOT / "strat_free_mode_bank_result.json"
    mode_npz_path = RUN_ROOT / "strat_free_mode_bank_clean.npz"
    c016_json_path = RUN_ROOT / "strat_free_c016_result.json"
    c016_npz_path = RUN_ROOT / "strat_free_c016_clean.npz"
    if any(
        path.exists()
        for path in (
            mode_json_path,
            mode_npz_path,
            c016_json_path,
            c016_npz_path,
        )
    ):
        raise FileExistsError("refusing to overwrite strat-free C016 outputs")

    folds = [load_strat_free_fold(fold) for fold in range(4)]
    prediction, fold_records = mode_clean.evaluate_variant(
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
    mode_score = mode_clean.rmse(prediction, truth)
    source_hashes = [str(record["sha256"]) for record in folds]
    mode_payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "raw_history_row_competition",
        "variant": "protected_mode_bank_stratified_parent_removed",
        "status": "complete_selection_clean_F0_F3",
        "baseline_rmse": baseline_score,
        "selected_rmse": mode_score,
        "gain": baseline_score - mode_score,
        "fold_records": fold_records,
        "source_bank_sha256": source_hashes,
        "removed_parent": "strat",
        "removed_candidate_count_per_fold": 16,
        "remaining_candidate_count_per_fold": 240,
        "quarantine_record": str(QUARANTINE),
        "quarantine_record_sha256": QUARANTINE_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "selection_clean_nested_selector_run": True,
        "forbidden_stratified_parent_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    np.savez_compressed(
        mode_npz_path,
        well_ids=well_ids,
        folds=assignments,
        row_starts=row_starts,
        baseline=baseline,
        truth=truth,
        prediction=prediction.astype(np.float32),
        source_bank_sha256=np.asarray(source_hashes),
        forbidden_stratified_parent_loaded=np.asarray(False),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
    )
    mode_payload["output"] = str(mode_npz_path)
    mode_payload["output_sha256"] = sha256(mode_npz_path)
    write_json(mode_json_path, mode_payload)

    c016.configure_ridge()
    data = c016.load_sources()
    data["protected_mode_bank"] = prediction.astype(np.float32)
    ridge_result, ridge_output = ridge.evaluate(data)
    half_result, half_output = c016.half_result(
        ridge_result, ridge_output
    )
    total_fold_records = []
    for held in range(4):
        mask = c016.role_mask(assignments, row_starts, held)
        total_fold_records.append(
            {
                "held_fold": held,
                "v20_rmse": c016.rmse(
                    baseline[mask], truth[mask]
                ),
                "selected_rmse": c016.rmse(
                    half_output["prediction"][mask], truth[mask]
                ),
                "gain_versus_v20": (
                    c016.rmse(baseline[mask], truth[mask])
                    - c016.rmse(
                        half_output["prediction"][mask],
                        truth[mask],
                    )
                ),
                "incremental_gain_versus_D570": (
                    c016.rmse(
                        half_output["parent"][mask], truth[mask]
                    )
                    - c016.rmse(
                        half_output["prediction"][mask],
                        truth[mask],
                    )
                ),
            }
        )
    result = {
        **half_result,
        "family": "clean_c016_without_any_stratified_parent",
        "variant": "fixed_half_amplitude_strat_free_mode_bank",
        "status": "complete_selection_clean_F0_F3",
        "source_sha256": sha256(Path(__file__)),
        "mode_bank_result": str(mode_json_path),
        "mode_bank_result_sha256": sha256(mode_json_path),
        "mode_bank_archive": str(mode_npz_path),
        "mode_bank_archive_sha256": sha256(mode_npz_path),
        "total_fold_records": total_fold_records,
        "positive_total_gain_folds": int(
            sum(
                record["gain_versus_v20"] > 0.0
                for record in total_fold_records
            )
        ),
        "forbidden_stratified_parent_loaded": False,
    }
    half_output["forbidden_stratified_parent_loaded"] = np.asarray(
        False
    )
    np.savez_compressed(c016_npz_path, **half_output)
    result["output"] = str(c016_npz_path)
    result["output_sha256"] = sha256(c016_npz_path)
    write_json(c016_json_path, result)
    print(
        json.dumps(
            {
                "mode_bank_rmse": mode_score,
                "mode_bank_gain": baseline_score - mode_score,
                "c016_rmse": result["selected_rmse"],
                "c016_gain": result["gain_versus_v20"],
                "positive_total_gain_folds": result[
                    "positive_total_gain_folds"
                ],
                "incremental_fold_gains": [
                    record["incremental_gain"]
                    for record in result["fold_records"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
