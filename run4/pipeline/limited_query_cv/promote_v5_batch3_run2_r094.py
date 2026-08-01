from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from limited_query_cv.prepare_v5_batch3_run2_combined_stack_stats import (
    PARENT,
    RUN1_ROOT,
    RUN2_ROOT,
    V20_ROOT,
    load_parent_fold,
    load_run1_candidates,
    load_run2_cache,
)
from limited_query_cv.train_v5_batch3_nested_linear_stack import (
    L2_GRID,
    TOTAL_SHARE_CAPS,
    aggregate,
    score,
)
from limited_query_cv.train_v5_batch3_run2_combined_reverse import (
    EXPECTED_REVERSE_NAMES,
)
from limited_query_cv.train_v5_batch3_run2_reverse_bank import (
    conditional_weight,
)
from limited_query_cv.v5_batch3_run2_historical_stack import (
    HISTORICAL_AMPLITUDES,
    extract_fold_prediction,
    load_historical_prediction,
    verify_historical_result,
)
from limited_query_cv.v5_batch3_run2_ramped_reverse import (
    RAMPED_REVERSE_TOTAL_CAPS,
    fit_ramped_reverse_bank,
    load_ramped_reverse_directions,
)
from limited_query_cv.v5_batch3_run2_reverse_residual_bank import (
    load_reverse_directions,
)
from limited_query_cv.v5_batch3_run2_secondary_reverse_bank import (
    load_secondary_reverse_directions,
)
from limited_query_cv.v5_batch3_run2_reverse_residual_bank import (
    REVERSE_TOTAL_CAPS,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "v5_batch3_run_002_goal_020"
FAMILY = "c016_buda_delayed_dip_cluster_surface"
RUN_ROOT = Path("limited_query_cv/runs") / RUN_ID
AUDIT_ROOT = RUN_ROOT / "promotion_audit"
AUTHORIZATION = (
    RUN_ROOT / "user_authorized_r094_exception_promotion_audit.json"
)
PREREGISTRATION = RUN_ROOT / "r094_promotion_preregistration.json"
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
HISTORICAL_INDEX = 28
REVERSE_INDICES = np.arange(29, 39, dtype=np.int64)
RAMPED_INDICES = np.arange(39, 49, dtype=np.int64)
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
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def cache_paths(
    fold: int,
    audit_root: Path,
) -> tuple[Path, Path, Path]:
    if fold < 4:
        return (
            RUN1_ROOT,
            RUN2_ROOT / f"surface_f{fold}_cache.npz",
            RUN2_ROOT / f"surface_selector_f{fold}_cache.npz",
        )
    return (
        audit_root / "candidates",
        audit_root / "candidates" / "surface_f4_cache.npz",
        audit_root / "candidates" / "surface_selector_f4_cache.npz",
    )


def load_fold_bank(
    fold: int,
    *,
    truth_allowed: bool,
    audit_root: Path = AUDIT_ROOT,
) -> dict[str, Any]:
    verify_historical_result()
    ids, parent_indices, parent_starts, clean = load_parent_fold(
        fold, PARENT
    )
    run1_root, raw_path, selector_path = cache_paths(fold, audit_root)
    names1, candidates1, records1 = load_run1_candidates(
        fold, parent_indices, run1_root
    )
    raw_names, raw, raw_record = load_run2_cache(
        raw_path, fold, ids
    )
    selector_names, selector, selector_record = load_run2_cache(
        selector_path, fold, ids
    )
    historical = extract_fold_prediction(
        load_historical_prediction(), fold, ids
    )
    primary_names, primary, primary_records = load_reverse_directions(
        fold, ids, clean
    )
    secondary_names, secondary, secondary_records = (
        load_secondary_reverse_directions(fold, ids, clean)
    )
    ramped_names, ramped, lengths, ramped_records = (
        load_ramped_reverse_directions(fold, ids, clean)
    )
    names = np.concatenate(
        [
            np.char.add("run1_", names1),
            np.char.add("surface_raw_", raw_names),
            np.char.add("surface_selector_", selector_names),
            np.asarray(["historical_clean_multimodel_ridge"]),
            primary_names,
            secondary_names,
            ramped_names,
        ]
    )
    ordinary = np.concatenate(
        [candidates1, raw, selector, historical[None]], axis=0
    )
    correction = np.concatenate(
        [
            ordinary - clean[None],
            primary,
            secondary,
            ramped,
        ],
        axis=0,
    )
    if (
        correction.shape != (49, len(clean))
        or HISTORICAL_INDEX != len(ordinary) - 1
        or list(names[REVERSE_INDICES]) != EXPECTED_REVERSE_NAMES
        or list(names[RAMPED_INDICES])
        != [f"ramped_{name}" for name in EXPECTED_REVERSE_NAMES]
        or int(np.sum(lengths)) != len(clean)
        or not np.isfinite(correction).all()
    ):
        raise RuntimeError("R094 candidate schema changed")
    v20_path = V20_ROOT / f"fold{fold}.npz"
    with np.load(v20_path, allow_pickle=False) as v20:
        if (
            int(v20["fold"]) != fold
            or bool(v20["audit_fold_loaded"])
            or bool(v20["layout_truth_array_accessed"])
            or not np.array_equal(v20["well_ids"].astype(str), ids)
        ):
            raise RuntimeError(f"fold {fold}: V20 firewall changed")
        baseline = v20["baseline"].astype(np.float64)
        truth = (
            v20["truth"].astype(np.float64)
            if truth_allowed
            else None
        )
    local_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(
                [
                    parent_starts[index + 1] - parent_starts[index]
                    for index in parent_indices
                ],
                dtype=np.int64,
            ),
        )
    )
    return {
        "fold": fold,
        "ids": ids,
        "parent_indices": parent_indices,
        "starts": local_starts,
        "clean": clean,
        "baseline": baseline,
        "truth": truth,
        "names": names,
        "correction": correction,
        "records": (
            records1
            + [raw_record, selector_record]
            + primary_records
            + secondary_records
            + ramped_records
        ),
    }


def empty_stats(group: int, names: np.ndarray) -> dict[str, Any]:
    return {
        "fold": group,
        "rows": 0,
        "clean_sse": 0.0,
        "v20_sse": 0.0,
        "cross": np.zeros(len(names), dtype=np.float64),
        "gram": np.zeros((len(names), len(names)), dtype=np.float64),
        "candidate_names": names,
    }


def add_rows(
    stats: dict[str, Any],
    correction: np.ndarray,
    clean: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
) -> None:
    error = clean - truth
    stats["rows"] += len(truth)
    stats["clean_sse"] += float(error @ error)
    baseline_error = baseline - truth
    stats["v20_sse"] += float(baseline_error @ baseline_error)
    stats["cross"] += correction @ error
    stats["gram"] += correction @ correction.T


def original_stats(
    *,
    audit_root: Path,
) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    stats = {}
    names = None
    for fold in range(5):
        bank = load_fold_bank(
            fold, truth_allowed=True, audit_root=audit_root
        )
        if names is None:
            names = bank["names"]
        elif not np.array_equal(names, bank["names"]):
            raise RuntimeError("outer candidate schema changed")
        record = empty_stats(fold, names)
        add_rows(
            record,
            bank["correction"],
            bank["clean"],
            bank["baseline"],
            bank["truth"],
        )
        stats[fold] = record
    return stats, names


def regrouped_development_stats(
    grouping: str,
) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    with np.load(LAYOUT, allow_pickle=False) as layout:
        layout_ids = layout["well_ids"].astype(str)
        raw_groups = layout[grouping].astype(np.int64)
    lookup = {
        well_id: int(group)
        for well_id, group in zip(layout_ids, raw_groups)
    }
    stats = None
    names = None
    for fold in range(4):
        bank = load_fold_bank(fold, truth_allowed=True)
        if names is None:
            names = bank["names"]
            stats = {
                group: empty_stats(group, names) for group in range(5)
            }
        for well_index, well_id in enumerate(bank["ids"]):
            left, right = bank["starts"][well_index : well_index + 2]
            group = lookup[str(well_id)]
            add_rows(
                stats[group],
                bank["correction"][:, left:right],
                bank["clean"][left:right],
                bank["baseline"][left:right],
                bank["truth"][left:right],
            )
    if set(stats) != set(range(5)) or any(
        record["rows"] == 0 for record in stats.values()
    ):
        raise RuntimeError(f"{grouping}: invalid confirmation grouping")
    return stats, names


def select_base_recipe(
    stats: dict[int, dict[str, Any]],
    training: list[int],
) -> dict[str, Any]:
    trials = []
    for amplitude in HISTORICAL_AMPLITUDES:
        for relative_l2 in L2_GRID:
            for original_cap in TOTAL_SHARE_CAPS:
                for reverse_cap in REVERSE_TOTAL_CAPS:
                    inner_sse = 0.0
                    inner_rows = 0
                    eligible = True
                    for inner in training:
                        fit = [group for group in training if group != inner]
                        cross, gram = aggregate(stats, fit)
                        weight = conditional_weight(
                            cross[:39],
                            gram[:39, :39],
                            HISTORICAL_INDEX,
                            REVERSE_INDICES,
                            amplitude,
                            relative_l2,
                            original_cap,
                            reverse_cap,
                        )
                        sse, rows = score(
                            stats[inner], augmented(weight)
                        )
                        eligible &= (
                            sse
                            <= float(stats[inner]["clean_sse"])
                            + 1e-7 * rows
                        )
                        inner_sse += sse
                        inner_rows += rows
                    trials.append(
                        {
                            "selected_historical_amplitude": amplitude,
                            "selected_relative_l2": relative_l2,
                            "selected_original_cap": original_cap,
                            "selected_reverse_cap": reverse_cap,
                            "eligible": bool(eligible),
                            "inner_rmse": float(
                                np.sqrt(inner_sse / inner_rows)
                            ),
                        }
                    )
    eligible = [trial for trial in trials if trial["eligible"]]
    if not eligible:
        return {
            "selected_historical_amplitude": 0.0,
            "selected_relative_l2": 1.0,
            "selected_original_cap": 0.0,
            "selected_reverse_cap": 0.0,
            "eligible": True,
            "inner_rmse": float(
                np.sqrt(
                    sum(float(stats[group]["clean_sse"]) for group in training)
                    / sum(int(stats[group]["rows"]) for group in training)
                )
            ),
            "exact_clean_C016_fallback": True,
        }
    return min(
        eligible,
        key=lambda row: (
            row["inner_rmse"],
            row["selected_reverse_cap"],
            row["selected_historical_amplitude"],
            row["selected_original_cap"],
            -row["selected_relative_l2"],
        ),
    )


def base_weight(
    stats: dict[int, dict[str, Any]],
    groups: list[int],
    recipe: dict[str, Any],
) -> np.ndarray:
    if recipe.get("exact_clean_C016_fallback", False):
        return np.zeros(39, dtype=np.float64)
    cross, gram = aggregate(stats, groups)
    return conditional_weight(
        cross[:39],
        gram[:39, :39],
        HISTORICAL_INDEX,
        REVERSE_INDICES,
        recipe["selected_historical_amplitude"],
        recipe["selected_relative_l2"],
        recipe["selected_original_cap"],
        recipe["selected_reverse_cap"],
    )


def augmented(
    base: np.ndarray,
    ramped: np.ndarray | None = None,
) -> np.ndarray:
    result = np.zeros(49, dtype=np.float64)
    result[:39] = base
    if ramped is not None:
        result[RAMPED_INDICES] = ramped
    return result


def conditional_ramped(
    cross: np.ndarray,
    gram: np.ndarray,
    base: np.ndarray,
    relative_l2: float,
    total_cap: float,
) -> np.ndarray:
    conditional_cross = (
        cross[RAMPED_INDICES]
        + gram[np.ix_(RAMPED_INDICES, np.arange(39))] @ base
    )
    return fit_ramped_reverse_bank(
        conditional_cross,
        gram[np.ix_(RAMPED_INDICES, RAMPED_INDICES)],
        relative_l2,
        total_cap,
    )


def select_outer(
    stats: dict[int, dict[str, Any]],
    outer: int,
) -> dict[str, Any]:
    training = [group for group in range(5) if group != outer]
    recipe = select_base_recipe(stats, training)
    ramped_trials = []
    for relative_l2 in L2_GRID:
        for total_cap in RAMPED_REVERSE_TOTAL_CAPS:
            inner_sse = 0.0
            inner_rows = 0
            eligible = True
            inner_positive = 0
            for inner in training:
                fit = [group for group in training if group != inner]
                base = base_weight(stats, fit, recipe)
                base_sse, rows = score(stats[inner], augmented(base))
                cross, gram = aggregate(stats, fit)
                ramped = conditional_ramped(
                    cross,
                    gram,
                    base,
                    relative_l2,
                    total_cap,
                )
                sse, _ = score(stats[inner], augmented(base, ramped))
                eligible &= sse <= base_sse + 1e-7 * rows
                inner_positive += int(np.sum(ramped) > 1e-10)
                inner_sse += sse
                inner_rows += rows
            ramped_trials.append(
                {
                    "relative_l2": relative_l2,
                    "total_cap": total_cap,
                    "eligible": bool(eligible),
                    "inner_rmse": float(
                        np.sqrt(inner_sse / inner_rows)
                    ),
                    "positive_inner_roles": inner_positive,
                }
            )
    selected = min(
        [trial for trial in ramped_trials if trial["eligible"]],
        key=lambda row: (
            row["inner_rmse"],
            row["total_cap"],
            -row["relative_l2"],
        ),
    )
    base = base_weight(stats, training, recipe)
    cross, gram = aggregate(stats, training)
    ramped = conditional_ramped(
        cross,
        gram,
        base,
        selected["relative_l2"],
        selected["total_cap"],
    )
    return {
        "outer": outer,
        "training_groups": training,
        "base_recipe": recipe,
        "ramped_relative_l2": selected["relative_l2"],
        "ramped_total_cap": selected["total_cap"],
        "inner_rmse": selected["inner_rmse"],
        "positive_inner_roles": selected["positive_inner_roles"],
        "weight": augmented(base, ramped),
    }


def selection_clean_summary(
    stats: dict[int, dict[str, Any]],
    names: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=5) as executor:
        roles = list(
            executor.map(
                partial(select_outer, stats),
                range(5),
            )
        )
    candidate_sse = 0.0
    clean_sse = 0.0
    v20_sse = 0.0
    group_records = []
    for role in roles:
        outer = role["outer"]
        held_sse, rows = score(stats[outer], role["weight"])
        clean = float(stats[outer]["clean_sse"])
        v20 = float(stats[outer]["v20_sse"])
        candidate_sse += held_sse
        clean_sse += clean
        v20_sse += v20
        group_records.append(
            {
                "held_group": outer,
                "rows": rows,
                "clean_C016_rmse": float(np.sqrt(clean / rows)),
                "v20_rmse": float(np.sqrt(v20 / rows)),
                "candidate_rmse": float(np.sqrt(held_sse / rows)),
                "gain_vs_clean_C016": float(
                    np.sqrt(clean / rows) - np.sqrt(held_sse / rows)
                ),
                "gain_vs_v20": float(
                    np.sqrt(v20 / rows) - np.sqrt(held_sse / rows)
                ),
            }
        )
    rows = sum(int(record["rows"]) for record in stats.values())
    summary = {
        "rows": rows,
        "clean_C016_rmse": float(np.sqrt(clean_sse / rows)),
        "v20_rmse": float(np.sqrt(v20_sse / rows)),
        "candidate_rmse": float(np.sqrt(candidate_sse / rows)),
        "gain_vs_clean_C016": float(
            np.sqrt(clean_sse / rows) - np.sqrt(candidate_sse / rows)
        ),
        "gain_vs_v20": float(
            np.sqrt(v20_sse / rows) - np.sqrt(candidate_sse / rows)
        ),
        "groups": group_records,
        "positive_inner_roles": int(
            sum(role["positive_inner_roles"] for role in roles)
        ),
    }
    serialized_roles = []
    for role in roles:
        serialized_roles.append(
            {
                key: value
                for key, value in role.items()
                if key != "weight"
            }
            | {
                "weights": [
                    {
                        "candidate": str(name),
                        "weight": float(weight),
                    }
                    for name, weight in zip(names, role["weight"])
                    if weight > 1e-10
                ]
            }
        )
    return roles, summary | {"roles": serialized_roles}


def verify_authorization_and_preregistration() -> dict[str, Any]:
    authorization = json.loads(AUTHORIZATION.read_text())
    preregistration = json.loads(PREREGISTRATION.read_text())
    if (
        authorization["status"] != "authorized_candidate_specific_exception"
        or authorization["candidate_specific_entry_threshold"] != 0.15
        or authorization["stored_run_goal_changed"] is not False
        or authorization["global_protocol_floor_changed"] is not False
        or authorization["f4_nonnegative_gain_veto_retained"] is not True
        or preregistration["status"]
        != "preregistered_r094_exception_promotion_audit"
        or preregistration["authorization_sha256"]
        != sha256(AUTHORIZATION)
        or preregistration["frozen_source_sha256"]
        != sha256(Path(__file__))
    ):
        raise RuntimeError("R094 exception contract changed")
    return preregistration


def confirm(output: Path) -> None:
    preregistration = verify_authorization_and_preregistration()
    results = {}
    for name, field in (
        ("balanced", "stratified"),
        ("independent", "independent"),
    ):
        stats, names = regrouped_development_stats(field)
        _, summary = selection_clean_summary(stats, names)
        results[name] = summary
    passed = all(
        result["gain_vs_clean_C016"] >= 0.10
        for result in results.values()
    )
    payload = {
        "schema_version": 1,
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "stage": "joint_F0_F3_confirmation_result",
        "status": (
            "confirmations_passed"
            if passed
            else "confirmations_failed"
        ),
        "minimum_each_gain_vs_clean_C016": 0.10,
        "results": results,
        "all_confirmations_passed": passed,
        "audit_fold_target_loaded": False,
        "f4_metric_computed": False,
        "protected_reveal_spent": False,
        "preregistration_sha256": sha256(PREREGISTRATION),
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "balanced_gain_vs_clean_C016": results["balanced"][
                    "gain_vs_clean_C016"
                ],
                "independent_gain_vs_clean_C016": results["independent"][
                    "gain_vs_clean_C016"
                ],
                "output_sha256": sha256(output),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise RuntimeError("R094 confirmations failed")


def candidate_files(audit_root: Path) -> list[Path]:
    files = [PARENT, Path(__file__)]
    for fold in range(5):
        run1_root, raw, selector = cache_paths(fold, audit_root)
        files.extend(
            sorted(
                (run1_root / f"f{fold}_highcap_structural").glob(
                    "shard*.npz"
                )
            )
        )
        files.extend([raw, selector])
    for path in (
        Path("limited_query_cv/v5_batch3_run2_surface.py"),
        Path("limited_query_cv/v5_batch3_highcap_path.py"),
        Path("limited_query_cv/v5_batch3_run2_ramped_reverse.py"),
        Path("limited_query_cv/v5_batch3_run2_reverse_residual_bank.py"),
        Path("limited_query_cv/v5_batch3_run2_secondary_reverse_bank.py"),
        Path(
            "limited_query_cv/"
            "prepare_v5_batch3_run2_combined_stack_stats.py"
        ),
        Path("limited_query_cv/train_v5_batch3_run2_combined_reverse.py"),
        Path("limited_query_cv/train_v5_batch3_run2_ramped_reverse_bank.py"),
    ):
        files.append(path)
    if any(not path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise RuntimeError(f"missing R094 candidate files: {missing}")
    return files


def freeze_candidates(audit_root: Path, output: Path) -> None:
    preregistration = verify_authorization_and_preregistration()
    confirmation = json.loads(
        (audit_root / "confirmation_result.json").read_text()
    )
    if (
        confirmation["status"] != "confirmations_passed"
        or confirmation["preregistration_sha256"]
        != sha256(PREREGISTRATION)
    ):
        raise RuntimeError("R094 confirmation contract changed")
    records = [
        {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in candidate_files(audit_root)
    ]
    payload = {
        "schema_version": 1,
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "all_target_free_candidate_predictions_frozen",
        "records": records,
        "record_count": len(records),
        "confirmation": str(audit_root / "confirmation_result.json"),
        "confirmation_sha256": sha256(
            audit_root / "confirmation_result.json"
        ),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "audit_fold_target_loaded": False,
        "f4_metric_computed": False,
        "protected_reveal_spent": False,
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "record_count": len(records),
                "output_sha256": sha256(output),
            },
            sort_keys=True,
        )
    )


def verify_candidate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if (
        manifest["status"]
        != "all_target_free_candidate_predictions_frozen"
        or manifest["confirmation_sha256"]
        != sha256(Path(manifest["confirmation"]))
        or manifest["preregistration_sha256"]
        != sha256(PREREGISTRATION)
        or manifest["audit_fold_target_loaded"] is not False
        or manifest["f4_metric_computed"] is not False
    ):
        raise RuntimeError("R094 candidate manifest changed")
    for record in manifest["records"]:
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"{path}: frozen candidate changed")
    return manifest


def assemble(
    candidate_manifest: Path,
    output_prediction: Path,
    output_selector: Path,
) -> None:
    if output_prediction.exists() or output_selector.exists():
        raise FileExistsError("refusing a second sealed R094 assembly")
    preregistration = verify_authorization_and_preregistration()
    verify_candidate_manifest(candidate_manifest)
    stats, names = original_stats(audit_root=AUDIT_ROOT)
    roles, summary = selection_clean_summary(stats, names)
    with np.load(PARENT, allow_pickle=False) as parent:
        well_ids = parent["well_ids"].astype(str)
        folds = parent["folds"].astype(np.int8)
        starts = parent["row_starts"].astype(np.int64)
        clean = parent["prediction"].astype(np.float32)
    prediction = clean.astype(np.float64).copy()
    for role in roles:
        fold = int(role["outer"])
        bank = load_fold_bank(
            fold, truth_allowed=False, audit_root=AUDIT_ROOT
        )
        corrected = (
            bank["clean"]
            + role["weight"] @ bank["correction"]
        )
        for local_index, parent_index in enumerate(
            bank["parent_indices"]
        ):
            source_left, source_right = bank["starts"][
                local_index : local_index + 2
            ]
            target_left, target_right = starts[
                parent_index : parent_index + 2
            ]
            prediction[target_left:target_right] = corrected[
                source_left:source_right
            ]
    output_prediction.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_prediction,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray(FAMILY),
        well_ids=well_ids,
        folds=folds,
        row_starts=starts,
        clean_C016=clean,
        prediction=prediction.astype(np.float32),
        hidden_truth_stored=np.bool_(False),
        protected_metrics_emitted=np.bool_(False),
        f4_metric_computed=np.bool_(False),
        protected_reveal_spent=np.bool_(False),
        candidate_manifest_sha256=np.asarray(
            sha256(candidate_manifest)
        ),
        preregistration_sha256=np.asarray(
            sha256(PREREGISTRATION)
        ),
    )
    selector_payload = {
        "schema_version": 1,
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "complete_all_outer_predictions_frozen",
        "selection": summary,
        "logical_inner_roles": 20,
        "unique_excluded_pair_stat_roles": 10,
        "outer_roles": 5,
        "truth_stored_in_prediction": False,
        "protected_metrics_emitted": False,
        "f4_metric_computed": False,
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "prediction_sha256": sha256(output_prediction),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output_selector, selector_payload)
    print(
        json.dumps(
            {
                "status": selector_payload["status"],
                "prediction_sha256": selector_payload[
                    "prediction_sha256"
                ],
                "selector_sha256": sha256(output_selector),
            },
            sort_keys=True,
        )
    )


def freeze_outer(
    candidate_manifest: Path,
    prediction: Path,
    selector: Path,
    output: Path,
) -> None:
    verify_authorization_and_preregistration()
    verify_candidate_manifest(candidate_manifest)
    if not prediction.is_file() or not selector.is_file():
        raise RuntimeError("sealed R094 outer artifact missing")
    payload = {
        "schema_version": 1,
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "frozen_ready_for_single_joint_reveal",
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "prediction": str(prediction),
        "prediction_sha256": sha256(prediction),
        "selector": str(selector),
        "selector_sha256": sha256(selector),
        "outer_folds": 5,
        "logical_inner_roles": 20,
        "unique_excluded_pair_stat_roles": 10,
        "truth_stored_in_prediction": False,
        "protected_metric_computed": False,
        "f4_metric_computed": False,
        "protected_reveal_spent": False,
        "preregistration_sha256": sha256(PREREGISTRATION),
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output_sha256": sha256(output),
            },
            sort_keys=True,
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
                "rows": int(np.sum(mask)),
            }
        )
    return {
        "gain": rmse(baseline, truth) - rmse(prediction, truth),
        "maximum_held_group_regression": max(
            0.0, max(-record["gain"] for record in records)
        ),
        "held_groups": records,
    }


def bootstrap(
    prediction: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    starts: np.ndarray,
    selected: np.ndarray | None = None,
) -> dict[str, Any]:
    indices = (
        np.arange(len(starts) - 1)
        if selected is None
        else np.asarray(selected, dtype=np.int64)
    )
    rows = np.diff(starts)[indices]
    before_sse = []
    after_sse = []
    for index in indices:
        left, right = starts[index : index + 2]
        before = baseline[left:right].astype(np.float64) - truth[left:right]
        after = prediction[left:right].astype(np.float64) - truth[left:right]
        before_sse.append(float(before @ before))
        after_sse.append(float(after @ after))
    rows = np.asarray(rows)
    before_sse = np.asarray(before_sse)
    after_sse = np.asarray(after_sse)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(
        0, len(indices), size=(BOOTSTRAP_RESAMPLES, len(indices))
    )
    sampled_rows = np.sum(rows[samples], axis=1)
    gain = np.sqrt(
        np.sum(before_sse[samples], axis=1) / sampled_rows
    ) - np.sqrt(
        np.sum(after_sse[samples], axis=1) / sampled_rows
    )
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "well_count": len(indices),
        "probability_gain_positive": float(np.mean(gain > 0)),
        "gain_quantile_0_10": float(np.quantile(gain, 0.10)),
        "gain_quantile_0_50": float(np.quantile(gain, 0.50)),
        "gain_quantile_0_90": float(np.quantile(gain, 0.90)),
    }


def evaluate(
    confirmation_path: Path,
    outer_manifest_path: Path,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError("refusing a second R094 protected reveal")
    preregistration = verify_authorization_and_preregistration()
    confirmation = json.loads(confirmation_path.read_text())
    manifest = json.loads(outer_manifest_path.read_text())
    if (
        confirmation["status"] != "confirmations_passed"
        or manifest["status"] != "frozen_ready_for_single_joint_reveal"
        or manifest["preregistration_sha256"]
        != sha256(PREREGISTRATION)
        or manifest["prediction_sha256"]
        != sha256(Path(manifest["prediction"]))
        or manifest["selector_sha256"]
        != sha256(Path(manifest["selector"]))
        or manifest["f4_metric_computed"] is not False
        or manifest["protected_reveal_spent"] is not False
    ):
        raise RuntimeError("R094 reveal contract changed")
    with (
        np.load(Path(manifest["prediction"]), allow_pickle=False) as sealed,
        np.load(LAYOUT, allow_pickle=False) as layout,
    ):
        if (
            bool(sealed["hidden_truth_stored"])
            or bool(sealed["protected_metrics_emitted"])
            or bool(sealed["f4_metric_computed"])
            or "truth" in sealed.files
            or not np.array_equal(
                sealed["well_ids"].astype(str),
                layout["well_ids"].astype(str),
            )
            or not np.array_equal(
                sealed["row_starts"].astype(np.int64),
                layout["row_starts"].astype(np.int64),
            )
        ):
            raise RuntimeError("sealed R094 prediction firewall changed")
        well_ids = sealed["well_ids"].astype(str)
        folds = sealed["folds"].astype(np.int64)
        starts = sealed["row_starts"].astype(np.int64)
        clean = sealed["clean_C016"].astype(np.float64)
        prediction = sealed["prediction"].astype(np.float64)
        baseline = layout["prediction"].astype(np.float64)
        truth = layout["truth"].astype(np.float64)
        groupings = {
            "original": folds,
            "balanced": layout["stratified"].astype(np.int64),
            "independent": layout["independent"].astype(np.int64),
        }
    fold_records = []
    for fold in range(5):
        mask = np.repeat(folds == fold, np.diff(starts))
        before = rmse(baseline[mask], truth[mask])
        after = rmse(prediction[mask], truth[mask])
        clean_score = rmse(clean[mask], truth[mask])
        fold_records.append(
            {
                "fold": fold,
                "wells": int(np.sum(folds == fold)),
                "rows": int(np.sum(mask)),
                "v20_rmse": before,
                "clean_C016_rmse": clean_score,
                "candidate_rmse": after,
                "gain_vs_v20": before - after,
                "gain_vs_clean_C016": clean_score - after,
            }
        )
    grouping_results = {
        name: grouping_audit(
            prediction, baseline, truth, starts, groups
        )
        for name, groups in groupings.items()
    }
    pooled_baseline = rmse(baseline, truth)
    pooled_candidate = rmse(prediction, truth)
    pooled_clean = rmse(clean, truth)
    audit_wells = np.flatnonzero(folds == 4)
    f4 = fold_records[4]
    bootstrap_result = bootstrap(
        prediction, baseline, truth, starts
    )
    f4_bootstrap = bootstrap(
        prediction, baseline, truth, starts, audit_wells
    )
    improved_folds = sum(
        record["gain_vs_v20"] > 0 for record in fold_records
    )
    maximum_regression = max(
        result["maximum_held_group_regression"]
        for result in grouping_results.values()
    )
    all_nonnegative = all(
        result["gain"] >= 0 for result in grouping_results.values()
    )
    all_positive = all(
        result["gain"] > 0 for result in grouping_results.values()
    )
    positive_inner = int(
        json.loads(Path(manifest["selector"]).read_text())[
            "selection"
        ]["positive_inner_roles"]
    )
    strong_clauses = {
        "pooled_gain_at_least_0p03": (
            pooled_baseline - pooled_candidate >= 0.03
        ),
        "at_least_four_original_folds_improve": improved_folds >= 4,
        "all_grouping_gains_nonnegative": all_nonnegative,
        "maximum_held_group_regression_at_most_0p01": (
            maximum_regression <= 0.01
        ),
    }
    small_clauses = {
        "pooled_gain_positive": pooled_baseline - pooled_candidate > 0,
        "all_grouping_gains_positive": all_positive,
        "at_least_ten_positive_inner_roles": positive_inner >= 10,
        "maximum_held_group_regression_at_most_0p01": (
            maximum_regression <= 0.01
        ),
        "bootstrap_probability_at_least_0p80": (
            bootstrap_result["probability_gain_positive"] >= 0.80
        ),
        "bootstrap_q10_at_least_negative_0p005": (
            bootstrap_result["gain_quantile_0_10"] >= -0.005
        ),
    }
    f4_passed = f4["gain_vs_v20"] >= 0
    strong_passed = all(strong_clauses.values())
    small_passed = all(small_clauses.values())
    accepted = bool((strong_passed or small_passed) and f4_passed)
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "batch": 3,
        "candidate": FAMILY,
        "stage": "joint_protected_outer_promotion_result",
        "status": (
            "complete_with_promotion"
            if accepted
            else "complete_rejected_by_outer_gate"
        ),
        "accepted": accepted,
        "disclosure_label": (
            "selection-clean, development-exposed; F4 prospective"
        ),
        "exception_entry_threshold": 0.15,
        "stored_run_goal": 0.20,
        "baseline_rmse": pooled_baseline,
        "clean_C016_rmse": pooled_clean,
        "candidate_rmse": pooled_candidate,
        "gain_vs_v20": pooled_baseline - pooled_candidate,
        "gain_vs_clean_C016": pooled_clean - pooled_candidate,
        "folds": fold_records,
        "f4_audit": f4
        | {
            "nonnegative_gain_veto_passed": f4_passed,
            "bootstrap": f4_bootstrap,
        },
        "f4_nonnegative_gain_veto_passed": f4_passed,
        "groupings": grouping_results,
        "maximum_held_group_regression": maximum_regression,
        "bootstrap": bootstrap_result,
        "positive_inner_roles": positive_inner,
        "strong_gate": {
            "passed": strong_passed,
            "clauses": strong_clauses,
        },
        "small_diverse_gate": {
            "passed": small_passed,
            "clauses": small_clauses,
        },
        "protected_outer_reveal_count": 1,
        "protected_outer_reveal_spent": True,
        "outer_manifest": str(outer_manifest_path),
        "outer_manifest_sha256": sha256(outer_manifest_path),
        "confirmation": str(confirmation_path),
        "confirmation_sha256": sha256(confirmation_path),
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "kaggle_dataset_authorized": False,
        "kaggle_kernel_authorized": False,
        "kaggle_submission_authorized": False,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    confirm_parser = commands.add_parser("confirm")
    confirm_parser.add_argument("--output", type=Path, required=True)
    freeze_candidates_parser = commands.add_parser("freeze-candidates")
    freeze_candidates_parser.add_argument(
        "--audit-root", type=Path, default=AUDIT_ROOT
    )
    freeze_candidates_parser.add_argument(
        "--output", type=Path, required=True
    )
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument(
        "--candidate-manifest", type=Path, required=True
    )
    assemble_parser.add_argument(
        "--output-prediction", type=Path, required=True
    )
    assemble_parser.add_argument(
        "--output-selector", type=Path, required=True
    )
    freeze_outer_parser = commands.add_parser("freeze-outer")
    freeze_outer_parser.add_argument(
        "--candidate-manifest", type=Path, required=True
    )
    freeze_outer_parser.add_argument(
        "--prediction", type=Path, required=True
    )
    freeze_outer_parser.add_argument(
        "--selector", type=Path, required=True
    )
    freeze_outer_parser.add_argument(
        "--output", type=Path, required=True
    )
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--confirmation", type=Path, required=True
    )
    evaluate_parser.add_argument(
        "--outer-manifest", type=Path, required=True
    )
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "confirm":
        confirm(args.output)
    elif args.command == "freeze-candidates":
        freeze_candidates(args.audit_root, args.output)
    elif args.command == "assemble":
        assemble(
            args.candidate_manifest,
            args.output_prediction,
            args.output_selector,
        )
    elif args.command == "freeze-outer":
        freeze_outer(
            args.candidate_manifest,
            args.prediction,
            args.selector,
            args.output,
        )
    else:
        evaluate(
            args.confirmation,
            args.outer_manifest,
            args.output,
        )


if __name__ == "__main__":
    main()
