from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from limited_query_cv import promote_v5_batch3_run2_r094 as legacy
from limited_query_cv import v5_batch3_run2_reverse_residual_bank as primary
from limited_query_cv import v5_batch3_run2_secondary_reverse_bank as secondary
from limited_query_cv.develop_v5_batch3_run4_well_utility_router import (
    NON_SURFACE_INDICES,
    STRUCTURAL_REVERSE_LOCAL,
    SURFACE_INDICES,
    select_single_route,
)
from limited_query_cv.prepare_v5_batch3_run4_router_features import (
    NONRAMPED_INDICES,
)
from limited_query_cv.promote_v5_batch3_run2_r094 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LAYOUT,
    PARENT,
    bootstrap,
    grouping_audit,
    rmse,
)
from limited_query_cv.train_v5_batch3_run2_reverse_bank import (
    conditional_weight,
)
from limited_query_cv.train_v5_batch3_run2_surface_selector import (
    DATA_ROOT,
    load_typewell,
    well_features,
)
from limited_query_cv.v5_batch3_run2_surface import typewell_clusters
from limited_query_cv.v5_batch3_run4_clean_bank import (
    RUN_ID,
    RUN_ROOT,
    clean_run4_overrides,
    load_clean_fold_bank,
)


FAMILY = "clean_boosted_whole_well_utility_router"
AUDIT_ROOT = RUN_ROOT / "promotion_audit"
SOURCE_AUDIT_ROOT = Path(
    "limited_query_cv/runs/v5_batch3_run_003_goal_020/promotion_audit"
)
AUTHORIZATION = RUN_ROOT / "user_authorized_confirmation_waiver.json"
POLICY_AMENDMENT = RUN_ROOT / "confirmation_waiver_policy_amendment.json"
CONFIRMATION = RUN_ROOT / "promotion_confirmation_result.json"
PREREGISTRATION = RUN_ROOT / "boosted_router_promotion_preregistration.json"
DEV_STATS = RUN_ROOT / "clean_well_stats.npz"
DEV_FEATURES = RUN_ROOT / "router_features.npz"
DEV_RESULT = RUN_ROOT / "clean_boosted_well_utility_router_development.json"
F4_FEATURES = AUDIT_ROOT / "target_free_f4_router_features.npz"
F4_SURFACE = SOURCE_AUDIT_ROOT / "candidates/surface_f4_cache.npz"
C016_OUTER = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "promotion_audit/sealed/outer_predictions.npz"
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


def verify_contract() -> dict[str, Any]:
    authorization = json.loads(AUTHORIZATION.read_text())
    amendment = json.loads(POLICY_AMENDMENT.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    promotion = json.loads(PREREGISTRATION.read_text())
    if (
        authorization["status"]
        != "authorized_failed_confirmation_waiver_and_F4_reveal"
        or authorization["protected_F4_reveal_authorized"] is not True
        or authorization["kaggle_submission_authorized"] is not False
        or amendment["status"] != "active_candidate_specific_exception"
        or amendment["authorization_sha256"] != sha256(AUTHORIZATION)
        or confirmation["fresh_confirmation_passed"] is not False
        or promotion["status"]
        != "frozen_before_single_protected_outer_reveal"
        or promotion["authorization_sha256"] != sha256(AUTHORIZATION)
        or promotion["policy_amendment_sha256"]
        != sha256(POLICY_AMENDMENT)
        or promotion["failed_confirmation_sha256"]
        != sha256(CONFIRMATION)
        or promotion["frozen_source_sha256"] != sha256(Path(__file__))
        or promotion["protected_outer_reveal"]["previously_spent"]
        is not False
        or promotion["kaggle_dataset_authorized"] is not False
        or promotion["kaggle_kernel_authorized"] is not False
        or promotion["kaggle_submission_authorized"] is not False
    ):
        raise RuntimeError("Run 4 promotion contract changed")
    return promotion


def extract_parent_field(
    field: str,
    fold: int,
    expected_ids: np.ndarray,
) -> np.ndarray:
    with np.load(C016_OUTER, allow_pickle=False) as parent:
        if (
            bool(parent["hidden_truth_stored"])
            or bool(parent["f4_metric_computed"])
            or "truth" in parent.files
        ):
            raise RuntimeError("C016 outer prediction firewall changed")
        ids = parent["well_ids"].astype(str)
        folds = parent["folds"].astype(np.int64)
        starts = parent["row_starts"].astype(np.int64)
        values = parent[field].astype(np.float64)
    parts = []
    for well_id in expected_ids.astype(str):
        found = np.flatnonzero((ids == well_id) & (folds == fold))
        if len(found) != 1:
            raise RuntimeError(f"{field}: incomplete fold {fold}")
        index = int(found[0])
        parts.append(values[starts[index] : starts[index + 1]])
    return np.concatenate(parts)


def fold_lengths(expected_ids: np.ndarray, fold: int) -> np.ndarray:
    with np.load(PARENT, allow_pickle=False) as parent:
        ids = parent["well_ids"].astype(str)
        folds = parent["folds"].astype(np.int64)
        starts = parent["row_starts"].astype(np.int64)
    output = []
    for well_id in expected_ids.astype(str):
        found = np.flatnonzero((ids == well_id) & (folds == fold))
        if len(found) != 1:
            raise RuntimeError(f"fold {fold}: parent layout incomplete")
        index = int(found[0])
        output.append(int(starts[index + 1] - starts[index]))
    return np.asarray(output, dtype=np.int64)


def f4_primary_directions(
    fold: int,
    expected_ids: np.ndarray,
    clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if fold != 4:
        raise RuntimeError("deployment fallback is F4-only")
    names = np.asarray([record[0] for record in primary.ARCHIVES])
    directions = np.zeros((len(names), len(clean)), dtype=np.float64)
    mode_index = int(
        np.flatnonzero(names == "reverse_protected_mode_maximin")[0]
    )
    mode_prediction = extract_parent_field(
        "protected_mode_bank", fold, expected_ids
    )
    directions[mode_index] = clean - mode_prediction
    records = []
    for index, name in enumerate(names):
        available = index == mode_index
        records.append(
            {
                "name": str(name),
                "path": str(C016_OUTER) if available else "unavailable",
                "sha256": sha256(C016_OUTER) if available else "zero-fallback",
                "field": "protected_mode_bank" if available else "zero",
                "target_free_available": available,
                "deployment_fallback": not available,
            }
        )
    return names, directions, records


def f4_secondary_directions(
    fold: int,
    expected_ids: np.ndarray,
    clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if fold != 4:
        raise RuntimeError("deployment fallback is F4-only")
    names = np.asarray([record[0] for record in secondary.SECONDARY_ARCHIVES])
    directions = np.zeros((len(names), len(clean)), dtype=np.float64)
    records = [
        {
            "name": str(name),
            "path": "unavailable",
            "sha256": "zero-fallback",
            "field": "zero",
            "target_free_available": False,
            "deployment_fallback": True,
        }
        for name in names
    ]
    return names, directions, records


def f4_ramped_directions(
    fold: int,
    expected_ids: np.ndarray,
    clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    primary_names, first, first_records = f4_primary_directions(
        fold, expected_ids, clean
    )
    secondary_names, second, second_records = f4_secondary_directions(
        fold, expected_ids, clean
    )
    lengths = fold_lengths(expected_ids, fold)
    ramp = np.concatenate(
        [
            np.zeros(1, dtype=np.float64)
            if length == 1
            else np.linspace(0.0, 1.0, length, dtype=np.float64)
            for length in lengths
        ]
    )
    names = np.char.add(
        "ramped_", np.concatenate([primary_names, secondary_names])
    )
    return (
        names,
        np.concatenate([first, second]) * ramp[None],
        lengths,
        first_records + second_records,
    )


@contextmanager
def promotion_bank_overrides() -> Iterator[None]:
    with clean_run4_overrides():
        saved_primary = legacy.load_reverse_directions
        saved_secondary = legacy.load_secondary_reverse_directions
        saved_ramped = legacy.load_ramped_reverse_directions
        legacy.load_reverse_directions = (
            lambda fold, ids, clean: (
                f4_primary_directions(fold, ids, clean)
                if fold == 4
                else saved_primary(fold, ids, clean)
            )
        )
        legacy.load_secondary_reverse_directions = (
            lambda fold, ids, clean: (
                f4_secondary_directions(fold, ids, clean)
                if fold == 4
                else saved_secondary(fold, ids, clean)
            )
        )
        legacy.load_ramped_reverse_directions = (
            lambda fold, ids, clean: (
                f4_ramped_directions(fold, ids, clean)
                if fold == 4
                else saved_ramped(fold, ids, clean)
            )
        )
        try:
            yield
        finally:
            legacy.load_reverse_directions = saved_primary
            legacy.load_secondary_reverse_directions = saved_secondary
            legacy.load_ramped_reverse_directions = saved_ramped


def load_promotion_bank(
    fold: int,
    *,
    truth_allowed: bool,
) -> dict[str, Any]:
    if fold < 4:
        return load_clean_fold_bank(fold, truth_allowed=truth_allowed)
    if truth_allowed:
        raise RuntimeError("F4 truth may not enter a candidate loader")
    with promotion_bank_overrides():
        return legacy.load_fold_bank(
            fold,
            truth_allowed=False,
            audit_root=SOURCE_AUDIT_ROOT,
        )


def build_f4_feature_matrix(bank: dict[str, Any]) -> np.ndarray:
    with np.load(F4_SURFACE, allow_pickle=False) as surface:
        if (
            int(surface["fold"]) != 4
            or bool(surface["F4_target_loaded"])
            or bool(surface["F4_metric_computed"])
            or bool(surface["protected_reveal_spent"])
            or not np.array_equal(
                surface["well_ids"].astype(str), bank["ids"].astype(str)
            )
        ):
            raise RuntimeError("F4 surface feature firewall changed")
        surfaces = surface["prediction"].astype(np.float64)
        cluster_sizes = surface["cluster_pool_sizes"].astype(np.int64)
    all_typewells = {
        str(well_id): load_typewell(str(well_id), DATA_ROOT)
        for well_id in np.load(PARENT, allow_pickle=False)["well_ids"].astype(str)
    }
    clusters = typewell_clusters(all_typewells)
    selector = []
    for index, well_id in enumerate(bank["ids"].astype(str)):
        left, right = bank["starts"][index : index + 2]
        numeric, _ = well_features(
            well_id,
            bank["clean"][left:right],
            bank["baseline"][left:right],
            surfaces[:, left:right],
            int(cluster_sizes[index]),
            int(clusters[well_id]),
            DATA_ROOT,
        )
        selector.append(numeric)
    selector_matrix = np.stack(selector).astype(np.float64)
    rows = np.diff(bank["starts"]).astype(np.float64)
    grams = []
    for index in range(len(bank["ids"])):
        left, right = bank["starts"][index : index + 2]
        correction = bank["correction"][:, left:right]
        grams.append(correction @ correction.T)
    gram = np.stack(grams)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    rms = np.sqrt(np.maximum(diagonal, 0.0) / rows[:, None])
    left_scale = np.sqrt(
        np.maximum(diagonal[:, NONRAMPED_INDICES], 1e-12)
    )
    right_scale = np.sqrt(
        np.maximum(diagonal[:, SURFACE_INDICES], 1e-12)
    )
    cosine = gram[
        :,
        NONRAMPED_INDICES[:, None],
        SURFACE_INDICES[None, :],
    ] / (left_scale[:, :, None] * right_scale[:, None, :])
    cosine = np.clip(cosine, -1.0, 1.0)
    disagreement = np.std(cosine[:, :12, :], axis=1)
    features = np.concatenate(
        (
            selector_matrix,
            np.log1p(rms),
            cosine.reshape(len(rows), -1),
            disagreement,
        ),
        axis=1,
    ).astype(np.float32)
    if features.shape != (154, 602) or not np.isfinite(features).all():
        raise RuntimeError("invalid target-free F4 router features")
    return features


def freeze_candidates(output: Path) -> None:
    verify_contract()
    if F4_FEATURES.exists() or output.exists():
        raise FileExistsError(F4_FEATURES if F4_FEATURES.exists() else output)
    bank = load_promotion_bank(4, truth_allowed=False)
    features = build_f4_feature_matrix(bank)
    F4_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        F4_FEATURES,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        well_ids=bank["ids"].astype(str),
        features=features,
        source_surface_sha256=np.asarray(sha256(F4_SURFACE)),
        candidate_names=bank["names"].astype(str),
        unavailable_reverse_candidates=np.asarray(
            [
                record[0]
                for record in primary.ARCHIVES
                if record[0] != "reverse_protected_mode_maximin"
            ]
            + [record[0] for record in secondary.SECONDARY_ARCHIVES]
        ),
        target_fields_read=np.asarray(False),
        hidden_truth_stored=np.asarray(False),
        F4_target_loaded=np.asarray(False),
        F4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
    )
    paths = {
        Path(__file__),
        DEV_STATS,
        DEV_FEATURES,
        DEV_RESULT,
        F4_FEATURES,
        F4_SURFACE,
        C016_OUTER,
        PARENT,
        *[
            Path(record["path"])
            for record in bank["records"]
            if record.get("path") not in {None, "unavailable"}
        ],
    }
    records = [
        {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths, key=str)
    ]
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "all_target_free_candidate_inputs_frozen",
        "records": records,
        "record_count": len(records),
        "F4_features": str(F4_FEATURES),
        "F4_features_sha256": sha256(F4_FEATURES),
        "deployment_availability_rule": (
            "checksum-verified candidate or exact zero correction; "
            "unavailable candidates cannot alter predictions"
        ),
        "unavailable_F4_reverse_candidates": 9,
        "available_F4_reverse_candidate": "reverse_protected_mode_maximin",
        "hidden_truth_stored": False,
        "F4_target_loaded": False,
        "F4_metric_computed": False,
        "protected_reveal_spent": False,
        "kaggle_dataset_authorized": False,
        "kaggle_kernel_authorized": False,
        "kaggle_submission_authorized": False,
        "promotion_preregistration_sha256": sha256(PREREGISTRATION),
    }
    write_json(output, payload)
    print(json.dumps({"status": payload["status"], "records": len(records)}))


def verify_candidate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if (
        manifest["status"] != "all_target_free_candidate_inputs_frozen"
        or manifest["promotion_preregistration_sha256"]
        != sha256(PREREGISTRATION)
        or manifest["F4_target_loaded"] is not False
        or manifest["F4_metric_computed"] is not False
        or manifest["protected_reveal_spent"] is not False
    ):
        raise RuntimeError("candidate manifest changed")
    for record in manifest["records"]:
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"{path}: frozen candidate changed")
    return manifest


def fitted_base(
    training: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    total_cross = np.sum(cross[training], axis=0)
    total_gram = np.sum(gram[training], axis=0)
    weight = np.zeros(49, dtype=np.float64)
    weight[:39] = conditional_weight(
        total_cross[:39],
        total_gram[:39, :39],
        28,
        np.arange(29, 39, dtype=np.int64),
        0.75,
        1e-4,
        0.75,
        0.75,
    )
    weight *= 0.875
    weight[17] = 0.0
    return weight


def qnorm_target(
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    indices: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    diagonal = np.stack([gram[:, index, index] for index in indices], axis=1)
    numerator = cross[:, indices] + np.einsum(
        "nij,j->ni", gram[:, indices, :], base
    )
    return -numerator / np.sqrt(
        np.maximum(diagonal * rows[:, None], 1e-12)
    )


def fit_route(
    features: np.ndarray,
    rows: np.ndarray,
    cross: np.ndarray,
    gram: np.ndarray,
    training: np.ndarray,
    predicted: np.ndarray,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = fitted_base(training, cross, gram)
    surface_target = qnorm_target(
        rows, cross, gram, SURFACE_INDICES, base
    )
    lower = np.quantile(surface_target[training], 0.01, axis=0)
    upper = np.quantile(surface_target[training], 0.99, axis=0)
    surface_target = np.clip(surface_target, lower, upper)
    surface_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=1,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_lambda=15.0,
        reg_alpha=0.1,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=16,
        random_state=20265600 + seed_offset,
    )
    surface_model.fit(
        features[training],
        surface_target[training],
        sample_weight=rows[training] / np.mean(rows[training]),
        verbose=False,
    )
    surface_index, surface_amplitude = select_single_route(
        surface_model.predict(features[predicted]),
        rows[predicted],
        gram[predicted],
        SURFACE_INDICES,
        allowed_local_indices=np.arange(4, dtype=np.int64),
        amplitude_cap=0.30,
        utility_threshold=0.20,
    )

    second_target = qnorm_target(
        rows, cross, gram, NON_SURFACE_INDICES, base
    )
    lower = np.quantile(second_target[training], 0.01, axis=0)
    upper = np.quantile(second_target[training], 0.99, axis=0)
    second_target = np.clip(second_target, lower, upper)
    second_model = ExtraTreesRegressor(
        n_estimators=700,
        min_samples_leaf=16,
        max_features=0.7,
        random_state=20265216 + seed_offset,
        n_jobs=16,
    )
    second_model.fit(
        features[training],
        second_target[training],
        sample_weight=rows[training] / np.mean(rows[training]),
    )
    second_index, second_amplitude = select_single_route(
        second_model.predict(features[predicted]),
        rows[predicted],
        gram[predicted],
        NON_SURFACE_INDICES,
        allowed_local_indices=STRUCTURAL_REVERSE_LOCAL,
        amplitude_cap=0.20,
        utility_threshold=0.01,
    )
    route = np.zeros((int(np.sum(predicted)), 49), dtype=np.float64)
    local = np.arange(len(route))
    route[local, surface_index] += 0.625 * surface_amplitude
    route[local, second_index] += 1.25 * second_amplitude
    return base, route


def sealed_inputs() -> dict[str, Any]:
    with np.load(PARENT, allow_pickle=False) as parent:
        parent_ids = parent["well_ids"].astype(str)
        parent_folds = parent["folds"].astype(np.int64)
        parent_starts = parent["row_starts"].astype(np.int64)
        clean = parent["prediction"].astype(np.float64)
    with np.load(LAYOUT, allow_pickle=False) as layout:
        if not np.array_equal(layout["well_ids"].astype(str), parent_ids):
            raise RuntimeError("sealed layout differs from C016 parent")
        truth = layout["truth"].astype(np.float64)

    rows = np.diff(parent_starts).astype(np.float64)
    clean_sse = np.zeros(len(parent_ids), dtype=np.float64)
    cross = np.zeros((len(parent_ids), 49), dtype=np.float64)
    gram = np.zeros((len(parent_ids), 49, 49), dtype=np.float64)
    banks: dict[int, dict[str, Any]] = {}
    names = None
    for fold in range(5):
        bank = load_promotion_bank(fold, truth_allowed=fold < 4)
        banks[fold] = bank
        if names is None:
            names = bank["names"].astype(str)
        elif not np.array_equal(names, bank["names"].astype(str)):
            raise RuntimeError("promotion candidate schema changed")
        for local_index, parent_index in enumerate(bank["parent_indices"]):
            left, right = bank["starts"][local_index : local_index + 2]
            target_left, target_right = parent_starts[
                parent_index : parent_index + 2
            ]
            local_truth = truth[target_left:target_right]
            error = bank["clean"][left:right] - local_truth
            correction = bank["correction"][:, left:right]
            clean_sse[parent_index] = float(error @ error)
            cross[parent_index] = correction @ error
            gram[parent_index] = correction @ correction.T

    with np.load(DEV_FEATURES, allow_pickle=False) as dev:
        dev_lookup = {
            well_id: dev["features"][index].astype(np.float64)
            for index, well_id in enumerate(dev["well_ids"].astype(str))
        }
    with np.load(F4_FEATURES, allow_pickle=False) as f4:
        if (
            bool(f4["F4_target_loaded"])
            or bool(f4["F4_metric_computed"])
            or bool(f4["protected_reveal_spent"])
        ):
            raise RuntimeError("F4 feature firewall changed")
        f4_lookup = {
            well_id: f4["features"][index].astype(np.float64)
            for index, well_id in enumerate(f4["well_ids"].astype(str))
        }
    features = np.stack(
        [
            (dev_lookup if fold < 4 else f4_lookup)[well_id]
            for well_id, fold in zip(parent_ids, parent_folds)
        ]
    )
    return {
        "well_ids": parent_ids,
        "folds": parent_folds,
        "starts": parent_starts,
        "clean": clean,
        "rows": rows,
        "clean_sse": clean_sse,
        "cross": cross,
        "gram": gram,
        "features": features,
        "names": names,
        "banks": banks,
    }


def assemble(
    candidate_manifest: Path,
    output_prediction: Path,
    output_selector: Path,
    output_inner_roles: Path,
) -> None:
    if any(
        path.exists()
        for path in (output_prediction, output_selector, output_inner_roles)
    ):
        raise FileExistsError("refusing a second sealed Run 4 assembly")
    verify_contract()
    verify_candidate_manifest(candidate_manifest)
    data = sealed_inputs()

    pair_masks = []
    pair_bases = []
    pair_routes = []
    pair_labels = []
    for pair_index, pair in enumerate(itertools.combinations(range(5), 2)):
        training = ~np.isin(data["folds"], pair)
        predicted = ~training
        base, local_route = fit_route(
            data["features"],
            data["rows"],
            data["cross"],
            data["gram"],
            training,
            predicted,
            100 + pair_index,
        )
        full_route = np.zeros((len(data["well_ids"]), 49), dtype=np.float32)
        full_route[predicted] = local_route.astype(np.float32)
        pair_masks.append(predicted)
        pair_bases.append(base)
        pair_routes.append(full_route)
        pair_labels.append(pair)

    prediction = data["clean"].copy()
    outer_routes = np.zeros((len(data["well_ids"]), 49), dtype=np.float32)
    outer_bases = np.zeros((5, 49), dtype=np.float64)
    for outer in range(5):
        training = data["folds"] != outer
        predicted = ~training
        base, local_route = fit_route(
            data["features"],
            data["rows"],
            data["cross"],
            data["gram"],
            training,
            predicted,
            outer,
        )
        outer_bases[outer] = base
        outer_routes[predicted] = local_route.astype(np.float32)
        bank = data["banks"][outer]
        for local_index, parent_index in enumerate(bank["parent_indices"]):
            left, right = bank["starts"][local_index : local_index + 2]
            target_left, target_right = data["starts"][
                parent_index : parent_index + 2
            ]
            weight = base + outer_routes[parent_index].astype(np.float64)
            prediction[target_left:target_right] = (
                bank["clean"][left:right]
                + weight @ bank["correction"][:, left:right]
            )

    output_inner_roles.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_inner_roles,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        excluded_pairs=np.asarray(pair_labels, dtype=np.int8),
        predicted_masks=np.stack(pair_masks),
        base_weights=np.stack(pair_bases).astype(np.float32),
        route_weights=np.stack(pair_routes).astype(np.float32),
        logical_inner_roles=np.int64(20),
        unique_symmetric_pair_models=np.int64(10),
        protected_metrics_emitted=np.asarray(False),
        f4_metric_computed=np.asarray(False),
    )
    with np.load(PARENT, allow_pickle=False) as parent:
        baseline = parent["baseline"].astype(np.float32)
    np.savez_compressed(
        output_prediction,
        schema_version=np.int64(1),
        protocol_revision=np.int64(11),
        run_id=np.asarray(RUN_ID),
        family=np.asarray(FAMILY),
        well_ids=data["well_ids"],
        folds=data["folds"].astype(np.int8),
        row_starts=data["starts"],
        baseline=baseline,
        clean_C016=data["clean"].astype(np.float32),
        prediction=prediction.astype(np.float32),
        outer_base_weights=outer_bases.astype(np.float32),
        outer_route_weights=outer_routes,
        hidden_truth_stored=np.asarray(False),
        protected_metrics_emitted=np.asarray(False),
        f4_metric_computed=np.asarray(False),
        protected_reveal_spent=np.asarray(False),
        candidate_manifest_sha256=np.asarray(sha256(candidate_manifest)),
        inner_roles_sha256=np.asarray(sha256(output_inner_roles)),
        promotion_preregistration_sha256=np.asarray(
            sha256(PREREGISTRATION)
        ),
    )
    selector = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "complete_all_outer_predictions_frozen",
        "fixed_recipe": json.loads(PREREGISTRATION.read_text())[
            "fixed_recipe"
        ],
        "logical_inner_roles": 20,
        "unique_symmetric_pair_models": 10,
        "outer_models": 5,
        "inner_roles": str(output_inner_roles),
        "inner_roles_sha256": sha256(output_inner_roles),
        "prediction": str(output_prediction),
        "prediction_sha256": sha256(output_prediction),
        "candidate_manifest_sha256": sha256(candidate_manifest),
        "truth_stored_in_prediction": False,
        "protected_metrics_emitted": False,
        "f4_metric_computed": False,
        "protected_reveal_spent": False,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output_selector, selector)
    print(
        json.dumps(
            {
                "status": selector["status"],
                "prediction_sha256": selector["prediction_sha256"],
                "selector_sha256": sha256(output_selector),
            }
        )
    )


def freeze_outer(
    candidate_manifest: Path,
    prediction: Path,
    selector: Path,
    inner_roles: Path,
    output: Path,
) -> None:
    verify_contract()
    verify_candidate_manifest(candidate_manifest)
    selector_record = json.loads(selector.read_text())
    if (
        selector_record["status"]
        != "complete_all_outer_predictions_frozen"
        or selector_record["prediction_sha256"] != sha256(prediction)
        or selector_record["inner_roles_sha256"] != sha256(inner_roles)
        or selector_record["protected_metrics_emitted"] is not False
        or selector_record["f4_metric_computed"] is not False
    ):
        raise RuntimeError("sealed Run 4 outer artifact changed")
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
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
        "inner_roles": str(inner_roles),
        "inner_roles_sha256": sha256(inner_roles),
        "outer_folds": 5,
        "logical_inner_roles": 20,
        "unique_symmetric_pair_models": 10,
        "truth_stored_in_prediction": False,
        "protected_metric_computed": False,
        "f4_metric_computed": False,
        "protected_reveal_spent": False,
        "promotion_preregistration_sha256": sha256(PREREGISTRATION),
    }
    write_json(output, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(output)}))


def evaluate(outer_manifest_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("refusing a second Run 4 protected reveal")
    verify_contract()
    manifest = json.loads(outer_manifest_path.read_text())
    if (
        manifest["status"] != "frozen_ready_for_single_joint_reveal"
        or manifest["promotion_preregistration_sha256"]
        != sha256(PREREGISTRATION)
        or manifest["prediction_sha256"]
        != sha256(Path(manifest["prediction"]))
        or manifest["selector_sha256"]
        != sha256(Path(manifest["selector"]))
        or manifest["inner_roles_sha256"]
        != sha256(Path(manifest["inner_roles"]))
        or manifest["f4_metric_computed"] is not False
        or manifest["protected_reveal_spent"] is not False
    ):
        raise RuntimeError("Run 4 reveal contract changed")
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
            raise RuntimeError("sealed Run 4 prediction firewall changed")
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
        name: grouping_audit(prediction, baseline, truth, starts, groups)
        for name, groups in groupings.items()
    }
    pooled_baseline = rmse(baseline, truth)
    pooled_candidate = rmse(prediction, truth)
    pooled_clean = rmse(clean, truth)
    f4 = fold_records[4]
    bootstrap_result = bootstrap(prediction, baseline, truth, starts)
    f4_bootstrap = bootstrap(
        prediction,
        baseline,
        truth,
        starts,
        np.flatnonzero(folds == 4),
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
    f4_passed = f4["gain_vs_v20"] >= 0
    strong_passed = all(strong_clauses.values())
    accepted = bool(strong_passed and f4_passed)
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "batch": 3,
        "candidate": "clean boosted whole-well utility router",
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
        "failed_confirmation_waived_by_user": True,
        "exception_entry_threshold": 0.17,
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
        "strong_gate": {
            "passed": strong_passed,
            "clauses": strong_clauses,
        },
        "small_diverse_gate": {
            "passed": False,
            "reason": (
                "No preregistered fold-local incremental weight estimates; "
                "promotion requires the ordinary strong gate."
            ),
        },
        "deployment_availability_fallback_applied_on_F4": True,
        "unavailable_F4_reverse_candidates": 9,
        "protected_outer_reveal_count": 1,
        "protected_outer_reveal_spent": True,
        "outer_manifest": str(outer_manifest_path),
        "outer_manifest_sha256": sha256(outer_manifest_path),
        "failed_confirmation": str(CONFIRMATION),
        "failed_confirmation_sha256": sha256(CONFIRMATION),
        "promotion_preregistration": str(PREREGISTRATION),
        "promotion_preregistration_sha256": sha256(PREREGISTRATION),
        "authorization": str(AUTHORIZATION),
        "authorization_sha256": sha256(AUTHORIZATION),
        "bootstrap_contract": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
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
    freeze_candidates_parser = commands.add_parser("freeze-candidates")
    freeze_candidates_parser.add_argument("--output", type=Path, required=True)
    assemble_parser = commands.add_parser("assemble")
    assemble_parser.add_argument("--candidate-manifest", type=Path, required=True)
    assemble_parser.add_argument("--output-prediction", type=Path, required=True)
    assemble_parser.add_argument("--output-selector", type=Path, required=True)
    assemble_parser.add_argument("--output-inner-roles", type=Path, required=True)
    freeze_outer_parser = commands.add_parser("freeze-outer")
    freeze_outer_parser.add_argument("--candidate-manifest", type=Path, required=True)
    freeze_outer_parser.add_argument("--prediction", type=Path, required=True)
    freeze_outer_parser.add_argument("--selector", type=Path, required=True)
    freeze_outer_parser.add_argument("--inner-roles", type=Path, required=True)
    freeze_outer_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--outer-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-candidates":
        freeze_candidates(args.output)
    elif args.command == "assemble":
        assemble(
            args.candidate_manifest,
            args.output_prediction,
            args.output_selector,
            args.output_inner_roles,
        )
    elif args.command == "freeze-outer":
        freeze_outer(
            args.candidate_manifest,
            args.prediction,
            args.selector,
            args.inner_roles,
            args.output,
        )
    else:
        evaluate(args.outer_manifest, args.output)


if __name__ == "__main__":
    main()
