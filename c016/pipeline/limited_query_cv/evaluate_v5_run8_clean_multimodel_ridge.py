from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


RUN_ID = "v5_batch2_run_008_goal_050"
RUN_ROOT = Path("limited_query_cv/runs") / RUN_ID
SOURCE_ROOT = Path("limited_query_cv/runs/v5_batch2_run_006_goal_050")
ARCHIVES = {
    "parent": (
        RUN_ROOT / "clean_dual_parent_clean.npz",
        "9ff61639bbca63e769b5a8bfacc4bb0e1674b1513b330cff403fc4ae11e46cfc",
        "prediction",
    ),
    "D570": (
        SOURCE_ROOT / "d209_amplitude_clean.npz",
        "5e3b6ee41f8705cc6b0044b36255bf8e8b3b6c4752c2afa9f6049d83f77cd310",
        "prediction",
    ),
    "D072": (
        SOURCE_ROOT / "purece_strat_clean.npz",
        "9e6f935f4fb6a831b9355d3db4c553be55a48c7899dfe3f75fc05f2335945f89",
        "prediction",
    ),
    "protected_posterior_incremental": (
        SOURCE_ROOT / "protected_posterior_clean.npz",
        "999123a3392b7af9d10e2a02ef26bd94b8a732f95b109a2a073a05b2b722c43c",
        "incremental_after_D570",
    ),
    "protected_mode_bank": (
        SOURCE_ROOT / "protected_mode_bank_clean.npz",
        "6240d9077188be3efcc2299f97fd59a879cac0a095201168e5250d7dbf341aee",
        "prediction",
    ),
}
COMPONENTS = (
    "D570",
    "D072",
    "protected_posterior_incremental",
    "protected_mode_bank",
)
RIDGE_GRID = (0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)
MAX_WEIGHT = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def load_sources(
    overrides: dict[str, Path] | None = None,
) -> dict[str, np.ndarray]:
    overrides = overrides or {}
    loaded = {}
    reference = None
    for name, (registered_path, expected_hash, prediction_key) in ARCHIVES.items():
        path = overrides.get(name, registered_path)
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(
                f"{name}: expected SHA256 {expected_hash}, observed {observed}"
            )
        with np.load(path, allow_pickle=False) as cache:
            for firewall in (
                "audit_fold_loaded",
                "confirmation_regroupings_loaded",
                "f4_loaded",
            ):
                if firewall in cache.files and bool(cache[firewall]):
                    raise ValueError(f"{name}: protected firewall failed")
            layout = {
                "well_ids": cache["well_ids"].astype(str),
                "folds": cache["folds"].astype(np.int8),
                "row_starts": cache["row_starts"].astype(np.int64),
                "baseline": cache["baseline"].astype(np.float32),
                "truth": cache["truth"].astype(np.float32),
            }
            if reference is None:
                reference = layout
            elif any(
                not np.array_equal(reference[key], layout[key])
                for key in reference
            ):
                raise ValueError(f"{name}: common clean layout changed")
            prediction = cache[prediction_key].astype(np.float32)
            if (
                prediction.shape != layout["truth"].shape
                or not np.isfinite(prediction).all()
            ):
                raise ValueError(f"{name}: invalid prediction")
            loaded[name] = prediction
    if reference is None:
        raise RuntimeError("no clean sources")
    loaded.update(reference)
    if set(loaded["folds"].astype(int).tolist()) != {0, 1, 2, 3}:
        raise ValueError("expected F0-F3 only")
    return loaded


def row_mask(
    folds: np.ndarray,
    row_starts: np.ndarray,
    selected_roles: set[int],
) -> np.ndarray:
    mask = np.zeros(int(row_starts[-1]), dtype=bool)
    for index, fold in enumerate(folds.astype(int)):
        if fold in selected_roles:
            mask[row_starts[index] : row_starts[index + 1]] = True
    return mask


def compose(
    parent: np.ndarray,
    components: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    if not np.any(weights):
        return parent.astype(np.float64).copy()
    prediction = parent.astype(np.float64).copy()
    for weight, component in zip(weights, components):
        if weight:
            prediction += float(weight) * (
                component.astype(np.float64) - parent.astype(np.float64)
            )
    return prediction


def role_and_well_gains(
    prediction: np.ndarray,
    parent: np.ndarray,
    truth: np.ndarray,
    folds: np.ndarray,
    row_starts: np.ndarray,
    roles: list[int],
) -> tuple[list[float], float, float, float]:
    role_gains = []
    well_gains = []
    parent_sse = 0.0
    candidate_sse = 0.0
    count = 0
    for role in roles:
        role_parts = []
        parent_parts = []
        truth_parts = []
        for index in np.flatnonzero(folds.astype(int) == role):
            item = slice(int(row_starts[index]), int(row_starts[index + 1]))
            role_parts.append(prediction[item])
            parent_parts.append(parent[item])
            truth_parts.append(truth[item])
            well_gains.append(
                rmse(parent[item], truth[item])
                - rmse(prediction[item], truth[item])
            )
        role_prediction = np.concatenate(role_parts)
        role_parent = np.concatenate(parent_parts)
        role_truth = np.concatenate(truth_parts)
        role_gains.append(
            rmse(role_parent, role_truth)
            - rmse(role_prediction, role_truth)
        )
        parent_error = role_parent.astype(np.float64) - role_truth
        candidate_error = role_prediction.astype(np.float64) - role_truth
        parent_sse += float(np.dot(parent_error, parent_error))
        candidate_sse += float(np.dot(candidate_error, candidate_error))
        count += len(role_truth)
    pooled_gain = float(
        np.sqrt(parent_sse / count) - np.sqrt(candidate_sse / count)
    )
    return (
        role_gains,
        pooled_gain,
        float(np.mean(well_gains)),
        float(min(role_gains)),
    )


def bounded_fit(
    parent: np.ndarray,
    components: list[np.ndarray],
    truth: np.ndarray,
    mask: np.ndarray,
    ridge: float,
) -> np.ndarray:
    selected_parent = parent[mask].astype(np.float64)
    error = selected_parent - truth[mask].astype(np.float64)
    design = np.stack(
        [
            component[mask].astype(np.float64) - selected_parent
            for component in components
        ],
        axis=1,
    )
    count = len(error)
    gram = design.T @ design / count
    linear = design.T @ error / count

    candidates = []
    # With four registered components the exact convex box solution can be
    # found by enumerating whether each coordinate is free, at zero, or at
    # its upper bound.  This avoids line-search sensitivity in L-BFGS-B.
    for states in itertools.product((-1, 0, 1), repeat=len(components)):
        weights = np.zeros(len(components), dtype=np.float64)
        upper = np.asarray(states) == 1
        free = np.asarray(states) == 0
        weights[upper] = MAX_WEIGHT
        if np.any(free):
            free_indices = np.flatnonzero(free)
            fixed_indices = np.flatnonzero(~free)
            system = gram[np.ix_(free_indices, free_indices)].copy()
            system.flat[:: len(free_indices) + 1] += ridge
            right = -linear[free_indices]
            if len(fixed_indices):
                right -= (
                    gram[np.ix_(free_indices, fixed_indices)]
                    @ weights[fixed_indices]
                )
            solution = np.linalg.lstsq(
                system, right, rcond=1e-12
            )[0]
            if np.any(solution < -1e-10) or np.any(
                solution > MAX_WEIGHT + 1e-10
            ):
                continue
            weights[free_indices] = np.clip(
                solution, 0.0, MAX_WEIGHT
            )
        objective = float(
            0.5 * weights @ gram @ weights
            + linear @ weights
            + 0.5 * ridge * (weights @ weights)
        )
        candidates.append((objective, float(weights.sum()), tuple(weights), weights))
    if not candidates:
        raise RuntimeError("exact bounded ridge enumeration found no candidate")
    weights = min(candidates, key=lambda item: item[:3])[3].copy()
    weights[weights < 1e-8] = 0.0
    prediction = compose(parent, components, weights)
    selected_score = rmse(prediction[mask], truth[mask])
    for index, weight in enumerate(weights.copy()):
        if weight == 0.0:
            continue
        removed = weights.copy()
        removed[index] = 0.0
        removed_prediction = compose(parent, components, removed)
        if rmse(removed_prediction[mask], truth[mask]) <= (
            selected_score + 1e-12
        ):
            weights[index] = 0.0
    return weights


def select_recipe(
    payload: dict[str, np.ndarray],
    selector_roles: list[int],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    parent = payload["parent"].astype(np.float64)
    components = [
        payload[name].astype(np.float64) for name in COMPONENTS
    ]
    truth = payload["truth"].astype(np.float64)
    folds = payload["folds"]
    row_starts = payload["row_starts"]
    mask = row_mask(folds, row_starts, set(selector_roles))
    records = []
    for ridge in RIDGE_GRID:
        weights = bounded_fit(
            parent, components, truth, mask, ridge
        )
        prediction = compose(parent, components, weights)
        role_gains, pooled_gain, mean_well_gain, minimum_gain = (
            role_and_well_gains(
                prediction,
                parent,
                truth,
                folds,
                row_starts,
                selector_roles,
            )
        )
        records.append(
            {
                "ridge": ridge,
                "weights": {
                    name: float(weight)
                    for name, weight in zip(COMPONENTS, weights)
                },
                "role_gains": role_gains,
                "pooled_gain": pooled_gain,
                "mean_complete_well_gain": mean_well_gain,
                "minimum_role_gain": minimum_gain,
                "sum_weights": float(weights.sum()),
                "active_components": int(np.count_nonzero(weights)),
                "eligible": bool(
                    np.count_nonzero(weights)
                    and all(gain > 0.0 for gain in role_gains)
                ),
            }
        )
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        return None, records
    selected = min(
        eligible,
        key=lambda record: (
            -round(float(record["pooled_gain"]), 12),
            -round(float(record["mean_complete_well_gain"]), 12),
            -round(float(record["minimum_role_gain"]), 12),
            -float(record["ridge"]),
            round(float(record["sum_weights"]), 12),
            int(record["active_components"]),
            tuple(record["weights"][name] for name in COMPONENTS),
        ),
    )
    return selected, records


def evaluate(
    payload: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    parent = payload["parent"].astype(np.float64)
    components = [
        payload[name].astype(np.float64) for name in COMPONENTS
    ]
    truth = payload["truth"].astype(np.float64)
    folds = payload["folds"]
    row_starts = payload["row_starts"]
    prediction = parent.copy()
    fold_records = []
    for held in range(4):
        selector_roles = [role for role in range(4) if role != held]
        selected, grid = select_recipe(payload, selector_roles)
        weights = np.zeros(len(COMPONENTS), dtype=np.float64)
        if selected is not None:
            weights = np.asarray(
                [selected["weights"][name] for name in COMPONENTS],
                dtype=np.float64,
            )
        local_prediction = compose(parent, components, weights)
        mask = row_mask(folds, row_starts, {held})
        prediction[mask] = local_prediction[mask]
        held_gain = rmse(parent[mask], truth[mask]) - rmse(
            prediction[mask], truth[mask]
        )
        fold_records.append(
            {
                "held_fold": held,
                "selector_folds": selector_roles,
                "selected": selected,
                "selector_grid": grid,
                "held_parent_rmse": rmse(parent[mask], truth[mask]),
                "held_selected_rmse": rmse(
                    prediction[mask], truth[mask]
                ),
                "held_incremental_gain": held_gain,
            }
        )
    baseline = payload["baseline"].astype(np.float64)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "clean_multimodel_bounded_ridge",
        "status": "complete_selection_clean_F0_F3",
        "source_sha256": sha256(Path(__file__)),
        "components": list(COMPONENTS),
        "ridge_grid": list(RIDGE_GRID),
        "maximum_component_weight": MAX_WEIGHT,
        "v20_rmse": rmse(baseline, truth),
        "parent_rmse": rmse(parent, truth),
        "selected_rmse": rmse(prediction, truth),
        "gain_versus_v20": rmse(baseline, truth) - rmse(
            prediction, truth
        ),
        "incremental_gain_versus_parent": rmse(parent, truth) - rmse(
            prediction, truth
        ),
        "positive_incremental_folds": int(
            sum(
                record["held_incremental_gain"] > 0.0
                for record in fold_records
            )
        ),
        "fold_records": fold_records,
        "selection_clean_nested_selector_run": True,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    output = {
        "well_ids": payload["well_ids"].astype(str),
        "folds": folds.astype(np.int8),
        "row_starts": row_starts.astype(np.int64),
        "baseline": baseline.astype(np.float32),
        "truth": truth.astype(np.float32),
        "parent": parent.astype(np.float32),
        **{
            name: payload[name].astype(np.float32)
            for name in COMPONENTS
        },
        "prediction": prediction.astype(np.float32),
        "selection_clean_nested_selector_run": np.asarray(True),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
    }
    return result, output


def synthetic_payload() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20262917)
    wells_per_role = 3
    rows_per_well = 64
    folds = np.repeat(np.arange(4, dtype=np.int8), wells_per_role)
    row_starts = np.arange(
        0, (len(folds) + 1) * rows_per_well, rows_per_well,
        dtype=np.int64,
    )
    parent = rng.normal(10000.0, 2.0, int(row_starts[-1]))
    directions = []
    for _ in COMPONENTS:
        value = rng.normal(size=len(parent))
        value -= value.mean()
        value /= np.sqrt(np.mean(np.square(value)))
        directions.append(value)
    truth = parent + 0.20 * directions[0] + 0.10 * directions[1]
    return {
        "well_ids": np.asarray(
            [f"synthetic_{index}" for index in range(len(folds))]
        ),
        "folds": folds,
        "row_starts": row_starts,
        "baseline": parent.copy(),
        "truth": truth,
        "parent": parent,
        **{
            name: parent + direction
            for name, direction in zip(COMPONENTS, directions)
        },
    }


def run_gate() -> dict[str, object]:
    real = load_sources()
    zero = compose(
        real["parent"],
        [real[name] for name in COMPONENTS],
        np.zeros(len(COMPONENTS)),
    )
    if not np.array_equal(zero, real["parent"].astype(np.float64)):
        raise AssertionError("parent zero no-op failed")
    planted = synthetic_payload()
    selected, _ = select_recipe(planted, [0, 1, 2])
    if selected is None:
        raise AssertionError("planted selector returned zero")
    planted_weights = selected["weights"]
    if (
        abs(planted_weights["D570"] - 0.20) > 0.01
        or abs(planted_weights["D072"] - 0.10) > 0.01
        or planted_weights["protected_posterior_incremental"] > 0.01
        or planted_weights["protected_mode_bank"] > 0.01
    ):
        raise AssertionError(f"planted weights not recovered: {selected}")
    changed = synthetic_payload()
    first, _ = select_recipe(changed, [1, 2, 3])
    mask = row_mask(
        changed["folds"], changed["row_starts"], {0}
    )
    changed["truth"] = changed["truth"].copy()
    changed["truth"][mask] = np.linspace(-1e6, 1e6, int(mask.sum()))
    second, _ = select_recipe(changed, [1, 2, 3])
    if first != second:
        raise AssertionError("excluded held truth changed selector")
    corrupted_rejected = False
    with tempfile.TemporaryDirectory(
        prefix="run8_clean_multimodel_gate_"
    ) as temp:
        corrupt_path = Path(temp) / "corrupt.npz"
        shutil.copyfile(ARCHIVES["protected_posterior_incremental"][0], corrupt_path)
        with corrupt_path.open("r+b") as handle:
            handle.seek(-17, 2)
            byte = handle.read(1)
            handle.seek(-1, 1)
            handle.write(bytes([byte[0] ^ 1]))
        try:
            load_sources(
                {"protected_posterior_incremental": corrupt_path}
            )
        except ValueError:
            corrupted_rejected = True
    if not corrupted_rejected:
        raise AssertionError("corrupted archive accepted")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "clean_multimodel_bounded_ridge",
        "status": "metric_free_gate_passed",
        "components": list(COMPONENTS),
        "ridge_grid": list(RIDGE_GRID),
        "maximum_component_weight": MAX_WEIGHT,
        "checks": {
            "all_archives_rehashed": True,
            "common_layout_exact": True,
            "protected_flags_false": True,
            "exact_parent_zero_noop": True,
            "planted_weights_recovered": True,
            "excluded_held_truth_invariant": True,
            "corrupted_archive_rejected": True,
        },
        "planted_selection": selected,
        "source_sha256": sha256(Path(__file__)),
        "held_target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if args.gate:
        if args.output_npz is not None:
            raise ValueError("metric-free gate does not write predictions")
        result = run_gate()
    else:
        if args.output_npz is None:
            raise ValueError("--output-npz required")
        if args.output_npz.exists():
            raise FileExistsError(f"refusing to overwrite {args.output_npz}")
        result, payload = evaluate(load_sources())
        np.savez_compressed(args.output_npz, **payload)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
