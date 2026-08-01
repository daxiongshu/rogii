from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


RUN_ID = "v5_batch2_run_006_goal_050"
VARIANTS = ("huber_h96", "metric_h128", "fusion_h128")
TREATMENTS = (
    "huber_h96",
    "metric_h128",
    "fusion_h128",
    "equal_huber_metric",
    "equal_all",
)
SNAPSHOTS = (4, 8, 12, 16, 24, 32)
AMPLITUDES = (1.0, 1.5, 2.0, 3.0)
SHARES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
TREATMENT_TIE_ORDER = {name: index for index, name in enumerate(TREATMENTS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def treatment_members(treatment: str) -> tuple[str, ...]:
    if treatment in VARIANTS:
        return (treatment,)
    if treatment == "equal_huber_metric":
        return ("huber_h96", "metric_h128")
    if treatment == "equal_all":
        return VARIANTS
    raise ValueError(f"unknown recurrent treatment: {treatment}")


def average_predictions(values: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values).astype(np.float64), axis=0)


def prediction_key(amplitude: float) -> str:
    return f"prediction_a{int(amplitude * 100):03d}"


def metric_sufficient_statistics(
    parent_parts: list[np.ndarray],
    truth_parts: list[np.ndarray],
    correction_parts: list[np.ndarray],
    starts_parts: list[np.ndarray],
) -> dict[str, np.ndarray | float | int]:
    pooled = np.zeros(3, dtype=np.float64)
    toe = np.zeros(3, dtype=np.float64)
    endpoint = np.zeros(3, dtype=np.float64)
    well_terms = []
    rows = 0
    toe_rows = 0
    wells = 0
    for parent, local_truth, local_correction, starts in zip(
        parent_parts,
        truth_parts,
        correction_parts,
        starts_parts,
    ):
        error = parent.astype(np.float64) - local_truth.astype(np.float64)
        correction = local_correction.astype(np.float64)
        pooled += np.asarray(
            (
                np.sum(np.square(error)),
                np.sum(error * correction),
                np.sum(np.square(correction)),
            )
        )
        rows += len(error)
        for left, right in zip(starts[:-1], starts[1:]):
            length = int(right - left)
            toe_left = int(right - max(length // 4, 1))
            local_error = error[left:right]
            local_correction = correction[left:right]
            well_terms.append(
                (
                    np.sum(np.square(local_error)),
                    np.sum(local_error * local_correction),
                    np.sum(np.square(local_correction)),
                    length,
                )
            )
            toe_error = error[toe_left:right]
            toe_correction = correction[toe_left:right]
            toe += np.asarray(
                (
                    np.sum(np.square(toe_error)),
                    np.sum(toe_error * toe_correction),
                    np.sum(np.square(toe_correction)),
                )
            )
            toe_rows += len(toe_error)
            endpoint_error = float(error[right - 1])
            endpoint_correction = float(correction[right - 1])
            endpoint += np.asarray(
                (
                    endpoint_error * endpoint_error,
                    endpoint_error * endpoint_correction,
                    endpoint_correction * endpoint_correction,
                )
            )
            wells += 1
    return {
        "pooled": pooled,
        "toe": toe,
        "endpoint": endpoint,
        "well_terms": np.asarray(well_terms, dtype=np.float64),
        "rows": rows,
        "toe_rows": toe_rows,
        "wells": wells,
    }


def quadratic_sse(terms: np.ndarray, scale: float) -> np.ndarray:
    return terms[..., 0] + 2.0 * scale * terms[..., 1] + (
        scale * scale * terms[..., 2]
    )


def blend_metrics(
    sufficient: dict[str, np.ndarray | float | int],
    scale: float,
) -> dict[str, float]:
    pooled = np.asarray(sufficient["pooled"])
    toe = np.asarray(sufficient["toe"])
    endpoint = np.asarray(sufficient["endpoint"])
    well_terms = np.asarray(sufficient["well_terms"])
    well_sse = np.maximum(quadratic_sse(well_terms[:, :3], scale), 0.0)
    return {
        "blend_rmse": float(
            np.sqrt(
                max(float(quadratic_sse(pooled, scale)), 0.0)
                / int(sufficient["rows"])
            )
        ),
        "mean_well_rmse": float(
            np.mean(np.sqrt(well_sse / well_terms[:, 3]))
        ),
        "toe_quartile_rmse": float(
            np.sqrt(
                max(float(quadratic_sse(toe, scale)), 0.0)
                / int(sufficient["toe_rows"])
            )
        ),
        "endpoint_rmse": float(
            np.sqrt(
                max(float(quadratic_sse(endpoint, scale)), 0.0)
                / int(sufficient["wells"])
            )
        ),
    }


def select_recipe(
    folds: list[dict[str, object]],
    training_folds: list[int],
    parent_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    parent_parts = [
        np.asarray(folds[fold][parent_name]) for fold in training_folds
    ]
    truth_parts = [
        np.asarray(folds[fold]["truth"]) for fold in training_folds
    ]
    starts_parts = [
        np.asarray(folds[fold]["row_starts"]) for fold in training_folds
    ]
    for treatment in TREATMENTS:
        for snapshot in SNAPSHOTS:
            members = treatment_members(treatment)
            correction_parts = []
            for fold in training_folds:
                member_corrections = [
                    np.asarray(
                        folds[fold]["corrections"][(variant, snapshot)]
                    )
                    for variant in members
                ]
                correction_parts.append(average_predictions(member_corrections))
            sufficient = metric_sufficient_statistics(
                parent_parts,
                truth_parts,
                correction_parts,
                starts_parts,
            )
            for amplitude in AMPLITUDES:
                share_scores = []
                for share in SHARES:
                    metrics = blend_metrics(
                        sufficient,
                        amplitude * share,
                    )
                    share_scores.append((metrics["blend_rmse"], share, metrics))
                _, share, metrics = min(share_scores)
                records.append(
                    {
                        "treatment": treatment,
                        "snapshot": snapshot,
                        "amplitude": amplitude,
                        "share": share,
                        "effective_scale": amplitude * share,
                        **metrics,
                    }
                )
    metric_names = (
        "blend_rmse",
        "mean_well_rmse",
        "toe_quartile_rmse",
        "endpoint_rmse",
    )
    ranks = np.column_stack(
        [
            rankdata([record[name] for record in records], method="average")
            for name in metric_names
        ]
    )
    for record, local_ranks in zip(records, ranks):
        record["metric_ranks"] = {
            name: float(value)
            for name, value in zip(metric_names, local_ranks.tolist())
        }
        record["mean_rank"] = float(np.mean(local_ranks))
    selected = min(
        records,
        key=lambda record: (
            record["mean_rank"],
            record["blend_rmse"],
            record["mean_well_rmse"],
            record["toe_quartile_rmse"],
            record["endpoint_rmse"],
            record["share"],
            record["amplitude"],
            len(treatment_members(str(record["treatment"]))),
            TREATMENT_TIE_ORDER[str(record["treatment"])],
            record["snapshot"],
        ),
    )
    return selected, records


def load_fold(root: Path, fold: int) -> dict[str, object]:
    well_ids = None
    row_starts = None
    baseline = None
    truth = None
    corrections: dict[tuple[str, int], np.ndarray] = {}
    hashes: dict[tuple[str, int], str] = {}
    for variant in VARIANTS:
        result_path = root / f"recurrent_residual_{variant}_f{fold}" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result["status"] != "complete"
            or result["variant"] != variant
            or int(result["fold"]) != fold
            or result["audit_fold_loaded"]
            or result["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError(f"{result_path}: recurrent result firewall failed")
        snapshots = {
            int(record["epoch"]): record for record in result["snapshots"]
        }
        for snapshot in SNAPSHOTS:
            path = (
                root
                / f"recurrent_residual_{variant}_f{fold}"
                / f"epoch{snapshot:03d}_prediction.npz"
            )
            if sha256(path) != snapshots[snapshot]["prediction_sha256"]:
                raise RuntimeError(f"{path}: recurrent prediction hash changed")
            with np.load(path, allow_pickle=False) as cache:
                local_ids = cache["well_ids"].astype(str)
                local_starts = cache["row_starts"].astype(np.int64)
                local_baseline = cache["baseline"].astype(np.float32)
                local_truth = cache["truth"].astype(np.float32)
                if bool(cache["audit_fold_loaded"]) or bool(
                    cache["confirmation_regroupings_loaded"]
                ):
                    raise RuntimeError(f"{path}: prediction firewall failed")
                correction = (
                    cache[prediction_key(1.0)].astype(np.float64)
                    - local_baseline.astype(np.float64)
                )
            if well_ids is None:
                well_ids = local_ids
                row_starts = local_starts
                baseline = local_baseline
                truth = local_truth
            elif not (
                np.array_equal(well_ids, local_ids)
                and np.array_equal(row_starts, local_starts)
                and np.array_equal(baseline, local_baseline)
                and np.array_equal(truth, local_truth)
            ):
                raise RuntimeError(f"{path}: recurrent fold layout changed")
            corrections[(variant, snapshot)] = correction
            hashes[(variant, snapshot)] = sha256(path)
    if any(value is None for value in (well_ids, row_starts, baseline, truth)):
        raise RuntimeError(f"F{fold}: no recurrent predictions loaded")
    return {
        "well_ids": well_ids,
        "row_starts": row_starts,
        "baseline": baseline,
        "truth": truth,
        "corrections": corrections,
        "hashes": hashes,
    }


def attach_d209_parent(folds: list[dict[str, object]], path: Path) -> None:
    with np.load(path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError(f"{path}: D209 firewall failed")
        all_ids = cache["well_ids"].astype(str)
        all_folds = cache["folds"].astype(np.int8)
        all_starts = cache["row_starts"].astype(np.int64)
        all_baseline = cache["baseline"].astype(np.float32)
        all_truth = cache["truth"].astype(np.float32)
        all_prediction = cache["prediction"].astype(np.float32)
    for fold, record in enumerate(folds):
        indices = np.flatnonzero(all_folds == fold)
        ids = all_ids[indices]
        lengths = all_starts[indices + 1] - all_starts[indices]
        starts = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        baseline = np.concatenate(
            [all_baseline[all_starts[index] : all_starts[index + 1]] for index in indices]
        )
        truth = np.concatenate(
            [all_truth[all_starts[index] : all_starts[index + 1]] for index in indices]
        )
        prediction = np.concatenate(
            [
                all_prediction[all_starts[index] : all_starts[index + 1]]
                for index in indices
            ]
        )
        if not (
            np.array_equal(ids, record["well_ids"])
            and np.array_equal(starts, record["row_starts"])
            and np.array_equal(baseline, record["baseline"])
            and np.array_equal(truth, record["truth"])
        ):
            raise RuntimeError(f"F{fold}: D209/recurrent layout changed")
        record["d209"] = prediction


def implementation_gate() -> dict[str, object]:
    source = np.asarray((1.0, 3.0, 5.0, 7.0), dtype=np.float64)
    metric = np.asarray((3.0, 5.0, 7.0, 9.0), dtype=np.float64)
    fusion = np.asarray((5.0, 7.0, 9.0, 11.0), dtype=np.float64)
    pair = average_predictions([source, metric])
    all_bag = average_predictions([source, metric, fusion])
    v20 = np.asarray((11000.0, 11001.0, 11002.0, 11003.0))
    d209 = v20 + np.asarray((0.0, 0.2, -0.1, 0.3))
    synthetic_folds: list[dict[str, object]] = []
    for fold in range(4):
        target = np.asarray(
            (0.0, 0.5, 1.0, 1.5, 0.0, -0.5, -1.0, -1.5),
            dtype=np.float64,
        )
        corrections: dict[tuple[str, int], np.ndarray] = {}
        for snapshot in SNAPSHOTS:
            corrections[("huber_h96", snapshot)] = target.copy()
            corrections[("metric_h128", snapshot)] = -target
            corrections[("fusion_h128", snapshot)] = 0.25 * target
        synthetic_folds.append(
            {
                "well_ids": np.asarray((f"f{fold}a", f"f{fold}b")),
                "row_starts": np.asarray((0, 4, 8), dtype=np.int64),
                "baseline": np.zeros(8, dtype=np.float64),
                "d209": np.full(8, 0.1, dtype=np.float64),
                "truth": target,
                "corrections": corrections,
            }
        )
    held_exclusion = True
    selected_treatments = []
    for held in range(4):
        training = [fold for fold in range(4) if fold != held]
        selected, _ = select_recipe(synthetic_folds, training, "baseline")
        altered = [dict(record) for record in synthetic_folds]
        altered[held]["truth"] = np.full(8, 1000.0 + held)
        selected_after_change, _ = select_recipe(altered, training, "baseline")
        identity = (
            selected["treatment"],
            selected["snapshot"],
            selected["amplitude"],
            selected["share"],
        )
        changed_identity = (
            selected_after_change["treatment"],
            selected_after_change["snapshot"],
            selected_after_change["amplitude"],
            selected_after_change["share"],
        )
        selected_treatments.append(str(selected["treatment"]))
        held_exclusion &= identity == changed_identity
    checks = {
        "equal_huber_metric_exact": np.array_equal(
            pair, (source + metric) / 2.0
        ),
        "equal_all_exact": np.array_equal(
            all_bag, (source + metric + fusion) / 3.0
        ),
        "v20_zero_share_exact": np.array_equal(v20 + 0.0 * pair, v20),
        "d209_zero_share_exact": np.array_equal(d209 + 0.0 * pair, d209),
        "finite": bool(np.isfinite(pair).all() and np.isfinite(all_bag).all()),
        "stable_layout": bool(
            all(
                len(record["well_ids"]) + 1 == len(record["row_starts"])
                and record["row_starts"][0] == 0
                and record["row_starts"][-1] == len(record["truth"])
                and np.all(np.diff(record["row_starts"]) > 0)
                for record in synthetic_folds
            )
        ),
        "held_exclusion_contract": held_exclusion,
        "constructed_huber_route_recovered": all(
            treatment == "huber_h96" for treatment in selected_treatments
        ),
    }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_recurrent_residual_refinement",
        "variant": "recurrent_clean_selector_implementation_gate",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "source_sha256": sha256(Path(__file__)),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }


def evaluate(
    root: Path,
    d209_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds = [load_fold(root, fold) for fold in range(4)]
    attach_d209_parent(folds, d209_path)
    modes = {"replacement": "baseline", "incremental_after_D209": "d209"}
    mode_results = {}
    mode_predictions = {}
    for mode, parent_name in modes.items():
        selected_predictions = []
        selected_records = []
        for held in range(4):
            training = [fold for fold in range(4) if fold != held]
            selected, grid = select_recipe(folds, training, parent_name)
            treatment = str(selected["treatment"])
            snapshot = int(selected["snapshot"])
            amplitude = float(selected["amplitude"])
            share = float(selected["share"])
            members = treatment_members(treatment)
            correction = average_predictions(
                [
                    np.asarray(folds[held]["corrections"][(variant, snapshot)])
                    for variant in members
                ]
            )
            parent = np.asarray(folds[held][parent_name])
            prediction = parent + amplitude * share * correction
            truth = np.asarray(folds[held]["truth"])
            selected_predictions.append(prediction.astype(np.float32))
            selected_records.append(
                {
                    "held_fold": held,
                    "selection_folds": training,
                    "selected": selected,
                    "training_grid": grid,
                    "held_parent_rmse": rmse(parent, truth),
                    "held_selected_rmse": rmse(prediction, truth),
                    "held_gain": rmse(parent, truth) - rmse(prediction, truth),
                }
            )
        parent = np.concatenate(
            [np.asarray(fold[parent_name]) for fold in folds]
        )
        truth = np.concatenate([np.asarray(fold["truth"]) for fold in folds])
        prediction = np.concatenate(selected_predictions)
        parent_score = rmse(parent, truth)
        selected_score = rmse(prediction, truth)
        mode_results[mode] = {
            "parent_rmse": parent_score,
            "selected_rmse": selected_score,
            "gain": parent_score - selected_score,
            "fold_records": selected_records,
        }
        mode_predictions[mode] = prediction.astype(np.float32)
    baseline = np.concatenate([np.asarray(fold["baseline"]) for fold in folds])
    d209 = np.concatenate([np.asarray(fold["d209"]) for fold in folds])
    truth = np.concatenate([np.asarray(fold["truth"]) for fold in folds])
    return (
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "family": "complete_well_recurrent_residual_refinement",
            "variant": "selection_clean_recurrent_diversity_expansion",
            "status": "complete_selection_clean_F0_F3",
            "treatments": list(TREATMENTS),
            "snapshots": list(SNAPSHOTS),
            "amplitudes": list(AMPLITUDES),
            "shares": list(SHARES),
            "modes": mode_results,
            "source_sha256": sha256(Path(__file__)),
            "d209_path": str(d209_path),
            "d209_sha256": sha256(d209_path),
            "audit_fold_loaded": False,
            "confirmation_regroupings_loaded": False,
        },
        {
            "well_ids": np.concatenate(
                [np.asarray(fold["well_ids"]) for fold in folds]
            ),
            "folds": np.concatenate(
                [
                    np.full(len(np.asarray(fold["well_ids"])), index, dtype=np.int8)
                    for index, fold in enumerate(folds)
                ]
            ),
            "row_starts": np.r_[
                0,
                np.cumsum(
                    np.concatenate(
                        [
                            np.diff(np.asarray(fold["row_starts"]))
                            for fold in folds
                        ]
                    )
                ),
            ].astype(np.int64),
            "baseline": baseline.astype(np.float32),
            "d209": d209.astype(np.float32),
            "truth": truth.astype(np.float32),
            "replacement": mode_predictions["replacement"],
            "incremental_after_D209": mode_predictions["incremental_after_D209"],
            "audit_fold_loaded": np.asarray(False),
            "confirmation_regroupings_loaded": np.asarray(False),
        },
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--d209", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    if args.gate:
        if args.root is not None or args.d209 is not None or args.output_npz is not None:
            raise ValueError("gate accepts only --output-json")
        payload = implementation_gate()
        write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if payload["status"] != "passed":
            raise RuntimeError("recurrent clean selector gate failed")
        return
    if args.root is None or args.d209 is None or args.output_npz is None:
        raise ValueError("evaluation requires --root, --d209, and --output-npz")
    if args.output_npz.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_npz}")
    result, cache = evaluate(args.root, args.d209)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **cache)
    result["prediction"] = str(args.output_npz)
    result["prediction_sha256"] = sha256(args.output_npz)
    write_json(args.output_json, result)
    print(json.dumps(result["modes"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
