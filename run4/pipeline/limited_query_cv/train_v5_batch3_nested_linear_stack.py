from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


L2_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
TOTAL_SHARE_CAPS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
INDIVIDUAL_SHARE_CAP = 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stats(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if (
            bool(data["raw_hidden_target_stored"])
            or bool(data["F4_loaded"])
            or bool(data["F4_metric_computed"])
            or bool(data["protected_reveal_spent"])
        ):
            raise ValueError(f"{path}: stack stats firewall failed")
        result = {name: data[name] for name in data.files}
    result["path"] = str(path)
    result["sha256"] = sha256(path)
    return result


def aggregate(
    stats: dict[int, dict],
    folds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cross = sum(stats[fold]["cross"].astype(np.float64) for fold in folds)
    gram = sum(stats[fold]["gram"].astype(np.float64) for fold in folds)
    return cross, gram


def fit_stack(
    cross: np.ndarray,
    gram: np.ndarray,
    relative_l2: float,
    total_cap: float,
) -> np.ndarray:
    dimensions = len(cross)
    scale = float(np.trace(gram) / max(dimensions, 1))
    regularized = gram + np.eye(dimensions) * relative_l2 * scale
    objective_scale = max(
        scale,
        float(np.max(np.abs(cross))),
        1.0,
    )

    def objective(weight: np.ndarray) -> float:
        return float(
            2.0 * np.dot(cross, weight)
            + np.dot(weight, regularized @ weight)
        ) / objective_scale

    def gradient(weight: np.ndarray) -> np.ndarray:
        return (
            2.0 * cross + 2.0 * (regularized @ weight)
        ) / objective_scale

    initial = np.full(
        dimensions,
        min(total_cap / dimensions, INDIVIDUAL_SHARE_CAP / 2.0),
        dtype=np.float64,
    )
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, INDIVIDUAL_SHARE_CAP)] * dimensions,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda weight: total_cap - np.sum(weight),
                "jac": lambda weight: -np.ones_like(weight),
            }
        ],
        options={"ftol": 1e-11, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"stack optimizer failed: {result.message}")
    weight = np.clip(result.x, 0.0, INDIVIDUAL_SHARE_CAP)
    if np.sum(weight) > total_cap + 1e-8:
        raise RuntimeError("stack optimizer violated total cap")
    return weight


def score(stats: dict, weight: np.ndarray) -> tuple[float, int]:
    sse = float(
        stats["clean_sse"]
        + 2.0 * np.dot(stats["cross"], weight)
        + np.dot(weight, stats["gram"] @ weight)
    )
    return sse, int(stats["rows"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    stats = {}
    for path in args.stats:
        record = load_stats(path)
        fold = int(record["fold"])
        if fold in stats:
            raise ValueError(f"duplicate fold {fold}")
        stats[fold] = record
    if set(stats) != set(range(4)):
        raise ValueError("nested stack requires F0-F3")
    names = stats[0]["candidate_names"].astype(str)
    for record in stats.values():
        if not np.array_equal(
            record["candidate_names"].astype(str), names
        ):
            raise ValueError("stack candidate schema differs")
    selected_roles = []
    outer_sse = 0.0
    total_rows = 0
    for outer_fold in range(4):
        training_folds = [
            fold for fold in range(4) if fold != outer_fold
        ]
        trials = []
        for l2_order, relative_l2 in enumerate(L2_GRID):
            for cap_order, total_cap in enumerate(TOTAL_SHARE_CAPS):
                inner_sse = 0.0
                inner_rows = 0
                inner_records = []
                for inner_fold in training_folds:
                    fit_folds = [
                        fold
                        for fold in training_folds
                        if fold != inner_fold
                    ]
                    cross, gram = aggregate(stats, fit_folds)
                    weight = fit_stack(
                        cross, gram, relative_l2, total_cap
                    )
                    sse, rows = score(stats[inner_fold], weight)
                    inner_sse += sse
                    inner_rows += rows
                    inner_records.append(
                        {
                            "inner_fold": inner_fold,
                            "rmse": float(np.sqrt(sse / rows)),
                            "total_share": float(np.sum(weight)),
                            "nonzero_components": int(
                                np.sum(weight > 1e-8)
                            ),
                        }
                    )
                trials.append(
                    {
                        "relative_l2": relative_l2,
                        "l2_order": l2_order,
                        "total_cap": total_cap,
                        "cap_order": cap_order,
                        "inner_rmse": float(
                            np.sqrt(inner_sse / inner_rows)
                        ),
                        "inner_records": inner_records,
                    }
                )
        selected = min(
            trials,
            key=lambda row: (
                row["inner_rmse"],
                row["total_cap"],
                -row["relative_l2"],
            ),
        )
        cross, gram = aggregate(stats, training_folds)
        weight = fit_stack(
            cross,
            gram,
            selected["relative_l2"],
            selected["total_cap"],
        )
        held_sse, held_rows = score(stats[outer_fold], weight)
        clean_sse = float(stats[outer_fold]["clean_sse"])
        v20_sse = float(stats[outer_fold]["v20_sse"])
        held_rmse = float(np.sqrt(held_sse / held_rows))
        clean_rmse = float(np.sqrt(clean_sse / held_rows))
        v20_rmse = float(np.sqrt(v20_sse / held_rows))
        outer_sse += held_sse
        total_rows += held_rows
        selected_roles.append(
            {
                "outer_fold": outer_fold,
                "training_folds": training_folds,
                "selected_relative_l2": selected["relative_l2"],
                "selected_total_share_cap": selected["total_cap"],
                "inner_pooled_rmse": selected["inner_rmse"],
                "inner_records": selected["inner_records"],
                "weights": [
                    {
                        "candidate": str(name),
                        "weight": float(value),
                    }
                    for name, value in zip(names, weight)
                    if value > 1e-10
                ],
                "total_share": float(np.sum(weight)),
                "held_rmse": held_rmse,
                "held_clean_C016_rmse": clean_rmse,
                "held_v20_rmse": v20_rmse,
                "held_gain_over_clean_C016": clean_rmse - held_rmse,
                "held_gain_over_v20": v20_rmse - held_rmse,
                "held_rows": held_rows,
            }
        )
        print(
            f"outer={outer_fold} l2={selected['relative_l2']:g} "
            f"cap={selected['total_cap']:g} "
            f"share={np.sum(weight):.4f} held={held_rmse:.6f}",
            flush=True,
        )
    clean_sse = sum(float(record["clean_sse"]) for record in stats.values())
    v20_sse = sum(float(record["v20_sse"]) for record in stats.values())
    clean_score = float(np.sqrt(clean_sse / total_rows))
    v20_score = float(np.sqrt(v20_sse / total_rows))
    stack_score = float(np.sqrt(outer_sse / total_rows))
    result = {
        "schema_version": 1,
        "run_id": "v5_batch3_run_001_goal_020",
        "family": "c016_residual_gr_path_pool",
        "stage": "fully_nested_nonnegative_structural_linear_stack",
        "status": (
            "development_readiness_passed"
            if v20_score - stack_score >= 0.2
            else "development_readiness_not_yet_passed"
        ),
        "effective_readiness_target": 0.2,
        "v20_pooled_rmse": v20_score,
        "clean_C016_pooled_rmse": clean_score,
        "nested_stack_pooled_rmse": stack_score,
        "clean_C016_gain_over_v20": v20_score - clean_score,
        "incremental_gain_over_clean_C016": clean_score - stack_score,
        "total_gain_over_v20": v20_score - stack_score,
        "l2_grid": list(L2_GRID),
        "total_share_caps": list(TOTAL_SHARE_CAPS),
        "individual_share_cap": INDIVIDUAL_SHARE_CAP,
        "selected_roles": selected_roles,
        "stats": [
            {
                "fold": fold,
                "path": record["path"],
                "sha256": record["sha256"],
            }
            for fold, record in sorted(stats.items())
        ],
        "source": str(Path(__file__)),
        "source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        "F4_loaded": False,
        "F4_metric_computed": False,
        "protected_reveal_spent": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
