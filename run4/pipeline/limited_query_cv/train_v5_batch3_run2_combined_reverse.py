from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from limited_query_cv.train_v5_batch3_nested_linear_stack import (
    INDIVIDUAL_SHARE_CAP,
    L2_GRID,
    TOTAL_SHARE_CAPS,
    aggregate,
    load_stats,
    score,
    sha256,
)
from limited_query_cv.train_v5_batch3_run2_reverse_bank import (
    conditional_weight,
)
from limited_query_cv.v5_batch3_run2_historical_stack import (
    HISTORICAL_AMPLITUDES,
)
from limited_query_cv.v5_batch3_run2_reverse_residual_bank import (
    REVERSE_TOTAL_CAPS,
)


EXPECTED_REVERSE_NAMES = [
    "reverse_physics_prior_mixed",
    "reverse_local_residual_seed_bag",
    "reverse_multiscale_residual_bag",
    "reverse_protected_mode_maximin",
    "reverse_target_surface",
    "reverse_local_residual_snapshot_bag",
    "reverse_multi_anchor_phase",
    "reverse_hardtail",
    "reverse_mtp",
    "reverse_trajectory_direction_projection",
]


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
        if (
            "reverse_candidate_indices" not in record
            or bool(
                record[
                    "archive_target_fields_loaded_by_candidate_loader"
                ]
            )
        ):
            raise ValueError("combined reverse stats firewall failed")
        stats[fold] = record
    if set(stats) != set(range(4)):
        raise ValueError("combined reverse stack requires F0-F3")
    names = stats[0]["candidate_names"].astype(str)
    historical_index = int(stats[0]["historical_candidate_index"])
    reverse_indices = stats[0]["reverse_candidate_indices"].astype(
        np.int64
    )
    if (
        names[historical_index] != "historical_clean_multimodel_ridge"
        or list(names[reverse_indices]) != EXPECTED_REVERSE_NAMES
        or any(
            not np.array_equal(record["candidate_names"], names)
            or int(record["historical_candidate_index"])
            != historical_index
            or not np.array_equal(
                record["reverse_candidate_indices"], reverse_indices
            )
            for record in stats.values()
        )
    ):
        raise ValueError("combined reverse schema changed")

    selected_roles = []
    outer_sse = 0.0
    total_rows = 0
    for outer_fold in range(4):
        training_folds = [
            fold for fold in range(4) if fold != outer_fold
        ]
        trials = []
        for amplitude in HISTORICAL_AMPLITUDES:
            for relative_l2 in L2_GRID:
                for original_cap in TOTAL_SHARE_CAPS:
                    for reverse_cap in REVERSE_TOTAL_CAPS:
                        inner_sse = 0.0
                        inner_rows = 0
                        inner_records = []
                        eligible = True
                        for inner_fold in training_folds:
                            fit_folds = [
                                fold
                                for fold in training_folds
                                if fold != inner_fold
                            ]
                            cross, gram = aggregate(stats, fit_folds)
                            weight = conditional_weight(
                                cross,
                                gram,
                                historical_index,
                                reverse_indices,
                                amplitude,
                                relative_l2,
                                original_cap,
                                reverse_cap,
                            )
                            sse, rows = score(
                                stats[inner_fold], weight
                            )
                            clean_sse = float(
                                stats[inner_fold]["clean_sse"]
                            )
                            eligible &= (
                                sse <= clean_sse + 1e-7 * rows
                            )
                            inner_sse += sse
                            inner_rows += rows
                            inner_records.append(
                                {
                                    "inner_fold": inner_fold,
                                    "rmse": float(
                                        np.sqrt(sse / rows)
                                    ),
                                    "clean_C016_rmse": float(
                                        np.sqrt(clean_sse / rows)
                                    ),
                                    "gain_over_clean_C016": float(
                                        np.sqrt(clean_sse / rows)
                                        - np.sqrt(sse / rows)
                                    ),
                                }
                            )
                        trials.append(
                            {
                                "historical_amplitude": amplitude,
                                "relative_l2": relative_l2,
                                "original_cap": original_cap,
                                "reverse_cap": reverse_cap,
                                "eligible": bool(eligible),
                                "inner_rmse": float(
                                    np.sqrt(inner_sse / inner_rows)
                                ),
                                "inner_records": inner_records,
                            }
                        )
        eligible_trials = [row for row in trials if row["eligible"]]
        selected = min(
            eligible_trials,
            key=lambda row: (
                row["inner_rmse"],
                row["reverse_cap"],
                row["historical_amplitude"],
                row["original_cap"],
                -row["relative_l2"],
            ),
        )
        cross, gram = aggregate(stats, training_folds)
        weight = conditional_weight(
            cross,
            gram,
            historical_index,
            reverse_indices,
            selected["historical_amplitude"],
            selected["relative_l2"],
            selected["original_cap"],
            selected["reverse_cap"],
        )
        held_sse, held_rows = score(stats[outer_fold], weight)
        clean_sse = float(stats[outer_fold]["clean_sse"])
        v20_sse = float(stats[outer_fold]["v20_sse"])
        outer_sse += held_sse
        total_rows += held_rows
        selected_roles.append(
            {
                "outer_fold": outer_fold,
                "training_folds": training_folds,
                "selected_historical_amplitude": selected[
                    "historical_amplitude"
                ],
                "selected_relative_l2": selected["relative_l2"],
                "selected_original_cap": selected["original_cap"],
                "selected_reverse_cap": selected["reverse_cap"],
                "inner_pooled_rmse": selected["inner_rmse"],
                "inner_records": selected["inner_records"],
                "weights": [
                    {"candidate": str(name), "weight": float(value)}
                    for name, value in zip(names, weight)
                    if value > 1e-10
                ],
                "held_rmse": float(
                    np.sqrt(held_sse / held_rows)
                ),
                "held_clean_C016_rmse": float(
                    np.sqrt(clean_sse / held_rows)
                ),
                "held_v20_rmse": float(
                    np.sqrt(v20_sse / held_rows)
                ),
                "held_gain_over_clean_C016": float(
                    np.sqrt(clean_sse / held_rows)
                    - np.sqrt(held_sse / held_rows)
                ),
                "held_rows": held_rows,
            }
        )
        print(
            f"outer={outer_fold} hist={selected['historical_amplitude']:.2f} "
            f"oldcap={selected['original_cap']:g} "
            f"revcap={selected['reverse_cap']:g} "
            f"held={np.sqrt(held_sse / held_rows):.6f}",
            flush=True,
        )

    clean_sse = sum(
        float(record["clean_sse"]) for record in stats.values()
    )
    v20_sse = sum(
        float(record["v20_sse"]) for record in stats.values()
    )
    clean_score = float(np.sqrt(clean_sse / total_rows))
    v20_score = float(np.sqrt(v20_sse / total_rows))
    stack_score = float(np.sqrt(outer_sse / total_rows))
    result = {
        "schema_version": 1,
        "protocol_revision": 11,
        "run_id": "v5_batch3_run_002_goal_020",
        "family": "c016_buda_delayed_dip_cluster_surface",
        "stage": "fully_nested_combined_ten_reverse_bank",
        "status": (
            "development_readiness_passed"
            if clean_score - stack_score >= 0.2
            else "development_readiness_not_yet_passed"
        ),
        "effective_readiness_target_over_original_C016": 0.2,
        "v20_pooled_rmse": v20_score,
        "original_C016_pooled_rmse": clean_score,
        "nested_stack_pooled_rmse": stack_score,
        "incremental_gain_over_original_C016": clean_score - stack_score,
        "total_gain_over_v20": v20_score - stack_score,
        "historical_amplitudes": list(HISTORICAL_AMPLITUDES),
        "relative_l2_grid": list(L2_GRID),
        "original_cap_grid": list(TOTAL_SHARE_CAPS),
        "reverse_cap_grid": list(REVERSE_TOTAL_CAPS),
        "individual_share_cap": INDIVIDUAL_SHARE_CAP,
        "inner_non_regression_required": True,
        "selected_roles": selected_roles,
        "stats": [
            {
                "fold": fold,
                "path": record["path"],
                "sha256": record["sha256"],
            }
            for fold, record in sorted(stats.items())
        ],
        "F4_loaded": False,
        "F4_metric_computed": False,
        "protected_reveal_spent": False,
        "source": str(Path(__file__)),
        "source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
