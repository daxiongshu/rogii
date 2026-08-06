from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from limited_query_cv.train_v5_batch3_run2_reverse_bank import (
    conditional_weight,
)
from limited_query_cv.v5_batch3_run4_clean_bank import (
    RUN_ID,
    RUN_ROOT,
    load_clean_fold_bank,
)


WELL_STATS = RUN_ROOT / "clean_well_stats.npz"
ROUTER = RUN_ROOT / "two_stage_router_screen_best.npz"
FRESH_ASSIGNMENT = Path(
    "limited_query_cv/runs/v5_batch3_run_003_goal_020/"
    "fresh_confirmation_assignment.json"
)
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
SURFACE_INDEX = 12
SMOOTHING_SIGMA = 127.0
FADE_ROWS = 128
ROW_SHARE = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_weight(
    labels: np.ndarray,
    held_group: int,
    cross: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    training = labels != held_group
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


def smooth_amplitude(
    raw: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    pieces = []
    for left, right in zip(starts[:-1], starts[1:]):
        values = gaussian_filter1d(
            raw[left:right].astype(np.float64),
            SMOOTHING_SIGMA,
            mode="nearest",
        )
        fade = min(FADE_ROWS, len(values))
        values[:fade] *= np.linspace(0.0, 1.0, fade)
        pieces.append(values)
    return np.concatenate(pieces)


def grouping_labels(well_ids: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(LAYOUT, allow_pickle=False) as layout:
        layout_ids = layout["well_ids"].astype(str)
        lookup = {
            well_id: index for index, well_id in enumerate(layout_ids)
        }
        balanced = np.asarray(
            [layout["stratified"][lookup[well_id]] for well_id in well_ids],
            dtype=np.int64,
        )
        independent = np.asarray(
            [layout["independent"][lookup[well_id]] for well_id in well_ids],
            dtype=np.int64,
        )
    fresh_record = json.loads(FRESH_ASSIGNMENT.read_text())
    fresh_lookup = fresh_record["assignments"]
    fresh = np.asarray(
        [fresh_lookup[well_id] for well_id in well_ids],
        dtype=np.int64,
    )
    return {
        "balanced_exposed": balanced,
        "independent_exposed": independent,
        "former_fresh_exposed": fresh,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--well-stats", type=Path, default=WELL_STATS)
    parser.add_argument("--router", type=Path, default=ROUTER)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "row_surface_router_development.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    with np.load(args.well_stats, allow_pickle=False) as stats:
        if (
            bool(stats["F4_loaded"])
            or bool(stats["F4_metric_computed"])
            or bool(stats["protected_reveal_spent"])
        ):
            raise ValueError("Run 4 well-stat firewall changed")
        well_ids = stats["well_ids"].astype(str)
        original = stats["original_folds"].astype(np.int64)
        rows = stats["rows"].astype(np.int64)
        clean_sse = stats["clean_sse"].astype(np.float64)
        cross = stats["cross"].astype(np.float64)
        gram = stats["gram"].astype(np.float64)
    with np.load(args.router, allow_pickle=False) as route:
        if (
            bool(route["F4_loaded"])
            or bool(route["F4_metric_computed"])
            or not np.array_equal(route["well_ids"].astype(str), well_ids)
        ):
            raise ValueError("two-stage router firewall changed")
        surface_coefficient = route["surface_coefficient"].astype(np.float64)
        surface_indices = route["surface_indices"].astype(np.int64)
        second_index = route["second_candidate_index"].astype(np.int64)
        second_amplitude = route["second_amplitude"].astype(np.float64)
        surface_scale = float(route["surface_scale"])
        second_scale = float(route["second_scale"])
    route_weight = np.zeros((len(well_ids), 49), dtype=np.float64)
    route_weight[:, surface_indices] += (
        surface_scale * surface_coefficient
    )
    route_weight[np.arange(len(well_ids)), second_index] += (
        second_scale * second_amplitude
    )

    new_cross = np.zeros(len(well_ids), dtype=np.float64)
    new_gram = np.zeros((len(well_ids), 49), dtype=np.float64)
    new_self = np.zeros(len(well_ids), dtype=np.float64)
    index_lookup = {well_id: index for index, well_id in enumerate(well_ids)}
    cache_records = []
    for fold in range(4):
        cache_path = RUN_ROOT / f"row_surface_router_f{fold}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                int(cache["held_fold"]) != fold
                or bool(cache["held_target_loaded"])
                or bool(cache["raw_hidden_target_stored"])
                or bool(cache["F4_loaded"])
                or bool(cache["F4_metric_computed"])
                or bool(cache["protected_reveal_spent"])
            ):
                raise ValueError(f"fold {fold}: row-router firewall changed")
            ids = cache["well_ids"].astype(str)
            starts = cache["row_starts"].astype(np.int64)
            raw = cache["raw_amplitude"].astype(np.float64)
            model_path = Path(str(cache["model_path"]))
            if sha256(model_path) != str(cache["model_sha256"]):
                raise ValueError(f"fold {fold}: row-router model changed")
        bank = load_clean_fold_bank(fold, truth_allowed=True)
        if not np.array_equal(ids, bank["ids"]):
            raise ValueError(f"fold {fold}: row-router layout changed")
        amplitude = ROW_SHARE * smooth_amplitude(raw, starts)
        correction = amplitude * bank["correction"][SURFACE_INDEX]
        error = bank["clean"] - bank["truth"]
        for local, well_id in enumerate(ids):
            left, right = starts[local : local + 2]
            global_index = index_lookup[str(well_id)]
            local_correction = correction[left:right]
            new_cross[global_index] = float(
                local_correction @ error[left:right]
            )
            new_gram[global_index] = (
                bank["correction"][:, left:right] @ local_correction
            )
            new_self[global_index] = float(
                local_correction @ local_correction
            )
        cache_records.append(
            {
                "fold": fold,
                "cache": str(cache_path),
                "cache_sha256": sha256(cache_path),
                "model": str(model_path),
                "model_sha256": sha256(model_path),
            }
        )

    geometries = {"original": original}
    geometries.update(grouping_labels(well_ids))
    results: dict[str, Any] = {}
    all_group_gains = []
    for name, labels in geometries.items():
        clean_total = float(np.sum(clean_sse))
        before_total = 0.0
        after_total = 0.0
        groups = []
        for held_group in sorted(set(labels.tolist())):
            mask = labels == held_group
            weight = base_weight(labels, held_group, cross, gram)
            combined = route_weight[mask] + weight[None, :]
            before = (
                clean_sse[mask]
                + 2.0 * np.einsum("ni,ni->n", cross[mask], combined)
                + np.einsum(
                    "ni,nij,nj->n",
                    combined,
                    gram[mask],
                    combined,
                )
            )
            row_numerator = (
                new_cross[mask]
                + np.einsum("ni,ni->n", new_gram[mask], combined)
            )
            after = before + 2.0 * row_numerator + new_self[mask]
            before_total += float(np.sum(before))
            after_total += float(np.sum(after))
            group_rows = int(np.sum(rows[mask]))
            clean_rmse = float(
                np.sqrt(np.sum(clean_sse[mask]) / group_rows)
            )
            before_rmse = float(np.sqrt(np.sum(before) / group_rows))
            after_rmse = float(np.sqrt(np.sum(after) / group_rows))
            gain = clean_rmse - after_rmse
            all_group_gains.append(gain)
            groups.append(
                {
                    "held_group": int(held_group),
                    "rows": group_rows,
                    "clean_C016_rmse": clean_rmse,
                    "routed_before_row_rmse": before_rmse,
                    "candidate_rmse": after_rmse,
                    "gain_vs_clean_C016": gain,
                    "incremental_row_gain": before_rmse - after_rmse,
                }
            )
        total_rows = int(np.sum(rows))
        clean_rmse = float(np.sqrt(clean_total / total_rows))
        before_rmse = float(np.sqrt(before_total / total_rows))
        after_rmse = float(np.sqrt(after_total / total_rows))
        results[name] = {
            "rows": total_rows,
            "clean_C016_rmse": clean_rmse,
            "routed_before_row_rmse": before_rmse,
            "candidate_rmse": after_rmse,
            "gain_vs_clean_C016": clean_rmse - after_rmse,
            "incremental_row_gain": before_rmse - after_rmse,
            "groups": groups,
        }

    pooled_gains = [
        result["gain_vs_clean_C016"] for result in results.values()
    ]
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 11,
        "run_id": RUN_ID,
        "stage": "development_exposed_clean_row_surface_router",
        "status": (
            "development_goal_passed"
            if min(pooled_gains) >= 0.20
            else "development_goal_not_yet_passed"
        ),
        "candidate": "clean two-stage router plus bounded row surface router",
        "fixed_recipe": {
            "surface_index": SURFACE_INDEX,
            "smoothing_sigma": SMOOTHING_SIGMA,
            "fade_rows": FADE_ROWS,
            "row_share": ROW_SHARE,
            "selected_from": "F0-only falsification control",
        },
        "geometries": results,
        "minimum_geometry_gain": min(pooled_gains),
        "mean_geometry_gain": float(np.mean(pooled_gains)),
        "minimum_held_group_gain": min(all_group_gains),
        "positive_held_groups": int(
            np.sum(np.asarray(all_group_gains) > 0.0)
        ),
        "held_groups": len(all_group_gains),
        "well_stats": str(args.well_stats),
        "well_stats_sha256": sha256(args.well_stats),
        "router": str(args.router),
        "router_sha256": sha256(args.router),
        "row_router_caches": cache_records,
        "firewall": {
            "development_original_folds_loaded": [0, 1, 2, 3],
            "F4_loaded": False,
            "F4_metric_computed": False,
            "protected_reveal_spent": False,
        },
        "kaggle_dataset_authorized": False,
        "kaggle_kernel_authorized": False,
        "kaggle_submission_authorized": False,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
