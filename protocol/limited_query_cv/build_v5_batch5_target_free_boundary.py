from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from limited_query_cv.v5_batch3_run2_surface import typewell_clusters


DATA_ROOT = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
SEED = 20260811
RESTARTS = 256
FOLDS = 5
LEGAL_HORIZONTAL_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_bins(values: np.ndarray, bins: int = 5) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.searchsorted(edges, values, side="right").astype(np.int8)


def load_target_free_well_summary(data_root: Path) -> tuple[list[str], np.ndarray, list[str]]:
    train_root = data_root / "train"
    wells = sorted(path.name.split("__")[0] for path in train_root.glob("*__horizontal_well.csv"))
    rows = []
    for index, well_id in enumerate(wells, start=1):
        frame = pd.read_csv(
            train_root / f"{well_id}__horizontal_well.csv",
            usecols=list(LEGAL_HORIZONTAL_COLUMNS),
        )
        hidden = frame["TVT_input"].isna().to_numpy()
        visible = ~hidden
        md = frame["MD"].to_numpy(np.float64)
        x = frame["X"].to_numpy(np.float64)
        y = frame["Y"].to_numpy(np.float64)
        z = frame["Z"].to_numpy(np.float64)
        gr = frame["GR"].to_numpy(np.float64)
        dz = np.gradient(z, md) if len(md) > 1 else np.zeros_like(z)
        curvature = np.gradient(dz, md) if len(md) > 2 else np.zeros_like(z)
        visible_tvt = frame.loc[visible, "TVT_input"].to_numpy(np.float64)
        visible_md = md[visible]
        prefix_slope = (
            float(np.polyfit(visible_md[-min(64, len(visible_md)) :] - visible_md[-1], visible_tvt[-min(64, len(visible_tvt)) :], 1)[0])
            if len(visible_md) >= 2 and np.ptp(visible_md[-min(64, len(visible_md)) :]) > 0
            else 0.0
        )
        rows.append(
            [
                float(hidden.sum()),
                float(visible.sum()),
                float(len(frame)),
                float(md[-1] - md[0]) if len(md) else 0.0,
                float(np.nanmedian(np.abs(dz[hidden]))) if np.any(hidden) else 0.0,
                float(np.nanmedian(np.abs(curvature[hidden]))) if np.any(hidden) else 0.0,
                float(np.nanstd(gr[hidden])) if np.any(hidden) else 0.0,
                float(np.nanmean(gr[hidden])) if np.any(hidden) else 0.0,
                prefix_slope,
                float(np.nanmedian(x)),
                float(np.nanmedian(y)),
                float(np.nanmedian(z)),
            ]
        )
        if index % 100 == 0:
            print(f"target-free summaries {index}/{len(wells)}", flush=True)
    names = [
        "hidden_rows",
        "visible_rows",
        "total_rows",
        "md_span",
        "median_abs_dz_dmd_hidden",
        "median_abs_d2z_dmd2_hidden",
        "hidden_gr_std",
        "hidden_gr_mean",
        "visible_prefix_tvt_slope",
        "median_x",
        "median_y",
        "median_z",
    ]
    return wells, np.asarray(rows, dtype=np.float64), names


def load_clusters(data_root: Path, wells: list[str]) -> dict[str, int]:
    typewells = {}
    for index, well_id in enumerate(wells, start=1):
        frame = pd.read_csv(
            data_root / "train" / f"{well_id}__typewell.csv",
            usecols=["TVT", "GR"],
        )
        typewells[well_id] = (
            frame["TVT"].to_numpy(np.float64),
            frame["GR"].to_numpy(np.float64),
        )
        if index % 100 == 0:
            print(f"typewell inputs {index}/{len(wells)}", flush=True)
    return typewell_clusters(typewells)


def balance_matrix(summary: np.ndarray) -> tuple[np.ndarray, list[str]]:
    columns = [
        np.ones(len(summary), dtype=np.float64),
        summary[:, 0],
        summary[:, 1],
        summary[:, 3],
    ]
    names = ["well_count", "hidden_rows", "visible_rows", "md_span"]
    bin_specs = (
        (0, "hidden_rows"),
        (4, "abs_dip"),
        (6, "hidden_gr_std"),
        (8, "prefix_slope"),
        (9, "x"),
        (10, "y"),
    )
    for column, label in bin_specs:
        bins = quantile_bins(summary[:, column])
        for value in range(5):
            columns.append((bins == value).astype(np.float64))
            names.append(f"{label}_q{value}")
    return np.column_stack(columns), names


def assign_clusters(
    wells: list[str],
    clusters: dict[str, int],
    well_matrix: np.ndarray,
) -> tuple[dict[int, int], dict]:
    cluster_ids = np.asarray([clusters[well_id] for well_id in wells], dtype=np.int64)
    unique = np.unique(cluster_ids)
    cluster_matrix = np.stack([well_matrix[cluster_ids == cluster].sum(axis=0) for cluster in unique])
    target = cluster_matrix.sum(axis=0) / FOLDS
    scale = np.maximum(target, 1.0)
    size_priority = np.sum(np.square(cluster_matrix / scale), axis=1)
    rng = np.random.default_rng(SEED)
    best = None
    for restart in range(RESTARTS):
        jitter = rng.uniform(0.0, 1e-8, size=len(unique))
        order = np.argsort(-(size_priority + jitter))
        loads = np.zeros((FOLDS, cluster_matrix.shape[1]), dtype=np.float64)
        counts = np.zeros(FOLDS, dtype=np.int64)
        assignment = np.full(len(unique), -1, dtype=np.int8)
        fold_tie_order = rng.permutation(FOLDS)
        for cluster_index in order:
            scores = []
            for fold in range(FOLDS):
                trial = loads.copy()
                trial[fold] += cluster_matrix[cluster_index]
                normalized = (trial - target) / scale
                imbalance = float(np.sum(np.square(normalized)))
                empty_penalty = 0.001 * float(np.square(counts[fold] + 1))
                scores.append((imbalance + empty_penalty, int(np.flatnonzero(fold_tie_order == fold)[0]), fold))
            selected = min(scores)[2]
            assignment[cluster_index] = selected
            loads[selected] += cluster_matrix[cluster_index]
            counts[selected] += 1
        normalized = (loads - target) / scale
        objective = float(np.sum(np.square(normalized)))
        candidate = (objective, assignment.copy(), loads.copy(), counts.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    objective, assignment, loads, counts = best

    # Designate the target-free most representative fold as F4. The remaining
    # fold numbers are assigned deterministically by their hidden-row load.
    fold_distance = np.sum(np.square((loads - target) / scale), axis=1)
    audit_old = int(np.argmin(fold_distance))
    remaining = [fold for fold in range(FOLDS) if fold != audit_old]
    remaining.sort(key=lambda fold: (loads[fold, 1], fold))
    remap = {old: new for new, old in enumerate(remaining)}
    remap[audit_old] = 4
    cluster_to_fold = {
        int(cluster): int(remap[int(assignment[index])]) for index, cluster in enumerate(unique)
    }
    remapped_loads = np.zeros_like(loads)
    remapped_counts = np.zeros_like(counts)
    for old in range(FOLDS):
        remapped_loads[remap[old]] = loads[old]
        remapped_counts[remap[old]] = counts[old]
    return cluster_to_fold, {
        "objective": objective,
        "target": target,
        "loads": remapped_loads,
        "cluster_counts": remapped_counts,
        "audit_fold_selection": "minimum target-free normalized balance distance",
        "audit_old_fold": audit_old,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    wells, summary, summary_names = load_target_free_well_summary(args.data_root)
    clusters = load_clusters(args.data_root, wells)
    matrix, balance_names = balance_matrix(summary)
    cluster_to_fold, balance = assign_clusters(wells, clusters, matrix)
    assignments = {well_id: cluster_to_fold[clusters[well_id]] for well_id in wells}
    cluster_members = {}
    for well_id in wells:
        cluster_members.setdefault(str(clusters[well_id]), []).append(well_id)
    fold_summary = []
    for fold in range(FOLDS):
        selected = np.asarray([assignments[well_id] == fold for well_id in wells])
        fold_summary.append(
            {
                "fold": fold,
                "well_count": int(selected.sum()),
                "cluster_count": int(len({clusters[well_id] for well_id in np.asarray(wells)[selected]})),
                "hidden_rows": int(summary[selected, 0].sum()),
                "visible_rows": int(summary[selected, 1].sum()),
                "md_span": float(summary[selected, 3].sum()),
                "balance_vector": balance["loads"][fold].tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 14,
        "boundary_id": "v5_batch5_target_free_cluster_balanced",
        "status": "frozen_before_any_batch5_target_metric",
        "seed": SEED,
        "restarts": RESTARTS,
        "fold_count": FOLDS,
        "development_folds": [0, 1, 2, 3],
        "prospective_audit_fold": 4,
        "cluster_rule": "exact-overlap typewell TVT/GR clustering; every cluster remains wholly inside one fold",
        "assignment_rule": "deterministic multistart greedy balancing of inference-visible well proxies, with the most representative target-free fold designated F4",
        "legal_horizontal_columns": list(LEGAL_HORIZONTAL_COLUMNS),
        "legal_typewell_columns": ["TVT", "GR"],
        "summary_feature_names": summary_names,
        "balance_feature_names": balance_names,
        "objective": balance["objective"],
        "audit_fold_selection": balance["audit_fold_selection"],
        "assignments": assignments,
        "clusters": {well_id: int(clusters[well_id]) for well_id in wells},
        "cluster_members": cluster_members,
        "fold_summary": fold_summary,
        "target_fields_read": False,
        "horizontal_TVT_read": False,
        "train_only_horizontal_columns_read": False,
        "source": str(Path(__file__)),
        "source_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"assignments", "clusters", "cluster_members"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
