from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from limited_query_cv.build_v5_batch5_target_free_boundary import (
    DATA_ROOT,
    LEGAL_HORIZONTAL_COLUMNS,
    balance_matrix,
    load_clusters,
    load_target_free_well_summary,
)


SEED = 20260831
RESTARTS = 512
FOLDS = 5
AUDIT_FRACTION = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_audit_wells(
    wells: list[str],
    clusters: dict[str, int],
    matrix: np.ndarray,
) -> tuple[set[str], dict[str, object]]:
    by_cluster: dict[int, np.ndarray] = {}
    for cluster in sorted(set(clusters.values())):
        by_cluster[cluster] = np.asarray(
            [index for index, well_id in enumerate(wells) if clusters[well_id] == cluster],
            dtype=np.int64,
        )
    target_count = int(round(AUDIT_FRACTION * len(wells)))
    expected = matrix.sum(axis=0) * AUDIT_FRACTION
    scale = np.maximum(expected, 1.0)
    rng = np.random.default_rng(SEED)
    best: tuple[float, np.ndarray, dict[int, int]] | None = None
    for _ in range(RESTARTS):
        quotas = {
            cluster: min(len(indices) - 1, int(np.floor(AUDIT_FRACTION * len(indices))))
            if len(indices) > 1
            else 0
            for cluster, indices in by_cluster.items()
        }
        capacity = {
            cluster: (len(indices) - 1 if len(indices) > 1 else 1) - quotas[cluster]
            for cluster, indices in by_cluster.items()
        }
        remaining = target_count - sum(quotas.values())
        priorities = []
        for cluster, indices in by_cluster.items():
            if capacity[cluster] <= 0:
                continue
            fractional = AUDIT_FRACTION * len(indices) - np.floor(
                AUDIT_FRACTION * len(indices)
            )
            priorities.append((-(fractional + rng.uniform(0.0, 1.0)), cluster))
        for _, cluster in sorted(priorities)[:remaining]:
            quotas[cluster] += 1
        chosen = []
        for cluster, indices in by_cluster.items():
            quota = quotas[cluster]
            if quota:
                chosen.extend(rng.choice(indices, size=quota, replace=False).tolist())
        chosen_array = np.asarray(sorted(chosen), dtype=np.int64)
        load = matrix[chosen_array].sum(axis=0)
        balance_error = float(np.sum(np.square((load - expected) / scale)))
        count_error = float((len(chosen_array) - target_count) ** 2)
        unseen_singletons = sum(
            quotas[cluster] for cluster, indices in by_cluster.items() if len(indices) == 1
        )
        candidate = (
            balance_error + 1000.0 * count_error + 1e-6 * unseen_singletons,
            chosen_array,
            quotas,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    selected = {wells[index] for index in best[1]}
    return selected, {
        "objective": best[0],
        "target_well_count": target_count,
        "selected_well_count": len(selected),
        "expected_balance_vector": expected.tolist(),
        "selected_balance_vector": matrix[best[1]].sum(axis=0).tolist(),
        "cluster_quotas": {str(key): value for key, value in sorted(best[2].items())},
    }


def assign_development_wells(
    wells: list[str],
    clusters: dict[str, int],
    matrix: np.ndarray,
    audit_wells: set[str],
) -> tuple[dict[str, int], dict[str, object]]:
    available = np.asarray([well_id not in audit_wells for well_id in wells])
    target = matrix[available].sum(axis=0) / 4.0
    scale = np.maximum(target, 1.0)
    by_cluster = {
        cluster: np.asarray(
            [
                index
                for index, well_id in enumerate(wells)
                if clusters[well_id] == cluster and well_id not in audit_wells
            ],
            dtype=np.int64,
        )
        for cluster in sorted(set(clusters.values()))
    }
    by_cluster = {cluster: indices for cluster, indices in by_cluster.items() if len(indices)}
    priority = sorted(
        by_cluster,
        key=lambda cluster: (
            -float(np.sum(np.square(matrix[by_cluster[cluster]].sum(axis=0) / scale))),
            cluster,
        ),
    )
    rng = np.random.default_rng(SEED + 1)
    best: tuple[float, dict[str, int], np.ndarray] | None = None
    for _ in range(RESTARTS):
        loads = np.zeros((4, matrix.shape[1]), dtype=np.float64)
        assignment: dict[str, int] = {}
        for cluster in priority:
            indices = by_cluster[cluster].copy()
            rng.shuffle(indices)
            fold_tie_order = rng.permutation(4)
            cluster_counts = np.zeros(4, dtype=np.int64)
            for index in indices:
                scores = []
                for fold in range(4):
                    trial = loads.copy()
                    trial[fold] += matrix[index]
                    imbalance = float(np.sum(np.square((trial - target) / scale)))
                    spread = 1e-3 * cluster_counts[fold]
                    tie = int(np.flatnonzero(fold_tie_order == fold)[0])
                    scores.append((imbalance + spread, tie, fold))
                selected = min(scores)[2]
                assignment[wells[index]] = selected
                loads[selected] += matrix[index]
                cluster_counts[selected] += 1
        objective = float(np.sum(np.square((loads - target) / scale)))
        missing = 0
        for cluster, indices in by_cluster.items():
            if len(indices) >= 4:
                missing += 4 - len({assignment[wells[index]] for index in indices})
        candidate = (objective + 1000.0 * missing, assignment, loads)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], {
        "objective": best[0],
        "target_balance_vector": target.tolist(),
        "fold_balance_vectors": best[2].tolist(),
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
    audit_wells, audit = choose_audit_wells(wells, clusters, matrix)
    assignments, development = assign_development_wells(
        wells, clusters, matrix, audit_wells
    )
    assignments.update({well_id: 4 for well_id in audit_wells})
    fold_summary = []
    for fold in range(FOLDS):
        selected = np.asarray([assignments[well_id] == fold for well_id in wells])
        fold_summary.append(
            {
                "fold": fold,
                "well_count": int(selected.sum()),
                "cluster_count": int(
                    len({clusters[well_id] for well_id in np.asarray(wells)[selected]})
                ),
                "hidden_rows": int(summary[selected, 0].sum()),
                "visible_rows": int(summary[selected, 1].sum()),
                "md_span": float(summary[selected, 3].sum()),
            }
        )
    cluster_members: dict[str, list[str]] = {}
    for well_id in wells:
        cluster_members.setdefault(str(clusters[well_id]), []).append(well_id)
    payload = {
        "schema_version": 1,
        "protocol_name": "cv-protocol-v5",
        "protocol_revision": 14,
        "boundary_id": "v5_batch6_target_free_representative",
        "status": "frozen_before_any_batch6_target_metric",
        "seed": SEED,
        "restarts": RESTARTS,
        "fold_count": FOLDS,
        "development_folds": [0, 1, 2, 3],
        "prospective_audit_fold": 4,
        "audit_fraction": AUDIT_FRACTION,
        "cluster_rule": "typewell TVT/GR clusters are proportionally represented in F4 when possible and spread over F0-F3",
        "assignment_rule": "deterministic target-free multistart balancing of inference-visible well proxies",
        "legal_horizontal_columns": list(LEGAL_HORIZONTAL_COLUMNS),
        "legal_typewell_columns": ["TVT", "GR"],
        "summary_feature_names": summary_names,
        "balance_feature_names": balance_names,
        "audit_selection": audit,
        "development_balance": development,
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
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"assignments", "clusters", "cluster_members"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
