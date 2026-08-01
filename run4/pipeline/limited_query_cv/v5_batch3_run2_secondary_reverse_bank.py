from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from limited_query_cv.v5_batch3_run2_reverse_residual_bank import (
    QUARANTINE,
    QUARANTINE_SHA256,
    extract_fold_prediction,
    load_prediction_archive,
    sha256,
)


ROOT = Path("limited_query_cv/runs/v5_batch2_run_006_goal_050")
SECONDARY_ARCHIVES = (
    (
        "reverse_local_residual_snapshot_bag",
        ROOT / "local_residual_snapshot_bag_clean.npz",
        "6700f299a1a14cbd505b188d3bf181f2d151c21034b5bf1a51011331bc1b1b73",
        ROOT / "local_residual_snapshot_bag_clean.json",
        "83ae3691f5cf815b5574999e2e05fba0d69263578b2d9c9326681010866eddf9",
        Path(
            "limited_query_cv/"
            "evaluate_v5_run6_local_residual_snapshot_bag.py"
        ),
        "8aede7474e49e7fe6e95c4005121cec70e3274427952cdd836b4adfd44f285f5",
    ),
    (
        "reverse_multi_anchor_phase",
        ROOT / "multi_anchor_phase_clean.npz",
        "b9efbd804660963a4cf6d40df6223ba74f80aae9d9504c7eb44fd92f6f8a3426",
        ROOT / "multi_anchor_phase_clean.json",
        "8c4634b994cb15c90d5ad0c2fa7a445919d41e9024fa07270012429da947fa3c",
        Path(
            "limited_query_cv/evaluate_v5_run6_multi_anchor_phase_clean.py"
        ),
        "2c5cb3aded6acca64a112140551f7dd06ddada02f63af16e1ff23e1bf3fd4508",
    ),
    (
        "reverse_hardtail",
        ROOT / "hardtail_clean.npz",
        "a7617b6e788d3643345f2dbc6d9db7d79c67d3c4dc6df32e85a2a156e26e903a",
        ROOT / "hardtail_clean.json",
        "a04a6842945e3d8f92933d09b797ff4d89609a2e5eab75eab584bdc33db2ffc3",
        Path("limited_query_cv/evaluate_v5_run6_hardtail_clean.py"),
        "a90ec62291afd65f75cb5bb0e3a61d739fe5bee7ad58bbe02a28ca2752b1e5d2",
    ),
    (
        "reverse_mtp",
        ROOT / "mtp_clean.npz",
        "c72b52bfb3b7a9364bd57237f4d297563642294088185d4e8d77ac2e6b644fc2",
        ROOT / "mtp_clean.json",
        "049a12ff14a7de4447f6c9eae85e142d272caeea370a9819c7542f2d89ee3d48",
        Path("limited_query_cv/evaluate_v5_run6_mtp_clean.py"),
        "653183b5c50535f3e290cd52af89efb488f63726ffdb4ac6c59830c3b30c6991",
    ),
    (
        "reverse_trajectory_direction_projection",
        ROOT / "trajectory_direction_projection_clean.npz",
        "d42510bc8e63d664fd77c0a89111d0598e33babb8c118847112206b0435ba1bd",
        ROOT / "trajectory_direction_projection_clean.json",
        "f5c1cedf37ab89a903655be0f556bb3eee8d77dca7139ad0af9f3fb67f25dfbb",
        Path(
            "limited_query_cv/"
            "finish_v5_run6_trajectory_direction_projection_clean.py"
        ),
        "42d5eef3d3a5f774cb775122fc6d93d51d719d21b3f3c398cff6830c6880d794",
    ),
)
PARENT_ATTESTATIONS = (
    (
        ROOT / "local_v20_residual_clean.npz",
        "9792284999535a5f7766cc1592f76ddceff34e057a935fd51e457fbb0171858e",
    ),
    (
        ROOT / "d209_amplitude_clean.npz",
        "5e3b6ee41f8705cc6b0044b36255bf8e8b3b6c4752c2afa9f6049d83f77cd310",
    ),
)
TRAJECTORY_SELECTOR_SOURCE = Path(
    "limited_query_cv/finish_v5_run6_trajectory_parent_gated_clean.py"
)
TRAJECTORY_SELECTOR_SOURCE_SHA256 = (
    "3a19d8f04dea673f20fb3733a9c7606a0658f3ed3513b58d1e036b07fdf9001b"
)


def verify_secondary_lineage() -> list[dict]:
    if sha256(QUARANTINE) != QUARANTINE_SHA256:
        raise ValueError("active lineage quarantine changed")
    quarantine = json.loads(QUARANTINE.read_text())
    forbidden = json.dumps(
        {
            "root": quarantine["root_cause"],
            "direct": quarantine["direct_invalid_consumers"],
            "downstream": quarantine[
                "representative_downstream_invalid_results"
            ],
        },
        sort_keys=True,
    )
    for path, expected_hash in PARENT_ATTESTATIONS:
        if sha256(path) != expected_hash or str(path) in forbidden:
            raise ValueError(f"{path}: clean parent attestation failed")
    if (
        sha256(TRAJECTORY_SELECTOR_SOURCE)
        != TRAJECTORY_SELECTOR_SOURCE_SHA256
    ):
        raise ValueError("trajectory other-fold selector source changed")
    records = []
    for (
        name,
        archive_path,
        archive_hash,
        result_path,
        result_hash,
        source_path,
        source_hash,
    ) in SECONDARY_ARCHIVES:
        if (
            sha256(archive_path) != archive_hash
            or sha256(result_path) != result_hash
            or sha256(source_path) != source_hash
            or str(archive_path) in forbidden
            or str(result_path) in forbidden
        ):
            raise ValueError(f"{name}: lineage hash changed")
        result = json.loads(result_path.read_text())
        selection_status = result.get("status", result.get("stage"))
        if (
            selection_status
            not in {
                "complete_selection_clean_F0_F3",
                "rejected_selection_clean_F0_F3",
            }
            or bool(result.get("audit_fold_loaded"))
            or bool(result.get("confirmation_regroupings_loaded"))
            or bool(result.get("f4_loaded"))
        ):
            raise ValueError(f"{name}: selection-clean status changed")
        selector_text = source_path.read_text()
        if name == "reverse_trajectory_direction_projection":
            selector_text += TRAJECTORY_SELECTOR_SOURCE.read_text()
        if not any(
            pattern in selector_text
            for pattern in (
                "other_three",
                "if fold != held",
                "if index != held",
                "if role != held",
                "selection_folds",
            )
        ):
            raise ValueError(f"{name}: other-fold selector not attested")
        records.append(
            {
                "name": name,
                "path": str(archive_path),
                "sha256": archive_hash,
                "result": str(result_path),
                "result_sha256": result_hash,
                "source": str(source_path),
                "source_sha256": source_hash,
                "selection_status": selection_status,
            }
        )
    return records


def load_secondary_reverse_directions(
    fold: int,
    expected_well_ids: np.ndarray,
    clean_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    records = verify_secondary_lineage()
    names = []
    directions = []
    for record in records:
        archive = load_prediction_archive(
            Path(record["path"]), record["sha256"]
        )
        prediction = extract_fold_prediction(
            archive, fold, expected_well_ids
        )
        if prediction.shape != clean_prediction.shape:
            raise ValueError(f"{record['name']}: shape changed")
        names.append(record["name"])
        directions.append(clean_prediction - prediction)
    result = np.stack(directions).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("secondary reverse bank is nonfinite")
    return np.asarray(names), result, records
