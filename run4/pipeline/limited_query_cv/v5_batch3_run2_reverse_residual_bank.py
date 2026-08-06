from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


RUN_ROOT = Path("limited_query_cv/runs/v5_batch3_run_002_goal_020")
ARCHIVE_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
MANIFEST = Path(
    "limited_query_cv/runs/v5_batch2_run_025_goal_050/"
    "clean_run17_expert_block_manifest.json"
)
MANIFEST_SHA256 = (
    "cde55877ef2fb4839af342f70739a084148a25a63098f4376520a68d0e348509"
)
QUARANTINE = Path(
    "limited_query_cv/v5_batch2_purece_strat_lineage_invalidation.json"
)
QUARANTINE_SHA256 = (
    "85cccc9982398b69f0210dc60c9177ac1d0fbabb5578ede60a967e0e345be2ec"
)
MODE_BANK_QUARANTINE = Path(
    "limited_query_cv/v5_batch2_mode_bank_strat_lineage_invalidation.json"
)
MODE_BANK_QUARANTINE_SHA256 = (
    "cbc470b5e5621e4f491c493b934c4111d92b0cae8a6821ffeafef923281bc56f"
)
ARCHIVES = (
    (
        "reverse_physics_prior_mixed",
        "physics_prior_mixed",
        ARCHIVE_ROOT / "physics_prior_mixed_clean.npz",
        "fe8133df00d65df6d921e191f722b9ef80db62445c761c09fc8de7c0012342ca",
    ),
    (
        "reverse_local_residual_seed_bag",
        "local_residual_seed_bag",
        ARCHIVE_ROOT / "local_residual_seed_bag_clean.npz",
        "19214f37dd5ca795008e1df9205e487e9617961d56904dff22d3d1b6b7e78974",
    ),
    (
        "reverse_multiscale_residual_bag",
        "multiscale_residual_bag",
        ARCHIVE_ROOT / "multiscale_residual_bag_clean.npz",
        "def04dfb50cec0cbd75c0df8e8f342cc5abd68e8f38c2fee3c108218176f2522",
    ),
    (
        "reverse_protected_mode_maximin",
        "protected_mode_maximin",
        ARCHIVE_ROOT / "protected_mode_bank_maximin.npz",
        "4206bef05db53bfd73bc1c9d5b14612667c3b16c2c1108fa88a38c11aacf416f",
    ),
    (
        "reverse_target_surface",
        "target_surface",
        ARCHIVE_ROOT / "target_surface_clean.npz",
        "11abc03394d8ec28489df5484dcb6b9e3307b2a3d8b5b12a98e1aec3a26ec48b",
    ),
)
REVERSE_TOTAL_CAPS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_clean_lineage() -> list[dict]:
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("clean expert manifest hash changed")
    if sha256(QUARANTINE) != QUARANTINE_SHA256:
        raise ValueError("lineage quarantine hash changed")
    if (
        sha256(MODE_BANK_QUARANTINE)
        != MODE_BANK_QUARANTINE_SHA256
    ):
        raise ValueError("mode-bank lineage quarantine hash changed")
    manifest = json.loads(MANIFEST.read_text())
    quarantine = json.loads(QUARANTINE.read_text())
    mode_quarantine = json.loads(MODE_BANK_QUARANTINE.read_text())
    if (
        manifest["status"] != "frozen_complete_artifact_manifest"
        or manifest["truth_array_loaded"]
        or manifest["audit_fold_loaded"]
        or manifest["f4_loaded"]
        or quarantine["status"] != "active_lineage_quarantine"
        or mode_quarantine["status"]
        != "active_lineage_quarantine_repaired"
    ):
        raise ValueError("clean lineage attestations changed")
    # The Run17 manifest predates the independent mode-bank quarantine and
    # still names protected_mode_bank_maximin.npz. That prediction is
    # transitively derived from the quarantined stratified-parent banks.
    # Fail closed instead of silently treating its pinned hash as clean.
    invalid_mode_prediction = str(
        ARCHIVE_ROOT / "protected_mode_bank_maximin.npz"
    )
    if any(
        name == "reverse_protected_mode_maximin"
        and str(path) == invalid_mode_prediction
        for name, _, path, _ in ARCHIVES
    ):
        raise ValueError(
            "reverse_protected_mode_maximin is transitively quarantined; "
            "R094 candidate banks must be rebuilt from the original-fold-"
            "safe mode lineage before reuse"
        )
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
    records = []
    for name, manifest_key, path, expected_hash in ARCHIVES:
        record = manifest["experts"][manifest_key]
        if (
            record["path"] != str(path)
            or record["field"] != "prediction"
            or record["archive_sha256"] != expected_hash
            or sha256(path) != expected_hash
            or str(path) in forbidden
        ):
            raise ValueError(f"{name}: clean manifest mismatch")
        records.append(
            {
                "name": name,
                "manifest_key": manifest_key,
                "path": str(path),
                "sha256": expected_hash,
                "field": "prediction",
                "prediction_sha256": record["prediction_sha256"],
            }
        )
    return records


def load_prediction_archive(
    path: Path,
    expected_sha256: str | None,
) -> dict[str, np.ndarray | bool]:
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f"{path}: archive hash changed")
    with np.load(path, allow_pickle=False) as data:
        result = {
            "well_ids": data["well_ids"].astype(str),
            "folds": data["folds"].astype(np.int64),
            "row_starts": data["row_starts"].astype(np.int64),
            "prediction": data["prediction"].astype(np.float64),
            "audit_fold_loaded": bool(data["audit_fold_loaded"]),
            "confirmation_regroupings_loaded": bool(
                data["confirmation_regroupings_loaded"]
            ),
        }
    if result["audit_fold_loaded"] or result["confirmation_regroupings_loaded"]:
        raise ValueError(f"{path}: archive firewall failed")
    return result


def extract_fold_prediction(
    archive: dict[str, np.ndarray | bool],
    fold: int,
    expected_well_ids: np.ndarray,
) -> np.ndarray:
    wells = np.asarray(archive["well_ids"]).astype(str)
    folds = np.asarray(archive["folds"]).astype(np.int64)
    starts = np.asarray(archive["row_starts"]).astype(np.int64)
    prediction = np.asarray(archive["prediction"]).astype(np.float64)
    pieces = []
    for well_id in expected_well_ids.astype(str):
        found = np.flatnonzero((wells == well_id) & (folds == fold))
        if len(found) != 1:
            raise ValueError(
                f"fold {fold}: expected one archive row for {well_id}"
            )
        index = int(found[0])
        pieces.append(prediction[starts[index] : starts[index + 1]])
    return np.concatenate(pieces)


def load_reverse_directions(
    fold: int,
    expected_well_ids: np.ndarray,
    clean_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    records = verify_clean_lineage()
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
            raise ValueError(f"{record['name']}: prediction shape changed")
        names.append(record["name"])
        directions.append(clean_prediction - prediction)
    result = np.stack(directions).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("reverse direction bank is nonfinite")
    return np.asarray(names), result, records
