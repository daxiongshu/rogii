from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "fold_purged_protected_posterior_fusion"
ROOT = Path("limited_query_cv/runs/v5_batch2_run_006_goal_050")
TRAIN_SOURCE = Path("limited_query_cv/train_v5_run6_protected_posterior_fusion.py")
TRAIN_SOURCE_SHA256 = (
    "62bb48fd63adb994df23a997f35eb5bfc0882cccd3ea8a26127c11aecf9454e8"
)
SELECTOR_GATE = ROOT / "protected_posterior_clean_selector_gate_c618.json"
SELECTOR_GATE_SHA256 = (
    "b750245752e675c4db513308d5f8bafbbc76e62cc1dd11cfa54c711e28ec7cb2"
)
VARIANTS = (
    "posterior_only_b12",
    "posterior_physical_b12",
    "posterior_physical_b16",
)
SNAPSHOTS = (4, 8, 12, 16, 20, 24)
TEMPERATURE_KEYS = (
    "correction_t075",
    "correction_t100",
    "correction_t150",
    "correction_t200",
)
POSTERIOR_CACHE_SHA256 = (
    "eb12c7ac801eda8837c611ec7832f8df6c4a175893bd9f9d84d512982b33c0aa",
    "43dbec13717c746d29939d1eb18922ff2ec36d56aedfd3d7da8b006ac8df95a6",
    "f7c9267a79bc7ea808db7bd28a511f0767d66813700f874529eff5b187a62eb9",
    "9376052566eb521cba17ca77a2d2c5bc594bd71c3088ee2928fbca3338d919bb",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_prediction(
    path: Path,
    *,
    fold: int,
    variant: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    prediction_sha256 = sha256(path)
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "fold",
            "variant",
            "well_ids",
            "row_starts",
            "baseline",
            "d570",
            "truth",
            "checkpoint_sha256",
            "source_sha256",
            "audit_fold_loaded",
            "confirmation_regroupings_loaded",
            *TEMPERATURE_KEYS,
        }
        missing = required - set(cache.files)
        if missing:
            raise RuntimeError(f"{path}: missing fields {sorted(missing)}")
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        d570 = cache["d570"].astype(np.float32)
        truth = cache["truth"].astype(np.float32)
        corrections = {
            key: cache[key].astype(np.float32) for key in TEMPERATURE_KEYS
        }
        if (
            int(cache["fold"]) != fold
            or str(cache["variant"]) != variant
            or str(cache["checkpoint_sha256"]) != checkpoint_sha256
            or str(cache["source_sha256"]) != TRAIN_SOURCE_SHA256
            or bool(cache["audit_fold_loaded"])
            or bool(cache["confirmation_regroupings_loaded"])
        ):
            raise RuntimeError(f"{path}: lineage/firewall mismatch")
    if (
        len(starts) != len(ids) + 1
        or starts[0] != 0
        or np.any(np.diff(starts) <= 0)
        or starts[-1] != len(baseline)
        or len(d570) != len(baseline)
        or len(truth) != len(baseline)
        or len(np.unique(ids)) != len(ids)
    ):
        raise RuntimeError(f"{path}: invalid held layout")
    arrays = (baseline, d570, truth, *corrections.values())
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError(f"{path}: nonfinite prediction data")
    for correction in corrections.values():
        if len(correction) != len(baseline):
            raise RuntimeError(f"{path}: correction length mismatch")
        if not np.all(correction[starts[:-1]] == 0.0):
            raise RuntimeError(f"{path}: first-hidden anchor changed")
        if float(np.max(np.abs(correction))) > 64.0 + 1e-6:
            raise RuntimeError(f"{path}: correction bound changed")
    return {
        "path": str(path),
        "sha256": prediction_sha256,
        "well_count": int(len(ids)),
        "row_count": int(len(baseline)),
        "well_ids_sha256": hashlib.sha256(
            "\n".join(ids.tolist()).encode("utf-8")
        ).hexdigest(),
        "row_starts_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
    }


def freeze(root: Path) -> dict[str, object]:
    if sha256(TRAIN_SOURCE) != TRAIN_SOURCE_SHA256:
        raise RuntimeError("protected-posterior trainer changed")
    if sha256(SELECTOR_GATE) != SELECTOR_GATE_SHA256:
        raise RuntimeError("C619 selector gate changed")
    cache_records = []
    for fold, expected_hash in enumerate(POSTERIOR_CACHE_SHA256):
        path = root / f"protected_posterior_volume_fold{fold}.npz"
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"F{fold} posterior cache changed")
        cache_records.append(
            {"fold": fold, "path": str(path), "sha256": actual_hash}
        )

    jobs = []
    for fold in range(1, 4):
        expected_training = [candidate for candidate in range(4) if candidate != fold]
        for variant in VARIANTS:
            directory = root / f"protected_{variant}_f{fold}"
            result_path = directory / "result.json"
            result_hash = sha256(result_path)
            result = json.loads(result_path.read_text())
            if (
                result.get("schema_version") != 1
                or result.get("run_id") != RUN_ID
                or result.get("family") != FAMILY
                or result.get("variant") != variant
                or result.get("status") != "complete_fixed_snapshots"
                or result.get("fold") != fold
                or result.get("training_folds") != expected_training
                or result.get("registered_snapshots") != list(SNAPSHOTS)
                or result.get("source_sha256") != TRAIN_SOURCE_SHA256
                or result.get("audit_fold_loaded")
                or result.get("confirmation_regroupings_loaded")
            ):
                raise RuntimeError(f"{result_path}: result lineage mismatch")
            records = {
                int(record["epoch"]): record for record in result["snapshots"]
            }
            if tuple(sorted(records)) != SNAPSHOTS:
                raise RuntimeError(f"{result_path}: snapshot set changed")
            snapshots = []
            layout_signature = None
            for epoch in SNAPSHOTS:
                record = records[epoch]
                checkpoint = directory / f"epoch{epoch:03d}.pt"
                prediction = directory / f"epoch{epoch:03d}_prediction.npz"
                checkpoint_hash = sha256(checkpoint)
                if (
                    record.get("checkpoint_sha256") != checkpoint_hash
                    or record.get("prediction_sha256") != sha256(prediction)
                ):
                    raise RuntimeError(
                        f"{variant} F{fold} epoch {epoch}: internal hash mismatch"
                    )
                prediction_record = validate_prediction(
                    prediction,
                    fold=fold,
                    variant=variant,
                    checkpoint_sha256=checkpoint_hash,
                )
                signature = (
                    prediction_record["well_count"],
                    prediction_record["row_count"],
                    prediction_record["well_ids_sha256"],
                    prediction_record["row_starts_sha256"],
                )
                if layout_signature is None:
                    layout_signature = signature
                elif signature != layout_signature:
                    raise RuntimeError(
                        f"{variant} F{fold}: snapshot layout changed"
                    )
                snapshots.append(
                    {
                        "epoch": epoch,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "prediction": prediction_record,
                    }
                )
            jobs.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "training_folds": expected_training,
                    "posterior_cache_sha256": POSTERIOR_CACHE_SHA256[fold],
                    "result": str(result_path),
                    "result_sha256": result_hash,
                    "snapshots": snapshots,
                }
            )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "stage": "complete_F1_F3_expansion_manifest",
        "status": "passed",
        "job_count": len(jobs),
        "prediction_count": len(jobs) * len(SNAPSHOTS),
        "checkpoint_count": len(jobs) * len(SNAPSHOTS),
        "train_source": str(TRAIN_SOURCE),
        "train_source_sha256": TRAIN_SOURCE_SHA256,
        "selector_gate": str(SELECTOR_GATE),
        "selector_gate_sha256": SELECTOR_GATE_SHA256,
        "posterior_caches": cache_records,
        "jobs": jobs,
        "target_metric_used_for_manifest": False,
        "target_metric_emitted": False,
        "f4_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.root)
    write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "job_count": payload["job_count"],
                "prediction_count": payload["prediction_count"],
                "checkpoint_count": payload["checkpoint_count"],
                "target_metric_emitted": payload["target_metric_emitted"],
                "f4_loaded": payload["f4_loaded"],
                "source_sha256": payload["source_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
