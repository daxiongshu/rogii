from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PROVENANCE_PATH = ROOT / "manifests" / "provenance.json"
MODEL_SUFFIXES = {".pt", ".pth", ".cbm", ".npz", ".ubj"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_weights_root() -> Path:
    override = os.environ.get("ROGII_V20_WEIGHTS_ROOT")
    if override:
        return Path(override)
    return ROOT.parent / "rogii_v20_checkpoint_vault"


def default_oof_root() -> Path:
    override = os.environ.get("ROGII_V20_OOF_ROOT")
    if override:
        return Path(override)
    return ROOT.parent / "rogii_v20_oof_vault"


def default_zero_weight_root() -> Path:
    override = os.environ.get("ROGII_V20_ZERO_WEIGHT_ROOT")
    if override:
        return Path(override)
    return ROOT.parent / "rogii_simple" / "checkpoints"


def default_data_root() -> Path:
    override = os.environ.get("ROGII_DATA_ROOT")
    if override:
        return Path(override)
    return (
        Path("/kaggle/input/competitions")
        / "rogii-wellbore-geology-prediction"
    )


def load_provenance() -> dict[str, Any]:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def verify_source(provenance: dict[str, Any]) -> dict[str, str]:
    checked = {}
    for record_name in ("inference", "kaggle_kernel_metadata"):
        record = provenance[record_name]
        path = ROOT / record["path"]
        observed = sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(
                f"{record_name} changed: {observed} != {record['sha256']}"
            )
        checked[record_name] = observed

    weights = provenance["weights"]
    manifest_path = ROOT / weights["manifest"]
    observed_manifest = sha256(manifest_path)
    if observed_manifest != weights["manifest_sha256"]:
        raise RuntimeError(
            "bundled weight manifest changed: "
            f"{observed_manifest} != {weights['manifest_sha256']}"
        )
    checked["weight_manifest"] = observed_manifest
    return checked


def functional_weight_hashes(
    provenance: dict[str, Any],
) -> dict[str, str]:
    weights = provenance["weights"]
    manifest = json.loads(
        (ROOT / weights["manifest"]).read_text(encoding="utf-8")
    )
    excluded = set(weights["runtime_only_zero_weight_checkpoints"])
    expected = {
        name: digest
        for name, digest in manifest["sha256"].items()
        if name not in excluded
    }
    if len(expected) != weights["functional_artifacts"]:
        raise RuntimeError(
            f"expected {weights['functional_artifacts']} functional "
            f"artifacts, found {len(expected)}"
        )
    return expected


def verify_weights(
    weights_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    expected = functional_weight_hashes(provenance)
    excluded = set(
        provenance["weights"]["runtime_only_zero_weight_checkpoints"]
    )
    missing = sorted(
        name for name in expected if not (weights_root / name).is_file()
    )
    forbidden = sorted(
        name for name in excluded if (weights_root / name).exists()
    )
    unexpected = sorted(
        path.name
        for path in weights_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in MODEL_SUFFIXES
        and path.name not in expected
    )
    if missing or forbidden or unexpected:
        raise RuntimeError(
            "weight-vault contents changed: "
            f"missing={missing}, forbidden={forbidden}, "
            f"unexpected={unexpected}"
        )

    mismatched = []
    total_bytes = 0
    for name, expected_digest in sorted(expected.items()):
        path = weights_root / name
        total_bytes += path.stat().st_size
        observed = sha256(path)
        if observed != expected_digest:
            mismatched.append((name, observed, expected_digest))
    if mismatched:
        raise RuntimeError(f"weight checksum mismatch: {mismatched}")

    counts = {
        "neural_checkpoints": sum(
            name.endswith((".pt", ".pth")) for name in expected
        ),
        "catboost_models": sum(name.endswith(".cbm") for name in expected),
        "reference_stores": sum(name.endswith(".npz") for name in expected),
    }
    for key, observed in counts.items():
        expected_count = provenance["weights"][key]
        if observed != expected_count:
            raise RuntimeError(
                f"{key} changed: {observed} != {expected_count}"
            )
    return {
        "root": str(weights_root.resolve()),
        "artifacts": len(expected),
        "bytes": total_bytes,
        **counts,
    }


def verify_zero_weight_runtime(
    zero_weight_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    manifest = json.loads(
        (
            ROOT / provenance["weights"]["manifest"]
        ).read_text(encoding="utf-8")
    )
    names = provenance["weights"]["runtime_only_zero_weight_checkpoints"]
    mismatched = []
    total_bytes = 0
    for name in names:
        path = zero_weight_root / name
        if not path.is_file():
            raise RuntimeError(f"missing runtime-only checkpoint: {path}")
        total_bytes += path.stat().st_size
        observed = sha256(path)
        expected = manifest["sha256"][name]
        if observed != expected:
            mismatched.append((name, observed, expected))
    if mismatched:
        raise RuntimeError(
            f"runtime-only checkpoint checksum mismatch: {mismatched}"
        )
    return {
        "root": str(zero_weight_root.resolve()),
        "artifacts": len(names),
        "bytes": total_bytes,
        "prediction_weight": 0.0,
        "purpose": "frozen-kernel runtime compatibility only",
    }


def stage_runtime_weights(
    target: Path,
    weights_root: Path,
    zero_weight_root: Path,
    provenance: dict[str, Any],
) -> None:
    functional = functional_weight_hashes(provenance)
    runtime_only = set(
        provenance["weights"]["runtime_only_zero_weight_checkpoints"]
    )
    for name in functional:
        (target / name).symlink_to((weights_root / name).resolve())
    for name in runtime_only:
        (target / name).symlink_to((zero_weight_root / name).resolve())
    observed = len(list(target.iterdir()))
    expected = provenance["weights"]["runtime_artifacts"]
    if observed != expected:
        raise RuntimeError(
            f"staged runtime artifact count changed: {observed} != {expected}"
        )


def row_mask(
    well_mask: np.ndarray, row_starts: np.ndarray
) -> np.ndarray:
    difference = np.zeros(int(row_starts[-1]) + 1, dtype=np.int32)
    selected = np.flatnonzero(well_mask)
    np.add.at(difference, row_starts[selected], 1)
    np.add.at(difference, row_starts[selected + 1], -1)
    return np.cumsum(difference[:-1]) > 0


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def verify_oof(
    oof_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    oof = provenance["oof"]
    paths = {
        key: oof_root / oof[key]["filename"]
        for key in ("layout", "prediction")
    }
    for key, path in paths.items():
        observed = sha256(path)
        expected = oof[key]["sha256"]
        if observed != expected:
            raise RuntimeError(
                f"{key} OOF archive changed: {observed} != {expected}"
            )

    with np.load(paths["layout"], allow_pickle=False) as layout:
        well_ids = layout["well_ids"].astype(str)
        row_starts = layout["row_starts"].astype(np.int64)
        truth = layout["truth"].astype(np.float32)
        folds = layout["original"].astype(np.int8)
    with np.load(paths["prediction"], allow_pickle=False) as prediction:
        if (
            not np.array_equal(well_ids, prediction["well_ids"].astype(str))
            or not np.array_equal(
                row_starts, prediction["row_starts"].astype(np.int64)
            )
        ):
            raise RuntimeError("OOF prediction/layout identity mismatch")
        values = prediction["prediction"].astype(np.float32)

    if (
        len(well_ids) != oof["wells"]
        or len(truth) != oof["rows"]
        or len(values) != oof["rows"]
        or row_starts[0] != 0
        or row_starts[-1] != oof["rows"]
        or np.any(np.diff(row_starts) <= 0)
        or not np.isfinite(truth).all()
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("OOF array contract changed")

    pooled = rmse(values, truth)
    fold_scores = [
        rmse(values[mask], truth[mask])
        for fold in range(5)
        for mask in [row_mask(folds == fold, row_starts)]
    ]
    if abs(pooled - oof["pooled_rmse"]) > 1e-12:
        raise RuntimeError(
            f"OOF score changed: {pooled} != {oof['pooled_rmse']}"
        )
    if not np.allclose(
        fold_scores,
        oof["original_fold_rmse"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("OOF fold scores changed")
    return {
        "root": str(oof_root.resolve()),
        "wells": len(well_ids),
        "rows": len(truth),
        "pooled_rmse": pooled,
        "original_fold_rmse": fold_scores,
    }


def command_verify(args: argparse.Namespace) -> None:
    provenance = load_provenance()
    result = {
        "status": "verified",
        "historical_source_commit": provenance[
            "historical_source_commit"
        ],
        "source": verify_source(provenance),
        "weights": verify_weights(args.weights_root, provenance),
        "runtime_only_zero_weight": verify_zero_weight_runtime(
            args.zero_weight_root, provenance
        ),
        "oof": verify_oof(args.oof_root, provenance),
        "kaggle_submission": provenance["kaggle_submission"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def command_score_oof(args: argparse.Namespace) -> None:
    provenance = load_provenance()
    verify_source(provenance)
    print(
        json.dumps(
            verify_oof(args.oof_root, provenance),
            indent=2,
            sort_keys=True,
        )
    )


def command_infer(args: argparse.Namespace) -> None:
    provenance = load_provenance()
    verify_source(provenance)
    if not args.skip_weight_verification:
        verify_weights(args.weights_root, provenance)
        verify_zero_weight_runtime(args.zero_weight_root, provenance)
    with tempfile.TemporaryDirectory(prefix="rogii-v20-runtime-") as temporary:
        runtime_root = Path(temporary)
        stage_runtime_weights(
            runtime_root,
            args.weights_root,
            args.zero_weight_root,
            provenance,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "ROGII_DATA_ROOT": str(args.data_root.resolve()),
                "ROGII_WEIGHTS_ROOT": str(runtime_root.resolve()),
                "ROGII_OUTPUT": str(args.output.resolve()),
            }
        )
        subprocess.run(
            [sys.executable, str(ROOT / provenance["inference"]["path"])],
            check=True,
            env=environment,
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Reproduce the immutable ROGII v20 baseline."
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify", help="Verify source, weights, and exact OOF score."
    )
    verify.add_argument(
        "--weights-root", type=Path, default=default_weights_root()
    )
    verify.add_argument("--oof-root", type=Path, default=default_oof_root())
    verify.add_argument(
        "--zero-weight-root",
        type=Path,
        default=default_zero_weight_root(),
    )
    verify.set_defaults(handler=command_verify)

    score = subparsers.add_parser(
        "score-oof", help="Recompute the exact frozen v20 OOF score."
    )
    score.add_argument("--oof-root", type=Path, default=default_oof_root())
    score.set_defaults(handler=command_score_oof)

    infer = subparsers.add_parser(
        "infer", help="Regenerate submission.csv with the frozen v20 model."
    )
    infer.add_argument("--data-root", type=Path, default=default_data_root())
    infer.add_argument(
        "--weights-root", type=Path, default=default_weights_root()
    )
    infer.add_argument(
        "--zero-weight-root",
        type=Path,
        default=default_zero_weight_root(),
    )
    infer.add_argument(
        "--output", type=Path, default=ROOT / "submission.csv"
    )
    infer.add_argument(
        "--skip-weight-verification",
        action="store_true",
        help="Skip the 87-artifact hash pass when it was already run.",
    )
    infer.set_defaults(handler=command_infer)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
