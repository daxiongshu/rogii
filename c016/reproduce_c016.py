from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCES_PATH = ROOT / "manifests" / "c016_sources.json"
ARTIFACTS_PATH = ROOT / "manifests" / "c016_artifacts.json"
PROVENANCE_PATH = ROOT / "manifests" / "c016_provenance.json"
RECIPE_PATH = ROOT / "weights-dataset" / "c016_recipe.json"
WEIGHTS_PATH = ROOT / "weights-dataset" / "c016_manifest.json"
KERNEL_PATH = ROOT / "kernel" / "infer.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_weights_root() -> Path:
    override = os.environ.get("ROGII_C016_WEIGHTS_ROOT")
    if override:
        return Path(override)
    return (
        ROOT.parent.parent
        / "rogii_v20_limited_query"
        / "deploy"
        / "kaggle"
        / "c016-weights-dataset"
    )


def default_legal_oof() -> Path:
    override = os.environ.get("ROGII_C016_OOF")
    if override:
        return Path(override)
    return (
        ROOT.parent.parent
        / "rogii_v20_limited_query"
        / "limited_query_cv"
        / "runs"
        / "v5_batch2_run_025_goal_050"
        / "kaggle_production"
        / "promotion_oof_legal_posterior_mask.npz"
    )


def verify_sources() -> dict[str, Any]:
    manifest = load_json(SOURCES_PATH)
    failures = []
    total = 0
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        observed = sha256(path)
        total += path.stat().st_size
        if observed != expected:
            failures.append(
                f"changed {relative}: {observed} != {expected}"
            )
    if failures:
        raise RuntimeError(
            "C016 source verification failed:\n" + "\n".join(failures)
        )
    return {
        "files": len(manifest["sha256"]),
        "bytes": total,
        "source_commit": manifest["source_commit"],
    }


def expected_checkpoints(recipe: dict[str, Any]) -> list[str]:
    names = []
    for record in recipe["outer_records"]:
        outer = record["outer_fold"]
        for seed in record["local_seeds"]:
            snapshot = record["local_snapshot"]
            names.append(
                f"c016_outer{outer}_local_seed{seed}"
                f"_epoch{snapshot:03d}.pt"
            )
        variant = record.get("posterior_variant")
        if variant is not None:
            snapshot = record["posterior_snapshot"]
            names.append(
                f"c016_outer{outer}_{variant}_epoch{snapshot:03d}.pt"
            )
    return sorted(names)


def verify_recipe() -> dict[str, Any]:
    recipe = load_json(RECIPE_PATH)
    weights = load_json(WEIGHTS_PATH)
    derived = expected_checkpoints(recipe)
    packaged = sorted(weights["sha256"])
    if derived != packaged:
        missing = sorted(set(derived) - set(packaged))
        extra = sorted(set(packaged) - set(derived))
        raise RuntimeError(
            "recipe does not describe the packaged checkpoints exactly: "
            f"missing={missing} unexpected={extra}"
        )
    if weights["checkpoint_count"] != len(packaged):
        raise RuntimeError("checkpoint_count disagrees with the manifest")
    posterior_folds = [
        record["outer_fold"]
        for record in recipe["outer_records"]
        if record.get("posterior_variant") is not None
    ]
    return {
        "outer_folds": len(recipe["outer_records"]),
        "local_seeds": sorted(
            {
                seed
                for record in recipe["outer_records"]
                for seed in record["local_seeds"]
            }
        ),
        "checkpoints": len(packaged),
        "checkpoint_bytes": weights["checkpoint_bytes"],
        "posterior_outer_folds": posterior_folds,
        "production_rule": recipe["production_rule"],
        "mode_bank_production_weight": recipe["mode_bank_production_weight"],
        "v20_parent_manifest_sha256": weights["v20_parent_manifest_sha256"],
    }


def verify_kernel() -> dict[str, Any]:
    provenance = load_json(PROVENANCE_PATH)
    expected = provenance["kaggle"]["inference_source_sha256"]
    observed = sha256(KERNEL_PATH)
    if observed != expected:
        raise RuntimeError(
            f"frozen C016 kernel changed: {observed} != {expected}"
        )
    return {"inference": observed}


def verify_weights(root: Path) -> dict[str, Any]:
    weights = load_json(WEIGHTS_PATH)
    failures = []
    total = 0
    for name, expected in weights["sha256"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"missing {name}")
            continue
        total += path.stat().st_size
        observed = sha256(path)
        if observed != expected:
            failures.append(f"changed {name}: {observed} != {expected}")
    if failures:
        raise RuntimeError(
            "C016 checkpoint verification failed:\n" + "\n".join(failures)
        )
    if total != weights["checkpoint_bytes"]:
        raise RuntimeError("packaged checkpoint bytes changed")
    return {
        "root": str(root),
        "checkpoints": len(weights["sha256"]),
        "bytes": total,
    }


def score_legal_oof(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        truth = cache["truth"].astype(np.float64)
        prediction = cache["prediction"].astype(np.float64)
        baseline = cache["baseline"].astype(np.float64)
        folds = cache["folds"].astype(np.int64)
        row_starts = cache["row_starts"].astype(np.int64)
        well_ids = cache["well_ids"]
        legal_mask = bool(cache["legal_geometric_posterior_mask"])

    def rmse(values: np.ndarray, rows: np.ndarray | None = None) -> float:
        if rows is None:
            return float(np.sqrt(np.mean((values - truth) ** 2)))
        return float(
            np.sqrt(np.mean((values[rows] - truth[rows]) ** 2))
        )

    fold_rmse = []
    baseline_fold_rmse = []
    for fold in range(int(folds.max()) + 1):
        wells = np.flatnonzero(folds == fold)
        rows = np.concatenate(
            [np.arange(row_starts[i], row_starts[i + 1]) for i in wells]
        )
        fold_rmse.append(rmse(prediction, rows))
        baseline_fold_rmse.append(rmse(baseline, rows))
    pooled = rmse(prediction)
    baseline_pooled = rmse(baseline)
    return {
        "path": str(path),
        "wells": int(len(well_ids)),
        "rows": int(len(truth)),
        "legal_geometric_posterior_mask": legal_mask,
        "v20_baseline_pooled_oof_rmse": baseline_pooled,
        "pooled_rmse": pooled,
        "gain_vs_v20": baseline_pooled - pooled,
        "fold_rmse": fold_rmse,
        "v20_baseline_fold_rmse": baseline_fold_rmse,
        "all_five_original_folds_improve": bool(
            all(b > c for b, c in zip(baseline_fold_rmse, fold_rmse))
        ),
    }


def check_recorded_metrics(score: dict[str, Any]) -> None:
    provenance = load_json(PROVENANCE_PATH)
    metrics = provenance["metrics"]
    if score["pooled_rmse"] != metrics["legal_production_oof_rmse"]:
        raise RuntimeError(
            "C016 OOF score changed: "
            f"{score['pooled_rmse']} != "
            f"{metrics['legal_production_oof_rmse']}"
        )
    if (
        score["v20_baseline_pooled_oof_rmse"]
        != metrics["v20_baseline_pooled_oof_rmse"]
    ):
        raise RuntimeError("v20 baseline inside the C016 archive changed")
    if score["fold_rmse"] != metrics["fold_rmse"]:
        raise RuntimeError("C016 fold scores changed")
    if not score["all_five_original_folds_improve"]:
        raise RuntimeError("C016 no longer improves every original fold")


def command_verify(args: argparse.Namespace) -> None:
    provenance = load_json(PROVENANCE_PATH)
    report: dict[str, Any] = {
        "solution": provenance["solution"],
        "parent_solution": provenance["parent_solution"],
        "source_commit": provenance["source_commit"],
        "sources": verify_sources(),
        "recipe": verify_recipe(),
        "kernel": verify_kernel(),
    }
    if args.weights_root.is_dir():
        report["weights"] = verify_weights(args.weights_root)
    else:
        report["weights"] = {
            "root": str(args.weights_root),
            "status": "unavailable_skipped",
        }
    if args.oof.is_file():
        score = score_legal_oof(args.oof)
        check_recorded_metrics(score)
        report["oof"] = score
    else:
        report["oof"] = {
            "path": str(args.oof),
            "status": "unavailable_skipped",
        }
    report["status"] = "verified"
    print(json.dumps(report, indent=2, sort_keys=True))


def command_score_oof(args: argparse.Namespace) -> None:
    score = score_legal_oof(args.oof)
    check_recorded_metrics(score)
    print(json.dumps(score, indent=2, sort_keys=True))


def command_list_recipe(_: argparse.Namespace) -> None:
    recipe = load_json(RECIPE_PATH)
    weights = load_json(WEIGHTS_PATH)
    print(
        f"candidate={recipe['candidate']} rule={recipe['production_rule']} "
        f"checkpoints={weights['checkpoint_count']}"
    )
    for record in recipe["outer_records"]:
        outer = record["outer_fold"]
        print(
            f"outer{outer} seeds={record['local_seeds']} "
            f"snapshot={record['local_snapshot']} "
            f"temperature={record['local_temperature']} "
            f"weight={record['local_weight']} "
            f"amplitude={record['local_amplitude']} "
            f"posterior={record.get('posterior_variant', 'none')}"
        )
    for name in expected_checkpoints(recipe):
        print(f"  {name}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Verify and score the promoted C016 solution, the "
            "user-approved successor to v20."
        )
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify",
        help="Verify C016 sources, recipe, kernel, weights, and OOF score.",
    )
    verify.add_argument(
        "--weights-root", type=Path, default=default_weights_root()
    )
    verify.add_argument("--oof", type=Path, default=default_legal_oof())
    verify.set_defaults(handler=command_verify)

    score = subparsers.add_parser(
        "score-oof",
        help="Recompute the exact legal production C016 OOF score.",
    )
    score.add_argument("--oof", type=Path, default=default_legal_oof())
    score.set_defaults(handler=command_score_oof)

    recipe = subparsers.add_parser(
        "list-recipe", help="Print the C016 production recipe."
    )
    recipe.set_defaults(handler=command_list_recipe)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
