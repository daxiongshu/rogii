from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCES_PATH = ROOT / "manifests" / "run4_sources.json"
ARTIFACTS_PATH = ROOT / "manifests" / "run4_artifacts.json"
PROVENANCE_PATH = ROOT / "manifests" / "run4_provenance.json"
RECIPE_PATH = ROOT / "weights-dataset" / "recipe.json"
KERNEL_PATH = ROOT / "kernel" / "infer.py"
PACKAGER_PATH = (
    ROOT
    / "pipeline"
    / "limited_query_cv"
    / "prepare_v5_batch3_run4_sr_deployment.py"
)

# The frozen kernel refuses to run unless the recipe still has this shape.
RECIPE_SHAPE = {
    "status": "deployment_candidate_complete",
    "production_correction_scale": 0.6,
    "outer_records": 5,
    "candidate_names": 18,
    "feature_names": 409,
    "well_ids": 773,
}
# float32 storage in the OOF archive costs about 2e-7 of pooled RMSE against
# the float64 value recorded at selection time.
RECORDED_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_limited_query_root() -> Path:
    override = os.environ.get("ROGII_LIMITED_QUERY_ROOT")
    if override:
        return Path(override)
    return REPO.parent / "rogii_v20_limited_query"


def default_router_root() -> Path:
    override = os.environ.get("ROGII_RUN4_ROUTER_ROOT")
    if override:
        return Path(override)
    return (
        default_limited_query_root()
        / "deploy"
        / "kaggle"
        / "run4-sr-weights-dataset"
    )


def default_oof(name: str) -> Path:
    override = os.environ.get(f"ROGII_RUN4_{name.upper()}")
    if override:
        return Path(override)
    artifacts = load_json(ARTIFACTS_PATH)
    if name == "recorded":
        return Path(artifacts["stage_outputs"]["recorded_oof_prediction"]["workspace_path"])
    if name == "parent_recorded":
        return Path(artifacts["lineage"]["parent_prediction"]["workspace_path"])
    return Path(artifacts["lineage"]["parent_legal_prediction"]["workspace_path"])


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
            failures.append(f"changed {relative}: {observed} != {expected}")
    if failures:
        raise RuntimeError(
            "Run4 source verification failed:\n" + "\n".join(failures)
        )
    return {
        "files": len(manifest["sha256"]),
        "bytes": total,
        "build_commit": manifest["build_commit"],
        "promotion_commit": manifest["promotion_commit"],
        "post_submission_records": manifest["post_submission_records"],
    }


def verify_recipe() -> dict[str, Any]:
    recipe = load_json(RECIPE_PATH)
    provenance = load_json(PROVENANCE_PATH)
    observed = {
        "status": recipe["status"],
        "production_correction_scale": recipe["production_correction_scale"],
        "outer_records": len(recipe["outer_records"]),
        "candidate_names": len(recipe["candidate_names"]),
        "feature_names": len(recipe["feature_names"]),
        "well_ids": len(recipe["well_ids"]),
    }
    if observed != RECIPE_SHAPE:
        raise RuntimeError(
            f"recipe shape changed: {observed} != {RECIPE_SHAPE}"
        )
    if len(recipe["well_folds"]) != len(recipe["well_ids"]):
        raise RuntimeError("well_folds and well_ids disagree")

    # The recipe records the sha256 of the packager that wrote it.
    packager = sha256(PACKAGER_PATH)
    if packager != recipe["source_sha256"]:
        raise RuntimeError(
            "deployment packager changed: "
            f"{packager} != {recipe['source_sha256']}"
        )
    recipe_hash = sha256(RECIPE_PATH)
    if recipe_hash != provenance["kaggle"]["recipe_sha256"]:
        raise RuntimeError(
            f"recipe changed: {recipe_hash} != "
            f"{provenance['kaggle']['recipe_sha256']}"
        )
    if recipe["pooled_rmse"] != provenance["metrics"]["recorded_pooled_oof_rmse"]:
        raise RuntimeError("recipe pooled RMSE disagrees with provenance")
    if recipe["v20_rmse"] != provenance["metrics"]["v20_pooled_oof_rmse"]:
        raise RuntimeError("recipe v20 baseline changed")
    if (
        recipe["clean_C016_rmse"]
        != provenance["metrics"]["clean_c016_pooled_oof_rmse"]
    ):
        raise RuntimeError("recipe C016 parent score changed")
    return {
        "recipe_sha256": recipe_hash,
        "packager_sha256": packager,
        "shape": observed,
        "routers": 2 * len(recipe["outer_records"]),
        "routed_wells": [
            {
                "outer_fold": record["outer_fold"],
                "structural": record["routed_structural_wells"],
                "surface": record["routed_surface_wells"],
            }
            for record in recipe["outer_records"]
        ],
    }


def verify_kernel() -> dict[str, Any]:
    provenance = load_json(PROVENANCE_PATH)
    expected = provenance["kaggle"]["inference_source_sha256"]
    observed = sha256(KERNEL_PATH)
    if observed != expected:
        raise RuntimeError(
            f"frozen Run4 kernel changed: {observed} != {expected}"
        )
    source_root = ROOT / "weights-dataset" / "source"
    modules = sorted(p.name for p in (source_root / "limited_query_cv").glob("*.py"))
    for required in (
        "transform_v5_batch3_highcap_structural.py",
        "v5_batch3_highcap_path.py",
        "v5_batch3_run2_surface.py",
        "__init__.py",
    ):
        if required not in modules:
            raise RuntimeError(
                f"deployment source snapshot is missing {required}"
            )
    return {
        "inference": observed,
        "deployment_source_modules": len(modules)
        + len(list((source_root / "rogii").glob("*.py"))),
    }


def verify_lineage() -> dict[str, Any]:
    artifacts = load_json(ARTIFACTS_PATH)
    parent = artifacts["lineage"]["parent_prediction"]
    bank = (
        ROOT / "pipeline" / "limited_query_cv" / "v5_batch3_run4_clean_bank.py"
    ).read_text(encoding="utf-8")
    if parent["sha256"] not in bank:
        raise RuntimeError(
            "the C016 parent hash pinned in v5_batch3_run4_clean_bank.py "
            "does not match the recorded parent artifact"
        )
    report = {
        "chain": artifacts["lineage"]["chain"],
        "parent_prediction_sha256": parent["sha256"],
        "pinned_in_code": True,
    }
    c016_artifacts = REPO / "c016" / "manifests" / "c016_artifacts.json"
    if c016_artifacts.is_file():
        sealed = load_json(c016_artifacts)["stage_outputs"][
            "sealed_outer_predictions"
        ]["sha256"]
        if sealed != parent["sha256"]:
            raise RuntimeError(
                "Run4's parent artifact is not the C016 subtree's sealed "
                "outer prediction"
            )
        report["matches_c016_subtree"] = True
    return report


def verify_routers(root: Path) -> dict[str, Any]:
    recipe = load_json(RECIPE_PATH)
    failures = []
    total = 0
    checked = 0
    for record in recipe["outer_records"]:
        for kind in ("structural", "surface"):
            name = record[f"{kind}_router"]
            expected = record[f"{kind}_router_sha256"]
            path = root / name
            if not path.is_file():
                failures.append(f"missing {name}")
                continue
            checked += 1
            total += path.stat().st_size
            observed = sha256(path)
            if observed != expected:
                failures.append(f"changed {name}: {observed} != {expected}")
    if failures:
        raise RuntimeError(
            "Run4 router verification failed:\n" + "\n".join(failures)
        )
    return {"root": str(root), "routers": checked, "bytes": total}


def score_oof(
    recorded: Path, parent_recorded: Path, parent_legal: Path
) -> dict[str, Any]:
    with np.load(recorded, allow_pickle=False) as run4:
        well_ids = run4["well_ids"].astype(str)
        folds = run4["folds"].astype(np.int64)
        row_starts = run4["row_starts"].astype(np.int64)
        prediction = run4["prediction"].astype(np.float64)
        stored_c016 = run4["clean_C016"].astype(np.float64)
        stored_v20 = run4["v20"].astype(np.float64)
    with np.load(parent_legal, allow_pickle=False) as legal:
        if not np.array_equal(legal["well_ids"].astype(str), well_ids):
            raise RuntimeError("C016 legal archive layout changed")
        truth = legal["truth"].astype(np.float64)
        c016_legal = legal["prediction"].astype(np.float64)
    with np.load(parent_recorded, allow_pickle=False) as sealed:
        if not np.array_equal(sealed["well_ids"].astype(str), well_ids):
            raise RuntimeError("C016 sealed archive layout changed")
        c016_recorded = sealed["prediction"].astype(np.float64)

    # The frozen-parent alignment record's deploy-parity reconstruction.
    parity = prediction + c016_legal - c016_recorded

    def rmse(values: np.ndarray, rows: np.ndarray | None = None) -> float:
        if rows is None:
            return float(np.sqrt(np.mean((values - truth) ** 2)))
        return float(np.sqrt(np.mean((values[rows] - truth[rows]) ** 2)))

    parity_folds = []
    c016_folds = []
    for fold in range(int(folds.max()) + 1):
        wells = np.flatnonzero(folds == fold)
        rows = np.concatenate(
            [np.arange(row_starts[i], row_starts[i + 1]) for i in wells]
        )
        parity_folds.append(rmse(parity, rows))
        c016_folds.append(rmse(c016_legal, rows))
    return {
        "recorded_path": str(recorded),
        "wells": int(len(well_ids)),
        "rows": int(len(truth)),
        "formula": "run4_recorded + c016_legal - c016_recorded",
        "recorded_pooled_rmse_float32_archive": rmse(prediction),
        "deploy_parity_pooled_rmse": rmse(parity),
        "legal_c016_pooled_rmse": rmse(c016_legal),
        "stored_clean_c016_pooled_rmse": rmse(stored_c016),
        "stored_v20_pooled_rmse": rmse(stored_v20),
        "deploy_parity_gain_vs_c016": rmse(c016_legal) - rmse(parity),
        "deploy_parity_fold_rmse": parity_folds,
        "c016_deploy_parity_fold_rmse": c016_folds,
        "all_five_folds_improve": bool(
            all(c > r for c, r in zip(c016_folds, parity_folds))
        ),
    }


def check_recorded_metrics(score: dict[str, Any]) -> None:
    metrics = load_json(PROVENANCE_PATH)["metrics"]
    exact = [
        ("deploy_parity_pooled_rmse", "deploy_parity_pooled_oof_rmse"),
        ("legal_c016_pooled_rmse", "legal_c016_pooled_oof_rmse"),
        ("stored_clean_c016_pooled_rmse", "clean_c016_pooled_oof_rmse"),
        ("stored_v20_pooled_rmse", "v20_pooled_oof_rmse"),
        ("deploy_parity_gain_vs_c016", "deploy_parity_gain_vs_c016"),
    ]
    for observed_key, expected_key in exact:
        if score[observed_key] != metrics[expected_key]:
            raise RuntimeError(
                f"{observed_key} changed: {score[observed_key]} != "
                f"{metrics[expected_key]}"
            )
    for key in ("deploy_parity_fold_rmse", "c016_deploy_parity_fold_rmse"):
        if score[key] != metrics[key]:
            raise RuntimeError(f"{key} changed")
    drift = abs(
        score["recorded_pooled_rmse_float32_archive"]
        - metrics["recorded_pooled_oof_rmse"]
    )
    if drift > RECORDED_TOLERANCE:
        raise RuntimeError(
            "recorded OOF score changed beyond float32 storage tolerance: "
            f"drift={drift}"
        )
    if not score["all_five_folds_improve"]:
        raise RuntimeError("Run4 no longer improves every fold over C016")


def command_verify(args: argparse.Namespace) -> None:
    provenance = load_json(PROVENANCE_PATH)
    report: dict[str, Any] = {
        "solution": provenance["solution"],
        "parent_solution": provenance["parent_solution"],
        "root_solution": provenance["root_solution"],
        "build_commit": provenance["build_commit"],
        "sources": verify_sources(),
        "recipe": verify_recipe(),
        "kernel": verify_kernel(),
        "lineage": verify_lineage(),
    }
    if args.router_root.is_dir():
        report["routers"] = verify_routers(args.router_root)
    else:
        report["routers"] = {
            "root": str(args.router_root),
            "status": "unavailable_skipped",
        }
    archives = [args.recorded, args.parent_recorded, args.parent_legal]
    if all(path.is_file() for path in archives):
        score = score_oof(*archives)
        check_recorded_metrics(score)
        report["oof"] = score
    else:
        report["oof"] = {
            "status": "unavailable_skipped",
            "missing": [str(p) for p in archives if not p.is_file()],
        }
    report["status"] = "verified"
    print(json.dumps(report, indent=2, sort_keys=True))


def command_score_oof(args: argparse.Namespace) -> None:
    score = score_oof(args.recorded, args.parent_recorded, args.parent_legal)
    check_recorded_metrics(score)
    print(json.dumps(score, indent=2, sort_keys=True))


def command_list_recipe(_: argparse.Namespace) -> None:
    recipe = load_json(RECIPE_PATH)
    print(
        f"family={recipe['family']} scale={recipe['production_correction_scale']} "
        f"candidates={len(recipe['candidate_names'])} "
        f"features={len(recipe['feature_names'])}"
    )
    for record in recipe["outer_records"]:
        active = [
            f"{name}={weight:.4f}"
            for name, weight in zip(
                recipe["candidate_names"], record["base_weights"]
            )
            if weight > 1e-6
        ]
        print(
            f"outer{record['outer_fold']} "
            f"structural_wells={record['routed_structural_wells']} "
            f"surface_wells={record['routed_surface_wells']}"
        )
        print(f"  routers: {record['structural_router']}, {record['surface_router']}")
        print(f"  base blend: {', '.join(active)}")
    for fold in recipe["folds"]:
        print(
            f"fold{fold['fold']}: rmse={fold['candidate_rmse']:.6f} "
            f"gain_vs_C016={fold['gain_vs_clean_C016']:.6f} "
            f"gain_vs_v20={fold['gain_vs_v20']:.6f}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Verify and score Run4, the structural-surface router promoted "
            "on top of C016."
        )
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify",
        help="Verify Run4 sources, recipe, kernel, lineage, routers, and OOF.",
    )
    verify.add_argument(
        "--router-root", type=Path, default=default_router_root()
    )
    verify.add_argument(
        "--recorded", type=Path, default=default_oof("recorded")
    )
    verify.add_argument(
        "--parent-recorded", type=Path, default=default_oof("parent_recorded")
    )
    verify.add_argument(
        "--parent-legal", type=Path, default=default_oof("parent_legal")
    )
    verify.set_defaults(handler=command_verify)

    score = subparsers.add_parser(
        "score-oof",
        help="Recompute the deploy-parity Run4 OOF score against C016.",
    )
    score.add_argument(
        "--recorded", type=Path, default=default_oof("recorded")
    )
    score.add_argument(
        "--parent-recorded", type=Path, default=default_oof("parent_recorded")
    )
    score.add_argument(
        "--parent-legal", type=Path, default=default_oof("parent_legal")
    )
    score.set_defaults(handler=command_score_oof)

    recipe = subparsers.add_parser(
        "list-recipe", help="Print the Run4 deployment recipe."
    )
    recipe.set_defaults(handler=command_list_recipe)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
