from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RECIPES_PATH = ROOT / "manifests" / "training_recipes.json"
SOURCES_PATH = ROOT / "manifests" / "training_sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_historical_sources() -> None:
    manifest = load_json(SOURCES_PATH)
    failures = []
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(f"changed {relative}: {observed} != {expected}")
    if failures:
        raise RuntimeError("historical source verification failed:\n" + "\n".join(failures))


def family_map(recipes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families = {family["id"]: family for family in recipes["families"]}
    if len(families) != len(recipes["families"]):
        raise RuntimeError("duplicate family id in training recipes")
    return families


def render(value: Any, fold: int, checkpoint_root: Path) -> str:
    return str(value).format(
        fold=fold,
        checkpoint_root=checkpoint_root.resolve(),
        repo_root=ROOT,
    )


def option_arguments(
    options: dict[str, Any], fold: int, checkpoint_root: Path
) -> list[str]:
    result: list[str] = []
    for name, value in options.items():
        if value is None or value is False:
            continue
        result.append(f"--{name}")
        if value is not True:
            result.append(render(value, fold, checkpoint_root))
    return result


def stage_command(
    recipes: dict[str, Any],
    family: dict[str, Any],
    stage: dict[str, Any],
    fold: int,
    data_root: Path,
    checkpoint_root: Path,
) -> tuple[list[str], list[Path], Path | None]:
    options = dict(recipes["standard_options"])
    options.update(family.get("overrides", {}))
    options.update(stage.get("overrides", {}))

    checkpoint = checkpoint_root / stage["checkpoint_pattern"].format(fold=fold)
    outputs = [checkpoint]
    command = [
        sys.executable,
        str(ROOT / recipes["program"]),
        "--data-root",
        str(data_root.resolve()),
        "--fold",
        str(fold),
        *option_arguments(options, fold, checkpoint_root),
        "--checkpoint",
        str(checkpoint.resolve()),
    ]

    last_pattern = stage.get("last_checkpoint_pattern")
    if last_pattern:
        last_checkpoint = checkpoint_root / last_pattern.format(fold=fold)
        outputs.append(last_checkpoint)
        command.extend(["--last-checkpoint", str(last_checkpoint.resolve())])

    resume = None
    resume_pattern = stage.get("resume_pattern")
    if resume_pattern:
        resume = checkpoint_root / resume_pattern.format(fold=fold)
        command.extend(["--resume", str(resume.resolve())])
    return command, outputs, resume


def selected_families(
    recipes: dict[str, Any], requested: str
) -> list[dict[str, Any]]:
    families = family_map(recipes)
    if requested == "all":
        return list(recipes["families"])
    if requested not in families:
        raise ValueError(
            f"unknown family {requested!r}; choose one of "
            + ", ".join([*families, "all"])
        )
    return [families[requested]]


def selected_folds(requested: str, fold_count: int) -> list[int]:
    if requested == "all":
        return list(range(fold_count))
    fold = int(requested)
    if fold not in range(fold_count):
        raise ValueError(f"fold must be 0..{fold_count - 1} or all")
    return [fold]


def command_list(_: argparse.Namespace) -> None:
    recipes = load_json(RECIPES_PATH)
    print(
        f"families={len(recipes['families'])} folds={recipes['folds']} "
        f"final_neural_checkpoints={len(recipes['families']) * recipes['folds']}"
    )
    for family in recipes["families"]:
        status = family.get("provenance_status", "historically_documented")
        role = "zero-weight-runtime" if family.get("prediction_weight") == 0 else "ensemble"
        print(
            f"{family['id']:20s} {family['checkpoint_pattern']:55s} "
            f"stages={len(family['stages'])} role={role} provenance={status}"
        )


def command_commands(args: argparse.Namespace) -> None:
    recipes = load_json(RECIPES_PATH)
    for family in selected_families(recipes, args.family):
        for fold in selected_folds(args.fold, recipes["folds"]):
            print(f"# family={family['id']} fold={fold}")
            for stage in family["stages"]:
                command, _, _ = stage_command(
                    recipes,
                    family,
                    stage,
                    fold,
                    args.data_root,
                    args.checkpoint_root,
                )
                print(shlex.join(command))


def command_run(args: argparse.Namespace) -> None:
    recipes = load_json(RECIPES_PATH)
    verify_historical_sources()
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (family, fold)
        for family in selected_families(recipes, args.family)
        for fold in selected_folds(args.fold, recipes["folds"])
    ]
    print(
        f"verified_historical_sources jobs={len(jobs)} "
        f"execute={args.execute} checkpoint_root={args.checkpoint_root.resolve()}"
    )
    for family, fold in jobs:
        for stage in family["stages"]:
            command, outputs, resume = stage_command(
                recipes,
                family,
                stage,
                fold,
                args.data_root,
                args.checkpoint_root,
            )
            existing = [path for path in outputs if path.exists()]
            if existing and not args.force:
                raise FileExistsError(
                    "refusing to overwrite existing output(s): "
                    + ", ".join(str(path) for path in existing)
                )
            if args.execute and resume is not None and not resume.is_file():
                raise FileNotFoundError(f"missing resume checkpoint: {resume}")
            print(
                f"family={family['id']} fold={fold} stage={stage['name']}\n"
                f"{shlex.join(command)}",
                flush=True,
            )
            if args.execute:
                subprocess.run(command, cwd=ROOT, check=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Render or execute the hash-locked historical v20 training recipes."
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="List all v20 neural families.")
    listing.set_defaults(handler=command_list)

    def add_recipe_selection(target: argparse.ArgumentParser) -> None:
        target.add_argument("--family", required=True, help="Family id or all.")
        target.add_argument("--fold", required=True, help="Fold 0..4 or all.")
        target.add_argument(
            "--data-root",
            type=Path,
            default=Path(
                "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
            ),
        )
        target.add_argument(
            "--checkpoint-root", type=Path, default=ROOT / "checkpoints"
        )

    commands = subparsers.add_parser(
        "commands", help="Print exact commands without creating directories."
    )
    add_recipe_selection(commands)
    commands.set_defaults(handler=command_commands)

    run = subparsers.add_parser(
        "run", help="Validate recipes and optionally execute them."
    )
    add_recipe_selection(run)
    run.add_argument(
        "--execute",
        action="store_true",
        help="Actually train; without this flag the command is a validated dry run.",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Allow the historical trainer to overwrite named output checkpoints.",
    )
    run.set_defaults(handler=command_run)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
