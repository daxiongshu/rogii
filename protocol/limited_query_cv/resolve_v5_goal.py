from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from decimal import Decimal
from pathlib import Path


POLICY = Path(__file__).with_name("cv_protocol_v5.json")
LEDGER = Path(__file__).with_name("cv_query_ledger_v5.json")
RUNS_ROOT = Path(__file__).with_name("runs")
RUN_DIRECTORY_PATTERN = re.compile(
    r"^v5_batch(?P<batch>[0-9]+)_run_(?P<run>[0-9]{3,})_goal_"
    r"(?P<goal>[0-9]+)$"
)


def parse_goal(raw: str, source: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{source} must be numeric, got {raw!r}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{source} must be finite and positive, got {raw!r}")
    return value


def policy_sha256() -> str:
    return hashlib.sha256(POLICY.read_bytes()).hexdigest()


def goal_code(value: float) -> str:
    text = format(Decimal(str(value)), "f")
    whole, separator, fraction = text.partition(".")
    if not separator:
        fraction = ""
    fraction = fraction.rstrip("0").ljust(2, "0")
    return f"{whole}{fraction}"


def active_batch(ledger: dict) -> int:
    if ledger["remaining_batches"] <= 0:
        raise ValueError("the V5 ledger has no remaining protected batch")
    batch = int(ledger["completed_batches"]) + 1
    if batch > int(ledger["maximum_batches"]):
        raise ValueError("the derived active batch exceeds the V5 budget")
    return batch


def current_run_numbering_epoch(ledger: dict) -> int:
    epoch = int(ledger["development_run_numbering"]["current_epoch"])
    if epoch <= 0:
        raise ValueError("the V5 run-numbering epoch must be positive")
    return epoch


def next_run_number(ledger: dict, batch: int) -> int:
    epoch = current_run_numbering_epoch(ledger)
    high_watermarks = ledger.get("development_run_high_watermarks", {})
    observed = [int(high_watermarks.get(str(batch), 0))]
    observed.extend(
        int(run["run_number"])
        for run in ledger.get("development_runs", [])
        if int(run["batch"]) == batch
        and int(run["run_numbering_epoch"]) == epoch
    )
    if RUNS_ROOT.is_dir():
        for child in RUNS_ROOT.iterdir():
            match = RUN_DIRECTORY_PATTERN.fullmatch(child.name)
            if match and int(match.group("batch")) == batch:
                observed.append(int(match.group("run")))
    return max(observed, default=0) + 1


def resolve_goal(cli_goal: str | None) -> dict:
    policy = json.loads(POLICY.read_text())
    readiness = policy["manual_analysis_firewall"]["development_readiness"]
    environment_name = readiness["goal_environment_variable"]
    environment_raw = os.environ.get(environment_name)

    environment_goal = (
        parse_goal(environment_raw, environment_name)
        if environment_raw is not None
        else None
    )
    argument_goal = (
        parse_goal(cli_goal, "--goal") if cli_goal is not None else None
    )

    if environment_goal is not None and argument_goal is not None:
        if not math.isclose(
            environment_goal,
            argument_goal,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{environment_name}={environment_goal} conflicts with "
                f"--goal={argument_goal}"
            )
        requested_goal = environment_goal
        goal_source = "environment_and_cli"
    elif environment_goal is not None:
        requested_goal = environment_goal
        goal_source = "environment"
    elif argument_goal is not None:
        requested_goal = argument_goal
        goal_source = "cli_prompt_transfer"
    else:
        requested_goal = float(readiness["default_requested_goal"])
        goal_source = "policy_default"

    protocol_floor = float(readiness["protocol_floor"])
    return {
        "schema_version": 1,
        "protocol_name": policy["protocol_name"],
        "protocol_revision": policy["protocol_revision"],
        "policy": str(POLICY.relative_to(POLICY.parents[1])),
        "policy_sha256": policy_sha256(),
        "goal_environment_variable": environment_name,
        "goal_source": goal_source,
        "requested_development_rmse_gain": requested_goal,
        "protocol_floor": protocol_floor,
        "effective_readiness_target": max(requested_goal, protocol_floor),
    }


def allocate_run(
    record: dict,
    prompt_text: str | None = None,
) -> tuple[dict, str]:
    ledger = json.loads(LEDGER.read_text())
    batch = active_batch(ledger)
    run_numbering_epoch = current_run_numbering_epoch(ledger)
    run_number = next_run_number(ledger, batch)
    code = goal_code(record["requested_development_rmse_gain"])
    run_id = f"v5_batch{batch}_run_{run_number:03d}_goal_{code}"
    run_directory = RUNS_ROOT / run_id
    registration = run_directory / "goal_registration.json"
    prompt_path = run_directory / "kickstart_prompt.txt"
    prompt_bytes = None
    if prompt_text is not None:
        if not prompt_text:
            raise ValueError("--prompt-save text must not be empty")
        if "\x00" in prompt_text:
            raise ValueError("--prompt-save text must not contain NUL bytes")
        prompt_bytes = prompt_text.encode("utf-8")

    RUNS_ROOT.mkdir(exist_ok=True)
    run_directory.mkdir()

    record.update(
        {
            "batch": batch,
            "run_numbering_epoch": run_numbering_epoch,
            "run_number": run_number,
            "run_id": run_id,
            "run_directory": str(run_directory.relative_to(POLICY.parents[1])),
        }
    )
    if prompt_bytes is not None:
        record["kickstart_prompt"] = {
            "saved": True,
            "scope": "literal user kickstart message only",
            "path": str(prompt_path.relative_to(POLICY.parents[1])),
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "authorization_role": "audit metadata only",
            "validation_gate": False,
        }
    else:
        record["kickstart_prompt"] = {
            "saved": False,
            "scope": "literal user kickstart message only",
            "path": None,
            "sha256": None,
            "authorization_role": "audit metadata only",
            "validation_gate": False,
        }
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    registration_sha256 = hashlib.sha256(serialized.encode()).hexdigest()

    relative_registration = str(registration.relative_to(POLICY.parents[1]))
    run_record = {
        "batch": batch,
        "run_numbering_epoch": run_numbering_epoch,
        "run_number": run_number,
        "run_id": run_id,
        "goal_code": code,
        "requested_development_rmse_gain": record[
            "requested_development_rmse_gain"
        ],
        "effective_readiness_target": record["effective_readiness_target"],
        "status": "registered_development",
        "legacy_layout": False,
        "run_directory": record["run_directory"],
        "goal_registration": relative_registration,
        "goal_registration_sha256": registration_sha256,
        "kickstart_prompt": record["kickstart_prompt"],
    }
    ledger.setdefault("development_runs", []).append(run_record)
    ledger.setdefault("development_run_high_watermarks", {})[
        str(batch)
    ] = run_number

    temporary_ledger = LEDGER.with_name(
        f".{LEDGER.name}.{os.getpid()}.tmp"
    )
    try:
        if prompt_bytes is not None:
            prompt_path.write_bytes(prompt_bytes)
        registration.write_text(serialized)
        temporary_ledger.write_text(json.dumps(ledger, indent=2) + "\n")
        os.replace(temporary_ledger, LEDGER)
    except Exception:
        temporary_ledger.unlink(missing_ok=True)
        registration.unlink(missing_ok=True)
        prompt_path.unlink(missing_ok=True)
        run_directory.rmdir()
        raise

    return record, serialized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and freeze a V5 kickstart development goal."
    )
    parser.add_argument(
        "--goal",
        help=(
            "Direct prompt-to-tool fallback. Must agree with "
            "ROGII_V5_DEVELOPMENT_GOAL when both are present."
        ),
    )
    parser.add_argument(
        "--prompt-save",
        metavar="USER_KICKSTART_TEXT",
        help=(
            "Save the literal user kickstart message as kickstart_prompt.txt "
            "inside a newly allocated run. Its digest is computed internally."
        ),
    )
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON record path; existing files are never overwritten.",
    )
    storage.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Allocate the next active-batch run directory, freeze its goal "
            "record, and register it in the V5 ledger."
        ),
    )
    args = parser.parse_args()

    if args.prompt_save is not None and not args.new_run:
        parser.error("--prompt-save requires --new-run")

    try:
        record = resolve_goal(args.goal)
    except ValueError as error:
        parser.error(str(error))

    if args.new_run:
        try:
            record, serialized = allocate_run(record, args.prompt_save)
        except (FileExistsError, OSError, ValueError) as error:
            parser.error(str(error))
    else:
        serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"

    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing goal record: {args.output}"
            )
        if not args.output.parent.is_dir():
            raise FileNotFoundError(
                f"goal record parent does not exist: {args.output.parent}"
            )
        args.output.write_text(serialized)

    print(serialized, end="")


if __name__ == "__main__":
    main()
