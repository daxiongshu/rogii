from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MANIFEST_PATH = ROOT / "manifests" / "protocol_sources.json"
ACTIVE_PATH = ROOT / "limited_query_cv" / "active_cv_protocol.json"
FROZEN_PATH = ROOT / "limited_query_cv" / "frozen_previous_best_v5.json"

# The startup surface COOKBOOK.md names and verify_active_cv_protocol.py
# asserts.
REQUIRED_HUMAN_READING = {
    "COOKBOOK.md",
    "limited_query_cv/CV_PROTOCOL_V5.md",
}
REQUIRED_MACHINE_SOURCES = {
    "limited_query_cv/cv_protocol_v5.json",
    "limited_query_cv/cv_query_ledger_v5.json",
    "limited_query_cv/frozen_previous_best_v5.json",
    "limited_query_cv/verify_active_cv_protocol.py",
}
# Which subtree manifest each frozen-parent entry must agree with.
SUBTREE_METRICS = {
    "v20": ("manifests/provenance.json", ("oof", "pooled_rmse")),
    "c016": (
        "c016/manifests/c016_provenance.json",
        ("metrics", "legal_production_oof_rmse"),
    ),
    "run4_structural_surface_router_s06": (
        "run4/manifests/run4_provenance.json",
        ("metrics", "deploy_parity_pooled_oof_rmse"),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        data = data[key]
    return data


def verify_sources() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    failures = []
    total = 0
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        total += path.stat().st_size
        observed = sha256(path)
        if observed != expected:
            failures.append(f"changed {relative}: {observed} != {expected}")
    if failures:
        raise RuntimeError(
            "protocol snapshot verification failed:\n" + "\n".join(failures)
        )
    return {
        "files": len(manifest["sha256"]),
        "bytes": total,
        "snapshot_commit": manifest["snapshot_commit"],
        "snapshot_utc": manifest["snapshot_utc"],
    }


def verify_startup_surface() -> dict[str, Any]:
    policy = load_json(ROOT / "limited_query_cv" / "cv_protocol_v5.json")
    startup = policy["agent_startup_surface"]
    if set(startup["required_human_reading"]) != REQUIRED_HUMAN_READING:
        raise RuntimeError("required human reading changed")
    if set(startup["machine_sources"]) != REQUIRED_MACHINE_SOURCES:
        raise RuntimeError("machine sources changed")
    missing = [
        name
        for name in REQUIRED_HUMAN_READING | REQUIRED_MACHINE_SOURCES
        if not (ROOT / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"startup surface incomplete: {missing}")
    return {
        "human_reading": sorted(REQUIRED_HUMAN_READING),
        "machine_sources": sorted(REQUIRED_MACHINE_SOURCES),
    }


def verify_amendments() -> dict[str, Any]:
    active = load_json(ACTIVE_PATH)
    checked = []
    for relative in active["active_amendments"]:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing active amendment: {relative}")
        record = load_json(path)
        for key in ("policy", "documentation", "authorization"):
            target = record.get(key)
            if not isinstance(target, str) or "/" not in target:
                continue
            expected = record.get(f"{key}_sha256")
            resolved = ROOT / target
            if not resolved.is_file():
                raise RuntimeError(f"{relative}: missing {key} {target}")
            if expected is not None and sha256(resolved) != expected:
                raise RuntimeError(
                    f"{relative}: {key} hash changed for {target}"
                )
        checked.append(relative)
    return {
        "protocol_revision": active["protocol_revision"],
        "active_amendments": len(checked),
        "baseline": active["baseline"],
        "baseline_oof_rmse": active["baseline_oof_rmse"],
    }


def verify_frozen_parent_chain() -> dict[str, Any]:
    frozen = load_json(FROZEN_PATH)
    parents = frozen["parents"]
    report: dict[str, Any] = {
        "active_parent_id": frozen["active_parent_id"],
        "checked_against_subtrees": {},
    }
    for solution, (relative, keys) in SUBTREE_METRICS.items():
        if solution not in parents:
            raise RuntimeError(f"{solution} missing from the frozen registry")
        manifest = REPO / relative
        if not manifest.is_file():
            report["checked_against_subtrees"][solution] = "subtree_absent"
            continue
        recorded = dig(load_json(manifest), keys)
        registry = parents[solution].get(
            "selection_clean_pooled_oof_rmse"
        )
        if registry != recorded:
            raise RuntimeError(
                f"{solution}: frozen registry says {registry}, "
                f"{relative} says {recorded}"
            )
        report["checked_against_subtrees"][solution] = recorded
    history = [entry["solution_id"] for entry in frozen["activation_history"]]
    if history != ["v20", "c016", "run4_structural_surface_router_s06"]:
        raise RuntimeError(f"unexpected activation history: {history}")
    report["activation_history"] = history
    if frozen["activation_policy"]["automatic"] is not False:
        raise RuntimeError("parent activation is no longer user-approved only")
    return report


def command_verify(_: argparse.Namespace) -> None:
    report = {
        "sources": verify_sources(),
        "startup_surface": verify_startup_surface(),
        "amendments": verify_amendments(),
        "frozen_parent_chain": verify_frozen_parent_chain(),
        "status": "verified",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Verify the CV Protocol v5 governance snapshot and its agreement "
            "with the v20, C016, and Run4 subtrees."
        )
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify", help="Check hashes, startup surface, amendments, lineage."
    )
    verify.set_defaults(handler=command_verify)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
