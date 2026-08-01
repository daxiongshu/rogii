from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from limited_query_cv import (
    evaluate_v5_run6_trajectory_segformer_clean as plain,
)
from limited_query_cv import (
    evaluate_v5_run6_trajectory_segformer_gated_clean as gated,
)
from limited_query_cv import (
    train_v5_run6_trajectory_categorical_segformer as model,
)


RUN_ID = plain.RUN_ID
FAMILY = plain.FAMILY
ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_006_goal_050"
)
D570_PATH = ROOT / "d209_amplitude_clean.npz"
D570_SHA256 = (
    "5e3b6ee41f8705cc6b0044b36255bf8e8b3b6c4752c2afa9f6049d83f77cd310"
)
SELECTOR_SOURCE_SHA256 = (
    "1402a82d99a0f902d8801db34a623a9beb0017266ae2b646c6a7bb91482bcc73"
)
SELECTOR_GATE_SHA256 = (
    "e5bcebafb3ce173da222208d6c5f7e75183f6b60f4e8a670e6317664b7217aee"
)
SELECTOR_GATE_PATH = (
    ROOT / "trajectory_categorical_gated_clean_selector_gate.json"
)
DIRECTORY_PATTERN = "trajectory_categorical_segformer_f{fold}"


def scalar(archive: np.lib.npyio.NpzFile, name: str):
    return np.asarray(archive[name]).item()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def expected_names() -> set[str]:
    names = {"result.json"}
    for snapshot in plain.SNAPSHOTS:
        names.add(f"epoch{snapshot:03d}.pt")
        names.add(f"epoch{snapshot:03d}_prediction.npz")
    return names


def verify_selector_lineage() -> None:
    if (
        plain.sha256(Path(gated.__file__)) != SELECTOR_SOURCE_SHA256
        or plain.sha256(SELECTOR_GATE_PATH) != SELECTOR_GATE_SHA256
    ):
        raise RuntimeError("A832 selector lineage changed")
    record = json.loads(SELECTOR_GATE_PATH.read_text())
    if (
        record.get("status") != "passed"
        or record.get("source_sha256") != SELECTOR_SOURCE_SHA256
        or record.get("model_source_sha256") != plain.SOURCE_SHA256
        or record.get("model_gate_sha256") != plain.GATE_SHA256
        or bool(record.get("audit_fold_loaded"))
        or bool(record.get("confirmation_regroupings_loaded"))
        or bool(record.get("f4_loaded"))
    ):
        raise RuntimeError("A832 selector gate changed")


def parent_role(
    parent: np.lib.npyio.NpzFile, fold: int
) -> dict[str, np.ndarray]:
    well_ids = np.asarray(parent["well_ids"]).astype(str)
    folds = np.asarray(parent["folds"], dtype=np.int64)
    starts = np.asarray(parent["row_starts"], dtype=np.int64)
    selected = np.flatnonzero(folds == fold)
    local_starts = [0]
    values: dict[str, list[np.ndarray]] = {
        "baseline": [],
        "truth": [],
        "d570": [],
    }
    for index in selected:
        left, right = int(starts[index]), int(starts[index + 1])
        values["baseline"].append(
            np.asarray(parent["baseline"][left:right]).copy()
        )
        values["truth"].append(
            np.asarray(parent["truth"][left:right]).copy()
        )
        values["d570"].append(
            np.asarray(parent["prediction"][left:right]).copy()
        )
        local_starts.append(local_starts[-1] + right - left)
    return {
        "well_ids": well_ids[selected],
        "row_starts": np.asarray(local_starts, dtype=np.int64),
        **{
            name: np.concatenate(parts)
            for name, parts in values.items()
        },
    }


def verify_prediction(
    path: Path,
    checkpoint: Path,
    fold: int,
    snapshot: int,
    parent_fold: dict[str, np.ndarray],
    *,
    reference: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "fold",
            "variant",
            "seed",
            "epoch",
            "well_ids",
            "row_starts",
            "d570",
            "truth",
            "source_sha256",
            "checkpoint_sha256",
            "gate_sha256",
            "audit_fold_loaded",
            "confirmation_regroupings_loaded",
            "f4_loaded",
            *{
                f"candidate_t{int(round(temperature * 100)):03d}"
                for temperature in plain.TEMPERATURES
            },
        }
        if set(archive.files) != required:
            raise RuntimeError(f"{path}: prediction field set changed")
        if (
            int(scalar(archive, "fold")) != fold
            or str(scalar(archive, "variant")) != model.VARIANT
            or int(scalar(archive, "seed")) != model.SEED
            or int(scalar(archive, "epoch")) != snapshot
            or str(scalar(archive, "source_sha256")) != plain.SOURCE_SHA256
            or str(scalar(archive, "gate_sha256")) != plain.GATE_SHA256
            or str(scalar(archive, "checkpoint_sha256"))
            != plain.sha256(checkpoint)
            or bool(scalar(archive, "audit_fold_loaded"))
            or bool(scalar(archive, "confirmation_regroupings_loaded"))
            or bool(scalar(archive, "f4_loaded"))
        ):
            raise RuntimeError(f"{path}: prediction provenance changed")
        loaded = {
            "well_ids": np.asarray(archive["well_ids"]).astype(str),
            "row_starts": np.asarray(
                archive["row_starts"], dtype=np.int64
            ).copy(),
            "d570": np.asarray(archive["d570"]).copy(),
            "truth": np.asarray(archive["truth"]).copy(),
        }
        for temperature in plain.TEMPERATURES:
            key = f"candidate_t{int(round(temperature * 100)):03d}"
            loaded[key] = np.asarray(archive[key]).copy()
    if (
        not np.array_equal(loaded["well_ids"], parent_fold["well_ids"])
        or not np.array_equal(
            loaded["row_starts"], parent_fold["row_starts"]
        )
        or not np.array_equal(loaded["d570"], parent_fold["d570"])
        or not np.array_equal(loaded["truth"], parent_fold["truth"])
    ):
        raise RuntimeError(f"{path}: D570/layout binding changed")
    if reference is not None:
        for name in ("well_ids", "row_starts", "d570", "truth"):
            if not np.array_equal(loaded[name], reference[name]):
                raise RuntimeError(
                    f"{path}: snapshot layout changed for {name}"
                )
    return loaded


def verify_result(path: Path, fold: int) -> None:
    record = json.loads(path.read_text())
    if (
        record.get("run_id") != RUN_ID
        or record.get("family") != FAMILY
        or record.get("variant") != model.VARIANT
        or int(record.get("fold", -1)) != fold
        or int(record.get("seed", -1)) != model.SEED
        or record.get("source_sha256") != plain.SOURCE_SHA256
        or record.get("gate_sha256") != plain.GATE_SHA256
        or bool(record.get("audit_fold_loaded"))
        or bool(record.get("confirmation_regroupings_loaded"))
        or bool(record.get("f4_loaded"))
    ):
        raise RuntimeError(f"{path}: result provenance changed")
    snapshots = record.get("snapshots")
    if (
        not isinstance(snapshots, list)
        or [int(item["epoch"]) for item in snapshots]
        != list(plain.SNAPSHOTS)
    ):
        raise RuntimeError(f"{path}: snapshot result bank changed")


def artifact_manifest(output: Path) -> None:
    verify_selector_lineage()
    if plain.sha256(D570_PATH) != D570_SHA256:
        raise RuntimeError("immutable D570 changed")
    with np.load(D570_PATH, allow_pickle=False) as parent:
        if (
            bool(scalar(parent, "audit_fold_loaded"))
            or bool(scalar(parent, "confirmation_regroupings_loaded"))
        ):
            raise RuntimeError("D570 protected flags changed")
        folds = []
        for fold in range(4):
            directory = ROOT / DIRECTORY_PATTERN.format(fold=fold)
            actual = {
                path.name
                for path in directory.iterdir()
                if path.is_file()
            }
            if actual != expected_names():
                missing = sorted(expected_names() - actual)
                extra = sorted(actual - expected_names())
                raise RuntimeError(
                    f"F{fold} incomplete artifact bank; "
                    f"missing={missing}, extra={extra}"
                )
            result_path = directory / "result.json"
            verify_result(result_path, fold)
            expected_parent = parent_role(parent, fold)
            reference = None
            artifacts = {"result.json": plain.sha256(result_path)}
            layout = None
            for snapshot in plain.SNAPSHOTS:
                checkpoint = directory / f"epoch{snapshot:03d}.pt"
                prediction = (
                    directory
                    / f"epoch{snapshot:03d}_prediction.npz"
                )
                loaded = verify_prediction(
                    prediction,
                    checkpoint,
                    fold,
                    snapshot,
                    expected_parent,
                    reference=reference,
                )
                if reference is None:
                    reference = loaded
                    layout = {
                        "well_count": int(len(loaded["well_ids"])),
                        "row_count": int(len(loaded["truth"])),
                        "well_ids_sha256": array_sha256(
                            loaded["well_ids"].astype("S8")
                        ),
                        "row_starts_sha256": array_sha256(
                            loaded["row_starts"]
                        ),
                        "truth_sha256": array_sha256(loaded["truth"]),
                        "d570_sha256": array_sha256(loaded["d570"]),
                    }
                artifacts[checkpoint.name] = plain.sha256(checkpoint)
                artifacts[prediction.name] = plain.sha256(prediction)
            folds.append(
                {
                    "fold": fold,
                    "directory": str(directory),
                    "layout": layout,
                    "artifacts": artifacts,
                }
            )
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "frozen_complete_artifact_manifest",
        "finisher_source_sha256": plain.sha256(Path(__file__)),
        "model_source_sha256": plain.SOURCE_SHA256,
        "model_gate_sha256": plain.GATE_SHA256,
        "selector_source_sha256": SELECTOR_SOURCE_SHA256,
        "selector_gate_sha256": SELECTOR_GATE_SHA256,
        "d570_path": str(D570_PATH),
        "d570_sha256": D570_SHA256,
        "folds": folds,
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    plain.write_json(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status")
        != "frozen_complete_artifact_manifest"
        or manifest.get("run_id") != RUN_ID
        or manifest.get("family") != FAMILY
        or manifest.get("finisher_source_sha256")
        != plain.sha256(Path(__file__))
        or manifest.get("model_source_sha256") != plain.SOURCE_SHA256
        or manifest.get("model_gate_sha256") != plain.GATE_SHA256
        or manifest.get("selector_source_sha256")
        != SELECTOR_SOURCE_SHA256
        or manifest.get("selector_gate_sha256")
        != SELECTOR_GATE_SHA256
        or manifest.get("d570_sha256") != D570_SHA256
        or bool(manifest.get("target_metric_computed"))
        or bool(manifest.get("audit_fold_loaded"))
        or bool(manifest.get("confirmation_regroupings_loaded"))
        or bool(manifest.get("f4_loaded"))
    ):
        raise RuntimeError("invalid immutable A830 artifact manifest")
    for fold_record in manifest["folds"]:
        directory = Path(fold_record["directory"])
        if set(fold_record["artifacts"]) != expected_names():
            raise RuntimeError("manifest artifact name set changed")
        for name, digest in fold_record["artifacts"].items():
            if plain.sha256(directory / name) != digest:
                raise RuntimeError(f"manifested artifact changed: {name}")
    return manifest


def load_fold(
    fold: int,
    parent: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    parent_fold = parent_role(parent, fold)
    directory = ROOT / DIRECTORY_PATTERN.format(fold=fold)
    reference = None
    candidates = {}
    for snapshot in plain.SNAPSHOTS:
        checkpoint = directory / f"epoch{snapshot:03d}.pt"
        prediction = directory / f"epoch{snapshot:03d}_prediction.npz"
        loaded = verify_prediction(
            prediction,
            checkpoint,
            fold,
            snapshot,
            parent_fold,
            reference=reference,
        )
        if reference is None:
            reference = loaded
        for temperature in plain.TEMPERATURES:
            key = f"candidate_t{int(round(temperature * 100)):03d}"
            candidates[plain.candidate_key(snapshot, temperature)] = (
                loaded[key]
            )
    result: dict[str, object] = {
        "fold": fold,
        "source_sha256": plain.SOURCE_SHA256,
        "gate_sha256": plain.GATE_SHA256,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
        "well_ids": parent_fold["well_ids"],
        "row_starts": parent_fold["row_starts"],
        "baseline": parent_fold["baseline"],
        "truth": parent_fold["truth"],
        "d570": parent_fold["d570"],
        "candidates": candidates,
    }
    gated.validate_fold(result, fold)
    return result


def coefficient_statistics(
    fold: dict[str, object],
    treatment: str,
    snapshot: int,
    temperature: float,
) -> dict[str, np.ndarray | float | int]:
    truth = np.asarray(fold["truth"], dtype=np.float64)
    parent = np.asarray(fold["d570"], dtype=np.float64)
    candidate = np.asarray(
        fold["candidates"][plain.candidate_key(snapshot, temperature)],
        dtype=np.float64,
    )
    mask = gated.treatment_mask(
        fold, snapshot, temperature, treatment
    )
    error = parent - truth
    correction = (candidate - parent) * mask
    starts = np.asarray(fold["row_starts"], dtype=np.int64)
    n_wells = len(starts) - 1
    count = np.empty(n_wells, dtype=np.float64)
    a = np.empty(n_wells, dtype=np.float64)
    b = np.empty(n_wells, dtype=np.float64)
    c = np.empty(n_wells, dtype=np.float64)
    toe_count = np.empty(n_wells, dtype=np.float64)
    toe_a = np.empty(n_wells, dtype=np.float64)
    toe_b = np.empty(n_wells, dtype=np.float64)
    toe_c = np.empty(n_wells, dtype=np.float64)
    endpoint_error = np.empty(n_wells, dtype=np.float64)
    endpoint_correction = np.empty(n_wells, dtype=np.float64)
    for index, (left, right) in enumerate(
        zip(starts[:-1], starts[1:])
    ):
        local_error = error[left:right]
        local_correction = correction[left:right]
        width = max(len(local_error) // 4, 1)
        toe_error = local_error[-width:]
        toe_correction = local_correction[-width:]
        count[index] = len(local_error)
        a[index] = np.dot(local_error, local_error)
        b[index] = np.dot(local_error, local_correction)
        c[index] = np.dot(local_correction, local_correction)
        toe_count[index] = width
        toe_a[index] = np.dot(toe_error, toe_error)
        toe_b[index] = np.dot(toe_error, toe_correction)
        toe_c[index] = np.dot(toe_correction, toe_correction)
        endpoint_error[index] = local_error[-1]
        endpoint_correction[index] = local_correction[-1]
    return {
        "count": count,
        "a": a,
        "b": b,
        "c": c,
        "toe_count": toe_count,
        "toe_a": toe_a,
        "toe_b": toe_b,
        "toe_c": toe_c,
        "endpoint_error": endpoint_error,
        "endpoint_correction": endpoint_correction,
        "corrected_rows": int(mask.sum()),
    }


def metrics_from_statistics(
    statistics: list[dict[str, np.ndarray | float | int]],
    share: float,
) -> dict[str, float]:
    def concatenate(name: str) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(record[name], dtype=np.float64)
                for record in statistics
            ]
        )

    count = concatenate("count")
    a = concatenate("a")
    b = concatenate("b")
    c = concatenate("c")
    sse = np.maximum(a + 2.0 * share * b + share * share * c, 0.0)
    toe_count = concatenate("toe_count")
    toe_sse = np.maximum(
        concatenate("toe_a")
        + 2.0 * share * concatenate("toe_b")
        + share * share * concatenate("toe_c"),
        0.0,
    )
    endpoint = (
        concatenate("endpoint_error")
        + share * concatenate("endpoint_correction")
    )
    return {
        "pooled_rmse": float(np.sqrt(sse.sum() / count.sum())),
        "mean_well_rmse": float(np.mean(np.sqrt(sse / count))),
        "toe_quartile_rmse": float(
            np.sqrt(toe_sse.sum() / toe_count.sum())
        ),
        "endpoint_rmse": float(np.sqrt(np.mean(np.square(endpoint)))),
    }


def fast_records(
    folds: list[dict[str, object]], training_roles: list[int]
) -> list[dict[str, object]]:
    parent_scores = {
        role: plain.rmse(
            np.asarray(folds[role]["d570"]),
            np.asarray(folds[role]["truth"]),
        )
        for role in training_roles
    }
    records = []
    for treatment in gated.TREATMENTS:
        for snapshot in plain.SNAPSHOTS:
            for temperature in plain.TEMPERATURES:
                statistics = {
                    role: coefficient_statistics(
                        folds[role],
                        treatment,
                        snapshot,
                        temperature,
                    )
                    for role in training_roles
                }
                for share in plain.SHARES:
                    role_scores = {
                        role: metrics_from_statistics(
                            [statistics[role]], share
                        )["pooled_rmse"]
                        for role in training_roles
                    }
                    eligible = all(
                        role_scores[role]
                        <= parent_scores[role] + 1e-12
                        for role in training_roles
                    )
                    aggregate = metrics_from_statistics(
                        [statistics[role] for role in training_roles],
                        share,
                    )
                    records.append(
                        {
                            "treatment": treatment,
                            "snapshot": snapshot,
                            "temperature": temperature,
                            "share": share,
                            "eligible": eligible,
                            "corrected_rows": sum(
                                int(statistics[role]["corrected_rows"])
                                for role in training_roles
                            ),
                            "role_rmse": {
                                str(key): value
                                for key, value in role_scores.items()
                            },
                            **aggregate,
                        }
                    )
    eligible_records = [
        record for record in records if record["eligible"]
    ]
    ranks = np.column_stack(
        [
            rankdata(
                [record[name] for record in eligible_records],
                method="average",
            )
            for name in plain.METRICS
        ]
    )
    for record, values in zip(eligible_records, ranks):
        record["metric_ranks"] = {
            name: float(value)
            for name, value in zip(plain.METRICS, values)
        }
        record["mean_rank"] = float(np.mean(values))
    return records


def fast_select(
    folds: list[dict[str, object]], held_role: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = fast_records(
        folds, [role for role in range(4) if role != held_role]
    )
    selected = min(
        (record for record in records if record["eligible"]),
        key=lambda record: (
            record["mean_rank"],
            *(record[name] for name in plain.METRICS),
            record["share"],
            record["corrected_rows"],
            gated.TREATMENTS.index(record["treatment"]),
            record["snapshot"],
            plain.TEMPERATURES.index(record["temperature"]),
        ),
    )
    return selected, records


def arithmetic_gate(output: Path) -> None:
    verify_selector_lineage()
    folds = [gated.synthetic_fold(role) for role in range(4)]
    maximum_difference = 0.0
    selections = {}
    for held in range(4):
        direct_selected, direct_records = gated.select_recipe(folds, held)
        fast_selected, optimized_records = fast_select(folds, held)
        if len(direct_records) != len(optimized_records):
            raise RuntimeError("optimized candidate count changed")
        direct_by_key = {
            (
                record["treatment"],
                record["snapshot"],
                record["temperature"],
                record["share"],
            ): record
            for record in direct_records
        }
        for record in optimized_records:
            key = (
                record["treatment"],
                record["snapshot"],
                record["temperature"],
                record["share"],
            )
            original = direct_by_key[key]
            if (
                bool(record["eligible"]) != bool(original["eligible"])
                or int(record["corrected_rows"])
                != int(original["corrected_rows"])
            ):
                raise RuntimeError("optimized eligibility changed")
            for name in plain.METRICS:
                difference = abs(
                    float(record[name]) - float(original[name])
                )
                maximum_difference = max(maximum_difference, difference)
                if difference > 1e-7:
                    raise RuntimeError(
                        f"optimized {name} changed by {difference}"
                    )
        recipe_names = (
            "treatment",
            "snapshot",
            "temperature",
            "share",
        )
        direct_recipe = {
            name: direct_selected[name] for name in recipe_names
        }
        fast_recipe = {
            name: fast_selected[name] for name in recipe_names
        }
        if direct_recipe != fast_recipe:
            raise RuntimeError("optimized selected recipe changed")
        selections[str(held)] = fast_recipe
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": "passed",
        "source_sha256": plain.sha256(Path(__file__)),
        "selector_source_sha256": SELECTOR_SOURCE_SHA256,
        "selector_gate_sha256": SELECTOR_GATE_SHA256,
        "candidate_count_per_role": len(direct_records),
        "maximum_metric_difference": maximum_difference,
        "selections": selections,
        "target_metric_computed": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    plain.write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def save_clean_archive(
    output: Path,
    folds: list[dict[str, object]],
    predictions: list[np.ndarray],
) -> None:
    if output.exists():
        raise FileExistsError(output)
    well_ids = []
    fold_ids = []
    starts = [0]
    values = {
        "baseline": [],
        "d570": [],
        "truth": [],
        "prediction": [],
    }
    for fold, prediction in enumerate(predictions):
        fold_record = folds[fold]
        local_ids = np.asarray(fold_record["well_ids"]).astype(str)
        local_starts = np.asarray(
            fold_record["row_starts"], dtype=np.int64
        )
        well_ids.append(local_ids)
        fold_ids.append(
            np.full(len(local_ids), fold, dtype=np.int8)
        )
        starts.extend((local_starts[1:] + starts[-1]).tolist())
        values["baseline"].append(
            np.asarray(fold_record["baseline"], dtype=np.float32)
        )
        values["d570"].append(
            np.asarray(fold_record["d570"], dtype=np.float32)
        )
        values["truth"].append(
            np.asarray(fold_record["truth"], dtype=np.float32)
        )
        values["prediction"].append(
            np.asarray(prediction, dtype=np.float32)
        )
    np.savez_compressed(
        output,
        well_ids=np.concatenate(well_ids),
        folds=np.concatenate(fold_ids),
        row_starts=np.asarray(starts, dtype=np.int64),
        baseline=np.concatenate(values["baseline"]),
        d570=np.concatenate(values["d570"]),
        truth=np.concatenate(values["truth"]),
        prediction=np.concatenate(values["prediction"]),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
        f4_loaded=np.asarray(False),
    )


def evaluate(
    manifest_path: Path, output_json: Path, output_npz: Path
) -> None:
    verify_selector_lineage()
    manifest = load_manifest(manifest_path)
    if plain.sha256(D570_PATH) != D570_SHA256:
        raise RuntimeError("immutable D570 changed")
    with np.load(D570_PATH, allow_pickle=False) as parent:
        folds = [load_fold(fold, parent) for fold in range(4)]
    selections = []
    predictions = []
    fold_gains = []
    for held in range(4):
        selected, records = fast_select(folds, held)
        prediction = gated.blend(
            folds[held],
            str(selected["treatment"]),
            int(selected["snapshot"]),
            float(selected["temperature"]),
            float(selected["share"]),
        )
        truth = np.asarray(folds[held]["truth"])
        parent_prediction = np.asarray(folds[held]["d570"])
        parent_rmse = plain.rmse(parent_prediction, truth)
        selected_rmse = plain.rmse(prediction, truth)
        gain = parent_rmse - selected_rmse
        fold_gains.append(gain)
        predictions.append(prediction)
        selections.append(
            {
                "held_fold": held,
                "selection_folds": [
                    role for role in range(4) if role != held
                ],
                "selected": {
                    name: selected[name]
                    for name in (
                        "treatment",
                        "snapshot",
                        "temperature",
                        "share",
                        "mean_rank",
                    )
                },
                "eligible_recipes": int(
                    sum(record["eligible"] for record in records)
                ),
                "held_d570_rmse": parent_rmse,
                "held_selected_rmse": selected_rmse,
                "held_gain_after_d570": gain,
            }
        )
    truth = np.concatenate(
        [np.asarray(fold["truth"]) for fold in folds]
    )
    baseline = np.concatenate(
        [np.asarray(fold["baseline"]) for fold in folds]
    )
    d570 = np.concatenate(
        [np.asarray(fold["d570"]) for fold in folds]
    )
    prediction = np.concatenate(predictions)
    baseline_rmse = plain.rmse(baseline, truth)
    d570_rmse = plain.rmse(d570, truth)
    selected_rmse = plain.rmse(prediction, truth)
    gain_after_d570 = d570_rmse - selected_rmse
    passed = (
        gain_after_d570 > 0.0
        and all(gain > 0.0 for gain in fold_gains)
    )
    save_clean_archive(output_npz, folds, predictions)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "status": (
            "accepted_selection_clean_F0_F3"
            if passed
            else "rejected_selection_clean_F0_F3"
        ),
        "source_sha256": plain.sha256(Path(__file__)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": plain.sha256(manifest_path),
        "manifest_status": manifest["status"],
        "selector_source_sha256": SELECTOR_SOURCE_SHA256,
        "selector_gate_sha256": SELECTOR_GATE_SHA256,
        "d570_sha256": D570_SHA256,
        "baseline_v20_rmse": baseline_rmse,
        "d570_rmse": d570_rmse,
        "selected_rmse": selected_rmse,
        "gain_after_d570": gain_after_d570,
        "gain_vs_v20": baseline_rmse - selected_rmse,
        "every_fold_positive_after_d570": bool(
            all(gain > 0.0 for gain in fold_gains)
        ),
        "fold_records": selections,
        "clean_archive_path": str(output_npz),
        "clean_archive_sha256": plain.sha256(output_npz),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "f4_loaded": False,
    }
    plain.write_json(output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--gate-only", action="store_true")
    modes.add_argument("--manifest-only", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-npz", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gate_only:
        if args.output_json is None:
            raise ValueError("--output-json is required for --gate-only")
        arithmetic_gate(args.output_json)
    elif args.manifest_only:
        if args.manifest is None:
            raise ValueError("--manifest is required for --manifest-only")
        artifact_manifest(args.manifest)
    else:
        if (
            args.manifest is None
            or args.output_json is None
            or args.output_npz is None
        ):
            raise ValueError(
                "--manifest, --output-json, and --output-npz "
                "are required for --evaluate"
            )
        evaluate(args.manifest, args.output_json, args.output_npz)


if __name__ == "__main__":
    main()
