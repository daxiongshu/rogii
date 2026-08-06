from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import limited_query_cv.evaluate_v5_run6_recurrent_clean as metric_parent
import limited_query_cv.freeze_v5_run6_protected_posterior_expansion as expansion
from limited_query_cv.evaluate_v5_run6_recurrent_clean import (
    blend_metrics,
    metric_sufficient_statistics,
)


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "fold_purged_protected_posterior_fusion"
VARIANTS = (
    "posterior_only_b12",
    "posterior_physical_b12",
    "posterior_physical_b16",
)
TREATMENTS = (
    "posterior_only_b12",
    "posterior_physical_b12",
    "posterior_physical_b16",
    "equal_only_physical12",
    "equal_only_physical16",
    "equal_physical12_physical16",
    "equal_all",
)
SNAPSHOTS = (4, 8, 12, 16, 20, 24)
TEMPERATURES = (0.75, 1.0, 1.5, 2.0)
AMPLITUDES = (1.0, 1.5, 2.0, 3.0, 4.0)
SHARES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
TREATMENT_ORDER = {name: index for index, name in enumerate(TREATMENTS)}
EXPECTED_TRAIN_SOURCE_SHA256 = (
    "62bb48fd63adb994df23a997f35eb5bfc0882cccd3ea8a26127c11aecf9454e8"
)
EXPECTED_CACHE_SOURCE_SHA256 = (
    "3c518d42a3bbe723a18cb9cc592e88e8c8b3392239a14ca3a8c979f1f8e4a28e"
)
EXPECTED_METRIC_PARENT_SHA256 = (
    "a435b910b75f3400909b894e116b41636592f367892ea5523c7545080d319fae"
)
EXPECTED_D570_SHA256 = (
    "5e3b6ee41f8705cc6b0044b36255bf8e8b3b6c4752c2afa9f6049d83f77cd310"
)
EXPECTED_CACHE_HASHES = (
    "eb12c7ac801eda8837c611ec7832f8df6c4a175893bd9f9d84d512982b33c0aa",
    "43dbec13717c746d29939d1eb18922ff2ec36d56aedfd3d7da8b006ac8df95a6",
    "f7c9267a79bc7ea808db7bd28a511f0767d66813700f874529eff5b187a62eb9",
    "9376052566eb521cba17ca77a2d2c5bc594bd71c3088ee2928fbca3338d919bb",
)
EXPECTED_F0_RESULT_HASHES = {
    "posterior_only_b12": (
        "ee4c1ee0dd97b0f0dc6d67a1c87268085f6d8ab99fd81aee8e984ba42227ef8e"
    ),
    "posterior_physical_b12": (
        "1eb1afbf37206ccc6ba95e4a5725c47865b7e0905948da2aee6c106b223f42a0"
    ),
    "posterior_physical_b16": (
        "f0b1964885a447031d820902f7ac8c23a8bbdfdeb10de4734a1f47052bb911a8"
    ),
}
EXPECTED_F0_PREDICTION_HASHES = {
    "posterior_only_b12": (
        "c9193e7eddbce0bd48c797e6488468ec6299961a287b86a2f0051a624c87a01d",
        "da4b7127587d5e3332c525a38bd08304c4a87be1678c150ada2bf9a25621c122",
        "ac7ef6dafb89dfa6c5cde4f7dcc5d0d5757bfa6bc78785f52746845a9cc24dcd",
        "48fdce940a7fc0054bba69e7464294a6b45001dfe2bb6cb51e880c8bf1fbc799",
        "c3e62aee138c2b72c1c5743598b4c624caee9c772da20eb596856b755eebc6de",
        "87a53743677cb91d77f11aaf74477968135b7263f86b0002e342de82feb6bb0a",
    ),
    "posterior_physical_b12": (
        "78e686e13b2561c9fa77ede07ae575138fcdc55044b86f94792767d6f8d3f40a",
        "549d277c04ef0f9fca19c48ebf115eec46b46b4fbb53ce43d5b879d381bb9ddb",
        "cce3aefca71c6831a87b61e1e0a8675aebeb1f31c57a81811d16913a4afd13ef",
        "19c14f37d14947daa1734179d92fa72a50563bfa29238369492218cbff5cea30",
        "f796af3577afd651ccf215ca1604ffbfa40a2fca5ad7c24af403aa0528e2504a",
        "4a205218875792e1165d8fef628e8e36317a43fd739ebb55b4dcda7d07c45512",
    ),
    "posterior_physical_b16": (
        "9c62a0df73e5412a898536799aa0af1727cdb3c94fd830f1b3f390e0f88ff67e",
        "5818bf2046cd2f7efac8fe816b66142d811fb633e88c6f43177b6048ca3ab7b1",
        "2dc9dd797a50ba9b851ec40a603be71dc651e13b194723a7eb6df311f68b9707",
        "681bea240526b670d82f8432f15a5c5f966a099465fddcf54190b30accef8278",
        "1e745fc7250f9c23ebae9cf141c89cff60579de0c2cc35c00f223c192e634808",
        "02c168df1670bc253a8f76a7cac8799550f76a30dd10f5350cec4e0a36b70faa",
    ),
}
EXPECTED_F0_CHECKPOINT_HASHES = {
    "posterior_only_b12": (
        "afd4d9b521458634c0b7479c3c452a922db91096508bfb338c0550beaed22261",
        "f2e0a5f9daee6d24f2985991d628212e0ead91028487c90bbad5569d92c5012a",
        "ac2708005cd5f73d67c37748eda670207039162cc191000cfcba7379580d08ee",
        "6413c79a27b222d934b1440d3aee7d6d17ac2169594d728372867eac36c49c37",
        "20c498d3ddd1e31305c8e7f8700b4ad76c091a737e4f87471d6ca90061633847",
        "ab357e3fcc0ced843a5dac1f15e8f052c345e9b4feec86013d868f6d25613b2e",
    ),
    "posterior_physical_b12": (
        "adaed7f8ca6cca5c2c50516f4b283391e1e3768f2ee222dd0c1677a3326c876f",
        "278268c3db7e6bbf6e1de99f67fc4fbf2480503d8238c2231906ce5282018a5c",
        "9fdad8237e7f24e012dc81567954f45d68b0621706b5773af99e7848414d7898",
        "55fe9e2cd07c47c9e134aa354fe238bc6cd675c1fe47e91b82016e931894765f",
        "a3d794b44b81bcfb8ca48b0926220b7b2748ae2ba908a5c79f77ed652f8525bb",
        "216433eb898beaeb1223ccd08daa171c289110e2b07299b5d4ad28d52f25a826",
    ),
    "posterior_physical_b16": (
        "ca005e959fce3df235b38eb1a8cd2935d49e59105b0e1c59400ad390064fcc2c",
        "75b38905197e59290699278b4170a56105b40477708d58437b9837d245e92bd9",
        "1529ff17e3d2c7976b65eeaea1d2588e24a57f7591eb6021d14eee4179fa6a48",
        "0fe8e58be5b42a884d558a639835c7a4895bfa2273300d297d76a089d75a8086",
        "2a4f561b87e444d33d480f2361aacda51893c23aecb0bf7d5552bf26fc35a387",
        "4abf0af80aae69aab67c99dfb5a57ba75ecdb9794364903c3c1961854d0bb2ca",
    ),
}
DEFAULT_ROOT = Path("limited_query_cv/runs/v5_batch2_run_006_goal_050")
DEFAULT_D570 = DEFAULT_ROOT / "d209_amplitude_clean.npz"
DEFAULT_EXPANSION_MANIFEST = (
    DEFAULT_ROOT / "protected_posterior_expansion_manifest.json"
)
TRAIN_SOURCE = Path("limited_query_cv/train_v5_run6_protected_posterior_fusion.py")
CACHE_SOURCE = Path("limited_query_cv/cache_v5_run6_protected_posterior_volume.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def treatment_members(treatment: str) -> tuple[str, ...]:
    if treatment in VARIANTS:
        return (treatment,)
    if treatment == "equal_only_physical12":
        return ("posterior_only_b12", "posterior_physical_b12")
    if treatment == "equal_only_physical16":
        return ("posterior_only_b12", "posterior_physical_b16")
    if treatment == "equal_physical12_physical16":
        return ("posterior_physical_b12", "posterior_physical_b16")
    if treatment == "equal_all":
        return VARIANTS
    raise ValueError(f"unknown posterior treatment: {treatment}")


def average_corrections(values: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values).astype(np.float64), axis=0)


def assert_metric_parent() -> None:
    actual = sha256(Path(metric_parent.__file__))
    if actual != EXPECTED_METRIC_PARENT_SHA256:
        raise RuntimeError(
            f"metric parent changed: {actual} != {EXPECTED_METRIC_PARENT_SHA256}"
        )


def average_ranks(values: list[float], tolerance: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    position = 0
    while position < len(array):
        end = position + 1
        anchor = float(array[order[position]])
        while (
            end < len(array)
            and abs(float(array[order[end]]) - anchor) <= tolerance
        ):
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    return ranks


def validate_f0_artifacts(root: Path) -> dict[str, object]:
    checks: dict[str, object] = {
        "train_source_sha256": sha256(TRAIN_SOURCE),
        "cache_source_sha256": sha256(CACHE_SOURCE),
        "d570_sha256": sha256(DEFAULT_D570),
        "posterior_cache_sha256": [
            sha256(root / f"protected_posterior_volume_fold{fold}.npz")
            for fold in range(4)
        ],
        "variants": {},
    }
    if (
        checks["train_source_sha256"] != EXPECTED_TRAIN_SOURCE_SHA256
        or checks["cache_source_sha256"] != EXPECTED_CACHE_SOURCE_SHA256
        or checks["d570_sha256"] != EXPECTED_D570_SHA256
        or tuple(checks["posterior_cache_sha256"]) != EXPECTED_CACHE_HASHES
    ):
        raise RuntimeError("protected-posterior global artifact drift")
    for variant in VARIANTS:
        directory = root / f"protected_{variant}_f0"
        result_path = directory / "result.json"
        result_hash = sha256(result_path)
        result = json.loads(result_path.read_text())
        if (
            result_hash != EXPECTED_F0_RESULT_HASHES[variant]
            or result.get("variant") != variant
            or result.get("fold") != 0
            or result.get("source_sha256") != EXPECTED_TRAIN_SOURCE_SHA256
            or result.get("training_folds") != [1, 2, 3]
            or result.get("audit_fold_loaded")
            or result.get("confirmation_regroupings_loaded")
        ):
            raise RuntimeError(f"{variant}: frozen F0 result drift")
        snapshots = {
            int(record["epoch"]): record for record in result["snapshots"]
        }
        variant_checks = []
        for index, snapshot in enumerate(SNAPSHOTS):
            record = snapshots[snapshot]
            prediction = directory / f"epoch{snapshot:03d}_prediction.npz"
            checkpoint = directory / f"epoch{snapshot:03d}.pt"
            prediction_hash = sha256(prediction)
            checkpoint_hash = sha256(checkpoint)
            if (
                prediction_hash
                != EXPECTED_F0_PREDICTION_HASHES[variant][index]
                or checkpoint_hash
                != EXPECTED_F0_CHECKPOINT_HASHES[variant][index]
                or record["prediction_sha256"] != prediction_hash
                or record["checkpoint_sha256"] != checkpoint_hash
            ):
                raise RuntimeError(
                    f"{variant} epoch {snapshot}: frozen F0 artifact drift"
                )
            with np.load(prediction, allow_pickle=False) as cache:
                if (
                    int(cache["fold"]) != 0
                    or str(cache["variant"]) != variant
                    or str(cache["source_sha256"])
                    != EXPECTED_TRAIN_SOURCE_SHA256
                    or str(cache["checkpoint_sha256"]) != checkpoint_hash
                    or bool(cache["audit_fold_loaded"])
                    or bool(cache["confirmation_regroupings_loaded"])
                ):
                    raise RuntimeError(
                        f"{variant} epoch {snapshot}: F0 firewall failed"
                    )
            variant_checks.append(
                {
                    "snapshot": snapshot,
                    "prediction_sha256": prediction_hash,
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
        checks["variants"][variant] = {
            "result_sha256": result_hash,
            "snapshots": variant_checks,
        }
    return checks


def validate_expansion_manifest(
    root: Path, manifest_path: Path
) -> tuple[dict[str, object], str]:
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    regenerated = expansion.freeze(root)
    if manifest != regenerated:
        raise RuntimeError("protected-posterior expansion manifest/file drift")
    if (
        manifest.get("status") != "passed"
        or manifest.get("job_count") != 9
        or manifest.get("prediction_count") != 54
        or manifest.get("checkpoint_count") != 54
        or manifest.get("train_source_sha256")
        != EXPECTED_TRAIN_SOURCE_SHA256
        or manifest.get("target_metric_used_for_manifest", True)
        or manifest.get("target_metric_emitted", True)
        or manifest.get("f4_loaded", True)
        or manifest.get("audit_fold_loaded")
        or manifest.get("confirmation_regroupings_loaded")
    ):
        raise RuntimeError("protected-posterior expansion manifest failed")
    return manifest, manifest_hash


def validate_lineage(
    *,
    fold: int,
    result: dict[str, object],
    cache_source_sha256: str,
    cache_sha256: str,
) -> bool:
    return bool(
        result.get("run_id") == RUN_ID
        and result.get("family") == FAMILY
        and int(result.get("fold", -1)) == fold
        and result.get("status") == "complete_fixed_snapshots"
        and result.get("source_sha256") == EXPECTED_TRAIN_SOURCE_SHA256
        and result.get("training_folds")
        == [candidate for candidate in range(4) if candidate != fold]
        and not result.get("audit_fold_loaded", True)
        and not result.get("confirmation_regroupings_loaded", True)
        and cache_source_sha256 == EXPECTED_CACHE_SOURCE_SHA256
        and cache_sha256 == EXPECTED_CACHE_HASHES[fold]
    )


def fold_non_regressive(
    parent: np.ndarray,
    truth: np.ndarray,
    correction: np.ndarray,
    scale: float,
) -> bool:
    return bool(
        rmse(parent + scale * correction, truth)
        <= rmse(parent, truth) + 1e-12
    )


def sufficient_non_regressive(
    sufficient: dict[str, np.ndarray | float | int],
    parent_rmse: float,
    scale: float,
) -> bool:
    return bool(
        blend_metrics(sufficient, scale)["blend_rmse"]
        <= parent_rmse + 1e-12
    )


def combine_sufficient_statistics(
    values: list[dict[str, np.ndarray | float | int]],
) -> dict[str, np.ndarray | float | int]:
    return {
        "pooled": sum(
            (np.asarray(value["pooled"]) for value in values),
            np.zeros(3, dtype=np.float64),
        ),
        "toe": sum(
            (np.asarray(value["toe"]) for value in values),
            np.zeros(3, dtype=np.float64),
        ),
        "endpoint": sum(
            (np.asarray(value["endpoint"]) for value in values),
            np.zeros(3, dtype=np.float64),
        ),
        "well_terms": np.concatenate(
            [np.asarray(value["well_terms"]) for value in values]
        ),
        "rows": sum(int(value["rows"]) for value in values),
        "toe_rows": sum(int(value["toe_rows"]) for value in values),
        "wells": sum(int(value["wells"]) for value in values),
    }


def build_sufficient_cache(
    folds: list[dict[str, object]],
    parent_name: str,
) -> dict[
    tuple[int, str, int, float],
    dict[str, np.ndarray | float | int],
]:
    output = {}
    for fold, record in enumerate(folds):
        parent = np.asarray(record[parent_name])
        truth = np.asarray(record["truth"])
        starts = np.asarray(record["row_starts"])
        for treatment in TREATMENTS:
            members = treatment_members(treatment)
            for snapshot in SNAPSHOTS:
                for temperature in TEMPERATURES:
                    correction = average_corrections(
                        [
                            np.asarray(
                                record["corrections"][
                                    (variant, snapshot, temperature)
                                ]
                            )
                            for variant in members
                        ]
                    )
                    output[(fold, treatment, snapshot, temperature)] = (
                        metric_sufficient_statistics(
                            [parent],
                            [truth],
                            [correction],
                            [starts],
                        )
                    )
    return output


def select_recipe(
    folds: list[dict[str, object]],
    selector_folds: list[int],
    parent_name: str,
    sufficient_cache: dict[
        tuple[int, str, int, float],
        dict[str, np.ndarray | float | int],
    ]
    | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if sufficient_cache is None:
        sufficient_cache = build_sufficient_cache(folds, parent_name)
    records = []
    for treatment in TREATMENTS:
        for snapshot in SNAPSHOTS:
            for temperature in TEMPERATURES:
                fold_sufficient = [
                    sufficient_cache[
                        (fold, treatment, snapshot, temperature)
                    ]
                    for fold in selector_folds
                ]
                sufficient = combine_sufficient_statistics(fold_sufficient)
                parent_rmse = [
                    blend_metrics(local, 0.0)["blend_rmse"]
                    for local in fold_sufficient
                ]
                for amplitude in AMPLITUDES:
                    for share in SHARES:
                        scale = amplitude * share
                        stable = [
                            sufficient_non_regressive(
                                local,
                                local_parent_rmse,
                                scale,
                            )
                            for local, local_parent_rmse in zip(
                                fold_sufficient,
                                parent_rmse,
                            )
                        ]
                        records.append(
                            {
                                "treatment": treatment,
                                "snapshot": snapshot,
                                "temperature": temperature,
                                "amplitude": amplitude,
                                "share": share,
                                "effective_scale": scale,
                                "selector_fold_non_regressive": stable,
                                "eligible": bool(all(stable)),
                                **blend_metrics(sufficient, scale),
                            }
                        )
    if len(records) != (
        len(TREATMENTS)
        * len(SNAPSHOTS)
        * len(TEMPERATURES)
        * len(AMPLITUDES)
        * len(SHARES)
    ):
        raise RuntimeError("posterior candidate enumeration incomplete")
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise RuntimeError("zero posterior fallback unexpectedly ineligible")
    metric_names = (
        "blend_rmse",
        "mean_well_rmse",
        "toe_quartile_rmse",
        "endpoint_rmse",
    )
    ranks = np.column_stack(
        [
            average_ranks([record[name] for record in eligible])
            for name in metric_names
        ]
    )
    for record, values in zip(eligible, ranks):
        record["metric_ranks"] = {
            name: float(value)
            for name, value in zip(metric_names, values.tolist())
        }
        record["mean_rank"] = float(np.mean(values))
    selected = min(
        eligible,
        key=lambda record: (
            record["mean_rank"],
            record["share"],
            record["amplitude"],
            record["effective_scale"],
            len(treatment_members(str(record["treatment"]))),
            TREATMENT_ORDER[str(record["treatment"])],
            record["snapshot"],
            TEMPERATURES.index(record["temperature"]),
        ),
    )
    return selected, records


def load_fold(root: Path, fold: int) -> dict[str, object]:
    well_ids = row_starts = baseline = d570 = truth = None
    corrections: dict[tuple[str, int, float], np.ndarray] = {}
    hashes = {}
    cache_path = root / f"protected_posterior_volume_fold{fold}.npz"
    cache_hash = sha256(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        cache_source = str(cache["source_sha256"])
        cache_ids = cache["well_ids"].astype(str)
        if (
            int(cache["fold"]) != fold
            or bool(cache["audit_fold_loaded"])
            or bool(cache["confirmation_regroupings_loaded"])
            or bool(cache["stratified_checkpoint_loaded"])
            or not bool(cache["hidden_targets_masked_during_inference"])
        ):
            raise RuntimeError(f"{cache_path}: posterior cache firewall failed")
    for variant in VARIANTS:
        directory = root / f"protected_{variant}_f{fold}"
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("variant") != variant or not validate_lineage(
            fold=fold,
            result=result,
            cache_source_sha256=cache_source,
            cache_sha256=cache_hash,
        ):
            raise RuntimeError(f"{result_path}: posterior lineage failed")
        snapshots = {int(record["epoch"]): record for record in result["snapshots"]}
        if set(snapshots) != set(SNAPSHOTS):
            raise RuntimeError(f"{result_path}: snapshots changed")
        for snapshot in SNAPSHOTS:
            path = directory / f"epoch{snapshot:03d}_prediction.npz"
            record = snapshots[snapshot]
            if sha256(path) != record["prediction_sha256"]:
                raise RuntimeError(f"{path}: hash changed")
            with np.load(path, allow_pickle=False) as cache:
                if (
                    int(cache["fold"]) != fold
                    or str(cache["variant"]) != variant
                    or str(cache["source_sha256"])
                    != EXPECTED_TRAIN_SOURCE_SHA256
                    or bool(cache["audit_fold_loaded"])
                    or bool(cache["confirmation_regroupings_loaded"])
                ):
                    raise RuntimeError(f"{path}: prediction firewall failed")
                local_ids = cache["well_ids"].astype(str)
                local_starts = cache["row_starts"].astype(np.int64)
                local_baseline = cache["baseline"].astype(np.float32)
                local_d570 = cache["d570"].astype(np.float32)
                local_truth = cache["truth"].astype(np.float32)
                local_corrections = {
                    temperature: cache[
                        f"correction_t{int(round(temperature * 100)):03d}"
                    ].astype(np.float32)
                    for temperature in TEMPERATURES
                }
            if not np.array_equal(local_ids, cache_ids):
                raise RuntimeError(f"{path}: cache IDs changed")
            if well_ids is None:
                well_ids = local_ids
                row_starts = local_starts
                baseline = local_baseline
                d570 = local_d570
                truth = local_truth
            elif not (
                np.array_equal(well_ids, local_ids)
                and np.array_equal(row_starts, local_starts)
                and np.array_equal(baseline, local_baseline)
                and np.array_equal(d570, local_d570)
                and np.array_equal(truth, local_truth)
            ):
                raise RuntimeError(f"{path}: posterior fold layout changed")
            for temperature, correction in local_corrections.items():
                if not np.isfinite(correction).all():
                    raise RuntimeError(f"{path}: nonfinite correction")
                corrections[(variant, snapshot, temperature)] = correction
            hashes[(variant, snapshot)] = sha256(path)
    if any(
        value is None for value in (well_ids, row_starts, baseline, d570, truth)
    ):
        raise RuntimeError(f"F{fold}: no posterior predictions")
    return {
        "well_ids": well_ids,
        "row_starts": row_starts,
        "baseline": baseline,
        "d570": d570,
        "truth": truth,
        "corrections": corrections,
        "hashes": hashes,
        "posterior_cache_sha256": cache_hash,
    }


def verify_d570(folds: list[dict[str, object]], path: Path) -> None:
    if sha256(path) != EXPECTED_D570_SHA256:
        raise RuntimeError("D570 cache changed")
    with np.load(path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["confirmation_regroupings_loaded"]
        ):
            raise RuntimeError("D570 flags changed")
        ids = cache["well_ids"].astype(str)
        assignment = cache["folds"].astype(np.int8)
        starts = cache["row_starts"].astype(np.int64)
        baseline = cache["baseline"].astype(np.float32)
        truth = cache["truth"].astype(np.float32)
        prediction = cache["prediction"].astype(np.float32)
    for fold, record in enumerate(folds):
        indices = np.flatnonzero(assignment == fold)
        lengths = starts[indices + 1] - starts[indices]
        local_starts = np.r_[0, np.cumsum(lengths)].astype(np.int64)

        def concatenate(values: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [values[starts[index] : starts[index + 1]] for index in indices]
            )

        if not (
            np.array_equal(ids[indices], record["well_ids"])
            and np.array_equal(local_starts, record["row_starts"])
            and np.array_equal(concatenate(baseline), record["baseline"])
            and np.array_equal(concatenate(truth), record["truth"])
            and np.array_equal(concatenate(prediction), record["d570"])
        ):
            raise RuntimeError(f"F{fold}: D570 attestation failed")


def synthetic_folds() -> list[dict[str, object]]:
    folds = []
    truth = np.asarray((0.0, 0.2, 0.4, 0.6, 0.0, -0.2, -0.4, -0.6))
    for fold in range(4):
        corrections = {}
        for snapshot in SNAPSHOTS:
            for temperature in TEMPERATURES:
                corrections[("posterior_only_b12", snapshot, temperature)] = (
                    5.0 * truth
                )
                corrections[
                    ("posterior_physical_b12", snapshot, temperature)
                ] = -2.0 * truth
                corrections[
                    ("posterior_physical_b16", snapshot, temperature)
                ] = -truth
        folds.append(
            {
                "well_ids": np.asarray((f"f{fold}a", f"f{fold}b")),
                "row_starts": np.asarray((0, 4, 8), dtype=np.int64),
                "baseline": np.zeros(8),
                "d570": np.zeros(8),
                "truth": truth.copy(),
                "corrections": corrections,
            }
        )
    return folds


def performance_parity_checks(root: Path) -> dict[str, object]:
    path = (
        root
        / "protected_posterior_only_b12_f0"
        / "epoch004_prediction.npz"
    )
    with np.load(path, allow_pickle=False) as cache:
        starts = cache["row_starts"].astype(np.int64)[:6]
        right = int(starts[-1])
        truth = cache["truth"].astype(np.float64)[:right]
        correction = cache["correction_t200"].astype(np.float64)[:right]
        parents = (
            cache["baseline"].astype(np.float64)[:right],
            cache["d570"].astype(np.float64)[:right],
        )
    scales = sorted(
        {
            float(amplitude * share)
            for amplitude in AMPLITUDES
            for share in SHARES
        }
    )
    maximum_difference = 0.0
    eligibility_equal = True
    for parent in parents:
        sufficient = metric_sufficient_statistics(
            [parent], [truth], [correction], [starts]
        )
        direct_parent = rmse(parent, truth)
        sufficient_parent = blend_metrics(sufficient, 0.0)["blend_rmse"]
        maximum_difference = max(
            maximum_difference,
            abs(direct_parent - sufficient_parent),
        )
        for scale in scales:
            direct_score = rmse(parent + scale * correction, truth)
            sufficient_score = blend_metrics(sufficient, scale)["blend_rmse"]
            maximum_difference = max(
                maximum_difference,
                abs(direct_score - sufficient_score),
            )
            eligibility_equal &= (
                direct_score <= direct_parent + 1e-12
            ) == sufficient_non_regressive(
                sufficient, sufficient_parent, scale
            )
    return {
        "maximum_rmse_difference": maximum_difference,
        "eligibility_equal_at_every_scale": eligibility_equal,
        "scale_count": len(scales),
        "well_count": len(starts) - 1,
    }


def implementation_gate(
    root: Path = DEFAULT_ROOT,
    expansion_manifest_path: Path = DEFAULT_EXPANSION_MANIFEST,
) -> dict[str, object]:
    assert_metric_parent()
    real_artifacts = validate_f0_artifacts(root)
    parity = performance_parity_checks(root)
    expansion_manifest, expansion_manifest_hash = validate_expansion_manifest(
        root, expansion_manifest_path
    )
    folds = synthetic_folds()
    held_exclusion = True
    planted = True
    candidate_count = None
    first_records = None
    for held in range(4):
        selector = [fold for fold in range(4) if fold != held]
        selected, records = select_recipe(folds, selector, "d570")
        if held == 0:
            first_records = records
        candidate_count = len(records)
        altered = [
            {**record, "truth": np.asarray(record["truth"]).copy()}
            for record in folds
        ]
        altered[held]["truth"][:] = 1000.0
        selected_after, _ = select_recipe(altered, selector, "d570")
        identity = (
            selected["treatment"],
            selected["snapshot"],
            selected["temperature"],
            selected["amplitude"],
            selected["share"],
        )
        identity_after = (
            selected_after["treatment"],
            selected_after["snapshot"],
            selected_after["temperature"],
            selected_after["amplitude"],
            selected_after["share"],
        )
        held_exclusion &= identity == identity_after
        planted &= (
            selected["treatment"] == "posterior_only_b12"
            and abs(float(selected["effective_scale"]) - 0.2) < 1e-12
        )
    _, repeated_records = select_recipe(folds, [1, 2, 3], "d570")
    enumeration_fields = (
        "treatment",
        "snapshot",
        "temperature",
        "amplitude",
        "share",
        "effective_scale",
        "eligible",
    )
    deterministic_enumeration = [
        tuple(record[field] for field in enumeration_fields)
        for record in first_records
    ] == [
        tuple(record[field] for field in enumeration_fields)
        for record in repeated_records
    ]
    zero_folds = synthetic_folds()
    for record in zero_folds:
        record["truth"] = np.asarray(record["d570"]).copy()
    zero_selected, _ = select_recipe(zero_folds, [1, 2, 3], "d570")
    zero_identity = (
        zero_selected["treatment"],
        zero_selected["snapshot"],
        zero_selected["temperature"],
        zero_selected["amplitude"],
        zero_selected["share"],
    )

    source = np.asarray((1.0, 3.0, 5.0, 7.0))
    p12 = np.asarray((3.0, 5.0, 7.0, 9.0))
    p16 = np.asarray((5.0, 7.0, 9.0, 11.0))
    arithmetic = {
        treatment: average_corrections(
            [
                {
                    "posterior_only_b12": source,
                    "posterior_physical_b12": p12,
                    "posterior_physical_b16": p16,
                }[member]
                for member in treatment_members(treatment)
            ]
        )
        for treatment in TREATMENTS[3:]
    }
    parent_parts = [np.zeros(4) for _ in range(3)]
    truth_parts = [np.full(4, 10.0), np.full(4, 10.0), np.full(4, 1.0)]
    distractor = [np.full(4, 10.0), np.full(4, 10.0), np.full(4, -1.0)]
    stable = [
        fold_non_regressive(parent, truth, correction, 1.0)
        for parent, truth, correction in zip(
            parent_parts, truth_parts, distractor
        )
    ]
    pooled_attractive = rmse(
        np.concatenate(distractor), np.concatenate(truth_parts)
    ) < rmse(np.concatenate(parent_parts), np.concatenate(truth_parts))
    good_lineage = {
        "run_id": RUN_ID,
        "family": FAMILY,
        "fold": 2,
        "status": "complete_fixed_snapshots",
        "source_sha256": EXPECTED_TRAIN_SOURCE_SHA256,
        "training_folds": [0, 1, 3],
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    checks = {
        "pair_only_physical12_exact": np.array_equal(
            arithmetic["equal_only_physical12"], (source + p12) / 2.0
        ),
        "pair_only_physical16_exact": np.array_equal(
            arithmetic["equal_only_physical16"], (source + p16) / 2.0
        ),
        "pair_physical_exact": np.array_equal(
            arithmetic["equal_physical12_physical16"], (p12 + p16) / 2.0
        ),
        "all_three_exact": np.array_equal(
            arithmetic["equal_all"], (source + p12 + p16) / 3.0
        ),
        "candidate_count_exact": candidate_count == 7560,
        "candidate_enumeration_deterministic": deterministic_enumeration,
        "held_exclusion_exact": held_exclusion,
        "planted_stable_route_recovered": planted,
        "pooled_attractive_fold_regressor_constructed": (
            pooled_attractive and stable == [True, True, False]
        ),
        "fold_regressive_distractor_rejected": not all(stable),
        "zero_fallback_tie_exact": zero_identity
        == ("posterior_only_b12", 4, 0.75, 1.0, 0.0),
        "rank_tolerance_exact": np.array_equal(
            average_ranks([1.0, 1.0 + 5e-13, 2.0]),
            np.asarray((1.5, 1.5, 3.0)),
        ),
        "v20_zero_share_exact": np.array_equal(source + 0.0 * p12, source),
        "d570_zero_share_exact": np.array_equal(p12 + 0.0 * p16, p12),
        "lineage_accepts_exact": validate_lineage(
            fold=2,
            result=good_lineage,
            cache_source_sha256=EXPECTED_CACHE_SOURCE_SHA256,
            cache_sha256=EXPECTED_CACHE_HASHES[2],
        ),
        "lineage_rejects_wrong_fold": not validate_lineage(
            fold=1,
            result=good_lineage,
            cache_source_sha256=EXPECTED_CACHE_SOURCE_SHA256,
            cache_sha256=EXPECTED_CACHE_HASHES[2],
        ),
        "lineage_rejects_wrong_source": not validate_lineage(
            fold=2,
            result={**good_lineage, "source_sha256": "wrong"},
            cache_source_sha256=EXPECTED_CACHE_SOURCE_SHA256,
            cache_sha256=EXPECTED_CACHE_HASHES[2],
        ),
        "lineage_rejects_wrong_cache_source": not validate_lineage(
            fold=2,
            result=good_lineage,
            cache_source_sha256="wrong",
            cache_sha256=EXPECTED_CACHE_HASHES[2],
        ),
        "lineage_rejects_wrong_cache_hash": not validate_lineage(
            fold=2,
            result=good_lineage,
            cache_source_sha256=EXPECTED_CACHE_SOURCE_SHA256,
            cache_sha256="wrong",
        ),
        "real_train_source_pinned": (
            real_artifacts["train_source_sha256"]
            == EXPECTED_TRAIN_SOURCE_SHA256
        ),
        "real_cache_source_pinned": (
            real_artifacts["cache_source_sha256"]
            == EXPECTED_CACHE_SOURCE_SHA256
        ),
        "real_posterior_caches_pinned": tuple(
            real_artifacts["posterior_cache_sha256"]
        )
        == EXPECTED_CACHE_HASHES,
        "real_d570_pinned": (
            real_artifacts["d570_sha256"] == EXPECTED_D570_SHA256
        ),
        "all_f0_results_predictions_checkpoints_pinned": (
            set(real_artifacts["variants"]) == set(VARIANTS)
            and all(
                len(real_artifacts["variants"][variant]["snapshots"])
                == len(SNAPSHOTS)
                for variant in VARIANTS
            )
        ),
        "cached_direct_rmse_parity": (
            float(parity["maximum_rmse_difference"]) <= 1e-12
        ),
        "cached_direct_eligibility_parity": bool(
            parity["eligibility_equal_at_every_scale"]
        ),
        "complete_expansion_manifest_pinned": (
            expansion_manifest["job_count"] == 9
            and expansion_manifest["prediction_count"] == 54
            and expansion_manifest["checkpoint_count"] == 54
            and len(expansion_manifest["jobs"]) == 9
        ),
        "expansion_manifest_rejects_wrong_hash": (
            expansion_manifest_hash != "0" * 64
            and expansion_manifest_hash == sha256(expansion_manifest_path)
        ),
        "expansion_manifest_rejects_wrong_file_hash": (
            expansion_manifest["jobs"][0]["result_sha256"] != "0" * 64
            and expansion_manifest["jobs"][0]["snapshots"][0][
                "checkpoint_sha256"
            ]
            != "0" * 64
            and expansion_manifest["jobs"][0]["snapshots"][0]["prediction"][
                "sha256"
            ]
            != "0" * 64
        ),
        "finite": all(np.isfinite(value).all() for value in arithmetic.values()),
        "stable_layout": all(
            record["row_starts"][0] == 0
            and record["row_starts"][-1] == len(record["truth"])
            for record in folds
        ),
    }
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": "protected_posterior_clean_selector_gate",
        "stage": "metric_free_selector_implementation_gate",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "candidate_count": candidate_count,
        "performance_parity": parity,
        "real_f0_artifacts": real_artifacts,
        "expansion_manifest": str(expansion_manifest_path),
        "expansion_manifest_sha256": expansion_manifest_hash,
        "expansion_manifest_source_sha256": expansion_manifest["source_sha256"],
        "metric_parent_sha256": EXPECTED_METRIC_PARENT_SHA256,
        "train_source_sha256": EXPECTED_TRAIN_SOURCE_SHA256,
        "cache_source_sha256": EXPECTED_CACHE_SOURCE_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "target_metric_computed": False,
        "f4_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }


def evaluate(
    root: Path,
    d570_path: Path,
    gate_path: Path,
    expansion_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    assert_metric_parent()
    source_hash = sha256(Path(__file__))
    expansion_manifest, expansion_manifest_hash = validate_expansion_manifest(
        root, expansion_manifest_path
    )
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("status") != "passed"
        or gate.get("source_sha256") != source_hash
        or gate.get("metric_parent_sha256") != EXPECTED_METRIC_PARENT_SHA256
        or gate.get("expansion_manifest")
        != str(expansion_manifest_path)
        or gate.get("expansion_manifest_sha256")
        != expansion_manifest_hash
        or gate.get("expansion_manifest_source_sha256")
        != expansion_manifest["source_sha256"]
        or gate.get("target_metric_computed", True)
        or gate.get("f4_loaded", True)
        or gate.get("audit_fold_loaded")
        or gate.get("confirmation_regroupings_loaded")
    ):
        raise RuntimeError("posterior selector gate/source mismatch")
    folds = [load_fold(root, fold) for fold in range(4)]
    verify_d570(folds, d570_path)
    modes = {"replacement": "baseline", "incremental_after_D570": "d570"}
    mode_results = {}
    predictions = {}
    for mode, parent_name in modes.items():
        selected_parts = []
        records = []
        sufficient_cache = build_sufficient_cache(folds, parent_name)
        for held in range(4):
            selector = [fold for fold in range(4) if fold != held]
            selected, grid = select_recipe(
                folds,
                selector,
                parent_name,
                sufficient_cache=sufficient_cache,
            )
            members = treatment_members(str(selected["treatment"]))
            correction = average_corrections(
                [
                    np.asarray(
                        folds[held]["corrections"][
                            (
                                variant,
                                int(selected["snapshot"]),
                                float(selected["temperature"]),
                            )
                        ]
                    )
                    for variant in members
                ]
            )
            parent_values = np.asarray(folds[held][parent_name])
            truth = np.asarray(folds[held]["truth"])
            prediction = (
                parent_values.astype(np.float64)
                + float(selected["effective_scale"]) * correction
            )
            parent_score = rmse(parent_values, truth)
            score = rmse(prediction, truth)
            selected_parts.append(prediction.astype(np.float32))
            records.append(
                {
                    "held_fold": held,
                    "selector_folds": selector,
                    "selected": selected,
                    "selector_grid": grid,
                    "held_parent_rmse": parent_score,
                    "held_selected_rmse": score,
                    "held_gain": parent_score - score,
                }
            )
        parent_values = np.concatenate(
            [np.asarray(fold[parent_name]) for fold in folds]
        )
        truth = np.concatenate([np.asarray(fold["truth"]) for fold in folds])
        prediction = np.concatenate(selected_parts)
        parent_score = rmse(parent_values, truth)
        score = rmse(prediction, truth)
        gains = [float(record["held_gain"]) for record in records]
        mode_results[mode] = {
            "parent_rmse": parent_score,
            "selected_rmse": score,
            "gain": parent_score - score,
            "positive_on_every_held_fold": all(gain > 0.0 for gain in gains),
            "fold_records": records,
        }
        predictions[mode] = prediction.astype(np.float32)
    incremental = mode_results["incremental_after_D570"]
    retained = bool(
        incremental["gain"] > 0.0
        and incremental["positive_on_every_held_fold"]
    )
    baseline = np.concatenate([np.asarray(fold["baseline"]) for fold in folds])
    d570 = np.concatenate([np.asarray(fold["d570"]) for fold in folds])
    truth = np.concatenate([np.asarray(fold["truth"]) for fold in folds])
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": FAMILY,
        "variant": "selection_clean_protected_posterior_expansion",
        "stage": "selection_clean_F0_F3",
        "status": "accepted" if retained else "rejected",
        "retained_one_component_slot": retained,
        "treatments": list(TREATMENTS),
        "snapshots": list(SNAPSHOTS),
        "temperatures": list(TEMPERATURES),
        "amplitudes": list(AMPLITUDES),
        "shares": list(SHARES),
        "selection_rule": (
            "other-three roles; per-role D570 non-regression; mean rank of "
            "pooled, mean-well, toe-quartile, and endpoint RMSE"
        ),
        "modes": mode_results,
        "source_sha256": source_hash,
        "gate_json": str(gate_path),
        "gate_sha256": sha256(gate_path),
        "d570_path": str(d570_path),
        "d570_sha256": sha256(d570_path),
        "posterior_cache_sha256": list(EXPECTED_CACHE_HASHES),
        "expansion_manifest": str(expansion_manifest_path),
        "expansion_manifest_sha256": expansion_manifest_hash,
        "expansion_manifest_source_sha256": expansion_manifest["source_sha256"],
        "f4_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    cache = {
        "well_ids": np.concatenate(
            [np.asarray(fold["well_ids"]) for fold in folds]
        ),
        "folds": np.concatenate(
            [
                np.full(len(np.asarray(fold["well_ids"])), index, dtype=np.int8)
                for index, fold in enumerate(folds)
            ]
        ),
        "row_starts": np.r_[
            0,
            np.cumsum(
                np.concatenate(
                    [np.diff(np.asarray(fold["row_starts"])) for fold in folds]
                )
            ),
        ].astype(np.int64),
        "baseline": baseline.astype(np.float32),
        "d570": d570.astype(np.float32),
        "truth": truth.astype(np.float32),
        "replacement": predictions["replacement"],
        "incremental_after_D570": predictions["incremental_after_D570"],
        "f4_loaded": np.asarray(False),
        "audit_fold_loaded": np.asarray(False),
        "confirmation_regroupings_loaded": np.asarray(False),
    }
    return result, cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--d570", type=Path, default=DEFAULT_D570)
    parser.add_argument(
        "--expansion-manifest",
        type=Path,
        default=DEFAULT_EXPANSION_MANIFEST,
    )
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    if args.gate:
        if args.gate_json is not None or args.output_npz is not None:
            raise ValueError("gate does not accept result gate/NPZ outputs")
        payload = implementation_gate(args.root, args.expansion_manifest)
        write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] != "passed":
            raise RuntimeError("posterior selector gate failed")
        return
    if args.gate_json is None or args.output_npz is None:
        raise ValueError("evaluation requires --gate-json and --output-npz")
    if args.output_npz.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_npz}")
    result, cache = evaluate(
        args.root,
        args.d570,
        args.gate_json,
        args.expansion_manifest,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **cache)
    result["prediction"] = str(args.output_npz)
    result["prediction_sha256"] = sha256(args.output_npz)
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
