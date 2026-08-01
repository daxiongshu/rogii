from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from cache_alignment_oof_predictions import COMPONENTS, indexed_prediction
from limited_query_cv import (
    cache_v5_run6_protected_posterior_volume as posterior,
)
from limited_query_cv import (
    build_v5_run25_original_safe_mode_bank as safe_mode,
)
from limited_query_cv import (
    evaluate_v5_run6_protected_mode_bank as mode_parent,
)
from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


RUN_ID = "v5_batch2_run_025_goal_050"
ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (
    ROOT
    / "limited_query_cv/runs/v5_batch2_run_025_goal_050/promotion_audit"
)
CACHE_ROOT = AUDIT_ROOT / "caches"
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
V20 = Path("/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz")
CHECKPOINT_ROOT = Path("/geo/geo_new/rogii_v20_checkpoint_vault")
COMPONENT_OUTPUT = CACHE_ROOT / "v20_components_fold4.npz"
POSTERIOR_OUTPUT = CACHE_ROOT / "protected_posterior_volume_fold4.npz"
MODE_OUTPUT = CACHE_ROOT / "original_safe_mode_bank_fold4.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected(
    identifiers: np.ndarray,
    selected: np.ndarray,
    data_root: Path,
    workers: int,
) -> list[Well]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda item: load_well(item, data_root),
                identifiers[selected].tolist(),
            )
        )


def build_components(
    data_root: Path,
    device_name: str,
    workers: int,
) -> dict[str, object]:
    if COMPONENT_OUTPUT.exists():
        raise FileExistsError(COMPONENT_OUTPUT)
    started = time.time()
    with np.load(LAYOUT, allow_pickle=False) as layout:
        identifiers = layout["well_ids"].astype(str)
        row_starts = layout["row_starts"].astype(np.int64)
        folds = layout["original"].astype(np.int64)
    if set(identifiers) != set(training_well_ids(data_root)):
        raise RuntimeError("layout and training IDs differ")
    selected = np.flatnonzero(folds == 4)
    wells = load_selected(identifiers, selected, data_root, workers)
    with np.load(V20, allow_pickle=False) as cache:
        if (
            not np.array_equal(identifiers, cache["well_ids"].astype(str))
            or not np.array_equal(
                row_starts, cache["row_starts"].astype(np.int64)
            )
        ):
            raise RuntimeError("v20/layout binding changed")
        protected_prediction = cache["prediction"].astype(np.float32)

    common = dict(
        prefix_points=64,
        tail_points=640,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=51,
    )
    grids = {
        "narrow": SequenceConfig(
            state_radius=64.0, state_step=0.5, **common
        ),
        "wide": SequenceConfig(
            state_radius=120.0, state_step=1.0, **common
        ),
    }
    device = torch.device(device_name)
    models = []
    checkpoint_hashes = {}
    for name, pattern, scale, temperature, weight in COMPONENTS:
        checkpoint = CHECKPOINT_ROOT / pattern.format(fold=4)
        model = AlignmentUNet(base=8).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        checkpoint_hashes[name] = sha256(checkpoint)
        models.append((name, model, scale, temperature, weight))

    well_ids: list[str] = []
    local_starts = [0]
    component_parts = []
    baseline_parts = []
    with torch.no_grad():
        for count, (global_index, well) in enumerate(
            zip(selected, wells), start=1
        ):
            examples = {
                key: make_alignment_example(
                    well, well.anchor_index, config
                )
                for key, config in grids.items()
            }
            values = []
            for _, model, scale, temperature, _ in models:
                image, _, metadata = examples[scale]
                values.append(
                    indexed_prediction(
                        model,
                        image,
                        metadata,
                        well,
                        grids[scale],
                        temperature,
                        device,
                        well.tail_indices,
                    )
                )
            left, right = row_starts[global_index : global_index + 2]
            length = len(well.prediction_indices)
            if length != right - left:
                raise RuntimeError(
                    f"{well.well_id}: protected layout mismatch"
                )
            component_parts.append(np.stack(values).astype(np.float32))
            baseline_parts.append(protected_prediction[left:right])
            well_ids.append(well.well_id)
            local_starts.append(local_starts[-1] + length)
            if count % 20 == 0 or count == len(wells):
                print(
                    json.dumps(
                        {
                            "stage": "component_cache",
                            "fold": 4,
                            "wells": count,
                            "total_wells": len(wells),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        COMPONENT_OUTPUT,
        fold=np.asarray(4, dtype=np.int8),
        well_ids=np.asarray(well_ids),
        row_starts=np.asarray(local_starts, dtype=np.int64),
        components=np.concatenate(component_parts, axis=1),
        baseline=np.concatenate(baseline_parts).astype(np.float32),
        component_names=np.asarray([item[0] for item in COMPONENTS]),
        component_weights=np.asarray(
            [item[4] for item in COMPONENTS], dtype=np.float64
        ),
        checkpoint_names=np.asarray(
            [item[1].format(fold=4) for item in COMPONENTS]
        ),
        checkpoint_sha256=np.asarray(
            [checkpoint_hashes[item[0]] for item in COMPONENTS]
        ),
        hidden_truth_stored=np.asarray(False),
        protected_f4_cache=np.asarray(True),
        audit_fold_loaded=np.asarray(False),
        layout_truth_array_accessed=np.asarray(False),
        source_sha256=np.asarray(sha256(Path(__file__))),
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "protected_f4_component_cache",
        "status": "complete_metric_suppressed",
        "output": str(COMPONENT_OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(COMPONENT_OUTPUT),
        "fold": 4,
        "well_count": len(wells),
        "row_count": int(local_starts[-1]),
        "component_count": len(COMPONENTS),
        "hidden_truth_stored": False,
        "metric_computed": False,
        "elapsed_seconds": time.time() - started,
        "source_sha256": sha256(Path(__file__)),
    }


def build_posterior(
    data_root: Path,
    device_name: str,
    workers: int,
) -> dict[str, object]:
    if not COMPONENT_OUTPUT.exists():
        raise FileNotFoundError(COMPONENT_OUTPUT)
    if POSTERIOR_OUTPUT.exists():
        raise FileExistsError(POSTERIOR_OUTPUT)
    original_pattern = posterior.COMPONENT_PATTERN
    try:
        posterior.COMPONENT_PATTERN = (
            CACHE_ROOT / "v20_components_fold{fold}.npz"
        )
        args = argparse.Namespace(
            fold=4,
            device=device_name,
            workers=workers,
            data_root=data_root,
            output=POSTERIOR_OUTPUT,
            firewall_only=False,
        )
        posterior.build(args)
    finally:
        posterior.COMPONENT_PATTERN = original_pattern
    with np.load(POSTERIOR_OUTPUT, allow_pickle=False) as cache:
        if (
            int(cache["fold"]) != 4
            or not bool(cache["hidden_targets_masked_during_inference"])
            or bool(cache["stratified_checkpoint_loaded"])
            or any(
                "strat" in name
                for name in cache["checkpoint_names"].astype(str)
            )
            or "truth" in cache.files
        ):
            raise RuntimeError("protected F4 posterior firewall failed")
        shape = list(cache["posterior_probability"].shape)
        wells = len(cache["well_ids"])
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "protected_f4_posterior_cache",
        "status": "complete_metric_suppressed",
        "output": str(POSTERIOR_OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(POSTERIOR_OUTPUT),
        "component_cache_sha256": sha256(COMPONENT_OUTPUT),
        "fold": 4,
        "well_count": wells,
        "posterior_shape": shape,
        "parent_count": 15,
        "hidden_targets_masked": True,
        "hidden_truth_stored": False,
        "stratified_parent_loaded": False,
        "metric_computed": False,
        "source_sha256": sha256(Path(__file__)),
    }


def build_mode(
    data_root: Path,
    device_name: str,
    workers: int,
) -> dict[str, object]:
    if MODE_OUTPUT.exists():
        raise FileExistsError(MODE_OUTPUT)
    started = time.time()
    with np.load(LAYOUT, allow_pickle=False) as layout:
        identifiers = layout["well_ids"].astype(str)
        starts = layout["row_starts"].astype(np.int64)
        folds = layout["original"].astype(np.int64)
    selected = np.flatnonzero(folds == 4)
    with np.load(V20, allow_pickle=False) as cache:
        if (
            not np.array_equal(identifiers, cache["well_ids"].astype(str))
            or not np.array_equal(
                starts, cache["row_starts"].astype(np.int64)
            )
        ):
            raise RuntimeError("v20/layout binding changed")
        protected_prediction = cache["prediction"].astype(np.float32)
    well_ids = identifiers[selected]
    row_lengths = starts[selected + 1] - starts[selected]
    local_starts = np.r_[0, np.cumsum(row_lengths)].astype(np.int64)
    baseline = np.concatenate(
        [
            protected_prediction[starts[index] : starts[index + 1]]
            for index in selected
        ]
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = list(
            pool.map(
                lambda item: mode_parent.load_inference_well(
                    item, data_root
                ),
                well_ids.tolist(),
            )
        )
    device = torch.device(device_name)
    configurations = mode_parent.configs()
    parents = safe_mode.safe_parents(4, device)
    candidate_names = mode_parent.candidate_schema(parents)
    if (
        len(parents) != 15
        or len(candidate_names) != 240
        or any("strat" in item.name for item in parents)
    ):
        raise RuntimeError("F4 safe mode schema changed")
    correction_parts = []
    metric_parts = []
    shifted_parts = []
    metric_names = None
    for index, well in enumerate(wells):
        left, right = local_starts[index : index + 2]
        correction = mode_parent.candidate_corrections(
            well, parents, configurations, device
        )
        if correction.shape != (240, right - left):
            raise RuntimeError(f"{well.well_id}: mode shape changed")
        paths = baseline[left:right][None].astype(np.float64) + correction
        metrics, names = mode_parent.calibrated_metrics(
            well, paths, shifted_control=False
        )
        shifted, shifted_names = mode_parent.calibrated_metrics(
            well, paths, shifted_control=True
        )
        if names != shifted_names or (
            metric_names is not None and names != metric_names
        ):
            raise RuntimeError("mode metric schema changed")
        metric_names = names
        correction_parts.append(correction.astype(np.float16))
        metric_parts.append(metrics)
        shifted_parts.append(shifted)
        if (index + 1) % 10 == 0 or index + 1 == len(wells):
            print(
                json.dumps(
                    {
                        "stage": "mode_cache",
                        "fold": 4,
                        "wells": index + 1,
                        "total_wells": len(wells),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    assert metric_names is not None
    np.savez_compressed(
        MODE_OUTPUT,
        fold=np.asarray(4, dtype=np.int8),
        well_ids=well_ids,
        row_starts=local_starts,
        baseline=baseline,
        corrections=np.concatenate(correction_parts, axis=1),
        candidate_names=candidate_names,
        metrics=np.stack(metric_parts),
        shifted_control_metrics=np.stack(shifted_parts),
        metric_names=np.asarray(metric_names),
        parent_names=np.asarray([item.name for item in parents]),
        checkpoint_names=np.asarray(
            [str(item.checkpoint) for item in parents]
        ),
        checkpoint_sha256=np.asarray(
            [item.checkpoint_sha256 for item in parents]
        ),
        hidden_truth_stored=np.asarray(False),
        protected_f4_cache=np.asarray(True),
        forbidden_stratified_parent_loaded=np.asarray(False),
        source_sha256=np.asarray(sha256(Path(__file__))),
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": "protected_f4_original_safe_mode_cache",
        "status": "complete_metric_suppressed",
        "output": str(MODE_OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(MODE_OUTPUT),
        "fold": 4,
        "well_count": len(wells),
        "row_count": int(local_starts[-1]),
        "parent_count": 15,
        "candidate_count": 240,
        "hidden_truth_stored": False,
        "stratified_parent_loaded": False,
        "metric_computed": False,
        "elapsed_seconds": time.time() - started,
        "source_sha256": sha256(Path(__file__)),
    }


def write_record(name: str, payload: dict[str, object]) -> None:
    path = CACHE_ROOT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "stage": payload["stage"],
                "output_sha256": payload["output_sha256"],
                "record": str(path.relative_to(ROOT)),
                "record_sha256": sha256(path),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("components", "posterior", "mode")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    if args.stage == "components":
        payload = build_components(
            args.data_root, args.device, args.workers
        )
        write_record("v20_components_fold4.json", payload)
    elif args.stage == "posterior":
        payload = build_posterior(
            args.data_root, args.device, args.workers
        )
        write_record("protected_posterior_volume_fold4.json", payload)
    else:
        payload = build_mode(
            args.data_root, args.device, args.workers
        )
        write_record("original_safe_mode_bank_fold4.json", payload)


if __name__ == "__main__":
    main()
