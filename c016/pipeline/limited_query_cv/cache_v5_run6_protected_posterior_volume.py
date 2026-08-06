from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from limited_query_cv.build_v5_run2_protected_view_bank import FAMILIES
from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


RUN_ID = "v5_batch2_run_006_goal_050"
LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
V20 = Path("/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz")
CHECKPOINT_ROOT = Path("/geo/geo_new/rogii_v20_checkpoint_vault")
SOURCE_MANIFEST = CHECKPOINT_ROOT / "source_manifest.json"
COMPONENT_PATTERN = Path(
    "limited_query_cv/runs/v5_batch2_run_003_goal_050/"
    "v20_components/fold{fold}.npz"
)
RESIDUAL_OFFSETS = np.arange(-32.0, 33.0, 1.0, dtype=np.float32)
PROBABILITY_SUM_TOLERANCE = 3e-6


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


def configs() -> dict[str, SequenceConfig]:
    common = dict(
        prefix_points=64,
        tail_points=640,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=51,
    )
    return {
        "narrow": SequenceConfig(
            state_radius=64.0, state_step=0.5, **common
        ),
        "wide": SequenceConfig(
            state_radius=120.0, state_step=1.0, **common
        ),
    }


def inference_masked(well: Well, mode: str = "nan") -> Well:
    anchor = well.anchor_index
    target = well.tvt.copy()
    hidden = np.arange(len(target)) > anchor
    if mode == "original":
        return well
    if mode == "reverse":
        target[hidden] = target[hidden][::-1]
    elif mode == "nan":
        target[hidden] = np.nan
    else:
        raise ValueError(f"unknown hidden-target mode: {mode}")
    # The visible prefix is retained exactly; hidden values are deliberately
    # unavailable to the image builder.
    return replace(well, tvt=target)


def load_models(
    fold: int, device: torch.device
) -> tuple[list[tuple[str, AlignmentUNet, str, float, Path, str]], dict[str, str]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())["sha256"]
    models = []
    hashes: dict[str, str] = {}
    for name, pattern, scale, base, temperature in FAMILIES:
        checkpoint = CHECKPOINT_ROOT / pattern.format(fold=fold)
        actual = sha256(checkpoint)
        if manifest.get(checkpoint.name) != actual:
            raise RuntimeError(f"{checkpoint}: immutable hash mismatch")
        # FAMILIES deliberately excludes the balanced/stratified checkpoint:
        # only numeric original-fold checkpoints have the required lineage.
        if "strat" in checkpoint.name:
            raise RuntimeError("stratified checkpoint entered A610")
        model = AlignmentUNet(base=base).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        models.append((name, model, scale, temperature, checkpoint, actual))
        hashes[checkpoint.name] = actual
    if len(models) != 15:
        raise RuntimeError(f"expected 15 protected parents, got {len(models)}")
    return models, hashes


def load_selected(
    identifiers: np.ndarray,
    selected: np.ndarray,
    data_root: Path,
    workers: int,
) -> list[Well]:
    ids = identifiers[selected].tolist()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: load_well(item, data_root), ids))


def sampled_prior(
    well: Well,
    positions: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    source = np.arange(len(well.tvt_input), dtype=np.float64)
    anchor = well.anchor_index
    visible = np.interp(
        np.minimum(positions, float(anchor)),
        source[: anchor + 1],
        well.tvt_input[: anchor + 1],
    )
    hidden = np.interp(
        positions,
        well.prediction_indices.astype(np.float64),
        baseline.astype(np.float64),
        left=float(well.tvt_input[anchor]),
        right=float(baseline[-1]),
    )
    return np.where(positions <= anchor + 1e-6, visible, hidden)


def residual_probability(
    probability: np.ndarray,
    source_offsets: np.ndarray,
    candidate_offsets: np.ndarray,
) -> np.ndarray:
    """Linearly sample a native state posterior on the V20 residual grid."""
    step = float(source_offsets[1] - source_offsets[0])
    coordinate = (candidate_offsets - float(source_offsets[0])) / step
    lower = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - lower
    supported = (lower >= 0) & (lower + 1 < len(source_offsets))
    lower = np.clip(lower, 0, len(source_offsets) - 2)
    # Native probability is [state, column]; gather on [column, residual].
    transposed = probability.T
    left = np.take_along_axis(transposed, lower, axis=1)
    right = np.take_along_axis(transposed, lower + 1, axis=1)
    sampled = left * (1.0 - fraction) + right * fraction
    sampled *= supported
    return sampled.T.astype(np.float32)


@torch.no_grad()
def make_volume(
    well: Well,
    baseline: np.ndarray,
    models: list[tuple[str, AlignmentUNet, str, float, Path, str]],
    grids: dict[str, SequenceConfig],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    masked = inference_masked(well, "nan")
    examples = {
        key: make_alignment_example(masked, masked.anchor_index, grid)
        for key, grid in grids.items()
    }
    reference_metadata = examples["wide"][2]
    positions = np.asarray(reference_metadata["positions"], dtype=np.float64)
    prior_tvt = sampled_prior(well, positions, baseline)
    candidate_offsets = (
        prior_tvt[:, None]
        - float(well.tvt_input[well.anchor_index])
        + RESIDUAL_OFFSETS[None]
    )
    planes = []
    native_sum_errors = []
    for _, model, scale, temperature, _, _ in models:
        image, _, metadata = examples[scale]
        other_positions = np.asarray(metadata["positions"], dtype=np.float64)
        if not np.array_equal(positions, other_positions):
            raise RuntimeError(f"{well.well_id}: parent positions differ")
        logits = model(torch.from_numpy(image)[None].to(device))[0]
        probability = (
            torch.softmax(logits.float() / temperature, dim=0).cpu().numpy()
        )
        probability /= probability.sum(axis=0, keepdims=True)
        native_sum_errors.append(
            float(np.max(np.abs(probability.sum(axis=0) - 1.0)))
        )
        planes.append(
            residual_probability(
                probability,
                grids[scale].offsets,
                candidate_offsets,
            )
        )
    volume = np.stack(planes).astype(np.float16)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise RuntimeError(f"{well.well_id}: invalid posterior volume")
    return (
        volume,
        positions.astype(np.float32),
        prior_tvt.astype(np.float32),
        native_sum_errors,
    )


def firewall(args: argparse.Namespace) -> None:
    with np.load(LAYOUT, allow_pickle=False) as layout:
        identifiers = layout["well_ids"].astype(str)
        original = layout["original"].astype(np.int64)
    selected = np.flatnonzero(original == args.fold)
    well = load_well(str(identifiers[selected[0]]), args.data_root)
    with np.load(
        Path(str(COMPONENT_PATTERN).format(fold=args.fold)),
        allow_pickle=False,
    ) as cache:
        ids = cache["well_ids"].astype(str)
        starts = cache["row_starts"].astype(np.int64)
        baseline_all = cache["baseline"].astype(np.float32)
    index = int(np.flatnonzero(ids == well.well_id)[0])
    left, right = starts[index : index + 2]
    baseline = baseline_all[left:right]
    device = torch.device(args.device)
    models, hashes = load_models(args.fold, device)
    grids = configs()

    arrays = {}
    native_errors = {}
    for mode in ("original", "reverse", "nan"):
        variant = inference_masked(well, mode)
        volume, positions, prior, errors = make_volume(
            variant, baseline, models, grids, device
        )
        arrays[mode] = (volume, positions, prior)
        native_errors[mode] = max(errors)
    hidden_invariant = all(
        np.array_equal(arrays["original"][i], arrays[mode][i])
        for mode in ("reverse", "nan")
        for i in range(3)
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "fold_purged_protected_posterior_fusion",
        "fold": args.fold,
        "well_id": well.well_id,
        "parents": len(models),
        "parent_names": [item[0] for item in models],
        "checkpoint_sha256": hashes,
        "posterior_shape": list(arrays["nan"][0].shape),
        "maximum_native_probability_sum_error": float(
            max(native_errors.values())
        ),
        "hidden_target_variants_bit_identical": hidden_invariant,
        "finite": bool(np.isfinite(arrays["nan"][0]).all()),
        "nonnegative": bool(np.all(arrays["nan"][0] >= 0)),
        "held_fold_checkpoint_lineage": args.fold,
        "stratified_checkpoint_loaded": False,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "status": (
            "passed"
            if hidden_invariant
            and max(native_errors.values()) < PROBABILITY_SUM_TOLERANCE
            and np.isfinite(arrays["nan"][0]).all()
            and np.all(arrays["nan"][0] >= 0)
            else "failed"
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise RuntimeError("A610 posterior firewall failed")


def build(args: argparse.Namespace) -> None:
    started = time.time()
    with np.load(LAYOUT, allow_pickle=False) as layout:
        identifiers = layout["well_ids"].astype(str)
        original = layout["original"].astype(np.int64)
    if set(identifiers) != set(training_well_ids(args.data_root)):
        raise RuntimeError("layout and training IDs differ")
    selected = np.flatnonzero(original == args.fold)
    wells = load_selected(identifiers, selected, args.data_root, args.workers)
    component_path = Path(str(COMPONENT_PATTERN).format(fold=args.fold))
    with np.load(component_path, allow_pickle=False) as cache:
        if bool(cache["audit_fold_loaded"]) or bool(
            cache["layout_truth_array_accessed"]
        ):
            raise RuntimeError("protected component cache firewall failed")
        component_ids = cache["well_ids"].astype(str)
        row_starts = cache["row_starts"].astype(np.int64)
        baseline_all = cache["baseline"].astype(np.float32)
    if not np.array_equal(component_ids, np.asarray([w.well_id for w in wells])):
        raise RuntimeError("component cache and fold well order differ")

    device = torch.device(args.device)
    models, hashes = load_models(args.fold, device)
    grids = configs()
    volumes = []
    positions = []
    priors = []
    maximum_native_error = 0.0
    for index, well in enumerate(wells):
        left, right = row_starts[index : index + 2]
        volume, position, prior, errors = make_volume(
            well, baseline_all[left:right], models, grids, device
        )
        volumes.append(volume)
        positions.append(position)
        priors.append(prior)
        maximum_native_error = max(maximum_native_error, max(errors))
        if (index + 1) % 10 == 0 or index + 1 == len(wells):
            print(
                f"fold={args.fold} wells={index + 1}/{len(wells)} "
                f"elapsed={time.time() - started:.1f}",
                flush=True,
            )
    if maximum_native_error >= PROBABILITY_SUM_TOLERANCE:
        raise RuntimeError("native posterior normalization failed")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fold=np.asarray(args.fold, dtype=np.int8),
        well_ids=np.asarray([well.well_id for well in wells]),
        row_starts=row_starts,
        baseline=baseline_all,
        posterior_probability=np.stack(volumes),
        positions=np.stack(positions),
        prior_tvt=np.stack(priors),
        residual_offsets=RESIDUAL_OFFSETS,
        parent_names=np.asarray([item[0] for item in models]),
        checkpoint_names=np.asarray([item[4].name for item in models]),
        checkpoint_sha256=np.asarray([item[5] for item in models]),
        parent_temperatures=np.asarray(
            [item[3] for item in models], dtype=np.float32
        ),
        maximum_native_probability_sum_error=np.asarray(
            maximum_native_error, dtype=np.float32
        ),
        source_sha256=np.asarray(sha256(Path(__file__))),
        component_cache_sha256=np.asarray(sha256(component_path)),
        hidden_targets_masked_during_inference=np.asarray(True),
        stratified_checkpoint_loaded=np.asarray(False),
        audit_fold_loaded=np.asarray(False),
        confirmation_regroupings_loaded=np.asarray(False),
    )
    print(
        f"wrote={args.output} fold={args.fold} wells={len(wells)} "
        f"shape={np.stack(volumes).shape} "
        f"sha256={sha256(args.output)} elapsed={time.time() - started:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--firewall-only", action="store_true")
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.firewall_only:
        firewall(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
