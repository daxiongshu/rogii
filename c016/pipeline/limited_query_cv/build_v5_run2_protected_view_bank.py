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

from rogii.alignment import AlignmentUNet, make_alignment_example
from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.sequence import SequenceConfig


LAYOUT = Path("/geo/geo_new/rogii_v20_oof_vault/layout.npz")
V20 = Path("/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz")
CHECKPOINT_ROOT = Path("/geo/geo_new/rogii_v20_checkpoint_vault")
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
GR_WINDOWS = (25, 51)
FAMILIES = (
    ("fixed", "alignment_fixed_synth_fold{fold}.pt", "narrow", 8, 1.5),
    ("distractor", "alignment_distractor_h05_fold{fold}.pt", "narrow", 8, 1.5),
    ("wide", "alignment_wide_synth_fold{fold}.pt", "wide", 8, 1.5),
    ("forward", "alignment_forward_std_fold{fold}.pt", "narrow", 8, 1.0),
    (
        "forward_distractor",
        "alignment_forward_distractor_fold{fold}.pt",
        "narrow",
        8,
        1.0,
    ),
    ("rare", "alignment_forward_exc4_fold{fold}.pt", "narrow", 8, 1.25),
    (
        "supervised",
        "alignment_forward_exc2sup_fold{fold}.pt",
        "narrow",
        8,
        1.0,
    ),
    (
        "wide_forward",
        "alignment_wide_forward_extreme_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    (
        "wide_supervised",
        "alignment_wide_forward_extreme_exc2sup_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    (
        "longsynthetic",
        "alignment_longsynthetic_exc2sup_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
    (
        "ordinal",
        "alignment_ordinal_wide_fold{fold}.pt",
        "wide",
        8,
        0.5,
    ),
    (
        "wide_base12",
        "alignment_wide_base12_exc2sup_fold{fold}.pt",
        "wide",
        12,
        1.0,
    ),
    (
        "wide_base16",
        "alignment_wide_base16_exc2sup_fold{fold}.pt",
        "wide",
        16,
        1.0,
    ),
    (
        "wide_base6",
        "alignment_wide_supervised_base6_fold{fold}.pt",
        "wide",
        6,
        1.0,
    ),
    (
        "wide_seed3",
        "alignment_wide_supervised_seed3_fold{fold}.pt",
        "wide",
        8,
        1.0,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_masked(well: Well) -> Well:
    anchor = well.anchor_index
    masked_target = np.where(
        np.isfinite(well.tvt_input),
        well.tvt_input,
        well.tvt_input[anchor],
    )
    return replace(well, tvt=masked_target)


def candidate_schema() -> tuple[list[str], list[str], list[str]]:
    names = []
    families = []
    decoders = []
    for family, _, _, _, _ in FAMILIES:
        for window in GR_WINDOWS:
            for temperature in TEMPERATURES:
                names.append(f"{family}_gr{window}_mean_t{temperature:g}")
                families.append(family)
                decoders.append("mean")
            for quantile in QUANTILES:
                names.append(f"{family}_gr{window}_q{quantile:g}")
                families.append(family)
                decoders.append(f"q{quantile:g}")
            names.append(f"{family}_gr{window}_mode")
            families.append(family)
            decoders.append("mode")
    return names, families, decoders


def interpolate_path(
    sampled_offset: np.ndarray,
    positions: np.ndarray,
    anchor_tvt: float,
    target_indices: np.ndarray,
) -> np.ndarray:
    return np.interp(
        target_indices, positions, anchor_tvt + sampled_offset
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the held-safe protected checkpoint view bank for one "
            "F0-F3 development fold without loading hidden truth or F4."
        )
    )
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=CHECKPOINT_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite protected view bank")

    started = time.time()
    with np.load(LAYOUT, allow_pickle=False) as layout:
        identifiers = layout["well_ids"].astype(str)
        original = layout["original"].astype(np.int8)
        global_starts = layout["row_starts"].astype(np.int64)
    if set(identifiers) != set(training_well_ids(args.data_root)):
        raise ValueError("protected layout and training IDs differ")
    selected = np.flatnonzero(original == args.fold)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        wells = list(
            pool.map(
                lambda value: load_well(value, args.data_root),
                identifiers[selected].tolist(),
            )
        )
    with np.load(V20, allow_pickle=False) as protected:
        if not np.array_equal(
            identifiers, protected["well_ids"].astype(str)
        ) or not np.array_equal(
            global_starts, protected["row_starts"].astype(np.int64)
        ):
            raise ValueError("protected v20/layout identity failed")
        baseline_all = protected["prediction"].astype(np.float32)

    configs = {}
    for window in GR_WINDOWS:
        configs[("narrow", window)] = SequenceConfig(
            prefix_points=64,
            tail_points=640,
            state_radius=64.0,
            state_step=0.5,
            target_kind="tvt_delta",
            target_scale=40.0,
            row_stride=16.0,
            gr_smooth_window=window,
        )
        configs[("wide", window)] = SequenceConfig(
            prefix_points=64,
            tail_points=640,
            state_radius=120.0,
            state_step=1.0,
            target_kind="tvt_delta",
            target_scale=40.0,
            row_stride=16.0,
            gr_smooth_window=window,
        )
    device = torch.device(args.device)
    models = []
    checkpoint_names = []
    checkpoint_hashes = []
    for family, pattern, scale, base, default_temperature in FAMILIES:
        checkpoint = args.checkpoint_root / pattern.format(fold=args.fold)
        if "strat" in checkpoint.name:
            raise ValueError("stratified checkpoint entered protected bank")
        model = AlignmentUNet(base=base).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(
            (family, model, scale, default_temperature)
        )
        checkpoint_names.append(checkpoint.name)
        checkpoint_hashes.append(sha256(checkpoint))

    names, candidate_families, decoders = candidate_schema()
    expected_candidates = (
        len(FAMILIES)
        * len(GR_WINDOWS)
        * (len(TEMPERATURES) + len(QUANTILES) + 1)
    )
    if len(names) != expected_candidates:
        raise AssertionError("protected view schema size changed")
    row_starts = [0]
    correction_parts = []
    well_ids = []
    maximum_probability_error = 0.0
    maximum_abs_correction = 0.0
    nonfinite = 0
    with torch.no_grad():
        for count, (global_index, well) in enumerate(
            zip(selected, wells), start=1
        ):
            masked = inference_masked(well)
            examples = {
                key: make_alignment_example(masked, masked.anchor_index, config)
                for key, config in configs.items()
            }
            tail = well.tail_indices
            left, right = global_starts[global_index : global_index + 2]
            baseline = baseline_all[left:right].astype(np.float64)
            if len(baseline) != len(tail):
                raise ValueError("protected baseline tail layout mismatch")
            candidates = []
            for _, model, scale, default_temperature in models:
                for window in GR_WINDOWS:
                    config = configs[(scale, window)]
                    image, _, metadata = examples[(scale, window)]
                    logits = model(
                        torch.from_numpy(image).unsqueeze(0).to(device)
                    )[0, :, config.prefix_points :]
                    valid = (
                        np.asarray(metadata["valid_positions"])[
                            config.prefix_points :
                        ]
                        > 0.5
                    )
                    logits = logits[:, valid]
                    positions = np.asarray(metadata["positions"])[
                        config.prefix_points :
                    ][valid]
                    offsets = config.offsets
                    for temperature in TEMPERATURES:
                        probability = (
                            torch.softmax(logits / temperature, dim=0)
                            .float()
                            .cpu()
                            .numpy()
                        )
                        maximum_probability_error = max(
                            maximum_probability_error,
                            float(
                                np.max(
                                    np.abs(
                                        np.sum(probability, axis=0) - 1.0
                                    )
                                )
                            ),
                        )
                        sampled = np.sum(
                            probability * offsets[:, None], axis=0
                        )
                        candidates.append(
                            interpolate_path(
                                sampled,
                                positions,
                                float(metadata["anchor_tvt"]),
                                tail,
                            )
                        )
                    probability = (
                        torch.softmax(logits / default_temperature, dim=0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    cdf = np.cumsum(probability, axis=0)
                    for quantile in QUANTILES:
                        state = np.argmax(cdf >= quantile, axis=0)
                        candidates.append(
                            interpolate_path(
                                offsets[state],
                                positions,
                                float(metadata["anchor_tvt"]),
                                tail,
                            )
                        )
                    state = np.argmax(probability, axis=0)
                    candidates.append(
                        interpolate_path(
                            offsets[state],
                            positions,
                            float(metadata["anchor_tvt"]),
                            tail,
                        )
                    )
            matrix = np.stack(candidates)
            correction = matrix - baseline[None, :]
            maximum_abs_correction = max(
                maximum_abs_correction,
                float(np.max(np.abs(correction))),
            )
            nonfinite += int(np.sum(~np.isfinite(correction)))
            correction_parts.append(correction.astype(np.float16))
            well_ids.append(well.well_id)
            row_starts.append(row_starts[-1] + len(tail))
            if count % 20 == 0 or count == len(wells):
                print(
                    f"fold={args.fold} wells={count}/{len(wells)} "
                    f"rows={row_starts[-1]} elapsed={time.time() - started:.1f}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correction = np.concatenate(correction_parts, axis=1)
    np.savez_compressed(
        args.output,
        fold=np.int8(args.fold),
        well_ids=np.asarray(well_ids),
        row_starts=np.asarray(row_starts, dtype=np.int64),
        correction=correction,
        candidate_names=np.asarray(names),
        candidate_families=np.asarray(candidate_families),
        candidate_decoders=np.asarray(decoders),
        checkpoint_names=np.asarray(checkpoint_names),
        checkpoint_sha256=np.asarray(checkpoint_hashes),
        hidden_truth_masked=np.bool_(True),
        audit_fold_loaded=np.bool_(False),
        layout_truth_array_accessed=np.bool_(False),
    )
    record = {
        "schema_version": 1,
        "status": "complete",
        "fold": args.fold,
        "wells": len(well_ids),
        "rows": int(correction.shape[1]),
        "candidate_paths": int(correction.shape[0]),
        "checkpoint_families": len(FAMILIES),
        "gr_smooth_windows": list(GR_WINDOWS),
        "posterior_mean_temperatures": list(TEMPERATURES),
        "posterior_quantiles": list(QUANTILES),
        "maximum_probability_normalization_error": maximum_probability_error,
        "maximum_absolute_correction_ft": maximum_abs_correction,
        "nonfinite_correction_elements": nonfinite,
        "stratified_checkpoints_loaded": False,
        "hidden_truth_masked": True,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "checkpoint_names": checkpoint_names,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": sha256(Path(__file__)),
        "audit_fold_loaded": False,
        "layout_truth_array_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
