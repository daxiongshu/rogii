from __future__ import annotations

from pathlib import Path

import numpy as np

from limited_query_cv import train_v5_run6_calibrated_separable_residual as calibrated
from limited_query_cv import train_v5_run6_scratch_corrected_unet as parent
from limited_query_cv import train_v5_run6_true_raw_absolute_residual as raw
from rogii.alignment import make_alignment_example
from rogii.data import Well
from rogii.sequence import SequenceConfig


RUN_ID = "v5_batch2_run_006_goal_050"
FAMILY = "scratch_complete_well_local_half_foot_unet"
PARENT_SOURCE_SHA256 = (
    "654efa73cd7dccaf6356fb7d18911fb779d01e6e166799aff3b5ec14ce206b09"
)
VARIANTS = {
    "scratch_true_raw_local_b12": {
        "extra_channels": 8,
        "base": 12,
        "seed": 20262161,
    },
    "scratch_calibrated_local_b12": {
        "extra_channels": 8,
        "base": 12,
        "seed": 20262169,
    },
}


def local_config() -> SequenceConfig:
    return SequenceConfig(
        prefix_points=64,
        tail_points=640,
        state_radius=64.0,
        state_step=0.5,
        target_kind="tvt_delta",
        target_scale=40.0,
        row_stride=16.0,
        gr_smooth_window=51,
    )


def local_corrected_planes(
    well: Well,
    cut: int,
    image: np.ndarray,
    metadata: dict[str, np.ndarray | float | int],
    grid,
    variant: str,
) -> np.ndarray:
    if variant == "scratch_true_raw_local_b12":
        return raw.true_raw_absolute_planes(well, cut, image, metadata, grid)
    if variant == "scratch_calibrated_local_b12":
        return calibrated.calibrated_separable_planes(
            well, cut, image, metadata, grid
        )[0]
    raise ValueError(f"unsupported A770 variant {variant!r}")


def local_source_feature_contract(
    well: Well,
    grid,
    variant: str,
    synthetic: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(20262173) if synthetic else None
    image, _, metadata = make_alignment_example(
        well,
        well.anchor_index,
        grid,
        synthetic_rng=rng,
        synthetic_profile="forward_extreme",
        synthetic_hardness=0.70,
        coordinate_kind="tvt_delta",
    )
    if variant == "scratch_true_raw_local_b12":
        direct = raw.true_raw_absolute_planes(
            well, well.anchor_index, image, metadata, grid
        )
        reconstruction_rmse = None
    elif variant == "scratch_calibrated_local_b12":
        direct, recovery = calibrated.calibrated_separable_planes(
            well, well.anchor_index, image, metadata, grid
        )
        reconstruction_rmse = float(recovery["reconstruction_rmse"])
    else:
        raise ValueError(f"unsupported A770 variant {variant!r}")
    generated = parent.core.make_example(
        well,
        well.anchor_index,
        grid,
        variant,
        synthetic_rng=(
            np.random.default_rng(20262173) if synthetic else None
        ),
        synthetic_hardness=0.70 if synthetic else 0.0,
    )
    return {
        "synthetic": synthetic,
        "base_image_bit_identical": bool(np.array_equal(image, generated[0])),
        "extra_features_bit_identical": bool(
            np.array_equal(direct, generated[1])
        ),
        "shape": list(generated[1].shape),
        "finite": bool(
            np.isfinite(generated[0]).all()
            and np.isfinite(generated[1]).all()
        ),
        "reconstruction_rmse": reconstruction_rmse,
        "state_radius": float(grid.state_radius),
        "state_step": float(grid.state_step),
        "state_count": int(len(grid.offsets)),
        "horizontal_points": int(grid.length),
    }


def configure_parent() -> None:
    actual = parent.core.sha256(Path(parent.__file__))
    if actual != PARENT_SOURCE_SHA256:
        raise RuntimeError(
            f"A770 frozen A730 source mismatch: {actual}"
        )
    parent.RUN_ID = RUN_ID
    parent.FAMILY = FAMILY
    parent.VARIANTS.clear()
    parent.VARIANTS.update(VARIANTS)
    for variant, specification in VARIANTS.items():
        parent.core.VARIANTS[variant] = dict(specification)
    parent.core.config = local_config
    parent.corrected_planes = local_corrected_planes
    parent.core.residual_planes = local_corrected_planes
    parent.source_feature_contract = local_source_feature_contract

    # A730's gate and result records deliberately hash their active source.
    # Point that provenance field at this wrapper after verifying the frozen
    # parent above, so production cannot silently omit the local-grid patch.
    parent.__file__ = __file__


def main() -> None:
    configure_parent()
    parent.main()


if __name__ == "__main__":
    main()
