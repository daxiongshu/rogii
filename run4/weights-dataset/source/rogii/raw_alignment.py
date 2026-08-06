from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .data import Well
from .sequence import _robust_affine


_SIGNAL_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class RawAlignmentExample:
    image: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    sdf: np.ndarray
    pixel_valid: np.ndarray
    typewell_tvt: np.ndarray
    typewell_valid: np.ndarray
    horizontal_positions: np.ndarray
    anchor_tvt: float


def _resample_typewell(
    tvt: np.ndarray, gr: np.ndarray, step: float, target_step: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    ratio = step / target_step
    if np.isclose(ratio, 1.0):
        return tvt, gr
    if ratio < 1.0:
        group = int(round(1.0 / ratio))
        pad = (-len(tvt)) % group
        if pad:
            tvt = np.pad(tvt, (0, pad), mode="edge")
            gr = np.pad(gr, (0, pad), mode="edge")
        return tvt.reshape(-1, group).mean(1), gr.reshape(-1, group).mean(1)
    factor = int(round(ratio))
    source = np.arange(len(tvt), dtype=np.float64)
    target = np.linspace(0, len(tvt) - 1, (len(tvt) - 1) * factor + 1)
    return np.interp(target, source, tvt), np.interp(target, source, gr)


def _block_average(
    values: np.ndarray, positions: np.ndarray, block: int, before: bool
) -> tuple[np.ndarray, np.ndarray]:
    pad = (-len(values)) % block
    if before:
        if pad < block // 2:
            values = np.pad(values, (pad, 0), mode="edge")
            positions = np.pad(positions, (pad, 0), mode="edge")
        else:
            drop = block - pad
            values = values[drop:]
            positions = positions[drop:]
    elif pad < block // 2:
        values = np.pad(values, (0, pad), mode="edge")
        positions = np.pad(positions, (0, pad), mode="edge")
    else:
        drop = block - pad
        values = values[:-drop]
        positions = positions[:-drop]
    return (
        values.reshape(-1, block).mean(1),
        positions.reshape(-1, block).mean(1),
    )


def _crop_pad(
    values: np.ndarray, center: int, history: int, future: int
) -> tuple[np.ndarray, np.ndarray]:
    raw_start = center - history
    raw_stop = center + future
    start = max(raw_start, 0)
    stop = min(raw_stop, len(values))
    left = max(0, -raw_start)
    right = max(0, raw_stop - len(values))
    result = np.pad(values[start:stop], (left, right))
    mask = np.pad(np.ones(stop - start, dtype=np.float32), (left, right))
    return result, mask


def make_raw_alignment_example(
    well: Well,
    cut: int,
    horizontal_step: int = 12,
    normalize_gr: bool = True,
    typewell_size: int = 256,
) -> RawAlignmentExample:
    if cut < 20 or cut >= len(well.md) - 20:
        raise ValueError(f"{well.well_id}: invalid cut {cut}")
    if typewell_size < 64 or typewell_size % 2:
        raise ValueError("typewell_size must be an even integer of at least 64")
    cached = _SIGNAL_CACHE.get(well.well_id)
    if cached is None:
        typewell_step = float(np.median(np.diff(well.typewell_tvt)))
        type_tvt, type_gr = _resample_typewell(
            well.typewell_tvt.copy(), well.typewell_gr.copy(), typewell_step
        )
        source = np.arange(len(well.gr), dtype=np.float64)
        finite = np.isfinite(well.gr)
        horizontal_gr = np.interp(source, source[finite], well.gr[finite])
        window = min(49, len(horizontal_gr) - (1 - len(horizontal_gr) % 2))
        if window >= 5:
            horizontal_gr = savgol_filter(horizontal_gr, window, 2)
        cached = (type_tvt, type_gr, source, horizontal_gr)
        _SIGNAL_CACHE[well.well_id] = cached
    type_tvt, type_gr, source, horizontal_gr = cached

    split = cut + 1
    before_gr, before_position = _block_average(
        horizontal_gr[:split], source[:split], horizontal_step, before=True
    )
    after_gr, after_position = _block_average(
        horizontal_gr[split:], source[split:], horizontal_step, before=False
    )
    visible_tvt_full = well.tvt[:split]
    last_tvt = float(np.interp(before_position[-1], source[:split], visible_tvt_full))
    center = int(np.argmin(np.abs(type_tvt - last_tvt))) + 1
    type_half = typewell_size // 2
    type_tvt_crop, type_valid = _crop_pad(
        type_tvt, center, type_half, type_half
    )
    type_gr_crop, _ = _crop_pad(type_gr, center, type_half, type_half)

    before_gr, before_valid = _crop_pad(before_gr, len(before_gr), 64, 0)
    before_position, _ = _crop_pad(before_position, len(before_position), 64, 0)
    after_gr, after_valid = _crop_pad(after_gr, 0, 0, 704)
    after_position, _ = _crop_pad(after_position, 0, 0, 704)
    horizontal_gr = np.concatenate([before_gr, after_gr])
    horizontal_position = np.concatenate([before_position, after_position])
    horizontal_valid = np.concatenate([before_valid, after_valid])

    if normalize_gr:
        reference_prefix = np.interp(
            well.tvt[:split],
            well.typewell_tvt,
            well.typewell_gr,
            left=np.nan,
            right=np.nan,
        )
        gain, offset, residual_scale = _robust_affine(
            reference_prefix, well.gr[:split]
        )
        expected_gr = gain * type_gr_crop + offset
        observed_values = np.clip(
            (horizontal_gr - offset) / residual_scale, -6.0, 6.0
        )
        expected_values = np.clip(
            (expected_gr - offset) / residual_scale, -6.0, 6.0
        )
    else:
        observed_values = horizontal_gr
        expected_values = type_gr_crop
    observed = np.broadcast_to(
        observed_values[None, :], (typewell_size, 768)
    )
    expected = np.broadcast_to(expected_values[:, None], (typewell_size, 768))

    history = np.zeros((typewell_size, 768), dtype=np.float32)
    visible_tvt = np.interp(
        before_position, source[:split], visible_tvt_full
    )
    matched = np.argmin(
        np.abs(type_tvt_crop[:, None] - visible_tvt[None, :]), axis=0
    )
    for column in range(63):
        if before_valid[column] <= 0 or before_valid[column + 1] <= 0:
            continue
        cv2.line(
            history,
            (column, int(matched[column])),
            (column + 1, int(matched[column + 1])),
            1.0,
            5,
            cv2.LINE_AA,
        )

    mask = type_valid[:, None] * horizontal_valid[None, :]
    image = np.stack([expected, observed, history], axis=0).astype(np.float32)
    image *= mask[None, :, :]
    sampled_tvt = np.interp(
        horizontal_position, source, well.tvt
    )
    labels = np.argmin(
        np.abs(type_tvt_crop[:, None] - sampled_tvt[None, :]), axis=0
    ).astype(np.int64)
    type_min = np.min(type_tvt_crop[type_valid > 0])
    type_max = np.max(type_tvt_crop[type_valid > 0])
    valid = horizontal_valid * (
        (sampled_tvt >= type_min) & (sampled_tvt <= type_max)
    )
    valid[:64] = 0.0
    sdf = np.clip(
        (sampled_tvt[None, :] - type_tvt_crop[:, None]) / 40.0,
        -3.0,
        3.0,
    ).astype(np.float32)
    return RawAlignmentExample(
        image=image,
        labels=labels,
        valid=valid.astype(np.float32),
        sdf=sdf,
        pixel_valid=mask.astype(np.float32),
        typewell_tvt=type_tvt_crop.astype(np.float32),
        typewell_valid=type_valid,
        horizontal_positions=horizontal_position.astype(np.float32),
        anchor_tvt=float(well.tvt[cut]),
    )
