from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from rogii.data import DEFAULT_DATA_ROOT, Well, load_well, training_well_ids
from rogii.folds import fold_indices
from rogii.sequence import _robust_affine


RUN_ID = "v5_batch2_run_006_goal_050"
DEFAULT_COMPONENT_ROOT = Path(
    "limited_query_cv/runs/v5_batch2_run_003_goal_050/v20_components"
)
VARIANTS = ("huber_h96", "metric_h128", "fusion_h128")
SNAPSHOTS = (4, 8, 12, 16, 24, 32)
AMPLITUDES = (1.0, 1.5, 2.0, 3.0)
BLEND_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
FIXED_TVT_OFFSETS = (-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0)
NATIVE_STRIDE = 8
OUTPUT_BOUND_FT = 64.0
FUSION_FIRST_WEIGHT = 0.35
FUSION_SECOND_WEIGHT = 0.08


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_interp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return np.zeros_like(values)
    return np.interp(np.arange(len(values)), finite, values[finite])


def smooth(values: np.ndarray) -> np.ndarray:
    values = finite_interp(values)
    window = min(101, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values.copy()
    return savgol_filter(values, window, min(3, window - 1), mode="interp")


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    return center, max(scale, 1.0)


def derivative(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if len(values) < 2:
        return np.zeros_like(values)
    safe = coordinate.copy()
    for index in range(1, len(safe)):
        if safe[index] <= safe[index - 1]:
            safe[index] = safe[index - 1] + 1e-3
    return np.gradient(values, safe, edge_order=1)


def broadcast(value: float, length: int) -> np.ndarray:
    return np.full(length, float(value), dtype=np.float64)


def sampled_indices(well: Well) -> tuple[np.ndarray, np.ndarray]:
    hidden = well.prediction_indices
    local = np.arange(0, len(hidden), NATIVE_STRIDE, dtype=np.int64)
    if not len(local) or local[-1] != len(hidden) - 1:
        local = np.concatenate((local, [len(hidden) - 1]))
    return local, hidden[local]


def visible_surface_summary(
    well: Well,
    window: int | None,
) -> tuple[float, float, float, float]:
    known = well.known_indices
    if window is not None:
        known = known[-window:]
    md = well.md[known]
    surface = well.tvt_input[known] + well.z[known]
    x = md - md[-1]
    if len(x) >= 3 and np.ptp(x) > 0:
        degree = min(2, len(x) - 1)
        coefficient = np.polyfit(x / max(np.ptp(x), 1.0), surface, degree)
        slope = float(coefficient[-2]) if degree >= 1 else 0.0
        bend = float(coefficient[-3]) if degree >= 2 else 0.0
    else:
        slope = 0.0
        bend = 0.0
    return (
        slope,
        bend,
        float(np.ptp(surface)) if len(surface) else 0.0,
        float(np.std(surface)) if len(surface) else 0.0,
    )


@dataclasses.dataclass(frozen=True)
class SequenceRecord:
    well_id: str
    features: np.ndarray
    target: np.ndarray
    sample_local: np.ndarray
    baseline: np.ndarray
    truth: np.ndarray


def make_sequence_record(
    well: Well,
    baseline_tail: np.ndarray,
    component_tail: np.ndarray,
    *,
    allow_missing_target: bool = False,
) -> SequenceRecord:
    sample_local, rows = sampled_indices(well)
    length = len(rows)
    baseline_tail = np.asarray(baseline_tail, dtype=np.float64)
    component_tail = np.asarray(component_tail, dtype=np.float64)
    if len(baseline_tail) != len(well.prediction_indices):
        raise RuntimeError(f"{well.well_id}: baseline-tail layout changed")
    if component_tail.shape != (9, len(baseline_tail)):
        raise RuntimeError(f"{well.well_id}: protected-component layout changed")

    known = well.known_indices
    anchor = well.anchor_index
    md = well.md[rows]
    md_span = max(float(md[-1] - well.md[anchor]), 1.0)
    progress = np.clip((md - well.md[anchor]) / md_span, 0.0, 1.0)

    horizontal_gr = finite_interp(well.gr)
    horizontal_smooth = smooth(horizontal_gr)
    gr_center, gr_scale = robust_center_scale(horizontal_gr[known])
    gr = (horizontal_gr[rows] - gr_center) / gr_scale
    gr_smooth = (horizontal_smooth[rows] - gr_center) / gr_scale

    type_valid = np.isfinite(well.typewell_tvt) & np.isfinite(well.typewell_gr)
    type_tvt = well.typewell_tvt[type_valid]
    type_gr = well.typewell_gr[type_valid]
    order = np.argsort(type_tvt)
    type_tvt = type_tvt[order]
    type_gr = type_gr[order]
    prefix_reference = np.interp(
        well.tvt_input[known],
        type_tvt,
        type_gr,
        left=np.nan,
        right=np.nan,
    )
    gain, intercept, response_scale = _robust_affine(
        prefix_reference,
        horizontal_gr[known],
    )
    response_scale = max(float(response_scale), gr_scale, 3.0)

    baseline_sample = baseline_tail[sample_local]
    component_sample = component_tail[:, sample_local]
    component_delta = component_sample - baseline_sample[None]
    anchor_surface = float(well.tvt_input[anchor] + well.z[anchor])
    surface = baseline_sample + well.z[rows] - anchor_surface
    surface_slope = derivative(surface, md)
    surface_bend = derivative(surface_slope, md)

    dx = derivative(well.x[rows], md)
    dy = derivative(well.y[rows], md)
    dz = derivative(well.z[rows], md)
    ddx = derivative(dx, md)
    ddy = derivative(dy, md)
    ddz = derivative(dz, md)

    planes: list[np.ndarray] = [
        gr,
        gr_smooth,
        gr - gr_smooth,
        derivative(gr, md) * 100.0,
        derivative(gr_smooth, md) * 100.0,
    ]
    for offset in FIXED_TVT_OFFSETS:
        reference = gain * np.interp(
            baseline_sample + offset,
            type_tvt,
            type_gr,
            left=np.nan,
            right=np.nan,
        ) + intercept
        reference = np.nan_to_num(reference, nan=gr_center)
        mismatch = (horizontal_gr[rows] - reference) / response_scale
        planes.extend(
            (
                (reference - gr_center) / gr_scale,
                mismatch,
                np.abs(mismatch),
            )
        )

    planes.extend(
        (
            surface / 32.0,
            surface_slope * 100.0,
            surface_bend * 10000.0,
            (baseline_sample - float(well.tvt_input[anchor])) / 32.0,
            component_delta.mean(axis=0) / 16.0,
            component_delta.std(axis=0) / 16.0,
            np.ptp(component_delta, axis=0) / 16.0,
        )
    )
    planes.extend(component_delta[index] / 16.0 for index in range(9))
    planes.extend(
        (
            (well.x[rows] - well.x[anchor]) / 4000.0,
            (well.y[rows] - well.y[anchor]) / 4000.0,
            (well.z[rows] - well.z[anchor]) / 400.0,
            dx * 10.0,
            dy * 10.0,
            dz * 10.0,
            ddx * 1000.0,
            ddy * 1000.0,
            ddz * 1000.0,
            progress,
            1.0 - progress,
            np.sqrt(
                np.square(well.x[rows] - well.x[anchor])
                + np.square(well.y[rows] - well.y[anchor])
            )
            / 4000.0,
        )
    )
    for window in (128, 512, None):
        slope_value, bend_value, span_value, std_value = (
            visible_surface_summary(well, window)
        )
        planes.extend(
            (
                broadcast(slope_value / 32.0, length),
                broadcast(bend_value / 32.0, length),
                broadcast(span_value / 64.0, length),
                broadcast(std_value / 32.0, length),
            )
        )
    planes.extend(
        (
            broadcast(np.clip(gain, -4.0, 4.0), length),
            broadcast(np.clip(intercept / 100.0, -4.0, 4.0), length),
            broadcast(np.clip(response_scale / 50.0, 0.0, 4.0), length),
            broadcast(len(well.prediction_indices) / 6000.0, length),
            broadcast(np.sign(well.y[rows[-1]] - well.y[anchor]), length),
            broadcast(np.sign(well.x[rows[-1]] - well.x[anchor]), length),
        )
    )

    features = np.stack(planes, axis=1).astype(np.float32)
    if allow_missing_target:
        target = np.zeros(length, dtype=np.float32)
        truth = np.zeros(len(well.prediction_indices), dtype=np.float32)
    else:
        target = np.clip(
            well.tvt[rows] - baseline_sample,
            -OUTPUT_BOUND_FT,
            OUTPUT_BOUND_FT,
        ).astype(np.float32)
        truth = well.tvt[well.prediction_indices].astype(np.float32)
    if not (
        np.isfinite(features).all()
        and np.isfinite(target).all()
        and np.isfinite(baseline_tail).all()
    ):
        raise RuntimeError(f"{well.well_id}: nonfinite recurrent example")
    return SequenceRecord(
        well_id=well.well_id,
        features=features,
        target=target,
        sample_local=sample_local,
        baseline=baseline_tail.astype(np.float32),
        truth=truth,
    )


def fit_normalizer(records: list[SequenceRecord]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([record.features for record in records], axis=0)
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center[None]), axis=0)
    scale = np.maximum(scale, 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


class RecurrentResidualNet(nn.Module):
    def __init__(
        self,
        input_features: int,
        variant: str,
        center: np.ndarray,
        scale: np.ndarray,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown recurrent variant: {variant}")
        hidden = 96 if variant == "huber_h96" else 128
        layers = 3 if variant == "metric_h128" else 2
        self.variant = variant
        self.register_buffer("feature_center", torch.from_numpy(center))
        self.register_buffer("feature_scale", torch.from_numpy(scale))
        self.input = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.refine = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.direct = nn.Linear(2 * hidden, 1)
        self.difference = (
            nn.Linear(2 * hidden, 1)
            if variant == "fusion_h128"
            else None
        )
        nn.init.zeros_(self.direct.weight)
        nn.init.zeros_(self.direct.bias)
        if self.difference is not None:
            nn.init.zeros_(self.difference.weight)
            nn.init.zeros_(self.difference.bias)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        values = torch.clamp(
            (features.float() - self.feature_center) / self.feature_scale,
            -8.0,
            8.0,
        )
        values = self.input(values)
        packed = pack_padded_sequence(
            values,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.gru(packed)
        values, _ = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=features.shape[1],
        )
        values = self.refine(values)
        direct = OUTPUT_BOUND_FT * torch.tanh(
            self.direct(values).squeeze(-1) / 16.0
        )
        direct = direct - direct[:, :1]
        difference = None
        if self.difference is not None:
            difference = 16.0 * torch.tanh(
                self.difference(values).squeeze(-1) / 4.0
            )
            difference = torch.cat(
                (torch.zeros_like(difference[:, :1]), difference[:, 1:]),
                dim=1,
            )
        return direct, difference


class SequenceDataset(Dataset):
    def __init__(self, records: list[SequenceRecord], repeats: int) -> None:
        self.records = records
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index % len(self.records)]
        return torch.from_numpy(record.features), torch.from_numpy(record.target)


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, targets = zip(*batch)
    lengths = torch.as_tensor([len(value) for value in features], dtype=torch.long)
    feature_pad = pad_sequence(features, batch_first=True)
    target_pad = pad_sequence(targets, batch_first=True)
    mask = (
        torch.arange(feature_pad.shape[1])[None]
        < lengths[:, None]
    ).float()
    return feature_pad, target_pad, lengths, mask


def recurrent_objective(
    direct: torch.Tensor,
    difference: torch.Tensor | None,
    target: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    variant: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    denominator = mask.sum().clamp_min(1.0)
    scaled_error = (direct - target) / 8.0
    if variant == "metric_h128":
        weight = 1.0 + torch.clamp(torch.abs(target) / 16.0, max=3.0)
        direct_loss = (torch.square(scaled_error) * weight * mask).sum() / (
            weight * mask
        ).sum().clamp_min(1.0)
    else:
        direct_loss = (
            F.smooth_l1_loss(
                direct / 8.0,
                target / 8.0,
                beta=1.0,
                reduction="none",
            )
            * mask
        ).sum() / denominator
    pair_mask = mask[:, 1:] * mask[:, :-1]
    direct_change = torch.diff(direct, dim=1)
    target_change = torch.diff(target, dim=1)
    change_loss = (
        torch.square((direct_change - target_change) / 4.0) * pair_mask
    ).sum() / pair_mask.sum().clamp_min(1.0)
    rows = torch.arange(len(lengths), device=direct.device)
    endpoint = direct[rows, lengths - 1]
    target_endpoint = target[rows, lengths - 1]
    endpoint_loss = torch.mean(torch.square((endpoint - target_endpoint) / 8.0))
    difference_loss = direct.new_tensor(0.0)
    if difference is not None:
        difference_loss = (
            torch.square((difference[:, 1:] - target_change) / 4.0) * pair_mask
        ).sum() / pair_mask.sum().clamp_min(1.0)
    total = (
        direct_loss
        + 0.05 * change_loss
        + 0.20 * endpoint_loss
        + 0.50 * difference_loss
    )
    return total, {
        "direct": direct_loss,
        "change": change_loss,
        "endpoint": endpoint_loss,
        "difference": difference_loss,
    }


def fusion_objective(
    path: np.ndarray,
    direct: np.ndarray,
    difference: np.ndarray,
) -> float:
    return float(
        np.sum(np.square(path - direct))
        + FUSION_FIRST_WEIGHT
        * np.sum(np.square(np.diff(path) - difference[1:]))
        + FUSION_SECOND_WEIGHT * np.sum(np.square(np.diff(path, n=2)))
    )


def fuse_direct_difference(
    direct: np.ndarray,
    difference: np.ndarray,
) -> np.ndarray:
    direct = np.asarray(direct, dtype=np.float64)
    difference = np.asarray(difference, dtype=np.float64)
    length = len(direct)
    if length < 2:
        return np.zeros_like(direct, dtype=np.float32)
    first = diags(
        [-np.ones(length - 1), np.ones(length - 1)],
        [0, 1],
        shape=(length - 1, length),
        format="csr",
    )
    if length >= 3:
        second = diags(
            [
                np.ones(length - 2),
                -2.0 * np.ones(length - 2),
                np.ones(length - 2),
            ],
            [0, 1, 2],
            shape=(length - 2, length),
            format="csr",
        )
        curvature = second.T @ second
    else:
        curvature = diags([np.zeros(length)], [0], format="csr")
    matrix = (
        diags([np.ones(length)], [0], format="csr")
        + FUSION_FIRST_WEIGHT * (first.T @ first)
        + FUSION_SECOND_WEIGHT * curvature
    )
    right = direct + FUSION_FIRST_WEIGHT * np.asarray(
        first.T @ difference[1:]
    ).reshape(-1)
    # Enforce the registered anchor exactly inside the solve. Translating an
    # unconstrained solution afterward does not minimize the frozen objective.
    fused = np.zeros(length, dtype=np.float64)
    fused[1:] = np.asarray(
        spsolve(matrix[1:, 1:], right[1:]),
        dtype=np.float64,
    )
    return np.clip(fused, -OUTPUT_BOUND_FT, OUTPUT_BOUND_FT).astype(np.float32)


@torch.no_grad()
def predict_sampled(
    model: RecurrentResidualNet,
    record: SequenceRecord,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    feature = torch.from_numpy(record.features[None]).to(device)
    length = torch.as_tensor([len(record.features)], device=device)
    direct, difference = model(feature, length)
    direct_np = direct[0, : length.item()].float().cpu().numpy()
    if difference is None:
        correction = direct_np
    else:
        correction = fuse_direct_difference(
            direct_np,
            difference[0, : length.item()].float().cpu().numpy(),
        )
    correction = correction - correction[0]
    return np.clip(correction, -OUTPUT_BOUND_FT, OUTPUT_BOUND_FT)


def native_correction(
    sampled: np.ndarray,
    sample_local: np.ndarray,
    native_length: int,
) -> np.ndarray:
    return np.interp(
        np.arange(native_length, dtype=np.float64),
        sample_local.astype(np.float64),
        sampled.astype(np.float64),
    ).astype(np.float32)


def sampled_rmse(
    model: RecurrentResidualNet,
    records: list[SequenceRecord],
    device: torch.device,
) -> float:
    error = []
    for record in records:
        prediction = predict_sampled(model, record, device)
        error.append(np.square(prediction - record.target))
    return float(np.sqrt(np.mean(np.concatenate(error))))


def hidden_target_invariance(
    model: RecurrentResidualNet,
    well: Well,
    baseline: np.ndarray,
    component: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    variants = (
        well,
        dataclasses.replace(well, tvt=well.tvt[::-1].copy()),
        dataclasses.replace(well, tvt=np.full_like(well.tvt, np.nan)),
    )
    feature_arrays = []
    corrections = []
    for variant in variants:
        record = make_sequence_record(
            variant,
            baseline,
            component,
            allow_missing_target=True,
        )
        feature_arrays.append(record.features)
        corrections.append(predict_sampled(model, record, device))
    return {
        "features_bit_identical": bool(
            all(np.array_equal(feature_arrays[0], value) for value in feature_arrays[1:])
        ),
        "corrections_bit_identical": bool(
            all(np.array_equal(corrections[0], value) for value in corrections[1:])
        ),
        "feature_center_sha256": hashlib.sha256(center.tobytes()).hexdigest(),
        "feature_scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        "maximum_abs_first_correction_ft": float(
            max(abs(float(value[0])) for value in corrections)
        ),
        "maximum_abs_correction_ft": float(
            max(np.max(np.abs(value)) for value in corrections)
        ),
    }


def fixed_batch_gate(
    records: list[SequenceRecord],
    wells: list[Well],
    baseline_paths: dict[str, np.ndarray],
    component_paths: dict[str, np.ndarray],
    variant: str,
    device: torch.device,
    steps: int,
    seed: int,
) -> dict[str, object]:
    fixed = records[:2]
    untouched = records[2:4]
    center, scale = fit_normalizer(fixed)
    model = RecurrentResidualNet(
        fixed[0].features.shape[1],
        variant,
        center,
        scale,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    initial_fixed = sampled_rmse(model, fixed, device)
    initial_untouched = sampled_rmse(model, untouched, device)
    feature, target, lengths, mask = collate(
        [
            (torch.from_numpy(record.features), torch.from_numpy(record.target))
            for record in fixed
        ]
    )
    feature = feature.to(device)
    target = target.to(device)
    lengths = lengths.to(device)
    mask = mask.to(device)
    nonfinite = 0
    history = []
    started = time.time()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        direct, difference = model(feature, lengths)
        loss, parts = recurrent_objective(
            direct, difference, target, mask, lengths, variant
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite recurrent gate loss")
        loss.backward()
        step_nonfinite = sum(
            int((~torch.isfinite(parameter.grad)).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        nonfinite += step_nonfinite
        if step_nonfinite:
            raise FloatingPointError("nonfinite recurrent gate gradients")
        gradient = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        if step == 1 or step % 200 == 0:
            item = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": gradient,
                **{key: float(value.detach()) for key, value in parts.items()},
            }
            history.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    final_fixed = sampled_rmse(model, fixed, device)
    final_untouched = sampled_rmse(model, untouched, device)
    invariance = hidden_target_invariance(
        model,
        wells[2],
        baseline_paths[wells[2].well_id],
        component_paths[wells[2].well_id],
        center,
        scale,
        device,
    )
    untouched_correction = predict_sampled(model, untouched[0], device)
    no_op_exact = bool(
        np.array_equal(
            untouched[0].baseline,
            untouched[0].baseline + np.float32(0.0) * native_correction(
                untouched_correction,
                untouched[0].sample_local,
                len(untouched[0].baseline),
            ),
        )
    )
    constructed = np.sin(np.linspace(0, 2 * np.pi, 65)) * 8.0
    constructed -= constructed[0]
    constructed_direct = constructed + np.linspace(0, 5, len(constructed))
    constructed_difference = np.r_[0.0, np.diff(constructed)]
    constructed_fused = fuse_direct_difference(
        constructed_direct, constructed_difference
    )
    constructed_direct_objective = fusion_objective(
        constructed_direct, constructed_direct, constructed_difference
    )
    constructed_fused_objective = fusion_objective(
        constructed_fused, constructed_direct, constructed_difference
    )
    passed = bool(
        final_fixed < 0.20 * initial_fixed
        and final_untouched != initial_untouched
        and np.max(np.abs(untouched_correction)) > 1e-5
        and np.max(np.abs(untouched_correction)) <= OUTPUT_BOUND_FT
        and invariance["features_bit_identical"]
        and invariance["corrections_bit_identical"]
        and invariance["maximum_abs_first_correction_ft"] == 0.0
        and no_op_exact
        and nonfinite == 0
        and (
            variant != "fusion_h128"
            or constructed_fused_objective < constructed_direct_objective
        )
    )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_recurrent_residual_refinement",
        "variant": variant,
        "stage": "fixed_batch_and_inference_firewall_gate",
        "status": "passed" if passed else "failed",
        "fixed_well_ids": [record.well_id for record in fixed],
        "untouched_well_ids": [record.well_id for record in untouched],
        "initial_fixed_rmse": initial_fixed,
        "final_fixed_rmse": final_fixed,
        "initial_untouched_rmse": initial_untouched,
        "final_untouched_rmse": final_untouched,
        "invariance": invariance,
        "zero_share_parent_exact": no_op_exact,
        "constructed_fusion": {
            "direct_objective": constructed_direct_objective,
            "fused_objective": constructed_fused_objective,
            "improved": constructed_fused_objective
            < constructed_direct_objective,
        },
        "feature_count": int(fixed[0].features.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "nonfinite_gradient_elements": nonfinite,
        "steps": steps,
        "history": history,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }


def merge_gates(paths: list[Path], output: Path) -> None:
    gates = [json.loads(path.read_text()) for path in paths]
    expected = set(VARIANTS)
    variants = {str(gate["variant"]) for gate in gates}
    source_hash = sha256(Path(__file__))
    passed = bool(
        variants == expected
        and all(gate["status"] == "passed" for gate in gates)
        and all(gate["source_sha256"] == source_hash for gate in gates)
        and all(not gate["audit_fold_loaded"] for gate in gates)
        and all(not gate["confirmation_regroupings_loaded"] for gate in gates)
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_recurrent_residual_refinement",
        "variant": "recurrent_residual_family_gate",
        "stage": "aggregate_metric_free_gate",
        "status": "passed" if passed else "failed",
        "variants": sorted(variants),
        "gate_paths": [str(path) for path in paths],
        "gate_sha256": {str(path): sha256(path) for path in paths},
        "source_sha256": source_hash,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("recurrent residual aggregate gate failed")


@torch.no_grad()
def evaluate(
    model: RecurrentResidualNet,
    records: list[SequenceRecord],
    device: torch.device,
) -> tuple[dict[str, object], dict[float, np.ndarray]]:
    baseline = np.concatenate([record.baseline for record in records]).astype(
        np.float32
    )
    truth = np.concatenate([record.truth for record in records]).astype(np.float32)
    corrections = []
    for record in records:
        sampled = predict_sampled(model, record, device)
        corrections.append(
            native_correction(sampled, record.sample_local, len(record.baseline))
        )
    correction = np.concatenate(corrections).astype(np.float32)
    predictions = {
        amplitude: (baseline + amplitude * correction).astype(np.float32)
        for amplitude in AMPLITUDES
    }
    baseline_rmse = float(
        np.sqrt(np.mean(np.square(baseline.astype(np.float64) - truth)))
    )
    grid = []
    for amplitude, candidate in predictions.items():
        standalone = float(
            np.sqrt(np.mean(np.square(candidate.astype(np.float64) - truth)))
        )
        for share in BLEND_WEIGHTS:
            prediction = baseline + share * (candidate - baseline)
            score = float(
                np.sqrt(np.mean(np.square(prediction.astype(np.float64) - truth)))
            )
            grid.append(
                {
                    "amplitude": amplitude,
                    "share": share,
                    "effective_scale": amplitude * share,
                    "standalone_rmse": standalone,
                    "rmse": score,
                    "gain": baseline_rmse - score,
                }
            )
    best = min(
        grid,
        key=lambda item: (
            item["rmse"],
            item["amplitude"],
            item["share"],
        ),
    )
    return {
        "baseline_rmse": baseline_rmse,
        "best_same_held_diagnostic_only": best,
        "grid": grid,
    }, predictions


def load_all(
    data_root: Path,
    component_root: Path,
    workers: int,
) -> tuple[
    list[Well],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[str, ...],
]:
    identifiers = np.asarray(training_well_ids(data_root))
    folds = np.empty(len(identifiers), dtype=np.int8)
    for fold in range(5):
        _, held = fold_indices(identifiers, fold, 5)
        folds[held] = fold
    eligible = identifiers[folds < 4]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        wells = list(pool.map(lambda well_id: load_well(well_id, data_root), eligible))
    baseline_paths, component_paths, component_names = load_component_paths(
        component_root
    )
    if set(eligible.tolist()) != set(baseline_paths):
        raise RuntimeError("protected F0-F3 component coverage changed")
    return wells, folds[folds < 4], baseline_paths, component_paths, component_names


def load_component_paths(
    root: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[str, ...],
]:
    baseline_paths: dict[str, np.ndarray] = {}
    component_paths: dict[str, np.ndarray] = {}
    component_names: tuple[str, ...] | None = None
    for fold in range(4):
        path = root / f"fold{fold}.npz"
        with np.load(path, allow_pickle=False) as cache:
            if bool(cache["audit_fold_loaded"]) or bool(
                cache["layout_truth_array_accessed"]
            ):
                raise RuntimeError(f"{path}: component-cache firewall failed")
            identifiers = cache["well_ids"].astype(str)
            starts = cache["row_starts"].astype(np.int64)
            baseline = cache["baseline"].astype(np.float32)
            component = cache["components"].astype(np.float32)
            names = tuple(cache["component_names"].astype(str))
        if component_names is None:
            component_names = names
        elif component_names != names:
            raise RuntimeError("protected component order differs by fold")
        for index, well_id in enumerate(identifiers):
            left, right = int(starts[index]), int(starts[index + 1])
            if well_id in baseline_paths:
                raise RuntimeError(f"duplicate protected path for {well_id}")
            baseline_paths[well_id] = baseline[left:right].copy()
            component_paths[well_id] = component[:, left:right].copy()
    if component_names is None:
        raise RuntimeError("no protected component cache loaded")
    return baseline_paths, component_paths, component_names


def run_gate(args: argparse.Namespace) -> None:
    wells, folds, baseline_paths, component_paths, component_names = load_all(
        args.data_root, args.component_root, args.workers
    )
    gate_wells = [well for well, fold in zip(wells, folds) if fold in (1, 2, 3)][:4]
    records = [
        make_sequence_record(
            well,
            baseline_paths[well.well_id],
            component_paths[well.well_id],
        )
        for well in gate_wells
    ]
    result = fixed_batch_gate(
        records,
        gate_wells,
        baseline_paths,
        component_paths,
        args.variant,
        torch.device(args.device),
        args.gate_steps,
        args.seed,
    )
    result["component_names"] = list(component_names)
    result["fixed_tvt_offsets"] = list(FIXED_TVT_OFFSETS)
    result["native_stride"] = NATIVE_STRIDE
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise RuntimeError(f"{args.variant}: recurrent fixed-batch gate failed")


def train(args: argparse.Namespace) -> None:
    gate = json.loads(args.gate_json.read_text())
    source_hash = sha256(Path(__file__))
    if (
        gate["status"] != "passed"
        or gate["source_sha256"] != source_hash
        or gate["variant"] != "recurrent_residual_family_gate"
    ):
        raise RuntimeError("recurrent aggregate gate does not match source")
    wells, folds, baseline_paths, component_paths, component_names = load_all(
        args.data_root, args.component_root, args.workers
    )
    train_wells = [
        well for well, fold in zip(wells, folds) if int(fold) != args.fold
    ]
    held_wells = [
        well for well, fold in zip(wells, folds) if int(fold) == args.fold
    ]
    train_records = [
        make_sequence_record(
            well,
            baseline_paths[well.well_id],
            component_paths[well.well_id],
        )
        for well in train_wells
    ]
    held_records = [
        make_sequence_record(
            well,
            baseline_paths[well.well_id],
            component_paths[well.well_id],
        )
        for well in held_wells
    ]
    center, scale = fit_normalizer(train_records)
    device = torch.device(args.device)
    model = RecurrentResidualNet(
        train_records[0].features.shape[1],
        args.variant,
        center,
        scale,
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    dataset = SequenceDataset(train_records, args.repeats)
    history = []
    snapshots = []
    started = time.time()
    nonfinite = 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        losses = []
        gradients = []
        component_values = []
        epoch_started = time.time()
        for feature, target, lengths, mask in loader:
            feature = feature.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            direct, difference = model(feature, lengths)
            loss, parts = recurrent_objective(
                direct,
                difference,
                target,
                mask,
                lengths,
                args.variant,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite recurrent training loss")
            loss.backward()
            step_nonfinite = sum(
                int((~torch.isfinite(parameter.grad)).sum().item())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            nonfinite += step_nonfinite
            if step_nonfinite:
                raise FloatingPointError("nonfinite recurrent training gradients")
            gradients.append(
                float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            component_values.append(
                {key: float(value.detach()) for key, value in parts.items()}
            )
        scheduler.step()
        record = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "mean_gradient_norm": float(np.mean(gradients)),
            "mean_components": {
                key: float(np.mean([item[key] for item in component_values]))
                for key in component_values[0]
            },
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch in SNAPSHOTS:
            evaluation, predictions = evaluate(model, held_records, device)
            checkpoint = args.output_dir / f"epoch{epoch:03d}.pt"
            prediction_path = args.output_dir / f"epoch{epoch:03d}_prediction.npz"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": args.variant,
                    "feature_center": center,
                    "feature_scale": scale,
                    "component_names": component_names,
                    "fixed_tvt_offsets": FIXED_TVT_OFFSETS,
                    "native_stride": NATIVE_STRIDE,
                },
                checkpoint,
            )
            np.savez_compressed(
                prediction_path,
                fold=np.asarray(args.fold, dtype=np.int8),
                epoch=np.asarray(epoch, dtype=np.int16),
                variant=np.asarray(args.variant),
                well_ids=np.asarray([record.well_id for record in held_records]),
                row_starts=np.concatenate(
                    (
                        [0],
                        np.cumsum([len(record.baseline) for record in held_records]),
                    )
                ).astype(np.int64),
                baseline=np.concatenate(
                    [record.baseline for record in held_records]
                ).astype(np.float32),
                truth=np.concatenate([record.truth for record in held_records]).astype(
                    np.float32
                ),
                **{
                    f"prediction_a{int(amplitude * 100):03d}": prediction
                    for amplitude, prediction in predictions.items()
                },
                audit_fold_loaded=np.asarray(False),
                confirmation_regroupings_loaded=np.asarray(False),
            )
            snapshot = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "evaluation": evaluation,
            }
            snapshots.append(snapshot)
            print(
                json.dumps(
                    {
                        "snapshot": epoch,
                        "best_same_held_diagnostic_only": evaluation[
                            "best_same_held_diagnostic_only"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "family": "complete_well_recurrent_residual_refinement",
        "variant": args.variant,
        "stage": "recurrent_residual_F0_capacity",
        "status": "complete",
        "fold": args.fold,
        "training_folds": [
            fold for fold in range(4) if fold != args.fold
        ],
        "training_wells": len(train_records),
        "held_wells": len(held_records),
        "feature_count": int(train_records[0].features.shape[1]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "configuration": {
            "snapshots": list(SNAPSHOTS),
            "amplitudes": list(AMPLITUDES),
            "blend_weights": list(BLEND_WEIGHTS),
            "fixed_tvt_offsets": list(FIXED_TVT_OFFSETS),
            "native_stride": NATIVE_STRIDE,
            "output_bound_ft": OUTPUT_BOUND_FT,
            "epochs": args.epochs,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "history": history,
        "snapshots": snapshots,
        "nonfinite_gradient_elements": nonfinite,
        "source_sha256": source_hash,
        "gate_json": str(args.gate_json),
        "gate_sha256": sha256(args.gate_json),
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "selection_clean_nested_selector_run": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT
    )
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261301)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-steps", type=int, default=1200)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--merge-gates", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge_gates:
        if args.gate_json is None:
            raise ValueError("--merge-gates requires --gate-json output")
        merge_gates(args.merge_gates, args.gate_json)
        return
    if args.variant is None or args.output_dir is None:
        raise ValueError("--variant and --output-dir are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.gate_only:
        args.output_dir.mkdir(parents=True)
        run_gate(args)
        return
    if args.gate_json is None:
        raise ValueError("training requires --gate-json")
    train(args)


if __name__ == "__main__":
    main()
