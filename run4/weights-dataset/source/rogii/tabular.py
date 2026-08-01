from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_DATA_ROOT, training_well_ids


GEOLOGY_MARKERS = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "LTHL",
    "LTGT",
    "LBHL",
    "MNSS",
    "BUDA",
)

LOCAL_FEATURES = (
    "gr",
    "anchor_tvt",
    "anchor_surface",
    "md_from_anchor",
    "x_from_anchor",
    "y_from_anchor",
    "z_from_anchor",
    "fraction",
    "known_length",
    "tail_length",
    "prefix_tvt_range",
    "prefix_surface_slope",
) + tuple(f"tw_{marker}" for marker in GEOLOGY_MARKERS)

SPATIAL_FEATURES = LOCAL_FEATURES + (
    "md",
    "x",
    "y",
    "z",
    "well_x_center",
    "well_y_center",
)


def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return 0.0
    x = x[mask]
    y = y[mask]
    scale = np.ptp(x)
    if scale < 1e-6:
        return 0.0
    # A centered least-squares slope is stable and cheap at this scale.  It is
    # diagnostic context, not an extrapolated prediction.
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def _typewell_markers(path: Path) -> dict[str, float]:
    tw = pd.read_csv(path, usecols=["TVT", "Geology"])
    tw = tw.dropna(subset=["TVT", "Geology"])
    tops = tw.groupby("Geology", sort=False)["TVT"].min()
    return {f"tw_{marker}": float(tops.get(marker, np.nan)) for marker in GEOLOGY_MARKERS}


def build_training_well(well_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    train = Path(data_root) / "train"
    hw = pd.read_csv(
        train / f"{well_id}__horizontal_well.csv",
        usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"],
    )
    known = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    if not len(known):
        raise ValueError(f"{well_id}: no known prefix")
    anchor = int(known[-1])
    tail = np.arange(anchor + 1, len(hw), dtype=np.int64)
    if not len(tail):
        raise ValueError(f"{well_id}: no hidden tail")

    md = hw["MD"].to_numpy(np.float64)
    x = hw["X"].to_numpy(np.float64)
    y = hw["Y"].to_numpy(np.float64)
    z = hw["Z"].to_numpy(np.float64)
    tvt_input = hw["TVT_input"].to_numpy(np.float64)
    anchor_tvt = float(tvt_input[anchor])
    prefix_surface = tvt_input[: anchor + 1] + z[: anchor + 1]
    marker_values = _typewell_markers(train / f"{well_id}__typewell.csv")

    n_tail = len(tail)
    frame = pd.DataFrame(
        {
            "well": well_id,
            "row": tail.astype(np.int32),
            "target": hw["TVT"].to_numpy(np.float32)[tail],
            "gr": hw["GR"].to_numpy(np.float32)[tail],
            "anchor_tvt": np.float32(anchor_tvt),
            "anchor_surface": np.float32(anchor_tvt + z[anchor]),
            "md_from_anchor": (md[tail] - md[anchor]).astype(np.float32),
            "x_from_anchor": (x[tail] - x[anchor]).astype(np.float32),
            "y_from_anchor": (y[tail] - y[anchor]).astype(np.float32),
            "z_from_anchor": (z[tail] - z[anchor]).astype(np.float32),
            "fraction": (np.arange(n_tail) / max(n_tail - 1, 1)).astype(np.float32),
            "known_length": np.float32(anchor + 1),
            "tail_length": np.float32(n_tail),
            "prefix_tvt_range": np.float32(np.ptp(tvt_input[: anchor + 1])),
            "prefix_surface_slope": np.float32(
                _robust_slope(md[: anchor + 1], prefix_surface)
            ),
            "md": md[tail].astype(np.float32),
            "x": x[tail].astype(np.float32),
            "y": y[tail].astype(np.float32),
            "z": z[tail].astype(np.float32),
            "well_x_center": np.float32(np.median(x)),
            "well_y_center": np.float32(np.median(y)),
        }
    )
    for name, value in marker_values.items():
        frame[name] = np.float32(value)
    return frame


def build_training_frame(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    workers: int = 32,
    limit: int | None = None,
) -> pd.DataFrame:
    ids = training_well_ids(data_root)
    if limit is not None:
        ids = ids[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda wid: build_training_well(wid, data_root), ids))
    return pd.concat(frames, ignore_index=True)
