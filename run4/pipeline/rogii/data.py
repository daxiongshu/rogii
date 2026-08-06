from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(
    "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
)

# Submission-safe feature contract.  The formation surfaces and typewell
# Geology are useful for training-data diagnostics, but do not exist in test.
# Keep them out of the core loader so a model cannot depend on them by accident.
HORIZONTAL_INFERENCE_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TYPEWELL_INFERENCE_COLUMNS = ("TVT", "GR")
HORIZONTAL_TARGET_COLUMN = "TVT"
TRAIN_ONLY_HORIZONTAL_COLUMNS = frozenset(
    {"TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
)
TRAIN_ONLY_TYPEWELL_COLUMNS = frozenset({"Geology"})


def _read_required_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read only the declared columns and fail with a clear schema error."""
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return pd.read_csv(path, usecols=list(columns))


def _validate_feature_contract() -> None:
    horizontal_leaks = (
        set(HORIZONTAL_INFERENCE_COLUMNS) & TRAIN_ONLY_HORIZONTAL_COLUMNS
    )
    typewell_leaks = set(TYPEWELL_INFERENCE_COLUMNS) & TRAIN_ONLY_TYPEWELL_COLUMNS
    if horizontal_leaks or typewell_leaks:
        raise RuntimeError(
            "Inference feature contract contains train-only columns: "
            f"horizontal={sorted(horizontal_leaks)}, "
            f"typewell={sorted(typewell_leaks)}"
        )


_validate_feature_contract()


@dataclass(frozen=True)
class Well:
    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    typewell_tvt: np.ndarray
    typewell_gr: np.ndarray

    @property
    def known_indices(self) -> np.ndarray:
        return np.flatnonzero(np.isfinite(self.tvt_input))

    @property
    def tail_indices(self) -> np.ndarray:
        return np.flatnonzero(~np.isfinite(self.tvt_input) & np.isfinite(self.tvt))

    @property
    def prediction_indices(self) -> np.ndarray:
        return np.flatnonzero(~np.isfinite(self.tvt_input))

    @property
    def anchor_index(self) -> int:
        known = self.known_indices
        if not len(known):
            raise ValueError(f"{self.well_id}: no visible TVT_input prefix")
        return int(known[-1])

    def validate(self) -> None:
        known = self.known_indices
        tail = self.tail_indices
        if not len(known) or not len(tail):
            raise ValueError(f"{self.well_id}: expected both visible prefix and labeled tail")
        split = known[-1] + 1
        if not np.all(np.isfinite(self.tvt_input[:split])):
            raise ValueError(f"{self.well_id}: TVT_input prefix is not contiguous")
        if np.any(np.isfinite(self.tvt_input[split:])):
            raise ValueError(f"{self.well_id}: TVT_input has values after the first hidden row")


def training_well_ids(data_root: Path = DEFAULT_DATA_ROOT) -> list[str]:
    train = Path(data_root) / "train"
    return sorted(p.name.split("__", 1)[0] for p in train.glob("*__horizontal_well.csv"))


def test_well_ids(data_root: Path = DEFAULT_DATA_ROOT) -> list[str]:
    test = Path(data_root) / "test"
    return sorted(p.name.split("__", 1)[0] for p in test.glob("*__horizontal_well.csv"))


def load_well(well_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> Well:
    train = Path(data_root) / "train"
    horizontal = _read_required_columns(
        train / f"{well_id}__horizontal_well.csv",
        HORIZONTAL_INFERENCE_COLUMNS + (HORIZONTAL_TARGET_COLUMN,),
    )
    typewell = _read_required_columns(
        train / f"{well_id}__typewell.csv",
        TYPEWELL_INFERENCE_COLUMNS,
    )
    well = Well(
        well_id=well_id,
        md=horizontal["MD"].to_numpy(np.float64),
        x=horizontal["X"].to_numpy(np.float64),
        y=horizontal["Y"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=horizontal["GR"].to_numpy(np.float64),
        tvt=horizontal["TVT"].to_numpy(np.float64),
        tvt_input=horizontal["TVT_input"].to_numpy(np.float64),
        typewell_tvt=typewell["TVT"].to_numpy(np.float64),
        typewell_gr=typewell["GR"].to_numpy(np.float64),
    )
    well.validate()
    return well


def load_test_well(well_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> Well:
    test = Path(data_root) / "test"
    horizontal = _read_required_columns(
        test / f"{well_id}__horizontal_well.csv",
        HORIZONTAL_INFERENCE_COLUMNS,
    )
    typewell = _read_required_columns(
        test / f"{well_id}__typewell.csv",
        TYPEWELL_INFERENCE_COLUMNS,
    ).sort_values("TVT")
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise ValueError(f"{well_id}: no visible TVT_input prefix")
    # The target is used only to draw the visible hint.  Hidden values are a
    # harmless anchor placeholder and never enter an inference image channel.
    tvt = np.where(np.isfinite(tvt_input), tvt_input, tvt_input[known[-1]])
    return Well(
        well_id=well_id,
        md=horizontal["MD"].to_numpy(np.float64),
        x=horizontal["X"].to_numpy(np.float64),
        y=horizontal["Y"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=horizontal["GR"].to_numpy(np.float64),
        tvt=tvt,
        tvt_input=tvt_input,
        typewell_tvt=typewell["TVT"].to_numpy(np.float64),
        typewell_gr=typewell["GR"].to_numpy(np.float64),
    )
