from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.data import DEFAULT_DATA_ROOT, training_well_ids


FAMILIES = (
    "alignment_wide_base12_exc2sup",
    "alignment_wide_base16_exc2sup",
    "alignment_longsynthetic_exc2sup",
    "alignment_longsynthetic_base12_exc2sup",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_ancc_reference(
    data_root: Path, output: Path, stride: int, workers: int
) -> None:
    well_ids = training_well_ids(data_root)

    def load(well_id: str) -> np.ndarray:
        path = data_root / "train" / f"{well_id}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["X", "Y", "ANCC"])
        values = frame[["X", "Y", "ANCC"]].to_numpy(np.float64)[::stride]
        return values[np.all(np.isfinite(values), axis=1)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(load, well_ids))
    starts = np.r_[0, np.cumsum([len(value) for value in values])].astype(np.int64)
    flat = np.concatenate(values)
    np.savez_compressed(
        output,
        well_ids=np.asarray(well_ids),
        starts=starts,
        xy=flat[:, :2],
        ancc=flat[:, 2],
        stride=np.asarray(stride),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--output", type=Path, default=Path("deploy/kaggle/weights-dataset")
    )
    parser.add_argument("--reference-stride", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    staged = []
    for family in FAMILIES:
        for fold in range(5):
            name = f"{family}_fold{fold}.pt"
            source = args.checkpoints / name
            if not source.exists():
                raise FileNotFoundError(source)
            target = args.output / name
            shutil.copy2(source, target)
            staged.append(target)

    ancc_path = args.output / "ancc_reference.npz"
    build_ancc_reference(
        args.data_root,
        ancc_path,
        args.reference_stride,
        args.workers,
    )
    staged.append(ancc_path)

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_channels"] = [8, 6, 12, 16]
    manifest["parameter_count"] = [1769417, 997399, 3972781, 7055249]
    manifest["models"] = 85
    manifest["validation"] = {
        "scheme": "whole-well OOF with all fixed post-v16 residual additions",
        "pooled_rmse": 6.166038,
        "rows": 3783989,
        "baseline_v16_rmse": 6.298881,
    }
    manifest["post_v16"] = {
        "ancc_weight": 0.01,
        "residual_gr_weight": 0.15,
        "mask_sensitivity_weight": -0.05,
        "mask_views": 4,
        "base12_residual_weight": 0.05,
        "base16_residual_weight": 0.05,
        "longsynthetic_residual_weight": 0.05,
        "feature_sensitivity_weight": -0.025,
        "feature_shift": 0.1,
        "longsynthetic_base12_residual_weight": 0.05,
    }
    hashes = manifest.setdefault("sha256", {})
    for path in staged:
        hashes[path.name] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"staged_models={len(staged) - 1} ancc={ancc_path.stat().st_size}")
    print(f"dataset_bytes={sum(path.stat().st_size for path in args.output.iterdir())}")
    for path in staged:
        print(f"{hashes[path.name]}  {path.name}")


if __name__ == "__main__":
    main()
