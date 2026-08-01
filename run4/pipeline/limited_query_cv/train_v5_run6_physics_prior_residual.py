from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from limited_query_cv import train_v5_run6_local_v20_residual as parent


PROFILES = ("linear_bend", "piecewise4", "mixed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def progress_axis(length: int, prefix_points: int) -> np.ndarray:
    anchor = prefix_points - 1
    progress = np.clip(
        (np.arange(length, dtype=np.float64) - anchor)
        / max(length - 1 - anchor, 1),
        0.0,
        1.0,
    )
    progress[:prefix_points] = 0.0
    return progress


def linear_bend_perturbation(
    length: int,
    prefix_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    progress = progress_axis(length, prefix_points)
    endpoint = float(rng.uniform(-28.0, 28.0))
    bend = float(rng.uniform(-18.0, 18.0))
    values = endpoint * progress + bend * (4.0 * progress * (1.0 - progress))
    values = np.clip(values, -32.0, 32.0)
    values[:prefix_points] = 0.0
    return values.astype(np.float32)


def piecewise4_perturbation(
    length: int,
    prefix_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    progress = progress_axis(length, prefix_points)
    knots = np.asarray((0.0, 0.25, 0.50, 0.75, 1.0), dtype=np.float64)
    increments = rng.uniform(-11.0, 11.0, size=4)
    controls = np.concatenate(([0.0], np.cumsum(increments)))
    controls = np.clip(controls, -32.0, 32.0)
    values = np.interp(progress, knots, controls)
    values = np.clip(values, -32.0, 32.0)
    values[:prefix_points] = 0.0
    return values.astype(np.float32)


def profile_perturbation(
    profile: str,
    original,
):
    if profile not in PROFILES:
        raise ValueError(f"unknown physics-prior profile: {profile}")

    def generate(
        length: int,
        prefix_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        selected = profile
        if profile == "mixed":
            selected = ("original_smooth", "linear_bend", "piecewise4")[
                int(rng.integers(0, 3))
            ]
        if selected == "original_smooth":
            values = original(length, prefix_points, rng)
        elif selected == "linear_bend":
            values = linear_bend_perturbation(length, prefix_points, rng)
        else:
            values = piecewise4_perturbation(length, prefix_points, rng)
        values = np.asarray(values, dtype=np.float32)
        values = np.clip(values, -32.0, 32.0)
        values[:prefix_points] = 0.0
        return values

    return generate


def identity_gate(args: argparse.Namespace) -> dict[str, object]:
    original = parent.smooth_perturbation
    records = parent.load_records(args.data_root)
    config = parent.grid()
    record = records[0]
    real_before = parent.make_residual_example(record, config, None)
    checks = []
    arrays = []
    for profile_index, profile in enumerate(PROFILES):
        generator = profile_perturbation(profile, original)
        for seed_index, seed in enumerate((20261183, 20261197)):
            values = generator(
                config.prefix_points + config.tail_points,
                config.prefix_points,
                np.random.default_rng(seed + 101 * profile_index),
            )
            arrays.append(values.astype(np.float64))
            checks.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "finite": bool(np.isfinite(values).all()),
                    "prefix_zero_exact": bool(
                        np.array_equal(
                            values[: config.prefix_points],
                            np.zeros(config.prefix_points, dtype=np.float32),
                        )
                    ),
                    "tail_nonzero": bool(
                        np.any(values[config.prefix_points :] != 0.0)
                    ),
                    "maximum_absolute_ft": float(np.max(np.abs(values))),
                    "tail_rms_ft": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    values[config.prefix_points :].astype(
                                        np.float64
                                    )
                                )
                            )
                        )
                    ),
                }
            )
    parent.smooth_perturbation = profile_perturbation(
        "linear_bend", original
    )
    real_after = parent.make_residual_example(record, config, None)
    real_identity = all(
        np.array_equal(before, after)
        for before, after in zip(real_before[:3], real_after[:3])
    ) and all(
        np.array_equal(real_before[3][name], real_after[3][name])
        for name in real_before[3]
    )
    pairwise_distinct = all(
        not np.array_equal(arrays[left], arrays[right])
        for left in range(len(arrays))
        for right in range(left + 1, len(arrays))
    )
    passed = bool(
        real_identity
        and pairwise_distinct
        and all(
            item["finite"]
            and item["prefix_zero_exact"]
            and item["tail_nonzero"]
            and item["maximum_absolute_ft"] <= 32.0
            for item in checks
        )
    )
    return {
        "schema_version": 1,
        "run_id": parent.RUN_ID,
        "family": "complete_well_structural_latent_alignment",
        "variant": "physics_shaped_residual_prior_identity",
        "stage": "training_only_implementation_gate",
        "status": "passed" if passed else "failed",
        "checks": checks,
        "all_draws_pairwise_distinct": pairwise_distinct,
        "real_no_rng_route_bit_identical_to_d024": real_identity,
        "audit_fold_loaded": False,
        "confirmation_regroupings_loaded": False,
        "wrapper_source_sha256": sha256(Path(__file__)),
        "parent_source_sha256": sha256(Path(parent.__file__)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--fold", type=int, choices=range(4), default=0)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--data-root", type=Path, default=parent.DEFAULT_DATA_ROOT
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20261183)
    result.add_argument("--base", type=int, default=12)
    result.add_argument("--epochs", type=int, default=16)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--loader-workers", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--synthetic-probability", type=float, default=0.5)
    result.add_argument("--residual-radius", type=float, default=32.0)
    result.add_argument("--residual-step", type=float, default=1.0)
    result.add_argument(
        "--direction",
        choices=("all", "negative_y", "positive_y"),
        default="all",
    )
    result.add_argument("--hard-alpha", type=float, default=0.0)
    result.add_argument("--classification-weight", type=float, default=1.0)
    result.add_argument(
        "--regression-kind", choices=("huber", "mse"), default="huber"
    )
    result.add_argument("--regression-weight", type=float, default=0.5)
    result.add_argument("--overfit-only", action="store_true")
    result.add_argument("--overfit-steps", type=int, default=1000)
    result.add_argument(
        "--prior-profile", choices=PROFILES, default="mixed"
    )
    result.add_argument("--profile-gate-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.profile_gate_only:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        result = identity_gate(args)
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise RuntimeError("physics-shaped residual-prior gate failed")
        return

    original = parent.smooth_perturbation
    parent.smooth_perturbation = profile_perturbation(
        args.prior_profile, original
    )
    profile = args.prior_profile
    delattr(args, "prior_profile")
    delattr(args, "profile_gate_only")
    parent.train(args)
    result_path = args.output_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["variant"] = "physics_shaped_prior_local_v20_residual"
    result["synthetic_prior_profile"] = profile
    result["wrapper_source_sha256"] = sha256(Path(__file__))
    result["parent_source_sha256"] = sha256(Path(parent.__file__))
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
