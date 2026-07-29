# ROGII v20 exact reproduction

This repository is the minimal, immutable reproduction surface for the
submitted ROGII Wellbore Geology Prediction v20 solution.

- Historical source commit:
  `cd2d68e9318cb04ea808b670cf6190d25c3da825`
- Kaggle submission: `54893107`
- Public leaderboard RMSE: `6.263`
- Five-fold whole-well OOF RMSE: `6.189484768154311`

`v20/infer.py` is the exact self-contained Kaggle inference source from that
commit. Its SHA-256 is frozen in `manifests/provenance.json`. No later V5
research, candidate model, or post-v20 artifact is included.

## Reproduction scope

This repository reproduces the **frozen v20 model artifact**:

1. verify every contributing checkpoint and reference store;
2. independently reconstruct the exact historical OOF metric and fold scores;
3. regenerate `submission.csv` from the competition test data.

It does not claim bit-for-bit retraining of the 80 historical neural
checkpoints. Those checkpoints arose from the earlier research and
validation-best selection process. Re-running that research is neither
necessary nor sufficient to reproduce the submitted v20 artifact.

## External immutable artifacts

Large artifacts are intentionally outside Git:

- weights: `/geo/geo_new/rogii_v20_checkpoint_vault`
- OOF: `/geo/geo_new/rogii_v20_oof_vault`
- competition data:
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction`

The defaults work with this workspace layout. Override them with
`ROGII_V20_WEIGHTS_ROOT`, `ROGII_V20_OOF_ROOT`, and `ROGII_DATA_ROOT`, or use
the corresponding command-line arguments.

The weight vault must contain exactly 80 neural checkpoints, five CatBoost
models, and two reference stores. The five historical
`alignment_longsynthetic_base12_exc2sup_fold*.pt` checkpoints have zero v20
weight and are deliberately excluded from that protected functional vault.

The exact frozen kernel nevertheless checked for and loaded those five inert
files. For source-identical execution, `reproduce.py` verifies them from
`/geo/geo_new/rogii_simple/checkpoints` and stages a temporary 92-artifact
symlink view. They never contribute to the prediction, the protected vault is
not modified, and the temporary view is removed after inference. Override
their location with `ROGII_V20_ZERO_WEIGHT_ROOT` or `--zero-weight-root`.

## Verify and score

```bash
python reproduce.py verify
```

This hashes all 87 functional artifacts and both OOF archives, then recomputes
the pooled and five original-fold scores. A quicker OOF-only check is:

```bash
python reproduce.py score-oof
```

Expected pooled output:

```text
6.189484768154311
```

## Regenerate the submission

```bash
python reproduce.py infer --output submission.csv
```

The frozen inference code automatically uses up to two GPUs, matching the
Kaggle dual-T4 deployment. After one successful `verify`, the redundant
weight hashing pass may be skipped:

```bash
python reproduce.py infer --skip-weight-verification \
  --output submission.csv
```

The generated CSV must contain every sample-submission ID exactly once with
no missing prediction. Small floating-point drift across GPU architectures is
expected; the historical remote output SHA-256 in the provenance record is a
byte-level Kaggle reference, not a cross-hardware requirement.

The standalone path was validated locally on two V100s: 14,151/14,151 rows
were produced in 23.9 seconds. Against the archived Kaggle v20 output,
prediction-vector RMSE was `0.000009374` ft and maximum absolute drift was
`0.000146399` ft.

## Dependencies

```bash
python -m pip install -r requirements.txt
```

The historical kernel metadata is retained in `kaggle/kernel-metadata.json`.
No Dataset, kernel, or competition submission is created by this repository.
