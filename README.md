# ROGII v20 complete reproduction

This standalone repository preserves the submitted v20 ensemble from ROGII
Wellbore Geology Prediction, including frozen inference, all single-model
training code, machine-readable training recipes, fold definitions,
postprocessing dependencies, and non-neural artifact builders.

The source is self-contained, but the large model and OOF artifacts are
private external inputs. A fresh clone needs Kaggle credentials with access to
the two private Datasets documented below.

- Historical source commit:
  `cd2d68e9318cb04ea808b670cf6190d25c3da825`
- Kaggle submission: `54893107`
- Public leaderboard RMSE: `6.263`
- Five-fold whole-well OOF RMSE: `6.189484768154311`

No V5 research, post-v20 candidate, or checkpoint is included.

## What is covered

The v20 runtime contains 17 neural families trained on five whole-well folds:
85 final checkpoints in total. Sixteen families (80 checkpoints) contribute
to predictions. The five long-base12 checkpoints are loaded by the historical
kernel but have coefficient zero.

| Family | Final checkpoint prefix | Training |
|---|---|---:|
| Fixed legacy synthetic | `alignment_fixed_synth` | 5 folds |
| Distractor synthetic | `alignment_distractor_h05` | 5 folds |
| Wide legacy synthetic | `alignment_wide_synth` | 5 folds |
| Forward synthetic | `alignment_forward_std` | 5 folds |
| Forward distractor | `alignment_forward_distractor` | 5 folds |
| Rare-excursion sampler | `alignment_forward_exc4` | 5 folds |
| Supervised-excursion sampler | `alignment_forward_exc2sup` | 5 folds |
| Wide forward-extreme | `alignment_wide_forward_extreme` | 5 folds |
| Wide supervised | `alignment_wide_forward_extreme_exc2sup` | 5 folds |
| Independent-seed wide supervised | `alignment_wide_supervised_seed3` | 5 folds |
| Base-6 wide supervised | `alignment_wide_supervised_base6` | 5 folds |
| Balanced-fold wide supervised | `alignment_wide_supervised_strat` | 5 folds |
| Ordinal wide | `alignment_ordinal_wide` | 5 folds |
| Base-12 wide | `alignment_wide_base12_exc2sup` | 5 folds |
| Base-16 wide | `alignment_wide_base16_exc2sup` | 5 folds |
| Long-synthetic base-8 | `alignment_longsynthetic_exc2sup` | 2 stages × 5 folds |
| Long-synthetic base-12 | `alignment_longsynthetic_base12_exc2sup` | 2 stages × 5 folds, zero v20 weight |

All neural families use the exact historical `evaluate_alignment.py`,
`AlignmentUNet`, data loader, sequence construction, objectives, and fold
logic. `manifests/training_recipes.json` records every option and intermediate
checkpoint. `train_v20.py` renders and safely executes those recipes.

The remaining learned/runtime artifacts are also covered:

- five CatBoost spatial-confidence models;
- `spatial_reference.npz`;
- `ancc_reference.npz`;
- the exact self-contained Kaggle inference program;
- the postprocessing and OOF evaluation modules it was developed from.

Every copied historical source file is hash-locked in
`manifests/training_sources.json`. The recipe-to-artifact verifier proves that
the 17 × 5 final checkpoint names equal the 85 neural checkpoints in the
frozen weight manifest exactly.

## Important provenance limits

This is the most complete reproduction the historical record permits, with
two gaps stated explicitly:

1. The independent-seed family documentation says only that the random seed
   changed; it did not commit the numeric seed. The recipe records
   `20260721`, inferred from the contemporaneous default/seed2/seed3 sequence,
   and labels that field as inferred.
2. The original generator of `spatial_meta.npz` was not committed. The exact
   8,402,753-byte cache survives and is SHA-256 locked, so the five CatBoost
   models and spatial reference store can be rebuilt from the exact historical
   preprocessing input.

Historical neural training enabled cuDNN benchmarking and selected the
validation-best checkpoint on the same held-out fold. A full rerun therefore
reproduces the code, data split, recipe, and artifact contract, but is not
promised to reproduce checkpoint bytes across different CUDA, cuDNN, PyTorch,
or GPU versions. This is historical v20 parity, not the later selection-clean
V5 protocol.

## Fresh clone and artifact access

```bash
git clone git@github.com:daxiongshu/rogii.git
cd rogii
python -m pip install -r requirements.txt

mkdir -p artifacts/runtime-weights artifacts/oof
kaggle datasets download \
  jiweiliu/rogii-compact-alignment-cv5-weights \
  --unzip -p artifacts/runtime-weights
kaggle datasets download \
  jiweiliu/rogii-v20-oof-vault \
  --unzip -p artifacts/oof
```

Both Datasets are private. The Kaggle account used by the CLI must be the
owner or an authorized collaborator.

The weights Dataset contains all 92 files loaded by the frozen kernel: 85
neural checkpoints, five CatBoost models, and two reference stores. Its
`manifest.json` is identical to `manifests/weights.json`, with SHA-256
`145fe76ca0a4fa48003d755ddcb661786dfe48e2ecfa9a83a2bd9b9e553e19d5`.

The OOF Dataset contains the exact `layout.npz` and
`v20_prediction.npz` archives plus their manifest. Verify the OOF download
immediately with:

```bash
python reproduce.py score-oof --oof-root artifacts/oof
```

Expected pooled OOF:

```text
6.189484768154311
```

The exact frozen inference source can use the combined remote weights layout
directly:

```bash
ROGII_DATA_ROOT=/path/to/rogii-wellbore-geology-prediction \
ROGII_WEIGHTS_ROOT="$PWD/artifacts/runtime-weights" \
ROGII_OUTPUT="$PWD/submission.csv" \
python v20/infer.py
```

Competition data is not redistributed. `ROGII_DATA_ROOT` must contain the
competition `train/`, `test/`, and `sample_submission.csv` layout.

## Full local verification

The workspace verifier intentionally keeps the 87 prediction-contributing
artifacts separate from the five zero-weight checkpoints. To create that
layout without duplicating the downloaded files:

```bash
runtime_root="$PWD/artifacts/runtime-weights"
functional_root="$PWD/artifacts/weights-functional"
zero_root="$PWD/artifacts/weights-zero"
mkdir -p "$functional_root" "$zero_root"

for path in "$runtime_root"/*.{pt,cbm,npz}; do
  name=${path##*/}
  case "$name" in
    alignment_longsynthetic_base12_exc2sup_fold?.pt)
      ln -s "$path" "$zero_root/$name"
      ;;
    *)
      ln -s "$path" "$functional_root/$name"
      ;;
  esac
done
```

Full verification also requires the exact `spatial_meta.npz` CatBoost
preprocessing cache. Its generator was never committed and the cache is not
currently in either Kaggle Dataset. In this workspace it is:

```text
/geo/geo_new/submit/5745790-94c03d6/artifacts/oof/spatial_meta.npz
```

With that cache available:

```bash
python reproduce.py verify \
  --weights-root artifacts/weights-functional \
  --zero-weight-root artifacts/weights-zero \
  --oof-root artifacts/oof \
  --spatial-meta /path/to/spatial_meta.npz
```

`verify` checks:

- the exact inference and historical training sources;
- all 17 family recipes and their exact 85-checkpoint coverage;
- the exact CatBoost preprocessing cache;
- all 87 functional runtime artifacts;
- the five zero-weight runtime checkpoints;
- both frozen OOF archives and the pooled/fold metrics.

Training-only verification is available separately:

```bash
python reproduce.py verify-training \
  --spatial-meta /path/to/spatial_meta.npz
```

## Inspect or retrain every single model

List the complete ensemble:

```bash
python train_v20.py list
```

Render a command without writing anything:

```bash
python train_v20.py commands --family wide_supervised --fold 0
python train_v20.py commands --family long_synthetic --fold 0
```

Validated dry run:

```bash
python train_v20.py run --family all --fold all \
  --checkpoint-root checkpoints
```

Actually train one family/fold:

```bash
python train_v20.py run --family wide_supervised --fold 0 \
  --checkpoint-root checkpoints --execute
```

The runner refuses to overwrite checkpoints unless `--force` is explicit.
The two long-synthetic families automatically run their 16-epoch
synthetic-only stage, save both best and last states, then fine-tune the actual
last state for six real-data epochs.

The exact spatial builder is:

```bash
python deploy/kaggle/prepare_spatial_deployment.py \
  --spatial-meta /geo/geo_new/submit/5745790-94c03d6/artifacts/oof/spatial_meta.npz \
  --output checkpoints
```

`deploy/kaggle/stage_post_v16_deployment.py` contains the exact ANCC reference
builder and historical post-v16 staging logic.

## External immutable artifacts

Large artifacts remain outside Git:

- contributing weights: `/geo/geo_new/rogii_v20_checkpoint_vault`
- private runtime-weights Dataset:
  `jiweiliu/rogii-compact-alignment-cv5-weights`
- zero-weight runtime checkpoints: `/geo/geo_new/rogii_simple/checkpoints`
- OOF archives: `/geo/geo_new/rogii_v20_oof_vault`
- private OOF backup: `jiweiliu/rogii-v20-oof-vault`
- exact spatial preprocessing cache:
  `/geo/geo_new/submit/5745790-94c03d6/artifacts/oof/spatial_meta.npz`
- competition data:
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction`

Defaults use this workspace layout. They can be overridden with
`ROGII_V20_WEIGHTS_ROOT`, `ROGII_V20_ZERO_WEIGHT_ROOT`,
`ROGII_V20_OOF_ROOT`, `ROGII_V20_SPATIAL_META`, and `ROGII_DATA_ROOT`, or
their corresponding command-line arguments.

## Regenerate the submission

```bash
python reproduce.py infer --output submission.csv
```

The frozen inference code automatically uses up to two GPUs, matching the
Kaggle dual-T4 deployment. To skip redundant weight hashing after a successful
verification:

```bash
python reproduce.py infer --skip-weight-verification \
  --output submission.csv
```

The standalone path was validated locally on two V100s: 14,151/14,151 rows
were produced in 23.9 seconds. Against the archived Kaggle v20 output,
prediction-vector RMSE was `0.000009374` ft and maximum absolute drift was
`0.000146399` ft. Small floating-point drift across GPU architectures is
expected.

The historical kernel metadata is retained in `kaggle/kernel-metadata.json`.
The artifact Datasets were created separately and are read-only inputs here.
No repository command automatically creates a Dataset or kernel, and nothing
submits a competition entry.
