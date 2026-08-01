# C016 reproduction

C016 (`clean_c016_without_quarantined_d072`) is the user-approved successor to
v20 under CV protocol v5. This directory holds the code, recipe, frozen
inference program, and audit records needed to rebuild it.

- Source repository: `rogii_v20_limited_query`
- Source commit: `64e1cf12320274ef2c26f2bf465e6346b57aa88a`
- Run id: `v5_batch2_run_025_goal_050`
- Kaggle kernel: `jiweiliu/rogii-v5-promoted-c016-dual-t4`
- Kaggle submission: `55115834`
- Public leaderboard RMSE: `6.127` (v20 was `6.263`)
- Legal production OOF RMSE: `6.1272727841976` (v20 was `6.189484768154311`)

## C016 is a correction on top of v20

C016 does not retrain v20 and does not replace it. Every stage consumes v20
output:

1. `cache_v5_run3_v20_components.py` runs the frozen v20 checkpoints over each
   development fold and stores the nine weighted component paths plus the v20
   baseline prediction. It re-hashes every v20 checkpoint against the protected
   v20 weight manifest (`145fe76c…`, the same manifest `../manifests/weights.json`
   locks) before writing anything.
2. Eleven small networks are trained on those cached components — they learn a
   residual against the v20 baseline, not the target from scratch.
3. The production rule is the arithmetic mean of five clean outer predictions,
   each of which is `v20 baseline + selected correction`.

So the parent artifacts in `../manifests/weights.json` and the v20 OOF vault are
hard prerequisites. The C016 OOF archive carries the v20 baseline in its own
`baseline` array, and `reproduce_c016.py` checks that it still scores exactly
`6.189484768154311`.

## What is packaged

| Path | Contents |
|---|---|
| `pipeline/limited_query_cv/` | 23 pipeline modules — the complete transitive import closure of the C016 entry points |
| `pipeline/rogii/` | the V5 copy of the shared library (see "Two copies of `rogii`" below) |
| `pipeline/cache_alignment_oof_predictions.py` | v20 component definitions and prediction indexing, byte-identical to the repository root copy |
| `kernel/` | the exact frozen Kaggle inference program (`610f8f70…`) and its metadata |
| `weights-dataset/` | the production recipe, the 11-checkpoint manifest, and the Dataset metadata |
| `records/` | the preregistration, gate, selector, promotion, repair, and push records |
| `manifests/` | source hash lock, external artifact inventory, provenance |

Every file in the first six groups is hash-locked in
`manifests/c016_sources.json` together with its path in the source repository.
`weights-dataset/dataset-metadata.json` is taken from the source commit rather
than the working tree, because the Kaggle CLI rewrites that file in place after
a push; the committed form is the one `records/package_validation.json`
validated.

## Two copies of `rogii`

The repository root `rogii/` is frozen at the v20 commit
(`cd2d68e9…`) and is hash-locked by `../manifests/training_sources.json`.
The C016 code needs the later `rogii/alignment.py`, which adds a
`coordinate_kind` argument and label smoothing to `make_alignment_example` and
`alignment_loss`. Both additions default to the v20 behaviour, and
`train_v5_run6_local_v20_residual.py` passes `coordinate_kind="tvt_delta"`
explicitly — the v20 default — but the v20 module does not accept the keyword
at all.

Rather than mutate the v20 lock, `pipeline/rogii/` carries the C016-commit copy
of the whole package. Five of its six modules (`__init__`, `data`, `dp`,
`folds`, `sequence`) are byte-identical to the v20 copies; only `alignment.py`
differs. Put `pipeline/` first on `PYTHONPATH` and C016 code resolves its own
version while `reproduce.py verify` at the repository root keeps passing
unchanged.

## Verification

```bash
python c016/reproduce_c016.py verify
```

Checks, in order:

- all 46 hash-locked C016 sources;
- that the recipe's outer records derive exactly the 11 packaged checkpoint
  names, with the packaged byte count;
- the frozen kernel hash against the recorded production validation;
- every packaged checkpoint against `weights-dataset/c016_manifest.json`;
- the legal production OOF archive: pooled RMSE, all five fold RMSEs, the
  embedded v20 baseline, and that every original fold improves.

Weights and OOF default to this workspace and are skipped with a status note if
absent. Override with `ROGII_C016_WEIGHTS_ROOT` / `ROGII_C016_OOF` or
`--weights-root` / `--oof`.

```bash
python c016/reproduce_c016.py score-oof     # OOF metrics only
python c016/reproduce_c016.py list-recipe   # the five outer records
```

## Rebuilding C016 from v20

Run everything with `c016/pipeline` as the working directory and on
`PYTHONPATH`. Run artifacts land under
`c016/pipeline/limited_query_cv/runs/…`, mirroring the source repository
layout the scripts expect.

```bash
cd c016/pipeline
export PYTHONPATH=$PWD
```

**1. Cache the v20 components for the four development folds.**

```bash
for fold in 0 1 2 3; do
  python limited_query_cv/cache_v5_run3_v20_components.py \
    --fold "$fold" \
    --output limited_query_cv/runs/v5_batch2_run_003_goal_050/v20_components/fold$fold.npz
done
```

**2. Build the run-25 fold-4 caches and the posterior volumes.**

```bash
python limited_query_cv/prepare_v5_run25_c016_nested.py components
python limited_query_cv/prepare_v5_run25_c016_nested.py posterior
python limited_query_cv/prepare_v5_run25_c016_nested.py mode
```

**3. Train the nested roles.** The audit trains 15 roles — five `outer` roles
and ten `pair` roles — across two local seeds and three posterior variants
(75 role/variant combinations). Only the five outer roles contribute
checkpoints to production; pair roles exist so the selector can choose without
touching the held fold.

```bash
# local residual models, both seeds, every outer role
for outer in 0 1 2 3 4; do
  for seed in 20260831 20261043; do
    python limited_query_cv/train_v5_run25_c016_nested.py local \
      --role-kind outer --first "$outer" --seed "$seed"
  done
done

# posterior fusion models
python limited_query_cv/train_v5_run25_c016_nested.py posterior \
  --role-kind outer --first 3 --variant posterior_physical_b16

# pair roles, e.g. the (0,1) pair
python limited_query_cv/train_v5_run25_c016_nested.py local \
  --role-kind pair --first 0 --second 1 --seed 20260831
```

**4. Select and assemble.**

```bash
R=limited_query_cv/runs/v5_batch2_run_025_goal_050/promotion_audit
python limited_query_cv/assemble_v5_run25_c016_nested.py gate --output $R/gate.json
python limited_query_cv/assemble_v5_run25_c016_nested.py manifest \
  --output $R/sealed/candidate_manifest.json
python limited_query_cv/assemble_v5_run25_c016_nested.py assemble \
  --candidate-manifest $R/sealed/candidate_manifest.json \
  --selector-output $R/sealed/selector_record.json \
  --prediction-output $R/sealed/outer_predictions.npz \
  --prediction-manifest $R/sealed/outer_prediction_manifest.json
python limited_query_cv/assemble_v5_run25_c016_nested.py confirm \
  --output $R/confirmation_result.json
```

**5. Score the promotion and rebuild the clean candidate.**

```bash
python limited_query_cv/evaluate_v5_run25_c016_promotion.py \
  --preregistration ../records/c016_promotion_preregistration.json \
  --confirmation $R/confirmation_result.json \
  --prediction-manifest $R/sealed/outer_prediction_manifest.json \
  --output-json $R/promotion_result.json \
  --output-npz $R/promotion_oof.npz
python limited_query_cv/rebuild_v5_run25_clean_c016.py --gate \
  --output-json limited_query_cv/runs/v5_batch2_run_025_goal_050/clean_c016_gate.json
```

**6. Apply the production parity repair.** This is the label-free posterior
interpolation mask that turns the recorded OOF (`6.124619174007348`) into the
legal production OOF (`6.1272727841976`). It changes neither the recipe nor any
weight.

```bash
python limited_query_cv/repair_v5_run25_c016_posterior_interpolation.py \
  --data-root /kaggle/input/competitions/rogii-wellbore-geology-prediction \
  --output-npz .../kaggle_production/promotion_oof_legal_posterior_mask.npz \
  --output-json .../kaggle_production/posterior_interpolation_leak_repair.json
```

## Regenerating the submission

`kernel/infer.py` is the exact program the private Kaggle kernel ran. It reads
the v20 weights Dataset (`rogii-compact-alignment-cv5-weights`) and the C016
weights Dataset (`rogii-v5-promoted-c016-weights`) and needs both. On Kaggle it
produced 14,151 rows in 42.8 s on dual T4s; locally the prediction vector
matched the kernel output to `4.85e-05` ft RMSE.

## External artifacts

Nothing large is committed, matching the v20 convention. `manifests/c016_artifacts.json`
records the sha256 and this workspace's location for each:

- 11 production checkpoints (187,900,137 bytes) — `jiweiliu/rogii-v5-promoted-c016-weights`,
  locally `rogii_v20_limited_query/deploy/kaggle/c016-weights-dataset`;
- five v20 component caches (folds 0-3 from run 3, fold 4 from the run-25 cache);
- five protected posterior volumes (~150 MB each);
- the sealed outer predictions (`737711c4…`), the promotion OOF, and the legal
  production OOF (`9707bdb5…`);
- the v20 parent vaults listed in the root README.

## Provenance limits

1. The C016 checkpoints are not currently mirrored anywhere but the private
   Kaggle Dataset and this workspace; there is no separate checkpoint vault
   directory as there is for v20.
2. `cache_v5_run3_v20_components.py` hard-codes this workspace's v20 vault
   paths (`/geo/geo_new/rogii_v20_oof_vault`,
   `/geo/geo_new/rogii_v20_checkpoint_vault`). It is copied unmodified, so a
   different machine needs those paths present or the file edited.
3. Full retraining reproduces the recipe, the fold logic, and the artifact
   contract. As with v20, checkpoint bytes are not promised across CUDA, cuDNN,
   PyTorch, or GPU versions.
4. The `records/` set is the decisive audit trail, not the complete run
   directory. The full 75-role audit tree, its logs, and the diagnostic
   candidates stay in `rogii_v20_limited_query`.
