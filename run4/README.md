# Run4 reproduction

Run4 (`run4_structural_surface_router_s06`) is the current frozen V5 parent and
the user-approved successor to C016. It routes each well between a structural
candidate blend and a raw surface candidate blend, then adds the selected
correction to the C016 parent prediction at production scale `0.6`.

- Source repository: `rogii_v20_limited_query`
- Build commit: `524eb22ceb56503eaf1bdca0762c67119eb68916`
- Promotion commit: `1ccb125` (added the returned leaderboard score and approval)
- Run id: `v5_batch3_run_004_goal_020`
- Kaggle kernel: `jiweiliu/rogii-v5-run4-sr-dual-t4`
- Kaggle submission: `55147410`
- Public leaderboard RMSE: `5.962` (C016 was `6.127`, v20 was `6.263`)
- Deploy-parity OOF RMSE: `6.066500604210546` (C016 was `6.1272727841976`)
- Recorded selection OOF RMSE: `6.063838803353486`

## The lineage is v20 → C016 → Run4

Run4 retrains nothing from either ancestor:

- the **C016 sealed outer prediction** is a hash-pinned input.
  `pipeline/limited_query_cv/v5_batch3_run4_clean_bank.py` carries
  `PARENT_SHA256 = 737711c4…` and refuses to load anything else. That is the
  same artifact `../c016/manifests/c016_artifacts.json` records as
  `sealed_outer_predictions`, and `reproduce_run4.py verify` cross-checks the
  two subtrees agree;
- the **v20 component caches** (folds 0-3 from run 3, fold 4 from the run-25
  cache) and the **v20 OOF layout** are read directly by the candidate bank;
- the recorded OOF archive stores its own `clean_C016` and `v20` arrays, which
  still score exactly `6.124619174007348` and `6.189484768154311`.

So all three generations must be present to verify Run4 end to end.

## Two OOF numbers, and why they differ

`records/run4_frozen_parent_alignment.json` defines the deploy-parity
reconstruction:

```text
run4_recorded_prediction + c016_legal_prediction - c016_recorded_prediction
```

Selection ran against the C016 prediction as *recorded* (`6.124619174007348`),
while production deploys the C016 prediction as *legally repaired*
(`6.1272727841976`). The reconstruction re-states Run4 on the deployed parent,
which is the number the frozen-parent decision used: `6.066500604210546`, a
gain of `0.060772179987053754` with all five folds improving.

`reproduce_run4.py` reproduces the deploy-parity pooled RMSE, both fold vectors,
and the gain **exactly**. It checks the recorded `6.063838803353486` only to
`1e-6`, because the archive stores predictions as float32 and that costs about
`2e-7` of pooled RMSE against the float64 value computed at selection time. The
observed archive value is `6.0638386078391315`.

## What is packaged

| Path | Contents |
|---|---|
| `pipeline/limited_query_cv/` | 62 modules — the closure of the 16 Run4 entry points over both imports *and* the caches they read by path, so the candidate bank builds from scratch |
| `pipeline/rogii/`, `pipeline/evaluate_alignment.py`, `pipeline/create_stratified_folds.py` | the V5 forms of the shared library and the v20 trainer that Run4's candidate bank imports |
| `kernel/` | the frozen Kaggle program (`02aa6105…`), its metadata, and the deployment validation record |
| `weights-dataset/` | the deployment recipe plus the `source/` snapshot the kernel inserts into `sys.path` at run time |
| `records/` | selector preregistration, recipe freeze, nested verification, confirmation, promotion, and the frozen-parent alignment record |
| `manifests/` | source hash lock, external artifact inventory, provenance |

All 119 copied files are hash-locked in `manifests/run4_sources.json`. 117 match
build commit `524eb22c` byte for byte; `kernel/deployment-validation.json` and
`records/run4_frozen_parent_alignment.json` are the post-submission forms from
promotion commit `1ccb125`, and the manifest names them explicitly. `README.md`,
`reproduce_run4.py`, `requirements.txt`, and `manifests/` are written for this
repository and are not hash-locked.

### Closure over paths, not just imports

Run4's candidate bank reads its 18 candidates from caches **by path**, not by
import, so an import-only closure silently omits the code that produces them.
The packaged closure follows both:

| Candidates | Read from | Producers, all packaged here |
|---|---|---|
| 12 × `run1_struct_*` | `runs/v5_batch3_run_001_goal_020/f{fold}_highcap_structural/shard*.npz` | `cache_v5_batch3_highcap_path.py`, `cache_v5_batch3_datum_path.py`, `combine_v5_batch3_structural_caches.py`, `transform_v5_batch3_highcap_structural.py`, `combine_v5_batch3_named_structural_caches.py` |
| 6 × `surface_raw_*` | `runs/v5_batch3_run_002_goal_020/surface_f{fold}_cache.npz` | `cache_v5_batch3_run2_surface.py` |
| v20 component paths | `runs/v5_batch2_run_00{1,3}_goal_050/v20_components/fold{n}.npz` | `cache_v5_run3_v20_components.py` |

With those included, the chain terminates at competition data, the v20
checkpoint vault, and the C016 sealed prediction — nothing else is required.

### Note on `pipeline/evaluate_alignment.py` and `pipeline/rogii/`

These are **not** the same files as the repository root copies. The root
`evaluate_alignment.py` and `rogii/` are frozen at the v20 commit and hash-locked
by `../manifests/training_sources.json`; the V5 branch evolved both. Run4 needs
the V5 forms, so they live here and the v20 locks stay untouched. `c016/pipeline/`
does the same thing for its own commit.

### Note on `weights-dataset/source/`

Unlike the v20 and C016 kernels, `kernel/infer.py` is not self-contained. At
run time it does:

```python
sys.path.insert(0, str(run4_root / "source"))
from limited_query_cv.transform_v5_batch3_highcap_structural import structural_bank
from limited_query_cv.v5_batch3_highcap_path import HIGHCAP_CONFIG, highcap_path_candidates
from limited_query_cv.v5_batch3_run2_surface import SurfaceWell, ...
```

That snapshot ships inside the Kaggle weights Dataset, so it is packaged here
verbatim: 23 `rogii` modules and 6 `limited_query_cv` modules, one of which is
the `__init__.py` that makes the snapshot a regular package. 28 of the 29 are
byte-identical to the repository modules at the build commit; that
`__init__.py` exists only in the snapshot. Keep this tree and
`pipeline/limited_query_cv/` on separate paths — the snapshot's `__init__.py`
would otherwise shadow the pipeline's namespace package.

The kernel also re-validates the recipe before predicting and aborts unless it
still has 5 outer records, 18 candidates, 409 features, and scale `0.6`.
`reproduce_run4.py verify` asserts that same shape.

## Verification

```bash
python run4/reproduce_run4.py verify
```

Checks, in order:

- all 105 hash-locked sources;
- the recipe shape the kernel enforces, its own sha256 against the recorded
  production value, and `recipe["source_sha256"]` against the actual packager
  `prepare_v5_batch3_run4_sr_deployment.py` — the recipe records the hash of
  the script that wrote it;
- the frozen kernel hash and the presence of the three snapshot modules it
  imports;
- the C016 parent hash pinned in the candidate bank, cross-checked against the
  `c016/` subtree;
- all ten routers against the per-outer hashes in the recipe;
- the deploy-parity OOF reconstruction, both fold vectors, the gain over C016,
  and the stored C016 and v20 baselines.

Routers and OOF archives default to this workspace and are skipped with a
status note if absent. Override with `ROGII_RUN4_ROUTER_ROOT`,
`ROGII_LIMITED_QUERY_ROOT`, or the matching command-line arguments.

```bash
python run4/reproduce_run4.py score-oof     # OOF metrics only
python run4/reproduce_run4.py list-recipe   # routers, blends, per-fold gains
```

## Rebuilding Run4 from C016

Run from `run4/pipeline` with it on `PYTHONPATH`; run artifacts land under
`run4/pipeline/limited_query_cv/runs/…`.

```bash
python -m pip install -r run4/requirements.txt
cd run4/pipeline
export PYTHONPATH=$PWD
R=limited_query_cv/runs/v5_batch3_run_004_goal_020
R1=limited_query_cv/runs/v5_batch3_run_001_goal_020
R2=limited_query_cv/runs/v5_batch3_run_002_goal_020
```

**0. Rebuild the candidate bank.** Skip this only if you already hold the run-1
and run-2 caches inventoried in `manifests/run4_artifacts.json`.

```bash
# v20 component paths, folds 0-3 (the fold-4 cache comes from the run-25 audit)
for fold in 0 1 2 3; do
  python limited_query_cv/cache_v5_run3_v20_components.py --fold "$fold" \
    --output limited_query_cv/runs/v5_batch2_run_003_goal_050/v20_components/fold$fold.npz
done

# 12 run1_struct_* candidates: cache, combine, transform, name
python limited_query_cv/cache_v5_batch3_highcap_path.py --help
python limited_query_cv/cache_v5_batch3_datum_path.py --help
python limited_query_cv/combine_v5_batch3_structural_caches.py \
  --self-cache <self.npz> --datum-cache <datum.npz> --output <combined.npz>
python limited_query_cv/transform_v5_batch3_highcap_structural.py \
  --input <combined.npz> --output <highcap.npz>
python limited_query_cv/combine_v5_batch3_named_structural_caches.py \
  --output $R1/f0_highcap_structural/shard0.npz

# 6 surface_raw_* candidates
for fold in 0 1 2 3; do
  python limited_query_cv/cache_v5_batch3_run2_surface.py --fold "$fold" \
    --output $R2/surface_f$fold\_cache.npz
done
```

The shard layout is per-fold and sharded; run `--help` on the two `cache_…path`
scripts for their fold and shard arguments before launching a full rebuild.

**1. Well statistics and router features** — both read the hash-pinned C016
parent prediction.

```bash
python limited_query_cv/prepare_v5_batch3_run4_clean_well_stats.py \
  --output $R/clean_well_stats.npz
python limited_query_cv/prepare_v5_batch3_run4_router_features.py \
  --well-stats $R/clean_well_stats.npz --output $R/router_features.npz
```

**2. Nested physics candidates.** One invocation per role; outer roles predict
the held fold, pair roles support selection without touching it.

```bash
python limited_query_cv/train_v5_batch3_run4_physics_nested.py \
  --role outer_0 --training-folds 1 2 3 4 --predicted-folds 0 \
  --output-dir $R/uniform_nested_physics
```

**3. Select and assemble the uniform nested prediction.**

```bash
python limited_query_cv/select_v5_batch3_run4_physics_nested.py \
  --root $R/uniform_nested_physics \
  --prediction-output $R/uniform_nested_physics/prediction.npz \
  --selector-output $R/uniform_nested_selector_preregistration.json \
  --manifest-output $R/uniform_nested_physics/manifest.json
python limited_query_cv/assemble_v5_batch3_run4_uniform_nested.py \
  --physics-root $R/uniform_nested_physics \
  --physics-prediction $R/uniform_nested_physics/prediction.npz \
  --physics-selector $R/uniform_nested_selector_preregistration.json \
  --output $R/uniform_nested_outer_predictions.npz \
  --manifest-output $R/uniform_nested_outer_manifest.json
python limited_query_cv/verify_v5_batch3_run4_uniform_nested.py \
  --physics-root $R/uniform_nested_physics \
  --physics-prediction $R/uniform_nested_physics/prediction.npz \
  --physics-selector $R/uniform_nested_selector_preregistration.json \
  --physics-manifest $R/uniform_nested_physics/manifest.json \
  --outer-prediction $R/uniform_nested_outer_predictions.npz \
  --outer-manifest $R/uniform_nested_outer_manifest.json \
  --record-output $R/uniform_nested_verification.json
```

**4. Train the row surface routers** (one per development fold) and develop the
boosted well-utility router.

```bash
for fold in 0 1 2 3; do
  python limited_query_cv/train_v5_batch3_run4_row_surface_router.py \
    --held-fold "$fold" --device cuda \
    --well-stats $R/clean_well_stats.npz \
    --output $R/row_surface_router_f$fold.npz \
    --model-output $R/row_surface_router_f$fold.cbm
done
python limited_query_cv/develop_v5_batch3_run4_boosted_router.py \
  --well-stats $R/clean_well_stats.npz --features $R/router_features.npz \
  --route-output $R/clean_boosted_well_utility_router.npz \
  --result-output $R/clean_boosted_well_utility_router_development.json
python limited_query_cv/evaluate_v5_batch3_run4_uniform_deployable.py \
  --well-stats $R/clean_well_stats.npz --features $R/router_features.npz \
  --route-output $R/uniform_deployable_physics_router.npz \
  --result-output $R/uniform_deployable_physics_router_development.json
```

**5. Package the deployment.** This writes `recipe.json`, the ten routers, and
`oof_prediction.npz`, and stamps its own sha256 into the recipe.

```bash
python limited_query_cv/prepare_v5_batch3_run4_sr_deployment.py \
  --output-dir $R/sr_deployment_scaled06 \
  --uniform-prediction $R/uniform_nested_outer_predictions.npz
```

## Regenerating the submission

`kernel/infer.py` needs three Kaggle inputs: the v20 weights Dataset, the C016
weights Dataset, and `jiweiliu/rogii-v5-run4-structural-surface-router` (the ten
routers, the recipe, and `source/`). On Kaggle it produced 14,151 rows on dual
T4s; locally the dual-GPU path took 91.1 s against 106.7 s single-GPU, with
dual-vs-single prediction RMSE of `2.58e-09` ft
(`kernel/deployment-validation.json`).

## Dependencies

`run4/requirements.txt` extends the v20 set. Run4 additionally needs `joblib`
(the structural routers are joblib pickles), `xgboost` and `lightgbm` (routers
and the master-frame booster), and — for the SegFormer candidates in the run-6
trajectory chain only — `transformers`, `torchvision`, and `opencv-python`.

### Pretrained backbone

`train_v5_run6_pretrained_categorical_segformer.py` and
`train_v5_run6_trajectory_categorical_segformer.py` load `nvidia/mit-b0` from
the local HuggingFace cache. It is the one external pretrained weight in the
chain; `manifests/run4_artifacts.json` records it under `pretrained_backbone`.
Set `HF_HOME` or pre-populate the cache before an offline rebuild.

## External artifacts

Nothing large is committed. `manifests/run4_artifacts.json` records sha256 and
this workspace's location for the ten routers (~34 MB), the candidate bank
(40 run-1 structural shards, ~73 MB; five run-2 surface caches, ~42 MB; five
selector caches, ~61 MB), the v20 component caches, the C016 safe-mode bank and
ridge archives, the uniform nested outer predictions, the router feature cache,
and the recorded OOF archive. Every one of these is now rebuildable by the code
in `pipeline/`.

## Provenance limits

1. The routers exist only in the private Kaggle Dataset, the run directory, and
   the deploy directory of `rogii_v20_limited_query`; there is no vault copy as
   there is for v20.
2. Retraining reproduces the recipe, the routing contract, and the fold logic.
   As with v20 and C016, byte-identical model artifacts are not promised across
   CUDA, cuDNN, PyTorch, XGBoost, CatBoost, or GPU versions — and the structural
   routers are `joblib` pickles, so they are additionally sensitive to the
   scikit-learn version that wrote them.
3. `records/` is the decisive audit trail, not the complete run directory. The
   full 15-role nested tree, the screening banks, and the diagnostic candidates
   stay in `rogii_v20_limited_query`.
4. Only one v20-component builder survives in the source history. The
   `v5_batch2_run_001_goal_050` caches, which the anchor-posterior path reads,
   were written by an earlier form of it: identical component names, weights,
   and parent checkpoint hashes, and an identical v20 baseline, but component
   values differing by up to `0.000977` ft against the `v5_batch2_run_003`
   caches the surviving builder writes. That is GPU-level rerun drift, not a
   different recipe, and it is the one place where a rebuild cannot be
   byte-identical to the historical input.
5. Run4 is the current frozen parent in `frozen_previous_best_v5.json`. Anything
   promoted after it is outside this repository.
