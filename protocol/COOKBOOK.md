# ROGII competition agent cookbook

## Start here

Develop an insight-driven solution for the ROGII Wellbore Geology Prediction
competition under **CV Protocol v5 revision 14**.

> Explore freely on F0-F3. Keep new target-derived F4 evidence hidden. Freeze
> one complete pipeline, evaluate it across all five outer folds once, and
> deploy only selection-clean improvements.

**v20** is the locked pre-V5 ensemble and immutable root checkpoint/OOF
baseline. Promotion compares the resulting solution against the preregistered
frozen previous-best parent recorded in
`limited_query_cv/frozen_previous_best_v5.json`. Here, “previous best” means
that active user-approved record, not an automatically chosen score.

Read these two documents at startup:

- `COOKBOOK.md` — working procedure;
- `limited_query_cv/CV_PROTOCOL_V5.md` — full protocol specification.

Then run the consistency check below. It validates the machine policy, live
query ledger, and frozen-parent registry; consult those JSON files when the
workflow needs their exact state.

Amendments and old results are append-only history. Read them only for an
audit or ambiguity. Start with:

```bash
python limited_query_cv/verify_active_cv_protocol.py
```

If this guide and the machine policy disagree, follow the machine policy and
fix the inconsistency before continuing.

## Non-negotiable rules

1. Split by complete well, never by random rows.
2. Use target-derived evidence from F0-F3 for interactive research. Keep F4
   targets, errors, ranks, summaries, and model choices hidden until the
   protected reveal.
3. A fold may not choose the checkpoint, decoder, postprocessor, or weight
   used to predict itself. Same-held-fold scores are never promotion evidence.
4. Freeze the global pipeline and all required artifacts before revealing the
   joint five-fold result. The same automatic procedure must run for every
   outer fold.
5. Never use hidden targets, same-well future labels, or train-only columns at
   inference.
6. Never modify the protected v20 checkpoints or OOF vaults, and never import
   unregistered post-v20 or rejected artifacts into them.
7. Never use public-leaderboard feedback to select models, checkpoints,
   postprocessing, or blend weights.

The old C016 mode-bank lineage is quarantined because it mapped a stratified
checkpoint to an original fold. Use only the original-fold-safe repair in
`limited_query_cv/runs/v5_batch2_run_025_goal_050/c016_mode_leak_repair.json`;
never use a stratified-fold checkpoint as an original-fold parent.

For registered family `c016_buda_delayed_dip_cluster_surface`, BUDA and ANCC
are train-only horizontal columns. A held-well prediction may use a role-purged
BUDA surface fitted on legal training wells, but may not read held-well BUDA,
ANCC, or hidden TVT. Its clean C016 F0-F3 baseline is `5.869105573928044` and
its original `+0.20` target is `5.669105573928044`; any later candidate-specific
threshold exception must be recorded in the ledger.

## Start a research run

Take one pooled F0-F3 RMSE-gain goal from the user's kickstart prompt:

```text
requested_goal = explicit prompt value, or 0.20
protocol_floor = 0.20
effective_readiness_target = max(requested_goal, protocol_floor)
```

If the prompt contains one clear value, use it without editing policy files.
Ask only if multiple values conflict. Before viewing development targets or
metrics, allocate the run and commit its registration with the updated ledger:

```bash
ROGII_V5_DEVELOPMENT_GOAL=0.50 \
  python limited_query_cv/resolve_v5_goal.py --new-run \
  --prompt-save 'LITERAL USER KICKSTART MESSAGE'
```

The environment variable is temporary; the immutable registration JSON and
ledger are authoritative. `--prompt-save` stores only the literal user message
as `kickstart_prompt.txt` and computes its digest. It is audit metadata, not an
authorization or validation gate. Never include hidden/system prompts or
secrets. Missing or different prompt metadata does not invalidate, abandon,
or renumber a run; omit capture rather than store sensitive text. Historical
hash-only records remain valid.

Batch and run numbers are independent. Run numbers advance monotonically
within each batch and numbering epoch, even when a run is removed. A full
reset requires an explicit user instruction and a new epoch; epoch 2 began at
run 1 after the prior full purge. Example names:

```text
v5_batch2_run_001_goal_050
v5_batch2_run_002_goal_050
v5_batch2_run_003_goal_030
```

The registered goal is immutable. A later kickstart creates a new run;
candidate-specific score exceptions follow the readiness rule below.

## Data and idea sources

### Inference contract

Competition data is under `/kaggle/input`; discover the competition directory
instead of hard-coding one layout. Before modeling, record:

- train/test shared columns and training-only columns;
- dtype and missingness differences;
- the visible `TVT_input` prefix and hidden-tail mask;
- one end-to-end inference example that uses no training-only field.

Legal horizontal inputs are `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`.
Inference-visible typewell inputs include typewell `TVT` and `GR`. Horizontal
`TVT` is supervised truth, never an input. Exclude every other training-only
geology or surface column unless the test schema independently proves it is
available.

For each feature, record how its test-time value is produced. Reject it if the
answer requires hidden truth, a same-well future label, or a train-only field.

### Optional local Kaggle cache

`/nvidia-kaggle/data` is an optional idea source; missing or stale cache data
must not block research. Its SQLite databases contain multiple competitions,
so always filter by `rogii-wellbore-geology-prediction`:

```text
/nvidia-kaggle/data/discussions.db
/nvidia-kaggle/data/kernels.db
/nvidia-kaggle/data/notebooks/<competition-id>/<owner_kernel>/
```

Useful read-only commands:

```bash
cd /nvidia-kaggle/skills/nvidia-kaggle-skill

python ./scripts/discussion_db_info.py
python ./scripts/discussion_query.py \
  rogii-wellbore-geology-prediction \
  --sort-by votes --sort-order DESC --limit 30
python ./scripts/discussion_read.py DISCUSSION_ID \
  --competition-id rogii-wellbore-geology-prediction

python ./scripts/kernel_db_info.py
python ./scripts/kernel_query.py \
  rogii-wellbore-geology-prediction \
  --sort-by total_votes --sort-order DESC --limit 30
python ./scripts/kernel_read.py OWNER/KERNEL-SLUG \
  --competition-id rogii-wellbore-geology-prediction
```

For a cache-only pass, inspect `data/notebooks` and skip missing references;
do not use `--force`. `kernel_read.py` may fetch a notebook that is not cached.
Treat posts and kernels as hypothesis generators, not trusted scores or
artifacts. Reproduce mechanisms under V5 and reject leaked labels, hidden-test
information, and unverified binaries.

For a sourced idea, record its source, claimed mechanism, expected complement
to the frozen parent, inference legality, smallest falsifying control, and
difference from rejected local work.

## Validation design

### What may be inspected

On F0-F3, the agent may inspect rows, wells, residuals, worst cases, segments,
rare excursions, tail lengths, clipping, subgroups, calibration, and model
disagreement. It may use these findings to change features, architecture,
target, loss, augmentation, curriculum, postprocessing, residual learning,
checkpoint policy, or ensemble logic.

Before the reveal, F4 allows only label-free checks: input schema and
missingness, prediction shape and completeness, finite values, runtime, and a
preregistered range check. Do not expose F4 target summaries, metrics,
residuals, worst-well lists, subgroup results, or target-selected choices.

After global freeze, a sealed runner may use F4 labels for training or inner
validation in outer contexts other than F4, but must suppress target-derived
logs and selector outcomes until every artifact is complete and checksummed.
The result cannot trigger a pipeline change. This firewall is prospective:
historical work saw F4, so call it protected, not pristine.

### Nested folds and checkpoint count

There are `M=5` outer folds and `N=4` inner roles per outer fold. For example,
with F0 held out:

| Inner validation | Inner training |
|---|---|
| F1 | F2 + F3 + F4 |
| F2 | F1 + F3 + F4 |
| F3 | F1 + F2 + F4 |
| F4 | F1 + F2 + F3 |

Roles `(outer=Fi, inner=Fj)` and `(outer=Fj, inner=Fi)` have the same training
folds. With identical configuration and seed, one model for excluded pair
`{Fi,Fj}` predicts both folds and fills both roles. Thus:

```text
logical inner roles              M*N    = 20
unique excluded-pair inner runs  C(5,2) = 10
outer runs on four folds         M      = 5
total logical roles                       25
total unique training runs                15
```

The inner runs select each outer model's clean recipe. The five outer models
then predict their held folds once, and all predictions are frozen before the
joint reveal. Snapshots add files, not training runs. Inner runs are research
artifacts; a promoted trainable family normally deploys only five outer
checkpoints.

## Research workflow

### 1. Lock the root and current parent

Verify v20 manifests and reproduce its expected score only through the
authorized evaluator. Read and preregister the active frozen parent from
`limited_query_cv/frozen_previous_best_v5.json`; do not infer it from the most
recent run or submission. Record the Git commit, data fingerprint, fold map,
and prediction checksums.

### 2. Register a candidate family

Register a mechanism rather than one fragile configuration: hypothesis,
failure mode, architecture/capacity range, legal inputs, target and loss,
augmentation/curriculum, optimizer/schedule, seed and snapshot policy,
decoder/postprocessing, inner selector and tie-break, ensemble-weight mode,
and whether bounded checkpoint refinement is enabled.

A trainable family needs at least three meaningful variants. Prefer staged,
mechanism-driven searches to one large undirected sweep.

### 3. Overfit a fixed batch

Before long training, make the model memorize one fixed small batch. Record
initial/final loss, prediction range, gradient norms/nonfinite counts, metric
on the memorized and untouched batches, parameter count, and throughput.
Diagnose underfitting, overfitting, data bugs, and loss/decoder mismatch. A
failed memorization test is an implementation control, not a CV result; fix it
and rerun.

### 4. Optimize on F0-F3

Use as many development iterations as time and compute justify. Do not abandon
a mechanism after one or two weak variants, and do not inspect F4 errors.
Explore target representation, receptive field and sequence/segment model,
trajectory/prefix conditioning, physics augmentation and curriculum,
ambiguity-aware loss/decoding, capacity and optimization, EMA/SWA/fixed
snapshots, neural and non-neural models, legal test-time augmentation,
postprocessing, v20 residuals, and ensemble diversity.

Use staged budgets:

1. fixed-batch control;
2. short run on one F0-F3 role;
3. four-fold F0-F3 cross-fit for promising variants;
4. confirmation on another F0-F3 whole-well grouping;
5. one globally frozen pipeline;
6. sealed five-outer-fold nested run.

Log meaningful successes and failures. Commit reusable changes separately;
never overwrite manifests or metric records.

### 5. Control development overfitting

Use the working F0-F3 split frequently, but preregister a second whole-well
grouping and query it only at meaningful milestones. Once its result changes a
decision, it becomes development evidence. Before freeze, require a still
untouched grouping or a new preregistered regrouping.

Report sensitivity to seed, fold geometry, rare wells, tail length, excursion
sign/magnitude, and clipping. Prefer simple selectors, rankings stable across
groupings, and conservative weights. Do not select the numerical best of
hundreds of trials without stability evidence, or use F4 to break a tie.

## Development goal

Let `G` be the registered `effective_readiness_target`. It is an ambitious
research target, not a promotion cutoff. A candidate may advance below `G`
when its selection-clean pooled F0-F3 gain is positive and the evidence is
stable enough to justify the protected audit.

For development fold `k` in F0-F3, train on the other three development folds
and compute on identical hidden-tail rows:

```text
baseline_rmse  = sqrt(sum_k SSE_base,k      / sum_k rows_k)
candidate_rmse = sqrt(sum_k SSE_candidate,k / sum_k rows_k)
development_gain = baseline_rmse - candidate_rmse
```

Every development well appears once. The later 20-role procedure is the
protected evaluation, not this manual readiness score.

Report `G`, fold signs, worst-fold regression, confirmation gains, gain
concentration, and reproducibility. The former reference levels—three of four
folds positive, worst regression `0.03`, confirmation gain `+0.10`, and at
most 60% concentration—are useful warnings, not hard limits.

The goal applies to the final ensemble, not each component. Several small,
diverse models may combine to reach it. A sub-target model may join when it
has positive clean ensemble contribution, removing it hurts the frozen
ensemble, and its mechanism is materially distinct.

A user instruction such as `+0.10 is good enough, promote` ends development
and starts the normal frozen audit. It needs no threshold exception document
and never waives integrity or overfitting checks.

`Keep this gain and continue` retains the candidate as the working development
stack in the current run; it does not open F4, update the frozen parent, or
authorize Kaggle packaging.

## Freeze and reveal

Before the evaluator sees any protected target, verify and checksum-freeze:

- candidate variants and one global selector, including snapshot grid,
  tie-break, seeds, augmentation, decoder, postprocessor, and weight rule;
- run goal, readiness result, confirmation results, and every manually derived
  heuristic;
- all 10 excluded-pair runs mapped to 20 roles, all five clean outer
  checkpoints, and hidden-truth-masked prediction caches;
- target-free deployment logic for every component, outer fold, and test;
- source, configuration, checkpoint, prediction, and metadata manifests;
- at most three preregistered refinement snapshots per fold, chosen without
  that held fold;
- a ledger entry authorizing this candidate's reveal and no F4 leakage.

Commit source and preregistration before the sealed nested run. Suppress F4
training logs and selector outcomes. Only the evaluator may expose protected
truth. Prefer a joint reveal of three to five distinct families; two is the
normal minimum, while an explicit `promote` permits one frozen candidate.

The official result is concatenated five-fold OOF labeled
`selection-clean, development-exposed`. F0-F3 were development evidence; F4
is only prospectively protected. Report pooled and per-fold RMSE/gain versus
the frozen previous-best solution, F4 gain and paired well bootstrap, balanced
and independent regrouping audits, maximum held-group regression, the
2,000-draw paired well bootstrap, positive inner-weight count, and each
passed/failed clause. Bootstrap whole wells with replacement, recomputing
pooled row RMSE, using seed `20260725`.

Numeric levels such as pooled `+0.03`, 4/5 improved folds, maximum regression
`0.01`, 10/15 positive weights, bootstrap probability `0.80`, quantile
`-0.005`, and the exact sign of F4 are diagnostics, not automatic gates.

The audit decision is:

- **Accept:** pooled selection-clean CV improves over the previous best, no
  integrity rule fails, and fold/regrouping/F4 evidence shows no material
  overfitting or transfer reversal.
- **Hold:** pooled CV improves but uncertainty or instability is substantial;
  do not package, and seek genuinely new evidence without tuning on the spent
  audit.
- **Reject:** pooled CV is not better, leakage or held-fold selection exists,
  or protected evidence materially reverses the development result.

Only **Accept** proceeds to Kaggle packaging. Hold or Reject returns
`LEGIT SUBMISSION: NO` and creates no Kaggle Dataset or kernel. A tiny F4
regression within uncertainty may be accepted; a material F4 reversal may not.
Repair engineering failures automatically, but never relabel scientific
failure as engineering.

Public LB is monitoring evidence, never an in-run tuning gate. Promotion and
submission do not change the frozen parent automatically. After a submission,
the user reviews CV-LB consistency; only explicit user approval permits an
agent to update `limited_query_cv/frozen_previous_best_v5.json` for future
runs. One small LB regression may be approved when CV is rigorous; repeated
CV-up/LB-down results require a validation review.

Do not tune a failed family on its protected result or query the same evidence
again.

## Promote: finish the job

`Promote this candidate` or `<score> is good enough, promote` is one terminal
instruction for the current named or frozen candidate. It authorizes the
agent, without further permission requests, to:

1. record the user's score target as research context, if supplied;
2. complete and freeze standard validation and integrity checks;
3. register and open the one candidate-specific protected audit if needed;
4. apply the Accept/Hold/Reject boundary above and stop unless accepted;
5. for a passing candidate, repair deployment uniformly and rerun checks;
6. create/update and push the **private** Kaggle weights Dataset;
7. create/update, push, run, and debug the **private** inference kernel;
8. validate its output and give the final report below.

This command creates no reusable reveal budget and authorizes neither public
artifacts nor a competition submission. Only `submit` spends a Kaggle
submission slot.

The score target does not waive any integrity check. Fix ordinary technical
failures autonomously. Do not stop at “ready for audit” or “ready for
packaging.” If legitimacy cannot be established, finish safe diagnostics and
say why instead of asking for another exception.

### Deployment completeness

Every development component needs a target-free implementation for every
outer context and test. If one is missing before reveal, either:

1. generate and freeze it for every context; or
2. remove/zero it uniformly, rerun readiness, and freeze the new pipeline.

Never add an F4-only or fold-specific fallback. `Promote` authorizes these
repairs. If neither uniform option works, the candidate is not legitimate.
This is an engineering requirement, not a new scientific gate: do not invent
target-scored mappings, thresholds, or vetoes outside the preregistered nested
selection and promotion audit.

### Checkpoint refinement and inference

Only a selection-clean passing family may refine checkpoints. For each outer
fold, choose the best already-frozen snapshot from the clean checkpoint and
its immediate predecessor/successor using the preregistered held metric; ties
choose the earliest. This may change only epoch/step. Architecture,
configuration, seed, augmentation, decoder, postprocessing, and clean fold
weight stay fixed. If an artifact is missing or has a bad checksum, use the
clean checkpoint; never regenerate after reveal.

Call the refined score `production-parity diagnostic`: it is optimistic,
cannot change promotion, and cannot guide research. The clean score remains
the official CV and sole promotion evidence.

Package the five selected outer checkpoints, omitting a physical candidate
file only when its frozen fold weight is zero. Never package the ten inner
trajectories. Test inference is:

```text
fold_prediction_h =
    (1 - clean_weight_h) * frozen_parent_prediction_h
    + clean_weight_h * selected_candidate_prediction_h

production_prediction = mean(fold_prediction_0, ..., fold_prediction_4)
```

OOF gives each well one model that excluded it, whereas test inference
averages five models. Consequently, OOF does not measure the variance
reduction from fold averaging and is conservative in that specific respect.

Checksum every artifact. Verify local/Kaggle prediction parity, row uniqueness,
missing/finite predictions, schema, and runtime. Use both Kaggle GPUs only when
a benchmark shows a real speedup, and require multi-GPU predictions and
pseudo-test RMSE to match single-GPU inference within numerical tolerance.
Wait for the private kernel to finish before reporting success.

For every completed protected audit, compare the candidate with the active
previous-best parent recorded before this candidate's reveal. Name that
reference and use identical folds, rows, masks, and RMSE calculation. Define
`gain = previous_best_rmse - candidate_rmse`, so positive is better. Do not
change the reference during the run.

End in plain language:

```text
LEGIT SUBMISSION: YES
Previous best: <name and immutable artifact/commit>

Fold    Previous best RMSE    Candidate RMSE    Gain
F0      <value>               <value>           <value>
F1      <value>               <value>           <value>
F2      <value>               <value>           <value>
F3      <value>               <value>           <value>
F4      <value>               <value>           <value>
Pooled  <value>               <value>           <value>

Private kernel: <slug and completed status>
```

or:

```text
LEGIT SUBMISSION: NO
Previous best and fold/pooled comparison: <include when a clean audit exists>
Reason: <one concrete reason>
Next: <best recovery action>
```

Avoid protocol jargon unless the user asks.

## Records and stopping

Each family must leave a source/hypothesis note, schema audit, registration,
fixed-batch control, full inner-trial ledger, F0-F3 analysis/readiness report,
preregistration and source hashes, role/checkpoint/prediction manifests,
protected result when revealed, refinement record when used, private Kaggle
artifact record when promoted, the per-fold/pooled comparison with the frozen
previous best, and Git commits separating mechanism, trials, freeze, reveal,
and packaging.

During research, continue while a credible mechanism, useful diagnostic, and
compute remain; one or two failures are not exhaustion. Do not reveal when
pooled clean development CV is not positive, evidence indicates overfitting,
leakage occurred, the pipeline changed after freeze, artifacts are incomplete,
or no ledger entry authorizes the reveal.

After `promote`, complete repairable work. Return `LEGIT SUBMISSION: NO` only
when a clean boundary or required integrity condition cannot be recovered, or
when the protected result rejects the candidate. Preserve a rejection and do
not tune against its spent evidence.
