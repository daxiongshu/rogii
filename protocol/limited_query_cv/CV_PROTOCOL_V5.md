# CV Protocol v5: inner exploration, outer confirmation

## Authority and scope

CV Protocol v5 revision 14 is active. It keeps **v20** as the immutable
pre-V5 checkpoint/OOF baseline while new candidates are developed under a
forward-only F4 firewall.

This document defines the validation structure. Use:

- `COOKBOOK.md` for the agent workflow;
- `limited_query_cv/cv_protocol_v5.json` for exact active rules and constants;
- `limited_query_cv/cv_query_ledger_v5.json` for live runs and reveal budget;
- `limited_query_cv/frozen_previous_best_v5.json` for the active parent;
- amendments and old results only for historical audit.

Run `python limited_query_cv/verify_active_cv_protocol.py` at startup. If these
sources disagree, the machine policy governs until the inconsistency is fixed.

Revision 14 makes numeric CV goals diagnostic rather than hard cutoffs. It
supersedes only earlier numeric-gate clauses; all integrity, isolation, freeze,
and provenance rules remain active. Sealed history is never rewritten.

## Validation design

### 1. Implementation control

A trainable family must first overfit one fixed labeled batch. Record loss,
prediction range, gradient health, and the underfit/overfit diagnosis. Failure
here is an implementation or capacity result, not CV evidence, and may be
repaired freely.

A training-free postprocessor instead needs mechanism-appropriate no-op,
boundary, monotonicity, and synthetic-recovery controls.

### 2. F0-F3 development

Interactive target-derived research is restricted to whole-well CV on F0-F3.
The agent may inspect errors, residuals, wells, segments, subgroups, and failure
cases, and may tune architecture, features, loss, augmentation, optimization,
checkpoints, postprocessing, and ensemble logic.

The primary view is a four-fold F0-F3 cross-fit. A second preregistered
whole-well grouping checks stability. Once a confirmation result changes a
decision, it becomes development evidence rather than an untouched holdout.
Every trial and selection decision is logged; a trainable family normally
tests at least three meaningful variants.

Each run records:

```text
effective_readiness_target = max(prompt_goal, 0.20)
```

This target guides research effort only. A frozen candidate may advance below
it when pooled selection-clean F0-F3 gain is positive and stability evidence
justifies the protected audit. Fold signs, regrouping gains, concentration,
and bootstrap results are reported as confidence evidence, not mechanical
score gates.

The goal resolver and immutable ledger record are authoritative. Prompt
capture contains only the literal user kickstart message, is audit metadata
only, and never authorizes, invalidates, abandons, or renumbers a run. Run
numbers remain monotonic within their batch and numbering epoch.

### 3. Protected five-fold audit

F4 target-derived evidence stays hidden during development: no targets,
metrics, residuals, rankings, subgroup results, or target-selected choices.
Label-free schema, completeness, finite-value, runtime, and preregistered range
checks are allowed. F4 is prospectively protected, not historically pristine.

Before opening protected targets, freeze one global pipeline: architecture,
features, loss, augmentation, seed policy, snapshot grid, checkpoint selector,
decoder, postprocessor, ensemble rule, and all error-derived heuristics. The
same automatic pipeline runs for every outer context without hand edits.

With five outer folds and four inner roles per outer fold:

```text
logical roles          = 20 inner + 5 outer = 25
unique training runs   = 10 paired + 5 outer = 15
deployed checkpoints   = 5 outer
```

Reciprocal inner roles reuse one model trained on the other three folds. For
outer fold `h`, only the other four folds may choose the clean recipe used to
train and predict `h`. A held fold may not select its own checkpoint, decoder,
postprocessor, or weight.

After global freeze, sealed jobs may use F4 labels as training or inner
validation data for outer contexts other than F4, but all target-derived logs
and selector outcomes stay hidden until every inner and outer artifact is
complete and checksummed.

The reveal requires complete role maps, five outer prediction caches, masked
truth, frozen selectors/weights, and source/checkpoint/prediction manifests.
All five predictions are revealed together and labeled
`selection-clean, development-exposed`; F4 is reported separately as the
prospective audit.

A fold-safe v20 prediction may be used as a model input only when
preregistered.

Protected results may explain a failure but may not tune a replacement and
then reconfirm it on the same evidence. Same-held-fold results are diagnostic
only, except for the bounded post-promotion checkpoint refinement below.

## Promotion decision

Every audit reports pooled and per-fold RMSE, regrouping sensitivity, maximum
regression, weight stability, F4 behavior, and a 2,000-draw paired whole-well
bootstrap using seed `20260725`.

Legacy levels such as `+0.03`, 4/5 improved folds, `0.01` maximum regression,
10/15 positive weights, bootstrap probability `0.80`, quantile `-0.005`, and
exact F4 sign remain useful diagnostics, not automatic gates.

- **Accept:** pooled selection-clean CV beats the frozen previous best, all
  integrity rules pass, and protected evidence shows no material CV
  overfitting or transfer reversal.
- **Hold:** gain is positive but uncertainty or instability is substantial.
- **Reject:** pooled gain is not positive, integrity fails, or protected
  evidence materially reverses development.

Only Accept may be packaged. A tiny F4 regression within uncertainty may be
accepted; a material reversal may not.

Public LB is monitoring evidence, never an in-run tuning gate. Promotion and
submission do not change the parent automatically. After a submission, the
user reviews CV-LB consistency and may explicitly approve that solution as the
frozen parent for future runs. One small public regression may be approved
when clean CV is rigorous; repeated CV-up/LB-down results trigger a validation
review.

“Previous best” means the active user-approved registry record, not an
automatic CV or LB argmin.

For every completed audit, use the active parent in
`limited_query_cv/frozen_previous_best_v5.json`, frozen before reveal, and
compare it with the candidate on identical F0-F4 rows and pooled OOF:

```text
gain = previous_best_rmse - candidate_rmse
```

Positive gain is improvement. The reference may not change during the run.

## Query discipline

The ledger is the sole authority for current batches, candidate scope, and
remaining reveals. A normal joint query contains several distinct frozen
families; an explicit `promote` may audit one frozen candidate. Promotion may
register only the candidate-specific audit it needs, never reusable query
budget.

Opening a protected result spends that audit. Any materially revised pipeline
needs a genuinely new validation boundary; it may not reuse the spent result
as confirmation.

## Ensemble and production

Promotion evaluates incremental cross-fitted contribution to the
preregistered frozen parent ensemble, not standalone RMSE alone; acceptance
compares the resulting solution with that parent. v20 remains the root
baseline. Parent activation is manual: only explicit user approval after a
submitted solution's CV-LB review updates the parent registry for future runs.
A family preregisters one weight mode:

1. dense grid: `0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20`; or
2. nonnegative ridge with a preregistered regularization grid and maximum
   candidate weight `0.20`.

For each outer fold, weights are estimated only inside its four-fold training
pool under the original, balanced, and independent groupings. Their median is
rounded down to the registered grid and applied unchanged to the held fold.
Production keeps the five fold-local weights and outer checkpoints:

```text
fold_h = (1 - weight_h) * frozen_parent_h + weight_h * candidate_h
test_prediction = mean(fold_0, ..., fold_4)
```

There is no full-data refit. Inner-pair checkpoints are never packaged. A
zero-weight candidate checkpoint may be omitted. Every deployed artifact and
weight is checksummed.

OOF gives each well one excluding model, while test inference averages five
models; OOF therefore does not measure test-time fold-averaging variance
reduction.

### Bounded checkpoint refinement

An accepted checkpoint-selecting family may use a preregistered window of at
most three already-frozen snapshots: the clean inner-selected checkpoint and
its immediate predecessor and successor. The held fold may select only the
best epoch/step in that window using the preregistered metric; ties choose the
earliest.

Every eligible snapshot and held prediction is frozen before reveal.

Architecture, configuration, seed, augmentation, decoder, postprocessing, and
clean fold weight cannot change. Missing or invalid refinement artifacts fall
back to the clean checkpoint and may not be regenerated after reveal.

The clean score remains the sole promotion evidence. The held-selected score
is an optimistic `production-parity diagnostic` and cannot guide research.

### `promote` and Kaggle

`Promote this candidate` or `<score> is good enough, promote` authorizes the
complete candidate-specific workflow without intermediate permission requests:
freeze, audit, decision, uniform deployment repair, and—only after Accept—the
private weights Dataset and private inference kernel.

Hold or Reject returns `LEGIT SUBMISSION: NO` and creates no Kaggle Dataset or
kernel. Repair ordinary engineering failures automatically; never relabel a
scientific failure as engineering.

An accepted kernel must pass local/Kaggle parity, output completeness/schema,
and runtime checks. The final report includes the previous-best versus
candidate F0-F4 and pooled CV table. `Promote` authorizes neither public
artifacts nor competition submission; only `submit` spends a submission slot.

## Forbidden behavior

- random-row rather than complete-well validation;
- hidden targets, same-well future labels, or train-only inference columns;
- same-held-fold selection used as clean promotion evidence;
- target-derived F4 access before the protected reveal;
- pipeline or fold-specific hand changes after freeze;
- post-reveal tuning on the same protected evidence;
- leaderboard-tuned models, blend weights, or offsets;
- automatic frozen-parent changes after promotion or submission;
- unregistered post-v20 checkpoints or OOF predictions;
- full-data refitting where the active policy forbids it;
- reporting the best of many protected queries as unbiased CV;
- rewriting a registered goal or trial record after metrics are seen;
- treating one or two failed variants as proof that a mechanism is exhausted.
