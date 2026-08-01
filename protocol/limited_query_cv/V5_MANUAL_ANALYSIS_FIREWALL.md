# V5 manual-analysis firewall and symmetric-pair rerun amendment

## Status and scope

This amendment is active for V5 batch 2 and later. It advances the active
protocol to revision 4 without changing V5 batch 1, either earlier amendment,
or the protected v20 baseline.

The designated development folds are `F0`, `F1`, `F2`, and `F3`. The
designated prospective audit fold is `F4`.

`F4` is not historically pristine: its labels and errors were examined during
the research that produced v20 and during earlier validation eras, including
V5 batch 1. This amendment therefore does not claim a brand-new statistically
independent holdout. It creates a forward-only manual-analysis firewall: from
this amendment onward, new candidate development may not inspect target-derived
`F4` evidence before the candidate's protected reveal.

## Unrestricted development inside F0-F3

The agent may iterate as many times as useful on whole-well cross-validation
restricted to `F0` through `F3`. Within those folds it may inspect and act on:

- per-well and per-row errors;
- residual plots, worst-well lists, segments, and subgroups;
- architecture, feature, target, loss, optimizer, and schedule changes;
- augmentation, curriculum, residual learning, and postprocessing;
- seeds, registered snapshot grids, automatic selectors, and ensemble rules;
- underfitting, overfitting, calibration, and error-concentration diagnostics.

Every trainable family must still pass the fixed-batch overfit control.
Inference features must still obey the train/test schema contract. Trials and
decisions must be logged, and the final mechanism must remain reproducible.

The default readiness view is four-fold development cross-fit:

```text
train F1+F2+F3 -> predict F0
train F0+F2+F3 -> predict F1
train F0+F1+F3 -> predict F2
train F0+F1+F2 -> predict F3
```

A batch is ready to consume protected evidence only when its frozen ensemble:

- improves pooled F0-F3 cross-fitted RMSE over protected v20 by at least
  `0.20`;
- improves at least three of the four development folds;
- has no development-fold RMSE regression worse than `0.03`;
- remains positive and gains at least `0.10` under each registered
  whole-well confirmation regrouping;
- receives no more than 60% of its positive SSE reduction from one
  development fold.

The `0.20` bar applies to the final batch ensemble. Smaller diverse components
may accumulate into it, but a sub-threshold component cannot trigger a reveal
by itself.

## F4 manual-analysis firewall

Before the protected joint reveal, the agent and all interactive research
processes must not receive any target-derived `F4` information for the new
candidate. Forbidden views include:

- `F4` targets, target summaries, or baseline errors;
- candidate RMSE, loss, residuals, rankings, or worst-well lists on `F4`;
- per-well, per-segment, or subgroup candidate behavior derived from `F4`
  truth;
- automatic selector choices or logs that disclose relative performance on
  `F4`;
- any aggregate, plot, exception, or failure message that can guide a recipe
  change from candidate performance against `F4` truth.

Before reveal, label-free `F4` checks remain allowed: input schema,
missingness, shape, inference runtime, prediction completeness, finite values,
and preregistered prediction-range safety checks. These checks may not compare
predictions with truth.

After the global pipeline is frozen, sealed noninteractive jobs may use `F4`
labels where the nested procedure requires them: as training data, or as an
inner validation fold for an outer context other than `F4`. Such jobs must not
expose target-derived logs, selected checkpoints, weights, or metrics until
all protected artifacts are complete and checksum-frozen. The pipeline may
not change in response to those sealed results.

## Global freeze

Before the full nested rerun, freeze one global pipeline for all outer
contexts. The freeze includes:

- legal inputs, preprocessing, architecture, capacity, and target
  representation;
- loss, augmentation, curriculum, optimizer, schedule, and seed policy;
- the ordered snapshot grid, automatic checkpoint selector, and tie-breaks;
- decoder, calibration, postprocessing, and test-time augmentation;
- candidate-to-v20 weight mode and every ensemble-selection rule;
- all error-derived heuristics discovered on `F0` through `F3`;
- source commit, environment, fold assignments, and data fingerprint.

The same code and rules run for every outer fold. No fold-specific hand edit
is allowed. A frozen F0-F3 model cache may be reused only if it is
byte-reproducible under the frozen source, configuration, seed, and data
fingerprint; otherwise it must be retrained.

## Full symmetric-pair nested rerun

Let `M = 5` outer folds and `N = 4` inner roles per outer fold. The complete
nested evaluation has `M*N = 20` logical inner roles.

The role with outer fold `Fi` and inner-validation fold `Fj` trains on the
other three folds. The reciprocal role with outer `Fj` and inner validation
`Fi` uses the same training set. Therefore one deterministic training
trajectory for each unordered excluded pair `{Fi, Fj}` can predict both
excluded folds:

```text
unique inner trainings = C(5, 2) = 10
logical inner roles    = 5 * 4    = 20
```

For example, the `{F0,F1}` trajectory trains on `F2+F3+F4`. Its prediction on
`F1` serves the `outer=F0, inner=F1` role, while its prediction on `F0` serves
the `outer=F1, inner=F0` role. Reuse is valid only when the training recipe is
identical for the reciprocal roles.

After inner selection, train one outer trajectory on the other four folds and
predict the held fold, for each of the five outer folds:

```text
outer trainings                = 5
total logical training roles   = 20 + 5 = 25
total unique model trainings   = 10 + 5 = 15
```

Saving several preregistered snapshots from one trajectory creates additional
checkpoint files, not additional training trajectories.

The sealed runner must:

1. generate both excluded-fold predictions from every unique inner
   trajectory;
2. reconstruct all 20 logical inner roles;
3. apply the frozen automatic selector separately for each outer context;
4. train and predict all five selected outer trajectories;
5. freeze every clean checkpoint, eligible refinement snapshot, prediction,
   fold-local weight, selector record, and checksum manifest;
6. confirm completeness before opening any protected metric.

## Joint reveal and promotion

All five selection-clean outer predictions are revealed together. The
concatenated five-fold OOF result remains the official V5 stability and
promotion estimate and must pass either the existing strong gate or the
existing small-diverse gate.

Because `F0` through `F3` were available for manual development, the result
must be labeled `selection-clean, development-exposed`. `F4` is reported
separately as the prospective manual-analysis audit. Promotion additionally
requires nonnegative RMSE gain versus protected v20 on `F4`; an `F4`
regression vetoes promotion even if the pooled five-fold gate passes. The
paired whole-well bootstrap on `F4` is reported as audit uncertainty but is
not an additional threshold.

After a clean promotion, revision 3's bounded, preregistered same-held-fold
snapshot refinement remains available. It may change only the production
epoch or step and remains an optimistic diagnostic, never promotion evidence.

## Production boundary

A promoted trainable family deploys the five outer checkpoints, or fewer
physical candidate files only where a frozen fold contribution has exactly
zero candidate weight. The ten inner trajectories and all inner snapshots are
research artifacts and are never packaged.

No Kaggle Dataset or kernel is authorized by this amendment. Packaging still
requires promotion and explicit batch authorization; a competition submission
always requires a separate explicit user instruction.
