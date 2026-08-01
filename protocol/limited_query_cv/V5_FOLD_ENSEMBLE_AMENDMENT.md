# CV Protocol v5 fold-ensemble production amendment

## Status and scope

This amendment is active for V5 batch 2 and every later V5 batch. It was
authorized after V5 batch 1 completed its single protected reveal.

Batch 1 remains an immutable historical result. Its preregistered full-data
production rule was never exercised because no family promoted. This
amendment may not be used to reinterpret, retune, or reconfirm batch 1.

## Checkpoint geometry

V5 uses `M = 5` protected outer folds. For outer fold `h`, the other four
folds form `N = 4` inner folds.

For each outer fold:

1. architecture, variant, seed policy, epoch or checkpoint, decoder,
   augmentation, postprocessing, and ensemble weight are selected without
   access to fold `h`;
2. a candidate is trained on the other four outer folds;
3. the selected outer checkpoint and its complete inference recipe are frozen;
4. the frozen candidate predicts fold `h` exactly once.

The held outer fold may not select a checkpoint. A same-held-fold
validation-best result may be reported only as an explicitly preregistered
diagnostic. It cannot promote a candidate, change a recipe, choose a weight,
or enter a Kaggle artifact.

## Production after promotion

An accepted trainable family packages the same five selection-clean outer
checkpoints used by protected confirmation. It does not retrain one model on
all labeled wells.

For test inference, fold `h` contributes

```text
fold_prediction_h =
    (1 - weight_h) * protected_v20_prediction
    + weight_h * candidate_prediction_h
```

and the deployed family prediction is the arithmetic mean of the five fold
contributions. Every `weight_h` and every fold recipe must have been selected
inside fold `h`'s four-fold inner pool and frozen before the protected reveal.

If a frozen fold weight is exactly zero, its logical contribution is the
protected v20 prediction. The physical candidate checkpoint may be omitted
from the Kaggle Dataset because it cannot affect inference, but the zero
contribution must remain in the five-way mean.

Inner-development checkpoints are selection artifacts only. They are never
packaged. Thus a normal accepted trainable family deploys `M = 5`
checkpoints, not `N * M = 20`; it may deploy fewer physical files only for
exactly zero-weight fold contributions.

## Why this is the V5 production estimand

This restores the production mechanism used by the original v20 kernel while
retaining V5's stricter checkpoint selection:

- protected OOF evaluates the exact fold member that would be deployed;
- fold averaging reduces model and training-set variance on unseen wells;
- there is no unvalidated full-data epoch or refit policy;
- checkpoint selection remains clean because no held outer target chooses its
  own checkpoint.

OOF cannot directly measure the additional variance reduction obtained by
averaging all five members on test data: each labeled well has only one
leakage-free outer prediction. This is the standard conservative limitation
of fold-ensemble OOF and must be stated in every promoted result.

## Packaging boundary

Every deployed outer checkpoint, fold recipe, and nonzero weight must appear
in a checksum manifest. The Kaggle kernel must reproduce the five-way mean.
Changing any fold checkpoint or recipe after the protected reveal requires a
new authorized validation boundary.
