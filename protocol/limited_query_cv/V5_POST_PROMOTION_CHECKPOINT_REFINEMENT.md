# V5 post-promotion checkpoint refinement amendment

## Status and scope

This amendment is active for V5 batch 2 and later. It refines the
selection-clean fold-ensemble production amendment without weakening the
official promotion gate.

V5 batch 1 and its result remain immutable. Revision 2's requirements for
five fold contributions, no full-data refit, inner-only clean checkpoint
selection, checksum manifests, and fold-ensemble test inference remain active.
This amendment supersedes only revision 2's absolute prohibition on packaging
a same-held-fold-selected snapshot.

## Two checkpoint views

Every checkpoint-selecting family must freeze two logically separate views:

1. **Selection-clean view.** For outer fold `h`, select the checkpoint using
   only the other four folds. These five checkpoints produce the official OOF
   result and are the only evidence allowed to pass the ensemble gate.
2. **Post-promotion production view.** Only after the family passes the clean
   gate, fold `h` may select one snapshot from a small, preregistered window
   using its own validation metric. These five snapshots may be packaged for
   Kaggle.

The post-promotion score is optimistically selected and must be labeled
`production-parity diagnostic`. It cannot alter acceptance, family order,
blend weight, or any other research decision.

## Frozen snapshot window

Before any protected outer target is opened, each family must register:

- an ordered snapshot grid;
- the checkpoint-selection metric and direction;
- an exact tie-break, with the earliest snapshot as the default;
- whether post-promotion refinement is enabled.

For fold `h`, inner CV selects a clean snapshot from the ordered grid. The
held-eligible window contains that snapshot plus its immediate predecessor and
successor, when present. The window therefore contains at most three
snapshots and is determined without fold `h`.

The outer model is trained on the other four folds. Every snapshot in the
eligible window, its hidden-truth-masked prediction for fold `h`, and its
metadata must be checksum-frozen before the joint protected reveal. The
training code, data, seed, and snapshot bytes may not change after the reveal.

## Conditional held-fold selection

The evaluator first computes the selection-clean ensemble result and applies
the registered V5 gate. For a rejected family, no held-selected production
checkpoint is authorized and the best-of-window metric is not reported as a
candidate result.

For a family that passes:

1. fold `h` selects the best frozen snapshot in its at-most-three-snapshot
   window using the preregistered checkpoint metric;
2. exact ties choose the earliest snapshot;
3. the selection-clean fold weight, decoder, augmentation, postprocessing,
   architecture, and seed remain unchanged;
4. the five held-selected snapshots become the production checkpoint set.

Held-fold selection may choose only the checkpoint epoch or training step. It
may not choose an architecture, configuration, seed, augmentation,
temperature, decoder, postprocessor, ensemble weight, or blend policy.

If an eligible snapshot or prediction fails its checksum or completeness
audit, the refinement is invalid for that fold and the clean checkpoint is
packaged. Retraining or regenerating a replacement after the reveal is
forbidden.

## Scoring and deployment

The official CV score is always the selection-clean five-fold OOF score.
Production-parity OOF may be reported after clean promotion, but must show:

- the clean official score;
- the optimistic production-parity score;
- the selected snapshot for every fold;
- the size of each eligible window;
- a warning that the production-parity score is not an unbiased estimate.

Kaggle test inference retains the revision-2 formula:

```text
fold_prediction_h =
    (1 - clean_weight_h) * protected_v20_prediction
    + clean_weight_h * held_selected_candidate_prediction_h

production_prediction = mean(fold_prediction_0, ..., fold_prediction_4)
```

No full-data model is trained. A normal accepted trainable family still
deploys five outer checkpoints, not the inner-development checkpoints.

## Interpretation

Using fold `h` to select its epoch is legitimate use of labeled training data
for final competition production, but it is leakage relative to that fold's
CV estimate. It does not put fold `h` into that model's gradients and is not
equivalent to training one model on all labeled wells.

The clean gate answers whether the family transfers. The conditional
held-fold step is a tightly bounded production heuristic intended to improve
early stopping after transfer has already been demonstrated.
