# Run 25 source review

Run 25 rebuilds the invalid Run 17 C016 lineage without using the
quarantined `purece_strat_clean.npz` prediction or any stratified-parent
checkpoint.

The first replacement family reuses four already-frozen Run 6 categorical
continuations whose parent checkpoints and continuation training both follow
the original whole-well fold:

- `wide_supervised`;
- `forward_distractor`;
- `base12`;
- `grpair_base8`.

For held development fold `h`, every continuation checkpoint was initialized
from an original-fold checkpoint named `fold{h}`, trained further on the
other three original development folds, and predicted `h`. The evaluator
requires that exact training-fold record, re-hashes every parent checkpoint
against the protected v20 checkpoint manifest, rejects any path containing
`strat`, and re-hashes every prediction cache.

The frozen selector compares family, snapshot, temperature, and conservative
v20-residual weight using only the other three folds. It applies the selected
recipe to the held fold once. Four family-specific nested predictions are
retained as diagnostics, while the jointly selected prediction is the
candidate D072 replacement.

No F4 target, confirmation regrouping, public-leaderboard result, or
post-Run17 prediction participates in this stage.
