# V5 batch-2 run-10 source review

## Competition and metric

The competition asks for the hidden horizontal-well `TVT` trajectory from the
legal inference columns `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`, together
with typewell `TVT` and `GR`. The local protected V20 baseline is evaluated by
pooled row RMSE on whole-well folds. On the open F0--F3 development roles it
scores `5.940191010295828`, so the requested `+0.50` readiness target is
`5.440191010295828`.

No public prediction, target, selected weight, or fitted constant is imported.
F4, both confirmation regroupings, and the remaining protected reveal remain
sealed.

## Current clean parents

- C016 is a selection-clean conservative ridge parent at `5.739422742`
  (`+0.200768269` versus V20 using the recomputed archive).
- A960 is the strongest selection-clean parent at `5.660076625`
  (`+0.280114385` versus V20), with positive gains on all four open folds.
  It routes four trajectory/posterior paths using inference-safe whole-well
  geometry and GR evidence.
- The independently preregistered A980 block-utility router may enter Run 10
  only if its already-running one-shot result is positive after A960 on every
  open fold. Its result and archive hashes must be frozen before any Run-10
  final selector reads targets.

## Open-fold mechanism diagnostic

A target-exposed scalar diagnostic, used only to test mechanism and never as a
deployable recipe, shows that A960 is systematically under-actuated. Multiplying
its clean correction relative to V20 by amplitudes from 1.25 through 3 improves
the pooled score over amplitude one across a broad range. A same-held amplitude
is not selection-clean and is ineligible. Run 10 therefore tests a prospective
other-three-role scalar selector and a separate inference-feature model that
predicts amplitude per well.

This differs from rejected local work. A950 capped individual route amplitudes
before route selection, whereas Run 10 calibrates the complete already-clean
A960 prediction. A970 applied fixed target-free GR treatment weights and failed
by fold-dependent sign. A980 predicts blockwise route utility. The new trainable
family instead predicts a single bounded actuator for the final A960 correction
from legal complete-well evidence, which preserves long-range coherence and
cannot switch paths row by row.

## Frozen development strategy

1. Rehash A960, its metric-free feature lineage, C016, and every protected
   firewall flag.
2. Gate an exact held-excluded scalar amplitude bank with amplitude-one no-op.
3. Gate four materially different per-well amplitude regressors on constructed
   roles, including held-truth replacement invariance and exact no-op.
4. Run each clean F0--F3 evaluation once. Retain a family only if it improves
   A960 pooled and on every held role.
5. Freeze a target-unscored final manifest. Optionally include accepted A980
   only under its independent retention rule.
6. Fit the registered nonnegative final ridge only on the other three roles for
   each held role. Open confirmations only if the primary ensemble reaches
   `+0.50` and passes every other readiness clause.

