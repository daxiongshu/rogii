# Run15 source review

## Clean boundary

Run15 starts after the active
`limited_query_cv/v5_batch2_purece_strat_lineage_invalidation.json`
quarantine. It must not load any prediction, feature, route, router, parent,
or ensemble transitively derived from `purece_strat_clean.npz`. In
particular, every Run-10 through Run-13 A960-derived artifact is forbidden.

The only prediction parent is the immutable protected-v20 OOF archive:

`/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz`

The legal per-well inputs are the inference-visible horizontal columns and
typewell `TVT`/`GR`.

## Development observation

On F0--F3, a label-separated diagnostic calibrated typewell GR to horizontal
GR using only the visible prefix, then compared the frozen v20 path and true
path on the hidden tail. The true path had lower normalized GR mismatch on
approximately 94% of wells. This establishes recoverable test-time signal,
but it does not prescribe an optimizer and is not promotion evidence.

## Primary mechanism

The first family treats the correction to v20 as a latent state sequence.
At every native tail row it evaluates typewell GR at a registered bank of
TVT offsets around v20. Multi-scale robust mismatch is the unary energy.
A dynamic program starts at zero correction, restricts row-to-row state
motion, and penalizes correction variation. The decoded path is smoothed,
re-anchored exactly at the final visible row, clipped, interpolated only
inside the well, and blended with v20.

This differs from the rejected Run-6 Viterbi family in three material ways:

1. costs are evaluated at native horizontal rows instead of the 640-column
   alignment tensor;
2. several observed/typewell GR smoothing scales contribute directly to the
   unary;
3. the transition is measured in TVT correction per native row and can use
   asymmetric/curvature follow-ups rather than a sampled-column jump alone.

## Follow-ups

If the primary decoder exposes useful but biased directions, Run15 may fit a
well-grouped supervised selector or residual sequence model from the same
inference-safe cost surface. All target-derived tuning remains restricted to
F0--F3, and every retained output must exclude its held well fold.

The first rowwise supervised cost-surface implementation memorized its
training wells and failed all three registered held-F0 blends. The next
prospective extension therefore keeps the complete-well categorical
representation intact: initialize the original-fold-safe protected base-12
model, synthesize query-specific complete wells around the protected-v20
path using only the query's visible inputs, and adapt controlled parameter
subsets. Its horizon is selected from synthetic validation alone. This is
not allowed to import the rejected booster output or any quarantined
purece-strat/A960 lineage.

## Diverse tracker-row extension

The locally cached Ravaghi LightGBM notebook contributes a materially
different hypothesis: predict the entire continuation from a bank of
independent physical trackers rather than regress a small correction from a
single GR cost surface. Its useful legal mechanisms are ANCC- and Z-state
particle filters, constrained typewell-GR beam paths, correlations against
the well's own visible horizontal-GR prefix, trajectory derivatives, and
multi-scale GR probes.

Run15 may use the notebook's CSV cache only for a nonpromotable F0 capacity
screen after excluding every formation-derived, cross-well spatial, public
prediction, and fitted-model field. It may not load public OOF predictions or
weights. Any positive family must be rebuilt from the organizer CSVs, with
feature invariance and lineage checked again, before clean F0--F3 expansion.
The detailed prospective contract is frozen in
`tracker_row_registration.json`.

## Visible-backtest decoder extension

The positive A009 decoder result motivates a legal same-query curriculum
rather than another cross-well row regressor. At inference the target well
supplies genuine TVT labels through the organizer cut. Earlier cuts inside
that visible prefix can therefore form real artificial forecasting tasks
without touching the hidden suffix. Run15 may update only the protected
base-12 decoder, normalization, and head on those tasks, select its snapshot
from disjoint visible-prefix backtests, and optionally finish from that state
on the already gated protected-v20 forward-extreme simulator.

The two learning rates, cuts, snapshots, target-free selectors, routes, and
capacity thresholds are fixed in
`visible_backtest_decoder_registration.json`. The same 12 F0 IDs remain a
capacity diagnostic only; no same-held temperature or blend can promote.
The implemented source is frozen separately in
`visible_backtest_decoder_source_registration.json` before any gate
invocation, optimizer update, or target-derived capacity metric.

Before the A013 capacity jobs completed, Run15 also froze a metric-aligned
follow-up in `metric_aligned_backtest_registration.json`. It keeps all
adaptation and validation labels inside the organizer-visible prefix, but
selects objective, rate, snapshot, and temperature per query by native-row
path RMSE on disjoint artificial continuations. This prospectively tests
whether categorical CE is the wrong early-stopping geometry rather than
tuning a selector after seeing the true hidden-tail result.
