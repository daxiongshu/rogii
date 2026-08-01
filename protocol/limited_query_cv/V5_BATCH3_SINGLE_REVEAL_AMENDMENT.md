# V5 Batch 3 single-reveal amendment

## Authorization

On 2026-07-30 the user explicitly authorized:

> Authorize V5 Batch 3 with one protected reveal for
> `c016_residual_gr_path_pool`, development goal `+0.20`. Do not submit.

The exact durable authorization is
`limited_query_cv/v5_batch3_authorization.json`.

## Scope

This amendment extends the V5 protected budget from two to exactly three
batches. Batch 3 may spend at most one joint protected outer-OOF reveal, and
only for the `c016_residual_gr_path_pool` family or its preregistered final
ensemble with the promoted clean C016 parent.

The requested and effective F0-F3 development-readiness target is `+0.20`
pooled RMSE over protected v20. Smaller variants may accumulate under the
existing removal-necessity rules, but the final frozen Batch 3 candidate must
clear the full readiness gate before the reveal.

## Unchanged safeguards

- Manual target-derived development remains confined to F0-F3.
- F4 stays behind the prospective manual-analysis firewall.
- No fold-pilot reveal is authorized.
- The complete ten-pair plus five-outer frozen procedure remains required.
- Every outer prediction must be complete and checksummed before the single
  joint reveal.
- A negative F4 gain vetoes promotion.
- No recipe may change after the reveal.
- Kaggle Dataset and kernel publication still require separate authorization.
- No Kaggle competition submission is authorized.
