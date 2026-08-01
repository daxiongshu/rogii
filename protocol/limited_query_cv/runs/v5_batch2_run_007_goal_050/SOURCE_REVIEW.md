# V5 batch-2 run-7 source review

## Competition mechanism

The legal prediction problem is a complete-well inverse problem.  The hidden
suffix contains roughly 0.9k--10.1k rows per development well (median 4.9k),
while the paired typewell contains roughly 0.7k--10.0k reference rows.  The
query trajectory and all horizontal GR values remain visible after the
prediction start.  Only horizontal `TVT` is hidden.

The useful physical decomposition is

```text
U(MD) = TVT(MD) + Z(MD)
TVT(MD) = U(MD) - Z(MD)
```

The known query trajectory therefore supplies the high-frequency wellbore
motion exactly.  A model must infer the smoother geological surface `U`, while
GR and the paired typewell constrain its stratigraphic phase.

## Public hypotheses reviewed

- Discussion 726465 reports whole-well grouped CV below 5 ft from one
  non-tabular model whose sample is a complete well and which uses only that
  well plus its typewell.  No architecture, prediction, weight, split, or
  checkpoint was disclosed or imported.
- Discussion 717573 reports that ordinary LightGBM, CatBoost, and XGBoost can
  reach the 5.x CV band when supplied the right inputs.  It specifically says
  the estimators and direct target are ordinary and suggests auxiliary
  future-related targets.  No feature array, model, prediction, or fitted
  constant was disclosed or imported.
- Discussion 700424 emphasizes raw full-well trajectory fields, direction,
  segmentation, typewell identity, complete context, and segment dropout.
  The source is used only to motivate input representation.
- Discussion 712037 and the Working Note Award announcement decompose most
  error into low-frequency datum/trend/shape rather than the trajectory
  wiggle.  The public oracle values and fitted coefficients are not used.
- The organizer task brief says the pre-start horizontal GR can be a
  higher-resolution reference than the paired vertical typewell and that
  future trajectory is visible.

These are hypothesis generators.  Run 7 imports no public prediction, target,
feature matrix, validation fold, model binary, fitted weight, or leaderboard
choice.

## Local evidence and non-duplication

- Frozen Run-6 component D570 is the only stable clean parent.  It improves
  pooled F0--F3 RMSE from 5.940191010 to 5.820599496 (+0.119591515), improves
  all four folds, and supplies at most one fixed parent slot.
- Run-6 residual TCN/Transformer and recurrent refiners compressed an existing
  V20-centered likelihood volume.  They did not directly cross-attend the raw
  full lateral to the raw full typewell.
- Run-6 master-frame boosters used row-local and summary features.  They did
  not expose a fixed wide raw sequence context to each training row.
- Run-6 direct-surface boosters added absolute coordinates but remained
  row-local.  Run-7 boosting instead supplies raw complete-well context and
  phase-relative reference windows.
- Run-6 random-cut refinement changed a protected tracker.  Run-7 auxiliary
  training learns legal future GR/trajectory representations before the TVT
  head and does not inherit a target-trained held-fold tracker.

## Smallest falsification controls

1. Cross-attention must memorize two fixed wells, change untouched
   predictions, remain invariant to every hidden-TVT replacement, and beat
   typewell-permuted attention on a synthetic aligned trace.
2. Wide-context boosting must recover a constructed future response, remain
   invariant to hidden-TVT replacement, and differ from a shuffled-context
   control before opening F0.
3. Auxiliary pretraining must reduce its legal masked/future loss, preserve
   target firewalls, and improve a frozen training-only TVT reconstruction
   control over the identical randomly initialized backbone.

Each family receives one preregistered F0 capacity screen.  It expands
unchanged only above its frozen threshold.  Same-held F0 recipes are
diagnostic and cannot select F0.

