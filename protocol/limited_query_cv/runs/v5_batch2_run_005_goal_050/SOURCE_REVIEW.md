# V5 batch-2 run-5 source review

## Boundary and objective

Run 5 targets a selection-clean pooled F0-F3 RMSE gain of at least `+0.50`
over the immutable V20 reference. The reference has 619 wells, 3,038,588
hidden-tail rows, and RMSE `5.940191010295828`, so readiness requires RMSE at
most `5.440191010295828`.

Only original-fold F0-F3 targets may be used for development. F4 target-derived
evidence, both registered confirmation regroupings, and the remaining
protected batch-2 reveal remain sealed. No Kaggle Dataset or kernel may be
created unless the frozen ensemble passes the complete V5 promotion gate, and
a competition submission is never authorized.

The source boundary is strict:

- reviewed public code may supply a mechanism template;
- no public or historical checkpoint, prediction, OOF decision, calibration,
  fitted weight, leaderboard-tuned constant, train-contact override, or hidden
  overlap answer substitution is imported;
- V20 OOF predictions are the fixed fold-safe baseline feature;
- the fixed Run-4 G004 prediction is eligible only as the checksummed,
  no-retuning legacy graph control declared in the family registration.

## Competition and baseline understanding

The test-time horizontal contract is `MD, X, Y, Z, GR, TVT_input`; the paired
typewell contract is `TVT, GR`. Horizontal `TVT` is the supervised target.
Horizontal formation columns and typewell `Geology` are training-only. At
inference, the horizontal target is visible only through the supplied prefix.

The useful physical decomposition is

`TVT = geological surface - Z`.

Trajectory supplies the high-frequency `-Z` shape for free. The difficult
part is the low-frequency geological surface: its per-well datum, gain/slope,
and occasional regime changes. V20 already ensembles 85 categorical 2-D
alignment checkpoints and is strong at local texture/path matching. Earlier
V5 runs showed that more checkpoints of the same kind, component
reweighting, likelihood smoothing, shallow path selection, direct derivative
sequence models, and simple cross-well transfer do not provide the required
step change.

An explicitly target-exposed F0-F3 anatomy diagnostic was run after Run 4
closed and before this registration. It is not eligible selection evidence
and no further metric was opened before registration. Relative to V20:

- an oracle constant correction per well reduces RMSE from
  `5.940191010` to `4.218841650`;
- an oracle affine correction in normalized measured depth reduces it to
  `3.070240799`;
- a quadratic correction reduces it to `2.467682016`;
- the affine correction removes 73.29% of baseline SSE.

This is not a deployable result. It establishes that the requested `+0.50`
exists inside signed per-well level/slope error and motivates mechanisms that
can infer those signs legally.

## Reviewed competition sources

The local `/nvidia-kaggle` cache was queried for this competition, and the
public Kaggle write-up API was used read-only where the cached discussion
contained only a summary.

### “The Wiggle Is Free, the Trend Is the Wall”

Danil Malyshev's organizer-recognized working note decomposes the target into
surface and trajectory, reports that the principal remaining uncertainty is
the smooth geological trend, and combines a neural corrector with robust
physics/postprocessing. It also documents train-contact answer overrides in
some public pipelines. Those overrides and all leaderboard-derived constants
are excluded here.

Mechanism retained: model surface residual and its low-order coefficients
directly, while giving the model the frozen V20 proposal and the known
trajectory.

### “When Better CV Scores Worse”

Radiant Allomancer's working note uses heel-adapted pseudo-typewells, multiple
bidirectional GRUs trained with prefix cuts, and an independent TVT/dip
trellis. Its error analysis says per-well low-frequency bias dominates and
that genuinely independent evidence can improve a strong particle-filter
track even when its standalone score is worse. Spatial models did not survive
leave-spatial-out controls.

Mechanisms retained: randomized legal prefix cuts, a separately decoded
TVT/dip posterior, and spatial-block controls for any regional transfer.

### STRIDE

Shrey Gandhi's public write-up treats prediction as a joint posterior over
segment length and TVT change. It reports material value from typewell
reference, persistence, drilled trend, and model diversity, while noting that
per-well oracle blending remains substantially better than deployable
blending.

Mechanism retained: preserve several posterior modes and combine evidence at
the complete-well level instead of reducing every row independently.

### SEGN

Kishore Vishal's public write-up presents a chunked autoregressive cost-map
view. It reports large gains from posterior-mean decoding and large oracle
gaps from anchor and tail-affine correction.

Mechanism retained: chunked re-anchoring, posterior-mean decoding, and an
explicit datum/slope head. No reported checkpoint or tuned constant is used.

### Public deterministic MHA/PF notebooks

The cached Canqiang notebook combines row-wise gradient boosting, a
particle/beam track, visible-prefix calibration, a formation-plane neighbor
model, and a forward-backward surface smoother. It also contains guarded
train-contact logic. Only the generic particle/trellis construction is used
as a template; contact overrides, fitted models, and leaderboard-tuned
parameters are excluded.

### Test-time adaptation discussion

Competition discussion 698002 describes self-supervised adaptation using
test-visible features. The useful legal special case is stronger than generic
adaptation: create artificial earlier cut points inside the actually visible
prefix, predict the pseudo-hidden interval, and compare it with the known
`TVT_input` values before the true anchor. That yields signed baseline error
features for the current well without revealing any real hidden-tail target.

Mechanism retained: the self-backtest residual actuator registered below.

## Registered mechanisms

### 1. Self-backtest residual actuator

For several earlier cut points inside the legal visible prefix, rerun the
frozen baseline with the later visible samples masked. Score those predictions
against the samples that remain legally visible at the true inference time.
Summarize signed level, slope, curvature, horizon, GR regime, and component
disagreement errors. A cross-fitted Ridge, CatBoost, or distributional MLP
maps those summaries to a conservative tail correction.

Smallest falsifier: pseudo-tail backtest coefficient signs must correlate with
true hidden-tail coefficient signs on training-side roles and add positive F0
gain at a registered cap. A shuffled-backtest control must not do so.

### 2. Masked multiview surface-residual sequence

Predict the entire V20 surface residual with a GRU, dilated TCN, or efficient
Transformer. Legal full-well geometry, V20 proposal, typewell likelihood
summaries, visible-prefix measurements, and component disagreement are
inputs. Masked reconstruction and artificial prefix cuts precede supervised
training. Direct path residual, datum, slope, quadratic, uncertainty, and
smoothness heads prevent the model from hiding all low-frequency error inside
one unstable derivative.

Smallest falsifier: fixed-batch memorization plus positive F0 incremental gain
over both V20 and a zero-correction control.

### 3. Heel-adapted anchor/posterior tracker

Blend the paired typewell with the legal heel segment, then preserve a
posterior over TVT and dip using a particle/beam track, a forward-backward
trellis, or a chunked autoregressive cost map. Decode posterior means and
uncertainty, and expose the independent track to the final residual ensemble.

Smallest falsifier: positive F0 incremental contribution beyond V20; raw
unadapted typewell and greedy-MAP controls must be weaker.

### 4. Regional residual field

Fit fold-purged regional fields for V20 residual datum, slope, and quadratic
coefficients from training wells. Query features are legal XY/trajectory,
typewell similarity, and visible-prefix descriptors. Variants are a robust
local polynomial/kriging model, a sparse anisotropic GP, and a graph-attention
field with an explicit no-neighbor no-op. Held wells and held target-derived
records never enter their reference store.

Smallest falsifier: positive contribution in both ordinary F0 and a
training-side spatial-block control. The family stops if its ordinary gain is
carried by spatial leakage.

## Staged decision

1. Lock schemas and pass implementation/negative controls.
2. Screen at least three meaningful variants per family on training-side
   roles and F0.
3. Expand only mechanisms with positive incremental evidence.
4. Build a selection-clean F0-F3 ensemble with every candidate weight at most
   `0.20`.
5. Open both registered confirmation regroupings only if primary gain is at
   least `+0.50`.
6. Freeze the global pipeline before any F4 target access or remaining
   protected reveal.
7. Create a private Kaggle Dataset and private offline inference kernel only
   after every promotion gate passes. Never submit.
