# V5 batch-2 run-4 source review

## Boundary and target

The user renewed development and explicitly encouraged a joint result from
several models or approaches. The durable goal record fixes the requested
pooled F0-F3 RMSE gain at `+0.50`, the protocol floor at `+0.20`, and the
effective readiness target at `+0.50`.

All new target-derived development is restricted to original folds F0-F3.
Current-run F4 targets, errors, metrics, ranks, selector outcomes, and
target-derived summaries remain sealed. The remaining protected batch-2
joint reveal is unspent. A competition submission is not authorized.

The locked F0-F3 v20 reference is:

- 619 complete wells and 3,038,588 hidden-tail rows;
- pooled RMSE `5.940191010295828`;
- required final batch-ensemble RMSE at most `5.440191010295828`;
- v20 OOF prediction SHA-256
  `ee94cbc42d9378514fa8d3e8cb23f9a42fea999c7ec457009c99fbc65fb9ebe0`.

## Competition and baseline

The inference contract is horizontal `MD, X, Y, Z, GR, TVT_input` and paired
typewell `TVT, GR`. Horizontal `TVT` is the supervised target. Horizontal
formation columns and typewell `Geology` are train-only; they may be auxiliary
training targets but never inference inputs.

V20 is already an 85-checkpoint stack dominated by compact categorical 2-D
alignment U-Nets. Its state images compare horizontal and typewell GR over
candidate TVT offsets, and its members vary state range, synthetic curriculum,
capacity, posterior decoder, and inference-safe postprocessing. Therefore
more checkpoints from the same categorical architecture are not a genuinely
new mechanism.

Historical V5 work has also falsified protected-path selectors, small
postprocessors, component reweighting, short checkpoint continuation, a
global best-of-K surface head, low-frequency residual bases, Bayesian
smoothing, candidate/segment rankers, and direct neighbor-profile transfer.
Run 4 does not import any post-v20 prediction, calibration, weight, or model.

## Competition-scoped sources

The local Kaggle cache was queried only for
`rogii-wellbore-geology-prediction`.

- Discussion 700424 records that a competitive complete-well U-Net scores the
  competition among candidate depth rows rather than a pointwise regression.
  Its follow-up distinguishes a signed-distance-field target, which ranks all
  candidate matches, from a single categorical path label. It also points to
  legal trajectory channels and synthetic-well pretraining.
- Cached public kernel `hengck23/cnn-sdf-example` demonstrates the concrete
  signed-distance formulation: typewell GR and horizontal GR form a 2-D image,
  the visible path is a history channel, and a SegFormer predicts the signed
  distance to the full path. It is a mechanism template only; its checkpoint,
  split, and target-derived artifacts are not used.
- Discussion 699853 and the referenced geosteering inversion material motivate
  multi-hypothesis, complete-path inference and learned correlation rather than
  committing to the lowest raw-GR mismatch.
- Discussion 698825 supplies the organizer's domain priors: lateral GR can
  correlate with itself, nearby wells can share formation dip, and trajectory
  direction matters.
- Discussions 702474 and 726465 distinguish full-curve models from row-wise
  models. The latter contains an independent report of sub-5 pooled,
  grouped five-fold CV from a complete-well model using the test-time contract
  and no neighbor wells. The score claim is not accepted as evidence here.
- The organizer-recognized Working Note summary in discussion 727171 says the
  recoverable error is mainly low-frequency datum/trend while trajectory gives
  much of the high-frequency shape. This motivates derivative/path and
  auxiliary-surface models rather than another residual selector.
- The public `LWD_inversion` and `inversion_school_geosteering` repositories
  motivate a regularized inverse solution with a forward GR operator,
  multistart initialization, and sequential or smoothness priors.

Every source is treated as a hypothesis generator. Only V5 grouped
development and, if readiness passes, the one sealed joint reveal can supply
promotion evidence.

## Registered mechanisms

### 1. Dense signed-distance alignment

Predict the signed distance from every candidate TVT row to the complete path,
then decode the zero contour. This supervises every state row and preserves
alternative depth rankings instead of reducing each column to one class.
SegFormer and residual U-Net variants operate on legal GR, visible-history,
support, and trajectory channels. Synthetic paths and real randomized-prefix
examples are both used.

Smallest falsifier: fixed-batch SDF memorization followed by one F0 role. The
family must beat its categorical/no-SDF control and contribute positively to
v20 before four-fold expansion.

### 2. Learned metric cost-volume path model

Learn separate multiscale embeddings for lateral and typewell GR, form a dense
cosine/difference cost volume, and decode a globally smooth path. Contrastive
alignment and path losses teach which GR structures are transferable; shuffled
GR and identity-metric controls test whether the learned metric carries signal.

Smallest falsifier: positive-pair retrieval on a fixed training batch and a
selection-clean F0 role gain over the fixed raw-GR metric.

### 3. Surface-derivative sequence model

Predict a distribution over low-frequency surface derivative and curvature,
integrate it from the last visible anchor, and reconstruct
`TVT = surface - Z`. Inputs include full-well trajectory derivatives, several
GR scales, typewell-relative likelihood summaries, and the visible prefix.
Auxiliary dip-sign and changepoint heads constrain the path. This is a direct
complete-curve model, not a residual selector around v20.

Smallest falsifier: memorize derivative/path targets on fixed wells and beat
carry plus a trajectory-only control on one held development fold.

### 4. Cross-well auxiliary-surface graph model

Build a fold-purged graph of training wells using legal XY geometry,
trajectory direction, typewell-GR similarity, and visible-prefix surface.
Neighbor paths are anchor-aligned and encoded as basis functions; graph
attention predicts a conservative whole-query surface correction. Training-only
formation surfaces can supervise auxiliary heads but cannot enter the input.

Smallest falsifier: recover planted neighbor fields, preserve an exact
zero-neighbor no-op, and add positive held-well contribution beyond v20 and a
fixed distance-weighted transfer control.

## Staged decision

1. Pass the implementation control for each trainable or training-free family.
2. Evaluate at least three meaningful variants in short F0-F3 roles.
3. Expand only mechanisms with positive incremental v20 contribution.
4. Form a selection-clean four-fold joint ensemble with no candidate weight
   above `0.20`.
5. Open the two registered F0-F3 confirmation regroupings only after the
   primary ensemble reaches `+0.50`.
6. Freeze one automatic pipeline before any F4 label use or sealed nested run.
7. Push a private Kaggle Dataset and private offline inference kernel only if
   the frozen clean ensemble passes every V5 promotion gate. Never submit.
