# V5 batch-2 run-6 source review

## Boundary

Run 6 is an independent development run for the explicit `+0.50` V5 goal.
Target-derived development is restricted to original folds F0-F3. F4, both
registered confirmation regroupings, and the remaining protected reveal stay
sealed. The immutable baseline is
`/geo/geo_new/rogii_v20_oof_vault/v20_prediction.npz`, with pooled F0-F3
RMSE `5.940191010295828`; readiness therefore requires `5.440191010295828`
or better.

## Why another postprocessor is not the primary bet

The already-recorded F0-F3 anatomy has substantial low-frequency oracle
headroom, but legal regressors of datum, slope, path choice, and historical
component weights have transferred only hundredths of a foot. That evidence
argues against spending this run on a fifth hand-designed v20 correction.
The target is a materially stronger, complete-well likelihood model.

## Local competition sources

The optional local cache under `/nvidia-kaggle/data/` was used only as a
hypothesis source.

- Discussion 726465 contains an independently reported whole-well grouped,
  inference-faithful, single-model pooled CV below 5 ft using no neighboring
  wells. It identifies each sample as a complete well.
- Discussion 700424 describes a simple U-Net whose synthetic-well pretraining
  helped, and gives the key target-design clue: typewell rows within one
  horizontal column are alternatives that should compete.
- Discussion 699853 supplies the concrete raw representation: separate
  typewell and horizontal GR planes, their difference, explicit visible-TVT
  history and mask, plus categorical or distance-field outputs. It also
  reports value from trajectory tangents, flip/stretch augmentation, and
  synthetic generation.
- Discussion 698002 reports smaller but repeatable improvement from legal
  online adaptation.
- The cached public CNN/SDF notebook is useful for shape and masking controls,
  but its dense SDF regression is not accepted as the only target. Run 4
  showed that faithful dense-SDF variants had very small incremental value.

No reported score is promotion evidence.

## Registered mechanisms

### Raw-history row competition

The model sees raw calibrated GR planes, explicit visible history, state
coordinates, and legal trajectory channels. Its principal loss is rowwise
categorical cross-entropy across typewell-depth alternatives. Dense SDF may
be a representation or auxiliary but is not the primary target.

### Hierarchical coarse-to-fine competition

A full-well coarse classifier first resolves global cycle choice. A second
high-resolution classifier operates in a crop centered on the coarse
posterior. This separates context length from TVT resolution.

Before its target-derived metric is opened, this family is prospectively
extended with a formation-surface-teacher variant. The public competition
discussion documents the useful piecewise-linear identity
`d(TVT + Z) = d(formation surface)` and a grouped auxiliary experiment that
could classify the sign of training-only formation-surface motion from legal
inputs. The reviewed public Q-3D toolkit independently reports trajectory
tortuosity as its largest grouped-CV feature ablation, although its complete
LightGBM is much weaker than V20. Run 6 therefore uses horizontal formation
columns only as training labels for coarse dip states and changepoints; at
inference the segmenter sees only legal XYZ, GR, visible TVT history,
typewell TVT/GR, and Q-3D tortuosity. No public prediction, fitted model,
weight, or target-derived constant is imported.

This differs from Run 4's failed derivative models: it predicts a small
piecewise-constant state sequence and integrates segment parameters rather
than accumulating thousands of unconstrained row derivatives. The required
shuffled-whole-well teacher control tests whether any gain comes from
transferable legal inputs rather than class imbalance.

Run-6 target-exposed F0 diagnosis then showed that the first segment regressor
learned transferable teacher signal but used only within-well coordinate
deltas. A prospective spatial-field variant therefore treats the same
training-only formation surfaces as regression targets at their legal XY
locations. At inference it queries only the new well's legal XY path, removes
the interpolated value at the final visible anchor, and restores that anchor's
observed `TVT_input + Z` surface datum. The support set is built only from the
training-side wells for each role. This is a nonparametric trained model, not
access to held-well geology; a whole-well shuffle of target tracks is its
negative control.

Before any metric for the following variant was opened, the public Working
Note Award source "The Wiggle Is Free, the Trend Is the Wall" and its public
notebook attachment were reviewed. The note reports a one-shot structural
surface postprocessor whose two independent 80/80 cross-fit directions chose
the same constants: four Tukey-IRLS iterations, polynomial degree 4,
projection share 0.75 after a 500-ft linear warm-up from the raw track, and a
51-point Savitzky-Golay smoother. Its reported proxy gain is hypothesis
evidence only. Run 6 imports no public predictions, fitted coefficients,
well split, validation labels, or target-derived selector. It freezes the
published constants and applies the deterministic transform independently to
each held well's immutable fold-safe V20 prediction in structural-surface
space `TVT + Z`.

This robust projection is registered as a geometry-only hierarchical variant:
it compresses high-frequency path jitter while retaining the exact supplied
`-Z` trajectory contribution. Its metric-free gate requires exact outer
zero-weight reconstruction, zero projection movement at the anchor end of the
warm-up, finite output, and rejection of a large synthetic surface outlier by
the standard Tukey bisquare. Because postprocessors may fail after upstream
corrections have spent the same error budget, only its incremental effect on
V20 is relevant.

After D004 showed that an all-well XY formation field mixed incompatible
geological frames, the public complete-well discussion and the public source
mechanism from the adjacent historical repository were reviewed. The
label-free master-frame rule is independently reproducible from legal
typewell `TVT/GR`: connect two profiles when Pearson correlation is at least
0.99 on at least 200 shared points of a one-foot absolute-TVT grid. Applied
without labels, this yields exactly 54 connected components, matching the
public mechanism description.

A prospective master-frame surface-transfer repair is therefore registered
before opening its target metric. For every held query it excludes the entire
held fold and every F4 reference target, restricts reference wells to the
query's legal typewell family,
and transfers fold-safe training-well `TVT + Z` curves in two ways: the three
closest complete analog paths and the 32 nearest local path points. Both use
inverse-square XY weights and estimate one bias only from the query's visible
prefix. The fixed analog/local distance limits are hypothesis-source
constants, not imported fitted weights. Candidate corrections use only the
already-registered cap and outer-blend grids. A deterministic permutation of
reference family membership is the negative control.

No historical spatial prediction, cached family assignment, confidence
model, target multiplier, correction weight, or calibration artifact is
reused. The 54-family map is rebuilt from raw legal typewell columns by the
new source.

The master-frame falsifier subsequently inverted: the eligible same-family
local transfer gained only `+0.001737` on F0, while its permuted-family
negative control gained `+0.060571`. The family hypothesis is therefore
rejected. The control is retained only as evidence that sparse distance
gating may deserve a separately registered future mechanism.

The formation-teacher implementation also left one public mechanism untested.
Discussion 699853 reports grouped OOF classification of the train-only ANCC
surface direction into `down / flat / up` from test-safe trajectory, GR, and
visible-history inputs. Run 6's first teacher implementation instead regressed
continuous segment changes directly. Before opening a new F0 metric, a
classifier-constrained repair is registered: define the three states using a
fixed +/-2-ft change threshold over the already frozen 16 hidden-tail
segments, train a three-class CatBoost on the existing legal segment
features, and reconstruct segment movement from either hard states or class
posterior means. The +/-2-ft threshold gives non-degenerate
28.5%/42.5%/29.0% development teacher-state prevalence and is fixed before
the diagnostic. Fold-training class means supply movement magnitudes; the
existing correction-cap and outer-weight grids are the only target-path
actuators. Whole-well shuffled states remain the negative control.

### Query-specific synthetic adaptation

After a global parent is selected cleanly, only inference-time-available
labels may adapt it: the true visible prefix, artificial cuts wholly inside
that prefix, and forward paths generated from the query's typewell and legal
trajectory. The hidden tail is never an adaptation label or stopping signal.

### Fixed-phase categorical expansion

A target-exposed F0 capacity diagnostic of the already registered base-16,
wide-grid categorical U-Net found that its fixed final checkpoint contributed
`+0.087827` at the registered `0.20` candidate-weight ceiling. The
same-held-fold validation-best checkpoint contributed only `+0.032820`, so it
is explicitly ineligible and is not the expansion rule.

Before opening any metric from the registered seeds, the repair is frozen as
16 forward-extreme synthetic epochs followed by an optimizer-restarted eight
real-data epochs, with snapshots at total epochs 4/8/12/16/24. Training is
full FP32 with finite-gradient assertions. The decoder grid is
0.75/1.0/1.25/1.5/2.0 posterior temperature, followed by the already
registered last-32-visible-column calibration and candidate-weight grid.
Both registered seeds are retained. For held development fold `h`, snapshot,
temperature, seed bag, and weight must be selected only from the other three
development folds. Thus the diagnostic F0 recipe cannot select itself.

The selector first chooses the registered weight minimizing pooled blend RMSE
for each seed-treatment/snapshot/temperature recipe on those three roles,
with smaller weight on ties. It then applies the registered minimum-mean-rank
rule over blend RMSE, standalone pooled RMSE, and mean worst-decile well RMSE.
Remaining ties prefer those metrics in order, smaller weight, earlier
snapshot, temperature nearest one, then the equal two-seed bag. These rules
are source-frozen before any registered-seed metric.

The mechanism is not new: it is the registered anisotropic residual U-Net
with full-field categorical competition on the registered +/-120-ft grid and
64-prefix/640-tail, stride-16 representation. The expansion only repairs the
staged training schedule, gradient-health audit, and clean selector.

### Surface-coordinate synthetic repair

The first A001 implementation applied the deterministic `U = TVT + Z` state
shear before calling the forward-synthetic generator. That generator already
constructs a geological surface and converts it back to TVT to sample the
typewell. It therefore double-counted Z, then returned TVT-coordinate labels
to the sheared surface likelihood. Its approximately 40-ft F0 standalone
diagnostic is invalid implementation evidence, not evidence against A001.

Before any repaired surface metric, the transformation is frozen in the
opposite order: build or synthesize the complete example in TVT coordinates,
then apply the same fractional Z shear to the likelihood, pseudo-likelihood,
support, exact target, and labels exactly once. A metric-free paired-RNG gate
must show identical synthetic GR, `surface_target = tvt_target + relative_Z`,
and a surface likelihood equal to a fractional shear of the TVT likelihood.
Only after that gate passes may the already registered surface-coordinate
variant enter a target-exposed capacity screen.

The repaired coordinate identity exposed a separate representation limit
before a model was launched. Across F0-F3, maximum absolute hidden surface
excursion has median 157.72 ft and exceeds the registered +/-120-ft grid for
474/619 wells. A wider absolute surface grid would either clip most wells or
lose the fine typewell resolution. The direct A001 expansion is therefore
stopped pre-metric.

The registered hierarchical successor keeps the fine +/-120-ft, 1-ft TVT
likelihood volume as input but predicts the direction of the structural
surface between stride-16 columns. At fixed +/-0.5-ft thresholds the F0-F3
training distribution is 23.1% down, 45.7% flat, and 31.2% up. A
position-preserving U-Net pools the typewell state axis into a per-column
three-class head. It is decoded through fold-training class movement means
and integrated from the exact visible surface anchor. This is the registered
three-class formation-motion mechanism at a dense resolution, not a wide
absolute surface category.

### Final-v20-centered local residual competition

The scratch row-competition expansion remains much weaker than the immutable
final v20 path, while the earlier global structural residual decoders reduce
the path to only 128 low-frequency controls and do not retain the local
typewell-row competition. A prospectively frozen structural-latent variant
therefore combines the two representations instead of replacing either.

For each complete well, the registered 19-plane alignment likelihood is
resampled on a fixed +/-32-ft, 1-ft grid around the fold-safe final-v20 prior.
Visible-prefix centers come from legal `TVT_input`; hidden centers come only
from the immutable fold-safe v20 prediction. Local residual coordinate, prior
offset, prior structural slope, and protected-component disagreement are
additional legal planes. A position-preserving anisotropic U-Net predicts the
training-only `TVT - prior` category. At native resolution, only its
interpolated posterior-mean residual is added to the untouched v20 path, so a
zero outer weight is an exact no-op.

Training mixes real fold-safe v20 centers with smooth anchor-zeroed
perturbations of training-side truth; held targets never construct an input.
The state radius, interpolation, loss, augmentation, and registered outer
weight grid are frozen before any F0 metric. Two fixed wells must be memorized
below 20% of initial residual-path RMSE with finite FP32 gradients, while two
untouched wells distinguish deliberate memorization from a broken optimizer.

Its schedule is source-frozen as eight forward-extreme synthetic epochs with
hardness ramped to 0.7, followed by an optimizer-restarted 16 real-data
epochs. Snapshots 4/8/12/16/24 and temperatures 0.5/0.75/1/1.5/2 are retained.
The categorical loss is accompanied by fixed 0.5 cumulative-path, 0.1
increment, and 0.005 increment-difference terms. A fixed-batch path gate must
pass with finite FP32 gradients before even an F0 capacity screen.

The A010 TVT expansion subsequently failed its frozen other-three-fold
selector: RMSE 5.941944094 versus 5.940191010, or -0.001753084 gain. Its fold
gains were +0.000916/-0.008359/-0.000771/+0.001667. The result is useful
controlled evidence that simply repeating the absolute categorical alignment
at new seeds is not a component; A012 changes the predicted object to dense
surface motion rather than tuning A010 again.

A012's fixed-batch gate passed: cumulative surface-path RMSE on the two fixed
wells fell from 73.6820 to 13.0736 in 600 FP32 steps with zero nonfinite
gradients, while two untouched wells worsened from 108.3215 to 137.4941.
Both registered seeds may therefore enter one F0 capacity screen; F1-F3 remain
unspent until that screen shows material transfer.

### Accepted-residual snapshot bag

The D024 local-v20 residual family is the only current component with positive
selection-clean contribution on every development role. Its clean selector
chooses late snapshots 12 or 16, and the registered independent-seed screen
tests variance across training trajectories. A complementary zero-training
amendment tests variance within the accepted trajectory by averaging already
frozen decoded predictions at adjacent late snapshots.

The prospective recipe bank contains exactly three bags: epochs 8+12, 12+16,
and 8+12+16, with equal component weights. Every snapshot in a bag uses the
same registered decoder temperature; the candidate paths are averaged before
the existing outer blend. All original single-snapshot recipes remain eligible.
For held role `h`, the unchanged D024 mean-rank selector compares singles and
bags using only the other three development roles, with the same blend RMSE,
standalone RMSE, toe-quartile RMSE, endpoint RMSE, temperature grid, and outer
weight grid. Remaining ties prefer fewer snapshots and then the original
snapshot ordering, so a numerically neutral bag cannot displace D024.

A metric-free gate must show exact arithmetic averaging, exact zero-weight
reconstruction, finite output, stable row layout, and a non-identical bag when
the frozen snapshot inputs differ. If the expanded selector is not better than
D024, the original output remains the family result.

### Independent accepted-residual trajectory

The first A192 independent seed clears both preregistered expansion thresholds:
its frozen epoch-16 F0 path gains 0.064444 ft at temperature 1.5 and share
0.20, and its row-level residual correlation with D024's frozen epoch-16 path
is 0.485810. The second independent seed gains only 0.044413 and is stopped.
No F1-F3 metric from the passing seed has been constructed or inspected.

The clean expansion retains every D024 recipe and adds seed 20261043 plus an
equal two-seed decoded-path bag. A bag always uses the same registered snapshot
and temperature in both trajectories; paths are averaged before the existing
outer share. For held role `h`, D024's unchanged other-three-fold mean-rank
selector compares original, independent, and bag treatments using blend RMSE,
standalone RMSE, toe-quartile RMSE, and endpoint RMSE. Remaining ties prefer
the original trajectory, then the equal bag, then the independent trajectory,
so a neutral expansion falls back exactly to D024. This changes only the
training-trajectory treatment, not the accepted model, loss, curriculum,
decoder grid, or maximum 0.20 candidate share.

A metric-free gate must establish exact bag averaging, exact original-treatment
fallback, exact zero-share reconstruction, finite output, and stable row layout
before training the passing seed on F1-F3. The expanded selector replaces D024
inside the same final component slot only when its frozen clean pooled score is
better.

### Multiscale-response accepted-residual trajectory

The A052 production trajectory completes its registered F0 screen with
+0.068983 gain at epoch 16, temperature 2, and share 0.20. Its row-level
correction correlation with D024 at the corresponding epoch and temperature is
only 0.344725, so the extra acquisition-response views learned a materially
different correction rather than reproducing the accepted seed.

Before constructing F1-F3 metrics, the clean expansion is frozen identically
to the independent-seed expansion: every original D024 recipe remains eligible,
alongside the multiscale-response trajectory and an equal decoded-path bag at
the same snapshot and temperature. D024's four-metric other-three-fold
mean-rank selector and outer share grid are unchanged. Exact ties prefer the
original path, then the equal bag, then the multiscale path. A metric-free gate
must prove exact averaging, original fallback, parent no-op, finite output, and
stable layout before the registered F1-F3 jobs begin.

### Stride-8 multiscale residual competition

The accepted D024 and A052 residual models compress the horizontal well at
one column per 16 native rows. The reviewed complete-well CNN evidence instead
attributes useful steering to denser horizontal GR heatmaps and trajectory
tangents. A056 tests that hypothesis by replacing V20 with a wide absolute
categorical path, but its early F0 trajectory is weak; the safer experiment is
to retain V20 as the exact prior and increase only the residual model's
horizontal resolution.

The prospective A202 model therefore keeps A052's 35 legal response planes,
the +/-32-ft residual states, loss, temperatures, and outer shares, while
changing the sequence grid to 128 visible-prefix plus 1,280 hidden-tail
columns at row stride 8 and native GR smoothing window 31. Base width 8,
batch size 1, two repeats, seed 20261141, and the existing 16-epoch schedule
control memory and training cost. A fixed two-well gate must reduce residual
RMSE below 20% of initialization with finite gradients and an exact zero-share
parent before one F0 capacity screen. No F1-F3 metric may be opened unless
that screen is material.

### Protected-parent hard-tail diversity

A045 did not clear the standalone expansion screen: its best hard-CE-plus-MSE
F0 gain is +0.044651. It is nevertheless mechanistically and numerically
diverse from the accepted residual path: the corresponding row correction has
only 0.045 correlation with D024, and a bounded nonnegative F0 ensemble
diagnostic assigns it positive weight while reducing RMSE from 5.9092 to
5.7485 across the currently available paths. That same-held diagnostic is
headroom evidence only and cannot select its checkpoint, temperature, or
weight.

The prospective diversity expansion retains only the already strongest
registered hard-CE-plus-direct-MSE variant, but keeps all of its frozen
snapshots 1/2/4/6, temperatures 0.75/1/1.25/1.5/2, and shares through 0.20.
For held role `h`, the unchanged four-metric mean-rank rule selects a recipe
using only the other three roles. Exact zero share remains eligible. This
component may survive only through positive contribution to the later frozen
multi-family ensemble; its F0 score alone cannot trigger a reveal.

### All-family nearest-curve transfer

The A006 same-typewell-family transfer was rejected because its eligible local
route gained only +0.001737, while the deterministically permuted family
control gained +0.060571. That inversion specifically falsifies the 0.99
typewell-family restriction, not fold-excluded full-curve transfer itself.
The public close-neighbor hypothesis also conditions primarily on geometry,
not an exact typewell-profile cluster.

A prospective minimal repair therefore reuses A006's implementation unchanged
except that every F0-F3 training-side reference belongs to one common eligible
family. Its top-three whole-well analog, local-32 point field, equal mean,
visible-prefix median bias, distance limits, caps, and shares remain frozen.
For held fold `h`, all references from `h` and all F4 target references remain
excluded. A target-free gate must verify the all-family assignment, exact
outer no-op, visible calibration, and fold exclusion before one F0 capacity
metric. The prior same-family route is the controlled comparator; no recipe
from its permuted target result is imported.

### Protected-v20 posterior-mode bank

The tracked historical source documents a useful mechanism after the current
v20 freeze: a posterior mean can average incompatible categorical paths, while
the query's hidden horizontal GR remains legal evidence for choosing among
complete posterior modes. Historical post-v20 checkpoints, cached paths,
chosen modes, calibration artifacts, fitted shares, and validation constants
remain excluded.

A prospective deterministic amendment therefore regenerates posterior
quantiles, the hard mode, and two temperature views from every fold-specific
checkpoint already present in the protected v20 vault. Each view contributes
only its anchored, smoothed difference from its own protected posterior mean.
The query's visible prefix supplies a robust affine typewell-to-horizontal-GR
calibration; candidate ranking then uses a registered grid of absolute,
Huber, correlation, and increment-correlation scores on inference-visible
hidden GR. For held development fold `h`, the score rules and at most three
nonnegative component weights are selected only on the other three roles.
A half-tail circular shift of query GR is the target-free negative control.
This rebuild imports the historical mechanism but none of its target-derived
decisions or artifacts.

### Multi-anchor hidden-GR phase correction

The toe-datum experiment establishes that inference-visible hidden GR can
recover part of the accumulated path drift, but one endpoint shift forces the
entire correction to have a single power-law shape. The next registered
variant keeps the accepted D024 path fixed and measures local phase at four
hidden-tail positions. Each measurement compares a short horizontal-GR
support with the typewell GR sampled around the D024 path over a fixed
continuous shift grid. Response-scale estimates are aggregated without
labels, then decoded as an anchor-zeroed piecewise-linear or
second-difference-smoothed correction.

The horizontal hidden TVT never enters construction. The only query evidence
is legal horizontal GR, legal typewell TVT/GR, the visible TVT prefix used for
affine response calibration, and the immutable fold-safe D024 prediction.
Within-well circularly shifted GR is processed through the identical
measurement and decoding path as the negative control.

## Smallest falsifiers

1. Every trainable architecture must memorize a fixed labeled batch with
   finite gradients.
2. Raw-GR models must outperform shuffled-GR and zero-GR controls on
   training-side roles.
3. The hierarchical fine stage must improve its coarse parent without using
   truth-centered crops.
4. Zero-step adaptation must reconstruct the parent exactly; shuffled-label
   adaptation must not look beneficial.
5. A family is expanded to all F0-F3 roles only after a useful F0 pilot or a
   clear underfit diagnosis with registered repair axes.
# A199: complete-well multi-trajectory prediction

The locally cached `hengck23/cnn-mtp-example` notebook, linked from discussion
699853, proposes a convolutional complete-well model that emits `K` full
trajectories and mode logits. Its loss assigns regression to the closest
trajectory and trains the logits to identify that trajectory. The useful
mechanism is multimodal trajectory competition, not the notebook's hard-coded
crop, claimed score, or checkpoint.

A199 ports only that mechanism into the legal protected-v20 residual setting:
all inputs are inference-visible, the known prefix and protected v20 path
provide the prior, every correction is anchor-zeroed, and F0-F3 remain the
only development labels. This differs from A007's single global path and
D024's single per-row posterior mean. The falsifying controls are fixed-batch
mode specialization, exact zero-share reconstruction, and a three-variant F0
capacity screen.
# A300: physical candidate-path scoring

The locally cached discussion 702474 develops an output-space view: generate
physically plausible geological-surface paths, project them through the
typewell, and learn which candidate explains the observed horizontal GR. It
also notes that the two projection directions carry different scoring
information and that the observed-versus-simulated GR gap is the main
sim-to-real difficulty.

A300 tests the narrow mechanism needed here without importing a public
checkpoint or train-only geology columns. It proposes bounded, anchor-zeroed
low-frequency corrections around protected V20, samples the existing legal
alignment likelihood and its local tube along each complete candidate, and
learns only the residual/trajectory cross-term from other wells. Candidate
norm and final SSE ordering stay analytic. This is materially richer than the
failed per-well summary rankers and materially different from A199's direct
best-of-K path regression.

# A305: physics-shaped residual-prior curriculum

The public output-space experiments and the F0-F3 V20 error decomposition
agree that the consequential uncertainty is dominated by coherent
anchor-zeroed datum, drift, and bend rather than arbitrary row noise. D024
already proves that a truth-centered synthetic-prior curriculum can teach a
fold-safe V20 residual network to use legal GR alignment evidence, but its
training perturbation is a randomly smoothed trace whose local variation can
spend much of the model's capacity away from those dominant modes.

A305 changes only the training-side prior-error distribution. One profile
draws a linear endpoint plus a symmetric bend, one draws four piecewise
control-point increments, and one mixes those two profiles with D024's
original smooth perturbation. Every path is exactly zero through the visible
prefix, bounded to the existing +/-32-ft residual support, and centered on
known training truth. Real validation and inference continue to use the
immutable fold-safe V20 prior; inputs, architecture, objective, snapshots,
temperatures, shares, seed, and hidden-tail masking otherwise remain D024's.
This isolates whether the synthetic-to-real gap is chiefly a prior-shape
mismatch without importing any public prediction, fitted constant, or
target-derived F0 recipe.

# A400: target-free physical proposal energy

A300's candidate bank exposes the two quantities that the public physical
output-space discussion says should score a proposed surface: observed GR
against the typewell projection and the complementary pseudo/inverse
projection. Those signed and absolute mismatch volumes are already legal
inference inputs, but A300 asks a small supervised scorer to rediscover their
basic energy from a limited number of complete wells.

A400 isolates the physical signal before adding learning. It samples the two
absolute mismatch channels along every frozen A300 path, aggregates only over
supported hidden-tail positions, and ranks paths by fixed observed, pseudo, or
equal energies. Fixed clipping tests robustness to isolated GR outliers, while
a dimensionless path-norm penalty expresses conservative shrinkage toward
V20. No label, OOF residual, protected component weight, or post-v20
checkpoint enters the score. Development truth is used only after the entire
recipe grid is frozen, to measure the resulting correction at an outer share
no larger than 0.20.

# A410: robust contrast and consensus proposal energy

A400's four-fold transfer establishes that direct physical likelihood is not
an F0 accident. Its remaining weakness is statistical rather than geological:
a raw complete-tail mean gives equal influence to locally flat GR, missing
intervals, isolated mismatch spikes, and informative contacts. The frozen
A300 tube contains enough inference-visible structure to distinguish them.

A410 therefore tests three label-free likelihood refinements. Eight-segment
one-high trimming prevents one damaged interval from dominating a complete
well. Per-column percentile rank compares paths without assuming mismatch
amplitude is calibrated across wells. Local-minimum contrast rewards a path
only when its observed and/or pseudo mismatch is lower than the same trace
sampled two and four feet to either side. The grids are frozen physical
robustness choices, not target-fitted well rules; truth is opened only after
the full scorer/decoder/share bank passes its constructed metric-free gate.

# A420: target-free residual Viterbi alignment

The repaired A411 experiment demonstrates transferable physical likelihood
without any learned scorer, but its 153 candidates span only one endpoint and
one symmetric bend. Real geological trajectories can encounter several
contacts, so a globally correct path may not live in that two-parameter bank
even when the rowwise likelihood ridge is informative.

A420 decodes the same inference-firewall-passed observed and pseudo mismatch
volumes directly over the existing +/-32-ft residual states. A Viterbi
transition limits how quickly the correction can move, and fixed Hann
smoothing plus a correction cap enforces a deployable geological path around
protected V20. The decoder sees no labels, checkpoints, component weights, or
hidden-tail-derived mask. This revisits the dynamic-alignment mechanism with
the stronger V20-centered two-projection likelihood that A411 validated,
rather than the weaker raw rolling-correlation representation rejected by
D027.

# A440: group-relative physical landscape ranker

The repaired A411 bank shows that direct likelihood is useful on every fold,
while A300's path-independent sequence CNN barely leaves its zero-correction
initialization. That contrast suggests the missing representation is not more
trace capacity but explicit competition among the 153 paths in one well.

A440 summarizes each path's two-projection energy, robust tail segments,
neighbor contrast, and physical endpoint/bend geometry, then adds within-well
ranks and standardized deviations for every summary. A small grouped boosted
tree model predicts either the cross-term needed by the exact SSE identity or
the pairwise path order. It is trained only on other wells and retains the
analytic correction norm and exact V20 no-op. This directly tests whether the
large candidate oracle can be recovered by calibrating a complete physical
landscape rather than by another high-capacity image model.

# D401/A411: proposal support firewall repair

The A300 proposal constructor used the residual trainer's `label_valid` array
as proposal support. That array is a training-loss mask: in addition to the
inference-visible row-stride mask, it removes columns whose hidden truth is
outside the +/-32-ft classification radius. It therefore made A300's features
and A400's aggregate energies depend on hidden TVT. The defect affected 106
sampled F0 rows in two extreme wells and invalidates every proposal-family
metric or derived registration through A410, regardless of apparent
cross-fold transfer.

A411 repairs the boundary rather than the recipe. For real proposal
construction it passes a copy of the well whose hidden TVT tail is replaced
by the inference-visible NaNs, derives support from deterministic sequence
geometry, and attaches truth only after proposal features and paths are
complete when a development metric explicitly requests targets. A
target-free route must be bit-invariant under arbitrary hidden-tail
replacement before any repaired score is opened. The rerun preserves A400's
original path bank, recipe order, shares, and A401 held-excluded selector and
does not inherit the invalid scaled or robust extensions.

# A440: MTP diversity expansion

The repaired MTP family completes its full registered F0 trajectories without
crossing the usual standalone expansion threshold: its strongest
K8/control256/component path gains 0.0451 at an outer share of 0.20. Its
mechanism nevertheless passed a demanding best-mode specialization gate, and
the selected F0 correction has only 0.084 row correlation with the accepted
two-seed local-residual correction. That makes it a plausible ensemble
direction rather than a replacement residual model.

A440 therefore expands only the strongest preregistered representation, not
the F0-selected epoch or decoder. All five original snapshots, all seven
original decoders, and the original share grid remain available, with each
held role selected by pooled row SSE on the other three development roles.
It cannot justify a protected reveal on its own and survives only if the
final frozen multi-family ensemble assigns it useful nonnegative weight.

# A422: Viterbi amplitude response

A420's repaired F0 path is both conservative and boundary-limited: the
winning decoded correction is capped at 8 ft, then enters V20 at the maximum
allowed component share of 0.20. This does not establish that the path shape
is wrong; it may simply leave its magnitude under-calibrated.

A422 preserves every inference-visible energy and path decision and tests
only five globally fixed correction multipliers before the unchanged outer
share grid. Scale one is an exact A420 fallback and zero share is exact for
all scales. Any expanded clean run must freeze its held-excluded selector
before opening the other folds, and the final component share remains at most
0.20.

# A460: repaired synthetic proposal scoring

D401 invalidated the learned proposal experiments because their real-example
support mask depended on hidden target range. That defect does not invalidate
the underlying training hypothesis: on training wells only, deliberately
perturb a known TVT prior, construct the same physical candidate bank, and
teach a small sequence scorer the candidate/error cross-term.

A460 repeats the originally frozen balanced-small, heavy-small, and
balanced-dilated curricula from scratch with the repaired proposal builder.
Every real validation feature is built after blanking hidden TVT and uses only
deterministic sequence support; target truth is attached after candidate
construction solely for development scoring. Synthetic truth-centered priors
remain training-only augmentation. No old checkpoint, target metric, or
recipe is retained.

# A450: repaired robust proposal energy

The robust A410 recipe grid was frozen before its original metric, but D401
voided that metric because the shared proposal support depended on hidden
target range. A450 does not reinterpret the old result. It reruns the exact
486 clipping, segment-trimming, column-rank, local-contrast, norm, and decoder
recipes from scratch after C405's inference-only repair.

D402 additionally shows why a pooled selector is insufficient even when the
features are legal: one aggressive recipe can regress a new fold sharply.
A450 therefore freezes strict held-excluded sign stability before its rerun.
Each candidate must improve every training role, then maximizes its worst
gain; V20 is the exact fallback. This tests whether robust aggregation makes
the physical direction genuinely portable rather than merely improving the
same-held capacity screen.

# A520: tracker-conditioned recurrent residual refinement

The public control-first working note, *When Better CV Scores Worse*, reports
its most durable learned family as several bidirectional GRU refiners over a
physical tracker, complete hidden-tail GR, future trajectory, fixed-offset GR
mismatch, and uncertainty features. A banded quadratic solve then reconciles
the direct neural path with a separately predicted structural rate. The same
writeup reports that wider GRUs saturated, stabilized TCN/transformer variants
added almost nothing, and full-well images learned a prior rather than the
correct mode. This is used only as a public mechanism hypothesis.

Protected V20 is a nine-member categorical U-Net stack. D209 learns a local
V20-centred likelihood correction, while A500 compresses that likelihood into
TCN/transformer column tokens. Neither family performs recurrent refinement on
native hidden-tail rows with explicit legal future trajectory and calibrated
typewell response. A520 therefore samples every eighth native row and combines
the protected baseline/components with raw and smoothed query GR, fixed-offset
typewell mismatch, component disagreement, trajectory derivatives, visible
surface-prefix summaries, and distance-to-end context.

Three genuinely different objectives are frozen: regularized Huber direct
residual, hard-tail metric-weighted MSE direct residual, and a direct-plus-rate
model whose two outputs enter a fixed banded quadratic fusion. No public
checkpoint, prediction, fold, fitted constant, or selected recipe is imported.
Every feature must be bit-identical after arbitrary hidden-target replacement,
and all three variants must deliberately fit the same two wells before a
single F0 capacity trajectory is allowed.

# A580: fold-safe random-cut protected tracker

The public control-first note and discussion 699853 both emphasize that a
tracker refiner must learn across forecast horizons and realistic tracker
failure states. A520 exposed only the single organizer cut and one fixed OOF
V20 path per training well. A560 tried to widen that distribution with
truth-centered analytic perturbations, but its three F0 routes collapsed to
the no-op.

A580 changes the source of augmentation rather than its scale. For each
development well, its own whole-well-excluded protected checkpoints are rerun
at several artificial cuts. Each cut reveals only the corresponding training
prefix while preserving the legal complete future GR and trajectory. This
produces multiple strong, model-generated tracker errors per well, including
different horizon and mode-selection failures, without using a checkpoint
that trained on that well. The actual-cut reconstruction is checked against
the immutable V20 cache, and hidden targets are permuted or blanked before any
development metric to prove that the cache builder is inference-faithful.

No public prediction, fitted constant, checkpoint, fold, or target-derived
recipe is imported. The public material supplies only the mechanism:
random-cut training around a physical/learned tracker followed by recurrent
direct/rate refinement.

Before implementing the cache, an algebra check on the already-protected
actual-cut arrays showed that the nine historical component weights alone do
not reconstruct v20: protected v20 also contains its accepted inference
corrections. A581 therefore uses only later artificial cuts. The exact
target-free difference between protected v20 and the weighted nine-component
actual path is a legal prior at every later cut, because the later prefix is a
superset of the organizer prefix. Translating all newly inferred component
paths by that common field preserves their disagreement exactly and recovers
protected v20 at the actual cut. This repair was frozen before cache code,
training, or any new target metric.

# A600: fold-purged super-typewell input refinement

The competition overview describes each query as alignment of a horizontal
well to a supplied typewell reference. A public target-free workflow note
further motivates treating nearby typewells as shifted regional reference
copies and combining aligned references when the assigned trace is noisy.
Only that qualitative mechanism is imported. No public prediction, weight,
fold, threshold, fitted constant, or score enters A600.

A600 tests the mechanism strictly from inference-visible inputs. For a held
development query it excludes every well in the query's fold and all of F4,
ranks the remaining F0-F3 donors by heel distance, aligns their typewell GR
to the query reference, and constructs a small fixed mean/median bank. The
route is selected only by the visible `TVT_input` prefix and horizontal GR.
The chosen reference is then passed through the immutable protected
categorical checkpoints; their weighted path difference from the original
reference becomes an anchor-zeroed bounded correction to V20 or D570.

A target-free audit on F0 justified one protected metric pilot: the visible
selector chose a non-base reference for 141 of 155 queries, and the
near-five mean improved visible-prefix fit on 46.5% of wells. Those are input
fit diagnostics, not target evidence. Hidden horizontal TVT is replaced,
reversed, or blanked before the implementation gate, donors are fold-purged,
and the original-reference inference must reproduce the immutable protected
component cache before any A600 target metric is permitted.

# A620: master-frame row residual boosting

Three public observations motivate a representation test rather than another
postprocessor. Discussion 726465 reports a whole-well, grouped, inference-faithful
single model below 5 ft without neighboring-well inputs. Discussion 717573
reports that ordinary LightGBM, CatBoost, and XGBoost become competitive when
the right inputs are supplied. Discussion 700424 explicitly identifies the
shared typewell identity, trajectory, multiscale smoothing, and segment dropout
as useful candidate inputs. These are mechanism claims only: A620 imports no
public model, prediction, validation split, fitted feature, weight, or score.

V20 represents the supplied typewell only through local candidate-state GR
values. It does not expose a stable identity for the complete reference trace
to its decoder. A label-free audit of F0-F3 typewell inputs on the preregistered
one-foot absolute-TVT grid produces 53 connected master frames at correlation
at least 0.99 and overlap at least 200 rows. For a held development well, the
other three roles contain a median 15--16 same-frame wells. There are 2/2/5/7
unseen held-frame identities, covering 3/3/5/8 held wells in F0/F1/F2/F3. The
frame calculation reads only the
supplied typewell TVT/GR and is therefore available for hidden test queries.

A620 retains A520's fold-safe V20/component, full-tail trajectory, visible
surface-prefix, and multiscale calibrated GR-response rows, then adds the
master-frame category and label-free whole-reference summaries. It predicts
the V20 residual every eight native rows and interpolates the anchor-zeroed
correction to the complete hidden interval. CatBoost, LightGBM, and XGBoost
are frozen as distinct learners; fixed pair/all-model arithmetic bags test
the user's multi-model route without learning within-family blend weights.
Every held role trains on only its other three development folds, and F4,
confirmation regroupings, horizontal formation columns, and hidden horizontal
TVT are excluded.

The metric-free C620 capacity gate subsequently found that the frozen
LightGBM regularization was too strong for its own fixed two-well batch:
anchor-zero residual RMSE fell from 2.471462 to 0.615800 ft, or 24.9164%,
missing the prospective below-20% threshold. CatBoost and XGBoost passed at
9.2550% and 7.2253%, respectively, and all firewall, parent-layout, no-op,
and untouched-well checks passed for all three learners. C623 therefore
changes only LightGBM's minimum child rows from 200 to 50 before any A620 F0
target metric. The learner count, feature representation, target, boosting
iterations, leaves, learning rate, regularization, sampling, seed, treatment
bank, and every selection threshold remain frozen. All three learner gates
must be rerun under the repaired common source hash.

The frozen F0 evaluator subsequently clears A621's prospective expansion
boundary. Around V20, LightGBM's best same-held diagnostic gains 0.075408
ft; after D570 it gains 0.0300075 ft, seven-and-a-half millionths above the
registered inclusive 0.030 threshold. This same-held recipe is not eligible
for deployment. The exact three learners now expand to F1--F3, after which
each held role's treatment, amplitude, and share are selected exclusively
from its other three roles under the already frozen A621 rule.

The selection-clean result rejects the family. F0, F1, and F3 correctly
choose the exact zero fallback after excluding their held targets. The
F2 recipe is selected from positive F0/F1/F3 evidence, but regresses F2 by
0.100143 ft after D570, leaving a pooled incremental loss of 0.022965 ft
and zero positive held folds. The master-frame learner outputs therefore
remain diagnostic artifacts only and do not enter the frozen ensemble.

# A680: frozen-parent raw-state residual U-Net

Discussion 700424 contains two unusually specific clues from a competitor
reporting roughly 5-ft grouped whole-well CV with a simple U-Net: score the
competition among candidate TVT rows rather than regress its shadow, and feed
the network the typewell and horizontal signals separately. The same thread
credits four epochs of forward-synthetic pretraining with a material gain.
Discussion 726465 later confirms that a complete-well, per-well-only neural
model can reach pooled CV below 5 ft without neighbor wells. These are
mechanism clues only. No public weights, predictions, folds, selected
constants, or target-derived artifacts are imported.

Protected v20 already performs categorical row competition and synthetic
pretraining, but its 19-plane input exposes calibrated GR mostly through
observed-minus-expected mismatch. A670 added separate observed/expected GR
and legal absolute state planes through a tiny zero-initialized stem. That
stem produced a real +0.0317-ft F0 increment while it alone was trainable.
Unfreezing the inherited network immediately collapsed the held gain even as
training loss fell; a label-free typewell-frame embedding added essentially
nothing. The diagnosis is neither "the representation is useless" nor "train
v20 harder": the new signal is useful, the small stem is capacity-limited,
and changing the protected feature extractor causes catastrophic
specialization to three folds.

A680 therefore freezes the exact protected parent permanently and adds a
separate residual categorical U-Net whose output logits are added to the
parent logits. Its head is exactly zero at initialization, so the candidate
begins as the protected parent without relying on approximate blending.
Three registered variants isolate raw-pair, legal absolute-state, and
multiscale raw-pair capacity. Each receives the original 19 planes as context;
the larger variants add only deterministic inference-visible channels.
Forward-extreme synthetic pretraining is followed by real complete-well and
random-cut training, while the parent remains in evaluation mode and receives
no gradients. This directly tests whether a higher-capacity learned
likelihood correction can exploit the representation Tucker described
without erasing v20's already transferable solution.

# A690: fixed-seed diversity for the absolute residual model

The first registered A680 checkpoints provide a narrow, mechanism-consistent
reason to spend compute on variance reduction rather than alter the model:
after D570, the absolute-coordinate member gains 0.065985 ft after synthetic
pretraining and 0.059598 ft after its first real-data block, while the simpler
raw-pair member independently rises to 0.052298 ft. These diagnostics were
already part of A680's frozen bank. They show a shared positive direction but
also substantial stochastic mode-selection noise near the +0.075 expansion
boundary.

Discussion 700424 reports that equal averaging of five seeds materially
improves the same simple categorical complete-well formulation. A690 imports
only that mechanism. It imports no public seed, checkpoint, prediction,
weight, fold, or score. Four prospectively fixed independent seeds repeat the
exact A680 `delta_absolute_b8` architecture and training trajectory. The
original seed is member one, and only fixed cumulative equal-prediction bags
in seed order are eligible; there is no learned blend or post-metric member
selection. A wrapper gate must pin the unchanged A680 core and protected
aggregate gate, prove deterministic and distinct initialization, and bind
every replica's provenance before any new held metric is opened.

# A700: repair the supposed raw typewell/horizontal planes

The A680 source audit exposed a precise mismatch between its implementation
and the public mechanism it intended to test. `pair_planes` robustly
normalizes horizontal GR using one scale, then subtracts a mismatch computed
with a different prefix-calibrated scale. The resulting second plane varies
with horizontal column. It is therefore neither a raw typewell plane nor a
clean calibrated expected-GR plane.

Hengck's public example makes the intended axes explicit: typewell GR has
shape `[B,1,T,1]` and is expanded across horizontal columns, while horizontal
GR has shape `[B,1,1,H]` and is expanded across candidate states. Tucker
specifically points to those separately supplied signals. A700 implements
that literal contract with independent robust normalization, which also
removes the synthetic generator's arbitrary positive gain and offset. The
six absolute inference-visible planes, frozen parent, residual architecture,
training curriculum, and evaluation bank remain unchanged. No public
prediction, weight, fold, checkpoint, selected constant, or score is used.

# A710: preserve calibration while repairing the signal axes

A700's first prospectively fixed snapshot is nearly null, showing that
independently normalizing the two literal raw traces removes useful
acquisition calibration. The original A680 pair retained that calibration
inside its mismatch plane but attempted to reconstruct expected GR after
renormalizing horizontal GR with a different scale.

A710 uses the algebra of the supplied likelihood image itself. Within each
candidate-state row, column-centering removes the state-only expected value,
so covariance with visible horizontal GR identifies the positive mismatch
scale. In that common scale, `horizontal - mismatch` should depend only on
candidate state. A robust row projection enforces exactly that separability.
The operation uses only inference-visible GR, validity/missing masks, and the
already legal mismatch plane; it works identically for real and forward-
synthetic examples and imports no public artifact or target-derived choice.

# A720: capacity and correctly oriented scale space

Before A710 exposes a target metric, A720 freezes the two natural capacity
tests implied by the earlier controlled bank. A680's contaminated base12
multiscale model is as strong as its base8 raw-pair member, so width and
scale-space context remain plausible rather than post-hoc rescues.

The first A720 member changes only residual width from eight to twelve. The
second smooths the calibrated horizontal signal only along horizontal
columns and the calibrated expected/typewell signal only along candidate
states, at the already registered widths 9 and 33, then appends their
axis-correct gradients. The first eight planes must remain bit-identical to
A710. No learned bag weight, public artifact, new target source, or held
metric informs these variants.

# A730: remove the common frozen-parent ceiling

The registered residual members use different seeds and representations but
are converging near the same small D570 increment. Their common factor is the
frozen protected parent: optimization begins from its already confident
posterior and learns a conservative logit correction.

The strongest public evidence instead describes a complete-well categorical
U-Net as the single model. The earlier Run-6 scratch controls are not a clean
test of that mechanism: they used the malformed reconstructed typewell plane
and stopped after only eight real epochs. A730 trains base12 U-Nets from
scratch on each of the now-gated corrected representations for 32 total
epochs. Pure categorical competition, visible-only calibration, random cuts,
and forward-synthetic pretraining remain unchanged. The fixed outer zero
share preserves D570 exactly, and a fixed equal pair tests diversity without
fitted weights. No public weights, predictions, fold choices, or constants
are imported.

# A732: preserve categorical modes at inference

A730's pure cross-entropy objective makes the candidate rows within a column
mutually exclusive, but its default deployment still returns a posterior
expectation. If two geological modes remain plausible, that expectation can
fall between both paths. Before either scratch model emits a target metric,
A732 freezes the simplest control: take the maximum-logit row with the
deterministic first-index tie rule, then reuse A730's visible-only calibration,
interpolation, outer shares, snapshots, and fixed equal decoded-path pair.
This imports no public prediction, fitted transition penalty, or
target-derived decoder setting.

# A740: literal three-plane public formulation

The corrected A730 inputs contain the three planes shown in Hengck's public
example, but also retain 24 engineered mismatch, support, coordinate,
trajectory, and absolute-state planes. Those shortcuts may encourage a
scratch model to specialize to three training roles. Before A730 exposes its
first target metric, A740 freezes the literal simple formulation: candidate
typewell GR, horizontal GR, and the visible-prefix path hint only. True-raw
and calibrated-axis versions isolate normalization, while architecture,
objective, curriculum, optimizer, snapshot bank, calibration, and outer
evaluation remain A730-identical. No public weight, prediction, fitted
constant, or target-derived choice is imported.

# A750: score the known part of the complete path

A730's image contains a Gaussian hint for every inference-visible TVT_input
column, but its inherited loss masks those columns and scores only the hidden
suffix. In a complete-image segmentation formulation, reconstructing the
known prefix is legal supervision and teaches the decoder what the hint
means before it must propagate the path. After the synthetic-only A730
snapshot and before any real-data snapshot, A750 freezes one calibrated
variant that applies the same categorical CE to every valid column. Every
feature, model, augmentation, optimizer, checkpoint, decoder, and outer
share remains unchanged, and no hidden target is added at inference.

# A760: adapt the parent without forgetting it

A670 established two facts: a zero-initialized absolute/raw stem has useful
direction, and unrestricted CE updates to the inherited network immediately
destroy held performance. A680 then showed that freezing every inherited
feature limits independent residual models to about +0.066 after D570.
A760 tests the missing middle ground. It injects A710's exact calibrated
planes through a zero stem, then allows full feature adaptation while a
fixed temperature-KL penalty keeps the student's posterior near the immutable
fold-safe teacher. Synthetic stem-only pretraining and the subsequent real
schedule are fixed prospectively. The teacher supplies no new target or F4
input; it only preserves knowledge already present in the protected parent.

# A770: restore the half-foot local competition

The source audit behind A740 uncovered an important distinction that its
three-plane ablation did not test. Hengck's cached public complete-well image
resamples the typewell at 0.5-ft spacing and crops 256 typewell rows around
the visible endpoint. That is a local, approximately +/-64-ft categorical
competition. A730 instead inherited Run-6's +/-120-ft grid at 1-ft spacing.
The two images have similar tensor heights, but not the same physical
resolution.

This matters independently of whether the input has three or 27 planes.
Tucker describes the useful output as rows competing within each horizontal
column, and the strongest historical v20 hypotheses also used a local
half-foot grid. A730's immutable epoch16 true-raw diagnostic is directionally
useful after D570 but still modest, which is consistent with a model locating
the right mode too coarsely rather than lacking all signal. A770 therefore
changes only the physical candidate grid to +/-64 ft at 0.5 ft. It preserves
the two prospectively corrected A730 representations, architecture,
curriculum, optimizer, decoder bank, and no-op blend. No public weight,
prediction, fold, target-derived constant, or F4 evidence is imported.

# A780: put the accepted path inside the competition

D570 is the only Run-6 addition that remains positive on all four
selection-clean development roles. Its residual is nevertheless highly
structured: removing a target-only per-well constant or line would reduce
pooled F0--F3 RMSE from 5.820599 to 4.159000 or 3.047250. Those oracle values
are anatomy, not deployable features, and no fitted coefficient is retained.
They show that the next model should refine a strong complete path rather than
reconstruct TVT independently.

Earlier V20-centered residual models predicted a bounded correction from a
narrow residual volume, and A680 added logits to a frozen protected posterior.
A780 is different in both representation and estimand. It places the exact
fold-safe D570 path on A770's half-foot candidate grid as two input planes and
as a fixed Gaussian log-prior. A zero residual head therefore begins at the
supplied path, while the corrected raw horizontal/typewell likelihood can
move probability mass to a different local mode. Real training uses the
organizer cut and the immutable OOF D570 path for each non-held training well.
Smooth target-centered prior corruption is allowed only as training
augmentation and is never constructed from a query target.

The hidden-test kernel can reproduce the same inputs by running D570 first.
Horizontal TVT, F4 targets, public predictions, post-v20 checkpoints, and
same-held fitted corrections are excluded. A fixed zero outer share continues
to reproduce D570 exactly even if the learned refiner fails.

# A790: isolate local synthetic diversity

A770's first immutable calibrated half-foot checkpoint improves F0 D570 by
0.069825 ft after exactly four forward-synthetic epochs. At the same frozen
stage, both coarse A730 scratch models select the exact no-op. The difference
is therefore attributable to the local likelihood resolution and seed/model
trajectory, not to supervised fitting on held-like real wells.

A790 repeats only that already fixed four-epoch trajectory under four new
seeds and evaluates cumulative equal-prediction bags in declared seed order.
It does not extend training, tune hardness, choose members, or import the
original same-held recipe. This is the narrow variance-reduction experiment
suggested by the public five-seed categorical result, now applied only where
the prospective local control shows a material direction. F1--F3 expansion
still requires a frozen treatment and held-excluded selection; F4 remains
sealed.

# A800: make forward synthesis look like observed acquisition noise

The half-foot calibrated A770 model is useful immediately after its four
forward-synthetic epochs, then its registered real-well continuation selects
the exact D570 no-op. This points to a sim-to-real mismatch rather than a
simple lack of model capacity. Hengck's cached forward-simulation discussion
explicitly identifies observed-minus-typewell GR noise as the remaining
augmentation problem.

A target-free audit of all inference-visible F0--F3 prefixes quantifies the
gap. After a per-well visible-only affine typewell response fit, the
horizontal GR residual has median RMS 12.77 GR units (10th--90th percentile
9.38--17.38), median lag-1 autocorrelation 0.722, and median lag-8
autocorrelation 0.221. The existing forward generator draws a substantially
cleaner, simpler residual. These summaries use no hidden tail labels and no
F4 rows.

A800 therefore keeps A770's calibrated 27-plane, half-foot categorical model
and changes only synthetic acquisition noise. For each artificial training
cut it extracts the correct-path mismatch from rows still visible at that
cut, then synthesizes a tail noise curve by either circular block bootstrap,
an AR process fitted to that visible curve, or their fixed hybrid. The
synthetic correct-path mismatch is replaced by that curve through a
state-common delta, preserving the candidate-row likelihood geometry and
labels. Training remains synthetic-only, with snapshots that include A770's
known epoch-4 regime. At deployment the ordinary observed input builder is
used; no target, augmentation curve, fitted held coefficient, or public
artifact enters inference.

# A810: adapt the inverse model to each query's legal simulator

The global model still has to invert hundreds of different typewell responses,
trajectory geometries, and acquisition-noise distributions with one set of
weights. Hengck's cached debug sequence recommends the opposite control:
simulate one query and fit one inverse network before mixing examples. The
held query supplies everything needed to do this legally: its typewell,
complete XYZ/MD/GR trajectory, visible TVT prefix, and a fold-safe D570 path.

A810 starts from the immutable A770 calibrated epoch-4 checkpoint. For each
query independently, it replaces the unavailable suffix target by D570,
forward-generates paths and horizontal GR around that legal prior, and applies
the query's visible-prefix noise distribution. Separate deterministic
synthetic examples train and select among steps 0/8/16/32; the actual hidden
tail is never a training or stopping label. Three bounded parameter scopes
test whether useful adaptation is acquisition calibration (normalization and
head), input calibration (stem plus normalization and head), or complete
query specialization. Deployment then runs the adapted network once on the
ordinary observed query image and discards it before the next well.

This is materially different from A510. A510 trained a protected global
parent only on artificial cuts of the observed visible prefix and its
pseudo-tail stopping improvement did not extrapolate. A810 trains on
full-horizon physical paths around the already accepted D570 prior, uses the
query's own typewell forward model, and validates on held synthetic paths.
It imports no public weights, held target, same-held coefficient, or F4
evidence.

# A820: combine pretrained global vision with the categorical competition

The cached public clues identify three ingredients together: one complete
well per sample, a three-plane typewell/horizontal/history image, and explicit
competition among the candidate rows in each column. Run-6 tested the first
and third with a scratch local U-Net, and Run 4 tested a pretrained MiT-B0
with dense signed-distance regression, but neither tested the complete
combination. Their failures therefore do not falsify a pretrained categorical
segmenter.

A820 consumes only A770's already-audited calibrated typewell and horizontal
GR planes plus the inference-visible path history. It uses a locally cached,
hash-attested ImageNet MiT-B0 encoder and a small multiscale decoder whose
single output field is scored by per-column categorical cross entropy. The
first eight epochs are physics-generated complete wells; the remaining
epochs retain a fixed half-synthetic mixture instead of abruptly replacing
the transferable simulation task with three-fold supervised fitting. No
public competition checkpoint, prediction, split, weight, or target-derived
recipe is imported.

# A830: condition categorical row competition on the legal future trajectory

Source clue: Tucker Arrants' comments in discussions 700424 and 699853.
Alongside the three-plane U-Net example, he asks what inference-visible input
is still missing and explicitly notes that post-prediction-start X/Y/Z is
causally downstream of the formation because the driller steered in response
to it. This is mechanism evidence only; no public prediction, fold, constant,
or checkpoint is imported.

The manual F0-F3 D570 decomposition gives an independent local reason to test
that mechanism. A per-well oracle constant reduces pooled RMSE from 5.820599
to 4.159000 and a quadratic trend to 2.454602, while the observed
high-frequency trajectory is already known exactly. The main unresolved
signal is therefore coherent datum/drift/shape, which a complete-horizon
trajectory encoder can condition without seeing any hidden target.

A830 keeps A820's pretrained three-plane image stream unchanged. A separate
dilated one-dimensional encoder consumes only the twelve row-constant legal
channels already produced by the inference builder: visibility, GR
missingness, Z delta, MD/X/Y displacement, three directional derivatives,
azimuth cosine/sine, and vertical curvature. Its per-column representation is
broadcast over candidate rows only at the decoder, so every row still
competes under categorical CE. Original, reversed, and all-NaN hidden-TVT
variants must produce identical deployable tensors; a constructed trajectory
perturbation must affect only the trajectory stream. This differs from prior
trajectory-aware scratch U-Nets by combining an ImageNet-pretrained global
GR-competition stream with an explicit complete-horizon physical
conditioning stream.

# A890: project rowwise model evidence onto the error modes that matter

A830's F0 diagnostic is useful but sparse: its raw disagreement from D570
improves pooled RMSE while losing on most individual wells. The development
oracle decomposition says the opposite representation should dominate:
constant, ramp, and low-order bend removal account for most remaining SSE,
while high-frequency wiggle has much less headroom.

A890 therefore changes no model and learns nothing from a target. Within each
query well it treats `candidate-D570` as an inference-visible noisy
measurement of the desired correction and projects it onto a small frozen
shape basis. Origin-constrained polynomial fits preserve the known boundary;
fixed moving averages test denoising; four segment controls retain one
mid-horizon change without permitting rowwise jumps. The raw path remains an
exact control and zero share remains an exact D570 fallback.

This is not a post-hoc oracle smoother: every coefficient comes only from the
candidate path being deployed. The one F0 screen may establish capacity, but
any retained transform must later be selected for each role exclusively from
the other three development roles and must add positive signal on every held
role.

# A900: let the global encoder see trajectory and proposal evidence early

A820 shows that a pretrained three-plane row-competition encoder carries a
small transferable direction. A830 materially strengthens that direction by
adding the legal complete-future trajectory, and D895 shows that its useful
correction concentrates on wells where the accepted D570 parent already
moved far from protected V20. The remaining limitation is architectural:
A830's pretrained image hierarchy processes only typewell GR, horizontal GR,
and path history. Its trajectory branch is fused after all four encoder
scales have already been computed, so the global representation cannot learn
candidate-state features conditioned on the future drilling path or D570
proposal.

A900 retains A840's audited D570-guided 15-plane builder and fixed curriculum,
but exposes all planes at the first encoder convolution. Added channels start
at exact zero in the pretrained stem, preserving the original three-channel
function before optimization. MiT-B0, ResNet-18, and Swin-T provide three
materially different pretrained global hierarchies under one common
four-scale categorical decoder. This is an architecture test, not a new data
source: no public prediction, competition checkpoint, held-fold recipe,
target-derived feature, F4 value, or confirmation grouping enters the family.

# A910: gate trajectory corrections by their inference-visible shape

A895 established that A830 contains a large but sparse correction: applying
epoch-12/temperature-0.75 only to the 20% of F0 wells on which accepted D570
moved farthest from protected v20 improves D570 by 0.145 ft. A subsequent F0
mechanism diagnostic found a stronger label-free discriminator. The absolute
linear slope of `A830 - D570` selects a top decile whose fixed 0.25 share
improves F0 by 0.239 ft. This is not an oracle quantity: it is computed from
the two predictions available at inference and is fixed before any joint
F1-F3 metric is opened.

A910 preregisters a compact treatment bank around that mechanism. It retains
all six C898 treatments, then ranks complete wells by (1) absolute candidate
correction slope, (2) absolute candidate endpoint correction, (3) D570-v20
RMS movement, or (4) the product of candidate-correction RMS and D570-v20 RMS.
The exact retained fractions are:

- candidate slope: 5%, 10%, 15%, 20%, 25%, and 30%;
- candidate endpoint: 10%, 15%, 20%, 25%, and 30%;
- parent RMS: 10%, 15%, 25%, and 30% (20% is already C898);
- product RMS: 10%, 15%, 20%, 25%, and 30%.

Ranks break ties by well ID. Snapshot, temperature, share, and treatment are
selected for each held development role using only the other three roles, with
the existing every-selection-role non-regression constraint and four-metric
rank aggregation. The source and metric-free replacement/invariance gate must
be committed before the complete F0-F3 selector is run. F4 and both
confirmation regroupings remain sealed.

# A915: require low-frequency directional agreement between two model stages

The A910 F0 analysis exposed a more mechanistic confidence signal than
disagreement magnitude. Accepted D570 is itself a learned correction from
protected v20. A830 supplies a second correction from D570. If their endpoint
and complete-well linear-slope corrections have the same sign, two independently
trained stages agree on the low-frequency direction; a large positive product
is therefore a target-free confidence score. On F0, the top 15% by endpoint
agreement improves D570 by 0.285 ft. Intersecting the top 15% endpoint and
top 15% slope agreement selects six wells and improves D570 by 0.451 ft at
share 0.50.

A915 freezes that direction-consistency mechanism before reading any joint
F1-F3 correction metric. It appends top 10/15/20/25/30% masks for endpoint
agreement and slope agreement, plus all nine intersections of endpoint
15/20/25% with slope 15/20/25%. Every score uses only v20, D570, and A830
predictions for the same role. Fractions are recomputed independently within
each inference role, ties break by well ID, and the existing held-excluded
selector chooses the treatment, snapshot, temperature, and share from the
other three roles. Exact zero remains available and acceptance still requires
positive contribution after D570 on every development role.

# A920: project only direction-confirmed corrections onto coherent shapes

A890 showed that target-free low-frequency projection improves A830's noisy
rowwise disagreement, but its all-well F0 gain did not clear its expansion
bar. A915 changes that context: its endpoint/slope agreement intersections
isolate a few wells on which both learned stages corroborate the same coherent
direction. On the fixed epoch-12/temperature-0.75 F0 diagnostic, the
endpoint-15%/slope-15% intersection gains 0.485 ft after D570 using the raw
correction. Applying A890's already-audited origin-constrained cubic transform
at share 0.75 gains 0.535 ft, while its four-control piecewise-linear transform
at share 1.0 gains 0.542 ft.

A920 therefore preserves all 45 A915 treatments and appends exactly cubic and
four-control piecewise transforms for each of the nine registered endpoint
15/20/25% by slope 15/20/25% intersections. Transform coefficients use only
the candidate disagreement within a well; the direction masks use only v20,
D570, and A830 predictions. The 63-treatment bank retains A915's snapshots,
temperatures, shares, held-excluded four-metric selector, non-regression
eligibility, tie breaks, and exact D570 fallback. No F0-selected recipe is
transferred to another role.

# A930: estimate correction utility, not the hidden geology

A915/A920 use hard top-fraction masks to identify wells where two independently
learned stages move in the same low-frequency direction. That mechanism is
deliberately robust, but it discards the continuous geometry of the candidate
correction and forces one share on every selected well.

A930 preregisters a narrower supervised task before A920's joint F0-F3 result
is known. For each immutable A830 candidate and its raw, cubic, or piecewise4
correction, inference-visible complete-well summaries describe correction
magnitude, endpoint, slope, curvature, frequency concentration, protected
parent movement, signed direction agreement, correction-parent correlation,
length, and stability across the frozen snapshot/temperature bank. Training
wells supply only the analytic scalar that minimizes their D570 residual along
that fixed correction direction, clipped to `[0,1]`.

For a held development role, model and recipe choice is itself cross-fitted
inside the other three roles. Only an actuator that is non-regressive on every
inner validation role may be refit on all three roles and applied to the held
role. Ridge and one shallow robust tree family are the complete registered
capacity bank; exact zero and fixed-share controls remain eligible. This does
not attempt to predict a residual path from metadata, which the earlier
visible-signature regressors falsified. It asks only whether the reliability
and safe amplitude of an independently generated A830 direction can be
estimated from that direction's own observable structure.

## A920 clean result

The one-shot held-excluded A920 evaluation rejects the hard confidence and
projection bank. It changes D570 from `5.820599496` to `5.890624631`, a
`-0.070025135` increment; held-role increments are `+0.009016`, `-0.282206`,
`+0.026725`, and `-0.009544`. The other-three-role selector never transfers
one of the F0 direction-intersection projections and instead chooses simple
parent-movement masks. The large F0 result was therefore a role-specific
mechanism diagnostic, not a stable ensemble improvement. Both confirmation
regroupings and F4 remain sealed.

# A940: route among diverse fold-safe solutions at whole-well scale

The aligned development archives expose a different kind of headroom. Relative
to D570 at `5.820599496`, an oracle that chooses D570 or the pure-CE
stratification path separately for each whole well reaches `5.481843926`.
Greedily adding the protected mode bank, protected posterior replacement, and
protected top-k route lowers that optimistic ceiling to `5.208053319`, a
`+0.612546` gain. The first three routes already supply `+0.583853` of
headroom. These values use targets and are diagnostics only.

A940 freezes those four routes before fitting a selector. They are materially
different: a trained stratified categorical path, posterior-mode selection,
posterior replacement, and a conservative top-k posterior rule. For each
well, the deployable features summarize each route's correction from D570,
their pairwise agreement, their consensus, D570's movement from v20, and row
count. The supervised targets are only the four routes' whole-well SSE gains
over D570, not a residual path.

Every held-role router is doubly excluded. Models are first cross-fitted inside
the other three roles to select one registered estimator, safety margin, and
route share while requiring non-regression on each inner role. The chosen
router is then refit on all three and applied to the held role. Exact D570 is
always available. The confirmation regroupings remain sealed until this or a
later frozen ensemble clears the primary `+0.50` gate.

## A940 result and A950 repair

A940 improves three roles strongly but is not safe: its pooled increment after
D570 is `+0.074372`, with role increments `+0.193232`, `-0.121392`,
`+0.054446`, and `+0.183656`. The F1 router sends 137 of 155 wells to the
pure-CE route, compared with 87, 80, and 53 in the other roles. This is a
label-free distribution-shift symptom, even though F1 truth confirms that it
is harmful.

A950 freezes one repair. For every fitted router, candidate-route proportions
come only from predictions on its selection-training roles. A registered
multiplier converts the median training-role proportion into a per-route cap,
bounded by 50%. Only the highest predicted-gain wells for that route survive;
the rest return to exact D570. Inner role cross-fitting chooses the multiplier
along with model, margin, and share. Neither the held feature distribution nor
held targets define the cap.

A950 does not repair transfer. Its pooled increment after D570 is only
`+0.002618`; held increments are `+0.148395`, `-0.334934`, `+0.034814`, and
`+0.193662`. F1 retains only 48 pure-CE wells and returns every other well to
D570, but those supposedly highest-utility wells are still strongly harmful.
The failure is therefore not merely route prevalence. Further work must test
whether the route is observably out of distribution or remove it; another
numeric cap on the same predictions is not justified.

# A960: let observed GR judge route geometry and normalize amplitude

The four routes still have substantial conditional headroom in every
development fold. Their calibration differs: the pure-CE correction RMS is
`1.162`, `2.451`, `1.046`, and `2.373` ft over F0-F3. Although the full route
regresses F1 and F3, one role-global clipped scalar remains positive in all
four roles. A950's hard routing ignored this continuous amplitude mismatch.

A960 adds evidence that is available for every competition query. Visible
TVT_input calibrates typewell GR to horizontal GR, after which each candidate
path predicts the hidden horizontal GR trace. The already audited protected
mode-bank scorer supplies MAE, Huber, correlation, derivative-MAE, and
derivative-correlation over five smoothings and three strides. For D570 plus
the four routes, min/median/mean/std within each metric family gives exactly
100 new features. A circularly shifted hidden-GR cache is retained only as a
negative-control provenance record.

The second control caps each route's correction RMS independently per well at
one of five frozen levels before applying the selected share. Targets and SSE
statistics are recomputed analytically for that cap. All model, cap, margin,
and share choices remain cross-fitted inside the three selection roles. The
real GR cache is built and committed without reading the truth array before
the single clean target evaluation.
