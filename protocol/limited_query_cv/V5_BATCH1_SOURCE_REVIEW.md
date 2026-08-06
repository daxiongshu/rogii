# V5 batch 1 source review: physics-generated path likelihood

## Sources

The source material came from the locally cached NVIDIA Kaggle discussion and
kernel indexes for `rogii-wellbore-geology-prediction`.

- [ML to learn generator for forward simulation](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702474)
  proposes sampling physically constrained complete TVT paths, forward
  simulating their GR from the typewell, and learning a likelihood or SDF
  scorer. It also observes that trajectory geometry supplies most of the prior
  while GR supplies a smaller steering signal.
- [Is the sub-6 regime end-to-end learned, or engineered alignment?](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/727149)
  reports that explicit alignment is most useful as a trust or ambiguity
  signal and that long cycle-skip regions can be genuinely multimodal.
- [Six independently-trained architectures, same blind spot](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/726834)
  shows multiple direct regressors converging to the same wrong mode over long
  intervals, motivating complete-path alternatives rather than another
  pointwise residual model.
- Cached kernel `canqiang/rogii-det-mha120sep4mpkg10` uses a bounded
  multi-hypothesis midpoint hedge and likelihood-PF features. Its title's
  `MHA` is not evidence for a new attention architecture; the transferable
  mechanism is uncertainty-aware path handling.
- Cached kernel `hengck23/cnn-sdf-example` confirms the original dense SDF
  representation. The local
  `SCRATCH_COMPLETE_WELL_VALIDATION.md` already showed that directly
  regressing this SDF generalizes poorly, so V5 will not repeat that objective.

## New hypothesis

The protected v20 stack already supplies a strong complete-well path. Instead
of predicting TVT again from scratch, generate a compact set of physically
smooth perturbations around that path and ask which complete trajectory best
explains the observed horizontal GR under the typewell forward model.

The candidate basis operates in structural-surface coordinates
`U = TVT + Z`. Every perturbation is zero at the visible boundary and is a
combination of:

- a rapidly introduced datum adjustment;
- an end-to-end slope adjustment;
- a middle-tail bend adjustment.

This preserves v20's high-frequency trajectory while exposing the
low-frequency datum, dip, and curvature degrees of freedom identified by the
discussion oracle ladder.

## Registered family batch

One mechanism is evaluated through three independently scored families:

1. `physics_path_energy_control`: a training-free robust forward-GR energy
   with fixed temperature and physics-prior grids;
2. `physics_path_catboost_ranker`: an engineered candidate-feature ranker;
3. `physics_path_conv_ranker`: a one-dimensional convolutional scorer over
   observed GR, simulated GR, mismatch, gradients, trajectory, and candidate
   correction.

The analytic family is a required control. The two trainable families satisfy
the V5 minimum of two candidates for a protected batch. Each trainable family
contains at least three meaningful registered configurations.

## Boundary from earlier failures

This is not the rejected surface U-Net or scratch SDF model:

- it starts from protected v20 rather than a flat or random path;
- the candidate output space is a bounded low-dimensional physical library;
- it scores whole candidate paths rather than predicting a dense contour;
- its final prediction is a conservative posterior path or v20 blend;
- all model and ensemble choices for outer fold `h` use only its other four
  folds.

No protected outer metric is opened during this review or family
registration.
