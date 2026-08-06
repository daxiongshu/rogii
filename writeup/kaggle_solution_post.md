# A Distributional U-Net for Whole-Well Geological Alignment

**Jiwei Liu · Team Kaggle Agent**

This write-up covers my part of our solution to the [ROGII Wellbore Geology Prediction competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction). Teammate components are intentionally outside its scope.

Our central decision was to stop treating each hidden row as an independent regression problem. A well follows one continuous geological path, so we modeled the complete hidden tail as a structured alignment between the well trajectory, the gamma-ray (GR) trace, and the typewell. The final system had three layers:

1. a **Core alignment** ensemble that predicted a distribution over TVT states along the entire well;
2. a **Residual refiner** that learned a small, bounded correction to that path; and
3. a **Routed ensemble** that applied sparse structural or cross-well surface corrections only when inference-visible evidence supported them.

| Stage | Five-fold whole-well OOF RMSE | Public LB RMSE | Private LB RMSE |
|---|---:|---:|---:|
| Core alignment | 6.1895 | 6.263 | 6.985 |
| Residual refiner | 6.1273 | 6.127 | 6.918 |
| Routed ensemble | **6.0665** | **5.962** | **6.729** |

The improvement was gradual, but it was consistent: every fold improved from the first stage to the final one, and the same ordering held on both leaderboard splits. The private set was harder in absolute RMSE, but the final system still improved by `0.256` over Core alignment there. More importantly, every prediction used by validation was generated without reading that well's held-out target.

## 1. Reframing the target as a geological path

For a formation surface `S`, the TVT path of a well can be written conceptually as:

```text
TVT(md) = S(X(md), Y(md)) - Z(md) + D_typewell
```

Here `md` is measured depth, `(X, Y, Z)` is the well trajectory, and `D_typewell` is the reference depth convention. This view immediately gives us useful structure:

- neighboring rows belong to one smooth trajectory;
- the visible prefix anchors the hidden tail;
- GR/typewell agreement supplies local alignment evidence; and
- nearby wells provide weak information about the same formation surface.

The competition mask makes the path interpretation especially natural. The visible prefix and hidden tail meet at one physical anchor, while the trajectory continues smoothly through it.

![A real training well under the competition mask: visible prefix, hidden tail truth, and the complete trajectory relative to the last visible row.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/real_target_path.png)

*The upper panel shows TVT relative to the visible anchor; the blue curve in the lower panel is the well's geometric trajectory, `TVT + Z`, relative to the same anchor.*

Instead of predicting a scalar independently at every row, we discretized possible TVT states and asked the network to align the full well with that ordered state axis.

## 2. Turning a well into an alignment volume

Each example is a two-dimensional tensor with axes **candidate TVT state × sampled position along the well**. We sampled 704 positions:

- 64 positions from the visible prefix;
- 640 positions from the hidden tail; and
- a stride of 16 original rows between sampled positions.

The usual local state grid covered `-64 ft` to `+64 ft` at `0.5 ft` spacing, giving `S = 257` states. A wide variant covered `-120 ft` to `+120 ft` at `1 ft` spacing, giving `S = 241`.

The network received 19 channels. They were not 19 arbitrary tabular features; each channel was broadcast or rasterized over the state-position grid so the model could reason jointly about local evidence and global path geometry.

| Channel group | Count | Contents |
|---|---:|---|
| GR alignment evidence | 4 | signed and absolute mismatch to the global typewell; signed and absolute mismatch to a prefix-derived pseudo-typewell |
| State-grid support | 2 | visible-prefix support and normalized candidate-state coordinate |
| Observation masks | 3 | visible path heatmap, visible-position mask, and missing-data mask |
| Anchor-relative position | 4 | relative `Z`, measured depth, `X`, and `Y` |
| Trajectory geometry | 6 | `dZ/dMD`, `dX/dMD`, `dY/dMD`, sine and cosine of azimuth, and vertical curvature |
| **Total** | **19** | |

![A real 19-channel alignment volume around the last visible anchor.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/real_alignment_volume.png)

*The target is a path through this volume. GR agreement creates local evidence, while masks, coordinates, and trajectory channels tell the model how that evidence fits the whole well.*

### GR as alignment evidence

GR was most useful when treated as a signal to align, rather than as one more scalar regressor input. At every candidate state we sampled the corresponding typewell response and compared it with the observed well GR. The signed mismatch retained direction; the absolute mismatch retained match quality.

We also built a pseudo-typewell from the visible prefix. This made the input less dependent on global calibration and gave the model a well-specific reference. During synthetic training we randomized GR gain and offset, injected noise, removed intervals, and introduced false cycles. The result was a model encouraged to use the *shape and continuity* of GR evidence instead of memorizing its absolute scale.

## 3. A compact distributional U-Net

The Core alignment model was a compact two-dimensional U-Net:

- `3 × 7` convolutions to mix nearby TVT states and a wider span of measured depth;
- GroupNorm and SiLU activations;
- channel widths from 8 to 64; and
- about 1.77 million parameters in the standard configuration.

Its output shape was:

```text
B × S × 704 logits
```

For every sampled position, the `S` logits described a categorical distribution over ordered TVT states. Training used the nearest state as the classification target, but inference decoded the posterior mean back to a continuous TVT offset. In that sense, this was a regression problem solved through ordinal, distributional classification.

The standard objective was approximately:

```text
cross-entropy
+ continuous posterior-mean regression
+ 0.001 × surface smoothness
```

One ordinal specialist also used:

```text
+ 5 × CDF distance
```

The cross-entropy located probability mass, the regression term improved sub-bin accuracy, and the smoothness term discouraged implausible jumps along the path. CDF distance additionally respected the ordering of the states: confusing adjacent bins was less severe than placing mass far away.

## 4. Synthetic-to-real training

The labeled dataset was small relative to the variety of plausible whole-well geometries. We therefore generated smooth geological paths and forward-sampled typewell GR along them. The generator varied:

- dip, curvature, and local excursions;
- GR gain, offset, and noise;
- missing intervals and gaps; and
- distracting or false correlation cycles.

All source traces and typewells came from the competition data; no external geology dataset was used. Synthetic training taught the network how alignment should behave under controlled distortions, and real-data fine-tuning adapted that prior to the exact competition mask.

Most families used a four-epoch synthetic stage. The long-curriculum variants used 16 epochs before fine-tuning. The complete Core alignment ensemble contained 17 families across five whole-well folds: 85 final checkpoints were packaged, 80 had nonzero weight, and five were retained with zero weight for exact historical reproducibility.

![Weights of the Core alignment model families.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/family_weights.png)

*The ensemble diversified state resolution, synthetic generators, capacity, seeds, sampling, and ordinal loss. No single family had to solve every geological regime.*

## 5. Nested residual refinement

The Core alignment path was strong, but its remaining error was not random. We trained a second distributional network to predict a bounded correction around the Core path:

```text
final local search window = Core prediction ± 32 ft
```

Each residual-training batch mixed real out-of-fold residual records and synthetic local residuals 50/50. The real examples taught the model the characteristic mistakes of the frozen parent; the synthetic examples stopped it from learning a narrow lookup table over those mistakes.

This stage used nested whole-well cross-validation. For each protected outer fold, all model and recipe choices were made only with the other four folds. Reciprocal excluded-pair roles let us represent 20 logical inner roles with 10 unique pair trainings. We then trained five outer models, each excluding the fold it predicted.

The deployed residual stage averaged five clean outer paths. Its neural package contained 10 local residual networks—two seeds for each outer role—plus one physical posterior network. This improved pooled OOF RMSE from `6.1895` to `6.1273`, public LB from `6.263` to `6.127`, and private LB from `6.985` to `6.918`.

## 6. Structural candidates from the target well

Even after residual refinement, long coherent errors remained. This suggested a geometry problem rather than row-level noise. Our first candidate bank therefore used only inference-visible evidence from the target well.

We sampled 8,192 smooth particle trajectories on a `±16 ft` grid at `0.25 ft` resolution. Candidates were scored by hidden-tail GR/typewell agreement, with momentum, 150-row smoothing, visible-prefix support, and a 128-row fade from the last visible anchor. The 12 retained reductions spanned top-64 and top-128 particle subsets, temperatures 10 and 30, surface bandwidths 4, 8, and 12, and uniform mean/median summaries.

![Twelve real structural correction candidates for a held-out well.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/real_structural_candidates.png)

*Blue lines are the 12 generated candidates, orange is the held-out residual truth used only for evaluation, and teal is the correction selected by the frozen router.*

The candidates were deliberately conservative near the visible anchor and could express a slowly accumulating correction farther into the hidden tail. They were hypotheses, not automatic edits: the router could abstain.

## 7. Cross-well surface transfer

The second candidate bank asked whether nearby wells supported a shared formation surface. We constructed six role-purged surface-transfer paths from donor pools that excluded every well in the target's held-out fold.

The variants used donor counts of 10, 16, or 24; global or cluster-local pools; a `138°` coordinate rotation selected inside clean development roles; and an unrotated control. An anisotropic quadratic surface was fit to the donor geometry, anchored at the target's last visible row, and allowed to decay with slope lengths of 500 or 1,000 rows.

![A real role-purged donor pool and target trajectory.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/real_surface_context.png)

*The orange line is the held-out target trajectory and contributes no donor BUDA values. Colored paths are real training-well donors; color encodes their BUDA formation-surface value.*

Cross-well transfer can be powerful, but only when the local neighborhood is genuinely informative. It was therefore another candidate source rather than a mandatory correction.

## 8. Sparse utility routing

For each well we built 409 inference-visible features describing the baseline path, GR match quality, trajectory geometry, candidate agreement, donor support, and uncertainty. Two small non-deep-learning routers estimated whether a candidate family was likely to help:

- a 600-tree depth-1 XGBoost model for surface candidates, with a `0.30` cap and `0.20` utility threshold;
- a 700-tree ExtraTrees model with minimum leaf size 16 for structural candidates, with a `0.20` cap and `0.01` threshold.

At most one surface route and one structural route could activate for a well. Surface amplitude was multiplied by `0.625`, structural amplitude by `1.25`, and the complete routed correction was globally scaled by `0.60`.

For outer path `o`, let `R_o` be the residual-refined path and `C_k` one of the 18 candidates. The deployed rule was:

```text
prediction[o]
= R[o] + 0.60 × sum_k ((weight[o,k] + route[o,k]) × (C[k] - R[o]))
```

Most frozen candidate weights were small. The dynamic part could add at most one surface route and one structural route, and either could be exactly zero.

The important behavior was abstention. A candidate was used only when the router found evidence of positive utility; otherwise the stronger parent prediction passed through unchanged.

![Router activation and gain on the five held-out folds.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/router_activation.png)

*The route rate remained sparse. Improvements came from selectively correcting suitable wells, not from perturbing every prediction.*

## 9. Validation and leakage control

All splits were grouped by complete wells. No row from a held-out well could appear in its training data, and the same automatic pipeline ran in every outer context. Hyperparameters, candidate recipes, thresholds, and tie-breaks were selected inside the four-fold inner pool before the protected outer fold was evaluated.

For outer folds `Fi` and `Fj`, the reciprocal inner roles share the model trained on the other three folds. This gives:

```text
20 logical inner roles
10 unique excluded-pair inner trainings
5 outer trainings
```

The public leaderboard was used for monitoring after promotion, not for training or blend selection. Inference loaded GR and well geometry, but it did not read target TVT, ANCC, BUDA, or Geology fields. Cross-well BUDA values were training-donor evidence only and were role-purged during OOF construction.

The Kaggle inference runtime used two T4 GPUs for the neural ensemble and packaged 85 Core checkpoints, 11 residual checkpoints, and five router pairs.

## 10. Results

The final routed correction improved all five held-out folds:

| Fold | Core alignment | Residual refiner | Routed ensemble | Routed gain vs. Core |
|---|---:|---:|---:|---:|
| F0 | 5.9092 | 5.8258 | **5.7400** | +0.1692 |
| F1 | 6.1954 | 6.1199 | **6.0757** | +0.1197 |
| F2 | 5.4032 | 5.3391 | **5.1929** | +0.2103 |
| F3 | 6.2145 | 6.1666 | **6.1168** | +0.0978 |
| F4 | 7.1159 | 7.0713 | **7.0704** | +0.0455 |
| **Pooled OOF** | **6.1895** | **6.1273** | **6.0665** | **+0.1230** |
| **Public LB** | **6.263** | **6.127** | **5.962** | **+0.301** |
| **Private LB** | **6.985** | **6.918** | **6.729** | **+0.256** |

The Core alignment, Residual refiner, and Routed ensemble leaderboard scores correspond to Kaggle submissions `54893107`, `55115834`, and `55147410`, respectively.

![Fold RMSE for the three solution stages.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/fold_rmse.png)

![Per-fold gains from residual refinement and routed geometry.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/fold_gain.png)

Across 11 non-buggy submitted milestones, CV and public LB preserved the exact same rank order (`Spearman ρ = 1.000`). After the competition, the private scores retained nearly the same ordering (`Spearman ρ = 0.991`); the only inversion was between the first two, weakest milestones. The three headline stages remained strictly ordered on private LB, including a `0.189` gain from Residual refiner to Routed ensemble.

The private scores were uniformly higher than the public scores for these milestones, so the private split was harder for this model lineage. The important retrospective result is not the public score alone, but that the leakage-controlled OOF ordering and both refinement gains survived on private LB. The public leaderboard was still used only for monitoring, never for training or blend selection.

![OOF CV and public leaderboard RMSE across submitted milestones.](https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup/plots/cv_lb_history.png)

## 11. What mattered most

The largest lesson was representational. Once the target became a path through an alignment volume, several pieces became natural:

- GR contributed as a correlation surface instead of an isolated scalar;
- categorical state logits represented ambiguity while posterior-mean decoding retained continuous accuracy;
- synthetic wells supplied controlled geological distortions before real-data fine-tuning;
- nested whole-well CV let a learned residual stage use parent OOF errors without leakage; and
- explicit structural candidates made the last correction interpretable and easy to gate.

The final system was not one giant model. It was a sequence of increasingly cautious decisions: predict the whole path, refine it locally, propose a few physically meaningful alternatives, and change the parent only when the visible evidence justified doing so.

The complete reproduction—including training recipes, fold definitions, frozen inference, manifests, and verification commands—is available in the [solution repository](https://github.com/daxiongshu/rogii).

Thanks to ROGII and Kaggle for a competition that rewarded combining signal processing, geometry, careful validation, and machine learning.
