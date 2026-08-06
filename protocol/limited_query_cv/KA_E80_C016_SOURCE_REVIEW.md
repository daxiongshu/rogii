# `ka-e80-v2-s65` source review for a C016 residual-path candidate

Date: 2026-07-30

## Exact source inspected

- Kernel:
  `https://www.kaggle.com/code/dott1718/ka-e80-v2-s65/notebook?scriptVersionId=338390204`
- Kaggle reports `currentVersionNumber=1`, so the linked historical script ID
  and the only available kernel version cannot refer to different source
  revisions.
- Kernel source SHA256:
  `fd02d9607355cbe389f0c68078556ec7f5af1139e21571974062f712e848d403`
- Attached private Datasets:
  - `dott1718/ka-e80-v2-s65-code`
  - `dott1718/ka-e80-v2-s65-weights`
- Important attached-file hashes:
  - `beam.py`:
    `9938dc820c21ba6709edf9b08afd1532ded801df3252e170fa2d71fe118a501c`
  - `beam_refine.py`:
    `cde35a3fcd9d88b2a9ee20cbfaea6c8d0e8496f7e0bdd672924dd52b7fb28b63`
  - `cluster.py`:
    `7a8ec59585ea240766884579c37102cb854c2316447c20f32f6c1c65f99705ca`
  - `buda.py`:
    `90fa467306e1d9d7f317e5534e6cd9d4632d3ec0a6fd4899e334086a784a9df8`
  - `buda_pool.npz`:
    `837725d87b5f262336a29a696949a940dc2766300f0a3c4726ddc394e9635a16`
  - `cluster_enrich.pkl`:
    `877cc255c028ebaa2affc81fc7c4a44ed4854dfddf59c66082f172e8393a4380`
  - `config.json`:
    `46c7ae080f112acc84abecc8ce407818944f5296976c178cbbd384c7c211f8ca`

The kernel completed on one T4 and wrote 14,151 finite, unique predictions in
about 26 seconds for the three-well preview test. Kaggle's kernel-score API
reports public RMSE `8.223`. The author's documented final three-fold,
150-well CV is `4.907 / 7.451 / 5.444`, pooled `6.054`. The approximately
2.17-ft CV/LB reversal, large fold spread, repeated reuse of the same 150
wells, and the author's measured approximately 0.15-ft beam-seed noise mean
that neither the reported CV optimum nor its selector constants are eligible
evidence for our pipeline.

The downloaded kernel prediction differs from the promoted C016 pseudo-test
prediction by relative RMSE `2.595854` over the same 14,151 rows. This is
label-free diversity evidence only, not a blend-selection result.

## Mechanism

The submission is not a neural model. It:

1. estimates a spatial formation surface from the training-only `BUDA`
   column;
2. enriches each typewell GR map with `(TVT, GR)` observations from training
   wells whose typewell traces identify a common datum;
3. generates 10,000 momentum-biased stochastic paths in a small residual
   coordinate system around the surface baseline;
4. Gaussian-smooths each residual path;
5. averages distinct low-GR-loss paths after adding baseline-divergence,
   heel-slope, and Z-shape terms.

The important insight is not the weak BUDA predictor. It is the separation of
two jobs:

- generate a diverse set of smooth, GR-consistent residual trajectories
  around a strong baseline;
- choose or average among them conservatively while retaining an exact
  baseline no-op.

This differs materially from our previous attempts:

- Run 5's particle/beam paths started from the heel rather than searching a
  residual around the promoted C016 trajectory. Their best same-F0 blend was
  `+0.038944`, but standalone RMSE was `15.256311`.
- The historical residual-GR dynamic program added only `+0.010013`.
- Run 6 A420 produced `+0.100892` on F0 but its single-route selector did not
  transfer: clean four-fold gain was `-0.015927`.

The sourced method addresses those failures with a much stronger C016 prior,
a large diverse path pool instead of one Viterbi path, residual-space
smoothing, and probability averaging rather than brittle route selection.

## Legal transfer and exclusions

The proposed candidate must be rebuilt locally; do not import either attached
binary artifact.

- Baseline: the fold-safe promoted C016 prediction.
- Inputs: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`, and typewell `TVT`/`GR`
  only.
- Optional same-datum enrichment may use evaluation-zone `(TVT, GR)` pairs
  only from wells in that role's training set. Every excluded/held well must
  be downgraded to its visible `TVT_input` prefix before it can contribute.
- Rebuild typewell-identity clusters per role. Never reuse
  `cluster_enrich.pkl`, because its full-data contributions are not
  fold-purged.
- Exclude the BUDA surface, `buda_pool.npz`, and every other training-only
  geology/surface column. Its direct kernel has poor LB alignment, and the
  active cookbook does not admit those columns into the inference contract.
- Do not use the kernel's public score to choose a parameter or blend weight.
- Do not copy the heavily tuned `config.json` values as validation evidence.

## Proposed falsification-first family

Name: `c016_residual_gr_path_pool`

Training-free implementation controls:

1. Exact no-op: zero correction weight and a zero residual path must reproduce
   C016 bit-for-bit.
2. Hidden-target firewall: original, reversed, and all-NaN hidden TVT must
   produce identical candidate pools.
3. Synthetic recovery: a planted smooth residual path must be recovered
   materially better with its aligned GR trace than with a circularly shifted
   GR trace.
4. Fold purge: replacing excluded-well hidden TVT must not change any map,
   path, selector feature, or prediction for that excluded role.
5. Numerical control: smooth anchor-relative residuals, never absolute
   approximately 11,000-ft TVT values, and compare CPU FP64 with GPU output.

Minimum meaningful variants:

1. `self_prefix`: C016 prior plus the well's visible-prefix GR calibration,
   with no cross-well label transfer.
2. `datum_pool`: add fold-purged, same-typewell-datum training observations.
3. `datum_pool_z`: add the inference-visible Z-shape term to the same path
   pool.

All variants should generate a shared paired-seed path pool. Candidate
selection must include C016 itself and conservative shares
`0, 0.025, 0.05, 0.10, 0.15, 0.20`. The real-GR route must beat its
circular-shift negative control before the family expands. A small positive
component is not rejected merely for missing the final readiness goal; it may
join a final ensemble only under the cookbook's removal-necessity and clean
incremental-contribution rules.

After a metric-free gate, optimize on F0-F3 only, use held-excluded recipe
selection, verify the two preregistered F0-F3 regroupings, and freeze one
global pipeline. If a protected batch is authorized later, rebuild the
fold-purged pools for all ten unordered excluded-pair roles and five outer
roles, freeze all predictions, and reveal once. Never submit a Kaggle
competition entry without a separate user instruction.

## Current protocol status

This source review opened no new target-derived metric. The active V5 ledger
has no remaining protected batch (`2/2`) and no remaining protected reveal
(`2/2`), and the goal resolver therefore cannot register a compliant new
research run. New F0-F3 scoring is paused pending an explicit protocol
amendment or a new protocol/batch authorization.
