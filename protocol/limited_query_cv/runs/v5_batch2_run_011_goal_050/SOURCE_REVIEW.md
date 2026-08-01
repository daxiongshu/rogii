# V5 batch-2 run-11 source review

## Mechanism

Run 10 established that the selection-clean A960 direction is useful but
under-actuated. Its inference-safe per-well amplitude ensemble improves V20 by
`+0.439251382`, leaving `+0.060748618` to the requested primary gate. A
target-exposed diagnostic on the already-open F0--F3 roles shows large
amplitude-only headroom at both complete-well and fixed 128-row block scale.
Those oracle amplitudes are diagnostics only and are never used as predictions,
features, constants, or selector choices.

The rejected A980 family attempted to choose among four alternative trajectory
paths in each block. Its direction was unstable. Run 11 makes a narrower
prediction: retain the single accepted A960 direction and estimate only its
bounded local strength. The actuator is centered on Run-10's selection-clean
per-well amplitude, so its exact zero-share fallback is the strongest retained
parent rather than raw A960.

## Inference contract

The family reuses the immutable 24,039-block, 220-feature A980 cache. That cache
was built from legal trajectory, GR, path-agreement, block-position, and
prediction-only summaries without loading truth, and passed hidden-target and
circular-shift firewalls. No new target-derived input is added.

For each training block only, the target is the analytic least-squares scalar
for `V20 + amplitude * (A960 - V20)`, clipped to `[0,4]`. Models predict this
scalar from the 220 legal cache features. Three fixed decoders compare raw
block amplitudes, within-well `[1,2,1]` smoothing, and within-well
row-count-weighted constant amplitude. Shares move from the retained Run-10
per-well amplitude toward the decoded block prediction.

## Protected boundary

F4, both confirmation regroupings, and the remaining protected reveal are
sealed. Confirmations may open only if the frozen primary F0--F3 result reaches
`+0.50` and passes every other readiness clause. No Kaggle artifact is
authorized before full V5 promotion, and competition submission remains
forbidden.

