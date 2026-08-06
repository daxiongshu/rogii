# Run 3 source review

Earlier local work tested absolute categorical U-Nets and hand-built GR/Viterbi
paths. The most encouraging hand-built screen gained 0.145 RMSE on only 12 F0
wells, improved 6/12, and was not expanded. A later clean cross-fit of a large
hand-built residual-Viterbi recipe bank regressed by 0.016 RMSE.

The unresolved mechanism is narrower: center the state space on the frozen
Run4 path, learn the horizontal/typewell GR compatibility surface from
role-training wells, and decode a globally smooth residual path. This removes
the absolute-path burden from the network and makes exact parent fallback a
zero residual.

Before training, an F0-F3-only oracle must separately measure residual-state
coverage, smooth-path representational capacity, and target-free GR evidence.
The family stops without F4 if those diagnostics show no useful headroom.
