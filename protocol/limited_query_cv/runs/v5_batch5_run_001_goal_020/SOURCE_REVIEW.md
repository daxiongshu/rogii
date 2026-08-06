# Batch 5 trajectory-state residual hypothesis

Source: the teammate's delayed-dip and typewell-cluster observations, plus the
large well-level bias remaining in the active Run4 parent's new F0-F3 errors.

The candidate models the complete hidden trajectory as a sequence. It combines
inference-visible geometry (including shifted `dZ/dMD`), horizontal/typewell GR
agreement, the visible TVT prefix, and fold-safe parent/component predictions.
A global trend head estimates well bias and drift; a recurrent local head models
remaining path variation. No horizontal TVT, ANCC, BUDA, or other train-only
field is an input.

The new Batch 5 folds keep exact typewell clusters intact. This directly tests
whether the mechanism generalizes beyond memorized cluster identity. The old
Batch 4 F4 result is not reused as a fresh gate.
