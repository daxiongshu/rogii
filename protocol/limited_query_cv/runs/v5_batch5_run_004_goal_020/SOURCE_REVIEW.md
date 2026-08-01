# Run 4 source review

The formation columns imply a strong geological identity: within a typewell
cluster, `TVT + Z - formation_surface` is constant to CSV rounding. ANCC has a
cluster-specific missingness hole, while BUDA and the other formation surfaces
remain complete.

This identity does not create six independent observations—the columns are
offset copies of the same contact surface. The useful unresolved question is
therefore geometric: can a role-purged interpolation use distinct neighboring
well trajectories more accurately than the existing global local-polynomial
and nearest-path averages?

This run tests per-row moving least squares over the nearest *distinct wells*.
The held well contributes only its inference-visible trajectory and prefix;
all formation surfaces and hidden TVT from that well remain unavailable. The
existing retained contact correction is the incremental comparison point.
