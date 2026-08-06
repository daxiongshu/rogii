# Batch 3 Run 2 source review

Date: 2026-07-30

## Hypotheses supplied by the teammate

1. `dZ/dMD` has its strongest relation to TVT at an approximately 300-row
   negative shift. The geometry suggests that steering reacts to delayed
   feedback.
2. `ANCC` is absent for one cluster represented in test, while `BUDA` is
   complete and is a better surface carrier.
3. Wells can be clustered by exact overlap of their inference-visible
   typewell TVT/GR traces. Wells in one cluster share a typewell datum.
4. Within a cluster, `TVT + Z - ANCC` is constant. The formation surfaces are
   parallel within a well, so anchor-relative BUDA differences can express
   the same inference-time geometry without requiring test BUDA.

Local schema checks support the missingness premise: seven training wells
have all-missing ANCC and no training well has all-missing BUDA. The
horizontal test schema contains only `MD`, `X`, `Y`, `Z`, `GR`, and
`TVT_input`; BUDA and ANCC are not test columns. Label-free exact-overlap
matching reconstructs 54 typewell clusters among 773 training wells and
matches all three public preview wells to a training cluster.

## Related cached kernel

The locally cached `dott1718/ka-e80-v2-s65` code already implements:

- a spatial BUDA P-spline surface fitted from pooled training wells;
- anchor reconstruction using
  `TVT_i = TVT_anchor + (BUDA_i-Z_i) - (BUDA_anchor-Z_anchor)`;
- exact-overlap typewell clustering and role-aware cluster enrichment;
- delayed Z-shape features with shifts near 250 rows;
- a GR-consistent trajectory pool around a surface baseline.

Its published public score (`8.223`) and documented CV (`6.054`) are not
selection evidence. Its full-data `buda_pool.npz`, `cluster_enrich.pkl`, and
tuned constants are forbidden because they are not V5 role-purged. The
useful transfer is the physical mechanism, rebuilt locally around the much
stronger clean C016 parent.

## Clean V5 transfer

The new family is `c016_buda_delayed_dip_cluster_surface`.

- Every held prediction starts from the fold-safe promoted clean C016 parent.
- A BUDA surface is fitted only from wells legal to the training role.
- Held-well inference reads only `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`,
  inference-visible typewell TVT/GR, and the fold-safe parent prediction.
- A held well's BUDA, ANCC, and hidden TVT are never loaded by its feature or
  prediction path.
- Typewell identity may be matched label-free, but horizontal target/surface
  contributions are restricted to role-training wells.
- F4 target-derived evidence remains sealed before global freeze and the
  protected joint reveal.

The family first tests the components separately, then their combination:
role-purged BUDA surface, delayed-dip residual, cluster-conditioned surface,
and a conservative C016-relative stack. Exact C016 is always present as a
zero-share option.

No Kaggle Dataset, kernel, or competition submission is authorized.
