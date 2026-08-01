# CV Protocol v5 governance snapshot

The three reproduction subtrees preserve the *artifacts*. This directory
preserves the *process* that produced them: the rules that made v20 immutable,
that quarantined the original C016 mode-bank lineage, and that require explicit
user approval before a frozen parent changes.

- Source repository: `rogii_v20_limited_query`
- Snapshot commit: `c34b92124faad762b469aba6408f00f224addf29`
- Snapshot date: 2026-08-01
- Protocol: `cv-protocol-v5` revision 14, 8 active amendments
- Baseline: `protected-v20`, `6.189484768154311`

## This is a snapshot, not a mirror

The query ledger, the frozen-parent registry, and the amendment set are **live**
files. Research continues in the source repository — at the snapshot commit it
was on batch 5, run numbering epoch 2, with 4 of 5 outer reveals spent. Those
files will move. What is here is what they contained at the snapshot commit, so
the `c016/` and `run4/` records can be read against the rules they were actually
written under.

For the live state, read `rogii_v20_limited_query` directly.

## What COOKBOOK.md is

The agent working procedure: explore freely on F0-F3, keep new target-derived F4
evidence hidden, freeze one complete pipeline, evaluate it across all five outer
folds once, deploy only selection-clean improvements. Its seven non-negotiable
rules are the reason this repository is shaped the way it is — rule 6 ("never
modify the protected v20 checkpoints or OOF vaults") is why C016 and Run4 layer
on v20 instead of touching it, and rule 7 ("never use public-leaderboard
feedback to select") is why the frozen-parent registry requires user approval
after reviewing CV-LB consistency rather than promoting automatically.

## What is packaged

55 hash-locked files, closed over the reference graph rooted at
`limited_query_cv/active_cv_protocol.json`:

| Group | Contents |
|---|---|
| Startup surface | `COOKBOOK.md`, `limited_query_cv/CV_PROTOCOL_V5.md`, and the four machine sources the policy names: `cv_protocol_v5.json`, `cv_query_ledger_v5.json`, `frozen_previous_best_v5.json`, `verify_active_cv_protocol.py` |
| Amendments | the 8 active amendment policies, their documentation (`V5_FOLD_ENSEMBLE_AMENDMENT.md`, `V5_POST_PROMOTION_CHECKPOINT_REFINEMENT.md`, `V5_MANUAL_ANALYSIS_FIREWALL.md`, `V5_KICKSTART_GOAL_AMENDMENT.md`, `V5_BATCH3_SINGLE_REVEAL_AMENDMENT.md`, `V5_BATCH3_RUN2_FAMILY_EXTENSION.md`, `V5_BATCH5_BOUNDARY_EXTENSION.md`, `V5_BATCH6_SINGLE_REVEAL_AMENDMENT.md`), and the batch authorizations they reference |
| Boundaries | `v5_batch5_target_free_boundary.json`, `v5_batch6_representative_boundary.json`, and the two builders that wrote them |
| Source reviews | the per-run `SOURCE_REVIEW.md` files the ledger cites, including run 25 (C016) and batch-3 run 2 |
| Run-time scripts | `resolve_v5_goal.py`, which allocates a run and writes its `goal_registration.json` and `kickstart_prompt.txt` |

All 55 match the source repository at the snapshot commit byte for byte.

## Verification

```bash
python protocol/verify_protocol_snapshot.py verify
```

Checks:

- all 55 hash-locked files;
- the startup surface the policy declares, and that every named file is present;
- each active amendment, including the recorded sha256 of its policy,
  documentation, and authorization targets;
- the frozen-parent chain — that `frozen_previous_best_v5.json` still records
  `v20 -> c016 -> run4`, that parent activation is not automatic, and that each
  entry's `selection_clean_pooled_oof_rmse` **agrees with the matching subtree's
  provenance manifest**: `6.189484768154311`, `6.1272727841976`,
  `6.066500604210546`.

That last check is the point of keeping the snapshot next to the
reproductions. If a subtree's numbers and the governance registry ever diverge,
this fails.

## The vendored checker does not run standalone

`limited_query_cv/verify_active_cv_protocol.py` is included because the cookbook
names it as startup reading and the policy lists it as a machine source. It will
**not** pass here:

```text
FileNotFoundError: protocol/limited_query_cv/runs/v5_batch3_run_004_goal_020/
                   run4_frozen_parent_alignment.json
```

It validates live run state — kickstart prompt capture, goal registrations, and
alignment records across the 35 run ids the ledger references — which is the
whole research tree, not a governance surface. Reproducing that would mean
copying the entire source repository. Run it there:

```bash
cd /geo/geo_new/rogii_v20_limited_query
python limited_query_cv/verify_active_cv_protocol.py
```

`verify_protocol_snapshot.py` covers the parts that are checkable without the
run tree, plus the cross-check against the subtrees that the original cannot do.

## Not vendored

`COOKBOOK.md` documents an optional Kaggle idea cache under `/nvidia-kaggle`:
the `discussions.db` and `kernels.db` SQLite databases, the cached notebook
tree, and the read-only query scripts in
`/nvidia-kaggle/skills/nvidia-kaggle-skill/scripts/`. The cookbook is explicit
that this is a hypothesis source only — "not trusted scores or artifacts" — and
that missing cache data must not block research. It is external tooling, not
part of any solution, so it is referenced rather than copied.
