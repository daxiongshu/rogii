# V5 kickstart-configurable development goal amendment

## Status and scope

This amendment is active for V5 batch 2 and later. Revision 9 adds explicit
run-numbering epochs after a user-authorized full artifact purge. It retains
revision 8's literal user-prompt capture, revision 7's automatic
development-run allocation, revision 5's goal semantics, and revision 6's
environment interface. It leaves V5 batch 1, the protected v20 baseline, the
manual-analysis firewall, the nested rerun geometry, and the promotion gates
unchanged.

It supersedes only the assumption that every new agent run must use exactly
`+0.20` as its development goal.

## Prompt-selected goal

A user kickstart prompt may set the run's pooled F0-F3 RMSE-gain goal without
editing or amending any repository file. For example:

```text
development goal: +0.50 RMSE
```

Natural-language equivalents such as “reach +0.5 pooled F0-F3 RMSE gain” are
valid when they contain one unambiguous numeric development target. If the
kickstart prompt contains no target, the default is `+0.20`.

The agent transfers the resolved prompt value to the task-specific environment
variable:

```bash
export ROGII_V5_DEVELOPMENT_GOAL=0.50
```

For nonpersistent shells, prefix the resolver command instead:

```bash
ROGII_V5_DEVELOPMENT_GOAL=0.50 \
  python limited_query_cv/resolve_v5_goal.py --new-run \
  --prompt-save 'LITERAL USER KICKSTART MESSAGE'
```

The environment variable is a runtime interface, not durable evidence. Run
the resolver and preserve its JSON output in the candidate's run directory
before using target-derived development metrics. The frozen goal record, not
ambient shell state, is authoritative for the remainder of the run.

The agent must resolve and report:

```text
requested_goal = explicit prompt target, or 0.20 when omitted
protocol_floor = 0.20
effective_readiness_target = max(requested_goal, protocol_floor)
```

Thus a prompt requesting `+0.20` creates a `+0.20` effective target, while a
prompt requesting `+0.50` creates a `+0.50` effective target. A positive goal
below `+0.20` may guide intermediate research, but it cannot lower the
protected-reveal floor.

## Run registration

Before using target-derived development metrics, the agent records in the run
or batch research log:

- the parsed `requested_goal`;
- whether the value was explicit or defaulted;
- the `protocol_floor`;
- the resulting `effective_readiness_target`.

For readable provenance, the standard resolver invocation passes the literal
user kickstart message through `--prompt-save`. The resolver writes its exact
UTF-8 bytes to `kickstart_prompt.txt` in the allocated run and computes the
SHA-256 itself. This capture is audit metadata only. It is not an
authorization source, validation gate, or substitute for the frozen parsed
goal record.

Only the literal user message belongs in this file. Hidden system, developer,
or composed agent instructions are deliberately excluded. No external digest
is supplied or compared, and a missing or unexpected prompt capture must
never abandon a run, consume a replacement run number, or invalidate its
metrics. Existing revision-7 hash-only records are grandfathered as
historical provenance and are not rewritten.

A clear prompt-selected goal needs no source edit, policy amendment, or
machine-policy hash change. Conflicting or ambiguous numeric goals require
clarification before development begins.

`limited_query_cv/resolve_v5_goal.py` reads
`ROGII_V5_DEVELOPMENT_GOAL`, validates that it is finite and positive, applies
the protocol floor from the consolidated machine policy, and emits the
registration record. Its optional `--goal` argument is a direct prompt-to-tool
fallback. If both inputs are present and disagree, it fails instead of
guessing.

## Batch and run naming

The protected batch number and development-run number are independent.
`batch2` remains fixed until its protected joint reveal is spent or the batch
is closed. Every kickstart inside that batch and numbering epoch receives the
next monotonic `run_###`:

```text
limited_query_cv/runs/v5_batch2_run_001_goal_050/
limited_query_cv/runs/v5_batch2_run_002_goal_050/
limited_query_cv/runs/v5_batch2_run_003_goal_030/
```

`goal_050` means `+0.50`; `goal_030` means `+0.30`. The encoding retains at
least two decimal places and adds more digits when needed.

The user explicitly deleted every directory under `limited_query_cv/runs`
and requested a complete numbering reset. Epoch 1 is archived with a
high-watermark of 9. Epoch 2 starts at zero, so its first allocation is run 1.
Each new registration records `run_numbering_epoch: 2`. `--new-run`:

1. derives the active protected batch from the ledger;
2. advances that batch and numbering epoch's durable run-number
   high-watermark;
3. creates the nonexisting run directory and goal registration;
4. appends the run to `cv_query_ledger_v5.json`.

The new directory and ledger change must be committed together before
target-derived development begins. Deleting a stopped run directory does not
lower the durable high-watermark or cause reuse within the current epoch.
Only an explicit user-authorized full reset with an empty run namespace may
start a new epoch at run 1.

The effective target is fixed for that run. A later explicit user message may
start a new run goal record, but the old target and all trials performed under
it remain in the ledger. Silently lowering the target after seeing development
metrics is forbidden.

## Readiness interpretation

Where revision 4 documentation says `+0.20 readiness bar`, read it as the
default and minimum floor. The numeric gain clause for the active run is:

```text
development_gain >= effective_readiness_target
```

All nonnumeric readiness requirements remain mandatory:

- improvement on at least three of four development folds;
- no development-fold regression worse than `0.03`;
- positive gain and at least `+0.10` under every registered confirmation
  regrouping;
- no more than 60% of positive SSE reduction from one development fold;
- reproducible frozen recipes, selectors, checkpoints, and weights.

The effective target applies to the final batch ensemble. Smaller,
individually sub-target components may still accumulate when they have
positive clean contribution and the final ensemble clears the effective
target.

## Unchanged boundaries

The kickstart goal does not:

- expose F4 or weaken its nonnegative-gain audit veto;
- authorize the sealed nested rerun before readiness;
- change the 20 logical inner roles, 10 unique inner trainings, five outer
  trainings, or five deployed checkpoints;
- change either protected promotion gate;
- authorize a Kaggle Dataset, kernel, or competition submission.
