# CV Protocol v5 — diagram set

Seven 16:9 SVGs explaining the validation strategy the v20, C016 and Run4
subtrees were developed under. Hand-written SVG, no external assets, no
webfonts — every mark that could depend on font coverage (ticks, crosses,
arrows, padlocks) is a drawn path.

Read them in order; each is self-contained.

**01–04 describe the method itself** — no competition, no dataset, no solution
names. They transfer to any grouped-data problem where selection pressure is the
threat. **05–07 are this project's instantiation**, with real figures from the
artifacts and the ledger.

| # | File | Answers | Scope |
|---|---|---|---|
| 01 | [`01-run-workflow.svg`](01-run-workflow.svg) | What happens in one research run, start to finish, and how the loop closes | general |
| 02 | [`02-folds-and-groupings.svg`](02-folds-and-groupings.svg) | Why the split unit is a whole group, what the five folds are, and what the three groupings are for | general |
| 03 | [`03-roles-and-pair-models.svg`](03-roles-and-pair-models.svg) | What a *role* is, and why 20 inner roles need only 10 models | general |
| 04 | [`04-selection-clean-choice.svg`](04-selection-clean-choice.svg) | Ordinary CV against selection-clean CV, side by side | general |
| 05 | [`05-reveal-and-decision.svg`](05-reveal-and-decision.svg) | What is frozen before truth opens, what the reveal reports, how Accept / Hold / Reject is decided | this project |
| 06 | [`06-weights-and-production.svg`](06-weights-and-production.svg) | How a candidate is blended into the parent, and what ships | this project |
| 07 | [`07-query-budget-and-lineage.svg`](07-query-budget-and-lineage.svg) | The economics — 36,450 trials against 5 reveals — and what those five bought | this project |

03 and 04 both work through **F4** as the held fold, so the two read together:
04's "inner roles of F4" are exactly the four roles 03 enumerates.

## The argument across the set

01 is the loop. 02–04 are the machinery that makes one turn of it trustworthy:
whole-group folds, the role system, and the rule that a fold never chooses
anything about itself. 05–06 are what happens at and after the measurement.
07 is why the whole apparatus exists — overfitting is treated as a function of
how often the judge is consulted, so search is free and the judge is rationed.

## Numbers in 05–07 are read from the artifacts

Not illustrative:

- the audit figures in 05 from C016's `package_validation.json`;
- the weight configurations in 06 from `c016_recipe.json` and Run4's
  `recipe.json`;
- trial, reveal and batch counts in 07 from `cv_query_ledger_v5.json`.

All at snapshot commit `c34b9212`, the same commit `../manifests/protocol_sources.json`
hash-locks. The ledger keeps moving in the source repository; these are its
values at the snapshot.

## Colour

One palette across all seven, validated for colour-vision deficiency at
`--pairs all` before anything was drawn: blue `#2a78d6` development and inner
roles, orange `#eb6834` protected and outer roles, aqua `#1baf7a` gains, violet
`#4a3aa7` the human gate, near-black `#2f2e2b` frozen or held-out. Status
colours appear only for Accept / Hold / Reject and always carry a text label,
never colour alone.

## Rendering

Every file was rasterised and visually checked, not just XML-validated — which
is what caught the defects worth knowing about if you edit them:

- the `→` glyph silently drops out in renderers without full Unicode coverage;
  use a drawn arrow marker instead;
- `<text text-anchor="middle">` containing `<tspan>` mis-positions in cairosvg
  and overlaps itself; keep centred text free of tspans.

```bash
python -c "import cairosvg; cairosvg.svg2png(url='01-run-workflow.svg', \
  write_to='01.png', output_width=1600, output_height=900, background_color='white')"
```
