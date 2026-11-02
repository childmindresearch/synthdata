# Model card template

Every evaluation writes one model card per candidate model to `model_cards/<model>.md`, next to `report.md` (set `evaluation.model_cards: false` to skip them). Each card is linked from the top of the report and from the run dashboard. The layout follows Model Cards (Mitchell et al., *FAT\** 2019), cut down to what a synthetic-data release needs. Every value is read from the run's own output files, so a card never says more than the report. The cards are run outputs: they describe your data, so share them only where you would share the report.

The template below shows what each section holds. `{...}` marks a value filled in from the run; the file each one comes from is in brackets. The section headings are fixed (a unit test keeps this page and `synthdata/evaluation/model_cards.py` in step).

```markdown
# Model card: `{model}`

Written by synthdata {version} on {date}, from this run's outputs. The evidence behind every number is in report.md.

## Model details

- Generator: `{generator}`; {how its settings were chosen: untuned, tuned (best trial),
  or tuned but no trial passed the screens} [hpo_best_params.json]
- Seed replicates: {n} (base seed `{seed}`); synthetic rows per replicate: {rows}
- Trained on: the train split of `{dataset}` (version `{version}`), target `{target}`
- Experiment: `{experiment id}`
- Provenance: commit, uncommitted changes, uv.lock and config hashes, key package
  versions [the experiment's manifest.json]
- The tuned hyperparameters, as JSON (tuned models only)

## Intended use

What the synthetic data is for (comparing generators, developing analysis code,
teaching) and what it is not for (clinical decisions, reporting results about the
real population unchecked, release before a privacy review). Fixed text.

## Ranking

Rank, overall score with its 95% interval, the type scores, the gap to the best
model with its interval and whether it is tied, and whether the model is on the
utility-privacy Pareto front [ranking_summary.csv]

## Primary metrics

One row per primary metric: mean value over replicates, the two baseline anchors
(which baseline scores 0 and which 1) and the anchored score [primary_metrics.csv]

## Utility

TSTR and TSTS macro AUPRC and Cohen's kappa for the model and both baselines, and
the TSTS minus TSTR gap [tstr_holdout.csv]

## Privacy

Anonymeter risk per attack for the model and both baselines [privacy_attacks.csv]

## Fairness

Holdout fairness gap errors and balanced accuracy for the model and both
baselines [fairness_gaps.csv]

## Known issues

Every output check that warned on one of the model's replicates (including the
dependency-rules check), metrics that failed or were skipped, and a search that
found no passing trial [checks.csv, combined_evaluation.csv]

## Caveats

Scores are relative to this run; the privacy evidence is not a formal guarantee;
the intervals' sources of noise; inherited bias. Fixed text.
```
