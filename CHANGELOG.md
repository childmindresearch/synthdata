# Changelog

What changed in each release, for people running the pipeline. Every pull request bumps the version in `pyproject.toml` and adds its entry here, following [Semantic Versioning](https://semver.org/): while the version starts with 0, the middle number goes up for new features or changes that need a config edit or change results, and the last number for fixes that don't.

Each entry says what a user sees change, which config keys or output files are new, renamed or removed, and whether old results stay comparable (a change to a metric or generator needs a new experiment id and a re-recorded golden run).

## 0.8.2 (2026-10-19)

Results are unchanged: no new experiment id or golden re-record is needed (no numerical library moved).

### Removed

- `scripts/document_pipeline/` (old PII and Markdown-parsing scripts, not part of the pipeline). Git history keeps it. This closes both CodeQL alerts.
- The `ydata` extra and `notebooks/ydata-test.py`. `ydata-synthetic` 1.4.0 pins TensorFlow 2.15, Keras 2, an old `requests`, `pytest<8` and old Jupyter, and caused most of the Dependabot alerts. `uv sync --extra ydata` no longer works.

### Fixed

- Security updates in `uv.lock` for packages with known vulnerabilities, among them aiohttp, pillow, gitpython, cryptography, urllib3, tornado, jupyterlab, notebook, jupyter-server, starlette, protobuf and pytest (7 to 9). numpy, pandas, scipy, scikit-learn, xgboost, torch and the generators stay at their old versions.
- `python-dotenv` is now a direct dependency: `synthdata/__init__.py` imports it, but it was only installed through other packages.

## 0.8.1 (2026-10-16)

### Fixed

- TabPFN no longer refits to top up a class that fell short of its quota (`generation.match_class_prior`). Each top-up was a whole new fit, about 10 h on wide data. The class now stays short, recorded in `diagnostics/class_sampling.csv` and flagged by `checks.csv`, as for any other model after its last batch. Other models are unchanged.

## 0.8.0 (2026-10-16)

Results are not comparable with 0.7.x: use a new experiment id.

### Added

- `data.protected_bins`: bin edges for numeric protected columns, `{column: [edges]}`. The fairness gap metrics compare each bin with the rest, and log disparity and the HPO protected-group screens use the same bins (the screens fall back to train quartiles without them).
- SynthEval's fairness gap metrics (`statistical_parity`, `equal_opportunity`, `equalized_odds`) now run when a protected column has more than two groups: each group with at least 5% of the rows is compared with the rest and the gaps are averaged. Before, they failed on every model unless a protected column had exactly two values.
- TSTS (train on synthetic, test on synthetic) columns `tsts_macro_f1`, `tsts_balanced_accuracy`, `tsts_macro_auprc` in `tstr_holdout.csv`; reported, not ranked.
- `evaluation/metric_directions.csv` and `evaluation/metrics_{utility,privacy,fairness}.png`: every metric with its mean and 95% interval per model, whether higher, lower or closer to an ideal value is better, and the ideal value.
- `docs/metrics-guide.md`: what every metric means, in plain language.

### Changed

- `report.md` shows one row per model (mean and 95% interval over replicates) instead of one row per replicate, and embeds the metric plots.
- Trade-off plots show one labelled point per model with its interval; replicates are faint unlabelled dots.
- The dashboard shows the one-vs-rest SynthEval passes as one step.
- Data readiness checks leave out the `__missing` indicator columns.
- `evaluation.log_disparity.protected_bins` defaults to `data.protected_bins`; set it only to use different bins for log disparity.

### Removed

- TRTR (train on real, test on real) rows in `tstr_holdout.csv`: they equalled TSTR on `baseline_train_copy`, which stays.
- The median split of numeric protected columns: a numeric protected column without `data.protected_bins` is left out of the gap metrics, with a warning.

### Fixed

- HPO under `generation.hpo.stage_time_limit_hours` now leaves time for the refits of each model's top trials (`refit_top_k` x `refit_seeds`) as well as its final fits; before, the refits ran after the search's own deadline and could push the stage past the limit. Refits that would not finish in time are skipped.
- HPO `patience` no longer counts the random start-up trials, so a study can't stop right after them; the stop message reports the true best value.
- `report.md` said data readiness and imputation validation had not run when they had (it looked in the wrong folder).

Changes before 0.8.0 are in the pull requests and git history.
