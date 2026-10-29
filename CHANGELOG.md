# Changelog

What changed in each release, for people running the pipeline. Every pull request bumps the version in `pyproject.toml` and adds its entry here, following [Semantic Versioning](https://semver.org/): while the version starts with 0, the middle number goes up for new features or changes that need a config edit or change results, and the last number for fixes that don't.

Each entry says what a user sees change, which config keys or output files are new, renamed or removed, and whether old results stay comparable (a change to a metric or generator needs a new experiment id and a re-recorded golden run).

## 0.27.0 (2026-10-29)

TabularARGN and CDTD are now on by default. Results of the other models are unchanged, but a run with the default config now also trains these two, so use a new experiment id. The golden runs need re-recording on the Linux GPU machine, because the fixture's baseline run now uses miceforest.

### Changed

- **TabularARGN and CDTD on by default** (`generation.tabularargn.enabled` and `generation.cdtd.enabled` are now `true`), and the example configs list both. `synthdata-run` installs their environments on first use. To run `synthdata-generate` on its own, or on a machine without internet, install them first (`uv sync --project envs/tabularargn`, `uv sync --project envs/cdtd`) or set `enabled: false`.
- **Integration tests follow the current defaults.** The fixture's baseline run now uses miceforest, the default imputer, and new checks cover the class quotas (`diagnostics/class_sampling.csv` and the synthetic class shares), the holdout fairness gaps (`fairness_gaps.csv` and the combined table) and the TSTR kappa in `tstr_holdout.csv`. MissForest stays covered by the imputation-only run. The fixture's DCR screen is set to 0.75: it is measured on about 40 tuning rows there, so the default 0.6 rejected sound trials by chance.
- **One slow test module for the two models with their own environments**, `tests/integration/test_sidecar_generators.py`, which replaces `test_tabularargn.py` and `test_cdtd.py`. It trains both in one run, as the default config does, and checks their class quotas and ranking. With `SYNTHDATA_REQUIRE_SIDECARS=1` a missing environment fails the tests instead of skipping them.
- **`scripts/scheduled_tests.sh`** runs the tiers CI skips on a Linux GPU box: the sidecar tests and the golden check. Its header has a weekly cron line.
- **Leaner unit tests:** config tests that only checked that an accepted value passes are now one table-driven test, two exact duplicates were removed, and two overlapping HPO screen tests were merged (one fewer HPO scorer build).

## 0.26.0 (2026-10-29)

Changes metric values and the ranking, so re-evaluate with a new experiment id (generation is unchanged) and re-record the golden runs on the Linux GPU machine. No config edits are needed. Every verification row that was Open has been re-checked against results and fixed, taken out of the ranking or the defaults, or accepted with a reason; `docs/verification.md` has the checks and numbers.

### Fixed

- **synthcity `max_mean_discrepancy`** gave every synthetic set that was not a copy the same score (about 2/n) on wide data, because its kernel used gamma = 1 on unscaled columns. Columns are now scaled on the real data and the kernel width is the median-distance heuristic (Gretton et al. 2012).
- **synthcity `prdc`** computed its neighbour radii on unscaled columns, so only the widest columns counted; it now uses the same scaling.
- **synthcity `jensenshannon_dist`** gives each category of a categorical column its own bin (codes of columns with more than 10 categories were merged).
- **synthcity performance metrics** use the macro (per-class) AUROC for 3+ classes instead of the micro average, and the XGBoost model gets the intended depth 3.
- **SynthEval `p_mse`** codes categorical columns as indicators, so a change in category shares is seen.
- **SynthEval `corr_diff`** uses the correlation ratio eta (it was eta squared), and a constant synthetic column no longer makes the whole metric blank.
- **SynthEval `mia_precision`**: an attacker that flags no row now scores as no skill instead of the worst privacy, and only precision above chance counts as risk.

### Changed

- **Reported but no longer ranked:** synthcity `ks_test` (SynthEval's ks_test covers it), the synthcity sanity distance scores and `common_rows_proportion`, and SynthEval `mia_recall`.
- **Off by default** (run only when named in `evaluation.synthcity.metrics`; they were already unranked after failing the known-answer checks): the three augmentation metrics, `detection_gmm`, the k-anonymity family (`k-anonymization`, `k-map`, `distinct l-diversity`, `delta-presence`) and `data_leakage_linear`/`data_leakage_mlp`. Evaluation runs fewer model fits.

## 0.25.1 (2026-10-29)

No result or config changes; no golden re-record is needed.

### Fixed

- Loading stopped with `data.ordinal_column_categories[...] does not include observed value(s) ['False', 'True']` for an ordinal True/False column with blanks. Since 0.24.0 such a column is loaded as the text "True"/"False", while its schema order `[false, true]` is read as true/false values; the ordinal encoding now accepts both forms.

## 0.25.0 (2026-10-29)

Changes the fairness score and so the ranking; re-evaluating an existing experiment is enough (generation is unchanged). The golden runs need re-recording on the Linux GPU machine: models that had blank fairness gap errors now have values.

### Changed

- **Every model gets holdout fairness gap errors.** From 0.20.0, a model whose synthetic-trained classifier did not beat chance on the holdout had blank errors and was left out of the fairness score. That gate was not needed: such a classifier predicts much the same for every group, so its errors come out about as large as the real gaps, the score of a dataset that erased the disparities. Leaving those models out let the independent-marginals baseline be ranked on log disparity alone (it came first overall in a simulated-data check) and made models flip between replicates. `fairness_signal` and the balanced-accuracy interval stay in `fairness_gaps.csv`, reported but not ranked.
- **`config_sim.yaml` uses the default TSTR XGBoost** (`evaluation.xgboost_params: null`) instead of parameters a collaborator tuned for another model. A run with this config needs a new experiment id.

### Added

- **`scripts/tune_tstr_xgboost.py`** tunes the TSTR XGBoost on the real train rows outside the tuning split (Optuna, macro AUPRC, patient-grouped stratified folds), scores the defaults and the configured parameters on the same folds with the gain's standard error, and prints the `evaluation.xgboost_params` line to paste. See the user manual, "Utility: holdout TSTR".

## 0.24.0 (2026-10-29)

Changes the default generators and the raw data for some columns, so use a new experiment id. No golden re-record is needed: the golden runs use neither PATE-GAN nor True/False columns with blanks.

### Changed

- **PATE-GAN is off by default.** It is no longer in the default `generation.synthcity.names` or in the shipped `config_sim.yaml`, `config_loris.yaml` and `config_hepatitis.yaml`. Its output currently collapses: most columns hold a single value, and some are blank in every row. Add `pategan` back to the list to run it.

### Fixed

- **True/False columns with blanks now match across stages.** pandas reads such a column as Python bools, but every CSV the pipeline writes and reads back holds the text `"True"`/`"False"`. The raw split and the imputed and synthetic data therefore had different labels, so `column_differences.csv` scored the column 1 for every model, the column topped every `<model>_top_columns` figure with real and synthetic bars side by side, and a run that imputed in the same process saw other labels than a resumed one. These columns are now read as `"True"`/`"False"` text from the start.

## 0.23.2 (2026-10-28)

No config changes and no golden re-record needed. Imputed values change only for binary columns where the imputation model is certain (a predicted probability of exactly 0 or 1), which used to stop the run.

### Fixed

- miceforest imputation no longer stops with `ValueError: data must be finite, check for nan or inf values` on a binary column with a very rare level. LightGBM then predicts a probability of exactly 0 or 1, and miceforest's log-odds for mean matching became infinite. The probabilities are now clipped to [1e-12, 1 - 1e-12] before the log-odds, which keeps their order.

## 0.23.1 (2026-10-28)

No result or config changes; old results stay comparable and no golden re-record is needed.

### Fixed

- miceforest imputation no longer stops with `LightGBMError: Do not support special JSON characters in feature name` when a column name contains `"`, `,`, `:`, brackets or braces. The imputer now gives LightGBM positional names (`c0`, `c1`, ...) and maps the results back, so output column names are unchanged.

## 0.23.0 (2026-10-28)

`configs/config_sim.yaml` now uses simulated dataset v1.3 (experiment id 1.3); results are not comparable with v1.2 runs.

## 0.22.1 (2026-10-28)

Dashboard and log only; results stay comparable with 0.22.0 and no golden re-record is needed.

### Changed

- **Models timeline labels say where a model is.** While the run goes on, the label names the step: *synthcity running*, or *generation done* between steps. Only once the run has ended does it say *done*, and it is green only when nothing failed. Failed metrics turn it amber (*done, 12 metrics failed*), and a failed step turns it red (*synthcity failed*). Before, PATEGAN showed a green *done* after its synthcity and SynthEval metrics failed.
- **HPO search table:** the median trial time stays after the search ends, and the last column shows the search's total time (first trial's start to last trial's end) instead of the time left once it is over. With `objective: auto` the main study's trials are copies of the probe's, stamped at copy time, so the median was lost once the search finished. It is now read from the probe study.
- **Resources by task:** each generation fit and each HPO search logs what it used (`[ctgan__rep2] fit took 42 s (peak RAM ..., CPU ... cores on average, peak GPU memory ...)`, `[ctgan_hpo] HPO search took ...`). Once a generation worker has ended, these rows replace its single row, so replicates (×3) and tuned fits (`ctgan_hpo`) show apart, and the search shows under *HPO search*, as they already did for the evaluation steps.

## 0.22.0 (2026-10-28)

Renames synthetic datasets, so use a new experiment id: cached sets under the old names are not reused. The generated data and the metrics are otherwise unchanged (same seeds per replicate). The golden runs need re-recording on the Linux GPU machine for the new row names (`ctgan` is now `ctgan__rep1`) and because tuned models whose search found no passing trial are no longer rows.

### Changed

- **Replicates are named `<model>__rep1`, `<model>__rep2`, `<model>__rep3`.** The first replicate used to keep the plain model name next to `__rep1` and `__rep2`, which read as one replicate too many. Even a single replicate is now `<model>__rep1`.
- **The TabPFN variant fitted on data with missing values is `tabpfn_standard_raw` / `tabpfn_custom_raw`** (was `tabpfn_standard` / `tabpfn_custom`), next to `tabpfn_standard_imputed` / `tabpfn_custom_imputed`. The config value (`generation.tabpfn.data_variants: [raw]`) is unchanged.
- **All generation runs in parallel.** TabPFN (each variant and data variant), TabPFGen, TabularARGN and CDTD now get a process of their own in the same pool as the synthcity models, instead of running one after another in the main process after them. TabPFN starts first, since its fits take longest on wide data. TabPFN picks its GPU itself and does not take a GPU slot, so watch its memory on a small GPU.
- **A tuned model whose HPO search found no passing trial is no longer fitted or evaluated.** It would only repeat the untuned model (same settings, same seeds); before, its datasets were saved as copies and every metric was computed twice. The report lists it as *Not ranked* with the reason, and evaluation skips `<model>_hpo` sets an earlier version saved for such a model.
- **Metric overview figures** (`metrics_utility`, `metrics_privacy`, `metrics_fairness`): no figure title (it overlapped the first row of panels; report.md explains the panels), metrics ordered by importance (TSTR and detection first for utility, Anonymeter and distance MIA first for privacy, the holdout fairness gap errors first for fairness; see `METRIC_PRIORITY` in `synthdata/plotting/evaluation_plots.py`), and models in one fixed order (generators by name, each tuned model after its untuned one, baselines last) in these and the rank trade-off plots. The report now embeds all three however tall they are, instead of linking them.
- **Run dashboard, Models timeline:** every model the config will produce is listed from the start, each tuned model on its own row under its untuned one. The untuned row shows the untuned fits, the tuned row the HPO search and the tuned fits, or *none passed*. Fits in the main process count as *generation* (no separate *generation (main process)* step), the baselines' generation shows where evaluation draws them, and SynthEval's main and one-vs-rest or binary-target passes are one *syntheval* step with the later passes shaded. How many models run at once and with how many threads is one line above the timeline rather than repeated per step.
- **Run dashboard, Resources:** tasks are no longer shown as running once the run has ended, and every model is listed with its peak use. Each worker process now logs its own peak RAM, average CPU and peak GPU memory when it finishes (`... finished after 42 s (peak RAM 1.20 GiB, CPU 2.10 cores on average, peak GPU memory 0.50 GiB); ...`), so models that finished between two once-a-minute samples are no longer missing.

### Fixed

- A worker process that had written its result but did not exit (synthcity's Bayesian network: joblib's idle process pool lingers for 300 s) held its slot and showed as still generating for 5 minutes. Workers now shut that pool down, and the parent stops a worker that has not exited 30 s after writing its result.

## 0.21.1 (2026-10-28)

No code or config changes; results stay comparable with 0.21.0.

### Changed

- The golden reference runs (`tests/integration/fixtures/golden/baseline_run_linux_cuda.csv` and `baseline_run_linux_cpu.csv`) are re-recorded on Linux for 0.21.0. They now include the fairness gap-error columns and `fairness_signal` (0.20.0), the detection metrics without `depth` ignored (0.19.0), and generator outputs rounded to the train number format (0.19.0). The CUDA reference had last been recorded before 0.12.0, so it also picks up the subgroup privacy gap columns.

## 0.21.0 (2026-10-27)

No result changes; old results stay comparable and no golden re-record is needed.

### Added

- **`synthdata-run` sets up and checks the separate generator environments first.** When the generate stage is selected and TabularARGN or CDTD is enabled, it checks the needed submodule checkout (`submodules/cdtd_simple`), runs `uv sync --frozen --project envs/<name>` (a no-op once installed) and imports PyTorch in that environment, printing whether it sees CUDA. Any failure stops the run before imputation, with the command to fix it, instead of failing when generation reaches that model hours later. You no longer need to run `uv sync --project envs/<name>` by hand.
- New config key `compute.sync_envs` (default `true`); `false` skips the `uv sync` for offline machines whose environments are already installed.

## 0.20.0 (2026-10-27)

Changes the fairness score and so the overall ranking: rankings from earlier versions are not comparable. Generated data is unchanged, so re-evaluating an existing experiment is enough (no new experiment id needed for generation). The golden runs need re-recording on the Linux GPU machine (new columns and fairness ranks).

### Changed

- **New ranked fairness metrics: holdout fairness gaps** (custom evaluator `fairness_gaps`, on by default). The holdout TSTR classifier is fitted on each synthetic dataset and on the real train split, and both predict the holdout. Per class and protected group, fairlearn gives the demographic parity, equal opportunity and equalized odds gaps, and each score (`demographic_parity_gap_error`, `equal_opportunity_gap_error`, `equalized_odds_gap_error`) is the average distance between the synthetic and the real gap: 0 keeps the real disparities, while erasing or inverting them scores high. When the synthetic-trained classifier does not beat chance on the holdout (bootstrap 95% interval of balanced accuracy), the scores are blank, `fairness_signal` is 0, and the model is left out of the fairness score instead of being scored worst.
- **SynthEval's statistical parity, equalized odds and equal opportunity are reported but no longer ranked.** Their classifier is fitted and scored inside the synthetic data, so a dataset with no class signal predicted one class and scored a perfect 0. On the simulated clinical dataset, ARF, TabularARGN, DDPM and the marginals baseline did, and CDTD, the one generator with a learnable signal, ranked last on fairness. With the new scores, TabularARGN keeps the real gaps best (demographic parity error 0.056), then ARF (0.077) and CDTD (0.110); DDPM and the marginals baseline are gated.
- New output files `fairness_gaps.csv` (per model) and `fairness_gaps_by_group.csv` (every gap per class and group), a new report subsection, and new `combined_evaluation.csv` columns under `custom/fairness`.
- New dependency: `fairlearn`.

## 0.19.0 (2026-10-27)

Changes generated data and the ranking: use a new experiment id, since cached synthetic sets keep their old format and old rankings counted detection. The golden runs need re-recording on the Linux GPU machine (detection values and rounded generator output move).

### Changed

- **Synthetic numbers get the train format.** Every numeric (non-categorical) column of every generator's output is clipped to the train min and max and rounded to the train column's decimal places, as SDV does by default (rdt's `FloatFormatter` rules, reimplemented in `synthdata/generation/numeric_format.py` and checked against rdt 2.0.0). Counts that are whole numbers in train stay whole numbers: ARF wrote decimals into them in about 95% of rows. No config key; categorical columns are not touched.
- **XGBoost and MLP detection are reported but no longer ranked.** On the simulated clinical dataset XGBoost detection was 1.000 for every generator and the marginals baseline (so it rewarded only the copy of train), and MLP detection scored the marginals baseline like a copy. Both stay in `combined_evaluation.csv`; the linear detector stays ranked.

### Fixed

- synthcity fork: `detection_xgb` passed `depth: 3`, which XGBoost ignores, so its trees grew to depth 6. It now passes `max_depth: 3`.

## 0.18.0 (2026-10-27)

Default generator list and a TabularARGN default change. Runs that keep the default `generation.synthcity.names` or TabularARGN params produce different models, so use a new experiment id; the golden runs do not use ddpm or TabularARGN and stay valid.

### Changed

- **ddpm is off by default.** It is no longer in the default `generation.synthcity.names` or in the shipped `config_sim.yaml`, `config_loris.yaml` and `config_hepatitis.yaml`. In the untuned simulated-data comparison it still learned the columns independently (its correlation and mutual-information errors matched the independent-marginals baseline, and its TSTR was below it). Add `ddpm` to `generation.synthcity.names` to run it. CDTD (`generation.cdtd`) is the diffusion model to use instead.
- **TabularARGN value protection is off by default** (`generation.tabularargn.params: {value_protection: false}`). With it on, mostlyai-engine 2.7.3 can still write its `_RARE_` token even with `rare_category_replacement_method: sample`, and the run then stops with "tabularargn made categories ... not in train". The reason and the repro are in `docs/verification.md`.

## 0.17.0 (2026-10-22)

Imputed values change, so results are not comparable with earlier runs: use a new experiment id and re-record the golden runs.

### Changed

- New default imputer `imputation.method: miceforest`: MICE with LightGBM forests and predictive mean matching (the `miceforest` library, new dependency; LightGBM capped below 4.7). Each gap gets a real observed value from a row with a close prediction, so filled columns keep their spread and category shares instead of MissForest's narrow spikes and single-class fills. Settings in `imputation.miceforest` (`iterations`, `mean_match_candidates`, `num_threads`). `missforest` is still available. The example configs now use `miceforest`. Run `uv sync` to install it.
- Real-vs-synthetic figures (`plots/generation/<model>.png`, `<model>_top_columns.png`) and `column_differences.csv` now compare the real train values that were recorded with the synthetic values as released (blanked where the synthetic missing indicator is 1), so imputed values appear on neither side. Panel titles show each side's share of blanks. The old filled-vs-filled view, which is what the metrics score, is `<model>_filled.png` (written when there are missing indicators).

### Added

- `missing_blocks.csv` (written by `synthdata-impute`, or on its own by the new `synthdata-missing-blocks --config ...`): columns that go missing together, and a candidate label for each missingness pattern (every column with a gap) (`skip_logic_candidate`, `form_not_given_candidate`, `item_missing`) with the decision-tree rule behind it, to review against the codebook.
- `imputation.missing_indicators.blocks` (`{name: [columns]}` or `"auto"`, default `{}`) and `block_min_jaccard` (default 0.9): one `<name>__missing` flag per block, 1 where the whole block is missing; the released copy blanks the whole block there. Column flags then count only the gaps outside whole-block rows.
- `dependencies.csv` from the new `synthdata-dependencies --config ...`: to help slim the dataset, an exact search over every column pair, with no cutoffs and nothing dropped. It covers functional dependencies, with a chance-corrected score so identifier-like columns are labelled `id_like` instead of passing as rules. It covers one-condition rules, which catch skip logic, kept when a Bonferroni-corrected binomial test says chance can't explain them. It also covers hierarchies, every identifying set of role columns (Desbordante's HyUCC) and every numeric column's linear R². Each finding has a suggested action. `--max-error` (default 1%) only picks the readable shortlist; `dependencies_pairs.parquet` and `dependencies_rules.parquet` hold everything. `--max-lhs 2` (opt-in, hours) adds two-column rules. Desbordante runs in its own environment (`uv sync --project envs/desbordante`; Linux or macOS, AGPL-licensed, so synthdata never imports it).
- `synthdata-profile --config ...` writes `missing_blocks.csv`, `dependencies.csv` and `skip_logic_comparison.csv` (skip logic found by the blocks report's tree and by conditional FDs, side by side). It needs `envs/desbordante` (Linux; WSL on Windows) and stops with an error without it; `--blocks-only` skips the Desbordante part.
- `evaluation.dependency_rules` (default `null`): a file of accepted rules (rows of `dependencies.csv` or `missing_blocks.csv` with a `check_rule`). Evaluation writes `dependency_violations.csv` and adds a `dependency_rules` row per model to `checks.csv`, warning when a model breaks the rules more often than the real holdout (plus 1 percentage point).

### Fixed

- Categorical axes in the plots sort labels that read as numbers by value (1, 2, ..., 10), even when they are stored as text; before, they sorted as text (1, 10, 11, 2). Schema `ordinal_order` still wins.

## 0.16.2 (2026-10-22)

No result or config changes; old results stay comparable and no golden re-record is needed.

### Fixed

- A config that turns on no generator now stops at once with an error that names the keys to turn on. `synthdata-run` checks before any stage starts (so imputation is not run for nothing) whenever the `generate` stage is selected, and `synthdata-generate` checks before loading data. Before, such a run did all of imputation first and only stopped after logging `Generated/loaded 0 synthetic datasets: []` (on versions before 0.9.0 it went on to evaluation and failed on an empty experiment folder). Runs with only the `impute` or `evaluate` stage are unaffected.

## 0.16.1 (2026-10-21)

No result or config changes; old results stay comparable and no golden re-record is needed.

### Fixed

- Security update in `uv.lock`: certifi 2026.2.25 to 2026.10.19 (GHSA-6q48-6c7w-9gw2).
- Still open, because a fix needs a pinned package to move: torch 2.7.1 and monai 1.5.2 (torch pin), pyarrow 21 (anonymeter needs pyarrow below 22, fix is 23.0.1), transformers 4.57 (be-great and mamba-ssm need 4.x, fixes are in 5.x) and datasets 2.14 (be-great pin, fix is 5.0.1). The `envs/cdtd` and `envs/tabularargn` lock files have no known vulnerabilities.

## 0.16.0 (2026-10-20)

A new optional generator; nothing changes unless you turn it on, so existing results and golden runs stay comparable.

### Added

- CDTD (continuous diffusion for mixed-type tabular data, ICLR 2025) as `generation.cdtd` (`enabled`, `env_dir`, `source_dir`, `params`, `fit_params`, `sample_params`, `pool_factor`; off by default). The authors' `cdtd_simple` code is the new submodule `submodules/cdtd_simple` (run `git submodule update --init`); it needs Python 3.13 and a newer torch, so it has its own environment (`uv sync --project envs/cdtd`) and runs as a subprocess. Writes `cdtd.csv` (plus replicates). A reference test checks that our runner reproduces the authors' notebook row for row; deviations are in `docs/verification.md`.

### Changed

- The subprocess plumbing TabularARGN used is now shared (`synthdata/generation/sidecar.py`), and progress bars from these subprocesses are logged at most once a minute.

## 0.15.0 (2026-10-20)

A new optional generator; nothing changes unless you turn it on, so existing results and golden runs stay comparable.

### Added

- TabularARGN (MOSTLY AI's autoregressive generator, `mostlyai-engine==2.7.3`) as `generation.tabularargn` (`enabled`, `env_dir`, `params`, `sample_params`; off by default). Its package pins torch 2.13 and `psutil<6`, which clash with this project's, so it lives in its own environment (`uv sync --project envs/tabularargn`) and runs as a subprocess. It fits on the imputed train split, writes `tabularargn.csv` (plus replicates), has no HPO, and fills class quotas by fixing the target as seed data. See the user manual (Generators) and `docs/verification.md`.

## 0.14.0 (2026-10-20)

TabPFN results change: use a new experiment id for runs with `generation.tabpfn.enabled`. No golden re-record is needed (the golden run has no TabPFN models). No config key changes.

### Changed

- TabPFN generation averages 1 permutation per column instead of tabpfn-extensions' default 3. Upstream's 3 permutations are 3 identical fits (the permuted column order is dropped and the seed is fixed), so this changes no results in distribution and cuts TabPFN generation time to about a third. A Hepatitis comparison, including 3 seeded permutations with the order fixed, found no metric difference beyond chance: `docs/verification.md`, row "tabpfn permutations".
- TabPFN's classifier and regressor, and the standard variant's label classifier, are seeded with each replicate's seed instead of TabPFN's fixed `random_state=0`, so replicates no longer share their fitted models and the replicate spread covers the model's own randomness.

## 0.13.0 (2026-10-20)

The simulated and LORIS configs change HPO selection: use a new experiment id for them (old Optuna studies are stale). The hepatitis config, and so the golden run, is unchanged; no re-record is needed.

### Added

- `generation.hpo.constraints.enforce` (default `true`). With `false`, every HPO screen is still measured, stored on each trial (`screens`, `screen_details`, and the new `screens_enforced`) and logged (`trial N fails screens [...] (logged, not enforced)`), but no trial is marked infeasible, so the best score wins and no search falls back to the default settings. The dashboard shows failed screens in the trial tooltip and counts only enforced failures as infeasible.
- `checks.csv` gains `screen_missing_classes`, `screen_category_coverage`, `screen_protected_coverage`, `screen_protected_share_gap` and `screen_dcr_closer_to_train` rows: the HPO screens measured on every final dataset, with train as the fitted data and the holdout as the unseen rows, whether or not the search enforced them.

### Changed

- `configs/config_sim.yaml` and `configs/config_loris.yaml`: `hpo.constraints.enforce: false`, and `hpo.objective: tstr_macro_auprc` for every model instead of `auto` (which chose three different objectives on the sim run). The sim config also sets `hpo.tstr_seeds: 3`, since its `evaluation.xgboost_params` make one TSTR fit seed-dependent. Why AUPRC rather than macro-F1, with evidence from the sim run's study database: `docs/verification.md`, HPO objective section. The sim config also sets `hpo.min_delta: 0.01` (about 1.5 bootstrap SEs, since seed noise matched the bootstrap SE on the sim run) and raises the TVAE and RTVAE epoch ranges to 25-300 (their untuned models now train 300 epochs) with timeouts raised to match.
- Class quotas: a short class the generator makes too rarely for the remaining batches to fill a tenth of it no longer sizes the top-up batches, and top-ups stop once every short class is like that (logged with each class's rate). Before, such a class drove batches of up to 10x the train size for nothing (PATE-GAN on the simulated data).
## 0.12.0 (2026-10-20)

Existing metrics and ranks are unchanged; two new reported (unranked) columns appear, so the golden runs gain those columns (Linux CPU golden updated; the Linux CUDA golden needs a re-record). No new experiment id is needed; old evaluations simply lack the new columns.

### Added

- Subgroup privacy gap (fairness of privacy): the holdout distance membership attack (`distance_mia_auc`) is now also run inside each protected group, using the same groups as SynthEval's fairness metrics (each category or `data.protected_bins` bin with at least 5% of the train rows). `combined_evaluation.csv` gains `subgroup_distance_mia_auc_gap` (largest AUC difference between groups of one protected column; 0 = equally exposed) and `subgroup_distance_mia_auc_max` (the most exposed group's AUC) under custom fairness. Both are reported, not ranked: a copy of train leaks equally in every group, so the gap cannot tell it from a fresh draw. It runs whenever `holdout_distance` is selected and protected columns are declared; there is no new config key.
- `report.md`: a new **Privacy by protected group** subsection under Fairness explains the two columns and shows each group's AUC per model and its patient counts; the privacy section links to it.
- `privacy_subgroups.csv` in the evaluation folder: each model x protected group with its train and test patient counts, AUC and standard error (Hanley & McNeil). Groups with fewer than 10 patients on either side get no AUC.

## 0.11.0 (2026-10-20)

ddpm results change: use a new experiment id for runs that include ddpm. No golden re-record is needed (the golden run has no ddpm).

### Fixed

- ddpm no longer collapses. Its sampler (the TabDDPM reference code, which synthcity ports) never clipped the predicted clean numeric values, so small denoiser errors grew at every reverse step until numeric values were in the thousands on a scale where real ones lie within about ±5. The inverse transform hid this by piling those values on each column's minimum or maximum, and the categorical part, fed them, produced a few categories only (the simulated run's ddpm trials covered 23% of a column's common categories and missed protected-group shares by 0.81). The fork now clips the predicted values to each column's training range at every sampling step. On a test table the fix took the per-column category share error from 0.28-0.47 to 0.04 (sampling noise 0.02). Details in `docs/verification.md`, ddpm row.
- ddpm's learning rate now decays per optimizer step, as in the TabDDPM reference, instead of per epoch.
- The synthcity fork also carries its `wasserstein_dist` fix (runs without KeOps).

## 0.10.1 (2026-10-20)

Results stay comparable: no new experiment id is needed, and a run can be resumed with its own experiment id. No golden re-record is needed (the fixture is far below the size where geomloss switched to KeOps, so its Wasserstein value is unchanged).

### Fixed

- `stats.wasserstein_dist` (synthcity fork) runs on geomloss's plain-torch backend instead of switching to KeOps above 5000 x 5000 points, so it works where KeOps cannot compile. When it does fail, it no longer leaves autograd off for the worker, which used to make `alpha_precision`, `identifiability_score`, `detection_mlp` and `data_leakage_mlp` fail too. Configs that dropped `wasserstein_dist` from `evaluation.synthcity.metrics` as a workaround can put it back.
- Generation refits a cached synthetic CSV that has no rows, no target column or cannot be read, with a warning, instead of passing it to evaluation. Evaluation skips an empty CSV with an error naming it rather than failing every metric on it.
- When no HPO trial passes the screens, `<model>_hpo` is saved as a copy of the untuned sets instead of being refitted with the same settings and seeds (about 2 h per replicate for ddpm on the clinical data). With `hpo.final_n_iter_override` set, the training length differs, so it is still refitted.

## 0.10.0 (2026-10-19)

Results are unchanged with the default config. A config that sets the new key changes the HPO objective and holdout TSTR: use a new experiment id. No golden re-record is needed (the golden config does not set it).

### Added

- `evaluation.xgboost_params`: your own `XGBClassifier` parameters for the classifier that predicts the target in holdout TSTR/TSTS, the HPO objective and its real-data ceiling, for example the parameters of a baseline model tuned on the real data. Keys you leave out keep the defaults; `random_state`, `seed`, `objective` and `num_class` are refused, and class-balanced weights still apply. SynthEval's and synthcity's utility metrics, `fid_detection_auc` and the privacy attacks keep their own classifiers. `configs/config_sim.yaml` sets it to a collaborator's tuned baseline parameters as an example.
- `manifest.json`: the evaluation entry (and the generation entry when HPO runs) records the classifier parameters used as `xgboost_params`.

## 0.9.0 (2026-10-19)

Results stay comparable and the golden run is unchanged: no new experiment id is needed, and a failed run can be resumed with its own experiment id. The one change to generated rows: a class top-up that asks for more than 10,000 rows at once is now drawn in chunks with their own seeds, so a model that needed one draws different (equally valid) rows than before.

### Added

- `generation.continue_on_model_failure` (default `true`): when one model's search, fit or sampling fails, the others carry on and evaluation runs on every model that finished. Before, one failed model (a PATE-GAN out-of-memory error after 36 h on the simulated data) stopped the run before evaluation. Failed models are listed in the new `<generation dir>/generation_failures.json` and under **Read with care** in `report.md`; the generate stage fails only if no synthetic dataset is left. Set it to `false` for the old behaviour.
- GPU memory is shared by measurement: a fit starts on a GPU only when the memory its model needs (its largest measured peak plus a margin, or a quarter of the GPU before its first fit) fits next to the fits already there. This applies with `compute.fits_per_gpu: auto` and with a number. `gpu_slots/fits.csv` gains a `peak_vram_gib` column (new files only).
- Dashboard: each HPO study that stopped short says why (*no gain in N trials*, *time budget*, *stage time limit*, *generation failed*); the objective and its range sit under each trial plot and in each trial's tooltip (the separate Objective column is gone); hovering a generation bar lists its phases (untuned fits, HPO search, tuned fits) with their lengths instead of repeating the model name.
- HPO logs `search ended at its time budget` when a study's timeout ends it.

### Fixed

- A model's process now hands its cached GPU memory back after every fit. Before, PyTorch kept each worker's highest use for the rest of its search (TVAE and RTVAE held 20 GiB each between trials), which left too little for the other models.
- Sampling draws at most 10,000 rows per call, and a fit or chunk that runs out of GPU memory is retried up to 3 times with the same seeds after the GPU frees up.
- Dashboard: hovering a hollow dot (a trial that failed a screen) now shows its tooltip; only the dot's outline reacted before, so models where no trial passed showed nothing.

## 0.8.2 (2026-10-19)

Results are unchanged: no new experiment id or golden re-record is needed (no numerical library moved).

### Removed

- Legacy code that was not part of the pipeline, was no longer maintained and caused most of the security alerts. The tag `legacy-archive` marks the last commit that has it; restore a path with `git checkout legacy-archive -- <path>`.
- `scripts/document_pipeline/` (old PII and Markdown-parsing scripts). This closes both CodeQL alerts.
- `apps/presidio/` (the Presidio PII Streamlit app) and the `presidio` extra, which pulled in an old `transformers` and `sqlitedict`. `uv sync --extra presidio` no longer works.
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
