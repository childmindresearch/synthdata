# Changelog

What changed in each release, for people running the pipeline. Every pull request bumps the version in `pyproject.toml` and adds its entry here, following [Semantic Versioning](https://semver.org/): while the version starts with 0, the middle number goes up for new features or changes that need a config edit or change results, and the last number for fixes that don't.

Each entry says what a user sees change, which config keys or output files are new, renamed or removed, and whether old results stay comparable (a change to a metric or generator needs a new experiment id and a re-recorded golden run).

## 0.11.0 (2026-10-20)

Existing metrics and ranks are unchanged; two new reported (unranked) columns appear, so the golden runs gain those columns (Linux CPU golden updated; the Linux CUDA golden needs a re-record). No new experiment id is needed; old evaluations simply lack the new columns.

### Added

- Subgroup privacy gap (fairness of privacy): the holdout distance membership attack (`distance_mia_auc`) is now also run inside each protected group, using the same groups as SynthEval's fairness metrics (each category or `data.protected_bins` bin with at least 5% of the train rows). `combined_evaluation.csv` gains `subgroup_distance_mia_auc_gap` (largest AUC difference between groups of one protected column; 0 = equally exposed) and `subgroup_distance_mia_auc_max` (the most exposed group's AUC) under custom fairness. Both are reported, not ranked: a copy of train leaks equally in every group, so the gap cannot tell it from a fresh draw. It runs whenever `holdout_distance` is selected and protected columns are declared; there is no new config key.
- `report.md`: a new **Privacy by protected group** subsection under Fairness explains the two columns and shows each group's AUC per model and its patient counts; the privacy section links to it.
- `privacy_subgroups.csv` in the evaluation folder: each model x protected group with its train and test patient counts, AUC and standard error (Hanley & McNeil). Groups with fewer than 10 patients on either side get no AUC.

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
