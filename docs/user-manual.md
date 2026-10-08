# SynthData user manual

This guide walks you through the SynthData pipeline from installation to reading the final report. It is written for someone who knows Python and basic machine learning but has not used this repository before. Read it once from top to bottom; afterwards, the section headings work as a reference.

Two other documents complete this one:

- [`synthdata/config.py`](../synthdata/config.py) documents every configuration option with its default. This manual covers the options you will actually change.
- [`docs/verification.md`](verification.md) records how each generator and metric was checked against its paper or reference implementation, and every place where we deliberately deviate from it.

## Contents

1. [What the pipeline does](#1-what-the-pipeline-does)
2. [Install](#2-install)
3. [Your first run (Hepatitis example)](#3-your-first-run-hepatitis-example)
4. [The config file](#4-the-config-file)
5. [Stage 1: load and split the data](#5-stage-1-load-and-split-the-data)
6. [Stage 2: imputation](#6-stage-2-imputation)
7. [Stage 3: generation and hyperparameter search](#7-stage-3-generation-and-hyperparameter-search)
8. [Stage 4: evaluation](#8-stage-4-evaluation)
9. [Stage 5: plots and the report](#9-stage-5-plots-and-the-report)
10. [Reading report.md](#10-reading-reportmd)
11. [Experiments, caching and reproducibility](#11-experiments-caching-and-reproducibility)
12. [Setting up your own dataset](#12-setting-up-your-own-dataset)
13. [Troubleshooting](#13-troubleshooting)
14. [Glossary](#14-glossary)

## 1. What the pipeline does

You give SynthData a real tabular dataset (for example, one row per hospital visit) with a target column you care about predicting. It produces synthetic datasets from several generative models, then scores each one on three questions:

- **Utility**: does the synthetic data look and behave like the real data? Can a model trained on it predict well on real data?
- **Privacy**: does the synthetic data leak information about the real people in the training data?
- **Fairness**: does the synthetic data distort outcomes for protected groups, such as sex or age bands?

It then ranks the models and writes one page, `report.md`, with the recommendation and all the evidence behind it.

The pipeline runs as four commands, in this order:

| Command | What it does | Main outputs |
| --- | --- | --- |
| `synthdata-impute` | Loads the data, splits it by patient into train, tuning and holdout, fills missing values | Split and imputed CSVs under `data_dir` |
| `synthdata-generate` | Tunes and trains each generator, writes synthetic datasets | One CSV per model under `generation.output_dir` |
| `synthdata-evaluate` | Scores every synthetic dataset on the holdout, ranks them | `combined_evaluation.csv`, `report.md` |
| `synthdata-plot` | Draws every figure and rewrites `report.md` with them embedded | PNG/HTML figures under `plots.output_dir` |

Every command takes the same `--config path/to/config.yaml`. That single YAML file is the source of truth for a run.

## 2. Install

You need [uv](https://docs.astral.sh/uv/) (the Python package manager this project uses) and git.

```bash
git clone https://github.com/childmindresearch/synthdata.git
cd synthdata
git submodule update --init --recursive   # our pinned forks of synthcity and SynthEval
uv sync --extra tabpfn
```

- `--extra tabpfn` installs the TabPFN generators. They run locally, but the model weights download only after your Prior Labs account has accepted the weights' license, so create a file named `.env` in the repository root containing `TABPFN_TOKEN=<your API key>` (see [TabPFN model license](#tabpfn-model-license)). If you do not have a key, set `generation.tabpfn.enabled: false` and skip the extra.
- On Linux x86-64, PyTorch is installed with CUDA 12.8 support, so an NVIDIA GPU is used automatically. `device: auto` picks the GPU when there is one.
- Run every command from the repository root. Relative paths in the config are resolved from the folder you run the command in.

### TabPFN model license

The `tabpfn` Python package and the TabPFN model weights are licensed separately. The weights come under a Prior Labs non-commercial license per model version, which you accept once per account on the Licenses tab at [ux.priorlabs.ai](https://ux.priorlabs.ai); your API key from the same site goes in `TABPFN_TOKEN`. The pipeline loads the version set in `generation.tabpfn.model_version` (default `v3`). TabPFN-3 License v1.0 (24 March 2026) and TabPFN-3.5 License v1.0 (9 September 2026) have the same terms. This summary is not legal advice; the license text governs.

Who it affects: anyone who runs `tabpfn_*` or `tabpfgen_*` generators, or uses their outputs. The terms cover outputs, so they apply to every synthetic table those generators write, including the labels TabPFN assigns.

| Term | What it means here |
| --- | --- |
| Non-commercial use only (sections 1c, 2a, 2b) | Testing, evaluation and research not tied to commercial gain, on public or private data. No production systems, revenue-generating work, client deliverables or commercial decision-making. |
| Outputs (2d) | Synthetic data from TabPFN may be used only for non-commercial purposes, and not to train a model that rivals TabPFN. Prior Labs claims no ownership of outputs. |
| Releasing synthetic data | Before sharing TabPFN-generated data outside your team, for example as a non-commercial research release, make sure the release terms keep recipients to non-commercial use. A release under terms that allow commercial reuse (such as CC BY or CC0) conflicts with 2d; release synthetic data from the other generators instead, or get a commercial license. |
| Who accepts (preamble) | You accept as a professional. Accept on behalf of your employer only if you are authorized to bind it; otherwise ask whoever handles licensing there. |
| Data protection (2e, 4a) | Inputs and outputs must comply with GDPR, the EU AI Act and similar rules, with safeguards suited to the data. Clinical data needs the usual approvals regardless of the model. |
| Weights (3, 4) | This repository does not ship weights. Copying them to others requires the license text and Prior Labs' attribution notice; hosting the model as a service needs a commercial license. |
| Revocation (7b, 7d) | Prior Labs can end the license by notice, after which the model and its copies must be deleted. The generation log records which weights each run used (`[tabpfn] using model weights ...`). |

Commercial use needs a separate license from Prior Labs (sales@priorlabs.ai).

Check the install with the fast test suite:

```bash
uv run pytest tests/unit
```

## 3. Your first run (Hepatitis example)

[`configs/config_hepatitis.yaml`](../configs/config_hepatitis.yaml) uses the public UCI Hepatitis dataset (155 patients). It downloads automatically and runs on a laptop CPU. Run the four stages:

```bash
uv run synthdata-impute   --config configs/config_hepatitis.yaml --plot
uv run synthdata-generate --config configs/config_hepatitis.yaml --plot
uv run synthdata-evaluate --config configs/config_hepatitis.yaml --plot
uv run synthdata-plot     --config configs/config_hepatitis.yaml
```

`--plot` draws that stage's figures as it goes; `synthdata-plot` at the end redraws everything and rewrites the report so every figure is in it. When the last command finishes, open:

```
output/hepatitis/evaluation/data_v_1.0/exp_v_0.1/report.md
```

`data_v_1.0` comes from `data.version` and `exp_v_0.1` from `experiment.id`. Read [section 10](#10-reading-reportmd) to understand what you see. To make the first run faster, set `generation.hpo.enabled: false` (models train with library defaults) or shorten `generation.synthcity.names` to two or three models.

## 4. The config file

A config has a few top-level keys and one section per stage. The example configs list only the settings worth reviewing; anything you leave out takes the default from [`synthdata/config.py`](../synthdata/config.py). A misspelled or unknown key stops the run and prints the list of valid keys, so a typo never fails silently.

```yaml
name: hepatitis       # short name used in default paths
seed: 42              # every random step derives from this
device: auto          # auto | cpu | cuda | mps

experiment:           # which experiment folder results go to (section 11)
data:                 # stage 1: source, schema, column roles, split
imputation:           # stage 2
generation:           # stage 3: models, HPO, number of rows, replicates
evaluation:           # stage 4: metrics, baselines, privacy attacks, ranking weights
plots:                # stage 5
```

The settings you will most often change, by section:

| Setting | What it controls | Example values |
| --- | --- | --- |
| `seed` | Split, imputation, model training and metrics | `42` |
| `experiment.id` | Folder name of this run; reusing it resumes the run | `"1.2"`, `null` (new id each run) |
| `data.source`, `data.path` | Where the data comes from | `csv`, `data/my.csv` |
| `data.version` | Label for this version of the data; change it when the data changes | `"1.0"` |
| `data.variable_schema_path` | CSV that says which columns are categorical or continuous | see [section 12](#12-setting-up-your-own-dataset) |
| `data.target_column` | The outcome column | `readmission_group` |
| `data.patient_id_column` | Groups rows by patient so no patient is in two splits | `research_subject_id` |
| `data.quasi_identifier_columns`, `sensitive_columns`, `protected_columns` | Column roles for privacy and fairness | see [section 5](#5-stage-1-load-and-split-the-data) |
| `data.stratify_columns`, `stratify_bins` | What the split keeps balanced | `[target, gender, age]` |
| `imputation.method` | How missing values are filled | `missforest`, `simple` |
| `generation.n_samples` | Synthetic rows per model | about the real train size |
| `generation.n_replicates` | Times each model is retrained with a new seed | `1`, or `3` for confidence intervals |
| `generation.synthcity.names` | Which synthcity generators run | `[ctgan, tvae, arf]` |
| `generation.hpo.*` | Hyperparameter search budgets | see [section 7](#7-stage-3-generation-and-hyperparameter-search) |
| `evaluation.baselines` | Reference rows scored next to the models | `[train_copy, marginals]` |
| `evaluation.rank_weights` | How much utility, privacy and fairness count in the overall rank | `{utility: 1, privacy: 2, fairness: 1}` |

The three example configs are templates, not defaults for your data:

- [`config_hepatitis.yaml`](../configs/config_hepatitis.yaml): small, one row per patient, CPU-friendly.
- [`config_sim.yaml`](../configs/config_sim.yaml): a wide (680-column), multi-visit, three-class dataset with profiled GPU budgets. Start here for real clinical data.
- [`config_loris.yaml`](../configs/config_loris.yaml): the LORIS dataset, which shows the `binary_target` option for a multi-class target.

## 5. Stage 1: load and split the data

Stage 1 runs inside `synthdata-impute`. It loads the data, cleans it, assigns column roles and splits it.

### Loading and cleaning

- `data.source` is `uci` (download by `uci_id`), `csv` or `parquet` (read `data.path`).
- `data.drop_columns` removes columns no model should see, such as free text or encounter numbers.
- `data.drop_rows_missing_target: true` drops rows with no target value. Every later stage needs a known target.
- `data.outlier_zscore_threshold` with `data.outlier_columns` turns sentinel codes (such as a lone 999 in a 0 to 30 scale) into missing values. List only columns you know contain such codes.
- The variable schema (`data.variable_schema_path`) must list every remaining feature and the target exactly once. The run stops if a column is missing, duplicated or no longer in the data.

### Column roles

Privacy and fairness metrics need to know what each column means. Three lists in `data` hold this:

| Role | Meaning | Used by | Example |
| --- | --- | --- | --- |
| `quasi_identifier_columns` | Public attributes an attacker could already know and use to link a record to a person | Anonymeter linkability and inference attacks (as attacker knowledge); both need at least one quasi-identifier and one sensitive column | age, sex, region |
| `sensitive_columns` | Secrets an attacker would try to learn | Anonymeter inference attacks (as targets); synthcity data_leakage and distinct l-diversity; SynthEval att_discl | a diagnosis, income |
| `protected_columns` | Groups whose fair treatment you want to check | Fairness metrics | sex, ethnicity, age band |

A column can be both a quasi-identifier and protected, or sensitive and protected, but never both a quasi-identifier and sensitive. None of the lists may contain the target or the patient ID. If you set `sensitive_columns`, you must also set `protected_columns` (even to `[]`).

Only Anonymeter reads the quasi-identifiers. synthcity and SynthEval have no quasi-identifier setting: their attribute attacks assume the attacker knows every other column and have no holdout control, so treat them as a worst-case screen and Anonymeter inference as the evidence for your declared threat model. The full table is in [verification.md, Column roles](verification.md#column-roles).

### The patient-level split

The data is split into three parts. The default shares are 60/20/20:

| Split | Share | Used for |
| --- | --- | --- |
| **search-train** | 60% | Fits the imputer and every HPO candidate model |
| **tuning** | 20% | Scores HPO candidates, so the search never sees the holdout |
| **holdout** (test) | 20% | Touched only by evaluation |

After the search, the final models are refitted on search-train plus tuning (80%), which the code and output files call **train**. So `train.csv` contains the tuning rows, and `test.csv` is the holdout.

Two rules keep the evaluation honest:

1. **Whole patients go to one split.** If `data.patient_id_column` is set, all visits of a patient land in the same split, so a model is never tested on a patient it was trained on. The ID column is then dropped before any model sees it. Leave it unset only when every row is a different person.
2. **The split is stratified.** `data.stratify_columns` lists columns whose joint distribution each split should match, most important first (put the target first). The default is the target alone. `data.stratify_bins` bins continuous columns, for example `[null, null, [30, 60]]` turns age into under 30, 30 to 60 and 60 or over. Groups too small to spread over the folds fall back to the first column alone.

The shares must sum to 1 and each must be a multiple of 1/k for some k up to 20, because the split takes whole folds of scikit-learn's `StratifiedGroupKFold` (0.6/0.2/0.2 uses 5 folds). `tuning_fraction` may be 0 only when HPO is off.

Output: `split_report.csv` in the data folder lists the rows, patients and class balance of each split. Check that the class shares are similar across splits.

## 6. Stage 2: imputation

Most generators cannot handle missing values, so `synthdata-impute` fills them. The target is never imputed.

### Methods

| `imputation.method` | What it is | When to use |
| --- | --- | --- |
| `missforest` (default) | Iterative random-forest imputation (scikit-learn `IterativeImputer`), as in Stekhoven and Bühlmann | Most datasets |
| `simple` | Median for continuous columns, most frequent value for categorical ones | Quick tests, CI |
| `tabimpute` | TabPFN-based imputation | Experimental |
| `refidiff` | Predictive plus diffusion hybrid; needs `uv sync --extra refidiff` | Experimental |

`imputation.missforest` sets the number of trees and rounds; the defaults are smaller than the paper's so a 20,000 by 400 table imputes in minutes.

### No leakage from held-out rows

The imputer is fitted twice:

1. On search-train only, then used to fill search-train and tuning. HPO uses these.
2. On all of train, then used to fill train and the holdout. The final models and evaluation use these.

In both cases the rows being scored never influence how values are filled. `imputation_drift.csv` compares the two fills of the cells both phases imputed; a column above `drift_warn_threshold` gets a warning, which usually means the column has very few observed values.

### Rounding

`round_to_int_default: true` rounds every imputed feature to a whole number unless `round_rules` gives it a number of decimals (for example `{BILIRUBIN: 2}`). Set it to `false` when your features are genuinely continuous.

### Missingness indicators

Whether a value is missing often carries information (a test that was not ordered). `imputation.missing_indicators` handles this:

- Columns missing in at least `min_missing_fraction` (default 5%) of search-train rows get an extra binary column `<column>__missing`. Generators learn these like any other column.
- After generation, synthetic values whose indicator is 1 are blanked again. The result is written to `released/<model>.csv`, the copy you would share. The CSV next to it, without `released/`, is the imputed version that evaluation scores.
- Columns missing in at least `indicator_only_fraction` (default 80%) of rows keep only their indicator. Their values are neither imputed nor generated; in the released copy, synthetic rows marked as recorded get values sampled from a CART model fitted on the observed train values (the synthpop method). Set it to `null` to impute and generate every column. List columns you always want generated in `keep_values_columns`.

### Caching

Imputed CSVs are cached in `data_dir/data_v_<version>/`. They are reused only if the data, split and imputation settings are unchanged; changing any of them refits the imputer. Set `imputation.cache: false` to always refit.

## 7. Stage 3: generation and hyperparameter search

`synthdata-generate` needs the imputed data from stage 2. It runs a hyperparameter search per model (if enabled), refits each model with its best settings, and writes synthetic data.

### Generators

| Name in config | Family | Notes |
| --- | --- | --- |
| `ctgan` | GAN | Conditional tabular GAN |
| `tvae` | Variational autoencoder | Often strong and fast |
| `rtvae` | Variational autoencoder | Robust TVAE variant |
| `adsgan` | GAN | Adds an identifiability penalty for privacy |
| `pategan` | GAN with differential privacy | Trains under a formal privacy budget; usually lower utility |
| `bayesian_network` | Probabilistic graphical model | Does not scale to hundreds of columns |
| `ddpm` | Diffusion model (TabDDPM) | Slow on wide data |
| `arf` | Adversarial random forests | CPU only |
| TabPFN `standard` | Foundation model | Generates features, then assigns labels from TabPFN's predicted class probabilities |
| TabPFN `custom` | Foundation model | Generates features and target jointly |
| TabPFGen | Energy-based sampling | Off by default: its output is near-copies of training rows (see [verification](verification.md)) |

Synthcity models are chosen with `generation.synthcity.names`. TabPFN runs when `generation.tabpfn.enabled` is true; `variants` picks `standard` and/or `custom`, and `data_variants` picks whether it learns from the `raw` data (TabPFN handles missing values itself) or the `imputed` data. Imputed-data outputs get an `_imputed` suffix, for example `tabpfn_custom_imputed`. `model_version` picks the TabPFN weights (default `v3`); `v3.5` works only after you accept its non-commercial license on ux.priorlabs.ai with the account behind your `TABPFN_TOKEN`.

### Rows, class balance and replicates

- `generation.n_samples` is the number of synthetic rows per model. Setting it close to the real train size makes utility and privacy scores easier to compare with the real data.
- `generation.match_class_prior: true` resamples every synthetic dataset to the class shares of the real train data. Without it, a generator that produces too many minority-class rows can look better on macro-F1 without being more faithful.
- `generation.n_replicates` retrains every model that many times with seeds `seed`, `seed+1`, and so on. Replicate r > 0 is saved as `<model>__rep<r>`. With 2 or more, the report gives every score a 95% confidence interval and says which models cannot be told apart from the best. HPO runs only once per model.

### How the hyperparameter search works

For each model, [Optuna](https://optuna.org/) tries a number of settings ("trials"):

1. Fit the model on search-train with the trial's settings.
2. Generate synthetic rows and resample them to the train class shares.
3. Train a fixed XGBoost classifier on the synthetic rows and test it on the real tuning rows (**TSTR**, train on synthetic, test on real). The score is macro-F1, averaged over `tstr_seeds` (3) seeds. Macro-F1 weighs every class equally, so a rare class counts as much as a common one. Set `objective: tstr_macro_auprc` for a threshold-free alternative.
4. Check three lenient screens in `hpo.constraints`. A trial that copies training rows (`copy_margin`), drops categories or classes (`min_category_coverage`), or produces out-of-range values (`max_out_of_range`) is marked infeasible and cannot be chosen as best.

The trial with the best feasible score wins. The final model is refitted with those settings on all of train. The best settings are saved in `hpo_best_params.json`, and every trial is stored in `optuna_studies.db`, so an interrupted search resumes where it stopped when you rerun with the same experiment id.

### Setting HPO budgets

Each model's search stops at its trial count or its time limit, whichever comes first:

| Setting | Meaning |
| --- | --- |
| `hpo.n_trials`, `hpo.timeout_seconds` | Defaults for models not listed below |
| `hpo.n_trials_per_model` | Trials per model, e.g. `{ctgan: 40, arf: 25}`; `0` skips the search for that model |
| `hpo.timeout_seconds_per_model` | Seconds per model |
| `hpo.epoch_ranges` | Training length searched, as `[low, high, step]`, e.g. `{ctgan: [25, 150, 25]}` |
| `hpo.pruner: median` | Stops a CTGAN or ADS-GAN trial early when its intermediate score is below the median of earlier trials |

A good way to set budgets for a new dataset:

1. Time one trial per model at your data's size (one fit on search-train, then generation and scoring).
2. Choose trials: about 10 plus 10 per hyperparameter the model tunes, capped by how long you can wait.
3. Set the timeout to trials × time per trial × 1.5, so one slow model cannot eat into another's budget.

[`config_sim.yaml`](../configs/config_sim.yaml) shows the result for a 12,600 by 682 search-train table on one GPU: about 30 hours per GAN or VAE and roughly 3.5 to 5 days in total. The timing table is in [verification.md, HPO budgets](verification.md#hpo-budgets). For a quick look at a new dataset, turn HPO off or use a handful of trials.

### Outputs

Under `generation.output_dir/data_v_<version>/exp_v_<id>/`:

- `<model>.csv`: synthetic data, imputed (what evaluation scores).
- `released/<model>.csv`: synthetic data with missing values put back (what you would share).
- `hpo_best_params.json`, `optuna_studies.db`: search results.

A rerun of the same experiment skips models whose CSV already exists. Set `generation.force_retrain: true` to retrain them.

## 8. Stage 4: evaluation

`synthdata-evaluate` scores every synthetic dataset of the latest experiment (or `--experiment-id`) against the **holdout**, which no generator, imputer or HPO trial has seen.

### Baselines

Two fixed reference datasets are scored next to the generators, so every run has the same anchors. They are never recommended.

| Baseline | What it is | What it tells you |
| --- | --- | --- |
| `baseline_train_copy` | The real training rows, released as is | The best utility possible and the worst privacy possible |
| `baseline_marginals` | Each column sampled independently from its real distribution | The utility of data with no relationships between columns |

A useful generator beats `baseline_marginals` on utility and stays well above `baseline_train_copy` on privacy.

### Metric families

Metrics come from three sources, each switched on in `evaluation.synthcity`, `evaluation.syntheval` and `evaluation.custom`. Each takes `categories` (any of `utility`, `privacy`, `fairness`) or an exact list of `metrics`; `null` runs everything. Metric names are listed in [`synthdata/evaluation/catalog.py`](../synthdata/evaluation/catalog.py).

| Source | Examples |
| --- | --- |
| [synthcity](https://github.com/vanderschaarlab/synthcity) | Distribution distances (Wasserstein, KL, PRDC), detection AUC (can a classifier tell real from synthetic?), identifiability, DOMIAS |
| [SynthEval](https://github.com/schneiderkamplab/syntheval) | Correlation and distribution differences, classifier accuracy and AUROC difference, nearest-neighbour adversarial accuracy, epsilon risk, statistical parity and other subgroup gaps |
| Custom (this project) | Holdout TSTR, Anonymeter attacks, holdout distance checks, log disparity, equalized odds and equal opportunity |

Both libraries run from our pinned forks, which fix the bugs listed in [verification.md](verification.md).

### Utility: holdout TSTR

The same XGBoost classifier used in HPO is trained on each synthetic dataset and tested on the holdout. `tstr_holdout.csv` reports macro-F1, balanced accuracy, macro AUPRC and per-class F1, next to the score of the same classifier trained on the real train data (the ceiling).

For a target with 3 or more classes, metrics that need two classes (AUROC difference, subgroup gaps) run once per class against the rest and are averaged with equal weight (`evaluation.class_averaging: ovr_macro`, per-class values in `ovr_per_class.csv`). Alternatively, set `class_averaging: binary` and define which classes count as positive in `evaluation.binary_target` (see `config_loris.yaml`).

### Privacy: evidence, not a verdict

Privacy is reported as evidence next to the baselines. There is no pass/fail gate, because no fixed threshold can certify that a dataset is private. The checks are:

- **Anonymeter** (Giomi et al., 2023): three attacks, each scored as success above a baseline attack, with the holdout as the control set.
  - *Singling out*: can the attacker find a rule that matches exactly one real person?
  - *Linkability*: can the attacker use the synthetic data to link a person's quasi-identifiers to their sensitive columns, when the two are held in separate records?
  - *Inference*: knowing a person's quasi-identifiers, can the attacker guess their sensitive columns?
- **Holdout distance checks**: are synthetic rows closer to training rows than to unseen real rows? (`dcr_*`, `nndr_*`, `distance_mia_auc`).

With `evaluation.privacy_attacks.unit: patient` (the default), each patient counts once, even if they had many visits. Settings for the attacks are under `evaluation.privacy_attacks.anonymeter`.

### Fairness

- **Subgroup gaps** (SynthEval, custom): how differently a classifier trained on the synthetic data treats protected groups.
- **Log disparity** (Bhanot et al., 2021): how far each protected subgroup's outcome share drifts from the real data. Continuous protected columns need bins, set in `evaluation.log_disparity.protected_bins`; `target_map` and `protected_map` give readable labels.

### Ranking

1. Every metric is oriented so higher is better, then min-max scaled across the models in this run. Scores therefore compare models within the run; they are not absolute quality measures.
2. Metrics are averaged within each source, then across sources, into a utility, a privacy and a fairness score between 0 and 1.
3. The overall score is the weighted geometric mean of the three, with weights from `evaluation.rank_weights`. A near-zero score on one dimension cannot be made up by the others.

With replicates, each score is the mean over seeds with a 95% confidence interval, and a model is "tied with best" when a one-sided Welch t-test cannot place its overall score below the best model's.

### Running on large data

Evaluation can process several models in parallel. `evaluation.syntheval_execution` controls how many (`model_workers`, `"auto"` by default), how many CPU cores each uses, and how much memory to keep free. Lower `model_workers` or raise `memory_per_model_gib` if the machine runs out of memory.

## 9. Stage 5: plots and the report

`synthdata-plot` redraws all figures for the sections in `plots.sections` (`data`, `imputation`, `generation`, `hpo`, `evaluation`) and then rewrites `report.md` with every figure embedded. It never reruns earlier stages, so it is safe to run any time. Formats and resolution are set with `plots.formats` and `plots.dpi`.

## 10. Reading report.md

`report.md` sits in the evaluation folder and is the one page to read after a run. Every table and figure it mentions is linked. Its seven sections:

1. **At a glance.** Dataset, experiment, seed, the recommended model and its overall score, how it compares with the two baselines, and a "Read with care" list of warnings (for example, a single seed, or an Anonymeter attack whose 95% interval is above 0).
2. **Ranking.** One row per model with overall, utility, privacy and fairness scores (with intervals and "tied with best" when there are replicates), followed by how the scores were built and the trade-off plots.
3. **Utility.** Real versus synthetic distribution plots per model, and the holdout TSTR table against the real-data ceiling.
4. **Privacy evidence.** The attack and distance results next to the baselines, with how to read each column.
5. **Fairness.** Subgroup gap metrics and log disparity, with links to sunburst pages that show which subgroups are over- or under-represented.
6. **Data and preparation.** The split table, missingness, imputation checks and the HPO results.
7. **File index.** Every output file with a one-line description.

How to interpret it:

- **Start with the baselines.** A generator with utility close to `baseline_marginals` has not learned the relationships between columns. A generator with privacy close to `baseline_train_copy` is close to releasing the real rows.
- **Look at real numbers, not only ranks.** Ranked scores are relative to this run. Check the TSTR table: a generator's macro-F1 near the real-data ceiling means a model trained on it predicts real outcomes nearly as well as one trained on real data.
- **Treat small gaps with suspicion.** With one seed there are no intervals. Use `n_replicates: 3` or more before choosing between close models.
- **Privacy results are not a guarantee.** A low risk means these attacks, with these settings, did not find a leak. Any flagged attack deserves a closer look, and any real release needs a privacy review.
- **Check the HPO plots.** If the best trial is at the edge of an epoch range, or the score is still rising at the last trial, the budget may be too small.

## 11. Experiments, caching and reproducibility

Outputs are organized by dataset version and experiment:

```
data/<name>/data_v_<version>/                               splits, imputed data, split_report.csv
output/<name>/synthetic_data/data_v_<version>/exp_v_<id>/   synthetic data, HPO
output/<name>/evaluation/data_v_<version>/exp_v_<id>/       metrics, report.md
output/<name>/plots/data_v_<version>/exp_v_<id>/            figures
output/<name>/experiments/data_v_<version>/exp_v_<id>/      manifest.json, config_snapshot.json
```

- **`data.version`**: change it whenever the source data changes, so old and new results never mix.
- **`experiment.id`**: with a fixed id, rerunning resumes and extends that experiment (finished models and HPO trials are reused). With `id: null`, `synthdata-generate` makes a new id from the time and `experiment.tag`. Evaluation and plotting use the latest experiment unless you pass `--experiment-id`.
- Every command accepts `--dataset-version` to override `data.version`; `synthdata-generate` also accepts `--tag` and `--experiment-id`.
- `manifest.json` records what each stage ran and produced, and `config_snapshot.json` holds the full configuration. Keep both with any result you report.
- A run with the same config, seed and machine reproduces the same data and metrics; the integration tests check this. GPU training can still differ slightly between machines or library versions.

Use a fresh experiment id for new work, and do not edit a config between stages of the same experiment.

## 12. Setting up your own dataset

1. **Copy a config.** Start from `configs/config_sim.yaml` for multi-visit clinical data, or `config_hepatitis.yaml` for small one-row-per-person data. Change `name`, all `output_dir` and `data_dir` paths, `data.source`, `data.path` and `data.version`.
2. **Write the variable schema.** One row per feature and the target:

   ```csv
   column,kind,ordinal_order
   Age,continuous,
   Sex,categorical,
   Severity,categorical,"[0, 1, 2, 3]"
   target,categorical,
   ```

   `kind` is `categorical` or `continuous`. For an ordered categorical column, `ordinal_order` lists its values from lowest to highest; leave it blank for unordered categories. Columns in `drop_columns` and the patient ID are left out.
3. **Set the target and patient ID.** `data.target_column` and, if a person can have several rows, `data.patient_id_column`. Add `drop_rows_missing_target: true` if some targets are missing.
4. **Assign column roles** ([section 5](#column-roles)). Decide with someone who knows the data what an attacker could plausibly know (quasi-identifiers) and what must stay secret (sensitive).
5. **Choose stratification.** Target first, then the most important protected columns, with bins for continuous ones.
6. **Run `synthdata-impute --plot`** and check `split_report.csv`, the missingness plot and `imputation_drift.csv`.
7. **Do a quick generation run** with HPO off and two or three fast models (`tvae`, `arf`) to check everything works end to end.
8. **Profile and set HPO budgets** ([section 7](#setting-hpo-budgets)), then run the full search under a new experiment id.
9. **Set `n_replicates` to 3 or more** for the run whose results you will report.

## 13. Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `Unknown config key 'x' for ...` | Typo or a key in the wrong section. The message lists the valid keys; check `synthdata/config.py`. |
| `... was replaced by ...` or `evaluation.privacy_gate was removed` | The config uses an option from an older version. Follow the message; delete `privacy_gate` entirely. |
| Schema errors (missing, duplicate or stale columns) | The variable schema does not match the data after `drop_columns` and the patient ID are removed. Add or remove rows in the schema. |
| `data.sensitive_columns now lists the secret columns ...` | Set `data.protected_columns` (even to `[]`) to confirm the column-role meaning. |
| `... cannot be both a quasi-identifier ... and sensitive` | Put the column in only one of those lists. |
| `data split fractions ... must each be a multiple of 1/k` | Use shares like 0.6/0.2/0.2 or 0.5/0.2/0.3. |
| `generation.hpo.enabled needs a tuning split` | Set `data.tuning_fraction` above 0, or turn HPO off. |
| Stratified split error about NaN in the target | Set `data.drop_rows_missing_target: true`. |
| `No imputed data found. Run synthdata-impute ... first.` | Run stage 2 first, with the same config and `data.version`. |
| TabPFN authentication errors | Add `TABPFN_TOKEN` to `.env` in the repository root, or set `generation.tabpfn.enabled: false`. |
| `TabPFNLicenseError: ... one-time license acceptance` | Your key is fine but your account has not accepted the license for the weights in `generation.tabpfn.model_version`. Accept it on the Licenses tab at ux.priorlabs.ai ([TabPFN model license](#tabpfn-model-license)), or switch to a version you have accepted. |
| Float overflow or NaN errors in TabPFN or imputation | A column has sentinel codes (999) or corrupt extremes. List it in `data.outlier_columns` with `outlier_zscore_threshold`. |
| A model is missing from the evaluation | Its generation failed; check the `synthdata-generate` log. Rerun with the same experiment id to retry only the missing models. |
| HPO finishes far below `n_trials` | The timeout was reached first. Raise `timeout_seconds_per_model` or shorten `epoch_ranges`. |
| Every HPO trial is infeasible | The generator copies rows, drops categories or produces out-of-range values. Look at the trial table in `optuna_studies.db`; a longer training range often helps. |
| Bayesian network or DDPM is extremely slow | Expected on wide data; remove them from `synthcity.names` for hundreds of columns. |
| Evaluation is killed or the machine runs out of memory | Lower `evaluation.syntheval_execution.model_workers` or raise `memory_per_model_gib`. |
| Warning that singling out stopped short | Anonymeter could not find enough unique predicates within `singling_out_max_attempts`; the risk may be underestimated. Raise the limit if time allows. |
| Report says a figure is "not rendered yet" | Run `synthdata-plot` with the same config (and `--experiment-id` if needed). |
| Results changed unexpectedly between runs | The data, `data.version` or config changed and the cache was refreshed, or a different experiment id was picked up. Compare `config_snapshot.json` and `manifest.json`. |

To run the tests after changing code:

```bash
uv run pytest tests/unit                                            # fast
uv run --with catboost pytest tests/integration -m "integration and not slow"   # full pipeline on a small fixture, a few minutes
```

## 14. Glossary

- **Baseline**: a fixed reference dataset (`train_copy`, `marginals`) scored next to the generators.
- **Holdout (test)**: the 20% of patients used only by evaluation.
- **HPO**: hyperparameter optimization, the search for each model's best settings.
- **Macro-F1**: the F1 score computed per class and averaged with equal weight, so rare classes count fully.
- **Quasi-identifier**: a public attribute that could help link a record to a person.
- **Replicate**: a retraining of the same model with a different seed, used to measure run-to-run variation.
- **Search-train**: the 60% of patients that HPO candidates and the first imputer are fitted on.
- **Sensitive column**: an attribute an attacker should not be able to learn.
- **Train**: search-train plus tuning (80%), used to fit the final models.
- **TSTR**: train on synthetic, test on real. A classifier is trained on synthetic data and scored on real data.
- **Tuning**: the 20% of patients that score HPO candidates.
