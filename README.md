# SynthData

SynthData is a config-driven pipeline for tabular-data imputation, synthetic data generation, evaluation, and plots. It is designed to run on **your own local CSV or Parquet data**.

## Quick start

Clone the repository, download its linked libraries (submodules), and install the packages needed to run the pipeline:

```bash
git clone https://github.com/childmindresearch/synthdata.git
cd synthdata
git submodule update --init --recursive
uv sync --extra tabpfn
```

Start with [`configs/config_hepatitis.yaml`](configs/config_hepatitis.yaml) for a small dataset example or [`configs/config_loris.yaml`](configs/config_loris.yaml) for a wider dataset example. Update its `data:` section and variable-schema path to point to your local data. A variable schema is a file that describes each column and its type. See [`synthdata/config.py`](synthdata/config.py) for every configuration setting and its default value.

```bash
uv run synthdata-impute   --config path/to/your-config.yaml --plot
uv run synthdata-generate --config path/to/your-config.yaml --plot
uv run synthdata-evaluate --config path/to/your-config.yaml --plot
uv run synthdata-plot     --config path/to/your-config.yaml
```

> [!NOTE]
> This repository includes config files for UCI ML Repo's Hepatitis and HBN's LORIS data only as local development test data and examples. Treat these configs as templates, not ready-to-use settings for your data. Review each project's dataset paths, column roles, target, version, generation methods, and evaluation settings.

### Audit a configuration

Run `synthdata-test` to check a configuration before running any pipeline steps:

```bash
uv run synthdata-test --config path/to/your-config.yaml
```

The audit checks that the configuration and its rules make sense, then loads the configured dataset and variable schema by default. It checks the data, schema, declared column roles, and data split using the same loader as the pipeline. The check may write processed data, split files, and related metadata under the configured data directory. If the config points to a remote source such as UCI and the data is not already cached locally, the audit may download and cache it.

## Your dataset configuration

Provide a variable-schema CSV with one row for each feature and the target. For each column, mark whether it is `categorical` or `continuous`. For categorical values with a meaningful order, the schema also records that order.

```csv
column,kind,ordinal_order
Age,continuous,
Sex,categorical,
Severity,categorical,"[0, 1, 2, 3]"
target,categorical,
```

Dataset profiles must define the standard `data.split` roles `train`, `tuning`, and `final_holdout`. The roles are patient-disjoint: each patient can appear in only one role, even if they have multiple encounters. `train` is used to fit candidate models, `tuning` is used for hyperparameter optimization (HPO) and candidate selection, and `final_holdout` is kept separate until after selection to provide final evidence. Set `data.patient_id_column` to the patient ID column. The pipeline uses it to keep each patient's records in one role, then removes it from the data given to models.

For patient-group splitting, the pipeline creates and reuses a local secret key (`.patient_id_hmac_key`) in the data folder by default. To use an external key, set `SYNTHDATA_PATIENT_ID_HMAC_KEY`; the pipeline uses that key instead. The pipeline uses it to create consistent patient-ID tokens in split and assignment files, reducing raw-ID exposure if those files are shared.

### Role overlap and threat-model boundaries

Declare each role explicitly; the pipeline does not guess one role from another. Important dataset roles are:
- `patient ID` identifies the same person across encounters,
- `target` is the outcome used to evaluate utility and fairness,
- `quasi-identifiers (QIs)` are explicitly selected columns an attacker could use to link records to people,
- `sensitive attributes` are columns whose disclosure is measured by privacy attacks,
- `protected attributes` identify groups used to check fairness. Checks include representation , equal opportunity, equalized odds, and log-disparity evidence.

## Pipeline behavior

These four commands form the pipeline and run in this order:

1. **Impute** missing feature values using fixed HyperImpute plugins. The supported imputation path uses HyperImpute only: candidate models are fit using `train`, while the final model is fit using `train` and `tuning` together. The transforms do not fit on `tuning` or `final_holdout`. TabImpute and RefiDiff are blocked or deferred in the supported path because they cannot keep the roles isolated yet. Retained benchmark and older code are not supported paths for producing release results.
2. **Generate** candidate synthetic datasets using the configured SynthCity, TabPFN, and TabPFGen models. Optional Optuna hyperparameter searches are saved and can resume. Each HPO context is stored in a companion file (sidecar) versioned by a digest, a value calculated from the context, with a pointer to the latest one. Changing the objective creates a new study identity and preserves the earlier context evidence.
3. **Evaluate** candidates for utility, privacy, and fairness. Candidate rankings use only `train` and `tuning`. After choosing a candidate, the pipeline fits its generator again on those two roles and evaluates it once against `final_holdout`. Evaluation can process models in parallel, within configured resource limits. It writes a ranked table, report, and diagnostic information.
4. **Plot** saves data quality, generation, HPO, and evaluation results without running earlier steps again.

Files (artifacts) produced by the pipeline are organized by dataset name, dataset version, and experiment ID. Folder names show both versions, for example, `data_v_1.2/exp_v_0.4/`. Generation creates a new experiment by default. Evaluation and plotting use the latest experiment or accept `--experiment-id` to open an earlier run. This keeps cached inputs, model outputs, HPO state, metrics, and figures separate across dataset revisions.

### Imputation artifacts and logs

`synthdata-impute` writes its validation table (a summary used to check the imputed data) to:

```text
output/<config.name>/imputation/data_v_<version>/imputation_validation_report.csv
```

For a config without `data.version`, `<version>` is `unversioned`, so the
report is written under `output/<config.name>/imputation/data_v_unversioned/`.
The saved imputed-data cache is separate from this report: raw, imputed, and
split CSV files are stored under `<data.data_dir>/data_v_<version>/` (or
`<data.data_dir>/data_v_unversioned/`), alongside metadata that records how the
cached files were produced. The CLI logs both locations instead of printing
the validation table in the terminal.

The supported candidate validation report covers only the roles transformed
for candidate use: `train` and `tuning`. It does not describe imputation of the
final holdout. Each row includes `datatype` from the variable schema's `kind`,
the number of distinct observed and imputed values (`obs_cardinality` and
`imp_cardinality`), and summaries for that column type. For continuous
columns, it includes the observed and imputed mean and standard deviation
(`obs_mean`, `obs_std`, `imp_mean`, and `imp_std`). These statistics do not
apply to categorical values, so those cells are empty. For categorical
columns, it includes the most common observed and imputed values (`obs_mode`
and `imp_mode`).

Evaluation always keeps the original metric values for audit. Ranked columns
are calculated separately and include only complete metrics that are allowed
to inform a decision. Failed, diagnostic, calibration-only, and blocked
results remain visible in the raw table or validation sidecar files. Each
evaluation bundle also records
`evaluation_artifacts-v1/metric_contract_manifest.json`,
`evaluation_artifacts-v1/synthcity_metric_status.json`,
`evaluation_artifacts-v1/syntheval_metric_status.json`, and
`evaluation_artifacts-v1/syntheval_execution.json`,
`evaluation_artifacts-v1/custom_metric_status.json`, and
`evaluation_artifacts-v1/final_holdout_evidence.json` when results from those
frameworks are present. The final-holdout sidecar is written after candidate
selection and points to the selected model's refit file. It is never merged
into `combined_evaluation.csv`. Its refit metadata records `fit_roles` as
`["train", "tuning"]`; successful evidence checks the synthetic CSV and its
cache metadata using SHA-256, a method for checking that file contents have
not changed. The SynthEval execution sidecar keeps the final status for each
method and both older and versioned rows with standardized formats. These
files are listed and integrity-checked by
`evaluation_artifacts-v1/manifest.json`. The sidecars contain each model's
metric status, fingerprints of the data roles, source information, and a
fingerprint of the metric rules used for that run. The experiment manifest
also records a separate
`final_holdout_evidence` stage. Legacy two-role datasets write an explicit
blocked final-evidence record instead of making a release claim.

By default, evaluation treats each row as a separate observation:

```yaml
evaluation:
  group_mode: row
  group_column: null
```

Set `group_mode: patient_group` and provide a non-empty `group_column` when
multiple rows represent the same patient or entity. The identifier is needed
to assign roles and record where data came from, but is excluded from all model
inputs and release-metric calculations. Real `train`, `tuning`, and
`final_holdout` data contain different patients. This separation does not make
encounters from one patient independent: privacy metrics cannot claim to
prevent links or inferences across that patient's encounters beyond keeping
them in one role.

## Release populations, privacy, and fairness evidence

Candidate utility (how useful the synthetic data is) compares synthetic data
prepared in the form intended for release with `tuning`. The final audit
compares the selected synthetic data, prepared in that same release form, with
`final_holdout`; it calculates new scores instead of
reusing candidate scores. For membership-inference attacks (MIA), members are
from `train+tuning`, and non-members are from the patient-disjoint
`final_holdout`. Attribute-disclosure attackers are trained on synthetic QIs
and tested on `final_holdout`. Representation, EO (equal opportunity), and
worst log disparity use final evidence. If required evidence is invalid, the
result is indeterminate; it is not silently replaced or reweighted.

If a synthetic categorical value does not occur in the source data, it is not
silently changed to a valid value. The affected evaluation records failed or
indeterminate evidence, depending on whether execution failed or required
evidence is incomplete; it never records such evidence as succeeded. Evidence
states have specific meanings: `succeeded` means every required table is
present; `failed` means execution or evaluation produced an error;
`indeterminate` means required evidence is missing or incomplete. Only
`succeeded` evidence can be used in metric values, plots, or policy rankings.
Failed and indeterminate records remain available for audit.

Formal privacy metrics for data prepared for release use transformed QIs and
sensitive fields: `S_k` and `S_l` are formal k-anonymity and l-diversity scores. K-anonymity
checks that records sharing the declared QIs occur in groups of at least k;
l-diversity checks diversity of sensitive values within those groups. `S_DCR`
and `S_epsilon` are separate formal release components. SynthCity's
`k-anonymization`, `l-diversity`, `k-map`, and `delta-presence` are blocked
KMeans-based proxy metrics. They are not formal k/l evidence and do not
authorize a release.

## Fixed utility, normalization, and final audit score

The utility score used during HPO is fixed:

```text
U_tuning = (S_TSTR + S_MMD + S_JSD) / 3
```

The final audit uses these scores:

```text
U = (S_TSTR + S_MMD + S_JSD) / 3
I = min(S_k, S_l, S_DCR, S_epsilon)
P = (I * S_MIA * S_attribute)^(1/3)
F = 0.40*S_representation + 0.40*S_EO + 0.20*S_worst_log_disparity
R_final = 0.45*U + 0.30*P + 0.25*F
```

Components that already have a standard scale are not rescaled. Raw anchored
metrics use fixed anchor values, which determine how scores are scaled and
clipped. Invalid required evidence makes its dimension and
`R_final` indeterminate. `R_final` is for audit only: do not rerank models
after seeing holdout results or use this score to authorize a release.

The included privacy thresholds are for calibration only and are off by
default. To enable `evaluation.privacy_gate` (the privacy decision check), each
threshold must name an exact
operational `contract_id` (the identifier for the approved metric rules).
Calibration-only and audit-only metrics remain visible as evidence but cannot
support a release recommendation. Evaluation sidecars record group counts and
fingerprints scoped to each data role, not raw identifiers. Until a metric has
an explicit `group_safe` contract, patient-group evaluation keeps its result
for audit but marks it `group_unsafe` and excludes it from policy rankings.
This compatibility step does not provide group-aware aggregation.

The general HPO objective is limited to the approved utility metrics (scores)
in `HPOConfig.metric_config` (`wasserstein_dist` and `inv_kl_divergence`, which
compare data distributions; nearest synthetic-neighbor distance, based on how
close records are to their nearest synthetic neighbor; and XGB utility, an
XGBoost-based utility measure). Privacy, diagnostic, and
calibration-only metrics are rejected when HPO is set up; they do not become
tuning objectives by default.

Before those objective metrics are calculated, every HPO candidate must pass
the checks in `generation.hpo.stage_a`: data shape and schema, numeric ranges
and categorical values allowed by the source, coverage of target categories
and protected groups, rules for column relationships that must always hold,
exact reuse of source rows (no reuse allowed),
and subgroup collapse (when a group is no longer adequately represented). A
failed check is saved as a pruned trial (a trial stopped before scoring)
under the generation experiment; it cannot count as
partial objective evidence. Stage A does not run privacy or fairness attacks.

If every trial for one HPO variant fails these Stage A checks, only that
variant's output is skipped. The pipeline does not invent fallback settings or
synthetic data; it may still produce other requested outputs. The existing
experiment manifest records generation as `partial`, lists expected,
produced, and failed outputs, and links to the saved Stage A results. This
manifest is the source of truth for the run status. Evaluation accepts a
missing variant only if the manifest explains why it is missing. It evaluates
available requested outputs and records incomplete coverage rather than
inventing metrics for the missing output. Partial reports and ranking plots
are labeled as partial: rankings compare only evaluated models, not all
requested outputs.

> [!TIP]
> See [`synthdata/config.py`](synthdata/config.py) for settings for cached data, computing devices, models, HPO, parallel evaluation, saved files, experiments, metrics, rankings, privacy gates, and plots.

## Optional: RefiDiff masked-cell benchmark

The retained `synthdata-imputation-benchmark` command and RefiDiff material are
for exploration or older work only. RefiDiff and TabImpute cannot currently
run in the supported mode that keeps dataset roles separate, so they cannot
support release claims. Do not use benchmark results to choose the supported
production imputation method.

The benchmark does not overwrite your regular imputed datasets. It saves each
study without replacing earlier studies. The study includes the masks (which
values were hidden for testing), parameter settings, per-column metrics, and
summary results under the configured imputation output directory.

```bash
uv run synthdata-imputation-benchmark \
  --config path/to/your-benchmark-config.yaml \
  --study-id refidiff-baseline-001
```

For an example with many columns, see the included LORIS-based [reference profile](configs/config_loris_refidiff_reference.yaml) and [HPO screening profile](configs/config_loris_refidiff_hpo.yaml). Copy and adapt them for your data; do not use their data paths or column choices as-is.

### Recommended benchmark sequence

1. **Establish a baseline.** Run a benchmark with fixed settings on a small, representative set of columns. This checks whether RefiDiff can recover values in your data and gives you a result to compare with later candidates.
2. **Screen settings.** Run an HPO study with a limited search to explore a practical range of model width, training budget, and sampling settings. Review its `best_trial.json` and per-column metrics; do not choose a candidate based only on its single summary score.
3. **Confirm a candidate.** Copy the selected settings into a new benchmark with fixed settings and a new study ID. Include the columns and missing-data scenarios that matter for your analysis (for example, MCAR, MAR, and MNAR: data missing completely at random, depending on observed data, or depending on unobserved data) before using the settings for production imputation.

> [!IMPORTANT]
> Reuse a study ID only to continue the exact same study. If you change the dataset, schema, masking plan, or settings, use a new study ID so results can be compared and traced to their inputs. For implementation details and the exact file layout, see [`synthdata/imputation/benchmark.py`](synthdata/imputation/benchmark.py) and [`synthdata/config.py`](synthdata/config.py).

> [!CAUTION]
> Benchmark and generation results are scientific records. Use a new study or experiment identifier for new work. Keep the generated manifests and saved configuration copies needed to reproduce each result.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests are in [`tests/`](tests/). See `pyproject.toml` for test markers (labels used to select tests) and optional dependency groups.

## Legacy and exploratory code

The pipeline above is the supported way to run new work. The following older or exploratory parts remain for reference and specific experiments:

### Apps

- [`apps/presidio/presidio_streamlit.py`](apps/presidio/presidio_streamlit.py): a Presidio Streamlit app designed to work offline for processing PHI/PII (protected health information/personally identifiable information). Install its dependencies with `uv sync --extra presidio`; see the [Presidio app guide](apps/presidio/PRESIDIO_APP_GUIDE.md) for setup and security guidance.

### Notebooks

- [`notebooks/ydata-test.py`](notebooks/ydata-test.py): experiments with ydata-synthetic; requires `uv sync --extra ydata`.
- [`notebooks/ctgan_hpo_hepatitis.ipynb`](notebooks/ctgan_hpo_hepatitis.ipynb) and [`notebooks/test_hepatitis_data.ipynb`](notebooks/test_hepatitis_data.ipynb): earlier Hepatitis-focused synthesis, imputation, HPO, and evaluation work.
- [`notebooks/tabpfn_demo.ipynb`](notebooks/tabpfn_demo.ipynb): experiments with TabPFN classification and data synthesis. When the API is needed, set `TABPFN_TOKEN` (and, optionally, `HF_TOKEN`) in `.env`.

### Scripts

- [`scripts/document_pipeline/`](scripts/document_pipeline): early work on PII anonymization and reading Markdown files; unrelated to the synthetic-data pipeline.

## Repository layout

- [`synthdata/`](synthdata/): code that runs the pipeline.
- [`configs/`](configs/): example configuration files.
- [`scripts/`](scripts/): command-line tools installed with the project.
- [`apps/`](apps/) and [`notebooks/`](notebooks/): optional tools and experiments.
