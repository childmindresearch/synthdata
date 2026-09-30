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

## Dataset configuration

Provide a variable-schema CSV with one row for each feature and the target. For each column, mark whether it is `categorical` or `continuous`. For categorical values with a meaningful order, the schema also records that order.

```csv
column,kind,ordinal_order
Age,continuous,
Sex,categorical,
Severity,categorical,"[0, 1, 2, 3]"
target,categorical,
```

Dataset profiles must define the standard `data.split` roles `train`, `tuning`, and `final_holdout`. The roles are patient-disjoint: each patient can appear in only one role, even if they have multiple encounters. `train` is used to fit candidate models, `tuning` is used for hyperparameter optimization (HPO) and candidate selection, and `final_holdout` is kept separate until after selection to provide final evidence. Set `data.patient_id_column` to the patient ID column. The pipeline uses it to keep each patient's records in one role, then removes it from the data given to models.

`data.stratification_variables` lists columns used to balance the split, and `data.stratification_bins` gives one entry per variable in the same order. The lists must have equal length. A `null` bin entry uses that column's observed values directly; a list of labels requests the corresponding categories or configured numeric intervals. Configured variables are combined for joint stratification, so the pipeline attempts to preserve their joint distribution across roles. `data.protected_attribute_bins` is also positional: it has one entry per `data.protected_columns` value, with `null` for unbinned columns and one label list for each binned numeric column. For example, `[Sex, Age, Ethnicity]` aligns with `[null, ["<18", "18-30", "30-45", "45-60", "60+"], null]`.

Age interval labels use explicit `<N`, `N-M`, and `N+` syntax. They derive lower-inclusive, upper-exclusive bounds `[lower, upper)`; `<N` and `N+` are open-ended. Use one shared Age scheme wherever Age is binned: `[−∞,18)`, `[18,30)`, `[30,45)`, `[45,60)`, `[60,+∞)`, labeled `<18`, `18-30`, `30-45`, `45-60`, `60+`. Thus age 18 belongs in `18-30`, while age 30 belongs in `30-45`. When Age is a stratification variable, `data.stratification_bins` labels must match `data.protected_attribute_bins` labels in order; do not maintain separate Age cuts or labels. The bins also define protected-attribute slices for release evaluation; they do not collapse or remap the target, which remains evaluated in its native categories.

Shipped profiles configure HPO with TSTR macro-F1 (`tstr_macro_f1.v1`) as its sole objective on `tuning`. HPO does not create a separate binary-target evaluation pass or a positive/negative target mapping.

For patient-group splitting, the pipeline creates and reuses a local secret key (`.patient_id_hmac_key`) in the data folder by default. To use an external key, set `SYNTHDATA_PATIENT_ID_HMAC_KEY`; the pipeline uses that key instead. The pipeline uses it to create consistent patient-ID tokens in split and assignment files, reducing raw-ID exposure if those files are shared.

### Role overlap and threat-model boundaries

Declare each role explicitly in the config file; the pipeline does not guess one role from another. Important dataset roles are:
- `patient ID` identifies the same person across encounters,
- `target` is the outcome used to evaluate utility and fairness,
- `quasi-identifiers (QIs)` are explicitly selected columns an attacker could use to link records to people,
- `sensitive attributes` are columns whose disclosure is measured by privacy attacks,
- `protected attributes` identify groups used to check fairness. Checks include representation , equal opportunity, equalized odds, and log-disparity evidence. May overlap with QIs and sensitive attributes.

## Pipeline behavior

These four commands form the pipeline and run in this order:

1. **Impute** missing feature values using fixed HyperImpute plugins. The supported imputation path uses HyperImpute only. Candidate imputation fits on raw `train` and transforms `train` and `tuning`; `final_holdout` remains untouched. Candidate HPO and selection use only these candidate-imputed roles. After selection, final imputation fits a fresh imputer on raw `train` and `tuning` together, then uses that same fit to transform combined `train`+`tuning` and `final_holdout`. TabImpute and RefiDiff are blocked or deferred in the supported path because they cannot keep the roles isolated yet. Retained benchmark and older code are not supported paths for producing release results.
2. **Generate** candidate synthetic datasets using the configured SynthCity, TabPFN, and TabPFGen models. Optional Optuna hyperparameter searches are saved and can resume. Each HPO context is stored in a companion file (sidecar) versioned by a digest, a value calculated from the context, with a pointer to the latest one. Changing the objective creates a new study identity and preserves the earlier context evidence.
3. **Evaluate** candidates for utility, privacy, and fairness. Candidate rankings use only candidate-imputed `train` and `tuning`. After choosing a candidate, the pipeline refits its generator on final-imputed combined `train`+`tuning` and evaluates it once against final-imputed `final_holdout`. Evaluation can process models in parallel, within configured resource limits. It writes a ranked table, report, and diagnostic information.
4. **Plot** saves data quality, generation, HPO, and evaluation results without running earlier steps again.

Files (artifacts) produced by the pipeline are organized by dataset name, dataset version, and experiment ID. Folder names show both versions, for example, `data_v_1.2/exp_v_0.4/`. Generation creates a new experiment by default. Evaluation and plotting use the latest experiment or accept `--experiment-id` to open an earlier run. This keeps cached inputs, model outputs, HPO state, metrics, and figures separate across dataset revisions.

### Imputation artifacts

For canonical `train`/`tuning`/`final_holdout` datasets, version **0.9.0** stores phase-specific imputation artifacts under the configured dataset `data_dir`:

- `imputation_initial/` writes and uses these seven candidate-phase artifacts in the current layout: `train_imputed.csv`, `tuning_imputed.csv`, `train_imputed_decoded.csv`, `tuning_imputed_decoded.csv`, `full_candidate_partial_imputed.csv`, `full_candidate_partial_imputed_decoded.csv`, and `.imputation_cache_key.json`. The role files hold candidate-imputed `train` and `tuning`; the decoded files are label-preserving views. The `full_candidate_partial` aggregates cover all original rows in original row order: candidate-imputed `train` and `tuning`, plus raw, unchanged `final_holdout`. There is no separate candidate-imputed holdout file.
- `imputation_final/` writes and uses these seven final-phase artifacts in the current layout: `train_tuning_imputed.csv`, `final_holdout_imputed.csv`, `train_tuning_imputed_decoded.csv`, `final_holdout_imputed_decoded.csv`, `full_final_imputed.csv`, `full_final_imputed_decoded.csv`, and `.imputation_cache_key.json`. The two role files contain combined final-imputed `train`+`tuning` and final-imputed `final_holdout`; decoded files are label-preserving views. The `full_final` aggregates cover every original row in original row order, with every role transformed by the same final imputer fit on raw `train`+`tuning`. These are current-layout artifacts, not an exclusive folder listing; older files from prior layouts, such as `train_imputed.csv` and `tuning_imputed.csv`, may remain in `imputation_final/`.

Each phase's cache-key file is scoped to that phase's imputation inputs and lineage. Raw `full.csv`, raw role CSVs (`train.csv`, `tuning.csv`, and `final_holdout.csv`), and role assignment files/manifests under the dataset-root `assignments/` directory remain at the dataset root; phase-specific imputed artifacts do not replace them. Legacy two-role datasets retain the historical root-level imputation artifact layout and do not use these phase directories.

> [!TIP]
> See [`synthdata/config.py`](synthdata/config.py) for settings for cached data, computing devices, models, HPO, parallel evaluation, saved files, experiments, metrics, rankings, privacy gates, and plots.

> [!IMPORTANT]
> Generation and evaluation results are scientific records. Use a new study or experiment identifier for new work. Keep the generated manifests and saved configuration copies needed to reproduce each result.

## Development

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
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
