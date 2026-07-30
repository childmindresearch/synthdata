# SynthData

SynthData is a config-driven pipeline for tabular-data imputation, synthetic data generation, evaluation, and plots. It is designed to run on **your own local CSV or Parquet data**.

> [!NOTE]
> This repository includes Hepatitis and LORIS data only as local development fixtures and examples. LORIS is used here to test the pipeline at a larger scale; users do not need it and should configure their own dataset.

## Quick start

Clone the repository, initialize its editable library submodules, and install the pipeline dependencies:

```bash
git clone https://github.com/childmindresearch/synthdata.git
cd synthdata
git submodule update --init --recursive
uv sync --extra tabpfn --extra refidiff
```

Start from [`configs/config_hepatitis.yaml`](configs/config_hepatitis.yaml) (or [`configs/config_loris.yaml`](configs/config_loris.yaml) for a wide-dataset example), then update its `data:` section and variable-schema path for your local data. [`synthdata/config.py`](synthdata/config.py) is the reference for all configuration settings and defaults.

```bash
uv run synthdata-impute   --config path/to/your-config.yaml --plot
uv run synthdata-generate --config path/to/your-config.yaml --plot
uv run synthdata-evaluate --config path/to/your-config.yaml --plot
uv run synthdata-plot     --config path/to/your-config.yaml
```

## Your dataset configuration

Provide a variable-schema CSV with one row for each feature and target. The schema explicitly declares a column as `categorical` or `continuous`, and records the ordering for ordinal categorical values.

```csv
column,kind,ordinal_order
Age,continuous,
Sex,categorical,
Severity,categorical,"[0, 1, 2, 3]"
target,categorical,
```

> [!TIP]
> Treat the example configurations as templates, not defaults for your data. Dataset paths, feature roles, target, version, generation methods, and evaluation settings should be reviewed for each project.

## Pipeline behavior

The four commands form an ordered pipeline:

1. **Impute** missing feature values using TabImpute or RefiDiff, with schema-aware caching and automatic CPU/GPU device selection.
2. **Generate** candidate synthetic datasets with configured SynthCity, TabPFN, and TabPFGen models; optional Optuna hyperparameter searches are persisted and resumable.
3. **Evaluate** candidates for utility, privacy, and fairness. Evaluation can process models in parallel within configured resource limits and writes a ranked table, report, and diagnostics.
4. **Plot** recorded data-quality, generation, HPO, and evaluation artifacts without rerunning earlier stages.

Artifacts are namespaced by dataset name, dataset version, and experiment ID. Their folder names make both levels explicit: for example, `data_v_1.2/exp_v_0.4/`. Generation creates a new experiment by default; evaluation and plotting use the latest one or accept `--experiment-id` to revisit a prior run. This preserves cached inputs, model outputs, HPO state, metrics, and figures across dataset revisions.

> [!TIP]
> See [`synthdata/config.py`](synthdata/config.py) for cache, device, model, HPO, parallel evaluation, artifact, experiment, metric, ranking, privacy-gate, and plot options.

## Optional: RefiDiff masked-cell benchmark

Use `synthdata-imputation-benchmark` when selecting RefiDiff settings for your dataset, especially when it has many columns. This is separate from normal imputation: it temporarily hides a sample of values that were originally observed in the **training split**, imputes them, and compares the predictions with the known values. Numeric columns are assessed by standardized error and categorical columns by accuracy and balanced accuracy. Sensitive columns are not masked or scored.

The benchmark does not overwrite your regular imputed datasets. It writes an append-only study containing the masks, parameter settings, per-column metrics, and aggregate results under the configured imputation output directory.

```bash
uv run synthdata-imputation-benchmark \
  --config path/to/your-benchmark-config.yaml \
  --study-id refidiff-baseline-001
```

For a wide-data example, see the included LORIS-based [reference profile](configs/config_loris_refidiff_reference.yaml) and [HPO screening profile](configs/config_loris_refidiff_hpo.yaml). Adapt a copy to your own data rather than using their data paths or column choices.

### Recommended benchmark sequence

1. **Establish a baseline.** Run a fixed-parameter benchmark on a small, representative panel of columns. This confirms that RefiDiff can recover values in your data and gives later candidates a comparison point.
2. **Screen settings.** Run a bounded HPO study to explore a practical range of model width, training budget, and sampling settings. Review its `best_trial.json` and per-column metrics; do not select a candidate only from its single aggregate score.
3. **Confirm a candidate.** Copy the selected settings into a new, fixed- parameter benchmark and use a fresh study ID. Expand to the columns and missingness scenarios that matter for the intended analysis (for example, MCAR, MAR, and MNAR) before adopting the settings for production imputation.

> [!IMPORTANT]
> Reuse a study ID only to resume the exact same study. Changing the dataset, schema, masking plan, or settings requires a new study ID so results remain comparable and traceable. For implementation details and the precise artifact layout, see [`synthdata/imputation/benchmark.py`](synthdata/imputation/benchmark.py) and [`synthdata/config.py`](synthdata/config.py).

> [!CAUTION]
> Benchmark and generation results are scientific artifacts. Use a fresh study or experiment identifier for new work, and retain the generated manifests and configuration snapshots needed to reproduce a result.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests are under [`tests/`](tests/). See `pyproject.toml` for available test markers and dependency extras.

## Legacy and exploratory code

The pipeline above is the supported path for new work. The following areas are older or exploratory tracks retained for reference and targeted experimentation:

### Apps

- [`apps/presidio/presidio_streamlit.py`](apps/presidio/presidio_streamlit.py): an offline-focused Presidio Streamlit app for PHI/PII processing. Install its dependencies with `uv sync --extra presidio`; see the [Presidio app guide](apps/presidio/PRESIDIO_APP_GUIDE.md) for setup and security guidance.

### Notebooks

- [`notebooks/ydata-test.py`](notebooks/ydata-test.py): ydata-synthetic experimentation; requires `uv sync --extra ydata`.
- [`notebooks/ctgan_hpo_hepatitis.ipynb`](notebooks/ctgan_hpo_hepatitis.ipynb) and [`notebooks/test_hepatitis_data.ipynb`](notebooks/test_hepatitis_data.ipynb): earlier Hepatitis-focused synthesis, imputation, HPO, and evaluation work.
- [`notebooks/tabpfn_demo.ipynb`](notebooks/tabpfn_demo.ipynb): TabPFN classification and synthesis exploration. Configure `TABPFN_TOKEN` (and, optionally, `HF_TOKEN`) in `.env` when the API is required.

### Scripts

- [`scripts/document_pipeline/`](scripts/document_pipeline): early PII-anonymization and Markdown-parsing work unrelated to the synthetic-data pipeline.

## Repository layout

- [`synthdata/`](synthdata/): pipeline implementation.
- [`configs/`](configs/): example configurations.
- [`scripts/`](scripts/): installed CLI entry points.
- [`apps/`](apps/) and [`notebooks/`](notebooks/): optional exploratory work.
