# SynthData

SynthData is a config-driven pipeline for tabular-data imputation, synthetic data generation, evaluation, and plots. It is designed to run on **your own local CSV or Parquet data**.

> [!NOTE]
> The example configurations cover the public UCI Hepatitis dataset (small, runs on a CPU), a simulated clinical dataset, and LORIS. The simulated and LORIS data are not distributed with the repository; they are used to test the pipeline at a larger scale. Configure your own dataset rather than relying on them.

New to the pipeline? Read the [user manual](docs/user-manual.md): it walks through every stage, the config file, the report and troubleshooting.

## Quick start

Clone the repository, initialize its editable library submodules, and install the pipeline dependencies:

```bash
git clone https://github.com/childmindresearch/synthdata.git
cd synthdata
git submodule update --init --recursive
uv sync --extra tabpfn
```

The `tabpfn` extra installs the TabPFN and TabPFGen generators; TabPFN needs a `TABPFN_TOKEN` in a `.env` file at the repository root, and its model weights are for non-commercial use only, outputs included; read [TabPFN model license](docs/user-manual.md#tabpfn-model-license) before sharing TabPFN-generated data. Add `--extra refidiff` only if you set `imputation.method: refidiff`.

Start from [`configs/config_hepatitis.yaml`](configs/config_hepatitis.yaml) (or [`configs/config_sim.yaml`](configs/config_sim.yaml) for a wide, multi-encounter dataset), then update its `data:` section and variable-schema path for your local data.

```bash
uv run synthdata-impute   --config path/to/your-config.yaml --plot
uv run synthdata-generate --config path/to/your-config.yaml --plot
uv run synthdata-evaluate --config path/to/your-config.yaml --plot
uv run synthdata-plot     --config path/to/your-config.yaml
```

Or run impute, generate and evaluate in one go (each drawing its figures) with a log file, in the background on a remote machine (see [Long runs and remote machines](docs/user-manual.md#long-runs-and-remote-machines)):

```bash
uv run synthdata-run --config path/to/your-config.yaml --detach
```

While it runs, `<output root>/logs/<UTC time>_dashboard.html` is a live status page for the run (stages, models, HPO progress, resources, errors, results); open it in a browser, or serve it with `synthdata-dashboard --serve` (see [The live status page](docs/user-manual.md#the-live-status-page)).

## Your dataset configuration

One YAML file drives every stage. Its sections (`data`, `imputation`, `generation`, `evaluation`, `compute`, `plots`, `experiment`) follow the pipeline order. The example files list only the options worth reviewing for each run: data source and column roles, the patient-level split, imputation method, generators and their HPO budgets, metric selection and ranking weights. Every option left out takes its default. [`synthdata/config.py`](synthdata/config.py) is the full reference: every option there has its default and an explanation, and an unknown or misspelled key stops the run with the list of valid keys.

The settings to check first for a new dataset:

- `data.target_column`, `data.patient_id_column` (so repeat visits of one patient never cross the train/test boundary) and `data.stratify_columns`.
- `data.quasi_identifier_columns`, `data.sensitive_columns` and `data.protected_columns`, which drive the privacy attacks and fairness metrics.
- `generation.n_samples`, `generation.hpo` budgets, and `generation.n_replicates` (2 or more gives every score a confidence interval).

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

1. **Impute** missing feature values with MissForest (default), median/mode, TabImpute or RefiDiff. The data are first split by patient into train, tuning and holdout (60/20/20, stratified by the target and chosen columns). The imputer is fitted on train minus tuning for HPO and refitted on all of train for the final models (with HPO off only the final fit runs, so the models use all train rows and the holdout stays out), each run cached in its own folder (`imputation_initial/`, `imputation_final/`) and checked by a validation report saved under `output/<name>/imputation/`; frequently missing columns get `<column>__missing` indicators, and synthetic values are blanked again where the synthetic indicator is 1 (`released/` next to the synthetic CSVs).
2. **Generate** candidate synthetic datasets with configured SynthCity and TabPFN models; optional Optuna hyperparameter searches are persisted and resumable. Each search trial fits on train minus tuning and maximizes the macro-F1 of a fixed XGBoost trained on the synthetic rows and tested on the real tuning rows (TSTR, averaged over 3 seeds; macro AUPRC and the real-data ceiling are logged too). Trials that copy training rows, drop categories or classes, produce out-of-range values, or leave a class short fail lenient screens and are never picked. Every synthetic dataset is drawn class by class to the real train class shares, by conditional or rejection sampling without repeating rows (`generation.match_class_prior`).
3. **Evaluate** candidates for utility, privacy, and fairness on the holdout split, next to two fixed baselines (a copy of the real training rows and independently sampled column marginals). Privacy includes Anonymeter attacks and holdout-referenced distance checks, counted per patient, reported as evidence next to the baselines rather than as a pass/fail verdict (no fixed threshold can certify privacy); models are ranked by a weighted geometric mean of their utility, privacy and fairness ranks. Evaluation can process models in parallel within configured resource limits and writes a ranked table, diagnostics, sanity checks of its own output (`checks.csv`: copied training rows, invented categories, out-of-range values, class shares, and whether the two baselines land where they must), and `report.md`, the one page that summarizes the run, embeds every plot and links every output file. Class-dependent scores weigh every class equally: holdout TSTR (the HPO classifier fitted on each dataset and scored on the test split) reports macro-F1, balanced accuracy, macro AUPRC and per-class F1 next to the real-data ceiling (the `baseline_train_copy` rows), plus train-on-synthetic, test-on-synthetic (TSTS) scores (`tstr_holdout.csv`); SynthEval's `cls_acc` uses macro F1; and metrics that need two classes run once per class against the rest and are averaged on a 3+ class target (`evaluation.class_averaging: ovr_macro`, or `binary` with `evaluation.binary_target`).
   What each metric measures and which values are good: [docs/metrics-guide.md](docs/metrics-guide.md).
4. **Plot** recorded data-quality, generation, HPO, and evaluation artifacts without rerunning earlier stages.

Artifacts are namespaced by dataset name, dataset version, and experiment ID. Their folder names make both levels explicit: for example, `data_v_1.2/exp_v_0.4/`. Generation creates a new experiment by default; evaluation and plotting use the latest one or accept `--experiment-id` to revisit a prior run. This preserves cached inputs, model outputs, HPO state, metrics, and figures across dataset revisions.

> [!TIP]
> See [`synthdata/config.py`](synthdata/config.py) for cache, device, model, HPO, parallel evaluation, artifact, experiment, metric, ranking, and plot options.

## Reproducibility

> [!CAUTION]
> Generation and evaluation results are scientific artifacts. Use a fresh experiment identifier for new work, and retain the generated manifests and configuration snapshots needed to reproduce a result.

## Development

```bash
uv run pre-commit install   # once: formatting, and refuses to commit data files
uv run pytest -m "not integration"
uv run --with catboost==1.2.10 pytest tests/integration -m "integration and not slow and not regression"
uv run ruff check .
uv run ruff format --check .
uv run ty check             # type check; only errors fail, see [tool.ty] in pyproject.toml
```

[ty](https://docs.astral.sh/ty/) is also a language server: install its editor extension (VS Code: "ty" by Astral) to see type errors as you type.

After a dependency or submodule bump, run the golden test (`pytest tests/integration -m regression`), which flags any metric that moved; see the [integration test guide](tests/integration/README.md).

Tests are under [`tests/`](tests/). See `pyproject.toml` for available test markers and dependency extras.

## Legacy and exploratory code

The pipeline above is the supported path for new work. The following areas are older or exploratory tracks retained for reference and targeted experimentation.

> **Deleted legacy code.** Version 0.8.2 removed the Presidio PII app (`apps/presidio/`, the `presidio` extra), the document-parsing scripts (`scripts/document_pipeline/`) and the ydata-synthetic notebook (`notebooks/ydata-test.py`, the `ydata` extra). None of them was part of the pipeline, they were no longer maintained, and their old pinned dependencies caused most of the repository's security alerts. They are still in git history: the tag `legacy-archive` points at the last commit that has them. To look at or reuse one, run `git checkout legacy-archive -- <path>` (or browse the tag on GitHub), and update its dependencies before running it.

### Notebooks

- [`notebooks/ctgan_hpo_hepatitis.ipynb`](notebooks/ctgan_hpo_hepatitis.ipynb) and [`notebooks/test_hepatitis_data.ipynb`](notebooks/test_hepatitis_data.ipynb): earlier Hepatitis-focused synthesis, imputation, HPO, and evaluation work.
- [`notebooks/tabpfn_demo.ipynb`](notebooks/tabpfn_demo.ipynb): TabPFN classification and synthesis exploration. Configure `TABPFN_TOKEN` (and, optionally, `HF_TOKEN`) in `.env` when the API is required.

### Imputation benchmark

- `synthdata-imputation-benchmark` ([`synthdata/imputation/benchmark.py`](synthdata/imputation/benchmark.py)): an exploratory masked-cell benchmark for RefiDiff settings. It has not been validated since its first implementation and is not part of the pipeline; its options (`imputation.benchmark`) are documented in [`synthdata/config.py`](synthdata/config.py).

## Repository layout

- [`docs/`](docs/): the [user manual](docs/user-manual.md) and the [verification of generators and metrics](docs/verification.md).
- [`synthdata/`](synthdata/): pipeline implementation.
- [`configs/`](configs/): example configurations (Hepatitis, simulated data, LORIS).
- [`scripts/`](scripts/): installed CLI entry points.
- [`notebooks/`](notebooks/): optional exploratory work.
