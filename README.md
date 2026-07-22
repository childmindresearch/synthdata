# SynthData

A sandbox for synthetic data generation and evaluation.

It keeps forks of [`syntheval`](https://github.com/schneiderkamplab/syntheval) and [`synthcity`](https://github.com/vanderschaarlab/synthcity) as editable submodules to make it easy to test new features and bug fixes in those libraries. It also contains some early versions of apps, notebooks, and scripts for testing out different synthetic data generation and evaluation techniques.

## Quick Start

Clone the repo, initialize submodules, and install the environment with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/childmindresearch/synthdata.git
cd synthdata
git submodule update --init --recursive
uv sync --extra tabpfn
```

The `tabpfn` extra (TabPFN, TabPFGen, TabImpute) is required to run the imputation/generation/evaluation pipeline below. An optional `refidiff` extra (`uv sync --extra tabpfn --extra refidiff`) adds RefiDiff, an alternative imputation backend (`imputation.method: refidiff`) for wide datasets where TabImpute's one-hot encoding runs out of GPU memory -- see the Imputation summary below.

## Pipeline: imputation -> generation -> evaluation -> plots

The `synthdata` package (and the four `synthdata-*` CLI commands it installs) turns the
exploratory work from `notebooks/test_hepatitis_data.ipynb` and
`notebooks/ctgan_hpo_hepatitis.ipynb` into a reusable, config-driven pipeline that any
user can run on their own local data. A single YAML file drives everything --
copy [`configs/config.yaml`](configs/config.yaml) (a working example on the UCI
Hepatitis dataset) and edit the `data:` section to point at your own CSV.

```bash
synthdata-impute   --config configs/config.yaml --plot   # load + impute missing data
synthdata-generate --config configs/config.yaml --plot   # synthcity + TabPFN + TabPFGen (+ Optuna HPO)
synthdata-evaluate --config configs/config.yaml --plot   # synthcity + SynthEval + custom fairness metrics
synthdata-plot      --config configs/config.yaml         # (re)generate every figure from cached artifacts
```

Each stage caches its outputs to disk (imputed data under `data/<name>/`, synthetic
CSVs + Optuna studies under `output/<name>/synthetic_data/`, the combined evaluation
table under `output/<name>/evaluation/`, figures under `output/<name>/plots/`) so
later stages -- or a re-run of `synthdata-plot` -- don't require recomputation.

### Experiment tracking & dataset versioning

Every `synthdata-generate` run is tracked as a new **experiment**: a timestamped id
(optionally labeled with `--tag <label>`), with that run's synthetic data /
evaluation / plot artifacts nested under `<output_dir>/<experiment_id>/`, and a
`manifest.json` under `output/<name>/experiments/<experiment_id>/` logging what each
stage produced (git commit, dataset version, artifact paths). `synthdata-evaluate`
and `synthdata-plot` automatically target the most recently generated experiment, or
a specific past one via `--experiment-id <id>`:

```bash
synthdata-generate --config configs/config.yaml --tag baseline
synthdata-evaluate --config configs/config.yaml            # picks up the "baseline" experiment automatically
synthdata-generate --config configs/config.yaml --tag hpo-v2
synthdata-evaluate --config configs/config.yaml --experiment-id <baseline's id>  # re-evaluate the earlier one explicitly
```

Separately, `data.version` (and `synthdata-impute --dataset-version <label>`) lets
you version the *dataset* itself: cached raw/imputed/split CSVs are nested under
`data/<name>/<version>/`, with a `dataset_manifest.json` recording the source config
that produced them. This is independent of experiments, since the same dataset
version is typically reused across many generation experiments.

 Summary of the four stages:

- **Imputation** (`synthdata/imputation/`): missing-data imputation with
  categorical-aware handling and post-imputation validation, via one of two
  backends selected by `imputation.method`:
  - `tabimpute` (default): TabImpute-based, one-hot categorical encoding.
  - `refidiff`: per-column XGBoost/CatBoost warm-up + Mamba-based EDM diffusion
    refinement, with binary (log2(k)-bit) categorical encoding. Scales better
    to wide datasets (many categorical columns) where TabImpute's one-hot
    encoding runs out of GPU memory -- see `configs/config_loris.yaml` for an
    example. Requires the `refidiff` extra; falls back to a plain MLP denoiser
    (`imputation.refidiff.denoiser: mlp`) if `mamba-ssm` isn't installed/importable.
- **Generation** (`synthdata/generation/`): synthcity plugins (CTGAN, TVAE, ADS-GAN,
  Bayesian network, PATE-GAN, RTVAE, DDPM, ...), TabPFN (standard/custom unsupervised
  synthesis), and TabPFGen (standard/custom SGLD-based synthesis), each with an
  optional Optuna-tuned `*_hpo` variant. HPO studies are persisted to SQLite
  (resumable, inspectable with `optuna-dashboard`) and best hyperparameters are
  cached to `output/<name>/hpo_best_params.json`.
- **Evaluation** (`synthdata/evaluation/`): combines synthcity metrics, SynthEval
  metrics, and custom fairness metrics (log disparity, plus the equalized-odds/
  equal-opportunity metrics added to this repo's SynthEval fork) into a single
  table with `(framework, type, metric)`-multi-indexed columns
  (`framework in {synthcity, syntheval, custom}`, `type in {utility, privacy, fairness}`),
  plus per-group, per-type, and overall ranking columns. Supports partial selection
  of metrics, either by `type` category or by explicit metric name, per framework.
- **Plotting** (`synthdata/plotting/`): every figure from the notebooks (column/
  missingness distributions, observed-vs-imputed validation, real-vs-synthetic
  comparisons, Optuna diagnostics, utility/privacy/fairness rank trade-offs,
  log-disparity sunburst reports, per-model SynthEval plots).

See `configs/config.yaml` for the full set of options (all documented inline).

## Testing & linting

```bash
uv sync --extra tabpfn        # dev tools (pytest/ruff) are installed by default via [dependency-groups]
uv run pytest                 # runs everything except @pytest.mark.slow/network tests
uv run ruff check .           # lint (scoped to synthdata/, scripts/*.py, tests/)
uv run ruff format .          # format
pre-commit install            # optional: run ruff automatically on commit
```

Tests live under `tests/`, mirroring `synthdata/`'s module layout, with shared
fixtures in `tests/conftest.py`. Markers (`unit`, `cache`, `integration`, `slow`,
`network`; see `[tool.pytest.ini_options]` in `pyproject.toml`) let you scope a run,
e.g. `uv run pytest -m unit`. CI (`.github/workflows/ci.yml`) runs `ruff check`/
`ruff format --check` plus `pytest -m "not slow and not network"` against base
dependencies only (no `tabpfn` extra, CPU-only torch) on every push/PR.

## Apps

These are older, less actively maintained tracks -- install their extras separately with `uv sync --extra <name>` when you need them.

- [`apps/presidio/presidio_streamlit.py`](apps/presidio/presidio_streamlit.py) (`uv sync --extra presidio`): Presidio's Streamlit app, modified for offline use. For the full version of the anonymizer, see [`anonymize-pii`](https://github.com/childmindresearch/anonymize-pii).

    See [PRESIDIO APP GUIDE](apps/presidio/PRESIDIO_APP_GUIDE.md) for details.

## Notebooks

- [`notebooks/ydata-test.py`](notebooks/ydata-test.py) (`uv sync --extra ydata`): Testing ydata-synthetic library for tabular data synthesis. To run using [`marimo`](https://github.com/marimo-team/marimo):

    ```bash
    uv run --extra ydata marimo run notebooks/ydata-test.py
    ```

- [`notebooks/ctgan_hpo_hepatitis.ipynb`](notebooks/ctgan_hpo_hepatitis.ipynb): Testing CTGAN synthesis with Optuna hyperparameter optimization on the hepatitis dataset.

- [`notebooks/test_hepatitis_data.ipynb`](notebooks/test_hepatitis_data.ipynb): Testing imputation (TabImpute), synthcity generators (+TabPFN/TabPFGen), hyperparameter optimization of all models (Optuna) and syntheval & synthcity evaluations on the hepatitis dataset.

- [`notebooks/tabpfn_demo.ipynb`](notebooks/tabpfn_demo.ipynb): Testing classification and synthetic data generation with TabPFN. Add a `TABPFN_TOKEN` (and optionally `HF_TOKEN`) to an `.env` file at the root of the project to access the TabPFN API (and download HuggingFace models faster).

## Scripts

- [`scripts/run_imputation.py`](scripts/run_imputation.py), [`scripts/run_generation.py`](scripts/run_generation.py), [`scripts/run_evaluation.py`](scripts/run_evaluation.py), [`scripts/run_plots.py`](scripts/run_plots.py): CLI entry points for the pipeline described above (installed as `synthdata-impute`/`synthdata-generate`/`synthdata-evaluate`/`synthdata-plot`).
- [`scripts/document_pipeline/`](scripts/document_pipeline): early PII-anonymization / markdown-parsing scripts (`NER.py`, `docparser.py`, `markdown_parser.py`, `section_loader.py`), unrelated to the synthetic data pipeline above. `markdown_parser.py` is an early, monolithic version; for the full version, see [`headhunter`](https://github.com/childmindresearch/headhunter).
