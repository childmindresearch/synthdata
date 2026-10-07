# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (Python 3.11 only, managed with `uv`; `submodules/synthcity` and `submodules/syntheval` are forks installed editable, so init them first):

```bash
git submodule update --init --recursive
uv sync --extra tabpfn        # base deps + TabPFN/TabPFGen/TabImpute
```

Checks (same as CI, `.github/workflows/ci.yml`):

```bash
uv run pytest -m "not slow and not network"     # what CI runs (CI installs base deps only, no tabpfn extra)
uv run pytest tests/unit/test_config.py::test_name   # single test
uv run ruff check && uv run ruff format --check
uv run ty check                                  # scope: synthdata/ + scripts/ (excl. scripts/document_pipeline)
```

Pytest markers (`--strict-markers` is on): `unit`, `cache`, `integration`, `slow` (real model training), `network`. `tests/conftest.py` forces pytest's basetemp into a repo-local `tmp/<run-id>/` and refuses symlinked/insecure scratch dirs; tests must not touch `data/` or `output/`.

Ruff and pre-commit are scoped to `synthdata/`, `scripts/` (minus `document_pipeline/`), and `tests/`. `apps/`, `notebooks/`, `scripts/document_pipeline/`, `benchmarks/`, and `submodules/` are legacy/exploratory or vendored and out of scope.

Pipeline CLIs (entry points in `scripts/run_*.py`, registered in `pyproject.toml`), run in order:

```bash
uv run synthdata-test     --config configs/config_hepatitis.yaml   # audit config + data/schema/split before running
uv run synthdata-impute   --config ... [--plot]
uv run synthdata-generate --config ... [--plot] [--tag T] [--experiment-id ID] [--dataset-version V]
uv run synthdata-evaluate --config ... [--plot] [--experiment-id ID]
uv run synthdata-plot     --config ...
```

`configs/config_hepatitis.yaml` (small, UCI download) and `configs/config_loris.yaml` (wide, local CSV) are the example/development configs.

## Architecture

Single YAML config → `synthdata.config.load_config` → nested dataclasses (`Config`, `DataConfig`, `GenerationConfig`, `EvaluationConfig`, …) in `synthdata/config.py`, which is the source of truth for every setting and default. Config validation is strict: removed settings raise errors via `_MigrationConfig` rather than being silently ignored.

**Data roles are the central invariant.** `synthdata/data.py` loads the source (UCI/CSV/Parquet) plus a variable-schema CSV into a `Dataset`; `synthdata/data_roles.py` allocates patient-disjoint roles `train` / `tuning` / `final_holdout` (grouped by `data.patient_id_column`, or `split.one_row_per_patient`). Patient IDs are replaced by HMAC tokens (local `.patient_id_hmac_key` or `SYNTHDATA_PATIENT_ID_HMAC_KEY`) and dropped before modeling. Everything downstream must preserve role isolation:
- Candidate phase: imputer fit on raw `train`, applied to `train`/`tuning`; HPO, candidate ranking and selection use only these. `final_holdout` is untouched.
- Final phase: fresh imputer fit on `train`+`tuning`; selected generator refit on that; evaluated once against final-imputed `final_holdout` (`synthdata/evaluation/release*.py`).

Stages:
- `synthdata/imputation/` — supported path is HyperImpute only (`hyperimpute_backend.py`). TabImpute/RefiDiff backends and `benchmark.py` are retained but blocked/deferred because they can't keep roles isolated.
- `synthdata/generation/` — `pipeline.py` orchestrates synthcity, TabPFN, and TabPFGen backends (tabpfgen imported lazily so base-deps CI works); `hpo.py` runs resumable Optuna studies with digest-versioned context sidecars and a latest pointer. Changing the objective creates a new study identity.
- `synthdata/evaluation/` — `__init__.py::run_evaluation` orchestrates synthcity, SynthEval (`syntheval_eval.py`, parallel workers bounded by `evaluation.syntheval_execution`), custom fairness/log-disparity (`custom_eval.py`, `synthdata/log_disparity/`), and TSTR; `combine.py` merges into one ranked multi-index table; `metric_contracts.py`/`catalog.py` define metric semantics; `privacy_gate.py` and `release_score.py` gate/score candidates; `artifacts.py` handles persisted evidence. Evaluation is currently documented as temporarily disabled/under validation.
- `synthdata/plotting/` — figures only; never reruns earlier stages.

**Artifacts and caching.** Outputs are scoped by dataset version and experiment: `.../data_v_<version>/exp_v_<experiment_id>/` (`synthdata/experiment.py`). `synthdata-generate` starts a new experiment and writes a `latest.json` pointer that evaluate/plot reuse unless `--experiment-id` is given; a per-experiment `manifest.json` records git commit and artifact paths. Stage caches are keyed by fingerprints/digests of data, role context, and config (e.g. `_cache_key_record`, `validate_imputation_cache_lineage`); lineage mismatches are meant to fail loudly rather than reuse stale results. Results are treated as scientific records — prefer new experiment/study IDs over overwriting.

**Import-time side effects** in `synthdata/__init__.py`: forces `TABPFN_DISABLE_TELEMETRY=1` (inherited by child workers — do not weaken), loads `.env`, preloads NVRTC on ARM64, and forces matplotlib `Agg`. Standalone notebooks don't get these and must set telemetry off themselves.

## Conventions

- Commit messages follow Conventional Commits with scopes, using `!` for breaking changes (e.g. `feat(data)!: ...`, `fix(evaluation): ...`).
- Line length 100, double quotes; ruff rule sets `E,F,W,I,UP,B,BLE,SIM`.
