# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. Humans should start with [README.md](README.md) and the [user manual](docs/user-manual.md).

SynthData generates synthetic versions of a clinical table and evaluates them for utility, privacy and fairness. Results are scientific artifacts: correctness and leakage-safety matter more than speed or new features.

## Rules

- Never say where the target dataset comes from or what event it was released for, in code, docs, configs, commit messages or PR text. It is not public; call it "the simulated clinical dataset" or "the clinical dataset".
- Never commit data. `data/` and `output/` are git-ignored and the `no-data-files` pre-commit hook (also run in CI) refuses data files. Hepatitis is the only dataset whose schema is committed (`configs/schemas/`).
- Every change to how a generator or metric behaves compared with its paper or library goes in [docs/verification.md](docs/verification.md), in that model's or metric's row, with the reason and the evidence (known-answer check, upstream issue, paper section). Mark it Fixed, Deviation, Open or Unranked as the legend at the top of that file describes.
- Prefer library code over custom code. Fix upstream bugs in the forks (below), not with monkey-patches here. Custom code needs a stated reason and a test against a reference.
- Never force-push or rewrite history on shared branches. Never modify `feat/final-release-evaluation` (old refactor, kept only as reference).
- Keep the user manual in step with the code: a change to a config key, CLI flag, stage or output file updates the matching section of `docs/user-manual.md` in the same PR, and the option's docstring in `synthdata/config.py`.

## Layout

- `synthdata/`: the pipeline. `config.py` (every option, with docs), `data.py` (load, clean, patient split), `data_readiness.py`, `imputation/`, `generation/` (`hpo.py`, `synthcity_backend.py`, `tabpfn_backend.py`, `class_quota.py`), `evaluation/` (`syntheval_eval.py`, `synthcity_eval.py`, `custom_eval.py`, `tstr.py`, `privacy_attacks.py`, `baselines.py`, `checks.py`, `report.py`), `compute.py` and `gpu_slots.py` (parallelism), `dashboard.py`, `plotting/`.
- `scripts/`: CLI entry points (`synthdata-impute`, `-generate`, `-evaluate`, `-plot`, `-run`, `-dashboard`), plus `profile_hpo.py`, `hpo_analysis.py`, `gpu_concurrency.py`.
- `configs/`: `config_hepatitis.yaml` (quick, public), `config_sim.yaml` (simulated clinical data, ~680 columns), `config_loris.yaml` (~1090 columns).
- `submodules/synthcity`, `submodules/syntheval`: our forks `alperkent-cmi/synthcity` and `alperkent-cmi/syntheval`, installed editable. A fix there is a commit on the fork's `main`, then a submodule bump here.
- `tests/unit/`, `tests/integration/` (real CLIs on the committed fixture `tests/integration/fixtures/clinic.csv`; see its README).
- `apps/`, `notebooks/`: legacy or exploratory, not part of the pipeline.

## Commands

```bash
git submodule update --init --recursive
uv sync --group dev --extra tabpfn        # plain `uv sync --group dev` removes TabPFN
uv run pre-commit install

uv run ruff check . && uv run ruff format --check .
uv run ty check                                                           # type check; CI fails on errors only
uv run pytest -m "not integration and not slow and not network"           # unit, as CI
uv run --with catboost==1.2.10 pytest tests/integration \
  -m "integration and not slow and not network and not regression"        # integration, as CI (needs catboost)
uv run --with catboost==1.2.10 pytest tests/integration -m regression     # golden check
SYNTHDATA_UPDATE_GOLDEN=1 uv run --with catboost==1.2.10 pytest tests/integration -m regression  # re-record

uv run synthdata-run --config configs/config_hepatitis.yaml               # full pipeline
uv run synthdata-run --config configs/config_sim.yaml --detach            # long run in background
```

- Golden references live in `tests/integration/fixtures/golden/baseline_run_<os>_<device>.csv`. Linux CUDA is canonical, Linux CPU secondary; other platforms skip. When a PR changes a metric or generator on purpose, re-record on the Linux GPU machine and commit the CSV with the change (or say in the PR that it needs re-recording).
- Run the golden check after any dependency or submodule bump.
- ty is beta and pinned in the dev group. Its warnings include a known false positive on pandas (a list passed as `columns=`); fix new errors rather than adding `# ty: ignore`, and when an ignore is unavoidable give the reason on the same line.
- If a container cannot reach download.pytorch.org, `uv sync` fails on the CUDA torch index; export requirements without `+cu128`/`nvidia-*` and `uv pip install --no-sources` instead.

## Config

One YAML file per dataset is the single source of truth; `synthdata/config.py` defines and documents every key, and an unknown or renamed key raises. Main sections: `data` (paths, target, column roles from the variable schema, split, readiness), `imputation`, `generation` (models, `n_samples` where null = train size, `match_class_prior`, `n_replicates`, `hpo`), `compute` (workers, cores, memory, GPU slots), `evaluation` (metric lists, `class_averaging`, ranking weights), `plots`. Column types come only from the variable schema, never from guessing.

## Pipeline invariants

Break any of these and results are invalid even if every test runs:

- Split by patient (60/20/20 search-train, tuning, holdout), stratified. No patient spans splits.
- Imputers are fitted on training rows only: search-train for HPO, all train for final fits. Holdout never reaches imputation, HPO or generation (canary tests in `test_reproducibility_and_leakage.py` check this).
- HPO fits on search-train and scores on tuning: macro-F1 of a fixed XGBoost, train-on-synthetic test-on-real, averaged over 3 seeds. Trials failing constraint screens (copies, missing categories or classes, out-of-range values, `protected_share_gap`) are never selected; if none pass, the model falls back to synthcity defaults.
- Evaluation uses the holdout, next to two baselines (copy of train, independent marginals) that must land where expected (`checks.csv`).
- Seeds are fixed everywhere, including SynthEval's random metrics. Same config and `cores_per_worker` give identical numbers.
- Ranks use only metrics that separate a copy of train from an independent draw; copy-blind metrics are reported but unranked (see verification.md).

## Pitfalls

- TabPFN unsupervised generation costs roughly columns squared; about 10 h per fit on the simulated data (~680 columns) and infeasible on LORIS (~1090). Turn `generation.tabpfn.enabled` off for wide data.
- DomiasMIA_prior fails on wide data (singular KDE) and is opt-in only.
- On wide, small datasets (LORIS) HPO trials can all fail screens because generators are undertrained in short budgets; check `hpo` logs before trusting `_hpo` models.
- `torch` results differ by OS and device, so compare numbers only within one platform.
- One run per output root: a lock in `<output root>/logs/run.lock` stops a second concurrent run.
- Changing the HPO objective or screens makes old Optuna studies stale: use a new experiment id rather than resuming.
- TabPFN telemetry is disabled by default (`TABPFN_DISABLE_TELEMETRY=1` in `synthdata/__init__.py`); keep it that way, the data are sensitive.

## Pull requests

- Use the conventional-commit style seen in the log: `fix(evaluation): ...`, `feat(generation): ...`, `docs(manual): ...`, `test(integration): ...`.
- Open PRs as drafts against `main`. Before pushing, run ruff and the unit tests; run the integration suite when a stage's behaviour changes.
- Describe what a reader would see change, and link the verification.md row for metric or generator changes.
- Every PR bumps `version` in `pyproject.toml` (and the `synthdata` entry in `uv.lock`) and adds an entry to [CHANGELOG.md](CHANGELOG.md): minor for new features, changed results or config edits users must make, patch for fixes that don't. The entry says what users see change, which config keys or output files moved, and whether a new experiment id or golden re-record is needed.
