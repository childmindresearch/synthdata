# End-to-end integration tests

These tests run the real pipeline CLIs (`synthdata-impute`, `-generate`, `-evaluate`, `-plot`) on a small committed dataset and check what they write. Unlike `tests/unit`, nothing is mocked.

## The fixture

`fixtures/clinic.csv` is a fully synthetic clinical-style table: 160 patients, 200 rows (a quarter of patients have a second visit), with numeric, binary, nominal and ordinal columns and a binary `target` drawn from a known logistic model. Some features have values missing completely at random; the target is never missing. `fixtures/make_fixture.py` regenerates it, and `fixtures/clinic_schema.csv` is its variable schema.

`fixtures/clinic_config.yaml` is the pipeline config. The harness (`conftest.py`) fills in `{root}` with a per-run scratch folder, so tests never touch `data/` or `output/` and need no network access.

## What runs where

The suite needs six independent pipeline runs (`RUNS` in `conftest.py`). They start together on a thread pool when the session begins, each stage in its own subprocess limited to one math thread, so the suite takes about as long as the CPU work divided by the cores rather than the sum of the runs.

| Run | Stages | Used by |
| --- | --- | --- |
| `baseline` | the pipeline defaults where CI can run them: impute (miceforest), generate (Bayesian network, CTGAN, both with HPO, class quotas), evaluate (holdout TSTR with kappa, fairness gaps), plot | `test_pipeline_end_to_end.py`, reproducibility and canary comparisons |
| `rerun` | the same config again, no plots | same-seed reproducibility and cache reuse |
| `holdout_canary` | impute and generate on a copy whose held-out rows were shifted | test rows never reach imputation, HPO or generation |
| `indicator_only` | median/mode imputation, mostly-missing columns kept as indicators, ARF with HPO | CART fill of recorded rows, ARF's fixed `delta` |
| `cart_canary` | `indicator_only` on shifted held-out rows | test rows never reach the CART fill or ARF |
| `imputed` | imputation only, with MissForest | `test_metric_known_answers.py`, which scores planted datasets in-process; MissForest's fill |

| File | Marker | What it checks |
| --- | --- | --- |
| `test_pipeline_end_to_end.py` | `integration` | Three-way patient split, the three CPU imputers (miceforest fills with observed train values), missing indicators, generated data, class quotas, HPO (TSTR objective, constraint screens, searched epochs, pruner reports), every metric family in range (SynthCity, SynthEval, class metrics, Anonymeter, DCR/NNDR, holdout fairness gaps, TSTR kappa), ranks, the seven report sections and their links, plots, no stray files |
| `test_reproducibility_and_leakage.py` | `integration` | A second identical run gives identical data and metrics; generation reuses its cache; held-out rows don't leak into imputation, tuning or generation |
| `test_metric_known_answers.py` | `integration` | Planted "copy of train" and "column-shuffled" datasets are scored in the right order; SynthEval role metrics are skipped when no sensitive or protected columns are declared |
| `test_golden_metrics.py` | `integration`, `regression` | Every raw metric of the `baseline` run against `fixtures/golden/baseline_run_<platform>_<device>.csv`, within tolerances (tight for the baselines, loose for trained generators). Torch training differs between CPU and CUDA and between operating systems, so each platform keeps its own reference; without one the test is skipped. Linux with CUDA is the canonical reference, Linux CPU a secondary one for machines without a GPU. Opt-in: run it after a dependency or submodule bump; when a change is intended, rewrite the reference with `SYNTHDATA_UPDATE_GOLDEN=1` and commit it |
| `test_full_generators.py` | `integration`, `slow` | Every SynthCity model, TabPFN and TabPFGen, with HPO (GPU recommended) |
| `test_sidecar_generators.py` | `integration`, `slow` | TabularARGN and CDTD (on by default, each in its own environment) in one run: class quotas, evaluation and ranking; CDTD's wrapper against the authors' notebook recipe. Skipped without the environments, failed instead with `SYNTHDATA_REQUIRE_SIDECARS=1` |

On a 4-core machine the non-slow suite takes about 5 minutes (16 before the runs went parallel).

`scripts/scheduled_tests.sh` runs what CI leaves out on a Linux GPU box (augur or falco): it syncs the main and sidecar environments, then runs `test_sidecar_generators.py` with `SYNTHDATA_REQUIRE_SIDECARS=1` and the `regression` golden check. The script's header has a weekly cron line.

```bash
# What CI runs (needs catboost for RefiDiff imputation):
uv run --with catboost==1.2.10 pytest tests/integration -m "integration and not slow and not regression"

# Golden run (opt-in): did a dependency bump move any metric?
uv run --with catboost==1.2.10 pytest tests/integration -m regression

# Full-generator run on a GPU machine (needs the tabpfn extra and a TabPFN token in .env):
uv sync --extra tabpfn
uv run --with catboost==1.2.10 pytest tests/integration -m slow
```

## Known bugs pinned as strict xfails

A test marked `xfail(strict=True)` describes behaviour the pipeline should have but doesn't yet. It fails today. When the bug is fixed the test passes, `strict=True` turns that into a failure, and the marker must be removed so the test guards the fix from then on. Each marker's `reason` names the bug.
