# End-to-end integration tests

These tests run the real pipeline CLIs (`synthdata-impute`, `-generate`, `-evaluate`, `-plot`) on a small committed dataset and check what they write. Unlike `tests/unit`, nothing is mocked.

## The fixture

`fixtures/clinic.csv` is a fully synthetic clinical-style table: 160 patients, 200 rows (a quarter of patients have a second visit), with numeric, binary, nominal and ordinal columns and a binary `target` drawn from a known logistic model. Some features have values missing completely at random; the target is never missing. `fixtures/make_fixture.py` regenerates it, and `fixtures/clinic_schema.csv` is its variable schema.

`fixtures/clinic_config.yaml` is the pipeline config. The harness (`conftest.py`) fills in `{root}` with a per-run scratch folder, so tests never touch `data/` or `output/` and need no network access.

## What runs where

The suite needs six independent pipeline runs (`RUNS` in `conftest.py`). They start together on a thread pool when the session begins, each stage in its own subprocess limited to one math thread, so the suite takes about as long as the CPU work divided by the cores rather than the sum of the runs.

| Run | Stages | Used by |
| --- | --- | --- |
| `baseline` | impute (MissForest), generate (Bayesian network, CTGAN, both with HPO), evaluate, plot | `test_pipeline_end_to_end.py`, reproducibility and canary comparisons |
| `rerun` | the same config again, no plots | same-seed reproducibility and cache reuse |
| `holdout_canary` | impute and generate on a copy whose held-out rows were shifted | test rows never reach imputation, HPO or generation |
| `indicator_only` | median/mode imputation, mostly-missing columns kept as indicators, ARF with HPO | CART fill of recorded rows, ARF's fixed `delta` |
| `cart_canary` | `indicator_only` on shifted held-out rows | test rows never reach the CART fill or ARF |
| `imputed` | imputation only | `test_metric_known_answers.py`, which scores planted datasets in-process |

| File | Marker | What it checks |
| --- | --- | --- |
| `test_pipeline_end_to_end.py` | `integration` | Three-way patient split, both imputers, missing indicators, generated data, HPO (TSTR objective, constraint screens, searched epochs, pruner reports), every metric family in range (SynthCity, SynthEval, class metrics, Anonymeter, DCR/NNDR), ranks, the seven report sections and their links, plots, no stray files |
| `test_reproducibility_and_leakage.py` | `integration` | A second identical run gives identical data and metrics; generation reuses its cache; held-out rows don't leak into imputation, tuning or generation |
| `test_metric_known_answers.py` | `integration` | Planted "copy of train" and "column-shuffled" datasets are scored in the right order; SynthEval role metrics are skipped when no sensitive or protected columns are declared |
| `test_full_generators.py` | `integration`, `slow` | Every SynthCity model, TabPFN and TabPFGen, with HPO (GPU recommended) |

On a 4-core machine the non-slow suite takes about 6 minutes (16 before the runs went parallel).

```bash
# What CI runs (needs catboost for RefiDiff imputation):
uv run --with catboost==1.2.10 pytest tests/integration -m "integration and not slow"

# Full-generator run on a GPU machine (needs the tabpfn extra and a TabPFN token in .env):
uv sync --extra tabpfn
uv run --with catboost==1.2.10 pytest tests/integration -m slow
```

## Known bugs pinned as strict xfails

A test marked `xfail(strict=True)` describes behaviour the pipeline should have but doesn't yet. It fails today. When the bug is fixed the test passes, `strict=True` turns that into a failure, and the marker must be removed so the test guards the fix from then on. Each marker's `reason` names the bug.
