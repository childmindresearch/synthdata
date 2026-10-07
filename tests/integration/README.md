# End-to-end integration tests

These tests run the real pipeline CLIs (`synthdata-impute`, `-generate`, `-evaluate`, `-plot`) on a small committed dataset and check what they write. Unlike `tests/unit`, nothing is mocked.

## The fixture

`fixtures/clinic.csv` is a fully synthetic clinical-style table: 160 patients, 200 rows (a quarter of patients have a second visit), with numeric, binary, nominal and ordinal columns and a binary `target` drawn from a known logistic model. Some features have values missing completely at random; the target is never missing. `fixtures/make_fixture.py` regenerates it, and `fixtures/clinic_schema.csv` is its variable schema.

`fixtures/clinic_config.yaml` is the pipeline config. The harness (`conftest.py`) fills in `{root}` with a per-run scratch folder, so tests never touch `data/` or `output/` and need no network access.

## What runs where

| File | Marker | What it checks | Time on a CPU runner |
| --- | --- | --- | --- |
| `test_pipeline_end_to_end.py` | `integration` | One full run: split, imputation, generated data, HPO, metric ranges, ranks, report, plots, no stray files | ~4 min (shared run) |
| `test_reproducibility_and_leakage.py` | `integration` | A second identical run gives identical data and metrics; generation reuses its cache; held-out rows don't leak into imputation | +4 min |
| `test_metric_known_answers.py` | `integration` | Planted "copy of train" and "column-shuffled" datasets are scored in the right order | +1–2 min |
| `test_full_generators.py` | `integration`, `slow` | Every SynthCity model, TabPFN and TabPFGen, with HPO | GPU recommended |

```bash
# What CI runs (needs catboost for RefiDiff imputation):
uv run --with catboost pytest tests/integration -m "integration and not slow"

# Full-generator run on a GPU machine (needs the tabpfn extra and a TabPFN token in .env):
uv sync --extra tabpfn
uv run --with catboost pytest tests/integration -m slow
```

## Known bugs pinned as strict xfails

A test marked `xfail(strict=True)` describes behaviour the pipeline should have but doesn't yet. It fails today. When the bug is fixed the test passes, `strict=True` turns that into a failure, and the marker must be removed so the test guards the fix from then on. Each marker's `reason` names the bug.
