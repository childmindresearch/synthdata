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

### Audit a configuration

Run `synthdata-test` to check a configuration before running pipeline stages:

```bash
uv run synthdata-test --config path/to/your-config.yaml
uv run synthdata-test --config configs/config_hepatitis.yaml
```

The audit validates configuration and semantic policies, then loads the
configured dataset and variable schema by default. It checks the configured
data, schema, declarations, and split using the normal dataset loader. This
check can write processed dataset and split artifacts plus related metadata
under the configured data directory. If the config points to a remote source
such as UCI and its cache is absent, the audit may fetch and cache that data.

An audit passing means only that the configuration and configured data/schema
passed these checks. It does not run imputation, generation, evaluation,
plotting, or hosted inference, and does not establish that those stages will
succeed.

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

New dataset profiles must define canonical `data.split` roles `train`,
`tuning`, and `final_holdout`. Roles are patient-disjoint: no patient may
occur in more than one role, including through repeated encounters. `train`
fits candidate learned state, `tuning` drives HPO and candidate selection,
and `final_holdout` is reserved for post-selection evidence. Direct patient
identity is required directly through `data.patient_id_column`, used for
disjoint assignment, and removed from model
frames. Raw identity remains only in local role-assignment metadata; it is
never a feature or release-form evaluation field.

Patient-group tokenization uses a local HMAC key. Without
`SYNTHDATA_PATIENT_ID_HMAC_KEY`, the loader creates and reuses
`<data.data_dir>/.patient_id_hmac_key` (outside versioned `data_v_*` directories)
with owner-only permissions where supported. CI and production should provision
the non-empty environment variable; it overrides the local file and is never
overwritten. Manifests record only a SHA-256 key fingerprint. If that
fingerprint changes, loading fails closed; restore the original key or perform
an explicit identity rotation. Losing the local key likewise requires recovery
or deliberate rotation. Keys are random rather than derived from dataset
contents, so patient namespaces remain secret and cannot be reconstructed from
published data; reproducibility requires retaining the key or providing the
same external secret.

Dataset roles are separate: **patient ID** identifies a person across
encounters; **target** is the outcome evaluated for utility/fairness;
**quasi-identifiers (QIs)** are explicitly declared linkage/attacker fields;
**sensitive attributes** are fields whose disclosure is measured by privacy
attacks; and **protected attributes** define fairness subgroups for
representation, EO, and log-disparity evidence. Sensitive attributes are not
automatically QIs or protected attributes, and protected attributes are not
inferred from sensitive fields. Patient ID is none of these model roles.

### Role overlap and threat-model boundaries

Role declarations are explicit and are not inferred from one another. Use this
matrix when reviewing a profile:

| Role | Meaning | May overlap | Must not overlap |
| --- | --- | --- | --- |
| Protected fields | Fairness groups for representation, EO, and disparity evidence | QIs or sensitive fields, when explicitly declared | — |
| QIs | Attacker-observable predictors and release equivalence-class fields | Protected fields | Sensitive fields, target, or patient ID |
| Sensitive fields | Disclosure/attribute-inference targets | Protected fields | QIs or target |
| Target | Utility/fairness outcome | Protected fields may also be used as fairness groups | — |
| Patient ID | Identity used for patient grouping and split assignment | — | QIs, sensitive fields, target, and protected fields |

Attribute inference uses only declared QIs as predictors and sensitive fields as
targets. Membership inference, re-identification, structural privacy screens,
and other attacks retain their own distinct threat models; their inputs and
claims must not be described as attribute inference by default.

Historical `train`/`test` artifacts are supported only when the profile sets
`data.legacy_two_role: true`. They remain readable for compatibility, but
evaluation writes an explicit blocked result without candidate metric
evaluation or ranking; they cannot produce new HPO, final-policy, or release
claims.

SynthCity attribute-inference attacks require an explicit quasi-identifier
panel in `data.quasi_identifier_columns`; an evaluation-level
`evaluation.synthcity.quasi_identifier_columns` value, when supplied, must
match it. Sensitive target kinds come from the dataset variable schema, and
`evaluation.synthcity.sensitive_target_types` may only repeat matching schema
values. Leaving the Dataset QI panel empty records an explicit failed attack
result instead of silently treating every remaining feature as attacker input.
Set `evaluation.synthcity.classification_score` to
`balanced_accuracy` (the default) or `macro_f1`; the selected score drives the
baseline-adjusted categorical attack value while both diagnostics are retained.

The SynthCity structural privacy screens (`k-anonymization`, `distinct
l-diversity`, `k-map`, and `delta-presence`) are configurable KMeans proxy
screens, not formal privacy guarantees or formal release-form k/l metrics.
Their calibration is tied to
`evaluation.synthcity.structural_n_clusters`,
`evaluation.synthcity.structural_min_rows_per_cluster`, the selected sensitive
features, the DataLoader representation, sample sizes, and the random seed.
Changing those inputs requires fresh calibration; prior thresholds do not
carry over automatically.

## Pipeline behavior

The four commands form an ordered pipeline:

1. **Impute** missing feature values with fixed HyperImpute plugins. Canonical imputation is HyperImpute-only: candidate state fits `train`, while final state fits `train` plus `tuning`; transforms never fit on `tuning` or `final_holdout`. Canonical TabImpute and RefiDiff execution paths are blocked/deferred because they cannot honor role isolation. Retained benchmark/legacy code is not a canonical release path.
2. **Generate** candidate synthetic datasets with configured SynthCity, TabPFN, and TabPFGen models; optional Optuna hyperparameter searches are persisted and resumable. Each HPO context is stored in a digest-versioned sidecar with a latest pointer, so changing the objective creates a new study identity without overwriting prior context evidence.
3. **Evaluate** candidates for utility, privacy, and fairness. Candidate ranking uses only `train` plus `tuning`; after selection, the chosen generator is refit on those two roles and evaluated once against `final_holdout`. Evaluation can process models in parallel within configured resource limits and writes a ranked table, report, and diagnostics.
4. **Plot** recorded data-quality, generation, HPO, and evaluation artifacts without rerunning earlier stages.

Artifacts are namespaced by dataset name, dataset version, and experiment ID. Their folder names make both levels explicit: for example, `data_v_1.2/exp_v_0.4/`. Generation creates a new experiment by default; evaluation and plotting use the latest one or accept `--experiment-id` to revisit a prior run. This preserves cached inputs, model outputs, HPO state, metrics, and figures across dataset revisions.

### Imputation artifacts and logs

`synthdata-impute` writes its validation table to:

```text
output/<config.name>/imputation/data_v_<version>/imputation_validation_report.csv
```

For a config without `data.version`, `<version>` is `unversioned`, so the
report is written under `output/<config.name>/imputation/data_v_unversioned/`.
The imputed-data cache is separate from this report: raw, imputed, and split
CSV artifacts are stored under `<data.data_dir>/data_v_<version>/` (or
`<data.data_dir>/data_v_unversioned/`), alongside its cache-lineage metadata.
The CLI logs both locations rather than printing validation-table contents to
the terminal.

The canonical candidate validation report covers only roles transformed for
candidate use: `train` and `tuning`. It does not represent imputation of the
final holdout. Each row includes `datatype` from the variable schema's
`kind`, observed and imputed cardinality (`obs_cardinality` and
`imp_cardinality`), and type-specific summaries. Continuous columns include
observed and imputed mean/std (`obs_mean`, `obs_std`, `imp_mean`, and
`imp_std`); these statistics are not meaningful for categorical values and
remain empty for them. Categorical columns include observed and imputed modes
(`obs_mode` and `imp_mode`).

Evaluation always retains raw metric values for audit. The ranked columns are
derived separately and include only complete, decision-eligible policy metrics;
failed, diagnostic, calibration-only, and blocked results remain visible in
the raw table or validation sidecars. Each evaluation bundle also records
`evaluation_artifacts-v1/metric_contract_manifest.json`,
`evaluation_artifacts-v1/synthcity_metric_status.json`,
`evaluation_artifacts-v1/syntheval_metric_status.json`, and
`evaluation_artifacts-v1/syntheval_execution.json`,
`evaluation_artifacts-v1/custom_metric_status.json`, and
`evaluation_artifacts-v1/final_holdout_evidence.json` when those framework
results are present. The final-holdout sidecar is written after candidate
selection, points to the selected model's refit artifact, and is never merged
into `combined_evaluation.csv`. Its refit metadata records `fit_roles` as
`["train", "tuning"]`; successful evidence verifies both the synthetic CSV
and its cache metadata by SHA-256. The SynthEval execution sidecar retains terminal
per-method status plus legacy and versioned normalized rows. All are
referenced and integrity-hashed by
`evaluation_artifacts-v1/manifest.json`. The sidecars contain per-model metric
states, role fingerprints, source metadata, and the contract digest used for
that run. The experiment manifest also records a separate
`final_holdout_evidence` stage. Legacy two-role datasets write an explicit
blocked final-evidence record instead of making a release claim.

Evaluation defaults to row-level populations:

```yaml
evaluation:
  group_mode: row
  group_column: null
```

Set `group_mode: patient_group` and provide a non-empty `group_column` when
rows represent repeated observations of the same patient or entity. The
identifier is required for role assignment and provenance, but is excluded
from every model and release metric frame. Real `train`, `tuning`, and
`final_holdout` populations are patient-disjoint. Patient-disjoint roles do
not make encounter records independent: privacy metrics cannot claim
protection against linkage or inference among multiple encounters from one
patient beyond this role isolation.

## Release populations, privacy, and fairness evidence

Candidate utility compares release-form synthetic data with `tuning`. Final
audit utility compares selected release-form synthetic data with
`final_holdout`; it does not reuse candidate scores. MIA members are
`train+tuning`, and non-members are patient-disjoint `final_holdout`.
Attribute-disclosure attackers fit on synthetic QIs and score `final_holdout`.
Representation, EO, and worst log disparity use final evidence. Invalid
required evidence is indeterminate, not silently substituted or reweighted.

Synthetic categorical values outside source support are not silently coerced.
The affected evaluation records failed or indeterminate evidence, as
appropriate to whether execution errored or required evidence is incomplete;
it never records that evidence as succeeded. Evidence states have strict
meaning: `succeeded` requires every required table; `failed` means execution
or an evaluation error; `indeterminate` means required evidence is insufficient
or incomplete. Only `succeeded` evidence contributes metric values, plots, or
policy ranking. Failed and indeterminate records remain visible for audit.

Formal release-form privacy metrics use transformed QIs and sensitive fields:
`S_k` and `S_l` are formal k-anonymity and l-diversity scores, with `S_DCR`
and `S_epsilon` as separate formal release components. SynthCity's
`k-anonymization`, `l-diversity`, `k-map`, and `delta-presence` are blocked
KMeans proxy metrics, not formal k/l evidence and not release authorization.

## Fixed utility, normalization, and final audit score

HPO utility is fixed:

```text
U_tuning = (S_TSTR + S_MMD + S_JSD) / 3
```

Final audit uses:

```text
U = (S_TSTR + S_MMD + S_JSD) / 3
I = min(S_k, S_l, S_DCR, S_epsilon)
P = (I * S_MIA * S_attribute)^(1/3)
F = 0.40*S_representation + 0.40*S_EO + 0.20*S_worst_log_disparity
R_final = 0.45*U + 0.30*P + 0.25*F
```

Direct normalized components are not double-normalized. Raw anchored metrics
use fixed clipped anchors. Invalid required evidence makes its dimension and
`R_final` indeterminate. `R_final` is audit-only; post-holdout reranking and
release authorization from this score are prohibited.

The shipped privacy thresholds are calibration-only and disabled by default.
Enabling `evaluation.privacy_gate` requires each threshold to name an exact
operational `contract_id`; calibration-only and audit-only metrics remain
visible as evidence but cannot authorize a release recommendation.
The evaluation sidecars record group counts and role-scoped fingerprints, not
raw identifiers. Until a metric has an explicit `group_safe` contract,
patient-group evaluation keeps its result for audit but marks it
`group_unsafe` and excludes it from policy ranking; this bridge does not claim
group-aware aggregation yet.

The generic HPO objective is limited to the approved utility set in
`HPOConfig.metric_config` (`wasserstein_dist`, `inv_kl_divergence`, nearest
synthetic-neighbor distance, and XGB utility). Privacy, diagnostic, and
calibration-only metrics fail closed at HPO setup rather than becoming tuning
objectives by default.

Before those objective metrics run, every HPO candidate passes the configured
`generation.hpo.stage_a` screens for shape/schema, source-derived numeric and
categorical validity, target/protected-group support, deterministic dependency
rules, exact row reuse (zero tolerance), and subgroup collapse. Failed screens
are persisted as per-trial prune results under the generation experiment and
cannot become partial objective evidence. No privacy or fairness attack is
invoked by Stage A.

If every trial for an HPO variant is rejected by these Stage A screens, only
that variant's output is skipped. No fallback parameters or synthetic output
are fabricated; other generation outputs can still be produced. The existing
experiment manifest records generation as `partial`, lists expected,
produced, and failed outputs, and references the persisted Stage A evidence.
This manifest is the authoritative source for status; evaluation accepts a
missing variant only when the manifest explains it. Evaluation processes
available requested outputs and records partial coverage rather than
inventing metrics for the missing output. Partial reports and ranking plots are
labeled as such: their rankings compare only evaluated models and are not a
complete comparison of all requested outputs.

> [!TIP]
> See [`synthdata/config.py`](synthdata/config.py) for cache, device, model, HPO, parallel evaluation, artifact, experiment, metric, ranking, privacy-gate, and plot options.

## Optional: RefiDiff masked-cell benchmark

The retained `synthdata-imputation-benchmark` and RefiDiff material is
exploratory/legacy only. RefiDiff and TabImpute paths are blocked or deferred
for canonical role-isolated execution and cannot support canonical release
claims. Do not use benchmark results to select canonical production
imputation.

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
