"""Config schema and YAML loader for the synthdata pipeline.

This module is the reference for every configuration option: each field below
carries its default and an explanation. A run is driven by one YAML file (see
``configs/``) whose sections mirror these dataclasses (``data``, ``imputation``,
``generation``, ``evaluation``, ``plots``, ``experiment``). The YAML files list
only the options most worth reviewing; anything left out takes the default
defined here. Unknown keys fail loudly, so a typo never falls back silently.
All four CLI commands (``synthdata-impute``, ``-generate``, ``-evaluate``,
``-plot``) load the same :class:`Config` via :func:`load_config`.

Relative paths in the config are resolved against the current working directory
at the time the scripts are invoked (i.e. run commands from the repository root,
or pass absolute paths).
"""

import dataclasses
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataReadinessConfig:
    """Thresholds of the data-readiness check (synthdata.data_readiness).

    The check runs on the train split at the start of ``synthdata-impute`` and
    only advises: it writes ``data_readiness.csv`` and logs warnings, it never
    drops or changes a column. Each threshold's source is in
    docs/verification.md (Data readiness); null turns that check off.
    """

    #: Run the check.
    enabled: bool = True
    #: Near-zero variance (caret ``nearZeroVar``): the most common value is at
    #: more than this many times as frequent as the second (caret's 95/5)...
    near_zero_freq_ratio: float | None = 95 / 5
    #: ...and at most this percent of the values are distinct (caret's 10).
    near_zero_unique_percent: float = 10.0
    #: Two continuous columns with abs(Spearman) at or above this are flagged
    #: as near-duplicates (caret ``findCorrelation`` uses 0.90; stricter here
    #: so only near-redundant pairs are flagged).
    max_abs_correlation: float | None = 0.95
    #: Columns missing in at least this share of the train rows.
    max_missing_fraction: float | None = 0.5
    #: Categories with fewer train rows than this (the "5 expected counts"
    #: rule, also the floor of the HPO coverage screens).
    min_category_rows: int | None = 5
    #: Nominal columns with more levels than this.
    max_levels: int | None = 50
    #: Continuous columns with abs(skewness) above this (West, Finch & Curran).
    max_abs_skew: float | None = 2.0
    #: Continuous columns with at least this share of zeros.
    max_zero_fraction: float | None = 0.5
    #: Fewer train rows per modelled column than this.
    min_rows_per_column: float | None = 10.0
    #: Target classes and protected groups with fewer rows than this in the
    #: tuning or test split.
    min_group_rows: int | None = 5


@dataclasses.dataclass
class DataConfig:
    """Where the raw dataset comes from and how columns should be interpreted."""

    #: "uci" to fetch+cache from the UCI ML repository, or "csv"/"parquet" for a
    #: local file (the actual reader used is auto-detected from ``path``'s file
    #: extension -- ".csv" vs. ".parquet"/".pq" -- regardless of which of the two
    #: local values is set here, so either works so long as it matches ``path``).
    source: str = "uci"
    #: UCI dataset id (only used when source == "uci").
    uci_id: int | None = None
    #: Path to a local CSV or Parquet file (only used when source == "csv"/"parquet").
    path: str | None = None

    #: Freeform dataset version label (e.g. "v1", "2024-06-01"). If set, cached
    #: raw/imputed/split CSVs are nested under `data_dir/data_v_<version>/` and every
    #: experiment manifest records which version was used, so results stay
    #: traceable when the underlying dataset changes over time.
    version: str | None = None

    #: Name of the outcome/label column.
    target_column: str = "target"
    #: If the source data uses a different name for the target column, set this
    #: to have it renamed to `target_column` on load (e.g. UCI's "CLASS" -> "target").
    raw_target_column: str | None = None
    #: Column roles in statistical disclosure control terms. Quasi-identifiers
    #: are public, linkable attributes an attacker may already know (age,
    #: sex, region). Sensitive columns are the secrets an attacker tries to
    #: infer (a diagnosis, income); privacy metrics that infer attributes
    #: target these. Protected columns define the groups fairness metrics
    #: compare. A column may be both a quasi-identifier and protected, or
    #: sensitive and protected, but never a quasi-identifier and sensitive.
    #: Set protected_columns (even to []) whenever sensitive_columns is set,
    #: so a config written for the old meaning of sensitive_columns (the
    #: fairness groups) fails instead of silently changing meaning.
    quasi_identifier_columns: list = dataclasses.field(default_factory=list)
    #: Secret attributes privacy metrics try to infer (see above).
    sensitive_columns: list = dataclasses.field(default_factory=list)
    #: Fairness groups (see above); also the default for
    #: ``evaluation.log_disparity.protected_columns``. SynthEval's gap
    #: metrics compare each group holding at least 5% of the rows with the
    #: rest. A numeric column (e.g. age, birth year) needs its groups set in
    #: ``protected_bins``; without them it is left out of those metrics.
    protected_columns: list | None = None
    #: Bin edges that turn a numeric protected column into groups,
    #: ``{column: [edges]}`` in ascending order, e.g.
    #: ``{age: [0, 18, 30, 45, 60, 200]}``. Bins include their upper edge
    #: (the first also its lower edge); values outside every bin belong to
    #: no group. Used by SynthEval's fairness metrics, the HPO protected-group
    #: screens (which fall back to train quartiles without bins) and, unless
    #: ``evaluation.log_disparity.protected_bins`` is set, by log disparity.
    protected_bins: dict = dataclasses.field(default_factory=dict)
    #: Columns to drop entirely before any modeling (e.g. free-text/ID columns).
    drop_columns: list = dataclasses.field(default_factory=list)
    #: Column identifying the patient (or other independent unit) each row
    #: belongs to. When set, the train/test split assigns whole patients, so
    #: repeat encounters of one patient never end up on both sides, and the
    #: column is then removed so no model ever sees it. Leave unset only when
    #: every row is a different patient.
    patient_id_column: str | None = None
    #: Drop rows where target_column is null before splitting/imputing. Every
    #: downstream stage assumes a fully-observed target (imputation only fills
    #: feature_columns; the target is passed through as-is), so datasets whose
    #: label is only sometimes assessed (e.g. an optional clinical scale) need
    #: this set to True -- otherwise stratified train_test_split raises on NaN.
    drop_rows_missing_target: bool = False

    #: Path to a CSV that explicitly defines how every retained feature and the
    #: target are modeled. It must contain ``column`` and ``kind`` columns;
    #: ``kind`` is ``categorical`` or ``continuous``. An optional
    #: ``ordinal_order`` uses square brackets to give the lowest-to-highest
    #: order for an ordinal categorical variable; a blank value means nominal.
    #:
    #: The schema is intentionally mandatory for new datasets. The loader validates exact
    #: coverage after source cleanup and fails on missing, duplicate, or stale declarations.
    variable_schema_path: str | None = None

    #: Transitional compatibility for existing configurations. New configs must
    #: use ``variable_schema_path``; these fields are only used if no schema path
    #: is supplied. ``"auto"`` is no longer accepted, so heuristics are never a
    #: default data-typing policy. Legacy support will be removed in the next
    #: breaking schema release.
    #: Legacy: categorical columns without an order.
    nominal_columns: list | None = None
    #: Legacy: categorical columns with an order.
    ordinal_columns: list = dataclasses.field(default_factory=list)
    #: Legacy: ``{column: [lowest, ..., highest]}`` category order per ordinal column.
    ordinal_column_categories: dict = dataclasses.field(default_factory=dict)

    #: Uppercase all column names on load (matches the hepatitis notebook convention).
    uppercase_columns: bool = False
    #: Dataset-specific quirk: remap columns whose only non-null values are {1, 2} to {0, 1}.
    remap_binary_one_two: bool = False

    #: If set (together with a non-empty ``outlier_columns``), numeric values in
    #: those columns further than this many std-devs from their column mean are
    #: treated as missing (NaN) rather than passed through as-is. Catches both
    #: "not administered" sentinel codes (e.g. a lone 999 among otherwise 0-30
    #: values) and corrupt outlier rows (e.g. a derived metric blown up by a
    #: division artifact), either of which can otherwise cause float32 overflow
    #: inside TabPFN/TabImpute. None (default) disables this check entirely.
    outlier_zscore_threshold: float | None = None
    #: Explicit list of columns to apply ``outlier_zscore_threshold`` to (no
    #: effect if that's None). Deliberately opt-in per-column rather than
    #: "all numeric columns": a blanket z-score check false-positives heavily
    #: on zero-/mode-inflated ordinal/Likert-style columns common in survey
    #: data (e.g. a 0-3 severity scale where 0 is the overwhelming majority --
    #: confirmed empirically, legitimate 2s/3s got flagged as "outliers" with
    #: z-scores >10), so only list columns confirmed to have genuine
    #: sentinel/corrupted values, not just a skewed distribution.
    outlier_columns: list = dataclasses.field(default_factory=list)

    #: Data-readiness check run before imputation (see DataReadinessConfig).
    readiness: DataReadinessConfig = dataclasses.field(default_factory=DataReadinessConfig)

    #: Patient-level three-way split, as approximate shares of rows: whole
    #: patients are assigned while the folds balance encounter counts. ``train`` fits the imputer and the HPO
    #: candidates, ``tuning`` scores the candidates, and ``holdout`` is touched
    #: only by evaluation. Final models are refitted on train + tuning. The
    #: three shares must sum to 1 and be multiples of 1/k for some k <= 20,
    #: because the split takes whole folds of
    #: :class:`sklearn.model_selection.StratifiedGroupKFold` with k folds
    #: (0.6/0.2/0.2 uses 5 folds). ``tuning_fraction`` may be 0 without HPO.
    train_fraction: float = 0.6
    tuning_fraction: float = 0.2
    holdout_fraction: float = 0.2
    #: Columns whose joint distribution the split keeps balanced, most
    #: important first (put the target first). None stratifies on the target
    #: alone; an empty list turns stratification off. Missing values form
    #: their own stratum. A joint stratum with fewer patients than folds is
    #: merged into the stratum of the first column alone.
    stratify_columns: list | None = None
    #: Optional bin edges per ``stratify_columns`` entry (null for none), e.g.
    #: ``[null, null, [30, 60]]`` bins age into <30, 30-60 and >=60.
    stratify_bins: list | None = None

    #: Where cached/derived CSVs (raw, imputed, train/test splits) are written.
    data_dir: str = "data/dataset"
    #: Subfolder of ``data_dir`` where a UCI download is cached.
    raw_cache_subdir: str = "raw"


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RefiDiffConfig:
    """Hyperparameters for the RefiDiff imputation backend (arXiv:2505.14451).

    Only used when ``ImputationConfig.method == "refidiff"``. Requires the
    `refidiff` extra (`uv sync --extra refidiff`); see
    synthdata/imputation/refidiff_backend.py for the ported algorithm.
    """

    #: Denoiser hidden width (diamond up/down-sampling network width).
    hidden_dim: int = 32
    #: Max training epochs (early stopping usually halts well before this).
    epochs: int = 10001
    #: Stop training if val loss hasn't improved for this many epochs.
    early_stopping_patience: int = 500
    #: Training mini-batch size (rows).
    batch_size: int = 8192
    #: Number of reverse-diffusion (EDM/VE-SDE) sampling steps.
    num_steps: int = 50
    #: Number of independent reverse-diffusion trajectories averaged together.
    num_trials: int = 10
    #: "auto" (use mamba-ssm if importable, else fall back to the MLP
    #: denoiser), "mamba" (require mamba-ssm, error if unavailable), or "mlp"
    #: (always use the plain residual-MLP denoiser, e.g. for CPU-only runs).
    denoiser: str = "auto"
    #: Save a training checkpoint every N epochs so an interrupted run
    #: (shared-GPU preemption/OOM) can resume instead of retraining from
    #: scratch.
    checkpoint_every: int = 1000
    #: Number of CatBoost boosting rounds used during each categorical
    #: warm-up/polishing refinement fit. ``100`` is the established practical
    #: LORIS default; the upstream RefiDiff reference uses CatBoost's default
    #: budget (normally 1000), which should be selected explicitly for a
    #: reproduction profile.
    catboost_warmup_iterations: int = 100
    #: How binary categorical codes that do not map to an observed category
    #: are repaired. ``clip`` preserves the historical local port behavior;
    #: ``nearest_valid`` projects to the valid binary code with minimum Hamming
    #: distance (ties resolve to the lower category index); ``error`` aborts
    #: rather than silently repairing, for strict diagnostic comparisons.
    categorical_decode_policy: str = "clip"


IMPUTATION_METHODS = ("missforest", "simple", "tabimpute", "refidiff")


@dataclasses.dataclass
class MissForestConfig:
    """Settings for ``imputation.method: missforest`` (scikit-learn ``IterativeImputer``).

    Stekhoven & Buehlmann's MissForest uses 100 trees and at most 10 rounds;
    the smaller defaults keep a 20k x 400 table in minutes on a CPU, and the
    rounds stop early once imputations stop changing.
    """

    #: Trees per random forest.
    n_estimators: int = 50
    #: Maximum imputation rounds over all incomplete columns.
    max_iter: int = 5
    #: Predict each column from only this many most-correlated other columns
    #: (``None`` = all of them); lowers run time on very wide tables.
    n_nearest_features: int | None = None


@dataclasses.dataclass
class MissingIndicatorConfig:
    """Binary ``<column>__missing`` features so generators learn missingness.

    Indicators come from the raw data (1 where the value is missing), so both
    imputation phases see the same ones. Synthetic values whose indicator is 1
    are blanked again in the released copy of each synthetic dataset.
    """

    #: Add indicators. Turning this off makes generators see only imputed
    #: values, so synthetic data cannot reproduce the missingness pattern.
    enabled: bool = True
    #: Columns missing in at least this share of the train rows outside the
    #: tuning split get an indicator; rarer missingness gets none.
    min_missing_fraction: float = 0.05
    #: Write ``released/<model>.csv`` with values blanked where the synthetic
    #: indicator is 1 and the indicator columns dropped.
    remask_synthetic: bool = True
    #: Columns missing in at least this share of those rows keep only their
    #: indicator: their values are neither imputed nor generated, and the
    #: released copy fills "recorded" synthetic rows by CART leaf sampling
    #: fitted on observed train values (synthpop's two-step method).
    #: ``null`` turns this off, so every column is imputed and generated.
    indicator_only_fraction: float | None = 0.8
    #: Columns to keep imputing and generating whatever their missingness.
    #: The target, stratification and role columns are always kept.
    keep_values_columns: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RefiDiffBenchmarkHPOConfig:
    """Narrow, staged search space for masked-cell RefiDiff validation.

    Exploratory and not used by the main pipeline; each list is the set of
    values the search may pick for the matching :class:`RefiDiffConfig` field.
    """

    #: Run an Optuna search instead of one fixed-parameter benchmark.
    enabled: bool = False
    #: Trials per study.
    n_trials: int = 12
    #: Wall-clock limit per study in seconds (None = no limit).
    timeout_seconds: int | None = None
    hidden_dims: list = dataclasses.field(default_factory=lambda: [16, 32, 64])
    num_steps: list = dataclasses.field(default_factory=lambda: [10, 25, 50])
    num_trials: list = dataclasses.field(default_factory=lambda: [1, 3, 5])
    epochs: list = dataclasses.field(default_factory=lambda: [1000, 3000])
    early_stopping_patience: list = dataclasses.field(default_factory=lambda: [100, 250])


@dataclasses.dataclass
class RefiDiffBenchmarkConfig:
    """Append-only masked-cell validation for RefiDiff candidates.

    Benchmarking is deliberately separate from ordinary imputation caching:
    it creates artificial masks only in the training split and writes studies
    beneath ``output/<dataset>/imputation/data_v_<version>/benchmark_<study-id>/``.
    Exploratory and not validated end to end; run only through the separate
    ``synthdata-imputation-benchmark`` command, never by the main pipeline.
    """

    #: Must be True for ``synthdata-imputation-benchmark`` to run.
    enabled: bool = False
    #: Root folder for benchmark studies.
    output_dir: str = "output/dataset/imputation"
    #: Share of observed training cells in each scored column that is hidden.
    mask_fraction: float = 0.3
    #: Independent random masks per study (results are averaged over them).
    n_masks: int = 3
    #: Missingness mechanisms to simulate: "mcar", "mar" and/or "mnar".
    mechanisms: list = dataclasses.field(default_factory=lambda: ["mcar"])
    #: Optional feature columns eligible for artificial masking/scoring. All
    #: feature columns remain visible to the imputer as context. ``None`` uses
    #: every non-sensitive feature; a small explicit panel is appropriate for
    #: an affordable screening study on a very wide dataset.
    score_columns: list | None = None
    #: Optional search over RefiDiff settings (see above).
    hpo: RefiDiffBenchmarkHPOConfig = dataclasses.field(default_factory=RefiDiffBenchmarkHPOConfig)


@dataclasses.dataclass
class ImputationConfig:
    """How missing feature values are filled before generation.

    The target is never imputed. Imputers are fitted on training rows only,
    so no information from the tuning or holdout rows leaks into the fill.
    """

    #: Run imputation. When False, generators that cannot handle missing
    #: values will fail; keep it on unless the data has no missing values.
    enabled: bool = True
    #: "missforest" (default; iterative random-forest imputation), "simple"
    #: (median for continuous columns, most frequent value for categorical
    #: ones; fast, used by CI), "tabimpute" (TabPFN-based) or "refidiff"
    #: (predictive+diffusion hybrid -- see synthdata/imputation/refidiff_backend.py).
    #: Every method is fitted twice: on train minus tuning for HPO, then on
    #: all of train for the final models and the holdout.
    method: str = "missforest"
    #: "auto" | "cpu" | "cuda" | "mps"
    device: str = "auto"
    #: Optional per-column rounding precision (decimal places) applied post-imputation.
    round_rules: dict = dataclasses.field(default_factory=dict)
    #: If True (default, matches the hepatitis notebook), feature columns not listed in
    #: round_rules are rounded to the nearest integer after imputation. Set to False for
    #: datasets with genuinely continuous features that shouldn't be integer-snapped.
    round_to_int_default: bool = True
    #: Reuse previously cached imputed CSVs if present *and* still valid: validity
    #: is determined by comparing a hash of the resolved schema/config fields
    #: (categorical roles, ordinal orders, method, round_rules,
    #: round_to_int_default, refidiff params) and exact source/full/train/test
    #: fingerprints against the sidecar ``.imputation_cache_key.json`` written
    #: alongside the cached CSVs. Editing e.g. ``data.nominal_columns``/
    #: ``data.ordinal_columns`` or refreshing source/split membership therefore
    #: retrains instead of silently reusing stale imputed data (see
    #: synthdata.imputation.pipeline.run_imputation).
    cache: bool = True
    #: Fractional margin used when validating imputed continuous values fall within range.
    validation_margin: float = 0.2
    #: Warn when a column's imputed values differ between the two fits by more
    #: than this (see ``imputation_drift.csv``).
    drift_warn_threshold: float = 0.1
    #: Only used when method == "missforest".
    missforest: MissForestConfig = dataclasses.field(default_factory=MissForestConfig)
    #: ``<column>__missing`` indicator features (see MissingIndicatorConfig).
    missing_indicators: MissingIndicatorConfig = dataclasses.field(
        default_factory=MissingIndicatorConfig
    )
    #: Only used when method == "refidiff".
    refidiff: RefiDiffConfig = dataclasses.field(default_factory=RefiDiffConfig)
    #: Optional train-only artificial-masking benchmark/HPO for RefiDiff
    #: (exploratory; used only by ``synthdata-imputation-benchmark``).
    benchmark: RefiDiffBenchmarkConfig = dataclasses.field(default_factory=RefiDiffBenchmarkConfig)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SynthcityModelsConfig:
    """Generators from the synthcity library."""

    #: Train synthcity generators at all.
    enabled: bool = True
    #: synthcity plugin names to train: "ctgan", "tvae", "rtvae", "adsgan",
    #: "pategan" (differentially private GAN), "bayesian_network" and "ddpm".
    #: Bayesian networks and DDPM scale poorly to hundreds of columns.
    names: list = dataclasses.field(
        default_factory=lambda: [
            "ctgan",
            "tvae",
            "adsgan",
            "bayesian_network",
            "pategan",
            "rtvae",
            "ddpm",
            "arf",
        ]
    )


@dataclasses.dataclass
class TabPFNConfig:
    """TabPFN unsupervised generation (needs the ``tabpfn`` extra and a
    ``TABPFN_TOKEN`` in ``.env``)."""

    #: Train TabPFN generators at all.
    enabled: bool = True
    #: "standard" (features only, label assigned post-hoc) and/or
    #: "custom" (features + target modeled jointly).
    variants: list = dataclasses.field(default_factory=lambda: ["standard", "custom"])
    #: Which train split(s) to fit on: "raw" (original data, pre-imputation --
    #: TabPFN handles missing values natively) and/or "imputed" (same imputed
    #: split used by every other model). Include both to compare how TabPFN
    #: performs with vs. without imputation; imputed-variant outputs are
    #: cached as e.g. "tabpfn_standard_imputed" (raw keeps the unsuffixed name).
    data_variants: list = dataclasses.field(default_factory=lambda: ["raw"])
    #: TabPFN weights to load ("v2", "v2.5", "v2.6", "v3", "v3.5"). Pinned so a
    #: tabpfn upgrade cannot change the model silently. Each version has its
    #: own non-commercial license to accept on ux.priorlabs.ai (see the user
    #: manual). Also used by TabPFGen's classifier.
    model_version: str = "v3.5"


@dataclasses.dataclass
class TabPFGenConfig:
    """TabPFGen (energy-based sampling with TabPFN as the classifier)."""

    #: Off by default: TabPFGen 0.1.4's SGLD energy is the distance to the
    #: nearest training row, so its output is jittered copies of training rows
    #: (see docs/verification.md). Kept for a later swap to TabEBM.
    enabled: bool = False
    #: "standard" (TabPFGen defaults) and/or "custom" (SGLD + nearest-neighbor relabeling).
    variants: list = dataclasses.field(default_factory=lambda: ["standard", "custom"])
    #: kwargs passed to TabPFGen() for the non-HPO "standard" variant (empty = library defaults).
    standard_params: dict = dataclasses.field(default_factory=dict)
    #: kwargs passed to TabPFGenSGLDLabels() for the non-HPO "custom" variant.
    custom_params: dict = dataclasses.field(
        default_factory=lambda: {"n_sgld_steps": 1000, "sgld_noise_scale": 0.1}
    )


@dataclasses.dataclass
class TabularARGNConfig:
    """MOSTLY AI's TabularARGN (``mostlyai-engine``), run in its own environment.

    mostlyai-engine's torch and psutil pins clash with this repo's, so it is
    installed separately (``uv sync --project envs/tabularargn``) and run as a
    subprocess; see ``synthdata/generation/tabularargn_backend.py``. It fits on
    the imputed train split, like the synthcity models. No HPO: the library
    stops training early on its own validation split.
    """

    #: Off by default: needs the separate environment above.
    enabled: bool = False
    #: The environment's folder (relative paths are from the repo root).
    env_dir: str = "envs/tabularargn"
    #: Arguments for ``TabularARGN()`` (e.g. ``model: MOSTLY_AI/Large``,
    #: ``max_epochs``, ``max_training_time`` in minutes, ``value_protection``).
    #: Empty keeps the library defaults.
    params: dict = dataclasses.field(default_factory=dict)
    #: Arguments for ``TabularARGN.sample()``. Value protection (on by default)
    #: hides categories too rare to release; ``sample`` draws a common category
    #: in their place instead of the library's ``_RARE_`` token, which is not a
    #: real value of the column.
    sample_params: dict = dataclasses.field(
        default_factory=lambda: {"rare_category_replacement_method": "sample"}
    )


@dataclasses.dataclass
class CDTDConfig:
    """CDTD (Mueller, Gruber & Fok, ICLR 2025), run in its own environment.

    The authors' ``cdtd_simple`` code is the submodule ``submodules/cdtd_simple``;
    it needs Python 3.13 and a newer torch, so it is installed separately
    (``uv sync --project envs/cdtd``) and run as a subprocess; see
    ``synthdata/generation/cdtd_backend.py``. It fits on the imputed train
    split, like the synthcity models. No HPO.
    """

    #: Off by default: needs the separate environment above.
    enabled: bool = False
    #: The environment's folder (relative paths are from the repo root).
    env_dir: str = "envs/cdtd"
    #: The cdtd_simple checkout.
    source_dir: str = "submodules/cdtd_simple"
    #: Arguments for ``CDTD()`` (e.g. ``mlp_n_units``, ``cat_emb_dim``). Empty
    #: keeps the authors' defaults.
    params: dict = dataclasses.field(default_factory=dict)
    #: Arguments for ``CDTD.fit()``: ``num_steps_train`` (default 30000),
    #: ``num_steps_warmup`` (1000), ``batch_size`` (4096, capped at the train
    #: rows), ``lr`` (1e-3).
    fit_params: dict = dataclasses.field(default_factory=dict)
    #: Arguments for ``CDTD.sample()``: ``num_steps`` (default 200), ``batch_size``.
    sample_params: dict = dataclasses.field(default_factory=dict)
    #: CDTD cannot condition on the label, so for class quotas one fit draws
    #: this many times the rows needed and the quotas are filled from them.
    pool_factor: float = 3.0


@dataclasses.dataclass
class HPOConstraintsConfig:
    """Lenient sanity screens for HPO trials, passed to Optuna as constraints.

    A trial that fails a screen keeps its objective value but is marked
    infeasible, so it can't be chosen as best and the sampler learns to avoid
    its region (constrained TPE). ``null`` turns a screen off.
    With ``enforce: false`` every screen is still measured and logged on each
    trial (and in the evaluation's ``checks.csv``), but no trial is marked
    infeasible for failing one.
    """

    #: Hold trials to the screens below. False: measure and log them only, so
    #: the best objective value wins whatever its screens say, and a search
    #: never falls back to the untuned defaults for want of a passing trial.
    enforce: bool = True

    #: Copies: the share of synthetic rows that exactly match a search-train
    #: row may exceed the same share for the real tuning rows by at most this.
    copy_margin: float | None = 0.02
    #: Mode collapse: for every categorical column, the share of its train
    #: categories with at least 1% frequency that the synthetic data contains.
    #: Every target class must also be present.
    #: The minimum over all categorical columns, so one rare level anywhere
    #: fails the trial; the protected screens below hold the fairness-relevant
    #: columns to a stricter standard.
    min_category_coverage: float | None = 0.8
    #: Implausible values: the share of numeric values outside the train
    #: [min, max] range may exceed the same share for the real tuning rows by
    #: at most this.
    max_out_of_range: float | None = 0.01
    #: Class shortfall (with ``generation.match_class_prior``): the total
    #: variation distance between the candidate's class shares and the
    #: search-train shares may be at most this. Candidates are drawn to those
    #: shares, so a gap means a class fell short of its quota.
    max_class_share_gap: float | None = 0.02
    #: Protected groups present: for every ``data.protected_columns`` column,
    #: the share of its groups (at least 1% frequency in train; numeric
    #: columns in ``data.protected_bins``, or train quartiles without them)
    #: the synthetic data contains.
    min_protected_coverage: float | None = 1.0
    #: Protected groups in proportion: the largest total variation distance,
    #: over the protected columns, between the synthetic and search-train
    #: group shares may exceed that of as many real tuning rows by at most this.
    max_protected_share_gap: float | None = 0.1
    #: Memorization: the share of synthetic rows nearer to a search-train row
    #: than to any tuning row, against a search-train sample as large as the
    #: tuning split (0.5 = no memorization; Platzer & Reutterer, 2021).
    max_dcr_closer_to_train: float | None = 0.6


@dataclasses.dataclass
class HPOConfig:
    """Optuna hyperparameter search, one study per model.

    Every trial fits on train minus tuning and is scored on the tuning split;
    the final model is refitted with the best parameters on all of train.
    Studies are stored on disk, so an interrupted search resumes.
    """

    #: Run the search. When False, models train with their library defaults.
    enabled: bool = True
    #: What a trial optimizes, scored on the tuning split by a fixed XGBoost
    #: trained on the synthetic rows (train on synthetic, test on real):
    #: "tstr_macro_auprc", "tstr_macro_auroc", "tstr_macro_f1", "tstr_kappa"
    #: (Cohen's kappa, quadratic-weighted for an ordinal target); all are
    #: logged on every trial. The XGBoost weights rows by inverse class frequency, as
    #: in every predictive fit of the pipeline.
    #: "auto" (default) picks one of ``objective_candidates`` per model by a
    #: fixed rule (see ``synthdata.generation.hpo.choose_objective``): drop
    #: candidates whose real-data headroom over chance is under
    #: ``min_headroom_z`` bootstrap standard errors, then take the one whose
    #: scores across the random start-up trials vary most beyond their noise.
    #: "synthcity_composite": the older average of ``metric_config``.
    objective: str = "auto"
    #: Candidates of ``objective: auto``, in order of preference (ties go to
    #: the first).
    objective_candidates: list = dataclasses.field(
        default_factory=lambda: [
            "tstr_macro_auprc",
            "tstr_macro_auroc",
            "tstr_macro_f1",
        ]
    )
    #: Headroom a candidate needs: (TRTR - chance) / bootstrap SE of TRTR.
    #: A fixed objective under it only gets a warning.
    min_headroom_z: float = 2.0
    #: XGBoost seeds averaged per TSTR score. The classifier is deterministic
    #: (no row or column subsampling), so more seeds only repeat the fit.
    tstr_seeds: int = 1
    #: Bootstrap resamples of the tuning rows behind each score's standard
    #: error (logged as ``tstr_boot_se``; used by ``objective: auto`` and
    #: ``min_delta``). 0 = off.
    n_bootstrap: int = 200
    #: Log cheap fidelity scores on every trial (marginal distance, pairwise
    #: association difference, detection AUC against the tuning rows), so a
    #: composite objective can be compared afterwards without a new run
    #: (``scripts/hpo_analysis.py compare``). Never part of the objective.
    log_fidelity: bool = True
    #: Sanity screens a trial must pass to be chosen (see HPOConstraintsConfig).
    constraints: HPOConstraintsConfig = dataclasses.field(default_factory=HPOConstraintsConfig)
    #: Trials and wall-clock seconds per study, for models not listed in the
    #: per-model budgets below.
    n_trials: int = 10
    #: None = no time limit.
    timeout_seconds: int | None = 300
    #: Trials per model (keys: synthcity plugin names, ``tabpfgen_standard``,
    #: ``tabpfgen_custom``). 0 skips the search; the model keeps its defaults.
    n_trials_per_model: dict = dataclasses.field(default_factory=dict)
    #: Seconds per model study; set from a profiling run as trials x measured
    #: trial time x 1.5, so one slow model can't use up another's budget.
    timeout_seconds_per_model: dict = dataclasses.field(default_factory=dict)
    #: Wall-clock hours for the whole generation stage, counted from its start
    #: (None = no limit). Every model's search stops starting trials early
    #: enough to leave time for the refits of its top trials (``refit_top_k``
    #: x ``refit_seeds``, each counted as its longest trial) and its tuned fit
    #: and replicates (counted as long as its untuned fits), so the stage
    #: ends near this limit with every model finished. Imputation and
    #: evaluation are not included: on a cluster, set it to the job's time
    #: limit minus the time those take. Per-model timeouts still apply.
    stage_time_limit_hours: float | None = None
    #: Training length searched per model as ``[low, high, step]`` (synthcity
    #: ``n_iter`` epochs, TabPFGen SGLD steps). Unlisted models use their
    #: library's own range (``EPOCH_RANGE_DEFAULTS`` where the library has none).
    #: The top of the range also caps the untuned (default) model's training
    #: length when the library default is longer.
    epoch_ranges: dict = dataclasses.field(default_factory=dict)
    #: "median": Optuna's MedianPruner stops a trial whose intermediate score
    #: (synthcity's GAN detection metric) falls below the median of earlier
    #: trials at the same step. null (default) = off: on the sim run that
    #: score sat at 1.000 for every GAN trial, did not predict the objective,
    #: and never checked the shortest trials, which biased the search to them.
    pruner: str | None = None
    #: Trials that finish before pruning starts, and intermediate scores a
    #: trial reports before it can be pruned (1: from its first check on).
    pruner_startup_trials: int = 5
    #: See above.
    pruner_warmup_steps: int = 1
    #: Random start-up trials before TPE models the results (Optuna's default).
    n_startup_trials: int = 10
    #: Multivariate TPE (with ``group``): models interactions between
    #: hyperparameters, such as learning rate and batch size.
    multivariate: bool = True
    #: Stop a study once this many finished trials in a row failed to improve
    #: the best feasible value by more than ``min_delta``. The first
    #: ``n_startup_trials`` trials set the value to beat but don't count, so
    #: TPE always gets ``patience`` trials of its own. null = off; timeouts
    #: stay as caps.
    patience: int | None = 10
    #: Improvement that resets ``patience``. null = the median bootstrap
    #: standard error of the objective over the trials so far.
    min_delta: float | None = None
    #: Enqueue the library's default hyperparameters as the first trial (the
    #: values inside the searched space), so the search starts from them.
    enqueue_defaults: bool = True
    #: Warm start from an earlier run's studies on similar data: the top
    #: ``top_k`` feasible trials of the study with the same name in the Optuna
    #: storage ``storage`` (a path or URL), ranked by the logged ``metric``,
    #: are enqueued first. null storage = off.
    warm_start: dict = dataclasses.field(
        default_factory=lambda: {"storage": None, "top_k": 3, "metric": "tstr_macro_auprc"}
    )
    #: Per model (synthcity plugin names), the hyperparameters to search;
    #: the rest stay at the library default. Unlisted models search their whole
    #: space. ``scripts/hpo_analysis.py importance`` shows which matter.
    search_params: dict = dataclasses.field(default_factory=dict)
    #: Before taking the best trial, refit the ``refit_top_k`` best feasible
    #: trials with ``refit_seeds`` new seeds each and pick the best mean: the
    #: best of many noisy trials is biased upward (the winner's curse).
    #: 0 seeds = off.
    refit_top_k: int = 3
    #: See above.
    refit_seeds: int = 2
    #: Metrics of the "synthcity_composite" objective, oriented to "higher is
    #: better" and averaged. Unused by the TSTR objectives.
    metric_config: dict = dataclasses.field(
        default_factory=lambda: {
            "stats": [
                "prdc",
                "alpha_precision",
                "wasserstein_dist",
                "inv_kl_divergence",
            ],
            "sanity": ["nearest_syn_neighbor_distance"],
            "performance": ["xgb"],
            # DOMIAS is deferred from HPO until its high-dimensional KDE
            # failure modes have a separately validated treatment.
            "privacy": ["identifiability_score"],
        }
    )
    #: Optuna storage URL, e.g. "sqlite:///output/dataset/optuna_studies.db".
    #: If None, a default sqlite file under the generation output dir is used.
    storage: str | None = None
    #: Where best-params-per-model are cached as JSON. If None, defaults under output_dir.
    best_params_path: str | None = None
    #: Override n_iter for the final "optimized" build of iterative models (None = no override).
    final_n_iter_override: int | None = None


@dataclasses.dataclass
class GenerationConfig:
    """Which generators run, how many rows they produce, and their HPO."""

    #: Synthetic rows produced per model (fewer if ``match_class_prior`` leaves
    #: a class short). None (the default) matches the real rows the model is
    #: fitted on: the train split for the final models, search-train for HPO
    #: candidates. Same-size synthetic data is the usual convention (synthpop),
    #: keeps TSTR comparable with the real-data ceiling, and adds no rows that
    #: carry no new signal but more chances to leak a memorized record. An
    #: integer fixes the count instead (e.g. to cap a slow generator); the
    #: ``baseline_train_copy`` reference then holds that many real train rows.
    n_samples: int | None = None
    #: Times each model is fitted and sampled, with seeds seed, seed+1, ...
    #: Replicate r > 0 is saved as "<model>__rep<r>". With 2 or more, the
    #: evaluation reports each score as a mean with a 95% confidence interval
    #: and flags models it cannot tell apart from the best. HPO runs once.
    n_replicates: int = 1
    #: Base artifact root. Runtime stage paths are versioned under
    #: ``<output_dir>/<data.version or 'unversioned'>/<experiment-id>/``.
    output_dir: str = "output/dataset/synthetic_data"
    #: Retrain and overwrite models whose synthetic CSV already exists in this
    #: experiment (False reuses them, so a rerun only fills in what is missing).
    force_retrain: bool = False
    #: When a model's search, fit or sampling fails (for example out of GPU
    #: memory), log it, list it in ``generation_failures.json`` and go on with
    #: the other models, so evaluation still runs on every model that
    #: finished; the stage fails only if no synthetic dataset is left. False
    #: stops the run at the first failed model instead. A rerun of the same
    #: experiment redoes only the failed models (the others are cached).
    continue_on_model_failure: bool = True
    #: synthcity generators (see SynthcityModelsConfig).
    synthcity: SynthcityModelsConfig = dataclasses.field(default_factory=SynthcityModelsConfig)
    #: TabPFN generators (see TabPFNConfig).
    tabpfn: TabPFNConfig = dataclasses.field(default_factory=TabPFNConfig)
    #: TabPFGen generators (see TabPFGenConfig).
    tabpfgen: TabPFGenConfig = dataclasses.field(default_factory=TabPFGenConfig)
    #: TabularARGN generator (see TabularARGNConfig).
    tabularargn: TabularARGNConfig = dataclasses.field(default_factory=TabularARGNConfig)
    #: CDTD generator (see CDTDConfig).
    cdtd: CDTDConfig = dataclasses.field(default_factory=CDTDConfig)
    #: Hyperparameter search (see HPOConfig).
    hpo: HPOConfig = dataclasses.field(default_factory=HPOConfig)
    #: Draw every synthetic dataset (and every HPO candidate) to the real train
    #: class shares of a categorical target: each class gets its share of
    #: ``n_samples`` rows, by conditional sampling where the model conditions
    #: on the label (ddpm) and rejection sampling otherwise, as SDV's
    #: conditional sampling does. Rows are never repeated; a class still short
    #: after 10 batches stays short and is recorded in ``diagnostics/class_sampling.csv``.
    #: Without it, a generator that over-produces minority classes raises
    #: macro-F1 without being more faithful.
    match_class_prior: bool = True

    def rows_for(self, n_real_rows: int) -> int:
        """Synthetic rows to draw for a model fitted on ``n_real_rows`` real rows."""
        return n_real_rows if self.n_samples is None else self.n_samples

    def enabled_generators(self) -> list[str]:
        """The generator families this config turns on (empty: nothing to generate)."""
        families = {
            "synthcity": self.synthcity.enabled and bool(self.synthcity.names),
            "tabpfn": self.tabpfn.enabled and bool(self.tabpfn.data_variants),
            "tabpfgen": self.tabpfgen.enabled,
            "tabularargn": self.tabularargn.enabled,
            "cdtd": self.cdtd.enabled,
        }
        return [name for name, on in families.items() if on]


#: Shown when no generator is enabled, so the run stops before any work.
NO_GENERATOR_MESSAGE = (
    "No generator is enabled in {path}, so generation would produce 0 synthetic datasets "
    "and evaluation would have nothing to score. Turn at least one on: "
    "generation.synthcity.enabled (with generation.synthcity.names), generation.tabpfn.enabled, "
    "generation.tabpfgen.enabled, generation.tabularargn.enabled or generation.cdtd.enabled."
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FrameworkSelectionConfig:
    """Partial-selection controls for one evaluation framework.

    ``metrics`` (explicit metric names) takes precedence over ``categories``
    (utility/privacy/... groupings) when both are given.
    """

    #: Run this framework's metrics at all.
    enabled: bool = True
    #: Metric types to run: any of "utility", "privacy", "fairness" (None =
    #: all). synthcity has no fairness metrics.
    categories: list | None = None
    #: Exact metric names to run (None = all in the selected categories); see
    #: synthdata/evaluation/catalog.py for the names.
    metrics: list | None = None


@dataclasses.dataclass
class LogDisparityConfig:
    """Log-disparity fairness metric: how much each protected subgroup's
    target distribution differs between real and synthetic data."""

    #: Defaults to data.protected_columns if left empty.
    protected_columns: list = dataclasses.field(default_factory=list)
    #: Readable labels for target values in the report, ``{value: label}``.
    target_map: dict | None = None
    #: Readable labels per protected column, aligned with
    #: ``protected_columns`` (null entries keep the raw values).
    protected_map: list | None = None
    #: Bin edges per protected column, aligned with ``protected_columns``
    #: (null = use the values as groups), e.g. ``[null, [0, 18, 30, 60, 200]]``.
    #: Leave null to use ``data.protected_bins``, which most configs should.
    protected_bins: list | None = None


#: What a privacy attack counts as one individual (see PrivacyAttacksConfig.unit).
PRIVACY_UNITS = ("patient", "row")

#: Anonymeter singling-out attack modes.
SINGLING_OUT_MODES = ("multivariate", "univariate")


@dataclasses.dataclass
class AnonymeterConfig:
    """Anonymeter attacks (Giomi et al., PoPETs 2023), each scored as attack
    success above a baseline attack, with the test split as the control set."""

    #: Attack targets per attack (capped at the control-set size).
    n_attacks: int = 500
    #: "multivariate" (predicates over several columns; realistic for wide
    #: data) or "univariate" (one column at a time).
    singling_out_mode: str = "multivariate"
    #: Columns per multivariate singling-out predicate.
    singling_out_n_cols: int = 3
    #: Candidate predicates tried before singling out stops short of
    #: ``n_attacks`` (Anonymeter's own default, 10 million, can run for
    #: hours on data that is hard to single out). Stopping short logs a
    #: warning and can underestimate the risk.
    singling_out_max_attempts: int = 100_000
    #: Neighbours a linkability attack may match a target among.
    linkability_n_neighbors: int = 10
    #: Confidence level of the reported risk intervals.
    confidence_level: float = 0.95


@dataclasses.dataclass
class PrivacyAttacksConfig:
    """Custom privacy metrics: Anonymeter attacks and holdout-referenced
    distance heuristics (DCR/NNDR; see synthdata.evaluation.privacy_attacks)."""

    #: "row" (default): every row is an individual, as each synthetic row is
    #: released as its own record. "patient": attacks target one encounter
    #: per patient and the distance membership check scores each patient by
    #: their closest encounter, so a frequent attender counts once.
    unit: str = "row"
    #: Anonymeter attack settings (see AnonymeterConfig).
    anonymeter: AnonymeterConfig = dataclasses.field(default_factory=AnonymeterConfig)


@dataclasses.dataclass
class BinaryTargetConfig:
    """Collapse a multi-class target into a binary (0/1) variable for a
    *second, separate* SynthEval pass, so metrics that require exactly 2
    target classes (auroc_diff, statistical_parity, equalized_odds,
    equal_opportunity) can run even when the real target has 3+ classes.

    This is evaluation-only: it never touches ``data.target_column``,
    generation, or any other metric's target -- the collapse is applied to
    disposable copies of the real/synthetic dataframes used only for this
    extra pass, replacing the target column's values in place (same column
    name, so it still gets excluded from the model's feature set exactly
    like the original target -- no leakage risk from the original,
    finer-grained labels lingering as a feature).

    ``positive_classes``/``negative_classes`` must together cover every
    observed value of the target column (fails loudly otherwise -- see
    :func:`synthdata.evaluation.syntheval_eval.build_binary_target_series`).
    """

    #: Required (and only used) when ``evaluation.class_averaging`` is "binary".
    enabled: bool = False
    #: Column to collapse; defaults to data.target_column if left None.
    column: str | None = None
    #: Original class values mapped to the binary "positive" (1) outcome.
    positive_classes: list = dataclasses.field(default_factory=list)
    #: Original class values mapped to the binary "negative"/reference (0) outcome.
    negative_classes: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ComputeConfig:
    """How many models run at once, in generation and in evaluation.

    Generation trains each synthcity model (untuned fit, HPO study, tuned fit,
    every replicate) in its own process; evaluation scores each synthetic
    dataset in its own process. Each model seeds itself, so results match a
    one-at-a-time run with the same threads per model. ``workers: "auto"``
    uses CPU cores // ``cores_per_worker``, capped by free memory, by
    ``max_workers`` and by the number of models.
    """

    #: Models processed at once: "auto" or a fixed number (1 = one at a time).
    workers: str | int = "auto"
    #: Upper limit on ``workers``.
    max_workers: int = 8
    #: CPU threads each model's process may use (torch, OpenMP, BLAS, joblib).
    cores_per_worker: int = 4
    #: RAM (GiB) kept free for the system and the main process.
    memory_reserve_gib: float = 16.0
    #: RAM (GiB) one model's process needs; None estimates it from the column count.
    memory_per_worker_gib: float | None = None
    #: GPU sharing while models train in parallel (synthdata.gpu_slots): each
    #: fit runs on the least busy GPU that has the memory its model needs
    #: (measured on the model's earlier fits; a share of the GPU before the
    #: first). "auto" sets no limit on the number of fits per GPU (sharing
    #: raised total speed in measurements); a number caps the fits per GPU,
    #: most remaining work first; 0/None keeps each model on one GPU,
    #: assigned round-robin, with no memory check.
    fits_per_gpu: str | int | None = "auto"
    #: Before any stage, when the generate stage is selected, ``synthdata-run``
    #: checks each enabled generator that has its own environment (TabularARGN,
    #: CDTD): its submodule checkout, ``uv sync --frozen --project <env>`` (a
    #: no-op when the environment matches its lock file) and whether its torch
    #: imports and sees CUDA. A failure stops the run at once. False skips the
    #: sync (for an offline machine whose environments are already set up); the
    #: other checks still run.
    sync_envs: bool = True


#: Values of ``evaluation.class_averaging``.
CLASS_AVERAGING = ("ovr_macro", "binary")

#: Reference rows ``evaluation.baselines`` may name; built by
#: synthdata.evaluation.baselines.
EVALUATION_BASELINES = ("train_copy", "marginals")


@dataclasses.dataclass
class EvaluationConfig:
    #: Base artifact root. Runtime stage paths are versioned under
    #: ``<output_dir>/<data.version or 'unversioned'>/<experiment-id>/``.
    output_dir: str = "output/dataset/evaluation"
    #: Restrict evaluation to a subset of generated model names (None = all found on disk).
    models: list | None = None
    #: Fixed reference rows scored alongside the generators so every run has
    #: the same anchors (see synthdata.evaluation.baselines): "train_copy"
    #: (real training rows: the utility ceiling and privacy floor) and
    #: "marginals" (each column sampled independently: no joint structure).
    #: They are never recommended. Empty list = no baselines.
    baselines: list = dataclasses.field(default_factory=lambda: ["train_copy", "marginals"])
    #: Target value SynthEval's fairness metrics treat as the positive outcome.
    #: Used only when the target itself has two classes.
    positive_class: Any = 1

    #: Metric selection per framework (see FrameworkSelectionConfig):
    #: synthcity's metrics, SynthEval's metrics, and this project's own
    #: ("custom": holdout TSTR, Anonymeter, holdout distances, log disparity,
    #: equalized odds / equal opportunity).
    synthcity: FrameworkSelectionConfig = dataclasses.field(
        default_factory=FrameworkSelectionConfig
    )
    syntheval: FrameworkSelectionConfig = dataclasses.field(
        default_factory=FrameworkSelectionConfig
    )
    custom: FrameworkSelectionConfig = dataclasses.field(default_factory=FrameworkSelectionConfig)

    #: "linear" (min-max scale + sum) or "summation" (SynthEval's built-in strategy).
    ranking_strategy: str = "linear"
    #: Log-disparity fairness metric settings (see LogDisparityConfig).
    log_disparity: LogDisparityConfig = dataclasses.field(default_factory=LogDisparityConfig)
    #: Save SynthEval's own figures for each model next to its results.
    save_per_model_syntheval_plots: bool = True
    #: How SynthEval metrics that need exactly 2 target classes (auroc_diff
    #: and the subgroup gap metrics) run when the target has 3+ classes:
    #: "ovr_macro" runs them once per class (that class vs the rest) and
    #: averages with equal weight per class, keeping the per-class values in
    #: ovr_per_class.csv; "binary" runs them once on the ``binary_target``
    #: collapse, which must then be enabled. A 2-class target runs them
    #: directly and ignores this setting.
    class_averaging: str = "ovr_macro"
    #: Binary collapse of the target (see BinaryTargetConfig).
    binary_target: BinaryTargetConfig = dataclasses.field(default_factory=BinaryTargetConfig)
    #: XGBoost seeds averaged for the holdout TSTR scores (fit on each
    #: synthetic dataset, score on the test split; see
    #: synthdata.evaluation.tstr). Needs a categorical target.
    tstr_seeds: int = 3
    #: Parameters for the XGBoost classifier that predicts the target in
    #: holdout TSTR/TSTS, the HPO objective and its TRTR ceiling, for example
    #: parameters tuned on the real data for your own baseline model. Keys
    #: are ``xgboost.XGBClassifier`` arguments and replace the matching
    #: defaults (n_estimators 80, max_depth 4, learning_rate 0.08, subsample
    #: 1, colsample_bytree 1, tree_method hist; see
    #: synthdata.evaluation.tstr.XGB_PARAMS); keys left out keep them.
    #: ``random_state``, ``seed``, ``objective`` and ``num_class`` are set by
    #: the pipeline and refused here. Class-balanced sample weights still
    #: apply. With ``subsample`` or ``colsample_bytree`` below 1 the fit
    #: depends on its seed, so ``tstr_seeds`` (and ``generation.hpo.tstr_seeds``)
    #: above 1 average that out. Not used by SynthEval's or synthcity's own
    #: utility metrics, the real-vs-synthetic detection score or the privacy
    #: attacks. Changing it changes the HPO objective: use a new experiment
    #: id. null = defaults.
    xgboost_params: dict | None = None
    #: Anonymeter attacks and holdout-referenced DCR/NNDR (custom privacy).
    privacy_attacks: PrivacyAttacksConfig = dataclasses.field(default_factory=PrivacyAttacksConfig)

    #: Per-"type" (utility/privacy/fairness) weight applied when rolling up
    #: type-level ranks into the overall rank, a weighted geometric mean (see
    #: synthdata.evaluation.combine.build_combined_table). Keys must be
    #: exactly {"utility","privacy","fairness"}; values must be non-negative.
    #: Default is equal weight; raise "privacy" to make privacy count for more.
    rank_weights: dict = dataclasses.field(
        default_factory=lambda: {"utility": 1.0, "privacy": 1.0, "fairness": 1.0}
    )
    #: Whether to write report.md (alongside combined_evaluation.csv): the one
    #: page that summarizes the run, embeds its plots and links every output.
    #: ``synthdata-plot`` rewrites it once the plots exist.
    generate_report: bool = True

    #: Sanity checks on every run, written to checks.csv and listed in the
    #: report (synthdata/evaluation/checks.py): synthetic rows that copy train
    #: rows, invented categories, out-of-range values, class shares, and the two
    #: baselines landing where they must. Warnings only; run
    #: ``synthdata-evaluate --strict-checks`` to fail on any warning.
    run_checks: bool = True


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlotsConfig:
    #: Base artifact root. Dataset QA figures use
    #: ``<output_dir>/<data.version or 'unversioned'>/dataset/``; experiment
    #: figures add ``<experiment-id>/`` beneath the version scope.
    output_dir: str = "output/dataset/plots"
    #: Which figure groups to (re)generate: "data", "imputation", "generation", "hpo", "evaluation".
    sections: list = dataclasses.field(
        default_factory=lambda: [
            "data",
            "imputation",
            "generation",
            "hpo",
            "evaluation",
        ]
    )
    #: Resolution of raster figures.
    dpi: int = 150
    #: File formats written for every figure, e.g. ["png", "pdf"].
    formats: list = dataclasses.field(default_factory=lambda: ["png"])


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExperimentConfig:
    """Identifies and tags this pipeline run for artifact versioning.

    Every invocation of the CLI scripts is treated as an "experiment": its
    generation/evaluation/plot artifacts are nested under
    `<stage_output_dir>/data_v_<data.version>/exp_v_<experiment_id>/`,
    and a manifest.json log at
    `<generation_output_dir>/../experiments/data_v_<data.version>/exp_v_<experiment_id>/manifest.json`
    records what each stage produced (see :mod:`synthdata.experiment`).
    """

    #: Freeform label (e.g. "baseline", "hpo-v2"). Included in the auto-generated
    #: experiment id, and recorded in the manifest regardless of `id`.
    tag: str | None = None
    #: Explicit experiment id. Re-using an id resumes/extends that experiment
    #: (e.g. reusing cached synthetic data, appending new manifest entries).
    #: If None, an id is auto-generated per run from a UTC timestamp (+ tag).
    id: str | None = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    #: Short dataset/run name, used to build default paths (data/<name>, output/<name>/...).
    name: str = "dataset"
    #: Master random seed (split, imputation, generators, metrics). Seed
    #: replicates use seed, seed+1, ... (``generation.n_replicates``).
    seed: int = 42
    #: "auto" | "cpu" | "cuda" | "mps"
    device: str = "auto"

    #: One section per pipeline stage; see each class for its options.
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    imputation: ImputationConfig = dataclasses.field(default_factory=ImputationConfig)
    generation: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    #: Parallel models in generation and evaluation (see ComputeConfig).
    compute: ComputeConfig = dataclasses.field(default_factory=ComputeConfig)
    plots: PlotsConfig = dataclasses.field(default_factory=PlotsConfig)
    experiment: ExperimentConfig = dataclasses.field(default_factory=ExperimentConfig)

    #: Populated by load_config(); not read from YAML.
    config_path: Path | None = None


def _from_dict(cls, data: dict | None):
    """Recursively build a dataclass instance from a (possibly nested) dict."""
    if data is None:
        return cls()
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if cls is DataConfig and key in _REMOVED_DATA_KEYS:
            raise ValueError(f"data.{key} was replaced by {_REMOVED_DATA_KEYS[key]}")
        if cls is HPOConfig and key in _REMOVED_HPO_KEYS:
            raise ValueError(f"generation.hpo.{key} was replaced by {_REMOVED_HPO_KEYS[key]}")
        if cls is EvaluationConfig and key == "syntheval_execution":
            raise ValueError(
                "evaluation.syntheval_execution was replaced by the top-level compute section "
                "(model_workers -> workers, max_model_workers -> max_workers, cores_per_model -> "
                "cores_per_worker, memory_per_model_gib -> memory_per_worker_gib), which "
                "generation uses too"
            )
        if cls is EvaluationConfig and key == "privacy_gate":
            raise ValueError(
                "evaluation.privacy_gate was removed: fixed thresholds cannot certify privacy. "
                "Delete the section; report.md shows the privacy attacks next to the baselines."
            )
        if key not in field_types:
            raise ValueError(
                f"Unknown config key '{key}' for {cls.__name__}. Valid keys: {sorted(field_types)}"
            )
        nested_cls = _NESTED_DATACLASSES.get((cls, key))
        if nested_cls is not None and isinstance(value, dict):
            kwargs[key] = _from_dict(nested_cls, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# Explicit registry of which fields are nested dataclasses (avoids relying on
# fragile string-based typing.get_type_hints resolution for forward refs).
_NESTED_DATACLASSES = {
    (Config, "data"): DataConfig,
    (Config, "imputation"): ImputationConfig,
    (Config, "generation"): GenerationConfig,
    (Config, "evaluation"): EvaluationConfig,
    (Config, "plots"): PlotsConfig,
    (Config, "compute"): ComputeConfig,
    (Config, "experiment"): ExperimentConfig,
    (ImputationConfig, "refidiff"): RefiDiffConfig,
    (ImputationConfig, "missforest"): MissForestConfig,
    (ImputationConfig, "missing_indicators"): MissingIndicatorConfig,
    (ImputationConfig, "benchmark"): RefiDiffBenchmarkConfig,
    (RefiDiffBenchmarkConfig, "hpo"): RefiDiffBenchmarkHPOConfig,
    (GenerationConfig, "synthcity"): SynthcityModelsConfig,
    (GenerationConfig, "tabpfn"): TabPFNConfig,
    (GenerationConfig, "tabpfgen"): TabPFGenConfig,
    (GenerationConfig, "tabularargn"): TabularARGNConfig,
    (GenerationConfig, "cdtd"): CDTDConfig,
    (GenerationConfig, "hpo"): HPOConfig,
    (HPOConfig, "constraints"): HPOConstraintsConfig,
    (EvaluationConfig, "synthcity"): FrameworkSelectionConfig,
    (EvaluationConfig, "syntheval"): FrameworkSelectionConfig,
    (EvaluationConfig, "custom"): FrameworkSelectionConfig,
    (EvaluationConfig, "log_disparity"): LogDisparityConfig,
    (EvaluationConfig, "binary_target"): BinaryTargetConfig,
    (EvaluationConfig, "privacy_attacks"): PrivacyAttacksConfig,
    (PrivacyAttacksConfig, "anonymeter"): AnonymeterConfig,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = _from_dict(Config, raw)
    cfg.config_path = config_path
    _validate(cfg)
    return cfg


#: Split keys removed in favour of the three-way ``*_fraction`` keys.
_REMOVED_DATA_KEYS = {
    "train_size": "data.train_fraction, data.tuning_fraction and data.holdout_fraction",
    "tuning_size": "data.train_fraction, data.tuning_fraction and data.holdout_fraction",
    "stratify": "data.stratify_columns (an empty list turns stratification off)",
}

#: Search-time training caps removed in favour of searched training length.
_REMOVED_HPO_KEYS = {
    key: "generation.hpo.epoch_ranges (training length searched per model) and generation.hpo.pruner"
    for key in ("n_iter_cap", "model_iter_caps", "sgld_step_cap")
}


def split_fold_count(fractions: tuple[float, ...]) -> int:
    """Smallest k <= 20 for which every fraction is a whole number of 1/k folds."""
    for k in range(2, 21):
        if all(abs(f * k - round(f * k)) < 1e-6 for f in fractions):
            return k
    raise ValueError(
        f"data split fractions {list(fractions)} must each be a multiple of 1/k for some "
        "k <= 20 (for example 0.6/0.2/0.2 or 0.5/0.2/0.3)"
    )


def _validate_split(cfg: Config) -> None:
    data = cfg.data
    fractions = (data.train_fraction, data.tuning_fraction, data.holdout_fraction)
    if not (0 < data.train_fraction < 1 and 0 < data.holdout_fraction < 1):
        raise ValueError("data.train_fraction and data.holdout_fraction must be between 0 and 1")
    if not 0 <= data.tuning_fraction < 1:
        raise ValueError("data.tuning_fraction must be at least 0 and below 1")
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError(
            f"data.train_fraction + tuning_fraction + holdout_fraction must be 1, got {sum(fractions)}"
        )
    split_fold_count(fractions)
    if cfg.generation.hpo.enabled and data.tuning_fraction == 0:
        raise ValueError(
            "generation.hpo.enabled needs a tuning split to score candidates on; set "
            "data.tuning_fraction above 0 (for example 0.2)"
        )
    _validate_hpo(cfg.generation.hpo)
    if data.stratify_bins is not None:
        columns = (
            data.stratify_columns if data.stratify_columns is not None else [data.target_column]
        )
        if len(data.stratify_bins) != len(columns):
            raise ValueError("data.stratify_bins needs one entry (or null) per stratify column")
        for edges in data.stratify_bins:
            if edges is not None and (not edges or list(edges) != sorted(set(edges))):
                raise ValueError(
                    f"data.stratify_bins entries must be strictly increasing edges, got {edges!r}"
                )


TSTR_OBJECTIVES = (
    "tstr_macro_auprc",
    "tstr_macro_auroc",
    "tstr_macro_f1",
    "tstr_kappa",
)
HPO_OBJECTIVES = (*TSTR_OBJECTIVES, "auto", "synthcity_composite")
HPO_PRUNERS = ("median", None)


def _validate_hpo(hpo: HPOConfig) -> None:
    if hpo.objective not in HPO_OBJECTIVES:
        raise ValueError(
            f"generation.hpo.objective must be one of {list(HPO_OBJECTIVES)}, got {hpo.objective!r}"
        )
    if (
        isinstance(hpo.tstr_seeds, bool)
        or not isinstance(hpo.tstr_seeds, int)
        or hpo.tstr_seeds < 1
    ):
        raise ValueError(
            f"generation.hpo.tstr_seeds must be a positive integer, got {hpo.tstr_seeds!r}"
        )
    bad = [c for c in hpo.objective_candidates if c not in TSTR_OBJECTIVES]
    if not hpo.objective_candidates or bad:
        raise ValueError(
            f"generation.hpo.objective_candidates must be a non-empty list of "
            f"{list(TSTR_OBJECTIVES)}, got {hpo.objective_candidates!r}"
        )
    for key in ("n_bootstrap", "n_startup_trials", "refit_top_k", "refit_seeds"):
        value = getattr(hpo, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"generation.hpo.{key} must be a non-negative integer, got {value!r}")
    if hpo.objective == "auto" and hpo.n_bootstrap == 0:
        raise ValueError("generation.hpo.objective: auto needs generation.hpo.n_bootstrap > 0")
    if hpo.patience is not None and (
        isinstance(hpo.patience, bool) or not isinstance(hpo.patience, int) or hpo.patience < 1
    ):
        raise ValueError(
            f"generation.hpo.patience must be a positive integer or null, got {hpo.patience!r}"
        )
    if hpo.min_delta is not None and (
        isinstance(hpo.min_delta, bool)
        or not isinstance(hpo.min_delta, int | float)
        or hpo.min_delta < 0
    ):
        raise ValueError(
            f"generation.hpo.min_delta must be a non-negative number or null, got {hpo.min_delta!r}"
        )
    unknown = set(hpo.warm_start) - {"storage", "top_k", "metric"}
    if unknown:
        raise ValueError(f"generation.hpo.warm_start has unknown keys {sorted(unknown)}")
    for model, names in hpo.search_params.items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"generation.hpo.search_params.{model} must be a list of parameter names, "
                f"got {names!r}"
            )
    if hpo.pruner not in HPO_PRUNERS:
        raise ValueError(
            f"generation.hpo.pruner must be one of {list(HPO_PRUNERS)}, got {hpo.pruner!r}"
        )
    limit = hpo.stage_time_limit_hours
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int | float) or limit <= 0
    ):
        raise ValueError(
            "generation.hpo.stage_time_limit_hours must be a positive number of hours "
            f"or null, got {limit!r}"
        )
    for key in ("n_trials_per_model", "timeout_seconds_per_model"):
        for model, value in getattr(hpo, key).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"generation.hpo.{key}.{model} must be a non-negative integer, got {value!r}"
                )
    for model, bounds in hpo.epoch_ranges.items():
        if (
            not isinstance(bounds, list | tuple)
            or len(bounds) != 3
            or not all(isinstance(b, int) and b > 0 for b in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError(
                f"generation.hpo.epoch_ranges.{model} must be [low, high, step] with "
                f"0 < low <= high, got {bounds!r}"
            )
    if not isinstance(hpo.constraints.enforce, bool):
        raise ValueError(
            f"generation.hpo.constraints.enforce must be true or false, got "
            f"{hpo.constraints.enforce!r}"
        )
    for field in dataclasses.fields(HPOConstraintsConfig):
        value = getattr(hpo.constraints, field.name)
        if field.name == "enforce":
            continue
        if value is not None and not 0 <= value <= 1:
            raise ValueError(
                f"generation.hpo.constraints.{field.name} must be between 0 and 1 or null, got {value!r}"
            )


def _validate_xgboost_params(params) -> None:
    """``evaluation.xgboost_params``: null or XGBClassifier arguments the pipeline leaves free."""
    if params is None:
        return
    if not isinstance(params, dict):
        raise ValueError(f"evaluation.xgboost_params must be a mapping or null, got {params!r}")
    from xgboost import XGBClassifier

    from synthdata.evaluation.tstr import RESERVED_XGB_KEYS

    reserved = sorted(set(params) & RESERVED_XGB_KEYS)
    if reserved:
        raise ValueError(
            f"evaluation.xgboost_params cannot set {reserved}: the pipeline sets them "
            "(seeds from tstr_seeds, objective from the target's classes)"
        )
    unknown = sorted(set(params) - set(XGBClassifier().get_params()))
    if unknown:
        raise ValueError(
            f"evaluation.xgboost_params has keys XGBClassifier does not accept: {unknown}"
        )


def _validate_column_roles(data: DataConfig) -> None:
    if data.sensitive_columns and data.protected_columns is None:
        raise ValueError(
            "data.sensitive_columns now lists the secret columns an attacker tries to infer "
            "(for example a diagnosis), not the fairness groups. Move fairness groups such as "
            "sex or age to data.protected_columns and public, linkable attributes to "
            "data.quasi_identifier_columns, then set data.protected_columns (even to []) to "
            "confirm the new meaning"
        )
    roles = {
        "quasi_identifier_columns": list(data.quasi_identifier_columns),
        "sensitive_columns": list(data.sensitive_columns),
        "protected_columns": list(data.protected_columns or []),
    }
    for key, columns in roles.items():
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        if duplicates:
            raise ValueError(f"data.{key} lists {duplicates} more than once")
        for reserved, label in (
            (data.target_column, "the target column"),
            (data.patient_id_column, "the patient ID column"),
        ):
            if reserved is not None and reserved in columns:
                raise ValueError(f"data.{key} must not contain {label} ({reserved!r})")
    for column, edges in (data.protected_bins or {}).items():
        if column not in roles["protected_columns"]:
            raise ValueError(
                f"data.protected_bins names {column!r}, which is not in data.protected_columns"
            )
        if (
            not isinstance(edges, list)
            or len(edges) < 2
            or not all(isinstance(e, int | float) and not isinstance(e, bool) for e in edges)
            or any(b <= a for a, b in zip(edges, edges[1:], strict=False))
        ):
            raise ValueError(
                f"data.protected_bins[{column!r}] must be at least two increasing numbers, "
                f"got {edges!r}"
            )
    overlap = sorted(set(roles["quasi_identifier_columns"]) & set(roles["sensitive_columns"]))
    if overlap:
        raise ValueError(
            f"{overlap} cannot be both a quasi-identifier (known to an attacker) and sensitive "
            "(what an attacker tries to infer); keep each in one of the two lists"
        )


def _validate(cfg: Config) -> None:
    if cfg.data.source not in ("uci", "csv", "parquet"):
        raise ValueError(f"data.source must be 'uci', 'csv', or 'parquet', got {cfg.data.source!r}")
    if cfg.data.source == "uci" and cfg.data.uci_id is None:
        raise ValueError("data.uci_id is required when data.source == 'uci'")
    if cfg.data.source in ("csv", "parquet") and not cfg.data.path:
        raise ValueError("data.path is required when data.source == 'csv'/'parquet'")
    if not cfg.data.target_column:
        raise ValueError("data.target_column must be set")
    if cfg.data.patient_id_column == cfg.data.target_column:
        raise ValueError("data.patient_id_column must not be the target column")
    _validate_column_roles(cfg.data)
    _validate_split(cfg)
    if cfg.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"device must be one of auto/cpu/cuda/mps, got {cfg.device!r}")
    if cfg.imputation.method not in IMPUTATION_METHODS:
        raise ValueError(
            f"imputation.method must be one of {IMPUTATION_METHODS}, got {cfg.imputation.method!r}"
        )
    indicators = cfg.imputation.missing_indicators
    if not 0 < indicators.min_missing_fraction <= 1:
        raise ValueError("imputation.missing_indicators.min_missing_fraction must be in (0, 1]")
    if indicators.indicator_only_fraction is not None and not (
        indicators.min_missing_fraction <= indicators.indicator_only_fraction <= 1
    ):
        raise ValueError(
            "imputation.missing_indicators.indicator_only_fraction must be between "
            "min_missing_fraction and 1"
        )
    missforest = cfg.imputation.missforest
    if missforest.n_estimators < 1 or missforest.max_iter < 1:
        raise ValueError("imputation.missforest.n_estimators and max_iter must be at least 1")
    if cfg.imputation.refidiff.denoiser not in ("auto", "mamba", "mlp"):
        raise ValueError(
            "imputation.refidiff.denoiser must be 'auto', 'mamba', or 'mlp', "
            f"got {cfg.imputation.refidiff.denoiser!r}"
        )
    if cfg.imputation.refidiff.categorical_decode_policy not in {
        "clip",
        "nearest_valid",
        "error",
    }:
        raise ValueError(
            "imputation.refidiff.categorical_decode_policy must be 'clip', 'nearest_valid', "
            f"or 'error', got {cfg.imputation.refidiff.categorical_decode_policy!r}"
        )
    refidiff = cfg.imputation.refidiff
    positive_refidiff_fields = (
        "hidden_dim",
        "epochs",
        "early_stopping_patience",
        "batch_size",
        "num_trials",
        "checkpoint_every",
        "catboost_warmup_iterations",
    )
    for field_name in positive_refidiff_fields:
        value = getattr(refidiff, field_name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"imputation.refidiff.{field_name} must be a positive integer, got {value!r}"
            )
    if not isinstance(refidiff.num_steps, int) or refidiff.num_steps < 2:
        raise ValueError(
            f"imputation.refidiff.num_steps must be an integer >= 2, got {refidiff.num_steps!r}"
        )
    benchmark = cfg.imputation.benchmark
    if not isinstance(benchmark.mask_fraction, (int, float)) or not 0 < benchmark.mask_fraction < 1:
        raise ValueError(
            "imputation.benchmark.mask_fraction must be a number strictly between 0 and 1, "
            f"got {benchmark.mask_fraction!r}"
        )
    if not isinstance(benchmark.n_masks, int) or benchmark.n_masks < 1:
        raise ValueError(
            f"imputation.benchmark.n_masks must be a positive integer, got {benchmark.n_masks!r}"
        )
    bad_mechanisms = set(benchmark.mechanisms) - {"mcar", "mar", "mnar"}
    if not benchmark.mechanisms or bad_mechanisms:
        raise ValueError(
            "imputation.benchmark.mechanisms must contain one or more of 'mcar', 'mar', or "
            f"'mnar', got {benchmark.mechanisms!r}"
        )
    if benchmark.score_columns is not None and (
        not isinstance(benchmark.score_columns, list) or not benchmark.score_columns
    ):
        raise ValueError(
            "imputation.benchmark.score_columns must be a non-empty list or null, "
            f"got {benchmark.score_columns!r}"
        )
    benchmark_hpo = benchmark.hpo
    if not isinstance(benchmark_hpo.n_trials, int) or benchmark_hpo.n_trials < 1:
        raise ValueError(
            "imputation.benchmark.hpo.n_trials must be a positive integer, "
            f"got {benchmark_hpo.n_trials!r}"
        )
    for field_name in (
        "hidden_dims",
        "num_steps",
        "num_trials",
        "epochs",
        "early_stopping_patience",
    ):
        values = getattr(benchmark_hpo, field_name)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"imputation.benchmark.hpo.{field_name} must be a non-empty list, got {values!r}"
            )
    n_samples = cfg.generation.n_samples
    if n_samples is not None and (
        isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1
    ):
        raise ValueError(
            f"generation.n_samples must be null or a positive integer, got {n_samples!r}"
        )
    if not isinstance(cfg.generation.n_replicates, int) or cfg.generation.n_replicates < 1:
        raise ValueError(
            f"generation.n_replicates must be a positive integer, got {cfg.generation.n_replicates!r}"
        )
    if cfg.evaluation.ranking_strategy not in ("linear", "summation"):
        raise ValueError(
            "evaluation.ranking_strategy must be 'linear' or 'summation', "
            f"got {cfg.evaluation.ranking_strategy!r}"
        )
    compute = cfg.compute
    if compute.workers != "auto" and (not isinstance(compute.workers, int) or compute.workers < 1):
        raise ValueError(
            f"compute.workers must be 'auto' or a positive integer, got {compute.workers!r}"
        )
    for field_name in ("max_workers", "cores_per_worker"):
        value = getattr(compute, field_name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"compute.{field_name} must be a positive integer, got {value!r}")
    fits = compute.fits_per_gpu
    if fits not in ("auto", None) and (
        isinstance(fits, bool) or not isinstance(fits, int) or fits < 0
    ):
        raise ValueError(
            f"compute.fits_per_gpu must be 'auto', a non-negative integer or null, got {fits!r}"
        )
    for field_name in ("memory_reserve_gib", "memory_per_worker_gib"):
        value = getattr(compute, field_name)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(
                f"compute.{field_name} must be a positive number or None, got {value!r}"
            )
    bad_data_variants = set(cfg.generation.tabpfn.data_variants) - {"raw", "imputed"}
    if bad_data_variants:
        raise ValueError(
            "generation.tabpfn.data_variants entries must be 'raw' and/or 'imputed', "
            f"got {sorted(bad_data_variants)}"
        )
    if cfg.data.nominal_columns == "auto":
        raise ValueError(
            "data.nominal_columns: 'auto' is no longer supported. Define every retained column "
            "in data.variable_schema_path instead."
        )
    if cfg.data.nominal_columns is not None and not isinstance(cfg.data.nominal_columns, list):
        raise ValueError(
            "data.nominal_columns must be a list or null; use data.variable_schema_path for new "
            "datasets."
        )
    if isinstance(cfg.data.nominal_columns, list):
        overlap = set(cfg.data.nominal_columns) & set(cfg.data.ordinal_columns)
        if overlap:
            raise ValueError(
                "data.nominal_columns and data.ordinal_columns must not overlap (a column is "
                f"either nominal or ordinal, not both): {sorted(overlap)}"
            )
    missing_ordinal = set(cfg.data.ordinal_column_categories) - set(cfg.data.ordinal_columns)
    if missing_ordinal:
        raise ValueError(
            "data.ordinal_column_categories references column(s) not listed in "
            f"data.ordinal_columns: {sorted(missing_ordinal)} -- add them to ordinal_columns so "
            "they're actually treated as ordinal/categorical instead of silently falling through "
            "to plain continuous numeric imputation/generation."
        )
    for col, categories in cfg.data.ordinal_column_categories.items():
        if not isinstance(categories, list) or len(categories) != len(set(categories)):
            raise ValueError(
                f"data.ordinal_column_categories[{col!r}] must be a list of unique values, "
                f"got {categories!r}"
            )
    if cfg.evaluation.class_averaging not in CLASS_AVERAGING:
        raise ValueError(
            f"evaluation.class_averaging must be one of {list(CLASS_AVERAGING)}, "
            f"got {cfg.evaluation.class_averaging!r}"
        )
    if (cfg.evaluation.class_averaging == "binary") != cfg.evaluation.binary_target.enabled:
        raise ValueError(
            "evaluation.binary_target.enabled must be true exactly when "
            "evaluation.class_averaging is 'binary' (the collapse replaces one-vs-rest "
            f"averaging); got class_averaging={cfg.evaluation.class_averaging!r}, "
            f"binary_target.enabled={cfg.evaluation.binary_target.enabled}"
        )
    if cfg.evaluation.tstr_seeds < 1:
        raise ValueError(f"evaluation.tstr_seeds must be >= 1, got {cfg.evaluation.tstr_seeds}")
    _validate_xgboost_params(cfg.evaluation.xgboost_params)
    attacks = cfg.evaluation.privacy_attacks
    if attacks.unit not in PRIVACY_UNITS:
        raise ValueError(
            f"evaluation.privacy_attacks.unit must be one of {list(PRIVACY_UNITS)}, "
            f"got {attacks.unit!r}"
        )
    anonymeter = attacks.anonymeter
    if anonymeter.singling_out_mode not in SINGLING_OUT_MODES:
        raise ValueError(
            "evaluation.privacy_attacks.anonymeter.singling_out_mode must be one of "
            f"{list(SINGLING_OUT_MODES)}, got {anonymeter.singling_out_mode!r}"
        )
    for key in (
        "n_attacks",
        "singling_out_n_cols",
        "singling_out_max_attempts",
        "linkability_n_neighbors",
    ):
        value = getattr(anonymeter, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"evaluation.privacy_attacks.anonymeter.{key} must be a positive integer, "
                f"got {value!r}"
            )
    if not 0 < anonymeter.confidence_level < 1:
        raise ValueError(
            "evaluation.privacy_attacks.anonymeter.confidence_level must be in (0, 1), "
            f"got {anonymeter.confidence_level!r}"
        )
    if cfg.evaluation.binary_target.enabled:
        bt = cfg.evaluation.binary_target
        if not bt.positive_classes or not bt.negative_classes:
            raise ValueError(
                "evaluation.binary_target.positive_classes and .negative_classes must both be "
                "non-empty when evaluation.binary_target.enabled is true."
            )
        overlap = set(bt.positive_classes) & set(bt.negative_classes)
        if overlap:
            raise ValueError(
                "evaluation.binary_target.positive_classes and .negative_classes must not "
                f"overlap: {sorted(overlap, key=str)}"
            )
    unknown_baselines = [b for b in cfg.evaluation.baselines if b not in EVALUATION_BASELINES]
    if unknown_baselines:
        raise ValueError(
            f"evaluation.baselines must be drawn from {list(EVALUATION_BASELINES)}, "
            f"got {unknown_baselines}"
        )
    rank_weight_keys = set(cfg.evaluation.rank_weights)
    if rank_weight_keys != {"utility", "privacy", "fairness"}:
        raise ValueError(
            "evaluation.rank_weights must have exactly keys {'utility', 'privacy', 'fairness'}, "
            f"got {sorted(rank_weight_keys)}"
        )
    negative_weights = {
        k: v
        for k, v in cfg.evaluation.rank_weights.items()
        if not isinstance(v, (int, float)) or v < 0
    }
    if negative_weights:
        raise ValueError(
            f"evaluation.rank_weights values must be non-negative numbers, got {negative_weights}"
        )
