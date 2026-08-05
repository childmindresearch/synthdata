"""Config schema and YAML loader for the synthdata pipeline.

A collaborator only needs to edit a single YAML file (see ``configs/config.yaml``)
to point the whole pipeline (imputation -> generation -> evaluation -> plots) at
their own dataset. All four ``scripts/run_*.py`` entry points load the same
:class:`Config` object via :func:`load_config`.

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
    #: Columns treated as protected/sensitive attributes for fairness evaluation.
    sensitive_columns: list = dataclasses.field(default_factory=list)
    #: Columns to drop entirely before any modeling (e.g. free-text/ID columns).
    drop_columns: list = dataclasses.field(default_factory=list)
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
    nominal_columns: list | None = None
    ordinal_columns: list = dataclasses.field(default_factory=list)
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

    #: Train/test split.
    train_size: float = 0.6667
    stratify: bool = True

    #: Where cached/derived CSVs (raw, imputed, train/test splits) are written.
    data_dir: str = "data/dataset"
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


@dataclasses.dataclass
class RefiDiffBenchmarkHPOConfig:
    """Narrow, staged search space for masked-cell RefiDiff validation."""

    enabled: bool = False
    n_trials: int = 12
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
    """

    enabled: bool = False
    output_dir: str = "output/dataset/imputation"
    mask_fraction: float = 0.3
    n_masks: int = 3
    mechanisms: list = dataclasses.field(default_factory=lambda: ["mcar"])
    #: Optional feature columns eligible for artificial masking/scoring. All
    #: feature columns remain visible to the imputer as context. ``None`` uses
    #: every non-sensitive feature; a small explicit panel is appropriate for
    #: an affordable screening study on a very wide dataset.
    score_columns: list | None = None
    hpo: RefiDiffBenchmarkHPOConfig = dataclasses.field(default_factory=RefiDiffBenchmarkHPOConfig)


@dataclasses.dataclass
class ImputationConfig:
    enabled: bool = True
    #: "tabimpute" (default, TabPFN-based) or "refidiff" (predictive+diffusion
    #: hybrid; better suited to wide datasets where tabimpute's one-hot
    #: categorical encoding OOMs -- see synthdata/imputation/refidiff_backend.py).
    method: str = "tabimpute"
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
    #: (categorical roles, ordinal orders, method,
    #: round_rules, round_to_int_default, refidiff params) against the sidecar
    #: ``.imputation_cache_key.json`` written alongside the cached CSVs, so
    #: editing e.g. ``data.nominal_columns``/``data.ordinal_columns`` and
    #: rerunning correctly retrains instead of silently reusing stale imputed
    #: data (see synthdata.imputation.pipeline.run_imputation).
    cache: bool = True
    #: Fractional margin used when validating imputed continuous values fall within range.
    validation_margin: float = 0.2
    #: Only used when method == "refidiff".
    refidiff: RefiDiffConfig = dataclasses.field(default_factory=RefiDiffConfig)
    #: Optional train-only artificial-masking benchmark/HPO for RefiDiff.
    benchmark: RefiDiffBenchmarkConfig = dataclasses.field(default_factory=RefiDiffBenchmarkConfig)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SynthcityModelsConfig:
    enabled: bool = True
    names: list = dataclasses.field(
        default_factory=lambda: [
            "ctgan",
            "tvae",
            "adsgan",
            "bayesian_network",
            "pategan",
            "rtvae",
            "ddpm",
        ]
    )


@dataclasses.dataclass
class TabPFNConfig:
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


@dataclasses.dataclass
class TabPFGenConfig:
    enabled: bool = True
    #: "standard" (TabPFGen defaults) and/or "custom" (SGLD + nearest-neighbor relabeling).
    variants: list = dataclasses.field(default_factory=lambda: ["standard", "custom"])
    #: kwargs passed to TabPFGen() for the non-HPO "standard" variant (empty = library defaults).
    standard_params: dict = dataclasses.field(default_factory=dict)
    #: kwargs passed to TabPFGenSGLDLabels() for the non-HPO "custom" variant.
    custom_params: dict = dataclasses.field(
        default_factory=lambda: {"n_sgld_steps": 1000, "sgld_noise_scale": 0.1}
    )


@dataclasses.dataclass
class HPOConfig:
    enabled: bool = True
    n_trials: int = 10
    timeout_seconds: int | None = 300
    #: Hard cap on generator training iterations during search (speed/quality tradeoff).
    n_iter_cap: int = 300
    #: Per-model overrides of n_iter_cap (e.g. pategan trains much slower per iteration).
    model_iter_caps: dict = dataclasses.field(default_factory=lambda: {"pategan": 50})
    #: Cap on TabPFGen custom variant's SGLD step count during search.
    sgld_step_cap: int = 500
    #: Composite objective: metrics oriented to "higher is better" and averaged.
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
    n_samples: int = 200
    #: Base artifact root. Runtime stage paths are versioned under
    #: ``<output_dir>/<data.version or 'unversioned'>/<experiment-id>/``.
    output_dir: str = "output/dataset/synthetic_data"
    force_retrain: bool = False
    synthcity: SynthcityModelsConfig = dataclasses.field(default_factory=SynthcityModelsConfig)
    tabpfn: TabPFNConfig = dataclasses.field(default_factory=TabPFNConfig)
    tabpfgen: TabPFGenConfig = dataclasses.field(default_factory=TabPFGenConfig)
    hpo: HPOConfig = dataclasses.field(default_factory=HPOConfig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FrameworkSelectionConfig:
    """Partial-selection controls for one evaluation framework.

    ``metrics`` (explicit metric names) takes precedence over ``categories``
    (utility/privacy/... groupings) when both are given.
    """

    enabled: bool = True
    categories: list | None = None
    metrics: list | None = None


@dataclasses.dataclass
class LogDisparityConfig:
    #: Defaults to data.sensitive_columns if left empty.
    protected_columns: list = dataclasses.field(default_factory=list)
    target_map: dict | None = None
    protected_map: list | None = None
    protected_bins: list | None = None


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

    enabled: bool = False
    #: Column to collapse; defaults to data.target_column if left None.
    column: str | None = None
    #: Original class values mapped to the binary "positive" (1) outcome.
    positive_classes: list = dataclasses.field(default_factory=list)
    #: Original class values mapped to the binary "negative"/reference (0) outcome.
    negative_classes: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PrivacyGateConfig:
    """Absolute (not merely relative-to-other-models) privacy safety floor.

    Unlike the ranked/scaled columns in the combined evaluation table (which
    only say "better/worse than the other candidate models in this run" via
    per-metric min-max scaling), this checks each model's RAW metric value
    against a fixed threshold, so a model can't look "best on privacy" by
    comparison alone while still leaking an unacceptable absolute amount.
    Gate failures are surfaced (a ``privacy_gate_pass``/
    ``privacy_gate_violations`` column pair in the combined table, plus a
    WARNING log line) but never silently remove a model from the ranked
    table -- see :mod:`synthdata.evaluation.privacy_gate`.

    ``thresholds`` maps a metric's exact result-column name (as it appears in
    the combined table -- e.g. ``"mia_recall"`` or
    ``"privacy.identifiability_score.score_OC"``) to
    ``{"bound": "max"|"min", "value": <float>}``. ``"max"`` means the metric's
    raw value must be ``<= value`` to pass; ``"min"`` means it must be
    ``>= value`` to pass. A metric not computed this run (selection/failure)
    is excluded from the gate check (logged), never silently treated as a pass.

    CAUTION: the defaults below are reasonable *starting points* (grounded in
    "meaningfully above chance/baseline"), NOT validated against any specific
    regulatory standard (e.g. HIPAA Safe Harbor/Expert Determination) -- get a
    domain/compliance sign-off before treating this as a real go/no-go gate
    for an actual data release or challenge submission.
    """

    enabled: bool = True
    thresholds: dict = dataclasses.field(
        default_factory=lambda: {
            # syntheval metrics (exact result-column names -- see catalog.py /
            # syntheval_eval.py's normalize_output-derived column names).
            "mia_recall": {"bound": "max", "value": 0.6},  # chance level ~0.5
            "mia_precision": {"bound": "max", "value": 0.6},  # chance level ~0.5
            "hit_rate": {"bound": "max", "value": 0.05},  # >5% near-duplicate rate
            "att_discl_risk": {"bound": "max", "value": 0.6},
            # synthcity metrics (dotted "category.metric.subkey" names).
            "privacy.identifiability_score.score_OC": {"bound": "max", "value": 0.3},
            "privacy.k-anonymization.syn": {"bound": "min", "value": 5.0},
            "privacy.k-map.score": {"bound": "min", "value": 5.0},
        }
    )


@dataclasses.dataclass
class SynthEvalExecutionConfig:
    """Resource policy for resumable per-model SynthEval evaluation.

    ``model_workers`` may be ``"auto"`` or an explicit positive integer.
    Automatic mode derives a safe bound from CPU count, available memory, and
    dataset width; the remaining fields constrain that estimate.
    """

    model_workers: str | int = "auto"
    max_model_workers: int = 8
    cores_per_model: int = 4
    memory_reserve_gib: float = 16.0
    #: Optional fixed estimate; automatic mode derives one from feature width when None.
    memory_per_model_gib: float | None = None


@dataclasses.dataclass
class EvaluationConfig:
    #: Base artifact root. Runtime stage paths are versioned under
    #: ``<output_dir>/<data.version or 'unversioned'>/<experiment-id>/``.
    output_dir: str = "output/dataset/evaluation"
    #: Restrict evaluation to a subset of generated model names (None = all found on disk).
    models: list | None = None
    positive_class: Any = 1

    synthcity: FrameworkSelectionConfig = dataclasses.field(
        default_factory=FrameworkSelectionConfig
    )
    syntheval: FrameworkSelectionConfig = dataclasses.field(
        default_factory=FrameworkSelectionConfig
    )
    custom: FrameworkSelectionConfig = dataclasses.field(default_factory=FrameworkSelectionConfig)

    #: "linear" (min-max scale + sum) or "summation" (SynthEval's built-in strategy).
    ranking_strategy: str = "linear"
    log_disparity: LogDisparityConfig = dataclasses.field(default_factory=LogDisparityConfig)
    save_per_model_syntheval_plots: bool = True
    binary_target: BinaryTargetConfig = dataclasses.field(default_factory=BinaryTargetConfig)
    syntheval_execution: SynthEvalExecutionConfig = dataclasses.field(
        default_factory=SynthEvalExecutionConfig
    )

    #: Per-"type" (utility/privacy/fairness) weight applied when rolling up
    #: type-level ranks into the overall rank (see
    #: synthdata.evaluation.combine.build_combined_table). Keys must be
    #: exactly {"utility","privacy","fairness"}; values must be non-negative.
    #: Default is equal weight -- ``privacy_gate`` (pass/fail) below is this
    #: project's primary safeguard for sensitive data, not this weight; raise
    #: "privacy" here too if you also want privacy to influence relative
    #: ranking among gate-passing models.
    rank_weights: dict = dataclasses.field(
        default_factory=lambda: {"utility": 1.0, "privacy": 1.0, "fairness": 1.0}
    )
    privacy_gate: PrivacyGateConfig = dataclasses.field(default_factory=PrivacyGateConfig)
    #: Whether to generate a human-readable Markdown evaluation report
    #: (report.md, alongside combined_evaluation.csv) summarizing the ranked
    #: table, privacy gate results, and a recommended model.
    generate_report: bool = True


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
    dpi: int = 150
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
    seed: int = 42
    #: "auto" | "cpu" | "cuda" | "mps"
    device: str = "auto"

    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    imputation: ImputationConfig = dataclasses.field(default_factory=ImputationConfig)
    generation: GenerationConfig = dataclasses.field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
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
    (Config, "experiment"): ExperimentConfig,
    (ImputationConfig, "refidiff"): RefiDiffConfig,
    (ImputationConfig, "benchmark"): RefiDiffBenchmarkConfig,
    (RefiDiffBenchmarkConfig, "hpo"): RefiDiffBenchmarkHPOConfig,
    (GenerationConfig, "synthcity"): SynthcityModelsConfig,
    (GenerationConfig, "tabpfn"): TabPFNConfig,
    (GenerationConfig, "tabpfgen"): TabPFGenConfig,
    (GenerationConfig, "hpo"): HPOConfig,
    (EvaluationConfig, "synthcity"): FrameworkSelectionConfig,
    (EvaluationConfig, "syntheval"): FrameworkSelectionConfig,
    (EvaluationConfig, "custom"): FrameworkSelectionConfig,
    (EvaluationConfig, "log_disparity"): LogDisparityConfig,
    (EvaluationConfig, "binary_target"): BinaryTargetConfig,
    (EvaluationConfig, "syntheval_execution"): SynthEvalExecutionConfig,
    (EvaluationConfig, "privacy_gate"): PrivacyGateConfig,
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


def _validate(cfg: Config) -> None:
    if cfg.data.source not in ("uci", "csv", "parquet"):
        raise ValueError(f"data.source must be 'uci', 'csv', or 'parquet', got {cfg.data.source!r}")
    if cfg.data.source == "uci" and cfg.data.uci_id is None:
        raise ValueError("data.uci_id is required when data.source == 'uci'")
    if cfg.data.source in ("csv", "parquet") and not cfg.data.path:
        raise ValueError("data.path is required when data.source == 'csv'/'parquet'")
    if not cfg.data.target_column:
        raise ValueError("data.target_column must be set")
    if cfg.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"device must be one of auto/cpu/cuda/mps, got {cfg.device!r}")
    if cfg.imputation.method not in ("tabimpute", "refidiff"):
        raise ValueError(
            f"imputation.method must be 'tabimpute' or 'refidiff', got {cfg.imputation.method!r}"
        )
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
    if cfg.evaluation.ranking_strategy not in ("linear", "summation"):
        raise ValueError(
            "evaluation.ranking_strategy must be 'linear' or 'summation', "
            f"got {cfg.evaluation.ranking_strategy!r}"
        )
    execution = cfg.evaluation.syntheval_execution
    if execution.model_workers != "auto" and (
        not isinstance(execution.model_workers, int) or execution.model_workers < 1
    ):
        raise ValueError(
            "evaluation.syntheval_execution.model_workers must be 'auto' or a positive integer, "
            f"got {execution.model_workers!r}"
        )
    for field_name in ("max_model_workers", "cores_per_model"):
        value = getattr(execution, field_name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"evaluation.syntheval_execution.{field_name} must be a positive integer, "
                f"got {value!r}"
            )
    for field_name in ("memory_reserve_gib", "memory_per_model_gib"):
        value = getattr(execution, field_name)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(
                f"evaluation.syntheval_execution.{field_name} must be a positive number or None, "
                f"got {value!r}"
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
    for metric, spec in cfg.evaluation.privacy_gate.thresholds.items():
        if not isinstance(spec, dict) or "bound" not in spec or "value" not in spec:
            raise ValueError(
                f"evaluation.privacy_gate.thresholds[{metric!r}] must be a dict with 'bound' and "
                f"'value' keys, got {spec!r}"
            )
        if spec["bound"] not in ("max", "min"):
            raise ValueError(
                f"evaluation.privacy_gate.thresholds[{metric!r}]['bound'] must be 'max' or 'min', "
                f"got {spec['bound']!r}"
            )
        if not isinstance(spec["value"], (int, float)):
            raise ValueError(
                f"evaluation.privacy_gate.thresholds[{metric!r}]['value'] must be a number, "
                f"got {spec['value']!r}"
            )
