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
import math
import re
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataSplitConfig:
    """Deterministic population roles used by new dataset profiles.

    ``data.split`` is intentionally optional on :class:`DataConfig` while the
    historical two-way loader remains readable. New profiles should provide a
    split explicitly; the resulting roles are always named ``train``,
    ``tuning``, and ``final_holdout``.
    """

    mode: str = "row"
    train_fraction: float = 0.60
    tuning_fraction: float = 0.20
    final_holdout_fraction: float = 0.20
    seed: int | None = None
    candidate_count: int = 32
    ratio_tolerance: float = 0.05
    target_balance_tolerance: float = 0.20

    #: Direct source-column identity, a validated local mapping, or an explicit
    #: assertion for data whose rows are already one distinct patient each.
    patient_id_column: str | None = None
    identity_mapping_path: str | None = None
    mapping_row_key_column: str | None = None
    mapping_patient_key_column: str | None = None
    one_row_per_patient: bool = False

    #: Optional explicit encounter label used when balancing grouped roles.
    encounter_label_column: str | None = None
    #: Maximum combined encounter-count and encounter-level target-balance error
    #: accepted for a candidate assignment.
    encounter_balance_tolerance: float = 0.20

    #: Global support floors and per-value overrides. Override maps are resolved
    #: against observed schema values before a split is accepted.
    minimum_class_count: int = 1
    class_count_overrides: dict = dataclasses.field(default_factory=dict)
    minimum_protected_group_count: int = 0
    protected_group_count_overrides: dict = dataclasses.field(default_factory=dict)
    minimum_target_by_protected_group_count: int = 0
    target_by_protected_group_count_overrides: dict = dataclasses.field(default_factory=dict)


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
    #: Direct source-column patient identity required for canonical evaluation.
    patient_id_column: str | None = None
    #: Marks profile as subject to canonical leakage-safe policy validation.
    canonical: bool = False

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
    #: Explicit protected attributes for fairness. ``sensitive_columns`` remains
    #: a compatibility alias for historical configurations.
    protected_columns: list = dataclasses.field(default_factory=list)
    #: Columns used to balance dataset roles and optional labels for binning
    #: their observed values. Both lists are positionally aligned.
    stratification_variables: list = dataclasses.field(default_factory=list)
    stratification_bins: list = dataclasses.field(default_factory=list)
    #: Optional numeric interval labels for protected columns, positionally
    #: aligned with ``protected_columns``. Bounds are lower-inclusive and
    #: upper-exclusive; Age labels use explicit forms such as ``18-30``.
    protected_attribute_bins: list = dataclasses.field(default_factory=list)
    #: Explicit quasi-identifiers for privacy protocols. These are distinct from
    #: protected attributes even when a named protocol intentionally overlaps.
    quasi_identifier_columns: list = dataclasses.field(default_factory=list)
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

    #: New three-role split contract. A configuration must provide this or set
    #: ``legacy_two_role`` explicitly; omission is not an implicit compatibility mode.
    split: DataSplitConfig | None = None
    #: Explicit opt-in for reading historical train/test artifacts. Legacy data
    #: is not eligible for new tuning, final-policy, or release claims.
    legacy_two_role: bool = False

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
    #: "hyperimpute" is canonical; tabimpute/refidiff are deferred legacy
    #: compatibility methods for explicit two-role datasets.
    method: str = "hyperimpute"
    #: Fixed HyperImpute plugin for continuous features; no automated selection.
    continuous_plugin: str = "median"
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
    #: Per-plugin keyword arguments for non-HPO SynthCity generation.
    params: dict = dataclasses.field(default_factory=dict)


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
class StageAScreenConfig:
    """Hard pre-evaluation screens applied to every HPO candidate."""

    #: Minimum count for every observed categorical target value.
    minimum_class_count: int = 1
    #: Minimum count for every observed protected-group value.
    minimum_protected_group_count: int = 1
    #: Minimum count for every observed protected-group/target cell.
    minimum_target_by_protected_group_count: int = 1
    #: Deterministic source relationships, e.g. ``child`` derived from ``parents``.
    dependency_rules: list = dataclasses.field(default_factory=list)


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
    #: The single configured HPO objective. Its optimization direction is derived
    #: from the registered metric contract; privacy/calibration metrics are rejected.
    metric_config: dict = dataclasses.field(
        default_factory=lambda: {
            "canonical_objectives": ["tstr_macro_f1.v1"],
        }
    )

    @property
    def utility_policy(self) -> dict[str, list]:
        """Expose the configured objective in the legacy policy shape."""
        objectives = self.metric_config.get("canonical_objectives", [])
        weight = 1 / len(objectives) if objectives else 0
        return {"metrics": list(objectives), "weights": [weight] * len(objectives)}

    #: Deterministic candidate screens run before any objective metrics.
    stage_a: StageAScreenConfig = dataclasses.field(default_factory=StageAScreenConfig)
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

    ``enabled=False`` disables framework execution and validation entirely;
    selection fields are ignored and no framework evidence is produced.
    ``metrics`` (explicit metric names) takes precedence over ``categories``
    (utility/privacy/... groupings) when both are given.
    """

    enabled: bool = True
    categories: list | None = None
    metrics: list | None = None
    #: Explicit quasi-identifiers for SynthCity attribute-inference attacks.
    #: Empty means those attacks fail closed instead of using every remaining feature.
    quasi_identifier_columns: list = dataclasses.field(default_factory=list)
    #: Categorical attribute-inference score used for baseline-adjusted risk.
    classification_score: str = "balanced_accuracy"
    #: Optional schema kinds for sensitive attack targets: categorical or continuous.
    sensitive_target_types: dict = dataclasses.field(default_factory=dict)
    #: KMeans partitions used by structural privacy proxy screens.
    structural_n_clusters: list = dataclasses.field(default_factory=lambda: [2, 5, 10, 15])
    #: Minimum average rows per cluster before a structural proxy partition runs.
    structural_min_rows_per_cluster: int = 10


@dataclasses.dataclass
class LogDisparityConfig:
    #: Defaults to data.sensitive_columns if left empty.
    protected_columns: list = dataclasses.field(default_factory=list)
    target_map: dict | None = None
    protected_map: list | None = None
    protected_bins: list | None = None


def _removed_setting_error(prefix: str, name: str) -> ValueError:
    return ValueError(
        f"{prefix}.{name} is no longer supported; remove this setting. "
        "It was not consumed by evaluation; there is no replacement control."
    )


class _MigrationConfigMeta(type):
    _config_prefix: str
    _removed_fields: tuple[str, ...]

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        for name in cls._removed_fields:
            if name in kwargs:
                raise _removed_setting_error(cls._config_prefix, name)
        return super().__call__(*args, **kwargs)


class _MigrationConfig(metaclass=_MigrationConfigMeta):
    _config_prefix = ""
    _removed_fields: tuple[str, ...] = ()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._removed_fields:
            raise _removed_setting_error(self._config_prefix, name)
        super().__setattr__(name, value)


@dataclasses.dataclass
class PrivacyPolicyConfig(_MigrationConfig):
    """Support requirements for release privacy screens, not formal guarantees."""

    _config_prefix = "evaluation.privacy_policy"
    _removed_fields = (
        "k_required",
        "l_required",
        "mia_epsilon_repetitions",
        "epsilon_excess_anchor",
        "mia_advantage_anchor",
        "attribute_disclosure_anchor",
    )

    role_population_floor: int = 20
    protected_slice_floor: int = 1


@dataclasses.dataclass
class ScoringPolicyConfig(_MigrationConfig):
    """Fairness anchors retained by evaluation scoring."""

    _config_prefix = "evaluation.scoring_policy"
    _removed_fields = ("bh_alpha", "practical_log_disparity_floor", "valid_comparison_fraction")

    equalized_odds_gap_anchor: float = 0.10
    worst_absolute_log_disparity_anchor: float = 0.69314718056


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

    ``thresholds`` maps an exact contract identity to
    ``{"contract_id": <id>, "emitted_key": <key>, "framework": <name>,
    "bound": "max"|"min", "value": <float>}``. The contract must explicitly
    allow gate use and be operational. A metric not computed this run
    (selection/failure) is excluded from the gate check (logged), never
    silently treated as a pass.

    CAUTION: the defaults below are reasonable *starting points* (grounded in
    "meaningfully above chance/baseline"), NOT validated against any specific
    regulatory standard (e.g. HIPAA Safe Harbor/Expert Determination) -- get a
    domain/compliance sign-off before treating this as a real go/no-go gate
    for an actual data release or challenge submission.
    """

    enabled: bool = False
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
            "release_privacy.v1": {
                "contract_id": "custom.release_privacy.v1",
                "emitted_key": "release_privacy.v1",
                "framework": "custom",
                "bound": "max",
                "value": 0.0,
            },
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
    #: Largest share of real holdout rows with categories absent from train that
    #: classifier-based holdout metrics (cls_acc, auroc_diff, mia, att_discl)
    #: resolve by mapping to the train mode. Above it those metrics are blocked
    #: as a material distribution shift; 0 always blocks.
    max_holdout_unknown_row_fraction: float = 0.05


@dataclasses.dataclass
class EvaluationConfig(_MigrationConfig):
    _config_prefix = "evaluation"
    _removed_fields = ("rank_weights",)

    #: Base artifact root. Runtime stage paths are versioned under
    #: ``<output_dir>/<data.version or 'unversioned'>/<experiment-id>/``.
    output_dir: str = "output/dataset/evaluation"
    #: Restrict evaluation to a subset of generated model names (None = all found on disk).
    models: list | None = None
    positive_class: Any = 1
    #: Evaluation population: ordinary row-level metrics, or a patient/group
    #: context that requires group-safe metric contracts before policy use.
    group_mode: str = "row"
    #: Identifier column required when ``group_mode == "patient_group"``.
    group_column: str | None = None

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
    syntheval_execution: SynthEvalExecutionConfig = dataclasses.field(
        default_factory=SynthEvalExecutionConfig
    )

    privacy_gate: PrivacyGateConfig = dataclasses.field(default_factory=PrivacyGateConfig)
    privacy_policy: PrivacyPolicyConfig = dataclasses.field(default_factory=PrivacyPolicyConfig)
    scoring_policy: ScoringPolicyConfig = dataclasses.field(default_factory=ScoringPolicyConfig)
    #: Whether to generate a human-readable Markdown evaluation report
    #: (report.md, alongside combined_evaluation.csv) summarizing the ranked
    #: table, privacy gate results, and a recommended model.
    generate_report: bool = True
    #: Final SynthEval is a mandatory audit pass, even when ordinary selection is disabled.
    final_syntheval_mandatory: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject attempts to restore the removed binary-target setting."""
        if name == "binary_target":
            raise AttributeError("evaluation.binary_target is no longer configurable")
        super().__setattr__(name, value)


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
        if issubclass(cls, _MigrationConfig) and key in cls._removed_fields:
            raise _removed_setting_error(cls._config_prefix, key)
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
    (HPOConfig, "stage_a"): StageAScreenConfig,
    (EvaluationConfig, "synthcity"): FrameworkSelectionConfig,
    (EvaluationConfig, "syntheval"): FrameworkSelectionConfig,
    (EvaluationConfig, "custom"): FrameworkSelectionConfig,
    (EvaluationConfig, "log_disparity"): LogDisparityConfig,
    (EvaluationConfig, "syntheval_execution"): SynthEvalExecutionConfig,
    (EvaluationConfig, "privacy_gate"): PrivacyGateConfig,
    (EvaluationConfig, "privacy_policy"): PrivacyPolicyConfig,
    (EvaluationConfig, "scoring_policy"): ScoringPolicyConfig,
    (DataConfig, "split"): DataSplitConfig,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    evaluation = raw.get("evaluation", {}) if isinstance(raw, dict) else {}
    if isinstance(evaluation, dict):
        removed = sorted({"release_generalization", "binary_target"} & set(evaluation))
        if removed:
            raise ValueError(
                "Removed evaluation config key(s): "
                + ", ".join(f"evaluation.{key}" for key in removed)
                + "; use data.protected_attribute_bins for release bins"
            )

    cfg = _from_dict(Config, raw)
    cfg.config_path = config_path
    cfg._provided_paths = _provided_paths(raw)
    if (
        cfg.data.canonical
        and cfg.data.split is not None
        and "data.split.patient_id_column" in cfg._provided_paths
    ):
        raise ValueError(
            "Canonical evaluation rejects nested split identity via data.split.patient_id_column; declare only "
            "data.patient_id_column"
        )
    _validate(cfg)
    return cfg


def _provided_paths(raw: dict) -> set[str]:
    """Return dotted YAML paths, allowing canonical validation to fail closed."""
    paths: set[str] = set()

    def visit(value: Any, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            visit(child, path)

    visit(raw)
    return paths


def _validate_nonnegative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


def _validate_support_override_map(value: Any, field_name: str, depth: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping, got {value!r}")
    for key, nested in value.items():
        if depth == 1:
            _validate_nonnegative_integer(nested, f"{field_name}[{key!r}]")
        else:
            _validate_support_override_map(nested, f"{field_name}[{key!r}]", depth - 1)


def _validate_policy_config(cfg: Any) -> None:
    """Validate canonical policy values and named release transformations."""
    privacy = cfg.evaluation.privacy_policy
    scoring = cfg.evaluation.scoring_policy
    for owner, expected_cls in (
        (cfg.evaluation, EvaluationConfig),
        (privacy, PrivacyPolicyConfig),
        (scoring, ScoringPolicyConfig),
    ):
        if isinstance(owner, dict):
            for name in expected_cls._removed_fields:
                if name in owner:
                    raise _removed_setting_error(expected_cls._config_prefix, name)
        if not isinstance(owner, expected_cls):
            raise ValueError(f"{expected_cls._config_prefix} must be {expected_cls.__name__}")
        for name in expected_cls._removed_fields:
            if hasattr(owner, name):
                raise _removed_setting_error(expected_cls._config_prefix, name)
    if cfg.data.canonical:
        provided = getattr(cfg, "_provided_paths", None)
        if provided is None:
            # Directly constructed dataclasses have no YAML omission information;
            # retain legacy unit-test ergonomics while load_config remains fail-closed.
            provided = None
        if provided is None:
            required = set()
        else:
            required = {
                "evaluation.privacy_policy",
                "evaluation.scoring_policy",
                "data.protected_attribute_bins",
                "generation.hpo.metric_config.canonical_objectives",
            }
            required.update(
                f"evaluation.privacy_policy.{name}"
                for name in (
                    "role_population_floor",
                    "protected_slice_floor",
                )
            )
            required.update(
                f"evaluation.scoring_policy.{name}"
                for name in (
                    "equalized_odds_gap_anchor",
                    "worst_absolute_log_disparity_anchor",
                )
            )
            missing = sorted(path for path in required if path not in (provided or set()))
            if missing:
                raise ValueError(
                    "Canonical policy is incomplete; explicitly configure: " + ", ".join(missing)
                )
    if cfg.data.canonical and cfg.evaluation.log_disparity.protected_bins is not None:
        raise ValueError(
            "Canonical evaluation rejects positional evaluation.log_disparity.protected_bins; "
            "declare named column-based data.protected_attribute_bins instead"
        )
    for name in (
        "role_population_floor",
        "protected_slice_floor",
    ):
        value = getattr(privacy, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"evaluation.privacy_policy.{name} must be a positive integer")
    for name in (
        "equalized_odds_gap_anchor",
        "worst_absolute_log_disparity_anchor",
    ):
        value = getattr(scoring, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise ValueError(f"evaluation.scoring_policy.{name} must be a finite positive number")
    if cfg.data.canonical and cfg.generation.hpo.metric_config != {
        "canonical_objectives": ["tstr_macro_f1.v1"]
    }:
        raise ValueError("Canonical HPO metric_config.canonical_objectives must select TSTR only")


def _protected_attribute_bin_intervals(
    protected_columns: list[str],
    value: Any,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Parse aligned age interval labels into the stable runtime declaration shape."""
    field_name = "data.protected_attribute_bins"
    if not isinstance(value, list) or len(value) != len(protected_columns):
        raise ValueError(f"{field_name} must be a list aligned with data.protected_columns")
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    label_patterns = (
        re.compile(r"<([0-9]+)"),
        re.compile(r"([0-9]+)-([0-9]+)"),
        re.compile(r"(?:([0-9]+)\+|>([0-9]+))"),
    )
    for column, labels in zip(protected_columns, value, strict=True):
        if labels is None:
            continue
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError(
                f"{field_name} entry for {column!r} must be null or a non-empty list of labels"
            )
        if len(labels) != len(set(labels)):
            raise ValueError(f"{field_name} labels for {column!r} must not contain duplicates")

        intervals: list[dict[str, Any]] = []
        for index, label in enumerate(labels):
            matched_pattern = None
            match = None
            for pattern_index, pattern in enumerate(label_patterns):
                match = pattern.fullmatch(label)
                if match is not None:
                    matched_pattern = pattern_index
                    break
            if match is None:
                raise ValueError(
                    f"{field_name} label {label!r} for {column!r} must use explicit age interval syntax"
                )
            if matched_pattern == 0:
                if index != 0:
                    raise ValueError(f"{field_name} lower-open label for {column!r} must be first")
                lower, upper = None, int(match.group(1))
            elif matched_pattern == 1:
                lower, upper = int(match.group(1)), int(match.group(2))
                if lower >= upper:
                    raise ValueError(
                        f"{field_name} interval {label!r} for {column!r} has lower >= upper"
                    )
            else:
                if index != len(labels) - 1:
                    raise ValueError(f"{field_name} upper-open label for {column!r} must be last")
                lower, upper = int(match.group(1) or match.group(2)), None
            if index and intervals[-1]["upper"] != lower:
                raise ValueError(
                    f"{field_name} intervals for {column!r} must be ordered and contiguous"
                )
            intervals.append({"label": label, "lower": lower, "upper": upper})
        if intervals[0]["lower"] is not None or intervals[-1]["upper"] is not None:
            raise ValueError(f"{field_name} intervals for {column!r} must cover both open ends")
        result[column] = {"intervals": intervals}
    return result


def _validate_stratification_config(
    data: DataConfig,
    protected_intervals: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    """Validate positional stratification columns and their optional labels."""
    variables = data.stratification_variables
    bins = data.stratification_bins
    if not isinstance(variables, list) or any(
        not isinstance(variable, str) or not variable.strip() for variable in variables
    ):
        raise ValueError("data.stratification_variables must be a list of non-empty column names")
    if len(variables) != len(set(variables)):
        raise ValueError("data.stratification_variables must not contain duplicates")
    if not isinstance(bins, list) or len(variables) != len(bins):
        raise ValueError(
            "data.stratification_bins must be a list aligned with data.stratification_variables"
        )
    for index, labels in enumerate(bins):
        if labels is not None:
            if (
                not isinstance(labels, list)
                or not labels
                or any(not isinstance(label, str) or not label.strip() for label in labels)
            ):
                raise ValueError(
                    f"data.stratification_bins[{index}] must be null or a non-empty list of labels"
                )
            if len(labels) != len(set(labels)):
                raise ValueError(
                    f"data.stratification_bins[{index}] must not contain duplicate labels"
                )
        variable = variables[index]
        declaration = protected_intervals.get(variable)
        if declaration is not None:
            interval_labels = [interval["label"] for interval in declaration["intervals"]]
            if labels != interval_labels:
                raise ValueError(
                    f"data.stratification_bins for {variable!r} must match "
                    "data.protected_attribute_bins labels in order"
                )


def _validate_data_split_config(cfg: DataConfig) -> None:
    split = cfg.split
    if split is None:
        if not isinstance(cfg.legacy_two_role, bool):
            raise ValueError("data.legacy_two_role must be a boolean when data.split is omitted")
        return
    if cfg.legacy_two_role:
        raise ValueError("data.split and data.legacy_two_role are mutually exclusive")
    if not isinstance(split, DataSplitConfig):
        raise ValueError(f"data.split must be a mapping/DataSplitConfig, got {split!r}")
    if split.mode not in {"row", "patient_group"}:
        raise ValueError(f"data.split.mode must be 'row' or 'patient_group', got {split.mode!r}")

    fractions = {
        "train_fraction": split.train_fraction,
        "tuning_fraction": split.tuning_fraction,
        "final_holdout_fraction": split.final_holdout_fraction,
    }
    for field_name, value in fractions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(
                f"data.split.{field_name} must be a number strictly between 0 and 1, got {value!r}"
            )
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(
            "data.split.train_fraction, tuning_fraction, and final_holdout_fraction must sum "
            f"to 1.0, got {sum(fractions.values())!r}"
        )
    if split.seed is not None and (isinstance(split.seed, bool) or not isinstance(split.seed, int)):
        raise ValueError(f"data.split.seed must be an integer or null, got {split.seed!r}")
    if isinstance(split.candidate_count, bool) or not isinstance(split.candidate_count, int):
        raise ValueError(
            f"data.split.candidate_count must be a positive integer, got {split.candidate_count!r}"
        )
    if split.candidate_count < 1:
        raise ValueError(
            f"data.split.candidate_count must be a positive integer, got {split.candidate_count!r}"
        )
    for field_name in (
        "ratio_tolerance",
        "target_balance_tolerance",
        "encounter_balance_tolerance",
    ):
        value = getattr(split, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(
                f"data.split.{field_name} must be a number between 0 and 1, got {value!r}"
            )

    identity_sources = sum(
        value is not None and value != ""
        for value in (split.patient_id_column, split.identity_mapping_path)
    ) + int(split.one_row_per_patient)
    if split.mode == "patient_group" and identity_sources not in (0, 1):
        raise ValueError(
            "data.split patient_group mode requires exactly one of patient_id_column, "
            "identity_mapping_path, or one_row_per_patient=true"
        )
    if split.mode == "row" and identity_sources:
        raise ValueError(
            "data.split identity settings are only valid when data.split.mode='patient_group'"
        )
    for field_name in (
        "patient_id_column",
        "identity_mapping_path",
        "mapping_row_key_column",
        "mapping_patient_key_column",
        "encounter_label_column",
    ):
        value = getattr(split, field_name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"data.split.{field_name} must be a non-empty string or null, got {value!r}"
            )
    if split.identity_mapping_path is not None and (
        split.mapping_row_key_column is None or split.mapping_patient_key_column is None
    ):
        raise ValueError(
            "data.split.mapping_row_key_column and mapping_patient_key_column are required "
            "when identity_mapping_path is configured"
        )
    if split.identity_mapping_path is None and (
        split.mapping_row_key_column is not None or split.mapping_patient_key_column is not None
    ):
        raise ValueError("data.split mapping key columns require identity_mapping_path")

    _validate_nonnegative_integer(split.minimum_class_count, "data.split.minimum_class_count")
    _validate_nonnegative_integer(
        split.minimum_protected_group_count,
        "data.split.minimum_protected_group_count",
    )
    _validate_nonnegative_integer(
        split.minimum_target_by_protected_group_count,
        "data.split.minimum_target_by_protected_group_count",
    )
    _validate_support_override_map(
        split.class_count_overrides,
        "data.split.class_count_overrides",
        depth=1,
    )
    _validate_support_override_map(
        split.protected_group_count_overrides,
        "data.split.protected_group_count_overrides",
        depth=2,
    )
    _validate_support_override_map(
        split.target_by_protected_group_count_overrides,
        "data.split.target_by_protected_group_count_overrides",
        depth=3,
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
    if not isinstance(cfg.data.legacy_two_role, bool):
        raise ValueError(
            f"data.legacy_two_role must be a boolean, got {cfg.data.legacy_two_role!r}"
        )
    for field_name in ("sensitive_columns", "protected_columns", "quasi_identifier_columns"):
        value = getattr(cfg.data, field_name)
        if not isinstance(value, list) or any(not isinstance(column, str) for column in value):
            raise ValueError(f"data.{field_name} must be a list of column names, got {value!r}")
        if len(value) != len(set(value)):
            raise ValueError(f"data.{field_name} must not contain duplicate columns")
    target_role_overlap = {
        role
        for role, columns in (
            ("protected_columns", cfg.data.protected_columns),
            ("sensitive_columns", cfg.data.sensitive_columns),
            ("quasi_identifier_columns", cfg.data.quasi_identifier_columns),
        )
        if cfg.data.target_column in columns
    }
    if target_role_overlap:
        raise ValueError(
            "data.target_column must not overlap protected_columns, sensitive_columns, or "
            f"quasi_identifier_columns: {sorted(target_role_overlap)}"
        )
    _validate_data_split_config(cfg.data)
    if cfg.data.canonical:
        if (
            not isinstance(cfg.data.patient_id_column, str)
            or not cfg.data.patient_id_column.strip()
        ):
            raise ValueError(
                "Canonical evaluation requires data.patient_id_column as a direct source column; "
                "mapping sidecars and one_row_per_patient are not accepted"
            )
        if cfg.data.split is None:
            raise ValueError("Canonical evaluation requires data.split with patient_group mode")
        if cfg.data.split.mode != "patient_group":
            raise ValueError(
                "Canonical evaluation rejects row split mode and legacy identity settings; "
                "use patient_group"
            )
        split_identity = cfg.data.split
        if (
            split_identity.identity_mapping_path is not None
            or split_identity.one_row_per_patient
            or split_identity.patient_id_column is not None
            or split_identity.mapping_row_key_column is not None
            or split_identity.mapping_patient_key_column is not None
        ):
            raise ValueError(
                "Canonical evaluation rejects nested split identity, mapping sidecars, and "
                "one_row_per_patient; use direct data.patient_id_column"
            )
        if cfg.evaluation.group_mode != "row" or cfg.evaluation.group_column is not None:
            raise ValueError(
                "Canonical evaluation rejects evaluation.group_mode/group_column; patient "
                "identity is configured only by data.patient_id_column"
            )
        qi = set(cfg.data.quasi_identifier_columns)
        forbidden_qi = qi & (
            set(cfg.data.sensitive_columns) | {cfg.data.target_column, cfg.data.patient_id_column}
        )
        if forbidden_qi:
            raise ValueError(
                "data.quasi_identifier_columns must not overlap sensitive_columns, target, or "
                f"patient ID: {sorted(forbidden_qi)}"
            )
    elif cfg.data.patient_id_column is not None:
        raise ValueError(
            "data.patient_id_column is reserved for canonical profiles; set data.canonical=true "
            "or remove it from this legacy/noncanonical profile"
        )
    if cfg.data.drop_columns is None:
        cfg.data.drop_columns = []
    drop_columns = cfg.data.drop_columns
    if not isinstance(drop_columns, list) or any(
        not isinstance(column, str) for column in drop_columns
    ):
        raise ValueError(f"data.drop_columns must be a list of column names, got {drop_columns!r}")
    if len(drop_columns) != len(set(drop_columns)):
        raise ValueError("data.drop_columns must not contain duplicate columns")
    split = cfg.data.split
    identity_columns = (
        {
            column
            for column in (
                split.patient_id_column,
                split.mapping_row_key_column,
                split.mapping_patient_key_column,
            )
            if column is not None
        }
        if split is not None
        else set()
    )
    if cfg.data.patient_id_column is not None:
        identity_columns.add(cfg.data.patient_id_column)
    declared_columns = (
        set(cfg.data.protected_columns)
        | set(cfg.data.sensitive_columns)
        | set(cfg.data.quasi_identifier_columns)
    )
    conflict_sets = {
        "target/drop": {cfg.data.target_column} & set(drop_columns),
        "target/identity": {cfg.data.target_column} & identity_columns,
        "declared/identity": declared_columns & identity_columns,
        "declared/drop": declared_columns & set(drop_columns),
        "drop/identity": set(drop_columns) & identity_columns,
    }
    if split is not None and split.encounter_label_column is not None:
        encounter_column = split.encounter_label_column
        conflict_sets.update(
            {
                "encounter/target": {encounter_column, cfg.data.target_column}
                if encounter_column == cfg.data.target_column
                else set(),
                "encounter/identity": {encounter_column} & identity_columns,
                "encounter/drop": {encounter_column} & set(drop_columns),
                "encounter/declared": {encounter_column} & declared_columns,
            }
        )
    conflicts = {name: sorted(values) for name, values in conflict_sets.items() if values}
    if conflicts:
        raise ValueError(
            "Conflicting data column declarations: "
            + "; ".join(f"{name}={values}" for name, values in conflicts.items())
        )
    protected_intervals = _protected_attribute_bin_intervals(
        cfg.data.protected_columns,
        cfg.data.protected_attribute_bins,
    )
    _validate_stratification_config(cfg.data, protected_intervals)
    if cfg.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"device must be one of auto/cpu/cuda/mps, got {cfg.device!r}")
    if cfg.imputation.method not in ("tabimpute", "refidiff", "hyperimpute"):
        raise ValueError(
            f"imputation.method must be 'tabimpute', 'refidiff', or 'hyperimpute', got {cfg.imputation.method!r}"
        )
    if cfg.imputation.continuous_plugin not in ("median", "mean"):
        raise ValueError(
            "imputation.continuous_plugin must be 'median' or 'mean', "
            f"got {cfg.imputation.continuous_plugin!r}"
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
    stage_a = cfg.generation.hpo.stage_a
    for field_name in (
        "minimum_class_count",
        "minimum_protected_group_count",
        "minimum_target_by_protected_group_count",
    ):
        _validate_nonnegative_integer(
            getattr(stage_a, field_name), f"generation.hpo.stage_a.{field_name}"
        )
    if not isinstance(stage_a.dependency_rules, list):
        raise ValueError(
            "generation.hpo.stage_a.dependency_rules must be a list of mappings, "
            f"got {stage_a.dependency_rules!r}"
        )
    for index, rule in enumerate(stage_a.dependency_rules):
        if not isinstance(rule, dict):
            raise ValueError(
                f"generation.hpo.stage_a.dependency_rules[{index}] must be a mapping, got {rule!r}"
            )
        if not isinstance(rule.get("child"), str) or not rule["child"].strip():
            raise ValueError(
                f"generation.hpo.stage_a.dependency_rules[{index}].child must be a non-empty string"
            )
        parents = rule.get("parents")
        if (
            not isinstance(parents, list)
            or not parents
            or any(not isinstance(parent, str) or not parent.strip() for parent in parents)
        ):
            raise ValueError(
                f"generation.hpo.stage_a.dependency_rules[{index}].parents must be a non-empty "
                "list of strings"
            )
    metric_config = cfg.generation.hpo.metric_config
    if not isinstance(metric_config, dict) or set(metric_config) != {"canonical_objectives"}:
        raise ValueError("generation.hpo.metric_config must declare only canonical_objectives")
    objectives = metric_config["canonical_objectives"]
    if (
        not isinstance(objectives, list)
        or not objectives
        or any(not isinstance(metric, str) or not metric.strip() for metric in objectives)
    ):
        raise ValueError(
            "generation.hpo.metric_config.canonical_objectives must be a non-empty list of metric identities"
        )
    if len(objectives) != len(set(objectives)):
        raise ValueError(
            "generation.hpo.metric_config.canonical_objectives must not contain duplicates"
        )
    if cfg.evaluation.ranking_strategy not in ("linear", "summation"):
        raise ValueError(
            "evaluation.ranking_strategy must be 'linear' or 'summation', "
            f"got {cfg.evaluation.ranking_strategy!r}"
        )
    if cfg.evaluation.group_mode not in ("row", "patient_group"):
        raise ValueError(
            "evaluation.group_mode must be 'row' or 'patient_group', "
            f"got {cfg.evaluation.group_mode!r}"
        )
    if cfg.evaluation.group_column is not None and (
        not isinstance(cfg.evaluation.group_column, str) or not cfg.evaluation.group_column.strip()
    ):
        raise ValueError(
            "evaluation.group_column must be a non-empty string or null, "
            f"got {cfg.evaluation.group_column!r}"
        )
    if cfg.evaluation.group_mode == "patient_group" and cfg.evaluation.group_column is None:
        raise ValueError(
            "evaluation.group_column is required when evaluation.group_mode is 'patient_group'"
        )
    if cfg.evaluation.group_column == cfg.data.target_column:
        raise ValueError("evaluation.group_column must not be the target column")
    if cfg.evaluation.group_column in (cfg.data.drop_columns or []):
        raise ValueError(
            "evaluation.group_column must not be listed in data.drop_columns; the identifier "
            "is required to validate patient/group evaluation"
        )
    _validate_policy_config(cfg)
    if cfg.data.canonical and cfg.imputation.method != "hyperimpute":
        raise ValueError(
            "Canonical profiles require imputation.method='hyperimpute'; refidiff is legacy "
            "and unsupported for canonical evaluation"
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
    unknown_fraction = execution.max_holdout_unknown_row_fraction
    if (
        isinstance(unknown_fraction, bool)
        or not isinstance(unknown_fraction, (int, float))
        or not 0.0 <= unknown_fraction <= 1.0
    ):
        raise ValueError(
            "evaluation.syntheval_execution.max_holdout_unknown_row_fraction must be a number "
            f"in [0, 1], got {unknown_fraction!r}"
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
    structural = cfg.evaluation.synthcity
    if structural.classification_score not in {"balanced_accuracy", "macro_f1"}:
        raise ValueError(
            "evaluation.synthcity.classification_score must be 'balanced_accuracy' or "
            f"'macro_f1', got {structural.classification_score!r}"
        )
    if not structural.structural_n_clusters or any(
        not isinstance(value, int) or value < 2 for value in structural.structural_n_clusters
    ):
        raise ValueError(
            "evaluation.synthcity.structural_n_clusters must contain positive integers >= 2, "
            f"got {structural.structural_n_clusters!r}"
        )
    if len(structural.structural_n_clusters) != len(set(structural.structural_n_clusters)):
        raise ValueError("evaluation.synthcity.structural_n_clusters must not contain duplicates")
    if (
        not isinstance(structural.structural_min_rows_per_cluster, int)
        or structural.structural_min_rows_per_cluster < 1
    ):
        raise ValueError(
            "evaluation.synthcity.structural_min_rows_per_cluster must be a positive integer, "
            f"got {structural.structural_min_rows_per_cluster!r}"
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
        if cfg.evaluation.privacy_gate.enabled and (
            not isinstance(spec.get("contract_id"), str) or not spec["contract_id"].strip()
        ):
            raise ValueError(
                f"evaluation.privacy_gate.thresholds[{metric!r}] must name an explicit "
                "operational contract_id when the privacy gate is enabled"
            )
