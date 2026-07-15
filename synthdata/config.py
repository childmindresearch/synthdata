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

    #: "uci" to fetch+cache from the UCI ML repository, or "csv" for a local file.
    source: str = "uci"
    #: UCI dataset id (only used when source == "uci").
    uci_id: int | None = None
    #: Path to a local CSV (only used when source == "csv").
    path: str | None = None

    #: Freeform dataset version label (e.g. "v1", "2024-06-01"). If set, cached
    #: raw/imputed/split CSVs are nested under `data_dir/<version>/` and every
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

    #: "auto" to infer categorical columns (nunique <= auto_categorical_unique_threshold,
    #: or dtype object/category/bool), or an explicit list of column names.
    categorical_columns: str | list = "auto"
    auto_categorical_unique_threshold: int = 10

    #: Uppercase all column names on load (matches the hepatitis notebook convention).
    uppercase_columns: bool = False
    #: Dataset-specific quirk: remap columns whose only non-null values are {1, 2} to {0, 1}.
    remap_binary_one_two: bool = False

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
class ImputationConfig:
    enabled: bool = True
    #: Currently only "tabimpute" is supported.
    method: str = "tabimpute"
    #: "auto" | "cpu" | "cuda" | "mps"
    device: str = "auto"
    #: Optional per-column rounding precision (decimal places) applied post-imputation.
    round_rules: dict = dataclasses.field(default_factory=dict)
    #: If True (default, matches the hepatitis notebook), feature columns not listed in
    #: round_rules are rounded to the nearest integer after imputation. Set to False for
    #: datasets with genuinely continuous features that shouldn't be integer-snapped.
    round_to_int_default: bool = True
    #: Reuse a previously cached imputed CSV if present.
    cache: bool = True
    #: Fractional margin used when validating imputed continuous values fall within range.
    validation_margin: float = 0.2


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
            "privacy": ["identifiability_score", "DomiasMIA_prior"],
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
    output_dir: str = "output/dataset/synthetic_data"
    force_retrain: bool = False
    synthcity: SynthcityModelsConfig = dataclasses.field(
        default_factory=SynthcityModelsConfig
    )
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
class EvaluationConfig:
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
    custom: FrameworkSelectionConfig = dataclasses.field(
        default_factory=FrameworkSelectionConfig
    )

    syntheval_preset: str = "complete_eval"
    #: "linear" (min-max scale + sum) or "summation" (SynthEval's built-in strategy).
    ranking_strategy: str = "linear"
    log_disparity: LogDisparityConfig = dataclasses.field(
        default_factory=LogDisparityConfig
    )
    save_per_model_syntheval_plots: bool = True


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlotsConfig:
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
    `<stage_output_dir>/<experiment_id>/`, and a manifest.json log at
    `<generation_output_dir>/../experiments/<experiment_id>/manifest.json`
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
                f"Unknown config key '{key}' for {cls.__name__}. "
                f"Valid keys: {sorted(field_types)}"
            )
        field_type = field_types[key]
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
    (GenerationConfig, "synthcity"): SynthcityModelsConfig,
    (GenerationConfig, "tabpfn"): TabPFNConfig,
    (GenerationConfig, "tabpfgen"): TabPFGenConfig,
    (GenerationConfig, "hpo"): HPOConfig,
    (EvaluationConfig, "synthcity"): FrameworkSelectionConfig,
    (EvaluationConfig, "syntheval"): FrameworkSelectionConfig,
    (EvaluationConfig, "custom"): FrameworkSelectionConfig,
    (EvaluationConfig, "log_disparity"): LogDisparityConfig,
}


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a :class:`Config`."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _from_dict(Config, raw)
    cfg.config_path = config_path
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.data.source not in ("uci", "csv"):
        raise ValueError(f"data.source must be 'uci' or 'csv', got {cfg.data.source!r}")
    if cfg.data.source == "uci" and cfg.data.uci_id is None:
        raise ValueError("data.uci_id is required when data.source == 'uci'")
    if cfg.data.source == "csv" and not cfg.data.path:
        raise ValueError("data.path is required when data.source == 'csv'")
    if not cfg.data.target_column:
        raise ValueError("data.target_column must be set")
    if cfg.device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"device must be one of auto/cpu/cuda/mps, got {cfg.device!r}")
    if cfg.evaluation.ranking_strategy not in ("linear", "summation"):
        raise ValueError(
            "evaluation.ranking_strategy must be 'linear' or 'summation', "
            f"got {cfg.evaluation.ranking_strategy!r}"
        )
