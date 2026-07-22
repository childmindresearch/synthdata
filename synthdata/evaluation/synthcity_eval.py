"""synthcity-based evaluation.

Rather than re-fitting every generator (which ``Benchmarks.evaluate`` does
internally and can be very expensive for GAN/diffusion-style plugins), this
module evaluates the *already-generated* synthetic CSVs from
:mod:`synthdata.generation` uniformly -- synthcity-native and TabPFN/TabPFGen
datasets alike -- by wrapping each cached DataFrame in a bootstrap-resampling
adapter (``PregeneratedSyntheticModel``). This mirrors the notebook's approach
for external (non-synthcity) generators and generalizes it to every model, so
evaluation is fast, reproducible from cached artifacts, and framework-agnostic.
"""

import pandas as pd

from synthdata.evaluation.catalog import (
    SYNTHCITY_CATEGORY_TO_TYPE,
    SYNTHCITY_METRIC_CONFIG,
    resolve_selection,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)


class PregeneratedSyntheticModel:
    """Wraps an already-generated synthetic DataFrame in a fit/sample API.

    Bootstrap-resamples the cached data with a caller-supplied seed, so that
    independent-looking draws (e.g. for DomiasMIA's reference set) can be
    produced without re-running the original (possibly expensive) generator.
    """

    def __init__(self, synthetic_df: pd.DataFrame):
        self._df = synthetic_df.reset_index(drop=True)

    def fit(self, X):
        return self

    def sample(self, count: int, random_state: int = 0) -> pd.DataFrame:
        return self._df.sample(n=count, replace=True, random_state=random_state).reset_index(
            drop=True
        )


class _ExternalGeneratorAdapter:
    """Minimal adapter for external models exposing fit(X) and sample(count)."""

    def __init__(self, model, random_state: int = 0):
        self.model = model
        self.random_state = random_state

    def fit(self, X):
        self.model.fit(X)
        return self

    def generate(self, count: int, random_state: int | None = None) -> pd.DataFrame:
        seed = random_state if random_state is not None else self.random_state
        x_syn = self.model.sample(count, random_state=seed)
        if not isinstance(x_syn, pd.DataFrame):
            x_syn = pd.DataFrame(x_syn)
        return x_syn


def _align_dtypes(x_syn: pd.DataFrame, x_ref: pd.DataFrame) -> pd.DataFrame:
    for col in x_ref.columns:
        if x_ref[col].dtype != x_syn[col].dtype:
            try:
                x_syn[col] = x_syn[col].astype(x_ref[col].dtype)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "[_align_dtypes] cast failed for column %s (ref dtype=%s, syn dtype=%s): %s",
                    col,
                    x_ref[col].dtype,
                    x_syn[col].dtype,
                    exc,
                )
    return x_syn


def run_synthcity_metrics(
    synthetic_df: pd.DataFrame,
    x_real_reference: pd.DataFrame,
    x_real_train: pd.DataFrame,
    n_samples: int,
    target_column: str,
    sensitive_features: list,
    metrics: dict,
    task_type: str = "classification",
    random_state: int = 42,
    workspace: str | None = None,
) -> pd.DataFrame:
    """Evaluate a cached synthetic DataFrame with synthcity's Metrics.evaluate.

    Mirrors how synthcity's Benchmarks calls Metrics.evaluate internally:
      X_gt        = held-out real data
      X_syn       = synthetic (bootstrap draw, seed=random_state)
      X_train     = real training data (for DomiasMIA)
      X_ref_syn   = second independent synthetic draw (seed=random_state + 1)
      X_augmented = X_real_train concatenated with X_syn (for augmentation metrics)
    """
    from pathlib import Path

    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    adapter = _ExternalGeneratorAdapter(
        PregeneratedSyntheticModel(synthetic_df), random_state=random_state
    )
    adapter.fit(x_real_train)

    x_syn_raw = _align_dtypes(
        adapter.generate(n_samples, random_state=random_state)[x_real_reference.columns],
        x_real_reference,
    )
    x_ref_syn_raw = _align_dtypes(
        adapter.generate(n_samples, random_state=random_state + 1)[x_real_reference.columns],
        x_real_reference,
    )
    x_augmented_raw = pd.concat([x_real_train, x_syn_raw], ignore_index=True)

    def _loader(df: pd.DataFrame):
        return GenericDataLoader(
            df, target_column=target_column, sensitive_features=sensitive_features
        )

    return Metrics.evaluate(
        _loader(x_real_reference),
        _loader(x_syn_raw),
        _loader(x_real_train),
        _loader(x_ref_syn_raw),
        _loader(x_augmented_raw),
        metrics=metrics,
        task_type=task_type,
        random_state=random_state,
        workspace=Path(workspace) if workspace else Path("workspace"),
    )


def resolve_metric_config(selection_cfg) -> dict:
    """Filter SYNTHCITY_METRIC_CONFIG down to the configured selection."""
    all_names = [n for names in SYNTHCITY_METRIC_CONFIG.values() for n in names]
    name_to_type = {
        n: SYNTHCITY_CATEGORY_TO_TYPE[cat]
        for cat, names in SYNTHCITY_METRIC_CONFIG.items()
        for n in names
    }
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        all_names,
        name_to_type,
    )
    metric_config = {
        cat: [n for n in names if n in selected] for cat, names in SYNTHCITY_METRIC_CONFIG.items()
    }
    return {cat: names for cat, names in metric_config.items() if names}


def run_synthcity_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    selection_cfg,
    n_samples: int | None = None,
    seed: int = 42,
    workspace: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run synthcity Metrics on every cached synthetic dataset.

    Returns ``{model_name: DataFrame}`` where each DataFrame is indexed by
    metric key (e.g. ``"stats.wasserstein_dist.joint"``) with at least
    ``mean``/``direction`` columns, as returned by ``Metrics.evaluate``.
    """
    metric_config = resolve_metric_config(selection_cfg)
    if not metric_config:
        logger.info("[synthcity] no metrics selected; skipping")
        return {}

    results = {}
    for name, syn_df in synthetic_datasets.items():
        n = n_samples or len(syn_df)
        logger.info("[synthcity] evaluating %s", name)
        try:
            results[name] = run_synthcity_metrics(
                syn_df,
                test_df,
                train_df,
                n,
                target_column,
                sensitive_features,
                metric_config,
                random_state=seed,
                workspace=workspace,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("[synthcity] evaluation failed for %s: %s", name, exc)
            results[name] = pd.DataFrame({"error": [str(exc)], "error_type": [type(exc).__name__]})
    return results
