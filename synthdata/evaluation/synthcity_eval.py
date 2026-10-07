"""synthcity-based evaluation.

Rather than re-fitting every generator (which ``Benchmarks.evaluate`` does
internally and can be very expensive for GAN/diffusion-style plugins), this
module evaluates the *already-generated* synthetic CSVs from
:mod:`synthdata.generation` with synthcity's ``Metrics.evaluate``, uniformly for
synthcity-native and TabPFN/TabPFGen datasets alike. Every model's synthetic
rows are scored exactly as generated: they are not resampled, so two datasets
that hold the same rows get the same scores.

Which real data a metric compares against depends on what it measures (see
:func:`synthdata.evaluation.catalog.synthcity_metric_uses_held_out`). Fidelity,
copy and nearest-neighbour checks, and the privacy and attack metrics compare
against the real train split, the rows the generator saw; against held-out
rows a generator that memorised train would look private. Train-on-synthetic
performance and DomiasMIA (which contrasts members with non-members) need rows
the generator never saw and use the held-out test split.
"""

import pandas as pd

from synthdata.evaluation.catalog import (
    SYNTHCITY_CATEGORY_TO_TYPE,
    SYNTHCITY_METRIC_CONFIG,
    resolve_selection,
    synthcity_metric_uses_held_out,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)


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


def _split_by_reference(metrics: dict) -> tuple[dict, dict]:
    """Split a synthcity metric config into (train-referenced, held-out-referenced)."""
    on_train: dict = {}
    on_held_out: dict = {}
    for category, names in metrics.items():
        for name in names:
            target = on_held_out if synthcity_metric_uses_held_out(category, name) else on_train
            target.setdefault(category, []).append(name)
    return on_train, on_held_out


def run_synthcity_metrics(
    synthetic_df: pd.DataFrame,
    x_real_held_out: pd.DataFrame,
    x_real_train: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    metrics: dict,
    task_type: str = "classification",
    random_state: int = 42,
    workspace: str | None = None,
) -> pd.DataFrame:
    """Evaluate a cached synthetic DataFrame with synthcity's Metrics.evaluate.

    Runs ``Metrics.evaluate`` once per real reference (see the module
    docstring) and stacks the results:
      X_gt        = real train for fidelity/sanity/privacy/attack metrics;
                    held-out real data for performance and DomiasMIA
      X_syn       = the synthetic rows, as generated
      X_train     = real training data (DomiasMIA's members)
      X_ref_syn   = the same synthetic rows; DomiasMIA_prior, the only DOMIAS
                    variant in the catalog, never reads it
      X_augmented = X_real_train concatenated with X_syn (for augmentation metrics)
    """
    from pathlib import Path

    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader

    x_syn_raw = _align_dtypes(
        synthetic_df[x_real_train.columns].reset_index(drop=True), x_real_train
    )
    x_augmented_raw = pd.concat([x_real_train, x_syn_raw], ignore_index=True)

    def _loader(df: pd.DataFrame):
        return GenericDataLoader(
            df, target_column=target_column, sensitive_features=sensitive_features
        )

    on_train, on_held_out = _split_by_reference(metrics)
    common = {
        "task_type": task_type,
        "random_state": random_state,
        "workspace": Path(workspace) if workspace else Path("workspace"),
    }
    results = []
    if on_train:
        results.append(
            Metrics.evaluate(_loader(x_real_train), _loader(x_syn_raw), metrics=on_train, **common)
        )
    if on_held_out:
        results.append(
            Metrics.evaluate(
                _loader(x_real_held_out),
                _loader(x_syn_raw),
                _loader(x_real_train),
                _loader(x_syn_raw),
                _loader(x_augmented_raw),
                metrics=on_held_out,
                **common,
            )
        )
    combined = pd.concat(results)
    missing = [
        f"{category}.{name}"
        for category, names in metrics.items()
        for name in names
        if not any(key.startswith(f"{category}.{name}.") for key in combined.index)
    ]
    if missing:
        # synthcity catches a metric's exception, logs it at error level on
        # its own logger and leaves the metric out of its results.
        logger.warning(
            "[synthcity] metric(s) failed and are missing from the results: %s "
            "(see synthcity's log for the error)",
            missing,
        )
    return combined


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
        logger.info("[synthcity] evaluating %s", name)
        try:
            results[name] = run_synthcity_metrics(
                syn_df,
                test_df,
                train_df,
                target_column,
                sensitive_features,
                metric_config,
                random_state=seed,
                workspace=workspace,
            )
        except Exception as exc:  # noqa: BLE001 -- one model's failure must not stop the rest
            logger.warning("[synthcity] evaluation failed for %s: %r", name, exc)
            results[name] = pd.DataFrame({"error": [str(exc)], "error_type": [type(exc).__name__]})
    return results
