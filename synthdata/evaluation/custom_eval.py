"""Custom evaluation: log disparity (Bhanot et al. 2021) fairness summary
metrics, holdout train-on-synthetic, test-on-real (TSTR) utility scores, and
privacy attacks (Anonymeter, holdout-referenced DCR/NNDR).

The equalized_odds/equal_opportunity metrics (custom additions to this repo's
SynthEval fork) are *computed* via :mod:`synthdata.evaluation.syntheval_eval`
but re-tagged to framework="custom" downstream in
:mod:`synthdata.evaluation.combine`; this module covers log disparity, which
has no SynthEval equivalent, and holdout TSTR, which reports the
imbalance-aware scores (macro-F1, balanced accuracy, macro AUPRC, per-class
F1) that the libraries' own classifier metrics do not.
"""

import functools

import pandas as pd

from synthdata.compute import run_per_model
from synthdata.config import ComputeConfig
from synthdata.data import Dataset
from synthdata.evaluation.catalog import LOG_DISPARITY_METRICS, TSTR_NAME, resolve_selection
from synthdata.evaluation.privacy_attacks import (
    ANONYMETER_NAME,
    HOLDOUT_DISTANCE_NAME,
    run_privacy_attack_evaluation,
)
from synthdata.evaluation.tstr import TSTRScores, target_classes, tstr_scores
from synthdata.utils import get_logger

logger = get_logger(__name__)

_LOG_DISPARITY_NAME = "log_disparity"

#: Every evaluator ``evaluation.custom`` selects from, with its type.
_CUSTOM_EVALUATORS = {
    _LOG_DISPARITY_NAME: "fairness",
    TSTR_NAME: "utility",
    ANONYMETER_NAME: "privacy",
    HOLDOUT_DISTANCE_NAME: "privacy",
}


def _selected(selection_cfg, name: str) -> bool:
    return name in resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        list(_CUSTOM_EVALUATORS),
        _CUSTOM_EVALUATORS,
    )


def run_log_disparity_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    log_disparity_cfg,
    selection_cfg,
) -> dict[str, dict]:
    """Compute a log-disparity fairness report for every synthetic dataset.

    Returns ``{model_name: report}`` (the full dict from
    ``compute_log_disparity_report``, including the Plotly ``report_figure``),
    or ``{}`` if log_disparity is not in the configured selection.
    """
    if not _selected(selection_cfg, _LOG_DISPARITY_NAME):
        return {}

    from synthdata.log_disparity.metric_log_disparity import compute_log_disparity_report

    protected_cols = log_disparity_cfg.protected_columns or list(dataset.protected_columns)
    if not protected_cols:
        logger.warning("[custom] log_disparity requires protected columns; skipping")
        return {}

    protected_bins = log_disparity_cfg.protected_bins or [
        dataset.protected_bins.get(c) for c in protected_cols
    ]

    # The imputed train split, which generators learn from and the train copy
    # repeats: against the raw split, a missing protected value is a subgroup
    # ("nan") that no imputed dataset can contain, so even a copy of train
    # scored about half its subgroups as misrepresented.
    real = dataset.require_imputed(dataset.train_imputed_df, "train_imputed_df")
    reports = {}
    for name, syn_df in synthetic_datasets.items():
        try:
            reports[name] = compute_log_disparity_report(
                real_data=real,
                synth_data=syn_df,
                target_col=dataset.target_column,
                protected_cols=protected_cols,
                model_name=name,
                target_map=log_disparity_cfg.target_map,
                protected_map=log_disparity_cfg.protected_map,
                protected_bins=protected_bins,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("[custom] log_disparity failed for %s: %s", name, exc)
            reports[name] = {"error": str(exc), "error_type": type(exc).__name__}
    return reports


def build_log_disparity_summary_table(reports: dict[str, dict]) -> pd.DataFrame:
    """Models x {log_disparity_mean_abs, _median_abs, _share_significant, _synthetic_only_row_share}.

    Models whose report failed (see ``run_log_disparity_evaluation``'s
    ``{"error": ...}`` entries) get all-NaN rows here rather than being
    silently dropped, so a failure is still visible in the summary table.
    """
    rows = {}
    for name, report in reports.items():
        if "error" in report:
            rows[name] = {
                "log_disparity_mean_abs": None,
                "log_disparity_median_abs": None,
                "log_disparity_share_significant": None,
                "log_disparity_synthetic_only_row_share": None,
            }
            continue
        stats = report["summary_stats"]
        rows[name] = {
            "log_disparity_mean_abs": stats.get("mean_abs_log_disparity"),
            "log_disparity_median_abs": stats.get("median_abs_log_disparity"),
            "log_disparity_share_significant": stats.get("share_significant_bh"),
            # Rows in subgroups the real data does not have; reported, not ranked.
            "log_disparity_synthetic_only_row_share": stats.get("synthetic_only_row_share"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


#: True => lower is "better" (orient as -value for ranking); mirrors LOG_DISPARITY_METRICS.
LOG_DISPARITY_MINIMIZE = dict(LOG_DISPARITY_METRICS)


#: Share of each synthetic dataset held back to score its TSTS fit.
TSTS_SCORE_FRACTION = 0.2


def tstr_and_tsts_scores(
    synthetic_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    nominal_columns: list,
    classes: list,
    seeds: list,
    ordinal: bool = False,
) -> tuple[TSTRScores, TSTRScores | None]:
    """TSTR (fit on all synthetic rows, score on the real test split) and TSTS.

    TSTS (train on synthetic, test on synthetic) fits the same classifier on
    80% of the synthetic rows and scores it on the other 20%, split stratified
    by the target with the first seed (unstratified when a class has fewer
    than two rows). It says how learnable the target is inside the synthetic
    data; next to TSTR it shows whether that signal carries over to real rows.
    None when the synthetic data has too few rows to split.
    """
    from sklearn.model_selection import train_test_split

    tstr = tstr_scores(
        synthetic_df, test_df, target, nominal_columns, classes, seeds, ordinal=ordinal
    )
    counts = synthetic_df[target].value_counts()
    if len(synthetic_df) < 10:
        return tstr, None
    stratify = synthetic_df[target] if len(counts) > 1 and counts.min() >= 2 else None
    fit_df, score_df = train_test_split(
        synthetic_df, test_size=TSTS_SCORE_FRACTION, random_state=seeds[0], stratify=stratify
    )
    tsts = tstr_scores(fit_df, score_df, target, nominal_columns, classes, seeds, ordinal=ordinal)
    return tstr, tsts


def run_tstr_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    n_seeds: int,
    seed: int,
    compute_cfg=None,
) -> dict:
    """Holdout TSTR and TSTS: the fixed XGBoost fitted on each dataset.

    The classifier is the one HPO tunes against (synthdata.evaluation.tstr),
    but scored here on the test rows, which no generator or search ever saw.
    The real-data reference is the ``baseline_train_copy`` row: its TSTR is
    the classifier fitted on real training rows (train on real, test on real),
    on as many rows as each synthetic dataset. TSTS (see
    :func:`tstr_and_tsts_scores`) is reported next to it, unranked.
    Returns ``{"scores": {model: TSTRScores}, "tsts": {model: TSTRScores},
    "classes": [...]}``, or ``{}`` when TSTR is not selected, the target is
    not categorical, or there is no test split. A model whose fit fails is
    logged and left out (its row is NaN in the combined table).
    """
    if not _selected(selection_cfg, TSTR_NAME):
        return {}
    if not dataset.target_is_categorical:
        logger.info("[custom] holdout TSTR needs a categorical target; skipping")
        return {}
    if dataset.test_imputed_df is None or dataset.test_imputed_df.empty:
        logger.warning("[custom] holdout TSTR needs a test split; skipping")
        return {}

    target = dataset.target_column
    train_df = dataset.require_imputed(dataset.train_imputed_df, "train_imputed_df")
    test_df = dataset.test_imputed_df
    ordinal = target in dataset.ordinal_columns
    classes = target_classes(train_df[target], ordinal)
    seeds = [seed + i for i in range(n_seeds)]

    # Models are scored in parallel processes per compute_cfg; a failed one
    # is logged there and left out.
    run = run_per_model(
        functools.partial(tstr_and_tsts_scores, ordinal=ordinal),
        {
            name: (
                syn_df[train_df.columns],
                test_df,
                target,
                dataset.nominal_columns,
                classes,
                seeds,
            )
            for name, syn_df in synthetic_datasets.items()
        },
        compute_cfg or ComputeConfig(workers=1),
        n_columns=train_df.shape[1],
        label="holdout TSTR",
    )
    return {
        "scores": {name: pair[0] for name, pair in run.results.items()},
        "tsts": {name: pair[1] for name, pair in run.results.items() if pair[1] is not None},
        "classes": [str(c) for c in classes],
    }


def build_tstr_table(tstr_result: dict) -> pd.DataFrame:
    """Models x (TSTR macro scores, ``f1_<class>``, TSTS macro scores) table."""
    if not tstr_result:
        return pd.DataFrame()
    tsts = tstr_result.get("tsts") or {}

    def _row(name, s):
        row = {
            "tstr_macro_f1": s.macro_f1,
            "tstr_balanced_accuracy": s.balanced_accuracy,
            "tstr_macro_auprc": s.macro_auprc,
            "tstr_kappa": s.kappa,
            **{f"f1_{c}": v for c, v in s.per_class_f1.items()},
        }
        t = tsts.get(name)
        row.update(
            {
                "tsts_macro_f1": t.macro_f1 if t else float("nan"),
                "tsts_balanced_accuracy": t.balanced_accuracy if t else float("nan"),
                "tsts_macro_auprc": t.macro_auprc if t else float("nan"),
            }
        )
        return row

    table = pd.DataFrame({name: _row(name, s) for name, s in tstr_result["scores"].items()}).T
    table.index.name = "model"
    return table


def run_privacy_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    selection_cfg,
    attacks_cfg,
    seed: int,
    compute_cfg=None,
) -> dict:
    """Anonymeter attacks and holdout-referenced DCR/NNDR, as selected in
    ``evaluation.custom`` (see synthdata.evaluation.privacy_attacks)."""
    return run_privacy_attack_evaluation(
        synthetic_datasets,
        dataset,
        attacks_cfg,
        run_anonymeter=_selected(selection_cfg, ANONYMETER_NAME),
        run_distances=_selected(selection_cfg, HOLDOUT_DISTANCE_NAME),
        seed=seed,
        compute_cfg=compute_cfg,
    )
