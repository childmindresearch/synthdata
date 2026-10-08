"""``report.md``: the one page to open after a run.

It summarizes the outcome first (recommended model, how it compares with the
baselines, what to be careful about), then walks through ranking, utility,
privacy evidence, fairness and data preparation, embedding every plot that
exists and linking every table behind it. A file index at the end lists every
output of the run. ``synthdata-evaluate`` writes it and ``synthdata-plot``
rewrites it once the plots exist; tables the caller did not pass are read back
from the evaluation directory, so both writes give the same page.
"""

import os
from pathlib import Path

import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.evaluation.baselines import is_baseline
from synthdata.evaluation.combine import simple_rank_summary, summarize_replicates
from synthdata.evaluation.privacy_attacks import ANONYMETER_METRICS, HOLDOUT_DISTANCE_METRICS
from synthdata.experiment import dataset_version_scope
from synthdata.utils import get_logger

logger = get_logger(__name__)

_IMAGE_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg")
_DIMS = ("overall", "utility", "privacy", "fairness")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_metric(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a flat (single-level-column) DataFrame as a Markdown table.

    Avoids depending on the optional ``tabulate`` package that
    ``pandas.DataFrame.to_markdown`` requires (not part of this repo's core
    dependency set).
    """
    if df.empty:
        return "_(no data)_"
    headers = [str(c) for c in df.columns]

    def _fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}" if pd.notna(value) else "NaN"
        return str(value)

    rows = [[_fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


class _Links:
    """Builds links relative to the directory ``report.md`` is written to."""

    def __init__(self, report_dir: Path):
        self.report_dir = report_dir

    def rel(self, path: Path) -> str:
        return Path(os.path.relpath(path, self.report_dir)).as_posix()

    def file(self, path: Path, label: str | None = None) -> str:
        return f"[{label or path.name}]({self.rel(path)})"

    def maybe(self, path: Path, label: str | None = None) -> str:
        """A link when ``path`` exists, else the name marked as not produced."""
        if path.exists():
            return self.file(path, label)
        return f"`{label or path.name}` (not produced this run)"

    def plots(self, directory: Path, pattern: str = "*") -> list[str]:
        """Images in ``directory`` embedded inline, interactive HTML pages linked."""
        if not directory.is_dir():
            return []
        lines = []
        for path in sorted(directory.glob(pattern)):
            title = path.stem.replace("_", " ")
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                lines += [f"**{title}**", "", f"![{title}]({self.rel(path)})", ""]
            elif path.suffix.lower() == ".html":
                lines += [f"- {self.file(path, title + ' (interactive)')}", ""]
        return lines


class _Paths:
    """Where each stage of this run wrote its outputs."""

    def __init__(self, cfg: Config, dataset: Dataset, experiment, report_dir: Path):
        self.evaluation = report_dir
        self.generation = Path(cfg.generation.output_dir)
        self.plots = Path(cfg.plots.output_dir)
        self.evaluation_plots = self.plots / "evaluation"
        experiment_plots = getattr(experiment, "plots_dir", None)
        if experiment_plots is not None:
            self.dataset_plots = Path(experiment_plots).parent / "dataset"
        else:
            self.dataset_plots = self.plots / dataset_version_scope(cfg) / "dataset"
        data_dir = getattr(dataset, "data_dir", None)
        self.data = Path(data_dir) if data_dir else None
        manifest = getattr(experiment, "manifest_path", None)
        self.manifest = Path(manifest) if manifest else None


def _load_extras(evaluation_dir: Path, combined: pd.DataFrame, extras: dict) -> dict:
    """Fill tables the caller did not pass from the files evaluation wrote."""
    extras = dict(extras)
    if extras.get("ranking_summary") is None and (
        ("__all__", "overall", "rank") in combined.columns
    ):
        extras["ranking_summary"] = summarize_replicates(combined)
    readers = {
        "tstr_table": ("tstr_holdout.csv", {"index_col": 0}),
        "ovr_per_class": ("ovr_per_class.csv", {"index_col": 0, "header": [0, 1]}),
        "privacy_attacks": ("privacy_attacks.csv", {}),
    }
    for key, (name, kwargs) in readers.items():
        path = evaluation_dir / name
        if extras.get(key) is None and path.exists():
            try:
                extras[key] = pd.read_csv(path, **kwargs)
            except (ValueError, pd.errors.ParserError) as exc:
                logger.warning("[report] could not read %s: %s", path, exc)
    if extras.get("privacy_attacks") is None:
        attacks = (extras.get("privacy_result") or {}).get("attacks")
        if attacks is not None:
            extras["privacy_attacks"] = attacks
    return extras


def _candidates(index) -> list:
    return [m for m in index if not is_baseline(m)]


def _summary(combined: pd.DataFrame, ranking_summary) -> pd.DataFrame:
    """One row per model with ``<dim>_mean`` columns, best first."""
    if ranking_summary is not None and "overall_mean" in ranking_summary.columns:
        return ranking_summary
    flat = simple_rank_summary(combined)
    return flat.rename(columns={d: f"{d}_mean" for d in flat.columns})


def _has_replicates(ranking_summary) -> bool:
    return (
        ranking_summary is not None
        and "overall_mean" in ranking_summary.columns
        and ranking_summary["n_replicates"].max() >= 2
    )


# ---------------------------------------------------------------------------
# 1. At a glance
# ---------------------------------------------------------------------------


def _recommendation(summary: pd.DataFrame) -> tuple[str | None, list]:
    """The best non-baseline model by mean overall score, and the models tied with it."""
    if "overall_mean" not in summary.columns:
        return None, []
    candidates = summary.loc[_candidates(summary.index)].sort_values(
        "overall_mean", ascending=False
    )
    if candidates.empty:
        return None, []
    best = candidates.index[0]
    tied = []
    if "tied_with_best" in candidates.columns:
        tied = [
            m
            for m in candidates.index[1:]
            if pd.notna(candidates.loc[m, "tied_with_best"])
            and bool(candidates.loc[m, "tied_with_best"])
        ]
    return best, tied


def _score_with_interval(row: pd.Series, dim: str) -> str:
    mean = row.get(f"{dim}_mean")
    low, high = row.get(f"{dim}_ci_low"), row.get(f"{dim}_ci_high")
    if low is None or pd.isna(low):
        return _fmt_metric(mean)
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _privacy_flags(attacks) -> list[str]:
    """Anonymeter attacks that measurably beat their baseline, per candidate model."""
    flags = []
    if attacks is not None and not attacks.empty and "ci_low" in attacks.columns:
        hits = attacks[(attacks["ci_low"] > 0) & attacks["reliable"].astype(bool)]
        hits = hits[~hits["model"].map(is_baseline).astype(bool)]
        for model, group in hits.groupby("model", sort=True):
            names = ", ".join(
                sorted(
                    {
                        a if pd.isna(s) or s == "" else f"{a} ({s})"
                        for a, s in zip(group["attack"], group["secret"], strict=True)
                    }
                )
            )
            flags.append(
                f"`{model}`: Anonymeter {names} succeeded more often than its baseline "
                "attack (95% interval above 0)."
            )
    return flags


def _glance_section(cfg, dataset, combined, extras, model_names, experiment) -> str:
    summary = _summary(combined, extras.get("ranking_summary"))
    lines = ["## 1. At a glance", ""]
    meta = [
        f"- **Dataset**: `{dataset.name}`"
        + (f" (version `{dataset.version}`)" if dataset.version else "")
        + f", target `{dataset.target_column}`",
    ]
    if experiment is not None:
        meta.append(f"- **Experiment**: `{experiment.id}`")
    replicates = int(summary["n_replicates"].max()) if "n_replicates" in summary else 1
    meta += [
        f"- **Seed**: `{cfg.seed}`; **seed replicates per model**: {replicates}; "
        f"**synthetic rows per model**: {cfg.generation.n_samples}",
        f"- **Models evaluated ({len(model_names)})**: " + ", ".join(f"`{m}`" for m in model_names),
    ]
    lines += meta + [""]

    best, tied = _recommendation(summary)
    if best is None:
        lines.append("**No candidate model to recommend** (only baselines, or no overall score).")
    else:
        row = summary.loc[best]
        lines.append(
            f"**Recommended model: `{best}`**, overall score "
            f"{_score_with_interval(row, 'overall')} (0 to 1, higher is better; "
            "relative to the other models in this run)."
        )
        if tied:
            lines += [
                "",
                "Tied with it within seed noise: "
                + ", ".join(f"`{m}`" for m in tied)
                + ". Choose among these on cost or on the score that matters most to you.",
            ]
        comparisons = []
        for baseline, role in (
            ("baseline_train_copy", "a copy of the real training rows"),
            ("baseline_marginals", "columns sampled independently"),
        ):
            if baseline in summary.index:
                parts = [
                    f"{dim} {_fmt_metric(summary.loc[baseline].get(f'{dim}_mean'))}"
                    for dim in ("utility", "privacy")
                    if f"{dim}_mean" in summary.columns
                ]
                comparisons.append(f"`{baseline}` ({role}): " + ", ".join(parts))
        if comparisons:
            own = ", ".join(
                f"{dim} {_fmt_metric(row.get(f'{dim}_mean'))}"
                for dim in ("utility", "privacy")
                if f"{dim}_mean" in summary.columns
            )
            lines += [
                "",
                f"Against the baselines: `{best}` scores {own}; "
                + "; ".join(comparisons)
                + ". A useful generator beats the marginals on utility and the train copy on "
                "privacy.",
            ]

    cautions = []
    if not _has_replicates(extras.get("ranking_summary")):
        cautions.append(
            "Single seed: scores carry no uncertainty estimate, so small gaps may be noise. "
            "Set `generation.n_replicates` to 2 or more for confidence intervals."
        )
    raw = combined[[c for c in combined.columns if c[0] != "__all__" and c[2] != "rank"]]
    raw = raw.loc[_candidates(raw.index), raw.notna().any()]
    missing = raw.isna().sum(axis=1)
    for model in missing.index[missing > 0]:
        cautions.append(f"`{model}`: {int(missing[model])} metric(s) missing (failed or skipped).")
    cautions += _privacy_flags(extras.get("privacy_attacks"))
    if cautions:
        lines += ["", "**Read with care**", ""] + [f"- {c}" for c in cautions]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Ranking
# ---------------------------------------------------------------------------


def _ranking_section(cfg, combined, extras, paths: _Paths, links: _Links) -> str:
    ranking_summary = extras.get("ranking_summary")
    lines = ["## 2. Ranking", ""]
    if _has_replicates(ranking_summary):
        dims = [d for d in _DIMS if f"{d}_mean" in ranking_summary]
        rows = []
        for model, row in ranking_summary.iterrows():
            tied = row.get("tied_with_best")
            rows.append(
                {
                    "rank": row["rank"],
                    "model": model,
                    "seeds": row["n_replicates"],
                    **{dim: _score_with_interval(row, dim) for dim in dims},
                    "tied with best": "" if pd.isna(tied) else ("yes" if tied else "no"),
                }
            )
        lines += [
            "Each score is the mean over seed replicates with its 95% confidence interval. "
            '"Tied with best" is yes when a one-sided Welch t-test (5% level) cannot place the '
            "model's overall score below the best candidate's.",
            "",
            _dataframe_to_markdown(pd.DataFrame(rows)),
        ]
        index = ranking_summary.index
    else:
        summary = simple_rank_summary(combined)
        if summary.empty:
            lines.append("No ranking columns were produced.")
        else:
            lines += [
                "Single seed: no confidence intervals.",
                "",
                _dataframe_to_markdown(summary.reset_index()),
            ]
        index = summary.index
    if any(is_baseline(m) for m in index):
        lines += [
            "",
            "Rows named `baseline_*` are fixed references, never recommended. "
            "`baseline_train_copy` is real training data: the utility a generator can at best "
            "reach, and the privacy it must stay well above. `baseline_marginals` samples each "
            "column independently: the utility floor a useful generator must beat.",
        ]
    weights = ", ".join(f"{k} {v:g}" for k, v in cfg.evaluation.rank_weights.items())
    lines += [
        "",
        "**How the scores are built.** Every metric is oriented so higher is better and "
        "min-max scaled across the models of this run, so a score says how a model compares "
        "with the others here, not how good it is in absolute terms. Metrics are averaged "
        "within each framework, then across frameworks into the utility, privacy and fairness "
        f"scores. The overall score is their weighted geometric mean (weights: {weights}; "
        "`evaluation.rank_weights`), so a near-zero score on one dimension cannot be bought "
        "back by the others.",
        "",
        f"Tables: {links.maybe(paths.evaluation / 'ranking_summary.csv')} (one row per model), "
        f"{links.maybe(paths.evaluation / 'combined_evaluation.csv')} (every metric, raw and "
        "per-group scores).",
        "",
    ]
    plots = links.plots(paths.evaluation_plots, "*_vs_*") + links.plots(
        paths.evaluation_plots, "rank_tradeoff*"
    )
    lines += plots or ["_Trade-off plots not rendered yet: run `synthdata-plot`._"]
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# 3. Utility
# ---------------------------------------------------------------------------


def _utility_section(cfg, extras, paths: _Paths, links: _Links) -> str:
    lines = ["## 3. Utility", "", "### Real vs synthetic distributions", ""]
    plots = links.plots(paths.plots / "generation")
    lines += plots or ["_Not rendered yet: run `synthdata-plot` with the `generation` section._"]
    lines += ["", "### Train on synthetic, test on real (holdout, imbalance-aware)", ""]
    tstr_table = extras.get("tstr_table")
    if tstr_table is not None and not tstr_table.empty:
        table = tstr_table.reset_index()
        table = table.rename(columns={table.columns[0]: "model"})
        lines += [
            "A fixed XGBoost classifier is fitted on each dataset and scored on the held-out "
            "test split. Macro-F1 and balanced accuracy weigh every class equally, so a minority "
            "class the synthetic data gets wrong pulls them down; `f1_<class>` shows which "
            "class. The `trtr (real train)` row is the same classifier fitted on the real "
            "training rows: the ceiling.",
            "",
            _dataframe_to_markdown(table),
            "",
            f"Table: {links.file(paths.evaluation / 'tstr_holdout.csv')}",
        ]
    else:
        lines.append("Holdout TSTR was not computed this run.")
    ovr = extras.get("ovr_per_class")
    if cfg.evaluation.class_averaging == "ovr_macro" and ovr is not None and not ovr.empty:
        flat = ovr.copy()
        flat.columns = [f"{metric} [{cls}]" for metric, cls in flat.columns]
        flat.index.name = "model"
        lines += [
            "",
            "SynthEval metrics that need two classes (AUROC difference, subgroup gaps) ran once "
            "per class against the rest and are averaged with equal weight per class. Values "
            "per class:",
            "",
            _dataframe_to_markdown(flat.reset_index()),
            "",
            f"Table: {links.maybe(paths.evaluation / 'ovr_per_class.csv')}",
        ]
    elif cfg.evaluation.class_averaging == "binary" and cfg.evaluation.binary_target.enabled:
        bt = cfg.evaluation.binary_target
        lines += [
            "",
            "SynthEval metrics that need two classes ran on the target collapsed to "
            f"positive {bt.positive_classes} versus negative {bt.negative_classes}.",
        ]
    native = paths.evaluation_plots / "syntheval_plots"
    if native.is_dir():
        lines += [
            "",
            f"SynthEval's own per-metric diagnostic plots: {links.file(native, 'folder')}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Privacy evidence
# ---------------------------------------------------------------------------


def _privacy_section(cfg, combined, extras, paths: _Paths, links: _Links) -> str:
    lines = [
        "## 4. Privacy evidence",
        "",
        "These are empirical attacks and distance checks, not guarantees. A low risk here "
        "means these particular attacks, with these settings and this many records, did not "
        "find a leak; it does not prove that no other attack would. Anonymeter's risk "
        "estimates in particular depend on the attack configuration and sample size and have "
        "been criticised as unreliable proof of safety. No threshold is applied: compare each "
        "model with `baseline_train_copy` (the worst case, real rows released as is) and with "
        "the other models, and get a privacy review before releasing any data.",
        "",
    ]
    metrics = [*ANONYMETER_METRICS, *HOLDOUT_DISTANCE_METRICS]
    columns = [("custom", "privacy", m) for m in metrics if ("custom", "privacy", m) in combined]
    if columns:
        table = combined[columns].copy()
        table.columns = [c[2] for c in columns]
        table.index.name = "model"
        unit = cfg.evaluation.privacy_attacks.unit
        lines += [
            f"Individuals are counted per **{unit}**. How to read each column:",
            "",
            "- `anonymeter_*_risk`: attack success above a baseline attack that only sees "
            "unseen real rows (0 = no better than guessing, 1 = every attack succeeds). "
            "Inference is the worst sensitive column.",
            "- `dcr_closer_to_train_share`, `distance_mia_auc`: near 0.5 when synthetic rows "
            "are no closer to the training rows than to unseen real rows; higher means closer "
            "to training rows.",
            "- `dcr_holdout_ratio`, `nndr_holdout_ratio`: near or above 1 when synthetic rows "
            "keep the same distance from training rows as unseen real rows do; below 1 means "
            "closer.",
            "",
            _dataframe_to_markdown(table.reset_index()),
            "",
            "Every Anonymeter attack with its 95% interval, its attack, baseline and control "
            "success rates, and whether Anonymeter judged it reliable: "
            f"{links.maybe(paths.evaluation / 'privacy_attacks.csv')}.",
        ]
    else:
        lines.append("Anonymeter and holdout-distance checks were not run this run.")
    other = [
        c
        for c in combined.columns
        if c[1] == "privacy" and c[0] in ("syntheval", "synthcity") and c[2] != "rank"
    ]
    if other:
        lines += [
            "",
            f"SynthEval and synthcity add {len(other)} more privacy metrics (membership "
            "inference, attribute disclosure, distances to closest record, identifiability, "
            f"k-anonymity); their raw values are in "
            f"{links.maybe(paths.evaluation / 'combined_evaluation.csv')}.",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Fairness
# ---------------------------------------------------------------------------

#: (framework, metric) -> (display label, description). All lower is better.
_FAIRNESS_GAP_METRICS = [
    (
        ("syntheval", "statistical_parity"),
        "Statistical parity gap",
        "Gap in positive-outcome rate across protected subgroups (0 = equal rates).",
    ),
    (
        ("custom", "equalized_odds"),
        "Equalized odds gap",
        "Gap in true-positive/false-positive rates across protected subgroups "
        "(0 = equal error rates).",
    ),
    (
        ("custom", "equal_opportunity"),
        "Equal opportunity gap",
        "Gap in true-positive rate across protected subgroups (0 = equal recall).",
    ),
]


def _fairness_section(combined, extras, paths: _Paths, links: _Links) -> str:
    lines = [
        "## 5. Fairness",
        "",
        "Two views: subgroup **gap metrics** (how differently a classifier trained on the "
        "data treats protected subgroups) and **log disparity** (Bhanot et al. 2021: how far "
        "each protected subgroup x outcome is over- or under-represented). Lower is better "
        "for every number below.",
        "",
        "### Subgroup gap metrics (0 = perfectly fair)",
        "",
    ]
    available = [
        (col, label, desc)
        for col, label, desc in _FAIRNESS_GAP_METRICS
        if (col[0], "fairness", col[1]) in combined.columns
    ]
    if available:
        rows = [
            {
                "model": model,
                **{
                    label: combined.loc[model, (fw, "fairness", metric)]
                    for (fw, metric), label, _ in available
                },
            }
            for model in combined.index
        ]
        lines += [_dataframe_to_markdown(pd.DataFrame(rows)), ""]
        lines += [f"- **{label}**: {desc}" for _, label, desc in available]
    else:
        lines.append("Subgroup gap metrics were not computed this run.")
    lines += ["", "### Log disparity", ""]
    reports = extras.get("log_disparity_reports") or {}
    if reports:
        rows = []
        for model, report in sorted(reports.items()):
            if "error" in report:
                rows.append({"model": model, "error": report["error"]})
                continue
            stats = report["summary_stats"]
            page = paths.evaluation_plots / "log_disparity" / f"{model}.html"
            rows.append(
                {
                    "model": model,
                    "mean_abs_log_disparity": stats.get("mean_abs_log_disparity"),
                    "median_abs_log_disparity": stats.get("median_abs_log_disparity"),
                    "share_significant_bh": stats.get("share_significant_bh"),
                    "subgroups": links.file(page, "sunburst") if page.exists() else "",
                }
            )
        lines += [
            "`mean`/`median_abs_log_disparity` is the typical drift (log-odds) of a subgroup's "
            "share from its real-data share; `share_significant_bh` is the fraction of subgroups "
            "whose drift is significant after Benjamini-Hochberg correction. The sunburst page "
            "shows which subgroups are over- or under-represented.",
            "",
            _dataframe_to_markdown(pd.DataFrame(rows)),
        ]
    else:
        lines.append("Log disparity was not computed this run.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Data and preparation
# ---------------------------------------------------------------------------


def _small_csv(path: Path, max_rows: int = 30) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except (ValueError, pd.errors.ParserError):
        return []
    if df.empty or len(df) > max_rows:
        return []
    return [_dataframe_to_markdown(df), ""]


def _data_section(paths: _Paths, links: _Links) -> str:
    lines = ["## 6. Data and preparation", "", "### Patient-level split", ""]
    if paths.data is not None:
        split = paths.data / "split_report.csv"
        lines += _small_csv(split)
        lines.append(f"Table: {links.maybe(split)}")
    lines += ["", "### Columns and missingness", ""]
    plots = links.plots(paths.dataset_plots / "data")
    lines += plots or ["_Not rendered yet: run `synthdata-plot` with the `data` section._"]
    lines += ["", "### Imputation", ""]
    lines += links.plots(paths.dataset_plots / "imputation") or [
        "_Not rendered yet: run `synthdata-plot` with the `imputation` section._"
    ]
    if paths.data is not None:
        lines += [
            "",
            "Drift of each column between observed and imputed values: "
            f"{links.maybe(paths.data / 'imputation_drift.csv')}",
        ]
    lines += ["", "### Hyperparameter search", ""]
    lines.append(
        f"Best parameters per model: {links.maybe(paths.generation / 'hpo_best_params.json')}"
    )
    lines.append("")
    lines += links.plots(paths.plots / "hpo") or [
        "_No HPO plots (HPO off, or not rendered yet: run `synthdata-plot`)._"
    ]
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# 7. File index
# ---------------------------------------------------------------------------

#: File name -> what it holds, for the file index.
_FILE_DESCRIPTIONS = {
    "combined_evaluation.csv": "Every metric for every model: raw values and per-group scores.",
    "ranking_summary.csv": "One row per model: mean scores, confidence intervals, ties.",
    "tstr_holdout.csv": "Holdout train-on-synthetic, test-on-real scores, per class.",
    "ovr_per_class.csv": "Two-class SynthEval metrics per class (one vs rest).",
    "privacy_attacks.csv": "Every Anonymeter attack with interval and success rates.",
    "split_report.csv": "Rows, patients and class balance of each split.",
    "imputation_drift.csv": "Observed vs imputed distribution drift per column.",
    "hpo_best_params.json": "Best hyperparameters found per model.",
    "optuna_studies.db": "Optuna studies (open with optuna-dashboard).",
    "manifest.json": "What each stage of this experiment ran and produced.",
    "config_snapshot.json": "The full configuration this experiment ran with.",
    "full.csv": "Full dataset after cleaning.",
    "train.csv": "Training split (before imputation).",
    "test.csv": "Held-out test split (before imputation).",
    "train_imputed.csv": "Training split after imputation (what generators learn from).",
    "test_imputed.csv": "Test split after imputation (what evaluation scores against).",
}


def _file_index_section(paths: _Paths, links: _Links, report_path: Path) -> str:
    lines = [
        "## 7. File index",
        "",
        "| File | What it holds |",
        "| --- | --- |",
    ]

    def add(path: Path, description: str | None = None):
        if path.exists() and path != report_path:
            desc = description or _FILE_DESCRIPTIONS.get(path.name, "")
            lines.append(f"| {links.file(path, path.name)} | {desc} |")

    for path in sorted(paths.evaluation.glob("*")):
        if path.is_file():
            add(path)
    for directory, description in (
        (paths.evaluation / "syntheval_benchmark", "SynthEval's raw benchmark output."),
        (paths.evaluation_plots / "syntheval_plots", "SynthEval's per-metric diagnostic plots."),
    ):
        add(directory, description)
    for path in sorted(paths.generation.glob("*.csv")):
        add(path, f"Synthetic data from `{path.stem}` (imputed, as scored).")
    released = paths.generation / "released"
    add(
        released,
        "Synthetic data to share: missing values put back where the real data has them.",
    )
    for name in ("hpo_best_params.json", "optuna_studies.db"):
        add(paths.generation / name)
    if paths.manifest is not None:
        add(paths.manifest)
        add(paths.manifest.parent / "config_snapshot.json")
    if paths.data is not None:
        for name in (
            "split_report.csv",
            "imputation_drift.csv",
            "full.csv",
            "train.csv",
            "test.csv",
            "train_imputed.csv",
            "test_imputed.csv",
        ):
            add(paths.data / name)
    add(paths.plots, "All plots of this experiment.")
    add(paths.dataset_plots, "Data and imputation plots (shared by every experiment).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_CONTENTS = (
    "1. [At a glance](#1-at-a-glance)\n"
    "2. [Ranking](#2-ranking)\n"
    "3. [Utility](#3-utility)\n"
    "4. [Privacy evidence](#4-privacy-evidence)\n"
    "5. [Fairness](#5-fairness)\n"
    "6. [Data and preparation](#6-data-and-preparation)\n"
    "7. [File index](#7-file-index)"
)


def build_evaluation_report(
    cfg: Config,
    dataset: Dataset,
    combined: pd.DataFrame,
    extras: dict,
    experiment=None,
    report_dir: Path | None = None,
) -> str:
    """Build the full Markdown evaluation report as a single string.

    ``report_dir`` is the directory the report will be written to; links are
    relative to it. It defaults to ``cfg.evaluation.output_dir``, which is
    also where tables missing from ``extras`` are read from.
    """
    report_dir = Path(report_dir) if report_dir else Path(cfg.evaluation.output_dir)
    extras = _load_extras(report_dir, combined, extras)
    model_names = sorted(extras.get("selected_datasets", {}) or combined.index.tolist())
    paths = _Paths(cfg, dataset, experiment, report_dir)
    links = _Links(report_dir)
    sections = [
        f"# Evaluation report: {dataset.name}",
        "",
        "The single page for this run: the outcome first, then the evidence behind it, "
        "with every plot shown and every table linked.",
        "",
        _CONTENTS,
        "",
        _glance_section(cfg, dataset, combined, extras, model_names, experiment),
        "",
        _ranking_section(cfg, combined, extras, paths, links),
        "",
        _utility_section(cfg, extras, paths, links),
        "",
        _privacy_section(cfg, combined, extras, paths, links),
        "",
        _fairness_section(combined, extras, paths, links),
        "",
        _data_section(paths, links),
        "",
        _file_index_section(paths, links, report_dir / "report.md"),
        "",
    ]
    return "\n".join(sections)


def save_evaluation_report(
    cfg: Config,
    dataset: Dataset,
    combined: pd.DataFrame,
    extras: dict,
    experiment=None,
    path: Path | None = None,
) -> Path:
    """Build and write the Markdown evaluation report, returning its path."""
    report_path = Path(path) if path else Path(cfg.evaluation.output_dir) / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_evaluation_report(
        cfg, dataset, combined, extras, experiment, report_dir=report_path.parent
    )
    report_path.write_text(report_text)
    logger.info("[report] wrote evaluation report to %s", report_path)
    return report_path
