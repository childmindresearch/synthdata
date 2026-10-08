"""Human-readable Markdown evaluation report: run metadata, the ranked
summary table, privacy-gate results, a recommended model, and fairness
highlights -- so a non-engineer reviewer can understand the evaluation
outcome without reading ``combined_evaluation.csv`` directly.
"""

import os
from pathlib import Path

import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.evaluation.baselines import is_baseline
from synthdata.evaluation.combine import simple_rank_summary
from synthdata.utils import get_logger

logger = get_logger(__name__)

_GATE_PASS_COL = ("__all__", "privacy_gate", "pass")
_GATE_VIOLATIONS_COL = ("__all__", "privacy_gate", "violations")


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


def _run_metadata_section(cfg: Config, dataset: Dataset, model_names: list, experiment) -> str:
    lines = [
        "## Run metadata",
        "",
        f"- Dataset: `{dataset.name}`"
        + (f" (version `{dataset.version}`)" if dataset.version else ""),
        f"- Target column: `{dataset.target_column}`",
        f"- Seed: `{cfg.seed}`",
        f"- Requested synthetic sample size (`generation.n_samples`): `{cfg.generation.n_samples}`",
        f"- Models evaluated ({len(model_names)}): " + ", ".join(f"`{m}`" for m in model_names),
    ]
    if experiment is not None:
        lines.append(f"- Experiment id: `{experiment.id}`")
    return "\n".join(lines)


def _has_replicates(ranking_summary: pd.DataFrame | None) -> bool:
    return (
        ranking_summary is not None
        and "overall_mean" in ranking_summary.columns
        and ranking_summary["n_replicates"].max() >= 2
    )


def _replicate_summary_section(ranking_summary: pd.DataFrame) -> str:
    def _interval(row, dim):
        mean, low, high = row[f"{dim}_mean"], row[f"{dim}_ci_low"], row[f"{dim}_ci_high"]
        if pd.isna(low):
            return _fmt_metric(mean)
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"

    dims = [
        d for d in ("overall", "utility", "privacy", "fairness") if f"{d}_mean" in ranking_summary
    ]
    rows = []
    for model, row in ranking_summary.iterrows():
        tied = row["tied_with_best"]
        rows.append(
            {
                "rank": row["rank"],
                "model": model,
                "seeds": row["n_replicates"],
                **{dim: _interval(row, dim) for dim in dims},
                "tied with best": "" if pd.isna(tied) else ("yes" if tied else "no"),
            }
        )
    return (
        "## Ranked summary (higher = better)\n\n"
        "Each score is the mean over seed replicates with its 95% confidence interval. "
        '"Tied with best" is yes when a one-sided Welch t-test (5% level) cannot place the '
        "model's overall score below the best eligible model's: the order among tied models "
        "is within seed noise.\n\n" + _dataframe_to_markdown(pd.DataFrame(rows))
    )


def _ranked_summary_section(combined: pd.DataFrame, ranking_summary=None) -> str:
    if _has_replicates(ranking_summary):
        text = _replicate_summary_section(ranking_summary)
        summary = ranking_summary
    else:
        text, summary = _point_summary_section(combined)
    if any(is_baseline(m) for m in summary.index):
        text += (
            "\n\nRows named `baseline_*` are fixed references, not candidates. "
            "`baseline_train_copy` is real training data: the utility a generator can at best "
            "reach, and the privacy it must stay well above. `baseline_marginals` samples each "
            "column independently: the utility floor a useful generator must beat."
        )
    return text


def _point_summary_section(combined: pd.DataFrame) -> "tuple[str, pd.DataFrame]":
    summary = simple_rank_summary(combined)
    if summary.empty:
        return "## Ranked summary\n\nNo ranking columns were produced.", summary
    table = _dataframe_to_markdown(summary.reset_index())
    text = (
        "## Ranked summary (higher = better)\n\n"
        + table
        + "\n\nSingle seed: these scores carry no uncertainty estimate, so small gaps "
        "may be noise. Set `generation.n_replicates` to 2 or more to get confidence intervals."
    )
    return text, summary


def _privacy_gate_section(combined: pd.DataFrame) -> str:
    if _GATE_PASS_COL not in combined.columns:
        return (
            "## Privacy gate\n\n"
            "Privacy gate was not run this evaluation (disabled, or none of its configured "
            "threshold metrics were computed -- see logs). No absolute privacy safety floor "
            "was checked; treat any 'recommended model' below with that caveat."
        )
    lines = ["## Privacy gate", ""]
    passing = combined.index[combined[_GATE_PASS_COL]].tolist()
    failing = combined.index[~combined[_GATE_PASS_COL]].tolist()
    lines.append(f"- Passing ({len(passing)}): " + (", ".join(f"`{m}`" for m in passing) or "none"))
    lines.append(
        f"- **FAILING ({len(failing)})**: " + (", ".join(f"`{m}`" for m in failing) or "none")
    )
    if failing:
        lines.append("")
        lines.append("### Violations")
        for model in failing:
            violations = combined.loc[model, _GATE_VIOLATIONS_COL]
            lines.append(f"- `{model}`: {violations}")
    return "\n".join(lines)


def _replicate_recommendation_section(ranking_summary: pd.DataFrame) -> str:
    eligible = ranking_summary[ranking_summary["eligible"].astype(bool)]
    if eligible.empty:
        return (
            "## Recommended model\n\n"
            "**No candidate model passed the privacy gate in every seed replicate** -- refusing "
            "to recommend one. See the Privacy gate section above."
        )
    best = eligible.index[0]
    row = eligible.loc[best]
    lines = [
        "## Recommended model",
        "",
        f"**`{best}`** (mean overall score {row['overall_mean']:.3f}, 95% CI "
        f"[{row['overall_ci_low']:.3f}, {row['overall_ci_high']:.3f}], "
        f"{row['n_replicates']} seeds)",
    ]
    tied = [
        m
        for m in eligible.index[1:]
        if pd.notna(eligible.loc[m, "tied_with_best"]) and bool(eligible.loc[m, "tied_with_best"])
    ]
    if tied:
        lines += [
            "",
            "Not distinguishable from it at the 5% level: "
            + ", ".join(f"`{m}`" for m in tied)
            + ". Choose among these on other grounds (cost, the type score that matters most).",
        ]
    if "privacy_gate_pass" in eligible.columns:
        lines += ["", "Only models that passed the privacy gate in every replicate were eligible."]
    return "\n".join(lines)


def _recommended_model_section(combined: pd.DataFrame, ranking_summary=None) -> str:
    if _has_replicates(ranking_summary):
        return _replicate_recommendation_section(ranking_summary)
    if ("__all__", "overall", "rank") not in combined.columns:
        return "## Recommended model\n\nNo overall rank column was produced."

    candidates = combined[[not is_baseline(m) for m in combined.index]]
    if candidates.empty:
        return "## Recommended model\n\nOnly baseline rows were evaluated; nothing to recommend."
    if _GATE_PASS_COL in combined.columns:
        eligible = candidates[candidates[_GATE_PASS_COL]]
        if eligible.empty:
            return (
                "## Recommended model\n\n"
                "**No model passed the privacy gate this run** -- refusing to recommend a "
                "gate-failing model regardless of its overall rank. See the Privacy gate "
                "section above for violation details, and either loosen/verify the configured "
                "thresholds (`evaluation.privacy_gate.thresholds`) or improve the "
                "generator(s) before re-evaluating."
            )
    else:
        eligible = candidates

    best = eligible.sort_values(("__all__", "overall", "rank"), ascending=False).index[0]
    overall_rank = eligible.loc[best, ("__all__", "overall", "rank")]
    lines = [
        "## Recommended model",
        "",
        f"**`{best}`** (overall rank score: {overall_rank:.3f})",
    ]
    if _GATE_PASS_COL in combined.columns:
        lines.append("")
        lines.append(
            "Selected as the top overall-ranked model among those that passed the privacy gate."
        )
    return "\n".join(lines)


#: (framework, metric) -> (display label, description). All three are "lower is
#: better" gap/disparity metrics (0 = perfectly fair), unlike log disparity's
#: mean/median columns which are also lower-is-better but on a different (log-odds)
#: scale -- keeping them in separate tables avoids implying they're comparable.
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


def _fmt_metric(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.4g}"


def _fairness_highlights_section(combined: pd.DataFrame, extras: dict) -> str:
    lines = ["## Fairness highlights", ""]
    lines.append(
        "Two independent views of fairness are computed: (1) subgroup **gap metrics** "
        "(statistical parity / equalized odds / equal opportunity -- how differently the "
        "model treats protected subgroups on average) and (2) the **log disparity** report "
        "(Bhanot et al. 2021 -- representation bias per protected subgroup x outcome, with "
        "significance testing). Lower is better for every number below."
    )
    lines.append("")

    lines.append("### Subgroup gap metrics (0 = perfectly fair)")
    lines.append("")
    available_gap_cols = [
        (col, label, desc)
        for col, label, desc in _FAIRNESS_GAP_METRICS
        if (col[0], "fairness", col[1]) in combined.columns
    ]
    if available_gap_cols:
        rows = []
        for model in combined.index:
            row = {"model": model}
            for (framework, metric), label, _desc in available_gap_cols:
                row[label] = combined.loc[model, (framework, "fairness", metric)]
            rows.append(row)
        gap_df = pd.DataFrame(rows)
        lines.append(_dataframe_to_markdown(gap_df))
        lines.append("")
        for _col, label, desc in available_gap_cols:
            lines.append(f"- **{label}**: {desc}")
    else:
        lines.append("Subgroup gap metrics were not computed this run.")
    lines.append("")

    lines.append("### Log disparity (Bhanot et al. 2021)")
    lines.append("")
    log_disparity_reports = extras.get("log_disparity_reports") or {}
    if log_disparity_reports:
        lines.append(
            "`mean`/`median_abs_log_disparity` summarize how far each subgroup's "
            "representation in the synthetic data drifts (in log-odds) from its real-data "
            "rate, averaged across every protected-attribute x outcome subgroup; "
            "`share_significant_bh` is the fraction of those subgroups whose drift is "
            "statistically significant after Benjamini-Hochberg correction (closer to 0 = "
            "fewer subgroups are meaningfully misrepresented)."
        )
        lines.append("")
        rows = []
        for model, report in sorted(log_disparity_reports.items()):
            if "error" in report:
                rows.append({"model": model, "error": report["error"]})
                continue
            stats = report["summary_stats"]
            rows.append(
                {
                    "model": model,
                    "mean_abs_log_disparity": stats.get("mean_abs_log_disparity"),
                    "median_abs_log_disparity": stats.get("median_abs_log_disparity"),
                    "share_significant_bh": stats.get("share_significant_bh"),
                }
            )
        lines.append(_dataframe_to_markdown(pd.DataFrame(rows)))
        lines.append("")
        lines.append(
            "See the per-model interactive sunburst reports linked under Plots below for a "
            "subgroup-by-subgroup breakdown (which subgroups are over/under-represented)."
        )
    else:
        lines.append("Log disparity was not computed this run.")
    return "\n".join(lines)


def _class_metrics_section(cfg: Config, extras: dict) -> str:
    """Imbalance-aware utility: holdout TSTR per class, and how binary-only metrics ran."""
    lines = ["## Class-level utility (imbalance-aware)", ""]
    tstr_table = extras.get("tstr_table")
    if tstr_table is not None and not tstr_table.empty:
        lines.append(
            "A fixed XGBoost classifier is fitted on each dataset and scored on the held-out "
            "test split (train on synthetic, test on real). Macro-F1 and balanced accuracy weigh "
            "every class equally, so a minority class the synthetic data gets wrong pulls them "
            "down; per-class F1 (`f1_<class>`) shows which class. The `trtr (real train)` row is "
            "the same classifier fitted on the real training rows: the ceiling."
        )
        lines.append("")
        lines.append(_dataframe_to_markdown(tstr_table.reset_index()))
    else:
        lines.append("Holdout TSTR was not computed this run.")
    lines.append("")
    ovr = extras.get("ovr_per_class")
    if cfg.evaluation.class_averaging == "ovr_macro" and ovr is not None and not ovr.empty:
        lines.append(
            "SynthEval metrics that need two classes (AUROC difference, subgroup gap metrics) "
            "ran once per class against the rest and are averaged with equal weight per class "
            "in the tables above. Values per class:"
        )
        lines.append("")
        flat = ovr.copy()
        flat.columns = [f"{metric} [{cls}]" for metric, cls in flat.columns]
        flat.index.name = "model"
        lines.append(_dataframe_to_markdown(flat.reset_index()))
    elif cfg.evaluation.class_averaging == "binary" and cfg.evaluation.binary_target.enabled:
        bt = cfg.evaluation.binary_target
        lines.append(
            "SynthEval metrics that need two classes ran on the target collapsed to "
            f"positive {bt.positive_classes} versus negative {bt.negative_classes}."
        )
    return "\n".join(lines)


def _plot_links_section(report_dir: Path, cfg: Config, log_disparity_reports: dict) -> str:
    """List links to evaluation plots, relative to where ``report.md`` is written.

    ``report.md`` lives under ``cfg.evaluation.output_dir`` while plots live under
    the separate ``cfg.plots.output_dir`` tree (both nested under the same
    ``<experiment_id>/``) -- so links must be computed relative to ``report_dir``
    itself via ``os.path.relpath``, not assumed to share a common ancestor at a
    fixed number of ``..`` hops up.
    """
    plots_dir = Path(cfg.plots.output_dir) / "evaluation"
    lines = ["## Plots", ""]
    candidates = [
        ("Utility vs privacy trade-off", plots_dir / "utility_vs_privacy.png"),
        ("Utility vs fairness trade-off", plots_dir / "utility_vs_fairness.png"),
        ("Privacy vs fairness trade-off", plots_dir / "privacy_vs_fairness.png"),
    ]
    found_any = False
    for label, path in candidates:
        if path.exists():
            found_any = True
            lines.append(f"- [{label}]({os.path.relpath(path, report_dir)})")
    for model in sorted(log_disparity_reports):
        html_path = plots_dir / "log_disparity" / f"{model}.html"
        if html_path.exists():
            found_any = True
            lines.append(
                f"- [Log disparity report ({model})]({os.path.relpath(html_path, report_dir)})"
            )
    if not found_any:
        lines.append(
            "No plots were found under `"
            + str(plots_dir)
            + "` (run `synthdata-plot` to render recorded plot artifacts)."
        )
    return "\n".join(lines)


def build_evaluation_report(
    cfg: Config,
    dataset: Dataset,
    combined: pd.DataFrame,
    extras: dict,
    experiment=None,
    report_dir: Path | None = None,
) -> str:
    """Build the full Markdown evaluation report as a single string.

    ``report_dir`` is the directory the report will be written to (needed to
    compute correct relative plot links); defaults to ``cfg.evaluation.output_dir``,
    matching :func:`save_evaluation_report`'s default write location.
    """
    model_names = sorted(extras.get("selected_datasets", {}) or combined.index.tolist())
    report_dir = Path(report_dir) if report_dir else Path(cfg.evaluation.output_dir)
    sections = [
        f"# Evaluation report: {dataset.name}",
        "",
        _run_metadata_section(cfg, dataset, model_names, experiment),
        "",
        _ranked_summary_section(combined, extras.get("ranking_summary")),
        "",
        _privacy_gate_section(combined),
        "",
        _recommended_model_section(combined, extras.get("ranking_summary")),
        "",
        _class_metrics_section(cfg, extras),
        "",
        _fairness_highlights_section(combined, extras),
        "",
        _plot_links_section(report_dir, cfg, extras.get("log_disparity_reports") or {}),
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
    report_text = build_evaluation_report(
        cfg, dataset, combined, extras, experiment, report_dir=report_path.parent
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    logger.info("[report] wrote evaluation report to %s", report_path)
    return report_path
