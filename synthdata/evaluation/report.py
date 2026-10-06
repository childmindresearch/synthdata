"""Human-readable Markdown evaluation report: run metadata, the ranked
summary table, privacy-gate results, a recommended model, and fairness
highlights -- so a non-engineer reviewer can understand the evaluation
outcome without reading ``combined_evaluation.csv`` directly.
"""

import math
import os
from pathlib import Path

import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.evaluation.artifacts import (
    _LOG_REPORT_TABLES,
    _SAFE_LOG_DISPARITY_REASONS,
    _model_artifact_id,
    _validate_release_score_binding,
    _validate_release_score_record,
    artifact_bundle_dir,
    expected_evaluation_context,
    validate_evaluation_bundle,
)
from synthdata.evaluation.combine import simple_rank_summary, validate_combined_table
from synthdata.utils import get_logger

logger = get_logger(__name__)

_GATE_PASS_COL = ("__all__", "privacy_gate", "pass")
_GATE_VIOLATIONS_COL = ("__all__", "privacy_gate", "violations")
_TUNING_UTILITY_COL = ("__all__", "utility", "U_tuning")


def _valid_log_disparity_success(report: object) -> bool:
    """Return whether report contains complete, safe success evidence."""
    if not isinstance(report, dict) or report.get("state") != "succeeded":
        return False
    if any(not isinstance(report.get(table), pd.DataFrame) for table in _LOG_REPORT_TABLES):
        return False
    stats = report.get("summary_stats")
    if not isinstance(stats, dict):
        return False
    metric_names = (
        "mean_abs_log_disparity",
        "median_abs_log_disparity",
        "share_significant_bh",
    )
    return all(
        isinstance(stats.get(name), (int, float))
        and not isinstance(stats.get(name), bool)
        and math.isfinite(float(stats[name]))
        for name in metric_names
    )


def _safe_log_disparity_reason(report: dict, fallback: str) -> str:
    """Return persisted allowlisted reason, never arbitrary report input."""
    reason = report.get("reason")
    return reason if isinstance(reason, str) and reason in _SAFE_LOG_DISPARITY_REASONS else fallback


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


def _ranked_summary_section(combined: pd.DataFrame, *, partial_coverage: bool = False) -> str:
    summary = simple_rank_summary(combined)
    if summary.empty:
        return "## Ranked summary\n\nNo ranking columns were produced."
    table = _dataframe_to_markdown(summary.reset_index())
    qualifier = " (partial coverage; higher = better)" if partial_coverage else " (higher = better)"
    return f"## Ranked summary{qualifier}\n\n" + table


def _coverage_section(coverage: dict) -> str:
    """Render missing generation outputs separately from evaluated model metrics."""
    failed_outputs = coverage.get("failed_outputs")
    evaluated_models = coverage.get("evaluated_models")
    if not isinstance(failed_outputs, list) or not all(
        isinstance(name, str) for name in failed_outputs
    ):
        raise ValueError("Partial evaluation coverage must list failed output names")
    if not isinstance(evaluated_models, list) or not all(
        isinstance(name, str) for name in evaluated_models
    ):
        raise ValueError("Partial evaluation coverage must list evaluated model names")
    lines = [
        "## Generation coverage",
        "",
        "**Partial:** failed generation outputs were omitted from metric tables and rankings.",
        "",
        "- Evaluated models: " + (", ".join(f"`{name}`" for name in evaluated_models) or "none"),
        "- Failed outputs: " + (", ".join(f"`{name}`" for name in failed_outputs) or "none"),
        "- Ranked summary compares evaluated models only; it is not a complete-run ranking.",
    ]
    return "\n".join(lines)


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
    lines.append("- Gate result is an audit warning; it does not filter candidate selection.")
    if failing:
        lines.append("")
        lines.append("### Violations")
        for model in failing:
            violations = combined.loc[model, _GATE_VIOLATIONS_COL]
            lines.append(f"- `{model}`: {violations}")
    return "\n".join(lines)


def _recommended_model_section(combined: pd.DataFrame, *, partial_coverage: bool = False) -> str:
    if _TUNING_UTILITY_COL not in combined.columns:
        return (
            "## Recommended model\n\n"
            "No overall rank column was produced; no complete fixed-transform `U_tuning` "
            "column was produced."
        )

    values = pd.to_numeric(combined[_TUNING_UTILITY_COL], errors="coerce")
    eligible = combined.loc[values.notna() & values.map(math.isfinite)]
    if eligible.empty:
        return "## Recommended model\n\nNo model has complete finite `U_tuning` evidence."

    best = eligible[_TUNING_UTILITY_COL].idxmax()
    tuning_utility = eligible.loc[best, _TUNING_UTILITY_COL]
    lines = [
        "## Recommended model",
        "",
        f"**`{best}`** (`U_tuning`: {tuning_utility:.3f})",
        "",
        (
            "Selected from available models only using highest complete fixed-transform "
            "tuning utility; coverage is partial."
            if partial_coverage
            else "Selected using highest complete fixed-transform tuning utility only; all candidate "
            "audit rows are retained."
        ),
    ]
    if _GATE_PASS_COL in combined.columns:
        lines.append("")
        gate_pass = bool(combined.loc[best, _GATE_PASS_COL])
        lines.append(
            "**Privacy-gate warning:** selected model "
            + ("passed." if gate_pass else "failed; selection was not blocked.")
        )
    return "\n".join(lines)


def _release_score_section(extras: dict) -> str:
    """Render final-holdout release score as audit evidence, never selection input."""
    evidence = extras.get("final_holdout_evidence") or {}
    score = evidence.get("release_score") or extras.get("release_score")
    lines = ["## Final-holdout release score audit", ""]
    if not isinstance(score, dict):
        lines.append("No final-holdout release score was recorded.")
        return "\n".join(lines)

    # Never print numeric success from an unbound or incomplete decomposition.
    # Failed/indeterminate states remain useful audit output.
    evidence_state = evidence.get("state", score.get("status"))
    if score.get("status") == "succeeded":
        try:
            if evidence.get("state") != "succeeded":
                raise ValueError("missing successful final-holdout evidence state")
            _validate_release_score_record(
                score, model_name=str(evidence.get("selected_model")), path=Path("<report>")
            )
            _validate_release_score_binding(score, evidence, label="Report release score")
        except ValueError:
            lines.append("- Status: `indeterminate`")
            lines.append(
                "- Claimed success withheld: final-holdout evidence is incomplete or tampered."
            )
            return "\n".join(lines)

    lines.append(
        "**Audit-only:** this score is computed after candidate selection and is not used to rerank models."
    )
    lines.append("")
    state = evidence_state
    if state not in {"succeeded", "failed", "indeterminate"}:
        state = "indeterminate"
    lines.append(f"- Status: `{state}`")
    for field in (
        "evidence_execution_state",
        "metric_completeness_state",
        "score_completeness_state",
        "audit_outcome_state",
    ):
        if field in evidence:
            lines.append(f"- {field.replace('_', ' ').capitalize()}: `{evidence[field]}`")
    indeterminate = score.get("indeterminate_dimensions") or []
    if indeterminate:
        lines.append(
            "- Indeterminate dimensions: " + ", ".join(f"`{item}`" for item in indeterminate)
        )
    if state != "succeeded":
        lines.append(
            "- Metrics and plots are withheld because final-holdout evidence is audit-only."
        )
        return "\n".join(lines)
    lines.append(f"- R_final: `{_fmt_metric(score.get('score'))}`")
    dimensions = score.get("dimensions") or {}
    for name in ("utility", "privacy", "fairness"):
        dimension = dimensions.get(name) or {}
        lines.append(f"- {name.capitalize()}: `{_fmt_metric(dimension.get('score'))}`")
    if "identity" in (dimensions.get("privacy") or {}):
        lines.append(f"- Identity safety: `{_fmt_metric(dimensions['privacy'].get('identity'))}`")
    lines.append(f"- Audit-only label: `{bool(score.get('audit_only', True))}`")
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
    release_evidence = extras.get("release_evidence_validation") or {}
    if release_evidence:
        lines.append("### Canonical release evidence")
        lines.append("")
        for model, validation in sorted(release_evidence.items()):
            status = validation.get("decision_status", validation.get("status", "unknown"))
            lines.append(f"- `{model}`: `{status}` (canonical release/fairness evidence)")
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
        has_incomplete = False
        for model, report in sorted(log_disparity_reports.items()):
            if not isinstance(report, dict):
                report = {}
            state = report.get("state")
            if state not in {"succeeded", "failed", "indeterminate"}:
                has_incomplete = True
                rows.append(
                    {
                        "model": model,
                        "state": "indeterminate",
                        "reason": "report_state_missing_or_unknown",
                    }
                )
                continue
            if state == "failed" or "error" in report:
                rows.append(
                    {
                        "model": model,
                        "state": "failed",
                        "reason": _safe_log_disparity_reason(
                            report, "log_disparity_evaluation_failed"
                        ),
                    }
                )
                continue
            if state == "indeterminate":
                has_incomplete = True
                rows.append(
                    {
                        "model": model,
                        "state": "indeterminate",
                        "reason": _safe_log_disparity_reason(
                            report, "log_disparity_evaluation_indeterminate"
                        ),
                    }
                )
                continue
            if not _valid_log_disparity_success(report):
                has_incomplete = True
                rows.append(
                    {
                        "model": model,
                        "state": "indeterminate",
                        "reason": "incomplete_report",
                    }
                )
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
        if not has_incomplete:
            lines.append(
                "See the per-model interactive sunburst reports linked under Plots below for a "
                "subgroup-by-subgroup breakdown (which subgroups are over/under-represented)."
            )
    else:
        lines.append("Log disparity was not computed this run.")
    return "\n".join(lines)


def _plot_links_section(report_dir: Path, cfg: Config, log_disparity_reports: dict) -> str:
    """List links to evaluation plots, relative to where ``report.md`` is written.

    Plot links use portable artifact identifiers rather than exposing configured paths.
    """
    plots_dir = Path(cfg.plots.output_dir) / "evaluation"
    formats = tuple(str(fmt).lstrip(".") for fmt in cfg.plots.formats)

    def _portable_link(path: Path) -> str:
        """Return an existence-checked, report-relative Markdown target."""
        return Path(os.path.relpath(path, report_dir)).as_posix()

    def _first_existing(stem: Path) -> Path | None:
        return next(
            (
                stem.with_suffix(f".{fmt}")
                for fmt in formats
                if stem.with_suffix(f".{fmt}").is_file()
            ),
            None,
        )

    lines = ["## Plots", ""]
    candidates = [
        ("Utility, privacy, and fairness rank trade-off (3D)", plots_dir / "rank_tradeoff_3d.html"),
    ]
    for label, stem in (
        ("Utility vs privacy trade-off", plots_dir / "utility_vs_privacy"),
        ("Utility vs fairness trade-off", plots_dir / "utility_vs_fairness"),
        ("Privacy vs fairness trade-off", plots_dir / "privacy_vs_fairness"),
    ):
        if path := _first_existing(stem):
            candidates.append((label, path))
    found_any = False
    for label, path in candidates:
        if path.exists():
            found_any = True
            lines.append(f"- [{label}]({_portable_link(path)})")
    for model in sorted(log_disparity_reports):
        report = log_disparity_reports[model]
        if not _valid_log_disparity_success(report):
            continue
        html_path = plots_dir / "log_disparity" / f"{_model_artifact_id(model)}.html"
        if html_path.exists():
            found_any = True
            lines.append(f"- [Log disparity report ({model})]({_portable_link(html_path)})")
    if not found_any:
        lines.append(
            "No plots were found under `plots/evaluation/` "
            "(run `synthdata-plot` to render recorded plot artifacts)."
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
    validate_combined_table(combined)
    artifact_manifest = extras.get("artifact_manifest")
    if artifact_manifest:
        expected_context = expected_evaluation_context(
            dataset,
            classification_score=cfg.evaluation.synthcity.classification_score,
        )
        validate_evaluation_bundle(
            Path(artifact_manifest).parent.parent,
            expected_config_path=cfg.config_path,
            expected_role_context_fingerprints=expected_context["role_context_fingerprints"],
            expected_semantic_context_fingerprint=expected_context["semantic_context_fingerprint"],
            expected_role_hashes=expected_context["role_hashes"],
            expected_role_hashes_by_framework=expected_context["role_hashes_by_framework"],
            expected_population_unit=(
                "patient_group" if cfg.evaluation.group_mode == "patient_group" else "row"
            ),
            expected_group_mode=cfg.evaluation.group_mode,
            allow_legacy=dataset.legacy_two_role,
        )
    elif artifact_bundle_dir(cfg.evaluation.output_dir).is_dir():
        raise FileNotFoundError(
            "An evaluation artifact bundle exists but report extras did not provide its "
            "manifest; refusing to render uncontextualized results"
        )
    model_names = sorted(extras.get("selected_datasets", {}) or combined.index.tolist())
    coverage = extras.get("evaluation_coverage")
    partial_coverage = isinstance(coverage, dict) and coverage.get("status") == "partial"
    report_dir = Path(report_dir) if report_dir else Path(cfg.evaluation.output_dir)
    sections = [
        f"# Evaluation report: {dataset.name}",
        "",
        _run_metadata_section(cfg, dataset, model_names, experiment),
        "",
    ]
    if partial_coverage:
        sections.extend([_coverage_section(coverage), ""])
    sections.extend(
        [
            _ranked_summary_section(combined, partial_coverage=partial_coverage),
            "",
            _privacy_gate_section(combined),
            "",
            _recommended_model_section(combined, partial_coverage=partial_coverage),
            "",
            _release_score_section(extras),
            "",
            _fairness_highlights_section(combined, extras),
            "",
            _plot_links_section(report_dir, cfg, extras.get("log_disparity_reports") or {}),
            "",
        ]
    )
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
    logger.info("[report] wrote evaluation report artifact report.md")
    return report_path
