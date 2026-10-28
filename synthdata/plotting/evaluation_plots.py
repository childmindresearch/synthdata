"""Evaluation figures: interactive rank trade-offs and log-disparity reports."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from synthdata.config import Config
from synthdata.plotting import save_matplotlib_figure, save_plotly_figure
from synthdata.utils import get_logger, model_sort_key, split_replicate_name

logger = get_logger(__name__)


def _is_hpo(name: str) -> bool:
    return split_replicate_name(name)[0].endswith("_hpo")


def _base_model(name: str) -> str:
    """Generator a row belongs to: ``"ctgan_hpo__rep2"`` -> ``"ctgan"``."""
    name = split_replicate_name(name)[0]
    return name[: -len("_hpo")] if name.endswith("_hpo") else name


def plot_rank_tradeoff(
    combined: pd.DataFrame,
    x_key: tuple,
    y_key: tuple,
    x_label: str,
    y_label: str,
    title: str,
):
    """Build a two-dimensional rank trade-off scatter plot.

    One labelled point per model: the mean over its seed replicates, with
    95% interval bars on both axes when it has two or more. Single seeds are
    small faint dots behind it, unlabelled. HPO-tuned variants are diamonds,
    regular ones circles; variants of one base model share a color. Both
    ranks are higher-is-better, so the top-right corner is best.
    """
    from matplotlib.lines import Line2D

    from synthdata.evaluation.combine import mean_and_interval

    groups = pd.Series(
        [split_replicate_name(name)[0] for name in combined.index], index=combined.index
    )
    base_models = sorted({_base_model(model) for model in groups.unique()})
    # tab20 has 20 colors; reuse them rather than fail when a run has more generators.
    colors = plt.get_cmap("tab20").colors  # ty: ignore[unresolved-attribute]
    palette = {base: colors[i % len(colors)] for i, base in enumerate(base_models)}

    fig, ax = plt.subplots(figsize=(11, 7))
    any_interval = False
    by_model = groups.groupby(groups, sort=False).groups
    for model in sorted(by_model, key=model_sort_key):
        names = by_model[model]
        is_hpo = _is_hpo(str(model))
        color = palette[_base_model(str(model))]
        xs = pd.to_numeric(combined.loc[names, x_key], errors="coerce")
        ys = pd.to_numeric(combined.loc[names, y_key], errors="coerce")
        x_mean, x_low, x_high = mean_and_interval(xs, 0.95)
        y_mean, y_low, y_high = mean_and_interval(ys, 0.95)
        if len(names) > 1:
            ax.scatter(xs, ys, s=18, color=color, alpha=0.35, linewidths=0, zorder=2)
        if pd.notna(x_low):
            any_interval = True
            # Ranks live in [0, 1]; a t-interval from few seeds can overshoot.
            x_low, x_high = max(x_low, 0.0), min(x_high, 1.0)
            y_low, y_high = max(y_low, 0.0), min(y_high, 1.0)
            ax.errorbar(
                x_mean,
                y_mean,
                xerr=[[x_mean - x_low], [x_high - x_mean]],
                yerr=[[y_mean - y_low], [y_high - y_mean]],
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                alpha=0.8,
                zorder=3,
            )
        ax.scatter(
            x_mean,
            y_mean,
            s=130,
            marker="D" if is_hpo else "o",
            color=color,
            alpha=0.9,
            edgecolors="black" if is_hpo else "white",
            linewidths=1.2 if is_hpo else 0.8,
            zorder=4,
        )
        ax.annotate(
            str(model),
            (x_mean, y_mean),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
            zorder=5,
        )

    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.annotate(
        "better",
        xy=(0.97, 0.97),
        xytext=(0.88, 0.88),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=9,
        color="#2e8b57",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#2e8b57"},
    )
    ax.set_xlabel(f"{x_label} (higher is better)", fontsize=12)
    ax.set_ylabel(f"{y_label} (higher is better)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="grey",
            markersize=9,
            markeredgecolor="white",
            label="Model (mean over seeds)",
        ),
    ]
    if any(_is_hpo(str(m)) for m in groups.unique()):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                color="grey",
                markersize=9,
                markeredgecolor="black",
                label="HPO-tuned model",
            )
        )
    if any_interval:
        handles += [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="grey",
                alpha=0.35,
                markersize=5,
                label="One seed",
            ),
            Line2D([0], [0], color="grey", lw=1.2, label="95% interval over seeds"),
        ]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_rank_tradeoff_3d(combined: pd.DataFrame):
    """Build an interactive utility/privacy/fairness rank scatter plot.

    HPO-tuned variants are diamonds; regular variants are circles. All
    variants use the same marker size, and each base model's variants share a
    color.
    """
    import plotly.graph_objects as go

    rank_keys = {
        "Utility": ("__all__", "utility", "rank"),
        "Privacy": ("__all__", "privacy", "rank"),
        "Fairness": ("__all__", "fairness", "rank"),
    }
    missing_keys = [key for key in rank_keys.values() if key not in combined.columns]
    if missing_keys:
        raise ValueError(f"Cannot plot 3D rank trade-off; missing rank columns: {missing_keys}")

    models = list(combined.index)
    base_models = sorted({_base_model(model) for model in models})
    palette = {
        base: f"hsl({index * 360 / max(len(base_models), 1):.0f}, 58%, 48%)"
        for index, base in enumerate(base_models)
    }
    fig = go.Figure()
    for model in models:
        ranks = [combined.loc[model, key] for key in rank_keys.values()]
        if pd.isna(ranks).any():
            raise ValueError(f"Cannot plot 3D rank trade-off; {model} has missing ranks: {ranks}")
    # One labelled point per model at its mean over seeds; single seeds are
    # small unlabelled points (hover shows their name).
    groups = pd.Series([split_replicate_name(m)[0] for m in models], index=models)
    replicated = groups.duplicated(keep=False)
    for model in [m for m in models if replicated[m]]:
        ranks = [combined.loc[model, key] for key in rank_keys.values()]
        fig.add_trace(
            go.Scatter3d(
                x=[ranks[0]],
                y=[ranks[1]],
                z=[ranks[2]],
                mode="markers",
                showlegend=False,
                marker={"size": 3, "color": palette[_base_model(model)], "opacity": 0.4},
                hovertemplate=(
                    f"{model}<br>Utility rank: %{{x:.3f}}<br>"
                    "Privacy rank: %{y:.3f}<br>Fairness rank: %{z:.3f}<extra></extra>"
                ),
            )
        )
    by_model = groups.groupby(groups, sort=False).groups
    for model in sorted(by_model, key=model_sort_key):
        names = by_model[model]
        is_hpo = _is_hpo(str(model))
        ranks = [float(combined.loc[names, key].astype(float).mean()) for key in rank_keys.values()]
        base = _base_model(str(model))
        fig.add_trace(
            go.Scatter3d(
                x=[ranks[0]],
                y=[ranks[1]],
                z=[ranks[2]],
                mode="markers+text",
                name=str(model),
                showlegend=False,
                text=[str(model)],
                textposition="top center",
                marker={
                    "size": 8,
                    "symbol": "diamond" if is_hpo else "circle",
                    "color": palette[base],
                    "line": {"color": "black" if is_hpo else palette[base], "width": 1},
                },
                hovertemplate=(
                    f"<b>{model}</b><br>Utility rank: %{{x:.3f}}<br>"
                    "Privacy rank: %{y:.3f}<br>Fairness rank: %{z:.3f}<extra></extra>"
                ),
            )
        )

    for base in base_models:
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=base,
                marker={"size": 8, "color": palette[base]},
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            name="Regular model",
            marker={"size": 8, "color": "grey", "symbol": "circle"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            name="HPO-tuned model",
            marker={"size": 8, "color": "grey", "symbol": "diamond"},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Utility, Privacy, and Fairness Rank Trade-off",
        scene={
            "xaxis_title": "Utility rank",
            "yaxis_title": "Privacy rank",
            "zaxis_title": "Fairness rank",
        },
        legend_title_text="Base model / variant",
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return fig


def save_rank_tradeoff_plots(cfg: Config, combined: pd.DataFrame, output_dir: str | Path) -> None:
    """Save interactive 3D and static pairwise rank trade-off plots."""
    output_dir = Path(output_dir) / "evaluation"
    fig = plot_rank_tradeoff_3d(combined)
    save_plotly_figure(fig, output_dir / "rank_tradeoff_3d", ("html",))

    pairs = [
        (
            ("__all__", "utility", "rank"),
            ("__all__", "privacy", "rank"),
            "Utility rank",
            "Privacy rank",
            "Utility vs Privacy Trade-off",
            "utility_vs_privacy",
        ),
        (
            ("__all__", "utility", "rank"),
            ("__all__", "fairness", "rank"),
            "Utility rank",
            "Fairness rank",
            "Utility vs Fairness Trade-off",
            "utility_vs_fairness",
        ),
        (
            ("__all__", "privacy", "rank"),
            ("__all__", "fairness", "rank"),
            "Privacy rank",
            "Fairness rank",
            "Privacy vs Fairness Trade-off",
            "privacy_vs_fairness",
        ),
    ]
    for x_key, y_key, x_label, y_label, title, filename in pairs:
        missing_keys = [key for key in (x_key, y_key) if key not in combined.columns]
        if missing_keys:
            logger.warning("Skipping %s; missing rank columns: %s", filename, missing_keys)
            continue
        figure = plot_rank_tradeoff(combined, x_key, y_key, x_label, y_label, title)
        try:
            save_matplotlib_figure(figure, output_dir / filename, cfg.plots.dpi, cfg.plots.formats)
        finally:
            plt.close(figure)


def save_log_disparity_plots(
    log_disparity_reports: dict[str, dict], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir) / "evaluation" / "log_disparity"
    for name, report in log_disparity_reports.items():
        fig = report.get("report_figure")
        if fig is None and "error" not in report:
            from synthdata.log_disparity.metric_log_disparity import (
                build_log_disparity_report_figure,
            )

            fig = build_log_disparity_report_figure(report)
        if fig is None:
            if "error" in report:
                logger.warning(
                    "[plot] skipping persisted failed log-disparity report for model %s: %s",
                    name,
                    report["error"],
                )
            continue
        save_plotly_figure(fig, output_dir / name, ("html",))


#: Short note shown with each direction in the metric overview titles.
_DIRECTION_NOTE = {
    "higher": "higher is better",
    "lower": "lower is better",
    "closer to 0": "closer to 0 is better",
    "unranked": "reported, not ranked",
    "unknown": "direction unknown",
}


#: Metrics of each type, most important first (the panel order of the
#: overview figures). Each entry is a name or a name prefix; metrics matching
#: none keep their order after the listed ones. Utility: how well models
#: trained on synthetic data predict real holdout rows, then whether real
#: and synthetic rows can be told apart, then joint and marginal fidelity,
#: then the synthetic-only scores and real-data references. Privacy: attacks
#: with a calibrated baseline (Anonymeter, distance MIA), then distance to
#: train against holdout, then the other attack and distance scores, then
#: the k-anonymity family. Fairness: how far a synthetic-trained model's group
#: gaps are from the real-trained model's, then subgroup privacy, then the
#: within-synthetic gaps.
METRIC_PRIORITY = {
    "utility": [
        "tstr_macro_auprc",
        "tstr_macro_f1",
        "tstr_balanced_accuracy",
        "avg_F1_diff_hout",
        "performance.xgb.syn_ood",
        "performance.mlp.syn_ood",
        "performance.linear_model.syn_ood",
        "detection.detection_xgb",
        "auroc",
        "avg_pMSE",
        "detection.",
        "stats.alpha_precision",
        "stats.prdc",
        "corr_mat_diff",
        "mutual_inf_diff",
        "stats.max_mean_discrepancy",
        "stats.wasserstein_dist",
        "pca_",
        "avg_dwm_diff",
        "avg_cio",
        "ks_tvd_stat",
        "frac_ks_sigs",
        "avg_h_dist",
        "stats.",
        "avg_qMSE",
        "nnaa",
        "performance.feat_rank_distance",
        "avg_F1_diff",
        "performance.",
        "tsts_",
        "sanity.",
    ],
    "privacy": [
        "anonymeter_",
        "distance_mia_auc",
        "dcr_closer_to_train_share",
        "dcr_holdout_ratio",
        "nndr_holdout_ratio",
        "mia_",
        "priv_loss_nnaa",
        "priv_loss_nndr",
        "eps_identif_risk",
        "priv_loss_eps",
        "privacy.identifiability_score",
        "att_discl_risk",
        "attack.",
        "hit_rate",
        "median_DCR",
        "avg_nndr",
        "privacy.delta-presence",
        "privacy.k-map",
        "privacy.k-anonymization",
        "privacy.distinct l-diversity",
    ],
    "fairness": [
        "equalized_odds_gap_error",
        "equal_opportunity_gap_error",
        "demographic_parity_gap_error",
        "subgroup_distance_mia_auc_gap",
        "subgroup_distance_mia_auc_max",
        "log_disparity_mean_abs",
        "log_disparity_share_significant",
        "log_disparity_",
        "equalized_odds",
        "eqo_",
        "equal_opportunity",
        "eo_",
        "statistical_parity",
        "sp_",
        "fairness_signal",
    ],
}


def metric_priority(metric_type: str, metric: str) -> int:
    """Position of ``metric`` in :data:`METRIC_PRIORITY` (unlisted metrics last)."""
    order = METRIC_PRIORITY.get(metric_type, [])
    for i, entry in enumerate(order):
        if metric == entry or metric.startswith(entry):
            return i
    return len(order)


def metric_overview_table(combined: pd.DataFrame, metric: tuple) -> pd.DataFrame:
    """Mean and 95% interval over seed replicates of one raw metric, per model.

    Rows are base models (``"ctgan__rep2"`` counts as ``"ctgan"``); columns
    ``mean``, ``ci_low``, ``ci_high`` and ``n`` (NaN interval for one seed).
    """
    from synthdata.evaluation.combine import mean_and_interval

    values = pd.to_numeric(combined[metric], errors="coerce")
    models = pd.Series(
        [split_replicate_name(name)[0] for name in combined.index], index=combined.index
    )
    rows = {}
    for model, names in models.groupby(models, sort=False).groups.items():
        mean, low, high = mean_and_interval(values[names], 0.95)
        rows[model] = {
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "n": values[names].notna().sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_metric_overview(
    combined: pd.DataFrame, directions: pd.DataFrame, metric_type: str, n_cols: int = 4
):
    """One small panel per raw metric of ``metric_type``: each model's mean over
    seeds with its 95% interval, the better direction in the title, the ideal
    value as a green line where one exists, and baselines in grey.
    Returns None when the type has no metrics.
    """
    import math

    from synthdata.evaluation.baselines import is_baseline

    metrics = [
        key
        for key in combined.columns
        if key[0] != "__all__"
        and key[1] == metric_type
        and key[2] != "rank"
        and pd.to_numeric(combined[key], errors="coerce").notna().any()
    ]
    if not metrics:
        return None
    metrics = sorted(metrics, key=lambda key: metric_priority(metric_type, key[2]))
    first = metric_overview_table(combined, metrics[0])
    models = sorted(first.index, key=model_sort_key)
    n_rows = math.ceil(len(metrics) / n_cols)
    height = max(1.6, 0.22 * len(models) + 0.9)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, height * n_rows + 0.4), squeeze=False, sharey=True
    )
    y = range(len(models))
    for ax, key in zip(axes.flat, metrics, strict=False):
        table = metric_overview_table(combined, key).reindex(models)
        direction = (
            directions["direction"].get(key, "unknown") if directions is not None else "unknown"
        )
        ideal = (
            directions["ideal"].get(key, float("nan")) if directions is not None else float("nan")
        )
        for i, model in enumerate(models):
            row = table.loc[model]
            color = "#8a8a8a" if is_baseline(model) else "#2a6fb0"
            if pd.notna(row["ci_low"]):
                ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=1.6)
            ax.plot(row["mean"], i, "o", color=color, ms=4.5)
        if pd.notna(ideal):
            ax.axvline(ideal, color="#2e8b57", lw=1.2, ls="--")
        arrow = {"higher": " →", "lower": " ←"}.get(direction, "")
        ax.set_title(f"{key[2]}\n({_DIRECTION_NOTE.get(direction, direction)}{arrow})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticks(list(y), models)
        ax.invert_yaxis()
    # No figure title: report.md explains the panels, and a title overlapped
    # the first row of panel titles.
    fig.tight_layout()
    return fig


def save_metric_overview_plots(
    cfg: Config, combined: pd.DataFrame, directions: pd.DataFrame | None, output_dir: str | Path
) -> None:
    """Save ``metrics_<type>.png`` (utility, privacy, fairness) under ``<output_dir>/evaluation``."""
    output_dir = Path(output_dir) / "evaluation"
    for metric_type in ("utility", "privacy", "fairness"):
        fig = plot_metric_overview(combined, directions, metric_type)
        if fig is None:
            continue
        try:
            save_matplotlib_figure(
                fig, output_dir / f"metrics_{metric_type}", cfg.plots.dpi, cfg.plots.formats
            )
        finally:
            plt.close(fig)
