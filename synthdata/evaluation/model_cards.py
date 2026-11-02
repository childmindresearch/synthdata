"""``model_cards/<model>.md``: one model card per candidate model of a run.

The layout follows Model Cards (Mitchell et al., FAT* 2019), cut down to what a
synthetic-data release needs: what the model is and how it was built, what it
is for, its primary metrics against the two baselines, privacy and fairness
evidence, the output checks it warned on, and caveats. Every value is read from
the run's own outputs, so a card says nothing the report does not. The section
headings are those of ``docs/model-card-template.md`` (a unit test keeps the
two in step).
"""

import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from synthdata.evaluation.baselines import is_baseline
from synthdata.evaluation.primary import BEST_ANCHOR, COPY, MARGINALS
from synthdata.utils import get_logger, split_replicate_name

logger = get_logger(__name__)

CARDS_DIRNAME = "model_cards"
#: Section headings, in order (docs/model-card-template.md uses the same).
SECTIONS = (
    "Model details",
    "Intended use",
    "Ranking",
    "Primary metrics",
    "Utility",
    "Privacy",
    "Fairness",
    "Known issues",
    "Caveats",
)

_INTENDED_USE = [
    "- **For**: comparing generators on this dataset, developing and testing analysis code, "
    "and teaching, where rows that look like the real data but are not real patients are enough.",
    "- **Not for**: clinical decisions about any person, estimates that will be reported as "
    "facts about the real population without checking them on the real data, or release "
    "outside the team before a privacy review of this card and the report.",
]

_CAVEATS = [
    "- Scores are relative to this run: anchored scores place the model between "
    "`baseline_marginals` and `baseline_train_copy` on the same holdout, so they move when "
    "the data, the split or the metric set changes.",
    "- The privacy evidence is a set of attacks that failed or succeeded on this holdout. "
    "It is not a formal guarantee (no differential privacy), and a stronger attack may find "
    "more.",
    "- Intervals cover seed replicates and holdout resampling only (see report.md, Ranking). "
    "A small holdout leaves wide intervals, so read small gaps between models as ties.",
    "- The synthetic data inherits the real data's biases; the fairness gaps say whether they "
    "are kept, not whether they are acceptable.",
]


def _package_version() -> str:
    try:
        return version("synthdata")
    except PackageNotFoundError:
        return "unknown"


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _base(name) -> str:
    return split_replicate_name(str(name))[0]


def _read(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (ValueError, pd.errors.ParserError) as exc:
        logger.warning("[model cards] could not read %s: %s", path, exc)
        return None


def _model_means(table: pd.DataFrame | None, columns: list) -> pd.DataFrame:
    """Mean of ``columns`` per model, over its replicates (``model`` holds replicate names)."""
    if table is None or "model" not in table.columns:
        return pd.DataFrame(columns=columns)
    present = [c for c in columns if c in table.columns]
    numeric = table[present].apply(pd.to_numeric, errors="coerce")
    return numeric.groupby(table["model"].map(_base)).mean()


def _generation_provenance(manifest_path: Path | None) -> dict:
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("[model cards] could not read %s: %s", manifest_path, exc)
        return {}
    runs = [r for r in manifest.get("runs", []) if r.get("stage") == "generation"]
    run = runs[-1] if runs else {}
    return {"config_sha256": run.get("config_sha256"), **(run.get("provenance") or {})}


def _hyperparameters(best_params: dict, model: str) -> tuple[str, dict | None]:
    """How the model's settings were chosen, and the tuned settings when there are any."""
    if not model.endswith("_hpo"):
        return "untuned: library defaults plus the settings in the config", None
    base = model.removesuffix("_hpo")
    for framework in best_params.values():
        if isinstance(framework, dict) and base in framework:
            params = framework[base]
            if not params:
                return (
                    "tuned model fitted with the untuned settings: no search trial passed "
                    "the screens, so it repeats the untuned model",
                    None,
                )
            return "tuned by the hyperparameter search (best trial below)", params
    return "tuned; the search result is not in the best-params file", None


def _details(model, cfg, dataset, experiment, summary_row, best_params, provenance, rows) -> list:
    how, params = _hyperparameters(best_params, model)
    replicates = summary_row.get("n_replicates") if summary_row is not None else None
    commit = provenance.get("git_commit")
    lines = [
        f"- **Generator**: `{model.removesuffix('_hpo')}`; {how}.",
        f"- **Seed replicates**: {_fmt(replicates, 0)} (base seed `{cfg.seed}`); "
        f"**synthetic rows per replicate**: {rows}.",
        f"- **Trained on**: the train split of `{dataset.name}`"
        + (f" (version `{dataset.version}`)" if dataset.version else "")
        + f", target `{dataset.target_column}`.",
    ]
    if experiment is not None:
        lines.append(f"- **Experiment**: `{experiment.id}`.")
    if provenance:
        packages = provenance.get("packages") or {}
        lines.append(
            "- **Provenance**: "
            + ", ".join(
                part
                for part in (
                    f"commit `{commit[:12]}`" if commit else "",
                    "with uncommitted changes" if provenance.get("git_dirty") else "",
                    f"uv.lock sha256 `{str(provenance.get('lockfile_sha256'))[:12]}`"
                    if provenance.get("lockfile_sha256")
                    else "",
                    f"config sha256 `{str(provenance.get('config_sha256'))[:12]}`"
                    if provenance.get("config_sha256")
                    else "",
                    ", ".join(f"{k} {v}" for k, v in sorted(packages.items())),
                )
                if part
            )
            + " (full record in the experiment's `manifest.json`)."
        )
    if params:
        lines += ["", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```"]
    return lines


def _ranking(summary_row) -> list:
    if summary_row is None:
        return ["Not in `ranking_summary.csv`."]
    lines = [
        f"- **Rank**: {_fmt(summary_row.get('rank'), 0)}; **overall score** "
        f"{_fmt(summary_row.get('overall_mean'))} (95% interval "
        f"{_fmt(summary_row.get('overall_ci_low'))} to {_fmt(summary_row.get('overall_ci_high'))}).",
    ]
    dims = [
        f"{dim} {_fmt(summary_row.get(f'{dim}_mean'))}"
        for dim in ("utility", "privacy", "fairness")
        if pd.notna(summary_row.get(f"{dim}_mean"))
    ]
    if dims:
        lines.append("- **Type scores**: " + ", ".join(dims) + ".")
    if pd.notna(summary_row.get("diff_vs_best")):
        lines.append(
            f"- **Gap to the best model**: {_fmt(summary_row.get('diff_vs_best'))} (95% interval "
            f"{_fmt(summary_row.get('diff_ci_low'))} to {_fmt(summary_row.get('diff_ci_high'))}); "
            + (
                "tied with the best within noise."
                if str(summary_row.get("tied_with_best")).lower() == "true"
                else "below the best."
            )
        )
    if "pareto_front" in summary_row.index:
        on_front = str(summary_row.get("pareto_front")).lower() == "true"
        lines.append(
            "- **Utility-privacy Pareto front**: "
            + ("yes" if on_front else "no")
            + (
                f" (on the front in {_fmt(summary_row.get('pareto_share'), 2)} of the bootstrap draws)"
                if pd.notna(summary_row.get("pareto_share"))
                else ""
            )
            + "."
        )
    return lines


def _primary(primary: pd.DataFrame | None, model: str) -> list:
    if primary is None or primary.empty:
        return ["No primary metric set in this run (`evaluation.primary_metrics: []`)."]
    rows = primary[primary["model"] == model]
    if rows.empty:
        return ["No primary metric for this model."]
    lines = [
        "Anchored score: 0 at the worse baseline, 1 at the better one, so it is the share of "
        "the gap between them this model covers (it can leave [0, 1]).",
        "",
        "| Metric | Type | Value | 0 at | 1 at | Anchored score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (metric, kind), group in rows.groupby(["metric", "type"], sort=False):
        best = BEST_ANCHOR.get(str(kind), COPY)
        worst = MARGINALS if best == COPY else COPY
        lines.append(
            f"| `{metric}` | {kind} | {_fmt(group['raw'].mean())} | "
            f"`{worst}` {_fmt(group['anchor_0'].mean())} | `{best}` {_fmt(group['anchor_1'].mean())} | "
            f"{_fmt(group['anchored'].mean())} |"
        )
    return lines


def _comparison(
    means: pd.DataFrame, model: str, columns: list, labels: dict, header: str = "Metric"
) -> list:
    """Table of ``columns`` for the model and both baselines."""
    present = [c for c in columns if c in means.columns]
    if model not in means.index or not present:
        return []
    names = [model] + [b for b in (MARGINALS, COPY) if b in means.index]
    lines = [
        f"| {header} | " + " | ".join(f"`{n}`" for n in names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for column in present:
        lines.append(
            f"| {labels.get(column, f'`{column}`')} | "
            + " | ".join(_fmt(means.loc[n, column]) for n in names)
            + " |"
        )
    return lines


def _utility(tstr: pd.DataFrame | None, model: str) -> list:
    means = _model_means(tstr, ["tstr_macro_auprc", "tstr_kappa", "tsts_macro_auprc", "tsts_kappa"])
    lines = _comparison(
        means,
        model,
        list(means.columns),
        {
            "tstr_macro_auprc": "TSTR macro AUPRC (train on synthetic, test on real)",
            "tstr_kappa": "TSTR Cohen's kappa",
            "tsts_macro_auprc": "TSTS macro AUPRC (train and test on synthetic)",
            "tsts_kappa": "TSTS Cohen's kappa",
        },
    )
    if not lines:
        return ["No holdout TSTR table in this run."]
    if {"tstr_macro_auprc", "tsts_macro_auprc"} <= set(means.columns):
        gap = means.loc[model, "tsts_macro_auprc"] - means.loc[model, "tstr_macro_auprc"]
        lines += [
            "",
            f"TSTS minus TSTR macro AUPRC: {_fmt(gap)}. A large positive gap means the "
            "synthetic data holds patterns the real data does not, so a model trained on it "
            "looks better on synthetic data than it is on real patients.",
        ]
    return lines


def _privacy(attacks: pd.DataFrame | None, model: str) -> list:
    if attacks is None or attacks.empty:
        return ["No Anonymeter attack in this run."]
    attacks = attacks.assign(
        name=attacks["attack"].astype(str)
        + attacks.get("secret", pd.Series("", index=attacks.index))
        .fillna("")
        .map(lambda s: f" ({s})" if str(s) else "")
    )
    wide = (
        attacks.assign(model=attacks["model"].map(_base))
        .groupby(["model", "name"])["risk"]
        .mean()
        .unstack("name")
    )
    lines = _comparison(wide, model, sorted(wide.columns), {}, header="Attack")
    if not lines:
        return ["No Anonymeter attack for this model."]
    return [
        "Anonymeter risk per attack (0 = the attacker does no better than on unseen "
        "patients), mean over replicates; intervals in `privacy_attacks.csv`.",
        "",
        *lines,
    ]


def _fairness(gaps: pd.DataFrame | None, model: str) -> list:
    columns = [
        "demographic_parity_gap_error",
        "equalized_odds_gap_error",
        "equal_opportunity_gap_error",
        "holdout_balanced_accuracy",
    ]
    lines = _comparison(_model_means(gaps, columns), model, columns, {})
    if not lines:
        return ["No holdout fairness gaps in this run."]
    return [
        "Gap errors: how far the protected-group gaps of a classifier trained on synthetic "
        "data are from those of one trained on real data (0 = the real data's gaps kept).",
        "",
        *lines,
    ]


def _issues(checks: pd.DataFrame | None, combined: pd.DataFrame, model: str, how: str) -> list:
    issues = []
    if "untuned settings" in how:
        issues.append("The search found no trial passing the screens (see Model details).")
    if checks is not None and not checks.empty:
        own = checks[(checks["model"].map(_base) == model) & (checks["status"] == "warn")]
        for row in own.itertuples():
            issues.append(
                f"Check `{row.check}` on `{row.model}`: {_fmt(row.value)} (limit "
                f"{_fmt(row.limit)})"
                + (f"; {row.detail}" if pd.notna(row.detail) and row.detail else "")
                + "."
            )
    raw = combined[
        [c for c in combined.columns if c[0] not in ("__all__", "legacy") and c[2] != "rank"]
    ]
    rows = raw[[_base(n) == model for n in raw.index]]
    missing = sorted({c[2] for c in rows.columns[rows.isna().any() & raw.notna().any()]})
    if missing:
        issues.append(
            f"{len(missing)} metric(s) missing for at least one replicate (failed or skipped): "
            + ", ".join(f"`{m}`" for m in missing[:10])
            + (" ..." if len(missing) > 10 else "")
            + "."
        )
    if not issues:
        return ["None found: every output check passed and every metric was computed."]
    return [f"- {issue}" for issue in issues]


def build_model_card(
    model: str, cfg, dataset, combined: pd.DataFrame, tables: dict, experiment=None
) -> str:
    """The Markdown card of one candidate model; ``tables`` holds the run's outputs."""
    summary = tables.get("ranking_summary")
    summary_row = summary.loc[model] if summary is not None and model in summary.index else None
    how, _ = _hyperparameters(tables.get("best_params", {}), model)
    rows = cfg.generation.rows_for(len(dataset.train_df))
    body = {
        "Model details": _details(
            model,
            cfg,
            dataset,
            experiment,
            summary_row,
            tables.get("best_params", {}),
            tables.get("provenance", {}),
            rows,
        ),
        "Intended use": _INTENDED_USE,
        "Ranking": _ranking(summary_row),
        "Primary metrics": _primary(tables.get("primary"), model),
        "Utility": _utility(tables.get("tstr"), model),
        "Privacy": _privacy(tables.get("attacks"), model),
        "Fairness": _fairness(tables.get("fairness_gaps"), model),
        "Known issues": _issues(tables.get("checks"), combined, model, how),
        "Caveats": _CAVEATS,
    }
    lines = [
        f"# Model card: `{model}`",
        "",
        f"Written by synthdata {_package_version()} on "
        f"{datetime.now(UTC).strftime('%Y-%m-%d')}, from this run's outputs. "
        "The evidence behind every number is in [report.md](../report.md).",
    ]
    for heading in SECTIONS:
        lines += ["", f"## {heading}", "", *body[heading]]
    return "\n".join(lines) + "\n"


def write_model_cards(
    cfg,
    dataset,
    combined: pd.DataFrame,
    evaluation_dir: Path,
    generation_dir: Path,
    manifest_path: Path | None = None,
    experiment=None,
) -> list[Path]:
    """Write ``model_cards/<model>.md`` for every candidate model; returns the paths."""
    from synthdata.generation.hpo import default_best_params_path
    from synthdata.utils import load_json

    evaluation_dir = Path(evaluation_dir)
    summary = _read(evaluation_dir / "ranking_summary.csv", index_col=0)
    best_path = cfg.generation.hpo.best_params_path or default_best_params_path(generation_dir)
    tables = {
        "ranking_summary": summary,
        "primary": _read(evaluation_dir / "primary_metrics.csv"),
        "tstr": _read(evaluation_dir / "tstr_holdout.csv"),
        "attacks": _read(evaluation_dir / "privacy_attacks.csv"),
        "fairness_gaps": _read(evaluation_dir / "fairness_gaps.csv"),
        "checks": _read(evaluation_dir / "checks.csv"),
        "best_params": load_json(Path(best_path), default={}),
        "provenance": _generation_provenance(manifest_path),
    }
    models = sorted({_base(m) for m in combined.index if not is_baseline(_base(m))})
    folder = evaluation_dir / CARDS_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in models:
        path = folder / f"{model}.md"
        path.write_text(build_model_card(model, cfg, dataset, combined, tables, experiment))
        paths.append(path)
    logger.info("[model cards] wrote %d model cards to %s", len(paths), folder)
    return paths
