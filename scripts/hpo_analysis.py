"""Inspect a finished HPO run's Optuna studies (``optuna_studies.db``).

Three views, each printed and written as CSV next to ``--out``:

``summary``
    One row per study: trials by state, feasible trials, the objective, the
    best trial and how many trials and trial-hours came after it.
``compare``
    Which trial each logged score (or a weighted composite of them) would have
    chosen, per study, so a different objective can be judged without a new
    run. Scores where lower is better (``fid_*``) are negated.
``importance``
    fANOVA parameter importance per study, on a logged score, with the
    spread over bootstrap resamples of the trials. A parameter is worth
    fixing only when its importance is low in most resamples and the score's
    spread across trials is above its bootstrap SE.

Examples::

    uv run python scripts/hpo_analysis.py summary output/sim/.../optuna_studies.db
    uv run python scripts/hpo_analysis.py compare optuna_studies.db \\
        --composite tstr_macro_auprc=1,fid_marginal=0.5
    uv run python scripts/hpo_analysis.py importance optuna_studies.db --metric tstr_macro_auprc
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from synthdata.generation.hpo import STUDY_OBJECTIVE_ATTR, _feasible

#: Logged scores compared by ``compare`` when present.
COMPARED = (
    "tstr_macro_auprc",
    "tstr_macro_auroc",
    "tstr_macro_f1",
    "tstr_worst_group_auprc",
    "fid_marginal",
    "fid_association",
    "fid_detection_auc",
)


def _url(path: str) -> str:
    return path if "://" in path else f"sqlite:///{Path(path).resolve()}"


def _studies(url: str) -> list[optuna.Study]:
    names = [s.study_name for s in optuna.get_all_study_summaries(url)]
    return [
        optuna.load_study(study_name=n, storage=url)
        for n in sorted(names)
        if not n.endswith("_probe")
    ]


def _hours(trials) -> float:
    return sum(t.duration.total_seconds() for t in trials if t.duration) / 3600


def summary(url: str) -> pd.DataFrame:
    rows = []
    for study in _studies(url):
        trials = sorted(study.trials, key=lambda t: t.number)
        feasible = [t for t in trials if _feasible(t)]
        best = max(feasible, key=lambda t: t.value) if feasible else None
        after = [t for t in trials if best is not None and t.number > best.number]
        rows.append(
            {
                "study": study.study_name,
                "objective": study.user_attrs.get(STUDY_OBJECTIVE_ATTR, "(not stored)"),
                "trials": len(trials),
                **{
                    state.name.lower(): sum(t.state == state for t in trials)
                    for state in (
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                        optuna.trial.TrialState.FAIL,
                    )
                },
                "feasible": len(feasible),
                "best_trial": best.number if best else None,
                "best_value": best.value if best else None,
                "ties_at_best": sum(
                    # Feasible trials are complete, so their value is never None.
                    np.isclose(t.value, best.value)  # ty: ignore[no-matching-overload]
                    for t in feasible
                )
                if best
                else 0,
                "trials_after_best": len(after),
                "hours_after_best": round(_hours(after), 2),
                "trial_hours": round(_hours(trials), 2),
            }
        )
    return pd.DataFrame(rows)


def _parse_composite(text: str | None) -> dict:
    if not text:
        return {}
    weights = {}
    for part in text.split(","):
        name, _, weight = part.partition("=")
        weights[name.strip()] = float(weight or 1)
    return weights


def _oriented(trial, name: str) -> float | None:
    value = trial.user_attrs.get(name)
    if value is None or not np.isfinite(value):
        return None
    return -float(value) if name.startswith("fid_") else float(value)


def compare(url: str, composite: dict) -> pd.DataFrame:
    rows = []
    for study in _studies(url):
        feasible = [t for t in study.trials if _feasible(t)]
        if not feasible:
            continue
        frame = pd.DataFrame(
            [
                {"number": t.number, **{n: _oriented(t, n) for n in COMPARED if n in t.user_attrs}}
                for t in feasible
            ]
        ).set_index("number")
        scores = {name: frame[name] for name in frame.columns}
        if composite:
            missing = [n for n in composite if n not in frame]
            if missing:
                raise SystemExit(f"{study.study_name}: no logged score {missing}")
            # Rank-normalize so scores on different scales weigh as given.
            ranks = frame[list(composite)].rank(pct=True)
            scores["composite"] = sum(ranks[n] * w for n, w in composite.items())
        for name, values in scores.items():
            values = values.dropna()
            if values.empty:
                continue
            pick = values.idxmax()
            rows.append(
                {
                    "study": study.study_name,
                    "chosen_by": name,
                    "trial": int(pick),
                    **{n: frame.at[pick, n] for n in frame.columns},
                }
            )
    return pd.DataFrame(rows)


def importance(url: str, metric: str, n_bootstrap: int, seed: int) -> pd.DataFrame:
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    rows = []
    for study in _studies(url):
        trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and metric in t.user_attrs
        ]
        if len(trials) < 5:
            continue
        names = sorted(set.intersection(*[set(t.params) for t in trials]))
        names = [n for n in names if len({str(t.params[n]) for t in trials}) > 1]
        if not names:
            continue
        rng = np.random.default_rng(seed)
        draws = []
        for b in range(n_bootstrap + 1):
            idx = rng.choice(len(trials), len(trials)) if b else np.arange(len(trials))
            sub = optuna.create_study(direction="maximize")
            for i in idx:
                t = trials[i]
                sub.add_trial(
                    optuna.trial.create_trial(
                        params={n: t.params[n] for n in names},
                        distributions={n: t.distributions[n] for n in names},
                        value=t.user_attrs[metric],
                    )
                )
            draws.append(get_param_importances(sub, evaluator=FanovaImportanceEvaluator(seed=b)))
        frame = pd.DataFrame(draws).fillna(0.0)
        top3 = frame.rank(axis=1, ascending=False) <= 3
        values = [t.user_attrs[metric] for t in trials]
        ses = [
            t.user_attrs.get("tstr_boot_se", {}).get(metric.removeprefix("tstr_")) for t in trials
        ]
        ses = [s for s in ses if s is not None]
        for name in frame.columns:
            rows.append(
                {
                    "study": study.study_name,
                    "param": name,
                    "importance": round(frame[name].iloc[0], 3),
                    "boot_median": round(frame[name].median(), 3),
                    "boot_p10": round(frame[name].quantile(0.1), 3),
                    "boot_p90": round(frame[name].quantile(0.9), 3),
                    "top3_share": round(top3[name].mean(), 2),
                    "metric_sd_across_trials": round(float(np.std(values, ddof=1)), 4),
                    "metric_boot_se": round(float(np.median(ses)), 4) if ses else None,
                    "n_trials": len(trials),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["study", "boot_median"], ascending=[True, False]) if len(out) else out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("view", choices=["summary", "compare", "importance"])
    parser.add_argument("storage", help="optuna_studies.db path or an Optuna storage URL")
    parser.add_argument("--out", help="CSV path (default: <view>.csv next to the database)")
    parser.add_argument(
        "--composite", help="compare: weighted scores, e.g. tstr_macro_auprc=1,fid_marginal=0.5"
    )
    parser.add_argument("--metric", default="tstr_macro_auprc", help="importance: logged score")
    parser.add_argument("--n-bootstrap", type=int, default=30, help="importance: resamples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    url = _url(args.storage)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        if args.view == "summary":
            table = summary(url)
        elif args.view == "compare":
            table = compare(url, _parse_composite(args.composite))
        else:
            table = importance(url, args.metric, args.n_bootstrap, args.seed)
    out = Path(args.out) if args.out else Path(args.storage).with_name(f"hpo_{args.view}.csv")
    if "://" not in str(out):
        table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
