"""Generic Optuna study management shared by all generation backends.

Provides:
- ``build_hpo_eval_fn``: scores a candidate synthetic DataFrame on the tuning
  split. The default objective is TSTR macro-F1 of a fixed XGBoost
  (:mod:`synthdata.evaluation.tstr`); privacy is reported by evaluation, not
  optimized. It also computes the sanity screens that Optuna treats as
  constraints (``screen_violations``).
- ``score_candidate``: records a candidate's screens and logged scores on its
  Optuna trial and returns the objective value.
- ``hpo_score``/``build_synthetic_eval_fn``: the older direction-aware
  synthcity composite (``hpo.objective: synthcity_composite``).
- ``create_study``/``run_study``: Optuna study creation with SQLite-backed
  persistence (resumable across runs, inspectable with optuna-dashboard).
- ``BestParamsCache``: JSON-backed cache of best hyperparameters per model,
  keyed by generator family (``synthcity`` / ``tabpfgen``), mirroring
  ``output/hepatitis/hpo_best_params.json`` from the notebooks.
"""

import dataclasses
import os
import re
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import optuna
import pandas as pd
from filelock import FileLock

from synthdata.config import HPOConfig, HPOConstraintsConfig
from synthdata.evaluation.tstr import (
    XGB_PARAMS,
    _encode,
    chance_scores,
    target_classes,
    tstr_scores,
)
from synthdata.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def hpo_score(report_df: pd.DataFrame) -> float:
    """Direction-aware composite score: orient metrics so higher=better, negate mean.

    ``report_df`` must have ``mean`` and ``direction`` columns (as returned by
    synthcity's ``Metrics.evaluate``/``Benchmarks.evaluate``). The result is
    suitable as an Optuna objective under ``direction="minimize"``.
    """
    sign = report_df["direction"].map({"maximize": 1.0, "minimize": -1.0})
    return -(report_df["mean"] * sign).mean()


def build_synthetic_eval_fn(
    train_reference_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    sensitive_features: list,
    metric_config: dict,
    seed: int,
    workspace: str | Path | None = None,
    *,
    discrete_columns: list,
) -> Callable[[pd.DataFrame], float]:
    """Build a ``syn_df -> score`` function via synthcity's Metrics.evaluate.

    ``discrete_columns`` are the columns synthcity's metrics treat as categorical.

    Mirrors the notebooks' ``_eval_syn_df`` helper: builds a second independent
    synthetic draw (bootstrap resample) for DomiasMIA's reference set, and an
    augmented train+synthetic set for augmentation metrics.
    """
    from synthcity.metrics import Metrics
    from synthcity.plugins.core.dataloader import GenericDataLoader
    from synthcity.utils.dataframe import declared_discrete_columns

    workspace_path = Path(workspace) if workspace else Path("workspace")

    def _loader(df: pd.DataFrame) -> GenericDataLoader:
        return GenericDataLoader(
            df, target_column=target_column, sensitive_features=sensitive_features
        )

    def eval_fn(syn_df: pd.DataFrame) -> float:
        ref_df = syn_df.sample(n=len(syn_df), replace=True, random_state=seed + 1).reset_index(
            drop=True
        )
        x_aug = pd.concat([train_reference_df, syn_df], ignore_index=True)
        with declared_discrete_columns(discrete_columns):
            report = Metrics.evaluate(
                _loader(holdout_df),
                _loader(syn_df),
                _loader(train_reference_df),
                _loader(ref_df),
                _loader(x_aug),
                metrics=metric_config,
                # The target is discrete exactly when the schema declares it categorical.
                task_type="classification" if target_column in discrete_columns else "regression",
                random_state=seed,
                workspace=workspace_path,
                # The cache keys on data and metric name, not code; see synthcity_eval.
                use_cache=False,
            )
        return hpo_score(report)

    return eval_fn


#: Constraint order stored on every trial; a value above 0 marks it infeasible.
SCREENS = (
    "copies",
    "missing_classes",
    "category_coverage",
    "out_of_range",
    "class_share_gap",
    "protected_coverage",
    "protected_share_gap",
    "dcr_closer_to_train",
)

#: Trial user attribute of a TSTR objective: ``tstr_macro_auprc`` -> its value.
#: Study user attribute holding the objective a study optimizes (the choice of
#: ``objective: auto``); trials take their value from that logged score.
STUDY_OBJECTIVE_ATTR = "objective"
#: User attribute that makes an objective function fit with another seed
#: (the top-k refit; see :func:`trial_seed`).
REFIT_SEED_ATTR = "refit_seed"

#: Categories rarer than this in train are not required in the synthetic data.
MIN_CATEGORY_FREQUENCY = 0.01


@dataclasses.dataclass
class TrialScore:
    """One candidate's objective value, constraint violations and logged scores."""

    value: float
    constraints: list = dataclasses.field(default_factory=lambda: [0.0] * len(SCREENS))
    attrs: dict = dataclasses.field(default_factory=dict)


class HPOEvalFn(Protocol):
    """Scores one synthetic dataset on the tuning split (see :func:`build_hpo_eval_fn`).

    ``headroom`` holds, per TSTR objective, the real-data ceiling, the chance
    level and their distance in bootstrap SEs; empty for a regression target.
    """

    headroom: dict

    def __call__(self, synthetic_df: pd.DataFrame, /) -> TrialScore: ...


def objective_direction(objective: str) -> str:
    return "minimize" if objective == "synthcity_composite" else "maximize"


def trial_seed(trial, seed: int) -> int:
    """The seed an objective fits with: ``seed``, or the top-k refit's seed."""
    return int(trial.user_attrs.get(REFIT_SEED_ATTR, seed))


def protected_groups(
    reference: pd.DataFrame, protected_columns: list, n_bins: int = 4
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Per protected column, a function mapping rows to group labels.

    Categorical columns (or numeric ones with at most 10 values in
    ``reference``) group by value; other numeric columns by the reference
    quartiles (``n_bins`` quantile bins), so age-like columns form groups.
    """
    groupers = {}
    for column in protected_columns:
        if column not in reference:
            continue
        values = reference[column]
        numeric = pd.api.types.is_numeric_dtype(values) and values.nunique() > 10
        if numeric:
            edges = np.unique(np.nanquantile(values.astype(float), np.linspace(0, 1, n_bins + 1)))
            inner = edges[1:-1]

            def grouper(df, column=column, inner=inner):
                codes = np.digitize(pd.to_numeric(df[column], errors="coerce"), inner)
                return pd.Series(
                    np.where(df[column].isna(), "nan", codes.astype(str)), index=df.index
                )
        else:

            def grouper(df, column=column):
                return _comparable(df[column]).astype(str)

        groupers[column] = grouper
    return groupers


def _shares(labels: pd.Series) -> pd.Series:
    return labels.value_counts(normalize=True)


def _tvd(a: pd.Series, b: pd.Series) -> float:
    keys = a.index.union(b.index)
    return float((a.reindex(keys, fill_value=0) - b.reindex(keys, fill_value=0)).abs().sum() / 2)


def protected_scores(
    synthetic_df: pd.DataFrame,
    search_train_df: pd.DataFrame,
    target_column: str,
    groupers: dict,
) -> dict:
    """Protected-group representation of a candidate against search-train.

    ``protected_coverage``: lowest share, over the protected columns, of the
    groups with at least 1% frequency (and at least 5 expected rows) that the
    candidate contains.
    ``protected_share_gap``: largest total variation distance between the
    candidate's and search-train's group shares. ``protected_target_gap``:
    largest total variation distance between the target class shares within
    one group (groups with at least 1% frequency), i.e. whether each group's
    relation to the target survives. Columns missing from the candidate
    count as fully missing.
    """
    coverage, share_gap, target_gap = [], [], []
    for column, grouper in groupers.items():
        real = grouper(search_train_df)
        real_shares = _shares(real)
        # A group expected fewer than 5 times in this many rows may be absent
        # by chance alone.
        floor = max(MIN_CATEGORY_FREQUENCY, 5 / max(len(synthetic_df), 1))
        required = set(real_shares[real_shares >= floor].index)
        if column not in synthetic_df:
            coverage.append(0.0)
            share_gap.append(1.0)
            continue
        syn = grouper(synthetic_df)
        coverage.append(len(required & set(syn)) / len(required) if required else 1.0)
        share_gap.append(_tvd(_shares(syn), real_shares))
        if target_column in synthetic_df:
            for group in required:
                syn_rows = synthetic_df[target_column][syn == group]
                if syn_rows.empty:
                    continue
                target_gap.append(
                    _tvd(
                        _shares(_comparable(syn_rows)),
                        _shares(_comparable(search_train_df[target_column][real == group])),
                    )
                )
    return {
        "protected_coverage": min(coverage) if coverage else 1.0,
        "protected_share_gap": max(share_gap) if share_gap else 0.0,
        "protected_target_gap": max(target_gap) if target_gap else 0.0,
    }


def fidelity_scores(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    categorical_columns: list,
    nominal_columns: list,
    seed: int,
) -> dict:
    """Cheap fidelity of ``synthetic_df`` to held-out real rows; lower is closer.

    ``fid_marginal``: mean over columns of the total variation distance of
    category shares (categorical columns) or the Kolmogorov-Smirnov statistic
    (numeric ones). ``fid_association``: mean absolute difference between the
    two Pearson correlation matrices of the columns as numbers (categories as
    integer codes), a fast proxy for pairwise structure. ``fid_detection_auc``:
    3-fold cross-validated AUC of the fixed XGBoost telling real from
    synthetic rows on equal-size samples (0.5 = indistinguishable).
    Logged for comparison only, never optimized.
    """
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    columns = [c for c in real_df.columns if c in synthetic_df.columns]
    marginal = []
    codes_real, codes_syn = {}, {}
    for column in columns:
        real, syn = real_df[column], synthetic_df[column]
        if column in categorical_columns or not pd.api.types.is_numeric_dtype(real):
            real, syn = _comparable(real).astype(str), _comparable(syn).astype(str)
            marginal.append(_tvd(_shares(syn), _shares(real)))
            levels = pd.Index(sorted(set(real) | set(syn)))
            codes_real[column] = levels.get_indexer(real).astype(float)
            codes_syn[column] = levels.get_indexer(syn).astype(float)
        else:
            real_values = pd.to_numeric(real, errors="coerce").dropna()
            syn_values = pd.to_numeric(syn, errors="coerce").dropna()
            if len(real_values) and len(syn_values):
                marginal.append(float(ks_2samp(real_values, syn_values).statistic))
            codes_real[column] = pd.to_numeric(real, errors="coerce").to_numpy(float)
            codes_syn[column] = pd.to_numeric(syn, errors="coerce").to_numpy(float)

    def _corr(codes):
        frame = pd.DataFrame(codes)
        return frame.fillna(frame.mean()).corr().to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(_corr(codes_real) - _corr(codes_syn))
    upper = diff[np.triu_indices_from(diff, k=1)]
    association = float(np.nanmean(upper)) if np.isfinite(upper).any() else float("nan")

    n = min(len(real_df), len(synthetic_df))
    rows = pd.concat(
        [
            real_df[columns].sample(n=n, random_state=seed),
            synthetic_df[columns].sample(n=n, random_state=seed),
        ],
        ignore_index=True,
    )
    labels = np.r_[np.zeros(n), np.ones(n)]
    x, _ = _encode(rows, rows.iloc[:1], [c for c in nominal_columns if c in columns])
    scores = np.zeros(len(labels))
    for fit_idx, score_idx in StratifiedKFold(3, shuffle=True, random_state=seed).split(x, labels):
        model = XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(x.iloc[fit_idx], labels[fit_idx])
        scores[score_idx] = model.predict_proba(x.iloc[score_idx])[:, 1]
    return {
        "fid_marginal": float(np.mean(marginal)) if marginal else float("nan"),
        "fid_association": association,
        "fid_detection_auc": float(roc_auc_score(labels, scores)),
    }


def _comparable(series: pd.Series) -> pd.Series:
    """Numbers as floats, everything else as strings, so 1, 1.0 and "1" agree."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype(float)
    return series.astype(str)


def _row_hashes(df: pd.DataFrame, columns: list) -> pd.Series:
    normalized = pd.DataFrame({c: _comparable(df[c]) for c in columns})
    return pd.util.hash_pandas_object(normalized, index=False)


def exact_match_rate(rows: pd.DataFrame, reference: pd.DataFrame) -> float:
    """Share of ``rows`` that equal some ``reference`` row on every shared column."""
    columns = [c for c in reference.columns if c in rows.columns]
    if rows.empty:
        return 0.0
    return float(_row_hashes(rows, columns).isin(set(_row_hashes(reference, columns))).mean())


def out_of_range_share(rows: pd.DataFrame, reference: pd.DataFrame, columns: list) -> float:
    """Share of numeric values in ``rows`` outside ``reference``'s [min, max] per column."""
    outside, total = 0, 0
    for column in columns:
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        low, high = reference[column].min(), reference[column].max()
        outside += int(((values < low) | (values > high)).sum())
        total += len(values)
    return outside / total if total else 0.0


def screen_violations(
    synthetic_df: pd.DataFrame,
    search_train_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    target_column: str,
    categorical_columns: list,
    constraints: HPOConstraintsConfig,
) -> dict:
    """Lenient sanity screens; each value is a violation size (<= 0 is feasible).

    - ``copies``: exact-match rate of synthetic rows against search-train,
      minus that of the real tuning rows, minus ``copy_margin``. Calibrating
      on held-out real rows keeps legitimately repeated clinical rows from
      failing the screen.
    - ``missing_classes``: number of real target classes the synthetic data lacks.
    - ``category_coverage``: ``min_category_coverage`` minus the lowest
      coverage of train categories (with at least 1% frequency) over the
      categorical columns.
    - ``out_of_range``: share of numeric values outside the search-train
      range, minus that share for the real tuning rows (fresh real rows fall
      outside too, about 2/n per column), minus ``max_out_of_range``.
    - ``class_share_gap`` is set by :func:`build_hpo_eval_fn`, not here.

    A screen turned off in the config reports -1.
    """
    violations = dict.fromkeys(SCREENS, -1.0)

    if constraints.copy_margin is not None:
        baseline = exact_match_rate(tuning_df, search_train_df)
        violations["copies"] = (
            exact_match_rate(synthetic_df, search_train_df) - baseline - constraints.copy_margin
        )

    if constraints.min_category_coverage is not None:
        real_classes = set(_comparable(search_train_df[target_column]).dropna())
        if target_column in synthetic_df:
            synthetic_classes = set(_comparable(synthetic_df[target_column]).dropna())
        else:
            synthetic_classes = set()
        violations["missing_classes"] = float(len(real_classes - synthetic_classes)) or -1.0
        coverages = []
        for column in categorical_columns:
            if column not in synthetic_df or column not in search_train_df:
                continue
            shares = _comparable(search_train_df[column]).value_counts(normalize=True)
            required = set(shares[shares >= MIN_CATEGORY_FREQUENCY].index)
            if required:
                present = set(_comparable(synthetic_df[column]).dropna())
                coverages.append(len(required & present) / len(required))
        if coverages:
            violations["category_coverage"] = constraints.min_category_coverage - min(coverages)

    if constraints.max_out_of_range is not None:
        numeric = [
            c
            for c in search_train_df.columns
            if c != target_column
            and c not in categorical_columns
            and c in synthetic_df
            and pd.api.types.is_numeric_dtype(search_train_df[c])
        ]
        if numeric:
            violations["out_of_range"] = (
                out_of_range_share(synthetic_df, search_train_df, numeric)
                - out_of_range_share(tuning_df, search_train_df, numeric)
                - constraints.max_out_of_range
            )

    return violations


def class_share_gap(labels: pd.Series, prior: pd.Series) -> float:
    """Total variation distance between the class shares of ``labels`` and ``prior``."""
    shares = labels.value_counts(normalize=True)
    classes = prior.index.union(shares.index)
    return float(
        (shares.reindex(classes, fill_value=0) - prior.reindex(classes, fill_value=0)).abs().sum()
        / 2
    )


def build_hpo_eval_fn(
    search_train_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    target_column: str,
    nominal_columns: list,
    categorical_columns: list,
    target_is_categorical: bool,
    sensitive_features: list,
    hpo_cfg: HPOConfig,
    seed: int,
    match_prior: bool = True,
    workspace: str | Path | None = None,
    protected_columns: Sequence[str] = (),
    ordinal_target: bool = False,
) -> HPOEvalFn:
    """Build a ``synthetic_df -> TrialScore`` function scored on the tuning split.

    Candidates are fitted on ``search_train_df`` and never see the tuning or
    test rows. With ``match_prior``, candidates are drawn to the search-train
    class shares (see :mod:`synthdata.generation.class_quota`), and the
    ``class_share_gap`` screen fails one whose shares still miss them by more
    than ``max_class_share_gap`` because a class fell short. The TSTR scores of the
    same classifier trained on the real search-train rows (TRTR) are the
    ceiling, logged on every trial; with :func:`chance_scores` they give each
    candidate objective's headroom, attached as ``eval_fn.headroom``
    (``{objective: {"trtr", "chance", "se", "z"}}``) for ``objective: auto``.
    ``protected_columns`` get the protected-group screens and per-group TSTR.
    """
    tstr = hpo_cfg.objective.startswith("tstr_") or hpo_cfg.objective == "auto"
    if tstr and not target_is_categorical:
        raise ValueError(
            f"hpo.objective {hpo_cfg.objective!r} needs a categorical target; use "
            "synthcity_composite for a numeric target"
        )
    prior = search_train_df[target_column].value_counts(normalize=True)
    classes = target_classes(pd.Series(prior.index), ordinal_target)
    seeds = [seed + i for i in range(hpo_cfg.tstr_seeds)]
    groupers = protected_groups(search_train_df, list(protected_columns))
    tuning_groups = {column: grouper(tuning_df) for column, grouper in groupers.items()}

    def _tstr(fit_df):
        return tstr_scores(
            fit_df,
            tuning_df,
            target_column,
            nominal_columns,
            classes,
            seeds,
            n_bootstrap=hpo_cfg.n_bootstrap,
            groups=tuning_groups,
            ordinal=ordinal_target,
        )

    def _worst_group(scores):
        values = [v for groups in scores.group_auprc.values() for v in groups.values()]
        return float(min(values)) if values else None

    trtr = None
    headroom = {}
    if target_is_categorical:
        trtr = _tstr(search_train_df)
        chance = chance_scores(tuning_df, target_column, classes)
        for objective in (
            "tstr_macro_auprc",
            "tstr_macro_auroc",
            "tstr_macro_f1",
            "tstr_kappa",
        ):
            metric = objective.removeprefix("tstr_")
            value = getattr(trtr, metric)
            se = trtr.boot_se.get(metric)
            z = (value - chance[metric]) / se if se else float("nan")
            headroom[objective] = {"trtr": value, "chance": chance[metric], "se": se, "z": z}
        logger.info(
            "HPO ceiling (fixed XGBoost trained on real search-train rows, scored on tuning) "
            "vs chance, with headroom in bootstrap SEs: %s",
            "; ".join(
                f"{name.removeprefix('tstr_')}={h['trtr']:.4f} (chance {h['chance']:.4f}, "
                f"z={h['z']:.1f})"
                for name, h in headroom.items()
            ),
        )
        fixed = headroom.get(hpo_cfg.objective)
        if fixed is not None and not fixed["z"] >= hpo_cfg.min_headroom_z:
            logger.warning(
                "hpo.objective %s has little headroom on this data: real-data ceiling %.4f vs "
                "chance %.4f is %.1f bootstrap SEs (< %.1f); the search may only be fitting "
                "noise. Consider objective: auto.",
                hpo_cfg.objective,
                fixed["trtr"],
                fixed["chance"],
                fixed["z"],
                hpo_cfg.min_headroom_z,
            )
    composite_fn = (
        None
        if tstr
        else build_synthetic_eval_fn(
            search_train_df,
            tuning_df,
            target_column,
            sensitive_features,
            hpo_cfg.metric_config,
            seed,
            workspace=workspace,
            discrete_columns=list(categorical_columns)
            + ([target_column] if target_is_categorical else []),
        )
    )

    dcr_reference = None
    if hpo_cfg.constraints.max_dcr_closer_to_train is not None:
        from synthdata.evaluation.privacy_attacks import _encoder, _nearest

        columns = list(search_train_df.columns)
        nominal = [
            c
            for c in columns
            if c in categorical_columns
            or (c == target_column and target_is_categorical)
            or not pd.api.types.is_numeric_dtype(search_train_df[c])
        ]
        encode = _encoder(search_train_df, nominal)
        sample = search_train_df.sample(
            n=min(len(tuning_df), len(search_train_df)), random_state=seed
        )
        dcr_reference = (columns, encode, encode(sample), encode(tuning_df[columns]), _nearest)

    def _dcr_closer_to_train(synthetic_df):
        assert dcr_reference is not None
        columns, encode, x_ref, x_hold, nearest = dcr_reference
        x_syn = encode(synthetic_df.reindex(columns=columns))
        d_ref = nearest(x_ref, x_syn, 1)[:, 0]
        d_hold = nearest(x_hold, x_syn, 1)[:, 0]
        return float(np.mean(d_ref < d_hold) + 0.5 * np.mean(d_ref == d_hold))

    real_gaps = {}

    def _real_share_gap(n_rows: int) -> float:
        """Protected share gap of ``n_rows`` real tuning rows (resampled if needed)."""
        if n_rows not in real_gaps:
            sample = tuning_df.sample(n=n_rows, replace=n_rows > len(tuning_df), random_state=seed)
            real_gaps[n_rows] = protected_scores(sample, search_train_df, target_column, groupers)[
                "protected_share_gap"
            ]
        return real_gaps[n_rows]

    constraints = hpo_cfg.constraints
    objective_name = (
        hpo_cfg.objective_candidates[0] if hpo_cfg.objective == "auto" else hpo_cfg.objective
    )

    def eval_fn(synthetic_df: pd.DataFrame) -> TrialScore:
        violations = screen_violations(
            synthetic_df,
            search_train_df,
            tuning_df,
            target_column,
            categorical_columns,
            constraints,
        )
        if not target_is_categorical:
            violations["missing_classes"] = -1.0
        elif match_prior and constraints.max_class_share_gap is not None:
            violations["class_share_gap"] = (
                class_share_gap(synthetic_df[target_column], prior)
                - constraints.max_class_share_gap
            )
        attrs = {"screens": violations}
        if groupers:
            protected = protected_scores(synthetic_df, search_train_df, target_column, groupers)
            attrs["protected_target_gap"] = protected["protected_target_gap"]
            attrs["protected_share_gap"] = protected["protected_share_gap"]
            if constraints.min_protected_coverage is not None:
                violations["protected_coverage"] = (
                    constraints.min_protected_coverage - protected["protected_coverage"]
                )
            if constraints.max_protected_share_gap is not None:
                # Calibrated like the copy screen: fresh real rows of the same
                # size also miss the train shares by sampling noise.
                violations["protected_share_gap"] = (
                    protected["protected_share_gap"]
                    - _real_share_gap(len(synthetic_df))
                    - constraints.max_protected_share_gap
                )
        if dcr_reference is not None:
            closer = _dcr_closer_to_train(synthetic_df)
            attrs["dcr_closer_to_train"] = closer
            violations["dcr_closer_to_train"] = closer - constraints.max_dcr_closer_to_train
        if hpo_cfg.log_fidelity:
            attrs |= fidelity_scores(
                synthetic_df, tuning_df, categorical_columns, nominal_columns, seed
            )
        if target_is_categorical:
            assert trtr is not None  # fitted above for a categorical target
            scores = _tstr(synthetic_df)
            attrs |= {
                "tstr_macro_f1": scores.macro_f1,
                "tstr_macro_auprc": scores.macro_auprc,
                "tstr_macro_auroc": scores.macro_auroc,
                "tstr_balanced_accuracy": scores.balanced_accuracy,
                "tstr_kappa": scores.kappa,
                "tstr_per_class_f1": scores.per_class_f1,
                "tstr_seed_sd": scores.seed_sd,
                "tstr_boot_se": scores.boot_se,
                "tstr_group_auprc": scores.group_auprc,
                "tstr_worst_group_auprc": _worst_group(scores),
                "trtr_macro_f1": trtr.macro_f1,
                "trtr_macro_auprc": trtr.macro_auprc,
                "trtr_macro_auroc": trtr.macro_auroc,
                "trtr_balanced_accuracy": trtr.balanced_accuracy,
                "trtr_kappa": trtr.kappa,
                "trtr_worst_group_auprc": _worst_group(trtr),
            }
        if tstr:
            value = float(attrs[objective_name])
        else:
            assert composite_fn is not None  # built above when the objective is not TSTR
            value = composite_fn(synthetic_df)
        if not np.isfinite(value):
            raise ValueError(f"HPO objective {objective_name} is not finite: {value}")
        return TrialScore(value, [violations[name] for name in SCREENS], attrs)

    # The real-data ceiling and chance level per objective, read by the
    # generation stage for its log (see HPOEvalFn).
    eval_fn.headroom = headroom  # ty: ignore[unresolved-attribute]
    return cast(HPOEvalFn, eval_fn)


def score_candidate(
    trial: optuna.Trial, eval_fn: Callable[[pd.DataFrame], TrialScore], synthetic_df
) -> float:
    """Score one candidate, store its screens and logged scores on ``trial``."""
    score = eval_fn(synthetic_df)
    trial.set_user_attr("constraints", list(score.constraints))
    for key, value in score.attrs.items():
        trial.set_user_attr(key, value)
    failed = [name for name, v in zip(SCREENS, score.constraints, strict=True) if v > 0]
    if failed:
        logger.info("trial %d is infeasible: failed screens %s", trial.number, failed)
    objective = trial.user_attrs.get(STUDY_OBJECTIVE_ATTR)
    if objective is None and hasattr(trial, "study"):
        objective = trial.study.user_attrs.get(STUDY_OBJECTIVE_ATTR)
    value = score.attrs.get(objective, score.value) if objective else score.value
    if not np.isfinite(value):
        raise ValueError(f"HPO objective {objective} is not finite: {value}")
    return value


def _trial_constraints(trial: optuna.trial.FrozenTrial) -> list:
    # Trials stopped before scoring have no screens; they are never "best".
    # Trials stored before a screen was added get 0 (feasible) for it.
    constraints = list(trial.user_attrs.get("constraints", []))
    return constraints + [0.0] * (len(SCREENS) - len(constraints))


def default_storage_url(output_dir: str | Path) -> str:
    db_path = Path(output_dir) / "optuna_studies.db"
    ensure_dir(db_path.parent)
    return f"sqlite:///{db_path}"


def default_best_params_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "hpo_best_params.json"


def untuned_fallbacks(best_params_path: str | Path) -> dict[str, str]:
    """``{"<model>_hpo": "<model>"}`` for searches where no trial passed the screens.

    Those tuned models were fitted with the untuned settings (an empty entry
    in the best-params cache), so with the same seeds they repeat the
    untuned model and should be read as such, not as a tuning result.
    """
    data = load_json(Path(best_params_path), default={})
    return {f"{name}_hpo": name for name, params in data.get("synthcity", {}).items() if not params}


def cleanup_hpo_generator_checkpoints(
    workspace: str | Path,
    study: optuna.Study,
    plugin_name: str,
    best_trial_number: int | None = None,
) -> int:
    """Compact completed synthcity HPO generator caches.

    Synthcity stores a fully serialized generator for every benchmark trial.
    Keep the best trial and the highest-numbered trial with a saved generator
    as conservative recovery artifacts once no trial is running; remove the
    other generator caches even when a time-limited study finished below its
    configured target. Metric and synthetic-data caches are intentionally left
    untouched.
    """
    if not plugin_name:
        raise ValueError("plugin_name must not be empty")

    running = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if running:
        logger.info(
            "[%s] HPO still has %d running trial(s); retaining generator checkpoints in %s",
            study.study_name,
            len(running),
            workspace,
        )
        return 0
    if not completed:
        logger.info(
            "[%s] HPO has no completed trials; retaining generator checkpoints in %s",
            study.study_name,
            workspace,
        )
        return 0

    workspace_path = Path(workspace)
    if not workspace_path.is_dir():
        logger.info(
            "[%s] no synthcity checkpoint workspace found at %s",
            study.study_name,
            workspace_path,
        )
        return 0

    trial_numbers = {trial.number for trial in study.trials}
    pattern = re.compile(rf"_trial_(?P<trial>\d+)_{re.escape(plugin_name)}_.*_generator_\d+\.bkp$")
    checkpoints_by_trial: dict[int, list[Path]] = {}
    for path in workspace_path.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if match is None:
            continue
        trial_number = int(match.group("trial"))
        if trial_number not in trial_numbers:
            continue
        checkpoints_by_trial.setdefault(trial_number, []).append(path)

    if not checkpoints_by_trial:
        logger.info(
            "[%s] no generator checkpoints found for plugin=%s in %s",
            study.study_name,
            plugin_name,
            workspace_path,
        )
        return 0

    if best_trial_number is None:
        best_trial_number = study.best_trial.number
    keep_trial_numbers = {best_trial_number, max(checkpoints_by_trial)}
    to_delete = [
        path
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number not in keep_trial_numbers
        for path in paths
    ]

    failures: list[tuple[Path, OSError]] = []
    for path in to_delete:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failures.append((path, exc))

    if failures:
        details = "; ".join(f"{path}: {exc}" for path, exc in failures)
        logger.error(
            "[%s] failed to delete %d HPO generator checkpoint(s): %s",
            study.study_name,
            len(failures),
            details,
        )
        raise RuntimeError(
            f"Failed to delete {len(failures)} HPO generator checkpoint(s)"
        ) from failures[0][1]

    kept_count = sum(
        len(paths)
        for trial_number, paths in checkpoints_by_trial.items()
        if trial_number in keep_trial_numbers
    )
    logger.info(
        "[%s] HPO checkpoint cleanup complete for plugin=%s: kept %d checkpoint(s) "
        "for trial(s) %s; deleted %d from %s",
        study.study_name,
        plugin_name,
        kept_count,
        sorted(keep_trial_numbers),
        len(to_delete),
        workspace_path,
    )
    return len(to_delete)


def model_budget(hpo_cfg: HPOConfig, model: str) -> tuple[int, int | None]:
    """``(n_trials, timeout_seconds)`` for ``model``'s study, per-model values first."""
    return (
        hpo_cfg.n_trials_per_model.get(model, hpo_cfg.n_trials),
        hpo_cfg.timeout_seconds_per_model.get(model, hpo_cfg.timeout_seconds),
    )


def epoch_range(
    hpo_cfg: HPOConfig, model: str, default: tuple | None
) -> tuple[int, int, int] | None:
    """Searched training length ``(low, high, step)``: the config's, else ``default``."""
    bounds = hpo_cfg.epoch_ranges.get(model, default)
    return tuple(bounds) if bounds is not None else None


def make_pruner(hpo_cfg: HPOConfig) -> optuna.pruners.BasePruner:
    if hpo_cfg.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=hpo_cfg.pruner_startup_trials,
            n_warmup_steps=hpo_cfg.pruner_warmup_steps,
        )
    return optuna.pruners.NopPruner()


def create_storage(hpo_cfg: HPOConfig, output_dir: str | Path) -> optuna.storages.BaseStorage:
    """The study storage; SQLite waits up to a minute for a lock held by another process."""
    url = hpo_cfg.storage or default_storage_url(output_dir)
    if url.startswith("sqlite"):
        # Models tuned in parallel processes write to one file.
        return optuna.storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 60}})
    return optuna.storages.get_storage(url)


def create_study(
    study_name: str,
    hpo_cfg: HPOConfig,
    output_dir: str | Path,
    seed: int,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> optuna.Study:
    storage = create_storage(hpo_cfg, output_dir)
    if sampler is None:
        with warnings.catch_warnings():
            # constraints_func, multivariate and group are marked experimental
            # but stable since Optuna 3.0.
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                constraints_func=_trial_constraints,
                n_startup_trials=hpo_cfg.n_startup_trials,
                multivariate=hpo_cfg.multivariate,
                group=hpo_cfg.multivariate,
            )
    return optuna.create_study(
        study_name=study_name,
        direction=objective_direction(hpo_cfg.objective),
        # Constrained TPE (Watanabe & Hutter, 2023): infeasible trials keep
        # their true value and teach the sampler where screens fail.
        sampler=sampler,
        # Pruning stops weak trials during training, from the intermediate
        # scores an objective reports (see synthcity_backend.PruningPatienceMetric).
        pruner=make_pruner(hpo_cfg),
        storage=storage,
        load_if_exists=True,
    )


#: Wall-clock start of the generation stage (seconds since the epoch), set
#: once by the stage and inherited by its worker processes; the stage time
#: limit counts from it.
STAGE_START_ENV_VAR = "SYNTHDATA_GENERATION_START"


def mark_stage_start() -> None:
    """Record the generation stage's start, unless a parent process already did."""
    os.environ.setdefault(STAGE_START_ENV_VAR, f"{time.time():.0f}")


def stage_deadline(hpo_cfg: HPOConfig) -> float | None:
    """When the generation stage must end (seconds since the epoch), or None."""
    if hpo_cfg.stage_time_limit_hours is None:
        return None
    start = float(os.environ.get(STAGE_START_ENV_VAR, time.time()))
    return start + hpo_cfg.stage_time_limit_hours * 3600


def _hours(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


class _TrialProgress:
    """Optuna callback: one log line per finished trial; stops at the search deadline.

    A trial is not started when the longest trial so far would not finish
    before ``search_deadline``.
    """

    def __init__(self, study_name: str, objective: str, n_target: int, search_deadline):
        self.study_name = study_name
        self.objective = objective
        self.n_target = n_target
        self.search_deadline = search_deadline
        self.longest = 0.0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        seconds = trial.duration.total_seconds() if trial.duration else 0.0
        self.longest = max(self.longest, seconds)
        finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        n_done = len([t for t in study.trials if t.state in finished])
        if trial.state == optuna.trial.TrialState.COMPLETE:
            feasible = all(v <= 0 for v in _trial_constraints(trial))
            outcome = f"{self.objective}={trial.value:.4f}" + ("" if feasible else " (infeasible)")
        else:
            outcome = trial.state.name.lower()
        best = _best_feasible(study)
        best_text = (
            f"best {best.value:.4f} (trial {best.number})" if best is not None else "no best yet"
        )
        left = ""
        if self.search_deadline is not None:
            left = f"; search stops in {_hours(max(self.search_deadline - time.time(), 0))}"
        logger.info(
            "[%s] trial %d %s after %s (%d of %d done); %s%s",
            self.study_name,
            trial.number,
            outcome,
            _hours(seconds),
            n_done,
            self.n_target,
            best_text,
            left,
        )
        if self.search_deadline is not None and time.time() + self.longest > self.search_deadline:
            logger.warning(
                "[%s] stopping the search at the stage time limit: the longest trial "
                "(%s) would not finish in time; %d of %d trials done",
                self.study_name,
                _hours(self.longest),
                n_done,
                self.n_target,
            )
            study.stop()


def _value(trial: optuna.trial.FrozenTrial) -> float:
    """Objective value of a complete trial (Optuna types it ``float | None``)."""
    if trial.value is None:
        raise ValueError(f"trial {trial.number} has no objective value")
    return trial.value


def _feasible(trial: optuna.trial.FrozenTrial) -> bool:
    return trial.state == optuna.trial.TrialState.COMPLETE and all(
        v <= 0 for v in _trial_constraints(trial)
    )


def _objective_se(trial: optuna.trial.FrozenTrial, objective: str) -> float | None:
    se = trial.user_attrs.get("tstr_boot_se", {}).get(objective.removeprefix("tstr_"))
    return float(se) if se is not None and np.isfinite(se) else None


class _Patience:
    """Optuna callback: stop once ``patience`` finished trials in a row failed
    to improve the best feasible value by more than ``min_delta``.

    Pruned and failed trials count towards the run. Never stops before
    ``min_trials`` trials. ``min_delta`` None: the median bootstrap standard
    error of the objective over the feasible trials so far (0 without one).
    """

    def __init__(self, study_name: str, patience: int, min_delta: float | None, min_trials: int):
        self.study_name = study_name
        self.patience = patience
        self.min_delta = min_delta
        self.min_trials = min_trials

    def threshold(self, trials: list, objective: str) -> float:
        if self.min_delta is not None:
            return self.min_delta
        ses = [se for t in trials if _feasible(t) and (se := _objective_se(t, objective))]
        return float(np.median(ses)) if ses else 0.0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = [t for t in study.trials if t.state.is_finished()]
        if len(done) < max(self.min_trials, self.patience):
            return
        objective = study.user_attrs.get(STUDY_OBJECTIVE_ATTR, "")
        delta = self.threshold(done, objective)
        sign = 1.0 if study.direction == optuna.study.StudyDirection.MAXIMIZE else -1.0
        best, since = None, 0
        for t in sorted(done, key=lambda t: t.number):
            if _feasible(t) and (best is None or sign * (_value(t) - best) > delta):
                best, since = _value(t), 0
            else:
                since += 1
        # Without a feasible trial yet, keep searching for one.
        if best is not None and since >= self.patience:
            logger.info(
                "[%s] stopping early: %d trials in a row without a gain above %.4f "
                "(best %s; patience %d)",
                self.study_name,
                since,
                delta,
                "none feasible" if best is None else f"{best:.4f}",
                self.patience,
            )
            study.stop()


def choose_objective(
    candidates: list, headroom: dict, trials: list, min_z: float
) -> tuple[str, dict]:
    """``objective: auto``: pick the objective from the random start-up trials.

    The rule is fixed before any score is seen, and uses tuning-split scores
    only:

    1. Drop candidates whose real-data headroom ``(TRTR - chance) / SE`` is
       under ``min_z``: the real data can't beat chance by more than the noise,
       so no synthetic data can be told apart on that score. (If none is left,
       keep the one with the largest headroom.)
    2. Of the rest, take the most reliable: ``1 - noise variance / observed
       variance`` across the feasible start-up trials, where the noise variance
       is the mean squared bootstrap SE. It is the share of the trials' spread
       that is not evaluation noise (an intraclass correlation). Ties, such as
       all zero, go to the first candidate.

    The start-up trials are random draws (Optuna's TPE only models results
    after them), so they do not depend on the objective. Returns the choice and
    the table behind it.
    """
    table: dict[str, dict[str, Any]] = {}
    for name in candidates:
        h = headroom.get(name, {})
        z = h.get("z", float("nan"))
        values = [t.user_attrs[name] for t in trials if _feasible(t) and name in t.user_attrs]
        ses = [se for t in trials if _feasible(t) and (se := _objective_se(t, name)) is not None]
        observed = float(np.var(values, ddof=1)) if len(values) >= 3 else float("nan")
        noise = float(np.mean(np.square(ses))) if ses else float("nan")
        reliability = (
            max(0.0, 1.0 - noise / observed)
            if np.isfinite(observed) and observed > 0 and np.isfinite(noise)
            else 0.0
        )
        table[name] = {
            "headroom_z": z,
            "passes_headroom": bool(z >= min_z),
            "n_trials": len(values),
            "trial_sd": float(np.sqrt(observed)) if np.isfinite(observed) else None,
            "noise_se": float(np.sqrt(noise)) if np.isfinite(noise) else None,
            "reliability": reliability,
        }
    eligible = [n for n in candidates if table[n]["passes_headroom"]]
    if not eligible:
        best_z = max(candidates, key=lambda n: np.nan_to_num(table[n]["headroom_z"], nan=-np.inf))
        logger.warning(
            "no HPO objective candidate has headroom of %.1f bootstrap SEs on this data; "
            "using %s, the one with the most",
            min_z,
            best_z,
        )
        return best_z, table
    top = max(table[n]["reliability"] for n in eligible)
    return next(n for n in eligible if table[n]["reliability"] == top), table


def _copy_trials(source: optuna.Study, target: optuna.Study, objective: str) -> None:
    """Add ``source``'s finished trials to ``target`` valued by ``objective``."""
    for t in source.trials:
        if not t.state.is_finished():
            continue
        value = None
        if t.state == optuna.trial.TrialState.COMPLETE:
            value = t.user_attrs.get(objective, t.value)
        target.add_trial(
            optuna.trial.create_trial(
                state=t.state,
                value=value,
                params=t.params,
                distributions=t.distributions,
                user_attrs=t.user_attrs,
                intermediate_values=t.intermediate_values,
            )
        )


def warm_start_params(hpo_cfg: HPOConfig, study_name: str) -> list[dict]:
    """Params of the top feasible trials of ``study_name`` in ``hpo.warm_start.storage``."""
    cfg = hpo_cfg.warm_start or {}
    storage = cfg.get("storage")
    if not storage:
        return []
    url = storage if "://" in str(storage) else f"sqlite:///{Path(storage).resolve()}"
    try:
        source = optuna.load_study(study_name=study_name, storage=url)
    except KeyError:
        logger.warning("[%s] warm start: no study of that name in %s", study_name, storage)
        return []
    metric = cfg.get("metric", "tstr_macro_auprc")
    ranked = sorted(
        (t for t in source.trials if _feasible(t) and metric in t.user_attrs),
        key=lambda t: t.user_attrs[metric],
        reverse=True,
    )
    return [dict(t.params) for t in ranked[: int(cfg.get("top_k", 3))]]


def _refit_top_k(
    study: optuna.Study,
    objective_fn: Callable,
    hpo_cfg: HPOConfig,
    seed: int,
    objective: str,
) -> optuna.trial.FrozenTrial | None:
    """Refit the best feasible trials with new seeds; return the best by mean.

    Each of the ``refit_top_k`` best feasible trials is fitted again with
    ``refit_seeds`` new seeds (through a ``FixedTrial`` of its params) and
    scored the same way. Results are kept in the study's ``refit`` user
    attribute, so a resumed run reuses them. A refit that fails counts as a
    failed candidate.
    """
    feasible = [t for t in study.trials if _feasible(t)]
    if not feasible:
        return None
    if hpo_cfg.refit_seeds == 0 or hpo_cfg.refit_top_k <= 1:
        return _best_feasible(study)
    sign = 1.0 if study.direction == optuna.study.StudyDirection.MAXIMIZE else -1.0
    top = sorted(feasible, key=lambda t: sign * _value(t), reverse=True)[: hpo_cfg.refit_top_k]
    done = dict(study.user_attrs.get("refit", {}))
    for t in top:
        key = str(t.number)
        if key in done:
            continue
        values = [_value(t)]
        for j in range(hpo_cfg.refit_seeds):
            fixed = optuna.trial.FixedTrial(dict(t.params), number=t.number)
            fixed.set_user_attr(REFIT_SEED_ATTR, seed + 1000 * (j + 1))
            fixed.set_user_attr(STUDY_OBJECTIVE_ATTR, objective)
            try:
                values.append(float(objective_fn(fixed)))
            except (ValueError, RuntimeError) as error:
                logger.warning(
                    "[%s] refit of trial %d failed: %s", study.study_name, t.number, error
                )
                values.append(float("nan"))
        done[key] = values
        study.set_user_attr("refit", done)
        logger.info(
            "[%s] refit trial %d with %d new seed(s): %s (mean %.4f)",
            study.study_name,
            t.number,
            hpo_cfg.refit_seeds,
            ", ".join(f"{v:.4f}" for v in values),
            float(np.mean(values)),
        )
    scored = [(t, float(np.mean(done[str(t.number)]))) for t in top if str(t.number) in done]
    scored = [(t, m) for t, m in scored if np.isfinite(m)]
    if not scored:
        return _best_feasible(study)
    return max(scored, key=lambda pair: sign * pair[1])[0]


def _with_fail_reason(objective_fn: Callable) -> Callable:
    """Store why a trial failed as its ``fail_reason`` user attribute."""

    def wrapped(trial):
        try:
            return objective_fn(trial)
        except optuna.TrialPruned:
            raise
        except Exception as error:
            trial.set_user_attr("fail_reason", f"{type(error).__name__}: {error}"[:1000])
            raise

    return wrapped


def _best_feasible(study: optuna.Study):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in completed if all(v <= 0 for v in _trial_constraints(t))]
    if not feasible:
        return None
    pick = max if study.direction == optuna.study.StudyDirection.MAXIMIZE else min
    return pick(feasible, key=lambda t: t.value)


def run_study(
    study_name: str,
    objective_fn: Callable[[optuna.Trial], float],
    hpo_cfg: HPOConfig,
    output_dir: str | Path,
    seed: int,
    drop_keys: tuple = (),
    checkpoint_workspace: str | Path | None = None,
    checkpoint_plugin: str | None = None,
    model: str | None = None,
    reserve_seconds: float = 0.0,
    default_params: dict | None = None,
    headroom: dict | None = None,
) -> dict:
    """Run (or resume, via SQLite storage) an Optuna study; return best params.

    The trial count and timeout come from ``model_budget`` for ``model``
    (default: ``study_name`` without its ``hpo_`` prefix); 0 trials skips the
    search and returns ``{}``. Pruned and failed trials count towards the
    trial budget. ``drop_keys`` are removed from the returned best-params dict.

    The first trials are ``default_params`` (with ``hpo.enqueue_defaults``)
    and the ``hpo.warm_start`` configs. With ``objective: auto``, the
    ``n_startup_trials`` random start-up trials run in a ``<study>_probe``
    study, :func:`choose_objective` picks the objective from them and
    ``headroom`` (the eval function's ``headroom``), and they are copied into
    the main study valued by that objective. The search stops early after
    ``hpo.patience`` trials without a gain, and the best trial is chosen
    after refitting the top ``refit_top_k`` with new seeds.

    With ``hpo.stage_time_limit_hours`` set, the search also stops starting
    trials ``reserve_seconds`` before the stage deadline (the time the
    model's final fits need), and a trial is not started when the longest
    one so far would run past that point.

    When both checkpoint arguments are provided, synthcity HPO studies with no
    running trial retain only their best and latest generator caches. Studies
    with a running trial or no completed trial retain every cache so an active
    or unsuccessful run remains recoverable.
    """
    if (checkpoint_workspace is None) != (checkpoint_plugin is None):
        raise ValueError("checkpoint_workspace and checkpoint_plugin must be provided together")

    n_trials, timeout = model_budget(hpo_cfg, model or study_name.removeprefix("hpo_"))
    if n_trials == 0:
        logger.info("[%s] hyperparameter search skipped (0 trials); using defaults", study_name)
        return {}
    objective_fn = _with_fail_reason(objective_fn)
    study = create_study(study_name, hpo_cfg, output_dir, seed)
    started = time.monotonic()

    def n_finished(s: optuna.Study) -> int:
        return len([t for t in s.trials if t.state.is_finished()])

    n_remaining = max(n_trials - n_finished(study), 0)
    deadline = stage_deadline(hpo_cfg)
    search_deadline = None if deadline is None else deadline - reserve_seconds
    if search_deadline is not None and n_remaining > 0:
        left = search_deadline - time.time()
        if left <= 0:
            logger.warning(
                "[%s] no time left for the search under the stage time limit "
                "(%.1f h; %s reserved for the final fits); %d trial(s) done",
                study_name,
                hpo_cfg.stage_time_limit_hours,
                _hours(reserve_seconds),
                n_finished(study),
            )
            n_remaining = 0
        else:
            timeout = int(left) if timeout is None else min(timeout, int(left))

    def time_left():
        return None if timeout is None else max(int(timeout - (time.monotonic() - started)), 0)

    initial = []
    if hpo_cfg.enqueue_defaults and default_params:
        initial.append(dict(default_params))
    initial += warm_start_params(hpo_cfg, study_name)

    def enqueue(s: optuna.Study) -> None:
        for params in initial:
            s.enqueue_trial(params, skip_if_exists=True)
        if initial:
            logger.info(
                "[%s] first trials: %d enqueued config(s) (library defaults: %s; warm start: %d)",
                s.study_name,
                len(initial),
                "yes" if hpo_cfg.enqueue_defaults and default_params else "no",
                len(initial) - bool(hpo_cfg.enqueue_defaults and default_params),
            )

    objective = study.user_attrs.get(STUDY_OBJECTIVE_ATTR)
    if objective is None and study.trials:
        # A study from before the objective was stored optimized macro-F1.
        objective = hpo_cfg.objective if hpo_cfg.objective != "auto" else "tstr_macro_f1"
        logger.warning(
            "[%s] resuming a study that predates objective selection; it keeps %s",
            study_name,
            objective,
        )
    elif objective is None and hpo_cfg.objective == "auto":
        probe = create_study(
            f"{study_name}_probe",
            hpo_cfg,
            output_dir,
            seed,
            sampler=optuna.samplers.RandomSampler(seed=seed),
        )
        if not probe.trials:
            enqueue(probe)
        n_probe = min(hpo_cfg.n_startup_trials, n_trials)
        if n_remaining > 0 and n_finished(probe) < n_probe and time_left() != 0:
            logger.info(
                "[%s] objective: auto; %d random start-up trial(s) first",
                study_name,
                n_probe - n_finished(probe),
            )
            probe.optimize(
                objective_fn,
                n_trials=n_probe - n_finished(probe),
                timeout=time_left(),
                catch=(ValueError, RuntimeError),
                show_progress_bar=False,
                callbacks=[
                    _TrialProgress(
                        probe.study_name, hpo_cfg.objective_candidates[0], n_probe, search_deadline
                    )
                ],
            )
        objective, table = choose_objective(
            hpo_cfg.objective_candidates, headroom or {}, probe.trials, hpo_cfg.min_headroom_z
        )
        logger.info(
            "[%s] objective: auto chose %s; %s",
            study_name,
            objective,
            "; ".join(
                f"{name.removeprefix('tstr_')}: headroom z={row['headroom_z']:.1f}, "
                f"reliability={row['reliability']:.2f}"
                for name, row in table.items()
            ),
        )
        study.set_user_attr("objective_selection", table)
        _copy_trials(probe, study, objective)
    elif objective is None:
        objective = hpo_cfg.objective
        enqueue(study)
    study.set_user_attr(STUDY_OBJECTIVE_ATTR, objective)

    n_remaining = 0 if n_remaining == 0 else max(n_trials - n_finished(study), 0)
    if n_remaining > 0 and time_left() == 0:
        logger.info("[%s] the start-up trials used the study's time budget", study_name)
        n_remaining = 0
    if n_remaining > 0:
        logger.info(
            "[%s] starting hyperparameter optimization of %s: %d trial(s) remaining "
            "(%d already finished, target=%d, timeout=%ss)",
            study_name,
            objective,
            n_remaining,
            n_finished(study),
            n_trials,
            time_left(),
        )
        callbacks = [_TrialProgress(study_name, objective, n_trials, search_deadline)]
        if hpo_cfg.patience is not None:
            callbacks.append(
                _Patience(study_name, hpo_cfg.patience, hpo_cfg.min_delta, hpo_cfg.n_startup_trials)
            )
        study.optimize(
            objective_fn,
            n_trials=n_remaining,
            timeout=time_left(),
            # A crashed fit or a non-finite score marks the trial failed.
            catch=(ValueError, RuntimeError),
            show_progress_bar=False,
            callbacks=callbacks,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("[%s] all trials pruned/failed; falling back to defaults", study_name)
        return {}
    feasible = [t for t in completed if _feasible(t)]
    if search_deadline is not None and time.time() >= search_deadline:
        logger.warning("[%s] no time left to refit the top trials; taking the best", study_name)
        best_trial = _best_feasible(study)
    else:
        best_trial = _refit_top_k(study, objective_fn, hpo_cfg, seed, objective)
    if best_trial is None:
        logger.warning(
            "[%s] no trial passed the HPO screens (%d completed); falling back to defaults",
            study_name,
            len(completed),
        )
        return {}

    best = {k: v for k, v in best_trial.params.items() if k not in drop_keys}
    logger.info(
        "[%s] best %s=%.4f (trial %d; %d feasible of %d completed) params=%s",
        study_name,
        objective,
        best_trial.value,
        best_trial.number,
        len(feasible),
        len(completed),
        best,
    )
    if checkpoint_workspace is not None and checkpoint_plugin is not None:
        cleanup_hpo_generator_checkpoints(
            checkpoint_workspace,
            study,
            checkpoint_plugin,
            best_trial_number=best_trial.number,
        )
    return best


class BestParamsCache:
    """JSON-backed cache of ``{family: {model_name: params}}`` best hyperparameters."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = load_json(self.path, default={})

    def get(self, family: str, model_name: str) -> dict:
        return self._data.get(family, {}).get(model_name, {})

    def has(self, family: str, model_name: str) -> bool:
        return model_name in self._data.get(family, {})

    def set(self, family: str, model_name: str, params: dict) -> None:
        # Models tuned in parallel processes share this file: re-read it under
        # a lock so one model's write never drops another's.
        ensure_dir(self.path.parent)
        with FileLock(f"{self.path}.lock"):
            self._data = load_json(self.path, default={})
            self._data.setdefault(family, {})[model_name] = params
            save_json(self.path, self._data)
