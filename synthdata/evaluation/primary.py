"""Primary metric set, baseline-anchored scores, Pareto front and the ranking bootstrap.

The ranking reads a short, pre-registered list of metrics
(``evaluation.primary_metrics``); every other metric is still computed and
reported as a diagnostic. Each primary metric is placed on the scale the two
baselines span instead of a min-max over the models of the run:

* utility and fairness: 0 = ``baseline_marginals`` (every column kept, every
  relationship between columns lost), 1 = ``baseline_train_copy`` (real rows);
* privacy: 0 = ``baseline_train_copy`` (every row a real record),
  1 = ``baseline_marginals`` (no row tied to a training record).

A TSTR macro AUPRC of 0.6 so reads "recovers 60% of the real-data signal
the marginals lack", whatever other models were run. Values outside [0, 1]
are kept in the type scores (a model can beat the train copy), and clipped
to [GEOMETRIC_FLOOR, 1] only inside the overall score, the weighted
geometric mean of the type scores. Anchors are the baselines' means over
their seed replicates. A metric whose baselines do not sit in the expected
order (or tie) has no scale and is left out of the scores, with a warning.

Noise (S3): :func:`bootstrap_draws` resamples, in each draw, the holdout
patients (one resample shared by every model, so comparisons are paired)
and each model's seed replicates (with replacement), recomputes the metrics
scored on the holdout from their per-row outputs (:class:`HoldoutRows`),
re-anchors, and recomputes every score. The other primary metrics vary by
seed replicate only. Reasons in docs/verification.md ("Ranking noise").
"""

import dataclasses

import numpy as np
import pandas as pd

from synthdata.utils import get_logger, split_replicate_name

logger = get_logger(__name__)

COPY = "baseline_train_copy"
MARGINALS = "baseline_marginals"
TYPES = ("utility", "privacy", "fairness")
#: Baseline scored 1 on each type (the other one scores 0).
BEST_ANCHOR = {"utility": COPY, "privacy": MARGINALS, "fairness": COPY}
#: Type scores are floored here inside the geometric mean (as the composite does).
GEOMETRIC_FLOOR = 0.01

#: Raw values past the no-leak reference earn no credit, as in the composite:
#: an attack below its baseline, or synthetic rows further from training rows
#: than unseen real rows, are not more private, only less useful.
_REFERENCE_CLIP = {
    "distance_mia_auc": (0.5, None),
    "domias_mia_auc": (0.5, None),
    "dcr_closer_to_train_share": (0.5, None),
    "dcr_holdout_ratio": (None, 1.0),
    "nndr_holdout_ratio": (None, 1.0),
    "anonymeter_singling_out_risk": (0.0, None),
    "anonymeter_linkability_risk": (0.0, None),
    "anonymeter_inference_risk": (0.0, None),
}

#: Direction of the default primary metrics (+1 higher is better, -1 lower).
KNOWN_SIGNS = {
    "tstr_macro_auprc": 1.0,
    "tstr_kappa": 1.0,
    "corr_mat_diff": -1.0,
    "distance_mia_auc": -1.0,
    "domias_mia_auc": -1.0,
    "dcr_closer_to_train_share": -1.0,
    "anonymeter_singling_out_risk": -1.0,
    "anonymeter_linkability_risk": -1.0,
    "anonymeter_inference_risk": -1.0,
    "demographic_parity_gap_error": -1.0,
    "equalized_odds_gap_error": -1.0,
    "equal_opportunity_gap_error": -1.0,
}

#: File (in the evaluation folder) with each primary metric's raw and anchored value.
PRIMARY_FILENAME = "primary_metrics.csv"

#: Holdout draws are computed in chunks of this many to bound memory.
_CHUNK = 200


@dataclasses.dataclass
class PrimarySpec:
    """The resolved primary metrics: combined-table columns, type and direction."""

    columns: list
    #: +1 when a higher raw value is better, -1 when lower is, per column.
    signs: dict
    #: Columns scored on |value| (signed gaps where 0 is ideal).
    absolute: set
    #: ``{column: reason}`` for configured metrics left out.
    dropped: dict

    @property
    def types(self) -> dict:
        return {c: c[1] for c in self.columns}


def resolve_primary(names: list, raw: pd.DataFrame, directions: pd.DataFrame) -> PrimarySpec:
    """Map ``evaluation.primary_metrics`` names to combined-table columns.

    A name is a metric (level 3 of the columns) or ``framework/type/metric``.
    A name matching no column is logged and left out (the metric was not
    selected or failed for every model); one matching several is an error.
    """
    columns, signs, absolute, dropped = [], {}, set(), {}
    for name in names:
        parts = tuple(name.split("/"))
        matches = [c for c in raw.columns if (c == parts if len(parts) == 3 else c[2] == name)]
        if len(matches) > 1:
            raise ValueError(
                f"evaluation.primary_metrics: {name!r} matches {matches}; "
                "write it as framework/type/metric"
            )
        if not matches:
            dropped[name] = "not computed in this run"
            continue
        column = matches[0]
        if column[1] not in TYPES:
            dropped[name] = f"type {column[1]!r} is not utility, privacy or fairness"
            continue
        direction = directions.loc[column, "direction"] if column in directions.index else None
        if column[2] in KNOWN_SIGNS:
            # Read from the data, a direction can come out "unknown" when the
            # models tie; the defaults' directions are fixed by definition.
            signs[column] = KNOWN_SIGNS[column[2]]
        elif direction == "closer to 0":
            absolute.add(column)
            signs[column] = -1.0
        elif direction in ("higher", "lower"):
            signs[column] = 1.0 if direction == "higher" else -1.0
        else:
            dropped[name] = f"direction {direction!r} unknown (reported, not ranked?)"
            continue
        columns.append(column)
    for name, reason in dropped.items():
        logger.warning("[ranking] primary metric %s left out: %s", name, reason)
    return PrimarySpec(columns, signs, absolute, dropped)


def anchor_input(raw: pd.DataFrame, spec: PrimarySpec) -> pd.DataFrame:
    """The raw primary values the anchors apply to: |value| for signed gaps,
    privacy values clipped at their no-leak reference."""
    table = raw[spec.columns].astype(float).copy()
    for column in spec.columns:
        if column in spec.absolute:
            table[column] = table[column].abs()
        lower, upper = _REFERENCE_CLIP.get(column[2], (None, None))
        table[column] = table[column].clip(lower=lower, upper=upper)
    return table


def _groups(names: list) -> dict:
    """``{model: positions in names}`` of each model's seed replicates."""
    groups: dict = {}
    for i, name in enumerate(names):
        groups.setdefault(split_replicate_name(str(name))[0], []).append(i)
    return {m: np.asarray(p) for m, p in groups.items()}


def _scores(
    values: np.ndarray,
    spec: PrimarySpec,
    names: list,
    picks: dict,
    weights: dict,
) -> dict:
    """Anchored scores of every replicate and model, for each of ``B`` draws.

    ``values``: ``(B, N, K)`` anchor inputs (draws x replicate names x primary
    columns). ``picks``: ``{model: (B, R) positions into names}``, the
    replicates each draw averages (the baselines' picks give the anchors).
    Returns ``{"anchored": (B, N, K), "<dim>_rep": (B, N), "<dim>": (B, M)}``
    for dim in overall and each type present, models in ``picks`` order.
    """
    n_draws = values.shape[0]
    rows = np.arange(n_draws)[:, None]

    def model_mean(array: np.ndarray, model: str) -> np.ndarray:
        # array (B, N, ...) -> mean over each draw's picked replicates.
        return np.nanmean(array[rows, picks[model]], axis=1)

    signs = np.array([spec.signs[c] for c in spec.columns])
    best = np.stack(
        [model_mean(values, BEST_ANCHOR[c[1]])[:, k] for k, c in enumerate(spec.columns)], 1
    )
    worst = np.stack(
        [
            model_mean(values, MARGINALS if BEST_ANCHOR[c[1]] == COPY else COPY)[:, k]
            for k, c in enumerate(spec.columns)
        ],
        1,
    )
    span = best - worst
    valid = (span * signs) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        anchored = (values - worst[:, None, :]) / np.where(valid, span, np.nan)[:, None, :]
    # A metric a replicate lacks while the scale exists scores 0, the worse
    # baseline (as a missing metric scores worst in the composite).
    anchored = np.where(np.isnan(values) & valid[:, None, :], 0.0, anchored)

    out: dict = {"anchored": anchored}
    log_total = np.zeros(values.shape[:2])
    weight_total = np.zeros(values.shape[:2])
    for type_ in TYPES:
        cols = [k for k, c in enumerate(spec.columns) if c[1] == type_]
        if not cols:
            continue
        with np.errstate(invalid="ignore"), _quiet():
            score = np.nanmean(anchored[:, :, cols], axis=2)
        out[f"{type_}_rep"] = score
        weight = float(weights.get(type_, 1.0))
        if weight > 0:
            ok = ~np.isnan(score)
            floored = np.clip(np.where(ok, score, 1.0), GEOMETRIC_FLOOR, 1.0)
            log_total += np.where(ok, weight * np.log(floored), 0.0)
            weight_total += np.where(ok, weight, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["overall_rep"] = np.where(weight_total > 0, np.exp(log_total / weight_total), np.nan)
    for dim in ("overall", *TYPES):
        if f"{dim}_rep" in out:
            with _quiet():
                out[dim] = np.stack([model_mean(out[f"{dim}_rep"], m) for m in picks], axis=1)
    return out


class _quiet(np.errstate):
    """Silence numpy's "mean of empty slice" warnings for all-NaN slices."""

    def __init__(self):
        super().__init__(invalid="ignore")

    def __enter__(self):
        import warnings

        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        return super().__enter__()

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self._catch.__exit__(*exc)


def point_scores(raw: pd.DataFrame, spec: PrimarySpec, weights: dict) -> dict:
    """Anchored primary scores on the full data.

    Returns ``{"anchored": DataFrame (replicates x primary columns),
    "replicates": DataFrame (replicates x overall/utility/privacy/fairness)}``.
    Metrics whose baselines are missing or out of order are moved to
    ``spec.dropped``.
    """
    names = list(raw.index)
    groups = _groups(names)
    missing = [b for b in (COPY, MARGINALS) if b not in groups]
    if missing:
        raise ValueError(f"primary metrics need both baselines; missing {missing}")
    values = anchor_input(raw, spec).to_numpy()[None]
    picks = {m: p[None, :] for m, p in groups.items()}
    out = _scores(values, spec, names, picks, weights)
    anchored = pd.DataFrame(
        out["anchored"][0], index=names, columns=pd.MultiIndex.from_tuples(spec.columns)
    )
    for column in spec.columns:
        if anchored[column].isna().all():
            spec.dropped[column[2]] = (
                "the baselines are not in the expected order on it (or tie), so it has no scale"
            )
            logger.warning(
                "[ranking] primary metric %s left out: %s", column[2], spec.dropped[column[2]]
            )
    replicates = pd.DataFrame(
        {dim: out[f"{dim}_rep"][0] for dim in ("overall", *TYPES) if f"{dim}_rep" in out},
        index=names,
    )
    return {"anchored": anchored, "replicates": replicates}


def primary_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Long table of the primary metrics: model, replicate, metric, type, raw
    value, the two anchors and the anchored score (``primary_metrics.csv``)."""
    spec = combined.attrs["primary_spec"]
    anchored = combined.attrs["primary_anchored"]
    values = anchor_input(combined[spec.columns], spec)
    rows = []
    for column in spec.columns:
        best = BEST_ANCHOR[column[1]]
        worst = MARGINALS if best == COPY else COPY
        means = (
            values[column].groupby([split_replicate_name(str(m))[0] for m in values.index]).mean()
        )
        for name in combined.index:
            rows.append(
                {
                    "model": split_replicate_name(str(name))[0],
                    "replicate": name,
                    "type": column[1],
                    "framework": column[0],
                    "metric": column[2],
                    "raw": combined.loc[name, column],
                    "anchor_0": means.get(worst, np.nan),
                    "anchor_1": means.get(best, np.nan),
                    "anchored": anchored.loc[name, column],
                }
            )
    return pd.DataFrame(rows)


def pareto_front(utility: pd.Series, privacy: pd.Series) -> pd.Series:
    """True for the models no other model beats on both utility and privacy.

    A model is dominated when another is at least as good on both and better
    on one. Models with a missing score are not on the front.
    """
    u, p = utility.to_numpy(float), privacy.to_numpy(float)
    front = np.zeros(len(u), dtype=bool)
    for i in range(len(u)):
        if np.isnan(u[i]) or np.isnan(p[i]):
            continue
        dominated = (u >= u[i]) & (p >= p[i]) & ((u > u[i]) | (p > p[i]))
        front[i] = not dominated.any()
    return pd.Series(front, index=utility.index)


# ---------------------------------------------------------------------------
# Holdout resampling (S3)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HoldoutRows:
    """Per-row outputs of the metrics scored on the holdout, for resampling.

    ``holdout_keys``: patient of each holdout row, in test-split order
    (rows when there is no patient id). TSTR and fairness arrays follow that
    order. Distance-attack arrays are per attack unit: ``member_keys`` (train
    units) and ``nonmember_rows`` (positions of each holdout unit's first row).
    """

    holdout_keys: np.ndarray
    #: TSTR: class codes of the holdout rows and each replicate's first-seed probabilities.
    y_true: np.ndarray | None = None
    proba: dict = dataclasses.field(default_factory=dict)
    ordinal: bool = False
    #: Fairness: the real-trained classifier's holdout classes, the group masks
    #: (``{(column, group): bool array}``) and each replicate's predicted classes.
    real_pred: np.ndarray | None = None
    masks: dict = dataclasses.field(default_factory=dict)
    syn_pred: dict = dataclasses.field(default_factory=dict)
    min_group_rows: int = 30
    #: Distance membership attack: ``{name: (member_keys, member_d, nonmember_rows, nonmember_d)}``.
    distances: dict = dataclasses.field(default_factory=dict)


#: Metrics :func:`bootstrap_draws` recomputes from :class:`HoldoutRows`.
RESAMPLED_METRICS = (
    "tstr_macro_auprc",
    "tstr_kappa",
    "distance_mia_auc",
    "demographic_parity_gap_error",
)


def _curve(score: np.ndarray, label: np.ndarray, w: np.ndarray) -> tuple:
    """Weighted true and false positive sums at each distinct threshold, high to low."""
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    pos = w[:, order] * label[order]
    neg = w[:, order] * (1 - label[order])
    return np.cumsum(pos, axis=1)[:, ends], np.cumsum(neg, axis=1)[:, ends]


def weighted_average_precision(score, label, w) -> np.ndarray:
    """Average precision (scikit-learn's step definition) per row of weights ``w``."""
    tp, fp = _curve(score, label.astype(float), w)
    total = tp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = tp / total
    gain = np.diff(np.c_[np.zeros(len(tp)), recall], axis=1)
    return np.where(total[:, 0] > 0, np.sum(gain * precision, axis=1), np.nan)


def weighted_auc(score, label, w) -> np.ndarray:
    """ROC AUC per row of weights ``w`` (trapezoids, ties count half)."""
    tp, fp = _curve(score, label.astype(float), w)
    tpr = np.c_[np.zeros(len(tp)), tp] / tp[:, -1:]
    fpr = np.c_[np.zeros(len(fp)), fp] / fp[:, -1:]
    return np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2, axis=1)


def weighted_macro_auprc(y: np.ndarray, proba: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Macro AUPRC over the classes present in each resample."""
    per_class = np.stack(
        [weighted_average_precision(proba[:, i], y == i, w) for i in range(proba.shape[1])], 1
    )
    with _quiet():
        return np.nanmean(per_class, axis=1)


def weighted_kappa(y: np.ndarray, pred: np.ndarray, n_classes: int, ordinal: bool, w) -> np.ndarray:
    """Cohen's kappa per row of ``w`` (quadratic weights when ``ordinal``)."""
    cell = np.zeros((len(y), n_classes * n_classes))
    cell[np.arange(len(y)), y * n_classes + pred] = 1.0
    confusion = (w @ cell).reshape(-1, n_classes, n_classes)
    total = confusion.sum(axis=(1, 2), keepdims=True)
    expected = confusion.sum(axis=2, keepdims=True) * confusion.sum(axis=1, keepdims=True) / total
    i, j = np.indices((n_classes, n_classes))
    cost = (i - j) ** 2 if ordinal else (i != j).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = 1 - (confusion * cost).sum(axis=(1, 2)) / (expected * cost).sum(axis=(1, 2))
    return np.nan_to_num(kappa, nan=0.0)


def weighted_dp_gap_error(syn_pred, real_pred, masks: dict, n_classes: int, w, min_rows: int):
    """Mean |synthetic - real| demographic parity gap per row of ``w``
    (fairness_gaps.gap_errors on a resample; group and rest need ``min_rows``)."""
    errors, counts = np.zeros(len(w)), np.zeros(len(w))
    for mask in masks.values():
        mask = np.asarray(mask, dtype=float)
        n_in, n_out = w @ mask, w @ (1 - mask)
        ok = (n_in >= min_rows) & (n_out >= min_rows)
        for cls in range(n_classes):
            gaps = []
            for pred in (syn_pred, real_pred):
                hit = (pred == cls).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    gaps.append(w @ (hit * mask) / n_in - w @ (hit * (1 - mask)) / n_out)
            errors += np.where(ok, np.abs(gaps[0] - gaps[1]), 0.0)
            counts += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, errors / counts, np.nan)


def _metric_draws(rows: HoldoutRows, metric: str, name: str, w_hold, w_member, member_codes):
    """One replicate's metric on each resample (``w_hold``: (b, holdout rows))."""
    if metric == "tstr_macro_auprc" and name in rows.proba:
        return weighted_macro_auprc(rows.y_true, rows.proba[name], w_hold)
    if metric == "tstr_kappa" and name in rows.proba:
        proba = rows.proba[name]
        pred = proba.argmax(axis=1)
        return weighted_kappa(rows.y_true, pred, proba.shape[1], rows.ordinal, w_hold)
    if (
        metric == "demographic_parity_gap_error"
        and name in rows.syn_pred
        and rows.real_pred is not None
    ):
        n_classes = int(max(rows.real_pred.max(), rows.syn_pred[name].max())) + 1
        return weighted_dp_gap_error(
            rows.syn_pred[name], rows.real_pred, rows.masks, n_classes, w_hold, rows.min_group_rows
        )
    if metric == "distance_mia_auc" and name in rows.distances:
        keys, member_d, nonmember_rows, nonmember_d = rows.distances[name]
        w = np.c_[w_member[:, member_codes[name]], w_hold[:, nonmember_rows]]
        label = np.r_[np.ones(len(member_d)), np.zeros(len(nonmember_d))]
        return weighted_auc(-np.r_[member_d, nonmember_d], label, w)
    return None


def bootstrap_draws(
    raw: pd.DataFrame,
    spec: PrimarySpec,
    weights: dict,
    rows: HoldoutRows | None,
    n_draws: int,
    seed: int,
) -> dict:
    """Bootstrap distribution of every model's scores: ``{dim: DataFrame (draws x models)}``.

    Each draw resamples the holdout patients once for all models (and the
    distance attack's member units), recomputes :data:`RESAMPLED_METRICS`
    from ``rows`` for every replicate, resamples each model's replicates with
    replacement, re-anchors and rescores (:func:`_scores`).
    """
    names = list(raw.index)
    groups = _groups(names)
    base = anchor_input(raw, spec).to_numpy()
    rng = np.random.default_rng(seed)
    resampled = [
        k for k, c in enumerate(spec.columns) if rows is not None and c[2] in RESAMPLED_METRICS
    ]
    member_codes, n_members = {}, 0
    if rows is not None and rows.distances:
        universe = pd.Index(np.unique(np.concatenate([v[0] for v in rows.distances.values()])))
        member_codes = {n: universe.get_indexer(v[0]) for n, v in rows.distances.items()}
        n_members = len(universe)
    hold_codes, patients = (
        pd.factorize(pd.Series(rows.holdout_keys)) if rows is not None else (None, [])
    )
    draws: dict = {}
    for start in range(0, n_draws, _CHUNK):
        b = min(_CHUNK, n_draws - start)
        values = np.repeat(base[None], b, axis=0)
        if resampled and rows is not None:
            counts = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), b)
            w_hold = counts[:, hold_codes].astype(float)
            w_member = (
                rng.multinomial(n_members, np.full(n_members, 1 / n_members), b).astype(float)
                if n_members
                else None
            )
            for k in resampled:
                column = spec.columns[k]
                for i, name in enumerate(names):
                    value = _metric_draws(rows, column[2], name, w_hold, w_member, member_codes)
                    if value is not None:
                        lower, upper = _REFERENCE_CLIP.get(column[2], (None, None))
                        value = np.abs(value) if column in spec.absolute else value
                        values[:, i, k] = np.clip(
                            value,
                            -np.inf if lower is None else lower,
                            np.inf if upper is None else upper,
                        )
        picks = {m: p[rng.integers(0, len(p), (b, len(p)))] for m, p in groups.items()}
        out = _scores(values, spec, names, picks, weights)
        for dim in ("overall", *TYPES):
            if dim in out:
                draws.setdefault(dim, []).append(out[dim])
    return {
        dim: pd.DataFrame(np.concatenate(parts), columns=list(groups))
        for dim, parts in draws.items()
    }


def holdout_rows(
    holdout_keys: np.ndarray,
    tstr_result: dict | None,
    fairness_result: dict | None,
    privacy_result: dict | None,
    ordinal: bool,
) -> HoldoutRows:
    """Collect the per-row outputs the evaluators kept (see :class:`HoldoutRows`)."""
    rows = HoldoutRows(holdout_keys=np.asarray(holdout_keys), ordinal=ordinal)
    for name, scores in ((tstr_result or {}).get("scores") or {}).items():
        if getattr(scores, "holdout_proba", None) is not None:
            rows.y_true = scores.holdout_labels
            rows.proba[name] = scores.holdout_proba
    for name, result in (fairness_result or {}).items():
        if getattr(result, "holdout_pred", None) is not None:
            rows.syn_pred[name] = result.holdout_pred
            rows.real_pred = result.real_pred
            rows.masks = result.masks
    rows.distances = dict((privacy_result or {}).get("distance_rows") or {})
    return rows


def summarize_draws(
    draws: dict, point: pd.DataFrame, eligible: list, confidence: float
) -> pd.DataFrame:
    """Percentile intervals, paired differences against the best eligible model and ties.

    ``point``: models x dims point scores. The best model is the eligible one
    with the highest point overall score; ``diff_vs_best`` is its score minus
    each model's, with a percentile interval from the same draws (paired:
    every draw resamples the same holdout patients for all models).
    ``tied_with_best`` is True when the one-sided ``confidence`` bound of
    that difference reaches 0. ``pareto_share``: share of draws in which the
    model is on the utility-privacy front among eligible models.
    """
    tail = (1 - confidence) / 2
    table = pd.DataFrame(index=point.index)
    for dim, frame in draws.items():
        frame = frame.reindex(columns=point.index)
        table[f"{dim}_ci_low"] = frame.quantile(tail).to_numpy()
        table[f"{dim}_ci_high"] = frame.quantile(1 - tail).to_numpy()
    if "overall" in draws and eligible:
        overall = draws["overall"].reindex(columns=point.index)
        best = point.loc[eligible, "overall"].idxmax()
        diff = overall[best].to_numpy()[:, None] - overall.to_numpy()
        table["diff_vs_best"] = point.loc[best, "overall"] - point["overall"]
        table["diff_ci_low"] = np.nanquantile(diff, tail, axis=0)
        table["diff_ci_high"] = np.nanquantile(diff, 1 - tail, axis=0)
        lower = np.nanquantile(diff, 1 - confidence, axis=0)
        tied = pd.Series(lower <= 0, index=point.index, dtype=object)
        tied[[m for m in point.index if m not in eligible]] = np.nan
        tied[best] = True
        table["tied_with_best"] = tied
    if {"utility", "privacy"} <= set(draws) and eligible:
        u = draws["utility"][eligible].to_numpy()
        p = draws["privacy"][eligible].to_numpy()
        on_front = np.zeros(u.shape)
        for i in range(len(eligible)):
            dominated = (
                (u >= u[:, [i]]) & (p >= p[:, [i]]) & ((u > u[:, [i]]) | (p > p[:, [i]]))
            ).any(axis=1)
            on_front[:, i] = ~dominated
        table["pareto_share"] = pd.Series(on_front.mean(axis=0), index=eligible)
    return table


def mia_rows(member: pd.Series, nonmember: pd.Series, holdout_unit_ids: pd.Series) -> tuple:
    """Per-unit distances of the distance attack, keyed for :class:`HoldoutRows`.

    ``holdout_unit_ids``: unit id of each holdout row, in test-split order.
    """
    first_row = (
        pd.Series(np.arange(len(holdout_unit_ids))).groupby(holdout_unit_ids.to_numpy()).first()
    )
    return (
        member.index.to_numpy(),
        member.to_numpy(float),
        first_row.reindex(nonmember.index).to_numpy(int),
        nonmember.to_numpy(float),
    )
