"""Exact search for dependencies with one (or, opt-in, two) determining columns.

Every ordered pair of columns is scored from its contingency counts, so
nothing is skipped for speed and no error cutoff decides what is computed:

* **FD error** of X -> Y: the g3 error (Kivinen & Mannila 1995), the share
  of rows to remove for "X fixes Y" to hold exactly, on train and holdout.
* **Chance-corrected score**: a near-unique X "fixes" any Y by accident (each
  value has one or two rows). The score compares the error with the error
  for a shuffled Y, scaled by the error of guessing Y's most common value:
  ``(null_error - train_error) / baseline_error``, 1 for a real dependency
  and near 0 for one that holds by chance. The same idea as the reliable
  fraction of information (Mandros, Boley & Vreeken, *KDD* 2017), with g3
  instead of entropy; ``null_error`` is the mean over a few shuffles.
* **One-condition rules** X = v -> Y = w (constant conditional FDs; skip logic
  is "gate = no -> item blank"): for each value v the most common Y value
  (the only one that can hold for more than half of the rows), its support
  and confidence. A rule counts when its confidence is too high to be chance:
  binomial test against Y's overall share of w, Bonferroni-corrected over all
  values of all pairs tested. Rows of a value seen once or twice cannot pass,
  however clean they look.

Blanks are a value of their own. Two determining columns (``max_lhs`` 2)
score every pair of columns against every other column; that is about
n_columns³ / 2 checks (157 million at 681 columns, hours), so it is opt-in.
Three or more determining columns are not searched: the count grows by
another factor of the number of columns.
"""

import itertools
import time

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Columns of the all-pairs table.
PAIR_COLUMNS = [
    "lhs",
    "rhs",
    "train_error",
    "holdout_error",
    "null_error",
    "baseline_error",
    "score",
    "lhs_values",
    "rhs_values",
]
#: Columns of the one-condition rules table.
RULE_COLUMNS = [
    "lhs",
    "lhs_value",
    "rhs",
    "rhs_value",
    "support",
    "confidence",
    "p_value",
    "holdout_support",
    "holdout_confidence",
]


def encode(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Integer codes per column, shared by both splits, and each column's labels."""
    both = pd.concat([train, holdout], ignore_index=True)
    codes = np.empty(both.shape, dtype=np.int64)
    labels = []
    for j, column in enumerate(both.columns):
        codes[:, j], uniques = pd.factorize(both[column])
        labels.append(np.asarray(uniques, dtype=object))
    return codes[: len(train)], codes[len(train) :], labels


def best_counts(a: np.ndarray, ka: int, b: np.ndarray, kb: int) -> tuple:
    """Per value of ``a``: its row count, the most common ``b`` value, and that value's count."""
    count_a = np.bincount(a, minlength=ka)
    if ka * kb <= 8 * len(a) + 1024:
        table = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb)
        best = table.argmax(axis=1)
        return count_a, best, table[np.arange(ka), best]
    keys, counts = np.unique(a * kb + b, return_counts=True)
    order = np.lexsort((counts, keys // kb))
    keys, counts = keys[order], counts[order]
    last = np.r_[np.flatnonzero(np.diff(keys // kb)), len(keys) - 1]
    best, best_count = np.zeros(ka, dtype=np.int64), np.zeros(ka, dtype=np.int64)
    best[keys[last] // kb] = keys[last] % kb
    best_count[keys[last] // kb] = counts[last]
    return count_a, best, best_count


def g3(a: np.ndarray, ka: int, b: np.ndarray, kb: int) -> float:
    """Share of rows to remove so that ``a`` fixes ``b``."""
    if len(a) == 0:
        return float("nan")
    return float(1 - best_counts(a, ka, b, kb)[2].sum() / len(a))


def _null_g3(a: np.ndarray, ka: int, b: np.ndarray, kb: int, shuffles: list) -> float:
    """Mean g3 of ``a`` -> ``b`` over the row shuffles of ``b``, in one count when it fits."""
    if not shuffles:
        return float("nan")
    if ka * kb > 8 * len(a) + 1024:
        return float(np.mean([g3(a, ka, b[s], kb) for s in shuffles]))
    size = ka * kb
    keys = np.concatenate([a * kb + b[s] + k * size for k, s in enumerate(shuffles)])
    table = np.bincount(keys, minlength=len(shuffles) * size).reshape(len(shuffles), ka, kb)
    return float(1 - table.max(axis=2).sum() / (len(shuffles) * len(a)))


def _pair_count(a: np.ndarray, b: np.ndarray, kb: int, va: np.ndarray, vb: np.ndarray):
    """Rows of each (``va``, ``vb``) value pair."""
    keys, counts = np.unique(a * kb + b, return_counts=True)
    query = va * kb + vb
    if not len(keys):
        return np.zeros(len(query), dtype=np.int64)
    at = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return np.where(keys[at] == query, counts[at], 0)


def single_column_search(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    permutations: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All ordered column pairs (see the module docstring), and the one-condition rules that pass.

    Columns with one value in ``train`` are left out: they fix nothing and
    every column fixes them.
    """
    from scipy.stats import binom

    start = time.time()
    columns = [c for c in train.columns if train[c].nunique(dropna=False) > 1]
    tr, ho, labels = encode(train[columns], holdout[columns])
    sizes = [len(lab) for lab in labels]
    n = len(tr)
    rng = np.random.default_rng(seed)
    shuffles = [rng.permutation(n) for _ in range(permutations)]
    shares = [np.bincount(tr[:, j], minlength=sizes[j]) / n for j in range(len(columns))]
    baseline = [1 - s.max() for s in shares]
    observed = [int((s > 0).sum()) for s in shares]
    tests = sum(observed) * (len(columns) - 1)
    log_cut = np.log(alpha / max(tests, 1))
    pairs, rules = [], []
    for i, j in itertools.permutations(range(len(columns)), 2):
        a, b, ka, kb = tr[:, i], tr[:, j], sizes[i], sizes[j]
        count_a, best, best_count = best_counts(a, ka, b, kb)
        train_error = 1 - best_count.sum() / n
        null = _null_g3(a, ka, b, kb, shuffles)
        pairs.append(
            (
                columns[i],
                columns[j],
                train_error,
                g3(ho[:, i], ka, ho[:, j], kb),
                null,
                baseline[j],
                (null - train_error) / baseline[j],
                observed[i],
                observed[j],
            )
        )
        # One-condition rules: values whose best match is too frequent to be chance.
        values = np.flatnonzero((count_a > 0) & (2 * best_count > count_a))
        if not len(values):
            continue
        share = shares[j][best[values]]
        support, hits = count_a[values], best_count[values]
        # P(at least `hits` of `support` rows by chance); exact rules need share^support <= cut.
        possible = support * np.log(share) <= log_cut
        if not possible.any():
            continue
        values, share, support, hits = (x[possible] for x in (values, share, support, hits))
        p_value = binom.sf(hits - 1, support, share)
        keep = p_value <= np.exp(log_cut)
        if not keep.any():
            continue
        values, support, hits, p_value = values[keep], support[keep], hits[keep], p_value[keep]
        targets = best[values]
        holdout_support = np.bincount(ho[:, i], minlength=ka)[values]
        holdout_hits = _pair_count(ho[:, i], ho[:, j], kb, values, targets)
        with np.errstate(invalid="ignore", divide="ignore"):
            holdout_confidence = np.where(
                holdout_support > 0, holdout_hits / holdout_support, np.nan
            )
        for k in range(len(values)):
            rules.append(
                (
                    columns[i],
                    labels[i][values[k]],
                    columns[j],
                    labels[j][targets[k]],
                    int(support[k]),
                    hits[k] / support[k],
                    float(p_value[k]),
                    int(holdout_support[k]),
                    float(holdout_confidence[k]),
                )
            )
    logger.info(
        "Dependency search: %d column pairs and %d one-condition rules (of %d tested) "
        "on %d columns in %.0fs.",
        len(pairs),
        len(rules),
        tests,
        len(columns),
        time.time() - start,
    )
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS), pd.DataFrame(rules, columns=RULE_COLUMNS)


def two_column_search(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    pairs: pd.DataFrame,
    max_error: float,
    permutations: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Every rule {A, B} -> C with train error at most ``max_error``, in ``PAIR_COLUMNS`` form.

    Each pair of columns is scored against every other column. C is skipped
    when A or B alone already fixes it exactly (the pair adds nothing), and
    the result keeps only rules within ``max_error``, because all scores
    would be ~n³/2 rows; the single-column table has every pair.
    """
    start = time.time()
    columns = list(dict.fromkeys([*pairs["lhs"], *pairs["rhs"]]))
    tr, ho, labels = encode(train[columns], holdout[columns])
    sizes = [len(lab) for lab in labels]
    n = len(tr)
    index = {c: k for k, c in enumerate(columns)}
    exact = pairs[pairs["train_error"] == 0]
    fixed = {k: set() for k in range(len(columns))}
    for lhs, rhs in zip(exact["lhs"], exact["rhs"], strict=True):
        fixed[index[lhs]].add(index[rhs])
    baseline = dict(zip(pairs["rhs"], pairs["baseline_error"], strict=False))
    rng = np.random.default_rng(seed)
    rows = []
    total = len(columns) * (len(columns) - 1) // 2
    for done, (i, j) in enumerate(itertools.combinations(range(len(columns)), 2), start=1):
        kb = sizes[j]
        joint, inverse = np.unique(
            np.r_[tr[:, i] * kb + tr[:, j], ho[:, i] * kb + ho[:, j]], return_inverse=True
        )
        lhs_tr, lhs_ho, k_lhs = inverse[:n], inverse[n:], len(joint)
        for c in range(len(columns)):
            if c in (i, j) or c in fixed[i] or c in fixed[j]:
                continue
            train_error = g3(lhs_tr, k_lhs, tr[:, c], sizes[c])
            if train_error > max_error:
                continue
            null = np.mean(
                [
                    g3(lhs_tr, k_lhs, rng.permutation(tr[:, c]), sizes[c])
                    for _ in range(permutations)
                ]
            )
            base = baseline[columns[c]]
            rows.append(
                (
                    f"{columns[i]}; {columns[j]}",
                    columns[c],
                    train_error,
                    g3(lhs_ho, k_lhs, ho[:, c], sizes[c]),
                    null,
                    base,
                    (null - train_error) / base,
                    int(len(np.unique(lhs_tr))),
                    int(len(np.unique(tr[:, c]))),
                )
            )
        if done % 1000 == 0:
            elapsed = time.time() - start
            logger.info(
                "Two-column search: %d/%d column pairs, %d rules, about %.0f min left.",
                done,
                total,
                len(rows),
                elapsed / done * (total - done) / 60,
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)
