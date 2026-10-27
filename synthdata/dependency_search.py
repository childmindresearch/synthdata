"""Exact search for dependencies with one (or, opt-in, two) determining columns.

Every ordered pair of columns is scored from its contingency counts, so
nothing is skipped for speed and no error cutoff decides what is computed:

* **FD error** of X -> Y: the g3 error (Kivinen & Mannila 1995), the share
  of rows to remove for "X fixes Y" to hold exactly, on train and holdout,
  counted on the rows where X is answered (``lhs_answered`` is their share).
  Otherwise two mostly blank columns "fix" each other through their shared
  blanks; what a blank X implies is in the one-condition rules
  (``X=<NA> -> Y=<NA>``) and the missing-blocks report. A blank Y counts as
  a value here (X may decide that Y is skipped).
* **Value view** (``value_*``): the same on rows where both X and Y are
  answered (``both_answered``), so a hierarchy whose child or parent is
  often missing at random still shows: sub_program -> major_program holds on
  the recorded values even if major_program is blank in half the rows.
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

Two determining columns (``max_lhs`` 2)
score every pair of columns (on rows where both are answered) against every
other column; that is about
n_columns³ / 2 checks (157 million at 681 columns, hours), so it is opt-in.
Three or more determining columns are not searched: the count grows by
another factor of the number of columns.
"""

import itertools
import time

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

#: Text of a blank cell (``synthdata.dependencies.BLANK``).
BLANK = "<NA>"

logger = get_logger(__name__)

#: Columns of the all-pairs table.
PAIR_COLUMNS = [
    "lhs",
    "rhs",
    "train_error",
    "null_error",
    "baseline_error",
    "score",
    "holdout_error",
    "lhs_answered",
    "value_error",
    "value_score",
    "value_holdout_error",
    "both_answered",
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


def _null_g3(a: np.ndarray, ka: int, shuffled: list, kb: int) -> float:
    """Mean g3 of ``a`` -> each array in ``shuffled`` (``b`` in random row orders)."""
    if not shuffled:
        return float("nan")
    if ka * kb > 8 * len(a) + 1024:
        return float(np.mean([g3(a, ka, b, kb) for b in shuffled]))
    size = ka * kb
    keys = np.concatenate([a * kb + b + k * size for k, b in enumerate(shuffled)])
    table = np.bincount(keys, minlength=len(shuffled) * size).reshape(len(shuffled), ka, kb)
    return float(1 - table.max(axis=2).sum() / (len(shuffled) * len(a)))


def _orders(mask: np.ndarray, permutations: list) -> list:
    """Random orders of the rows in ``mask``: each global permutation filtered to them."""
    return [perm[mask[perm]] for perm in permutations]


def _blank_code(labels: np.ndarray) -> int:
    """Code of the blank label, or -1 when the column has no blanks."""
    blank = np.flatnonzero(labels == BLANK)
    return int(blank[0]) if len(blank) else -1


def _answered(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Boolean mask of rows where the column is not blank."""
    return codes != _blank_code(labels)


def _fd_scores(a: np.ndarray, ka: int, b: np.ndarray, kb: int, shuffled: list) -> tuple:
    """g3 error, shuffled-``b`` error, most-common-value error and chance-corrected score."""
    if len(a) == 0:
        return (np.nan,) * 4
    error = g3(a, ka, b, kb)
    null = _null_g3(a, ka, shuffled, kb)
    baseline = 1 - np.bincount(b, minlength=kb).max() / len(b)
    score = (null - error) / baseline if baseline > 0 else np.nan
    return error, null, baseline, score


def _views(a, a_ho, ka, b, b_ho, kb, masks, orders_a, permutations) -> tuple:
    """Scores of ``a`` -> ``b`` where ``a`` is answered, then where both are answered.

    ``masks`` holds the answered masks of ``a``, ``a_ho``, ``b`` and ``b_ho``;
    ``orders_a`` the random orders of ``a``'s answered rows.
    """
    mask_a, mask_a_ho, mask_b, mask_b_ho = masks
    answered = (
        *_fd_scores(a[mask_a], ka, b[mask_a], kb, [b[o] for o in orders_a]),
        g3(a_ho[mask_a_ho], ka, b_ho[mask_a_ho], kb),
        mask_a.mean(),
    )
    both = mask_a & mask_b
    if both.sum() == mask_a.sum():  # b has no blanks where a is answered: the same rows
        return answered + (answered[0], answered[3], answered[4], answered[5])
    both_ho = mask_a_ho & mask_b_ho
    shuffled = [b[o] for o in _orders(both, permutations)]
    scores = _fd_scores(a[both], ka, b[both], kb, shuffled)
    return answered + (scores[0], scores[3], g3(a_ho[both_ho], ka, b_ho[both_ho], kb), both.mean())


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
    shares = [np.bincount(tr[:, j], minlength=sizes[j]) / n for j in range(len(columns))]
    observed = [int((s > 0).sum()) for s in shares]
    tests = sum(observed) * (len(columns) - 1)
    log_cut = np.log(alpha / max(tests, 1))
    masks_tr = [_answered(tr[:, j], labels[j]) for j in range(len(columns))]
    masks_ho = [_answered(ho[:, j], labels[j]) for j in range(len(columns))]
    blank_code = [_blank_code(lab) for lab in labels]
    permutation_list = [rng.permutation(n) for _ in range(permutations)]
    pairs, rules = [], []
    last_i, orders_i = -1, []
    for i, j in itertools.permutations(range(len(columns)), 2):
        a, b, ka, kb = tr[:, i], tr[:, j], sizes[i], sizes[j]
        if i != last_i:
            rows_i = np.flatnonzero(masks_tr[i])
            orders_i = _orders(masks_tr[i], permutation_list)
            last_i = i
        pairs.append(
            (
                columns[i],
                columns[j],
                *_views(
                    a,
                    ho[:, i],
                    ka,
                    b,
                    ho[:, j],
                    kb,
                    (masks_tr[i], masks_ho[i], masks_tr[j], masks_ho[j]),
                    orders_i,
                    permutation_list,
                ),
                observed[i],
                observed[j],
            )
        )
        count_a, best, best_count = best_counts(a, ka, b, kb)
        # One-condition rules: values whose best match is too frequent to be chance.
        values = np.flatnonzero((count_a > 0) & (2 * best_count > count_a))
        if not len(values):
            continue
        # Chance level for an answered X value: Y's shares where X is answered, so
        # columns blank together do not make every answered value look like a rule.
        answered_share = np.bincount(b[rows_i], minlength=kb) / max(len(rows_i), 1)
        blank_value = values == blank_code[i]
        share = np.where(blank_value, shares[j][best[values]], answered_share[best[values]])
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
    """Every rule {A, B} -> C within ``max_error`` in either view, in ``PAIR_COLUMNS`` form.

    Each pair of columns is scored against every other column, on rows where
    A and B are answered (and, for the value view, C too). C is skipped when
    A or B alone already fixes it exactly (the pair adds nothing), and the
    result keeps only rules within ``max_error``, because all scores would be
    ~n³/2 rows; the single-column table has every pair.
    """
    start = time.time()
    columns = list(dict.fromkeys([*pairs["lhs"], *pairs["rhs"]]))
    tr, ho, labels = encode(train[columns], holdout[columns])
    sizes = [len(lab) for lab in labels]
    n = len(tr)
    index = {c: k for k, c in enumerate(columns)}
    exact = pairs[(pairs["train_error"] == 0) | (pairs["value_error"] == 0)]
    fixed = {k: set() for k in range(len(columns))}
    for lhs, rhs in zip(exact["lhs"], exact["rhs"], strict=True):
        fixed[index[lhs]].add(index[rhs])
    masks_tr = [_answered(tr[:, k], labels[k]) for k in range(len(columns))]
    masks_ho = [_answered(ho[:, k], labels[k]) for k in range(len(columns))]
    rng = np.random.default_rng(seed)
    permutation_list = [rng.permutation(n) for _ in range(permutations)]
    rows = []
    total = len(columns) * (len(columns) - 1) // 2
    for done, (i, j) in enumerate(itertools.combinations(range(len(columns)), 2), start=1):
        kb = sizes[j]
        joint, inverse = np.unique(
            np.r_[tr[:, i] * kb + tr[:, j], ho[:, i] * kb + ho[:, j]], return_inverse=True
        )
        lhs_tr, lhs_ho, k_lhs = inverse[:n], inverse[n:], len(joint)
        mask, mask_ho = masks_tr[i] & masks_tr[j], masks_ho[i] & masks_ho[j]
        if not mask.any():
            continue
        orders = _orders(mask, permutation_list)
        for c in range(len(columns)):
            if c in (i, j) or c in fixed[i] or c in fixed[j]:
                continue
            target, kc = tr[:, c], sizes[c]
            both = mask & masks_tr[c]
            within = g3(lhs_tr[mask], k_lhs, target[mask], kc) <= max_error or (
                both.any() and g3(lhs_tr[both], k_lhs, target[both], kc) <= max_error
            )
            if not within:
                continue
            scores = _views(
                lhs_tr,
                lhs_ho,
                k_lhs,
                target,
                ho[:, c],
                kc,
                (mask, mask_ho, masks_tr[c], masks_ho[c]),
                orders,
                permutation_list,
            )
            rows.append(
                (
                    f"{columns[i]}; {columns[j]}",
                    columns[c],
                    *scores,
                    int(len(np.unique(lhs_tr[mask]))),
                    int(len(np.unique(target))),
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
