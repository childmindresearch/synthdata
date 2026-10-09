"""Give each synthetic dataset the real train class shares without duplicating rows.

With ``generation.match_class_prior``, every model is asked for a fixed number
of rows per class (its quota): the train class shares times ``n_samples``,
rounded so the quotas add up to ``n_samples``. This is conditional sampling as
SDV does it (``sample_from_conditions``):

* A model that conditions on the label (synthcity's ddpm on a categorical
  target) is given the quota labels and generates those rows directly.
* Any other model is rejection-sampled: draw a batch, keep each class's rows up
  to its quota, and draw more batches for the classes still short, at most
  ``MAX_ROUNDS`` batches in all.

A class still short after the last batch is left short, and the shortfall is
recorded. Rows are never repeated to fill a quota, so the output can have fewer
than ``n_samples`` rows; the HPO ``class_share_gap`` screen and the
``diagnostics/class_sampling.csv`` record make that visible instead of hiding it.

Without conditioning, a generator that makes a class rarely needs many batches
for that class; a missing class is never filled.
"""

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Batches drawn per dataset at most, the first included.
MAX_ROUNDS = 10
#: A later batch asks for at most this many times the total quota.
MAX_BATCH_FACTOR = 10
#: A later batch asks for this many times the rows expected to fill the
#: scarcest short class, so one top-up batch usually suffices. Backends that
#: refit per batch (TabPFN, TabPFGen) pay mostly per batch, not per row.
TOP_UP_MARGIN = 1.5
#: Batch ``r`` is drawn with seed ``seed + ROUND_SEED_STRIDE * r``, far from the
#: replicate seeds ``seed + replicate``.
ROUND_SEED_STRIDE = 10_000

#: ``sample_fn(count, seed, labels)`` returns a synthetic DataFrame. ``labels``
#: is None, or the target value for each requested row when the model
#: conditions on the label.
SampleFn = Callable[[int, int, np.ndarray | None], pd.DataFrame]


def class_quotas(prior: pd.Series, n: int) -> pd.Series:
    """Rows per class: ``prior * n`` with largest-remainder rounding, summing to ``n``."""
    prior = prior / prior.sum()
    exact = prior * n
    sizes = np.floor(exact).astype(int)
    shortfall = n - int(sizes.sum())
    sizes[(exact - sizes).sort_values(ascending=False).index[:shortfall]] += 1
    return sizes


@dataclasses.dataclass
class ClassSamplingReport:
    """How one synthetic dataset met its class quotas."""

    #: "conditional" (labels given to the model) or "rejection".
    method: str
    #: Rows wanted per class.
    quota: dict
    #: Rows kept per class (equal to quota unless the class fell short).
    filled: dict
    #: Class shares in the model's first unconditioned batch (empty for
    #: conditional sampling, whose batch has the requested labels).
    raw_shares: dict
    #: Batches drawn.
    rounds: int

    @property
    def shortfall(self) -> dict:
        return {c: self.quota[c] - self.filled.get(c, 0) for c in self.quota}

    def to_frame(self, model: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": model,
                "class": [str(c) for c in self.quota],
                "method": self.method,
                "quota": list(self.quota.values()),
                "filled": [self.filled.get(c, 0) for c in self.quota],
                "raw_share": [self.raw_shares.get(c, np.nan) for c in self.quota],
                "rounds": self.rounds,
            }
        )


def _feature_hashes(df: pd.DataFrame, target_column: str) -> pd.Series:
    return pd.util.hash_pandas_object(df.drop(columns=[target_column]), index=False)


def _labels_for(need: pd.Series) -> np.ndarray:
    return np.repeat(need.index.to_numpy(), need.to_numpy())


def sample_to_quota(
    sample_fn: SampleFn,
    target_column: str,
    prior: pd.Series,
    n: int,
    seed: int,
    conditional: bool = False,
    first_batch: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ClassSamplingReport]:
    """Draw rows from ``sample_fn`` until each class of ``prior`` has its quota.

    ``first_batch`` is a batch already drawn with ``seed`` (unconditioned); it
    is used as batch 0 instead of calling ``sample_fn`` again. Rows of a later
    batch whose features equal a row already drawn are dropped, so a backend
    that ignores the seed cannot fill a quota with repeats. The result is
    shuffled with ``seed``.
    """
    quota = class_quotas(prior, n)
    quota = quota[quota > 0]
    kept: dict = {c: [] for c in quota.index}
    have = pd.Series(0, index=quota.index)
    seen: set = set()
    raw_shares: dict = {}
    rounds = 0
    total_drawn = 0
    observed = pd.Series(0, index=quota.index)

    while rounds < MAX_ROUNDS:
        need = (quota - have).clip(lower=0)
        if not need.any():
            break
        round_seed = seed + ROUND_SEED_STRIDE * rounds
        if rounds == 0 and first_batch is not None:
            batch = first_batch
        elif conditional:
            batch = sample_fn(int(need.sum()), round_seed, _labels_for(need))
        else:
            if rounds == 0:
                count = n
            else:
                # Draw enough for the scarcest short class at its observed
                # rate (one pseudo-row for a class not seen yet), with a
                # margin, capped.
                rate = (observed[need > 0] + 1) / (total_drawn + 1)
                expected = (need[need > 0] / rate).max()
                count = min(math.ceil(TOP_UP_MARGIN * expected), MAX_BATCH_FACTOR * n)
            batch = sample_fn(count, round_seed, None)
        rounds += 1
        if target_column not in batch:
            raise ValueError(f"synthetic batch has no target column {target_column!r}")
        batch = batch.reset_index(drop=True)
        hashes = _feature_hashes(batch, target_column)
        if rounds > 1:
            fresh = ~hashes.isin(seen)
            batch, hashes = batch[fresh.to_numpy()], hashes[fresh]
        seen.update(hashes)
        labels = batch[target_column]
        counts = labels.value_counts()
        total_drawn += len(batch)
        observed = observed.add(counts.reindex(quota.index, fill_value=0), fill_value=0)
        if rounds == 1 and not (conditional and first_batch is None):
            raw_shares = (counts / max(len(batch), 1)).to_dict()
        for cls in need[need > 0].index:
            rows = batch[labels == cls].iloc[: need[cls]]
            if len(rows):
                kept[cls].append(rows)
                have[cls] += len(rows)
        if rounds > 1 and len(batch) == 0 and not conditional:
            logger.warning("a batch repeated earlier rows only; the backend may ignore its seed")
            break

    filled = {c: int(have[c]) for c in quota.index}
    short = {c: int(quota[c] - have[c]) for c in quota.index if have[c] < quota[c]}
    if short:
        logger.warning(
            "class quotas not met after %d batch(es); rows short per class: %s "
            "(left short rather than repeating rows)",
            rounds,
            short,
        )
    parts = [part for cls in quota.index for part in kept[cls]]
    out = pd.concat(parts) if parts else batch.iloc[:0]
    rng = np.random.default_rng(seed)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    report = ClassSamplingReport(
        method="conditional" if conditional else "rejection",
        quota={c: int(quota[c]) for c in quota.index},
        filled=filled,
        raw_shares=raw_shares,
        rounds=rounds,
    )
    return out, report
