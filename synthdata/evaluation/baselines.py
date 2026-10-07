"""Fixed reference rows scored alongside the generators.

Min-max scaling in :mod:`synthdata.evaluation.combine` places every model
relative to the others in the same run, so a score of 1 only means "best of
whatever was evaluated". Two baselines with known behavior go into the same
pool so the scale has fixed ends in every run:

* ``baseline_train_copy`` -- real training rows, sampled without replacement.
  Its utility and fidelity are what real data achieves (train-on-real,
  test-on-real), and its privacy is the worst possible: every row is a real
  record. A generator near it on privacy is memorizing.
* ``baseline_marginals`` -- synthcity's ``marginal_distributions`` plugin,
  which samples each column independently from its training marginal. It
  keeps the marginals and destroys all joint structure, including the
  feature/target relationship, so it is the floor a generator must beat on
  utility.

The test split is deliberately not offered as a baseline: it is the reference
the held-out metrics (TSTR, DomiasMIA, SynthEval's holdout metrics) score
against, so scoring it as "synthetic" data would train and test on the same
rows.

Baselines are evaluated and ranked like any model but are never recommended
(see :func:`is_baseline`).
"""

from pathlib import Path

import pandas as pd

from synthdata.config import EVALUATION_BASELINES
from synthdata.utils import get_logger

logger = get_logger(__name__)

BASELINE_PREFIX = "baseline_"


def baseline_name(kind: str) -> str:
    return f"{BASELINE_PREFIX}{kind}"


def is_baseline(model_name: str) -> bool:
    return str(model_name).startswith(BASELINE_PREFIX)


def build_baselines(
    kinds: list,
    train_df: pd.DataFrame,
    target_column: str,
    sensitive_columns: list,
    n_samples: int,
    seed: int,
    workspace: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ``{baseline_name: frame}`` for each requested baseline kind."""
    baselines: dict[str, pd.DataFrame] = {}
    for kind in kinds:
        if kind not in EVALUATION_BASELINES:
            raise ValueError(f"Unknown baseline {kind!r}; choose from {EVALUATION_BASELINES}")
        if kind == "train_copy":
            n = min(n_samples, len(train_df))
            frame = train_df.sample(n=n, replace=False, random_state=seed)
        else:
            from synthdata.generation import synthcity_backend as sc

            loader = sc.make_loader(train_df, target_column, sensitive_columns, random_state=seed)
            frame = sc.fit_generate(
                "marginal_distributions",
                {},
                loader,
                n_samples,
                seed,
                workspace=workspace,
                device="cpu",
            )
        baselines[baseline_name(kind)] = frame.reset_index(drop=True)[list(train_df.columns)]
    if baselines:
        logger.info("Scoring baseline reference rows: %s", sorted(baselines))
    return baselines
