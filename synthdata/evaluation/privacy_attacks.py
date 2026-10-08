"""Custom privacy metrics: Anonymeter attacks and holdout-referenced distance
heuristics, both counting a patient (not an encounter) as the individual by
default (``evaluation.privacy_attacks.unit``).

Anonymeter (Giomi et al., PoPETs 2023) runs singling-out, linkability and
attribute-inference attacks against the training rows and reports each as
attack success *above* a baseline attack, with the test split as the control
set the generator never saw. Quasi-identifiers are the attacker's auxiliary
knowledge and sensitive columns are the secrets it tries to infer.

The distance metrics (DCR, NNDR) carry no privacy guarantee: Ganev & De
Cristofaro (2023) build reconstruction attacks against data that passes
them. They are reported only relative to unseen real rows (Platzer &
Reutterer, 2021): synthetic rows should be no closer to the training rows
than the test rows are, and a nearest-synthetic-row membership check should
not tell training patients from test patients apart.
"""

import dataclasses
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from synthdata.data import Dataset
from synthdata.utils import get_logger

logger = get_logger(__name__)

#: Name of each evaluator in ``evaluation.custom`` selection.
ANONYMETER_NAME = "anonymeter"
HOLDOUT_DISTANCE_NAME = "holdout_distance"

#: Anonymeter risks: attack success above the baseline attack, in [0, 1];
#: lower is better. ``anonymeter_inference_risk`` is the worst secret.
ANONYMETER_METRICS = (
    "anonymeter_singling_out_risk",
    "anonymeter_linkability_risk",
    "anonymeter_inference_risk",
)

#: Holdout-referenced distance heuristics -> True when lower is better.
#: share and AUC are 0.5 when synthetic rows are no closer to the training
#: rows than to unseen ones; the ratios are 1 when synthetic rows are as far
#: from the training rows as the test rows are.
HOLDOUT_DISTANCE_METRICS = {
    "dcr_closer_to_train_share": True,
    "distance_mia_auc": True,
    "dcr_holdout_ratio": False,
    "nndr_holdout_ratio": False,
}

#: Quantile of the DCR and NNDR distributions the ratios compare: privacy
#: risk sits in the closest rows, not the typical ones.
DISTANCE_QUANTILE = 0.05


def _one_row_per_patient(df: pd.DataFrame, patient_ids: pd.Series, seed: int) -> pd.DataFrame:
    """One random encounter per patient, so attack targets count patients."""
    ids = patient_ids.loc[df.index]
    order = np.random.default_rng(seed).permutation(len(df))
    keep = ~ids.iloc[order].duplicated()
    return df.iloc[order[keep.to_numpy()]].sort_index()


def _patient_ids(dataset: Dataset, unit: str, index: pd.Index) -> pd.Series:
    """Patient id per row; every row is its own patient for ``unit="row"`` or no id column."""
    if unit == "patient" and dataset.patient_ids is not None:
        return dataset.patient_ids.loc[index]
    return pd.Series(np.arange(len(index)), index=index)


# ---------------------------------------------------------------------------
# Holdout-referenced distances
# ---------------------------------------------------------------------------


def _encoder(reference: pd.DataFrame, nominal: list):
    """Min-max scale numeric columns and one-hot nominal ones, fitted on ``reference``."""
    numeric = [c for c in reference.columns if c not in nominal]
    lo = reference[numeric].astype(float).min()
    span = (reference[numeric].astype(float).max() - lo).replace(0, 1.0)
    levels = {c: pd.Index(reference[c].dropna().unique()) for c in nominal}

    def encode(frame: pd.DataFrame) -> np.ndarray:
        parts = [((frame[numeric].astype(float) - lo) / span).fillna(0.0).to_numpy()]
        for c in nominal:
            codes = levels[c].get_indexer(frame[c])
            onehot = np.zeros((len(frame), len(levels[c])))
            seen = codes >= 0
            onehot[np.flatnonzero(seen), codes[seen]] = 1.0
            parts.append(onehot)
        return np.hstack(parts)

    return encode


def _nearest(reference: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Euclidean distances from each query row to its ``k`` nearest reference rows."""
    nn = NearestNeighbors(n_neighbors=min(k, len(reference))).fit(reference)
    return nn.kneighbors(query)[0]


def _nndr(distances: np.ndarray) -> np.ndarray:
    """Nearest / second-nearest distance ratio (0 for an exact copy)."""
    d1, d2 = distances[:, 0], distances[:, -1]
    return np.divide(d1, d2, out=np.zeros_like(d1), where=d2 > 0)


def holdout_distance_scores(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    nominal: list,
    train_patients: pd.Series,
    holdout_patients: pd.Series,
    seed: int,
) -> dict:
    """Holdout-referenced DCR/NNDR heuristics for one synthetic dataset.

    ``dcr_closer_to_train_share``: share of synthetic rows nearer to a
    training row than to any test row (ties count half), against a random
    training sample with as many patients as the test split, so 0.5 is the
    no-memorization value (Platzer & Reutterer, 2021).
    ``distance_mia_auc``: AUC of telling those training patients from test
    patients by the distance from each patient's closest encounter to the
    synthetic data (a distance membership attack; 0.5 = no signal).
    ``dcr_holdout_ratio`` / ``nndr_holdout_ratio``: the 5th percentile of
    the synthetic rows' DCR (NNDR) to all training rows over the same
    percentile for the test rows; 1 = as far as unseen real rows.
    """
    columns = list(train.columns)
    encode = _encoder(train, [c for c in nominal if c in columns])
    x_train = encode(train)
    x_holdout = encode(holdout)
    x_syn = encode(synthetic[columns])

    holdout_ids = holdout_patients.to_numpy()
    n_holdout_patients = len(np.unique(holdout_ids))
    train_ids = train_patients.to_numpy()
    unique_train = np.unique(train_ids)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        unique_train, size=min(n_holdout_patients, len(unique_train)), replace=False
    )
    in_sample = np.isin(train_ids, sampled)
    x_ref = x_train[in_sample]

    d_ref = _nearest(x_ref, x_syn, 1)[:, 0]
    d_hold = _nearest(x_holdout, x_syn, 1)[:, 0]
    closer = float(np.mean(d_ref < d_hold) + 0.5 * np.mean(d_ref == d_hold))

    member_d = _nearest(x_syn, x_ref, 1)[:, 0]
    nonmember_d = _nearest(x_syn, x_holdout, 1)[:, 0]
    member = pd.Series(member_d).groupby(train_ids[in_sample]).min()
    nonmember = pd.Series(nonmember_d).groupby(holdout_ids).min()
    labels = np.r_[np.ones(len(member)), np.zeros(len(nonmember))]
    auc = float(roc_auc_score(labels, -np.r_[member.to_numpy(), nonmember.to_numpy()]))

    syn_train = _nearest(x_train, x_syn, 2)
    hold_train = _nearest(x_train, x_holdout, 2)

    def _ratio(syn_values: np.ndarray, hold_values: np.ndarray) -> float:
        reference = np.quantile(hold_values, DISTANCE_QUANTILE)
        value = np.quantile(syn_values, DISTANCE_QUANTILE)
        if reference == 0:
            return 1.0 if value == 0 else float("inf")
        return float(value / reference)

    return {
        "dcr_closer_to_train_share": closer,
        "distance_mia_auc": auc,
        "dcr_holdout_ratio": _ratio(syn_train[:, 0], hold_train[:, 0]),
        "nndr_holdout_ratio": _ratio(_nndr(syn_train), _nndr(hold_train)),
    }


# ---------------------------------------------------------------------------
# Anonymeter
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AttackRisk:
    """One Anonymeter attack: risk above baseline with its confidence interval."""

    attack: str
    secret: str | None
    risk: float
    ci_low: float
    ci_high: float
    attack_rate: float
    baseline_rate: float
    control_rate: float
    #: False when Anonymeter flags the analysis as untrustworthy: the attack
    #: did no better than random guessing, or the control attack always won.
    reliable: bool


def _as_labels(series: pd.Series) -> pd.Series:
    """Categorical codes as strings, with 1.0 and 1 the same label."""

    def _label(value):
        if pd.isna(value):
            return "nan"
        if isinstance(value, float | np.floating) and float(value).is_integer():
            return str(int(value))
        return str(value)

    return series.map(_label)


def _anonymeter_frame(frame: pd.DataFrame, categorical: list) -> pd.DataFrame:
    """Anonymeter treats object columns as categorical, numeric ones as continuous."""
    out = frame.copy()
    for column in categorical:
        out[column] = _as_labels(out[column])
    for column in out.columns.difference(categorical):
        out[column] = out[column].astype(float)
    return out


def _risk(evaluator, attack: str, secret: str | None, confidence_level: float) -> AttackRisk:
    risk = evaluator.risk(confidence_level=confidence_level)
    results = evaluator.results(confidence_level=confidence_level)
    attack_rate = float(results.attack_rate.value)
    baseline_rate = float(results.baseline_rate.value)
    control_rate = float(results.control_rate.value)
    return AttackRisk(
        attack=attack,
        secret=secret,
        risk=float(risk.value),
        ci_low=float(risk.ci[0]),
        ci_high=float(risk.ci[1]),
        attack_rate=attack_rate,
        baseline_rate=baseline_rate,
        control_rate=control_rate,
        reliable=attack_rate > baseline_rate and control_rate < 1,
    )


def anonymeter_risks(
    ori: pd.DataFrame,
    synthetic: pd.DataFrame,
    control: pd.DataFrame,
    categorical: list,
    quasi_identifiers: list,
    secrets: list,
    anonymeter_cfg,
    seed: int,
) -> list[AttackRisk]:
    """Run Anonymeter's attacks on one synthetic dataset.

    Singling out uses every column; linkability links the quasi-identifier
    half of a record to its sensitive half; inference guesses each sensitive
    column from the quasi-identifiers. An attack whose columns are not
    configured is skipped. Singling out takes ``seed``; linkability and
    inference sample their targets with numpy's global generator, so it is
    reseeded before each of them. Anonymeter's random linkability baseline
    draws from an unseeded generator, so only its ``baseline_rate`` varies
    between runs; the risk compares the attack with the control set, not
    with that baseline.
    """
    from anonymeter.evaluators import (
        InferenceEvaluator,
        LinkabilityEvaluator,
        SinglingOutEvaluator,
    )

    columns = list(ori.columns)
    categorical = [c for c in categorical if c in columns]
    ori = _anonymeter_frame(ori, categorical)
    control = _anonymeter_frame(control, categorical)
    synthetic = _anonymeter_frame(synthetic[columns], categorical)
    n_attacks = min(anonymeter_cfg.n_attacks, len(control), len(ori))
    level = anonymeter_cfg.confidence_level
    risks = []

    with warnings.catch_warnings():
        # Anonymeter's sanity warnings are recorded per attack as ``reliable``.
        warnings.filterwarnings("ignore", category=UserWarning, module="anonymeter")
        singling_out = SinglingOutEvaluator(
            ori=ori,
            syn=synthetic,
            control=control,
            n_attacks=n_attacks,
            n_cols=min(anonymeter_cfg.singling_out_n_cols, len(columns)),
            max_attempts=anonymeter_cfg.singling_out_max_attempts,
            seed=seed,
        )
        singling_out.evaluate(mode=anonymeter_cfg.singling_out_mode)
        risks.append(_risk(singling_out, "singling_out", None, level))

        if quasi_identifiers and secrets:
            np.random.seed(seed)
            linkability = LinkabilityEvaluator(
                ori=ori,
                syn=synthetic,
                control=control,
                n_attacks=n_attacks,
                aux_cols=(list(quasi_identifiers), list(secrets)),
                n_neighbors=anonymeter_cfg.linkability_n_neighbors,
            )
            linkability.evaluate(n_jobs=1)
            risks.append(_risk(linkability, "linkability", None, level))

        for secret in secrets if quasi_identifiers else []:
            np.random.seed(seed)
            inference = InferenceEvaluator(
                ori=ori,
                syn=synthetic,
                control=control,
                aux_cols=[c for c in quasi_identifiers if c != secret],
                secret=secret,
                regression=secret not in categorical,
                n_attacks=n_attacks,
            )
            inference.evaluate(n_jobs=1)
            risks.append(_risk(inference, "inference", secret, level))
    return risks


def summarize_anonymeter(risks: list[AttackRisk]) -> dict:
    """The three ranked risks; inference is the worst secret's (NaN when skipped)."""
    by_attack = {r.attack: r.risk for r in risks if r.attack != "inference"}
    inference = [r.risk for r in risks if r.attack == "inference"]
    return {
        "anonymeter_singling_out_risk": by_attack.get("singling_out", np.nan),
        "anonymeter_linkability_risk": by_attack.get("linkability", np.nan),
        "anonymeter_inference_risk": max(inference) if inference else np.nan,
    }


# ---------------------------------------------------------------------------
# Evaluation stage entry point
# ---------------------------------------------------------------------------


def run_privacy_attack_evaluation(
    synthetic_datasets: dict[str, pd.DataFrame],
    dataset: Dataset,
    attacks_cfg,
    run_anonymeter: bool,
    run_distances: bool,
    seed: int,
) -> dict:
    """Score every synthetic dataset with the selected privacy evaluators.

    Returns ``{"scores": {model: {metric: value}}, "attacks": DataFrame}``
    (``attacks`` holds every Anonymeter attack with its interval and its
    attack and baseline success rates), or ``{}`` when neither evaluator is
    selected or there is no test split to act as unseen data. A model whose
    scoring fails is logged and left out (NaN in the combined table).
    """
    if not (run_anonymeter or run_distances):
        return {}
    if dataset.test_imputed_df is None or dataset.test_imputed_df.empty:
        logger.warning("[custom] privacy attacks need a test split as the control set; skipping")
        return {}

    train = dataset.train_imputed_df
    holdout = dataset.test_imputed_df[train.columns]
    unit = attacks_cfg.unit
    train_patients = _patient_ids(dataset, unit, train.index)
    holdout_patients = _patient_ids(dataset, unit, holdout.index)
    categorical = list(dataset.nominal_columns)
    if dataset.target_is_categorical:
        categorical.append(dataset.target_column)
    if run_anonymeter:
        ori = _one_row_per_patient(train, train_patients, seed)
        control = _one_row_per_patient(holdout, holdout_patients, seed)
        # Same size as the control set: whether a predicate singles out one
        # record depends on how many records there are, so a larger target
        # set would bias the singling-out risk down.
        ori = ori.sample(n=min(len(ori), len(control)), random_state=seed).sort_index()
        if not dataset.quasi_identifier_columns or not dataset.sensitive_columns:
            logger.info(
                "[custom] Anonymeter linkability and inference need quasi-identifier and "
                "sensitive columns; running singling out only"
            )

    scores, attack_rows = {}, []
    for name, syn_df in synthetic_datasets.items():
        row = {}
        try:
            if run_distances:
                row.update(
                    holdout_distance_scores(
                        train,
                        holdout,
                        syn_df,
                        categorical,
                        train_patients,
                        holdout_patients,
                        seed,
                    )
                )
            if run_anonymeter:
                risks = anonymeter_risks(
                    ori,
                    syn_df,
                    control,
                    categorical,
                    dataset.quasi_identifier_columns,
                    dataset.sensitive_columns,
                    attacks_cfg.anonymeter,
                    seed,
                )
                row.update(summarize_anonymeter(risks))
                attack_rows += [{"model": name, **dataclasses.asdict(r)} for r in risks]
        except Exception as exc:  # noqa: BLE001 -- one bad dataset must not stop the rest
            logger.warning("[custom] privacy attacks failed for %s: %s", name, exc)
            continue
        scores[name] = row
    return {"scores": scores, "attacks": pd.DataFrame(attack_rows), "unit": unit}
