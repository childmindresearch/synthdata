"""Unit tests for synthdata.evaluation.privacy_attacks."""

import numpy as np
import pandas as pd
import pytest

from synthdata.config import FrameworkSelectionConfig, PrivacyAttacksConfig
from synthdata.evaluation.combine import _privacy_attack_frames
from synthdata.evaluation.custom_eval import run_privacy_evaluation
from synthdata.evaluation.privacy_attacks import (
    AttackRisk,
    _one_row_per_patient,
    holdout_distance_scores,
    summarize_anonymeter,
)

pytestmark = pytest.mark.unit


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.normal(50, 15, n),
            "sex": rng.integers(0, 2, n),
            "diagnosis": rng.integers(0, 3, n),
            "target": rng.integers(0, 2, n),
        }
    )


def _ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.arange(len(frame)), index=frame.index)


@pytest.fixture
def splits():
    train = _frame(300, 0)
    holdout = _frame(100, 1)
    holdout.index = holdout.index + 1000
    return train, holdout


NOMINAL = ["sex", "diagnosis", "target"]


class TestHoldoutDistances:
    def test_a_copy_of_train_is_flagged_and_fresh_data_is_not(self, splits):
        train, holdout = splits
        copy = train.sample(200, random_state=0).reset_index(drop=True)
        fresh = _frame(200, 2)
        scores = {
            name: holdout_distance_scores(
                train, holdout, syn, NOMINAL, _ids(train), _ids(holdout), seed=0
            )
            for name, syn in {"copy": copy, "fresh": fresh}.items()
        }
        assert scores["copy"]["dcr_holdout_ratio"] == 0.0
        assert scores["copy"]["nndr_holdout_ratio"] == 0.0
        assert scores["copy"]["distance_mia_auc"] > 0.6
        assert scores["copy"]["dcr_closer_to_train_share"] > 0.6
        assert scores["fresh"]["dcr_holdout_ratio"] == pytest.approx(1.0, abs=0.35)
        assert scores["fresh"]["distance_mia_auc"] == pytest.approx(0.5, abs=0.1)
        assert scores["fresh"]["dcr_closer_to_train_share"] == pytest.approx(0.5, abs=0.1)

    def test_patients_count_once_in_the_membership_auc(self, splits):
        train, holdout = splits
        # Three train encounters per patient: the AUC is over 100 patients, not 300 rows.
        patients = pd.Series(np.arange(len(train)) // 3, index=train.index)
        copy = train.iloc[:150]
        scores = holdout_distance_scores(
            train, holdout, copy, NOMINAL, patients, _ids(holdout), seed=0
        )
        assert scores["distance_mia_auc"] > 0.6


def test_one_row_per_patient_keeps_one_encounter_each():
    frame = _frame(9, 0)
    ids = pd.Series([0, 0, 0, 1, 1, 2, 3, 3, 3], index=frame.index)
    kept = _one_row_per_patient(frame, ids, seed=0)
    assert sorted(ids.loc[kept.index]) == [0, 1, 2, 3]


def test_summary_takes_the_worst_secret():
    risks = [
        AttackRisk("singling_out", None, 0.1, 0, 0.2, 0.3, 0.2, 0.2, True),
        AttackRisk("inference", "income", 0.05, 0, 0.1, 0.4, 0.35, 0.35, True),
        AttackRisk("inference", "diagnosis", 0.2, 0.1, 0.3, 0.5, 0.3, 0.3, True),
    ]
    summary = summarize_anonymeter(risks)
    assert summary["anonymeter_singling_out_risk"] == 0.1
    assert summary["anonymeter_inference_risk"] == 0.2
    assert np.isnan(summary["anonymeter_linkability_risk"])


def test_combined_frames_give_no_credit_past_the_holdout_reference():
    scores = {
        "far": {"dcr_holdout_ratio": 3.0, "distance_mia_auc": 0.3, "anonymeter_inference_risk": 0},
        "same": {"dcr_holdout_ratio": 1.0, "distance_mia_auc": 0.5, "anonymeter_inference_risk": 0},
        "close": {
            "dcr_holdout_ratio": 0.2,
            "distance_mia_auc": 0.9,
            "anonymeter_inference_risk": 0.4,
        },
    }
    raw, oriented = _privacy_attack_frames({"scores": scores}, ["far", "same", "close"])
    assert raw.loc["far", ("custom", "privacy", "dcr_holdout_ratio")] == 3.0
    assert oriented.loc["far"].equals(oriented.loc["same"])
    assert (oriented.loc["close"] < oriented.loc["same"]).all()


class TestRunPrivacyEvaluation:
    @pytest.fixture
    def dataset(self, make_dataset):
        df = _frame(240, 0)
        dataset = make_dataset(
            df=df,
            nominal_columns=["sex", "diagnosis"],
            quasi_identifier_columns=["age", "sex"],
            sensitive_columns=["diagnosis", "income"],
        )
        dataset.train_imputed_df, dataset.test_imputed_df = dataset.train_df, dataset.test_df
        return dataset

    def test_deselected_returns_empty(self, dataset):
        selection = FrameworkSelectionConfig(categories=["utility"])
        assert run_privacy_evaluation({}, dataset, selection, PrivacyAttacksConfig(), 0) == {}

    def test_runs_every_attack_and_flags_a_copy(self, dataset):
        pytest.importorskip("anonymeter")
        cfg = PrivacyAttacksConfig()
        cfg.anonymeter.n_attacks = 50
        real = dataset.train_imputed_df
        result = run_privacy_evaluation(
            {"copy": real, "fresh": _frame(len(real), 5)},
            dataset,
            FrameworkSelectionConfig(),
            cfg,
            seed=0,
        )
        assert set(result["scores"]) == {"copy", "fresh"}
        attacks = result["attacks"]
        assert set(attacks["attack"]) == {"singling_out", "linkability", "inference"}
        assert set(attacks.loc[attacks["attack"] == "inference", "secret"]) == {
            "diagnosis",
            "income",
        }
        copy, fresh = result["scores"]["copy"], result["scores"]["fresh"]
        assert copy["anonymeter_linkability_risk"] > fresh["anonymeter_linkability_risk"]
        assert copy["distance_mia_auc"] > fresh["distance_mia_auc"]

    def test_is_reproducible_for_a_seed(self, dataset):
        pytest.importorskip("anonymeter")
        cfg = PrivacyAttacksConfig()
        cfg.anonymeter.n_attacks = 30
        syn = {"fresh": _frame(150, 5)}
        first = run_privacy_evaluation(syn, dataset, FrameworkSelectionConfig(), cfg, seed=3)
        second = run_privacy_evaluation(syn, dataset, FrameworkSelectionConfig(), cfg, seed=3)
        assert first["scores"] == second["scores"]
