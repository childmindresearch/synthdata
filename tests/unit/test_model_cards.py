"""Model cards: layout in step with the docs template, filled from the run's tables."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from synthdata.config import Config
from synthdata.evaluation.model_cards import SECTIONS, build_model_card

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).parents[2] / "docs" / "model-card-template.md"


def test_docs_template_has_the_cards_sections_in_order():
    headings = [line[3:] for line in TEMPLATE.read_text().splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)


def _combined(models):
    columns = pd.MultiIndex.from_tuples(
        [("custom", "utility", "tstr_macro_auprc"), ("custom", "privacy", "distance_mia_auc")]
    )
    values = [[0.6, 0.55], [0.5, float("nan")], [0.5, 0.5], [0.8, 0.9]]
    return pd.DataFrame(values[: len(models)], index=models, columns=columns)


def test_card_reads_tuned_settings_warnings_and_missing_metrics():
    models = ["ctgan_hpo__rep1", "ctgan_hpo__rep2", "baseline_marginals", "baseline_train_copy"]
    dataset = SimpleNamespace(
        name="toy", version="1", target_column="y", train_df=pd.DataFrame({"a": range(10)})
    )
    tables = {
        "best_params": {"synthcity": {"ctgan": {"n_iter": 300}}},
        "checks": pd.DataFrame(
            {
                "check": ["dependency_rules", "columns_match"],
                "model": ["ctgan_hpo__rep2", "ctgan_hpo__rep1"],
                "status": ["warn", "pass"],
                "value": [0.2, 0.0],
                "limit": [0.05, 0.0],
                "detail": ["worst: a -> b", ""],
            }
        ),
        "tstr": pd.DataFrame(
            {
                "model": models,
                "tstr_macro_auprc": [0.6, 0.5, 0.5, 0.8],
                "tsts_macro_auprc": [0.9, 0.9, 0.6, 0.8],
            }
        ),
    }
    card = build_model_card("ctgan_hpo", Config(), dataset, _combined(models), tables)
    assert [line[3:] for line in card.splitlines() if line.startswith("## ")] == list(SECTIONS)
    assert "tuned by the hyperparameter search" in card and '"n_iter": 300' in card
    # The warning of one replicate is listed; the passing check is not.
    assert "`dependency_rules` on `ctgan_hpo__rep2`: 0.200 (limit 0.050); worst: a -> b" in card
    assert "columns_match" not in card
    assert "1 metric(s) missing" in card and "`distance_mia_auc`" in card
    # Mean over the two replicates next to both baselines, and the TSTS gap.
    assert "| 0.550 | 0.500 | 0.800 |" in card
    assert "TSTS minus TSTR macro AUPRC: 0.350" in card


def test_card_says_when_the_search_found_no_passing_trial():
    models = ["ctgan_hpo__rep1", "baseline_marginals", "baseline_train_copy"]
    dataset = SimpleNamespace(name="toy", version=None, target_column="y", train_df=pd.DataFrame())
    tables = {"best_params": {"synthcity": {"ctgan": {}}}}
    card = build_model_card("ctgan_hpo", Config(), dataset, _combined(models), tables)
    assert "no search trial passed the screens" in card
    assert "The search found no trial passing the screens" in card
