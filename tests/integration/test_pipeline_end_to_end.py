"""End-to-end checks on one real impute → generate → evaluate → plot run.

Every assertion here reads artifacts the real CLIs wrote for the committed
clinic fixture (see ``fixtures/make_fixture.py``). Tests marked
``xfail(strict=True)`` pin a known validity bug in the current pipeline: they
fail today, and the strict marker makes the suite go red the moment the bug
is fixed so the marker gets removed and the check starts guarding the fix.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from synthdata.data import load_dataset, load_imputed_splits
from synthdata.generation.hpo import SCREENS

from .conftest import FIXTURE_CSV

pytestmark = pytest.mark.integration

FEATURES = ["AGE", "SEX", "SITE", "SMOKER", "BMI", "LAB_A", "LAB_B", "SEVERITY", "DIAGNOSIS"]
COLUMNS = [*FEATURES, "target"]
CATEGORICAL = ["SEX", "SITE", "SMOKER", "SEVERITY", "DIAGNOSIS", "target"]
NUMERIC = ["AGE", "BMI", "LAB_A", "LAB_B"]
HPO_MODELS = ["bayesian_network", "ctgan"]
MODELS = sorted([*HPO_MODELS, *(f"{m}_hpo" for m in HPO_MODELS)])
BASELINES = ["baseline_marginals", "baseline_train_copy"]


@pytest.fixture(scope="module")
def dataset(pipeline_run):
    return load_imputed_splits(load_dataset(pipeline_run.cfg))


@pytest.fixture(scope="module")
def source():
    return pd.read_csv(FIXTURE_CSV)


def _category_set(values: pd.Series) -> set[str]:
    """Category labels with numeric codes normalised (1 == 1.0)."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return {repr(float(v)) for v in numeric}
    return set(values.astype(str))


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def test_split_is_disjoint_complete_and_stratified(dataset, pipeline_run, source):
    cfg = pipeline_run.cfg
    train, test = dataset.train_df, dataset.test_df
    assert set(train.index).isdisjoint(test.index)
    assert sorted([*train.index, *test.index]) == list(range(len(dataset.full_df)))
    # Fractions are row shares built from whole patients; train includes tuning.
    n_patients = source["patient_id"].nunique()
    assert dataset.n_patients["train"] + dataset.n_patients["test"] == n_patients
    assert abs(len(test) / len(dataset.full_df) - cfg.data.holdout_fraction) <= 0.03
    # Stratified on the row-level target; balance stays close.
    assert abs(train["target"].mean() - test["target"].mean()) <= 0.05


def test_patient_identifier_never_reaches_models(dataset, pipeline_run):
    frames = {
        "full": dataset.full_df,
        "train_imputed": dataset.train_imputed_df,
        "test_imputed": dataset.test_imputed_df,
        **pipeline_run.synthetic(),
    }
    for name, frame in frames.items():
        assert "patient_id" not in frame.columns, name


def test_patients_do_not_span_train_and_test(dataset, source):
    train_patients = set(source.loc[dataset.train_df.index, "patient_id"])
    test_patients = set(source.loc[dataset.test_df.index, "patient_id"])
    assert train_patients.isdisjoint(test_patients)


def test_tuning_split_is_a_patient_disjoint_part_of_train(dataset, source, pipeline_run):
    tuning = dataset.tuning_df
    search = dataset.search_train_df
    assert set(tuning.index) <= set(dataset.train_df.index)
    tuning_patients = set(source.loc[tuning.index, "patient_id"])
    assert tuning_patients.isdisjoint(source.loc[search.index, "patient_id"])
    assert tuning_patients.isdisjoint(source.loc[dataset.test_df.index, "patient_id"])
    share = len(tuning) / len(dataset.full_df)
    assert abs(share - pipeline_run.cfg.data.tuning_fraction) <= 0.03


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def test_imputation_fills_every_missing_feature(dataset):
    assert dataset.full_df[FEATURES].isna().any().any(), "fixture must contain missing values"
    for name in ("full_imputed_df", "train_imputed_df", "test_imputed_df"):
        frame = getattr(dataset, name)
        assert not frame[COLUMNS].isna().any().any(), name


def test_imputation_preserves_observed_values(dataset):
    raw, imputed = dataset.full_df, dataset.full_imputed_df
    assert raw.index.equals(imputed.index)
    for column in COLUMNS:
        observed = raw[column].notna()
        if column in NUMERIC:
            np.testing.assert_allclose(
                imputed.loc[observed, column].astype(float),
                raw.loc[observed, column].astype(float),
                rtol=0,
                atol=1e-5,
                err_msg=column,
            )
        else:
            assert (
                imputed.loc[observed, column].astype(str) == raw.loc[observed, column].astype(str)
            ).all(), column


def test_imputed_values_are_valid_and_rounded(dataset, pipeline_run):
    raw, imputed = dataset.full_df, dataset.full_imputed_df
    round_rules = pipeline_run.cfg.imputation.round_rules
    for column in FEATURES:
        missing = raw[column].isna()
        if not missing.any():
            continue
        filled = imputed.loc[missing, column]
        if column in CATEGORICAL:
            assert set(filled.astype(str)) <= set(raw[column].dropna().astype(str)), column
        else:
            values = filled.astype(float)
            assert np.isfinite(values).all(), column
            if column in round_rules:
                decimals = round_rules[column]
                np.testing.assert_allclose(values, values.round(decimals), atol=1e-9)


def test_imputed_splits_hold_the_same_rows_as_raw_splits(dataset):
    # Compare on columns that are never missing, as row multisets: the imputed
    # split files are not stored in the raw split's row order.
    complete = [c for c in COLUMNS if dataset.full_df[c].notna().all()]

    def rows(frame):
        values = frame[complete].copy()
        for column in complete:
            if column in NUMERIC:
                values[column] = values[column].astype(float).round(4)
            else:
                values[column] = values[column].astype(str)
        return sorted(map(tuple, values.to_numpy().tolist()))

    assert rows(dataset.train_imputed_df) == rows(dataset.train_df)
    assert rows(dataset.test_imputed_df) == rows(dataset.test_df)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_every_configured_model_produced_valid_synthetic_data(pipeline_run, dataset):
    synthetic = pipeline_run.synthetic()
    assert sorted(synthetic) == sorted(MODELS)
    n_samples = pipeline_run.cfg.generation.n_samples
    train = dataset.train_imputed_df
    indicators = list(dataset.missing_indicator_columns)
    for name, frame in synthetic.items():
        assert list(frame.columns) == [*COLUMNS, *indicators], name
        assert set(frame[indicators].stack().unique()) <= {0, 1}, name
        assert len(frame) == n_samples, name
        assert not frame.isna().any().any(), name
        for column in NUMERIC:
            assert np.isfinite(frame[column].astype(float)).all(), (name, column)
        for column in CATEGORICAL:
            generated = _category_set(frame[column])
            categories = _category_set(train[column])
            assert generated <= categories, (name, column, generated - categories)
        # Both classes present, so downstream classifiers are trainable.
        assert frame["target"].nunique() == 2, name


def test_missing_indicators_and_released_copies(pipeline_run, dataset):
    indicators = dataset.missing_indicator_columns
    assert indicators, "fixture columns with missing values must get indicators"
    for name, column in indicators.items():
        assert (dataset.full_df[name] == dataset.full_df[column].isna()).all(), name
    for name, frame in pipeline_run.synthetic().items():
        released = pd.read_csv(pipeline_run.generation_dir / "released" / f"{name}.csv")
        assert list(released.columns) == COLUMNS, name
        for indicator, column in indicators.items():
            flagged = frame[indicator] == 1
            assert released.loc[flagged, column].isna().all(), (name, column)
            assert released.loc[~flagged, column].notna().all(), (name, column)


def test_hpo_uses_an_imputer_fitted_without_tuning_rows(dataset):
    assert dataset.search_imputed_df is not None
    drift = pd.read_csv(dataset.data_dir / "imputation_drift.csv")
    assert set(drift["column"]) <= set(FEATURES)
    assert (drift["drift"] >= 0).all()


def test_hpo_studies_ran_and_recorded_best_params(pipeline_run):
    optuna = pytest.importorskip("optuna")
    best = json.loads((pipeline_run.generation_dir / "hpo_best_params.json").read_text())
    assert set(best["synthcity"]) == set(HPO_MODELS)
    storage = f"sqlite:///{pipeline_run.generation_dir / 'optuna_studies.db'}"
    for model in HPO_MODELS:
        study = optuna.load_study(study_name=f"hpo_{model}", storage=storage)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(complete) == pipeline_run.cfg.generation.hpo.n_trials, model
        assert all(math.isfinite(t.value) for t in complete), model
        # Training length is searched like any other hyperparameter and kept.
        assert study.best_params == best["synthcity"][model], model


def _hpo_studies(run, models=HPO_MODELS):
    optuna = pytest.importorskip("optuna")
    storage = f"sqlite:///{run.generation_dir / 'optuna_studies.db'}"
    return optuna, {
        model: optuna.load_study(study_name=f"hpo_{model}", storage=storage) for model in models
    }


def test_hpo_trials_score_tstr_on_tuning_rows_and_record_screens(pipeline_run):
    optuna, studies = _hpo_studies(pipeline_run)
    for model, study in studies.items():
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            attrs = trial.user_attrs
            # The objective is TSTR macro-F1; the other scores are logged next to it.
            assert trial.value == pytest.approx(attrs["tstr_macro_f1"]), model
            for key in ("tstr_macro_auprc", "tstr_balanced_accuracy", "trtr_macro_f1"):
                assert 0 <= attrs[key] <= 1, (model, key)
            assert sorted(attrs["tstr_per_class_f1"]) == ["0", "1"], model
            # Every sanity screen is passed to Optuna as a constraint.
            assert len(attrs["constraints"]) == len(SCREENS), model
            assert set(attrs["screens"]) == set(SCREENS), model


def test_hpo_searches_epochs_in_range_and_reports_to_the_pruner(pipeline_run):
    optuna, studies = _hpo_studies(pipeline_run)
    low, high, step = pipeline_run.cfg.generation.hpo.epoch_ranges["ctgan"]
    ctgan = studies["ctgan"].trials
    epochs = {t.params["n_iter"] for t in ctgan}
    assert epochs <= set(range(low, high + 1, step)), epochs
    # CTGAN's early-stopping metric is reported to Optuna's median pruner.
    assert all(t.intermediate_values for t in ctgan), [t.number for t in ctgan]


def test_experiment_manifest_records_each_stage(pipeline_run):
    manifest = pipeline_run.experiment_manifest
    assert manifest["dataset_name"] == "clinic"
    assert manifest["dataset_version"] == "fixture-1"
    stages = [entry["stage"] for entry in manifest["runs"]]
    assert stages[:2] == ["generation", "evaluation"]
    assert "plots" in stages
    generation = manifest["runs"][0]
    assert sorted(generation["artifacts"]["models"]) == sorted(MODELS)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

#: Metrics that must be present for every model, with their valid range.
BOUNDED_METRICS = {
    ("synthcity", "utility", "stats.ks_test.marginal"): (0, 1),
    ("synthcity", "utility", "stats.prdc.precision"): (0, 1),
    ("synthcity", "utility", "stats.prdc.recall"): (0, 1),
    ("synthcity", "utility", "stats.prdc.coverage"): (0, 1),
    ("synthcity", "utility", "stats.alpha_precision.authenticity_OC"): (0, 1),
    ("synthcity", "utility", "performance.xgb.gt"): (0, 1),
    ("synthcity", "utility", "performance.xgb.syn_id"): (0, 1),
    ("synthcity", "utility", "performance.xgb.syn_ood"): (0, 1),
    ("synthcity", "utility", "detection.detection_xgb.mean"): (0, 1),
    ("synthcity", "privacy", "privacy.identifiability_score.score"): (0, 1),
    ("synthcity", "privacy", "privacy.DomiasMIA_prior.aucroc"): (0, 1),
    ("syntheval", "utility", "avg_h_dist"): (0, 1),
    ("syntheval", "utility", "ks_tvd_stat"): (0, 1),
    ("syntheval", "privacy", "hit_rate"): (0, 1),
    ("syntheval", "utility", "nnaa"): (0, 1),
    ("custom", "fairness", "log_disparity_share_significant"): (0, 1),
    ("custom", "utility", "tstr_macro_f1"): (0, 1),
    ("custom", "utility", "tstr_balanced_accuracy"): (0, 1),
    ("custom", "utility", "tstr_macro_auprc"): (0, 1),
    ("custom", "privacy", "anonymeter_singling_out_risk"): (0, 1),
    ("custom", "privacy", "anonymeter_linkability_risk"): (0, 1),
    ("custom", "privacy", "anonymeter_inference_risk"): (0, 1),
    ("custom", "privacy", "dcr_closer_to_train_share"): (0, 1),
    ("custom", "privacy", "distance_mia_auc"): (0, 1),
    ("custom", "privacy", "dcr_holdout_ratio"): (0, math.inf),
    ("custom", "privacy", "nndr_holdout_ratio"): (0, math.inf),
}


def test_combined_table_has_one_complete_row_per_model(pipeline_run):
    combined = pipeline_run.combined()
    assert sorted(combined.index) == sorted(MODELS + BASELINES)
    assert set(combined.columns.get_level_values("framework")) >= {
        "synthcity",
        "syntheval",
        "custom",
        "__all__",
    }
    metrics = list(combined.columns)
    all_missing = [c for c in metrics if combined[c].isna().all()]
    assert not all_missing, all_missing


@pytest.mark.parametrize("column", sorted(BOUNDED_METRICS), ids=lambda c: c[2])
def test_bounded_metrics_are_present_finite_and_in_range(pipeline_run, column):
    combined = pipeline_run.combined()
    assert column in combined.columns
    low, high = BOUNDED_METRICS[column]
    values = combined[column].astype(float)
    assert np.isfinite(values).all()
    assert ((values >= low) & (values <= high)).all(), values.to_dict()


def test_tstr_has_signal_on_the_fixture(pipeline_run, dataset):
    # The fixture's target follows a known logistic model, so a model trained on
    # real train must clearly beat chance on test. Guards against target/feature
    # misalignment. (SynthCity's performance.xgb.gt cross-validates inside the
    # small test split, so it is too noisy to threshold here.)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def features(frame):
        nominal = ["SEX", "SITE", "SMOKER", "DIAGNOSIS"]
        encoded = frame[FEATURES].astype({c: str for c in nominal})
        return pd.get_dummies(encoded, columns=nominal).astype(float)

    train, test = dataset.train_imputed_df, dataset.test_imputed_df
    x_train = features(train)
    x_test = features(test).reindex(columns=x_train.columns, fill_value=0.0)
    model = LogisticRegression(max_iter=2000).fit(x_train, train["target"])
    auc = roc_auc_score(test["target"], model.predict_proba(x_test)[:, 1])
    assert auc > 0.65, auc
    real_auc = pipeline_run.combined()[("synthcity", "utility", "performance.xgb.gt")]
    assert real_auc.astype(float).between(0, 1).all()


def test_ranks_are_normalised_and_overall_is_their_geometric_mean(pipeline_run):
    combined = pipeline_run.combined()
    dims = ["utility", "privacy", "fairness"]
    for dim in dims:
        rank = combined[("__all__", dim, "rank")].astype(float)
        assert ((rank >= 0) & (rank <= 1)).all(), dim
    overall = combined[("__all__", "overall", "rank")].astype(float)
    weights = pipeline_run.cfg.evaluation.rank_weights
    logs = sum(
        weights[d] * np.log(combined[("__all__", d, "rank")].astype(float).clip(lower=0.01))
        for d in dims
    )
    expected = np.exp(logs / sum(weights[d] for d in dims))
    np.testing.assert_allclose(overall, expected, rtol=1e-6)


def test_copying_training_rows_does_not_pay_off_overall(pipeline_run):
    # The copy has the best utility and the worst privacy; a sum of the two
    # ranked it above every generator but one. Its privacy must now keep it
    # below the best generator. (The fixture's generators are tiny, so a
    # stricter "below every generator" would test the fixture, not the score.)
    overall = pipeline_run.combined()[("__all__", "overall", "rank")].astype(float)
    generators = overall[[m for m in overall.index if not m.startswith("baseline_")]]
    assert overall["baseline_train_copy"] < generators.max(), overall.to_dict()


def test_combined_table_has_no_privacy_gate(pipeline_run):
    assert "privacy_gate" not in pipeline_run.combined().columns.get_level_values("type")


def test_syntheval_fairness_metrics_are_reported(pipeline_run):
    columns = set(pipeline_run.combined().columns.get_level_values("metric"))
    assert {"statistical_parity", "equalized_odds", "equal_opportunity"} <= columns


def test_syntheval_privacy_results_are_classified_as_privacy(pipeline_run):
    combined = pipeline_run.combined()
    types = {
        metric: kind for framework, kind, metric in combined.columns if framework == "syntheval"
    }
    for metric in ("mia_recall", "mia_precision", "att_discl_risk", "median_DCR", "avg_nndr"):
        assert types.get(metric) == "privacy", (metric, types.get(metric))


def test_feature_importance_rank_distance_is_reported(pipeline_run):
    metrics = set(pipeline_run.combined().columns.get_level_values("metric"))
    assert any(m.startswith("performance.feat_rank_distance") for m in metrics)


def test_holdout_tstr_is_ranked_and_reported_per_class(pipeline_run):
    combined = pipeline_run.combined()
    tstr = combined[("custom", "utility", "tstr_macro_f1")]
    assert tstr.notna().all() and tstr.between(0, 1).all()
    assert ("custom", "utility", "rank") in combined.columns
    table = pd.read_csv(pipeline_run.evaluation_dir / "tstr_holdout.csv", index_col=0)
    assert "trtr (real train)" in table.index
    assert {"f1_0", "f1_1"} <= set(table.columns)
    assert "trtr (real train)" in (pipeline_run.evaluation_dir / "report.md").read_text()


def test_ranking_summary_has_one_row_per_model(pipeline_run):
    summary = pd.read_csv(pipeline_run.evaluation_dir / "ranking_summary.csv", index_col=0)
    assert sorted(summary.index) == sorted(MODELS + BASELINES)
    assert (summary["n_replicates"] == 1).all()
    assert summary.loc[summary["baseline"], "eligible"].eq(False).all()
    report = (pipeline_run.evaluation_dir / "report.md").read_text()
    assert "Single seed" in report


def test_report_covers_every_model(pipeline_run):
    report = (pipeline_run.evaluation_dir / "report.md").read_text()
    for model in MODELS:
        assert model in report


def test_privacy_attacks_table_has_every_attack_with_an_interval(pipeline_run):
    table = pd.read_csv(pipeline_run.evaluation_dir / "privacy_attacks.csv")
    assert set(table["model"]) == set(MODELS + BASELINES)
    sensitive = pipeline_run.cfg.data.sensitive_columns
    for model, rows in table.groupby("model"):
        assert set(rows["attack"]) == {"singling_out", "linkability", "inference"}, model
        assert set(rows.loc[rows["attack"] == "inference", "secret"]) == set(sensitive), model
        assert (rows["ci_low"] <= rows["risk"]).all() and (rows["risk"] <= rows["ci_high"]).all()


def test_report_has_its_seven_sections_and_working_links(pipeline_run):
    report_path = pipeline_run.evaluation_dir / "report.md"
    report = report_path.read_text()
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings == [
        "## 1. At a glance",
        "## 2. Ranking",
        "## 3. Utility",
        "## 4. Privacy evidence",
        "## 5. Fairness",
        "## 6. Data and preparation",
        "## 7. File index",
    ]
    # Every embedded plot and linked file resolves (no privacy gate left).
    targets = re.findall(r"\]\(([^)#\s]+)\)", report)
    assert targets
    broken = [t for t in targets if "://" not in t and not (report_path.parent / t).exists()]
    assert not broken, broken
    assert "privacy gate" not in report.lower()


# ---------------------------------------------------------------------------
# Plots and housekeeping
# ---------------------------------------------------------------------------


def test_plots_were_written_for_each_section(pipeline_run):
    dataset_plots = pipeline_run.plots_dir.parent / "dataset"
    expected = [
        dataset_plots / "data" / "column_distributions.png",
        dataset_plots / "data" / "missingness.png",
        dataset_plots / "imputation" / "observed_vs_imputed.png",
        pipeline_run.plots_dir / "evaluation" / "utility_vs_privacy.png",
        *(pipeline_run.plots_dir / "generation" / f"{model}.png" for model in MODELS),
        *(pipeline_run.plots_dir / "hpo" / f"hpo_{m}_history.html" for m in ("ctgan",)),
    ]
    missing = [str(p) for p in expected if not p.is_file() or p.stat().st_size == 0]
    assert not missing, missing


def test_nothing_is_written_outside_the_configured_roots(pipeline_run):
    assert list(pipeline_run.cwd.iterdir()) == []


def test_indicator_only_columns_are_filled_in_recorded_rows_only(indicator_only_run, source):
    dataset = load_dataset(indicator_only_run.cfg)
    sparse = dataset.indicator_only_values.columns.tolist()
    assert sorted(sparse) == ["BMI", "LAB_A"]
    synthetic = indicator_only_run.synthetic()["arf"]
    assert not set(sparse) & set(synthetic.columns), "values are never generated directly"
    released_dir = indicator_only_run.generation_dir / "released"
    released = pd.read_csv(released_dir / "arf.csv")
    for column in sparse:
        recorded = synthetic[f"{column}__missing"] == 0
        assert released.loc[recorded, column].notna().all(), column
        assert released.loc[~recorded, column].isna().all(), column
        observed = source[column].dropna()
        assert released[column].min() >= observed.min(), column
    report = pd.read_csv(released_dir / "arf_recorded_values.csv")
    assert sorted(report["column"]) == sorted(sparse)


def test_holdout_rows_do_not_influence_the_cart_fill(indicator_only_run, cart_canary_run):
    for name in ("arf.csv", "arf_recorded_values.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(indicator_only_run.generation_dir / "released" / name),
            pd.read_csv(cart_canary_run.generation_dir / "released" / name),
        )


def test_arf_and_its_search_run_with_delta_held_at_arfpys_default(indicator_only_run):
    assert sorted(indicator_only_run.synthetic()) == ["arf", "arf_hpo"]
    # synthcity searches delta over 0-50 but arfpy only accepts 0-0.5, so it is
    # held at 0 instead of searched; every trial must then complete.
    optuna, studies = _hpo_studies(indicator_only_run, ["arf"])
    trials = studies["arf"].trials
    assert len(trials) == indicator_only_run.cfg.generation.hpo.n_trials
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    assert all("delta" not in t.params for t in trials)
