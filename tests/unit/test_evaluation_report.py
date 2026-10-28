"""Unit tests for synthdata.evaluation.report: Markdown evaluation report
generation from a combined table + extras dict.
"""

from pathlib import Path

import pandas as pd
import pytest

from synthdata.evaluation.report import (
    MAX_EMBED_HEIGHT_PX,
    _Links,
    build_evaluation_report,
    save_evaluation_report,
)
from synthdata.experiment import imputation_output_dir

pytestmark = pytest.mark.unit


def _combined_table():
    df = pd.DataFrame(index=["model_a", "model_b"])
    df[("syntheval", "utility", "ks_test")] = [0.9, 0.1]
    df[("__all__", "utility", "rank")] = [1.0, 0.0]
    df[("__all__", "privacy", "rank")] = [0.5, 0.5]
    df[("__all__", "fairness", "rank")] = [0.5, 0.5]
    df[("__all__", "overall", "rank")] = [2.0, 1.0]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _section(text: str, title: str) -> str:
    return text.split(f"## {title}")[1].split("\n## ")[0]


class TestBuildEvaluationReport:
    def test_contains_expected_section_headers(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        for header in (
            "# Evaluation report",
            "## 1. At a glance",
            "## 2. Ranking",
            "## 3. Utility",
            "## 4. Privacy evidence",
            "## 5. Fairness",
            "## 6. Data and preparation",
            "## 7. File index",
        ):
            assert header in text
        assert "privacy gate" not in text.lower()

    def test_fairness_section_shows_holdout_gaps_and_the_no_signal_gate(
        self, make_config, make_dataset
    ):
        from synthdata.evaluation.fairness_gaps import FAIRNESS_GAP_METRICS

        gaps = pd.DataFrame(
            {
                "fairness_signal": [1, 0],
                "holdout_balanced_accuracy": [0.61, 0.33],
                "balanced_accuracy_ci_low": [0.58, 0.31],
                "balanced_accuracy_ci_high": [0.64, 0.35],
                "chance_balanced_accuracy": [1 / 3, 1 / 3],
                **{m: [0.04, float("nan")] for m in FAIRNESS_GAP_METRICS},
            },
            index=pd.Index(["model_a", "model_b"], name="model"),
        )
        extras = {"selected_datasets": {}, "fairness_gaps": gaps}
        text = build_evaluation_report(make_config(), make_dataset(), _combined_table(), extras)
        fairness = _section(text, "5. Fairness")
        assert "### Holdout fairness gaps" in fairness
        assert "demographic_parity_gap_error" in fairness
        assert "reaches chance (0.333)" in fairness
        assert "### Subgroup gap metrics (reported, not ranked)" in fairness
        assert "fairness_gaps_by_group.csv" in fairness

    def test_recommends_top_overall_rank_model(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        assert "Recommended model: `model_a`" in _section(text, "1. At a glance")

    def test_baseline_rows_are_never_recommended(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        combined = combined.rename(index={"model_a": "baseline_train_copy"})
        extras = {"selected_datasets": {"baseline_train_copy": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        assert "Recommended model: `model_b`" in _section(text, "1. At a glance")
        assert "fixed references" in _section(text, "2. Ranking")

    def test_replicates_report_intervals_and_ties(self, make_config, make_dataset):
        from synthdata.evaluation.combine import summarize_replicates

        cfg = make_config()
        dataset = make_dataset()
        combined = pd.DataFrame(index=["a", "a__rep1", "b", "b__rep1", "c", "c__rep1"])
        combined[("__all__", "overall", "rank")] = [2.0, 2.2, 1.9, 2.3, 0.1, 0.2]
        combined.columns = pd.MultiIndex.from_tuples(combined.columns)
        extras = {"ranking_summary": summarize_replicates(combined)}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        ranked = _section(text, "2. Ranking")
        assert "95% confidence interval" in ranked
        assert "2.100 [" in ranked
        glance = _section(text, "1. At a glance")
        assert "Recommended model: `a`" in glance or "Recommended model: `b`" in glance
        assert "Tied with it within seed noise:" in glance
        assert "`c`" not in glance.split("Tied with it")[1]
        assert "Single seed" not in glance

    def test_hpo_that_fell_back_to_untuned_settings_is_marked(
        self, make_config, make_dataset, tmp_path
    ):
        import json

        cfg = make_config()
        cfg.generation.output_dir = str(tmp_path)
        (tmp_path / "hpo_best_params.json").write_text(
            json.dumps({"synthcity": {"model": {}, "other": {"n_iter": 5}}})
        )
        combined = _combined_table().rename(index={"model_a": "model_hpo"})
        text = build_evaluation_report(cfg, make_dataset(), combined, {"selected_datasets": {}})
        ranked = _section(text, "2. Ranking")
        assert "`model_hpo` = `model`" in ranked and "other_hpo" not in ranked

    def test_single_seed_report_says_there_is_no_uncertainty(self, make_config, make_dataset):
        text = build_evaluation_report(
            make_config(), make_dataset(), _combined_table(), {"selected_datasets": {}}
        )
        assert "Single seed" in _section(text, "1. At a glance")

    def test_fairness_highlights_include_log_disparity_error_rows(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {
            "selected_datasets": {"model_a": None, "model_b": None},
            "log_disparity_reports": {
                "model_a": {
                    "summary_stats": {
                        "mean_abs_log_disparity": 0.2,
                        "median_abs_log_disparity": 0.15,
                        "share_significant_bh": 0.0,
                    }
                },
                "model_b": {"error": "boom", "error_type": "KeyError"},
            },
        }
        text = build_evaluation_report(cfg, dataset, combined, extras)
        fairness_section = _section(text, "5. Fairness")
        assert "model_a" in fairness_section
        assert "model_b" in fairness_section

    def test_experiment_id_included_when_provided(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}

        class _FakeExperiment:
            id = "20260101T000000Z_test"

        text = build_evaluation_report(cfg, dataset, combined, extras, experiment=_FakeExperiment())
        assert "20260101T000000Z_test" in text

    def test_class_level_section_shows_tstr_and_per_class_values(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        tstr = pd.DataFrame(
            {"tstr_macro_f1": [0.4, 0.6], "f1_rare": [0.0, 0.5]},
            index=pd.Index(["model_a", "trtr (real train)"], name="model"),
        )
        ovr = pd.DataFrame(
            {("auroc_diff", "0"): [0.1], ("auroc_diff", "1"): [0.3]}, index=["model_a"]
        )
        extras = {"selected_datasets": {}, "tstr_table": tstr, "ovr_per_class": ovr}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        utility = _section(text, "3. Utility")
        assert "trtr (real train)" in utility
        assert "auroc_diff [1]" in utility

    def test_privacy_evidence_shows_attacks_and_flags_beaten_baselines(
        self, make_config, make_dataset
    ):
        cfg = make_config()
        combined = _combined_table()
        combined[("custom", "privacy", "anonymeter_linkability_risk")] = [0.3, 0.0]
        combined[("custom", "privacy", "distance_mia_auc")] = [0.55, 0.5]
        attacks = pd.DataFrame(
            {
                "model": ["model_a", "model_b"],
                "attack": ["linkability", "linkability"],
                "secret": [None, None],
                "risk": [0.3, 0.0],
                "ci_low": [0.1, -0.05],
                "ci_high": [0.5, 0.05],
                "reliable": [True, True],
            }
        )
        extras = {"selected_datasets": {}, "privacy_result": {"attacks": attacks}}
        text = build_evaluation_report(cfg, make_dataset(), combined, extras)
        privacy = _section(text, "4. Privacy evidence")
        assert "not guarantees" in privacy
        assert "anonymeter_linkability_risk" in privacy and "distance_mia_auc" in privacy
        glance = _section(text, "1. At a glance")
        assert "`model_a`: Anonymeter linkability succeeded" in glance
        assert "`model_b`: Anonymeter" not in glance

    def test_fairness_lists_privacy_by_protected_group(self, make_config, make_dataset):
        cfg = make_config()
        combined = _combined_table()
        combined[("custom", "privacy", "distance_mia_auc")] = [0.6, 0.5]
        combined[("custom", "fairness", "subgroup_distance_mia_auc_gap")] = [0.2, 0.02]
        combined[("custom", "fairness", "subgroup_distance_mia_auc_max")] = [0.7, 0.51]
        subgroups = pd.DataFrame(
            {
                "model": ["model_a", "model_a", "model_b", "model_b"],
                "column": ["sex"] * 4,
                "group": ["F", "M"] * 2,
                "n_train_patients": [40, 35] * 2,
                "n_test_patients": [12, 11] * 2,
                "distance_mia_auc": [0.7, 0.5, 0.51, 0.49],
                "auc_se": [0.08, 0.09, 0.09, 0.1],
            }
        )
        extras = {"selected_datasets": {}, "privacy_result": {"subgroups": subgroups}}
        text = build_evaluation_report(cfg, make_dataset(), combined, extras)
        fairness = _section(text, "5. Fairness")
        part = fairness.split("### Privacy by protected group")[1]
        assert "subgroup_distance_mia_auc_gap" in part and "not ranked" in part
        assert "sex = F" in part and "sex = M" in part
        assert "sex = F 40/12" in part
        assert "Privacy by protected group" in _section(text, "4. Privacy evidence")

    def test_privacy_by_group_says_when_it_was_not_run(self, make_config, make_dataset):
        text = build_evaluation_report(
            make_config(), make_dataset(), _combined_table(), {"selected_datasets": {}}
        )
        part = _section(text, "5. Fairness").split("### Privacy by protected group")[1]
        assert "Not computed this run" in part

    def test_reads_tables_back_and_embeds_plots_from_disk(self, make_config, make_dataset):
        cfg = make_config()
        eval_dir = Path(cfg.evaluation.output_dir)
        eval_dir.mkdir(parents=True)
        tstr = pd.DataFrame({"tstr_macro_f1": [0.4]}, index=pd.Index(["model_a"], name="model"))
        tstr.to_csv(eval_dir / "tstr_holdout.csv")
        plots = Path(cfg.plots.output_dir)
        (plots / "evaluation").mkdir(parents=True)
        (plots / "evaluation" / "utility_vs_privacy.png").write_bytes(b"png")
        (plots / "generation").mkdir()
        (plots / "generation" / "model_a.png").write_bytes(b"png")
        text = build_evaluation_report(cfg, make_dataset(), _combined_table(), {})
        assert "![utility vs privacy](../plots/evaluation/utility_vs_privacy.png)" in text
        assert "![model a](../plots/generation/model_a.png)" in _section(text, "3. Utility")
        assert "[tstr_holdout.csv](tstr_holdout.csv)" in _section(text, "7. File index")

    def test_wide_data_gets_a_column_summary_and_tall_figures_are_linked(
        self, make_config, make_dataset
    ):
        cfg = make_config()
        generation = Path(cfg.plots.output_dir) / "generation"
        generation.mkdir(parents=True)
        pd.DataFrame(
            {"model_a": [0.02, 0.4, 0.05], "model_b": [0.01, 0.03, 0.02]},
            index=pd.Index(["age", "income", "sex"], name="column"),
        ).to_csv(generation / "column_differences.csv")
        tall_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (1000).to_bytes(4, "big")
        tall_png += (MAX_EMBED_HEIGHT_PX + 1).to_bytes(4, "big")
        for model in ("model_a", "model_b"):
            (generation / f"{model}.png").write_bytes(tall_png)
            (generation / f"{model}_top_columns.png").write_bytes(b"png")
        utility = _section(
            build_evaluation_report(cfg, make_dataset(), _combined_table(), {}), "3. Utility"
        )
        assert "income (0.40)" in utility  # model_a's worst column
        assert utility.index("**model_b**") < utility.index("**model_a**")  # closest first
        assert (
            "![model_a: columns that differ most](../plots/generation/model_a_top_columns.png)"
            in utility
        )
        assert "Every column: [model_a.png](../plots/generation/model_a.png)" in utility
        assert "![model a](" not in utility

        dataset_plots = Path(cfg.plots.output_dir).parent / "dataset" / "data"
        links = _Links(Path(cfg.evaluation.output_dir))
        dataset_plots.mkdir(parents=True)
        (dataset_plots / "column_distributions.png").write_bytes(tall_png)
        assert "large figure" in links.plots(dataset_plots)[0]


class TestSaveEvaluationReport:
    def test_writes_report_to_evaluation_output_dir(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        path = save_evaluation_report(cfg, dataset, combined, extras)
        assert path.exists()
        assert path.name == "report.md"
        assert path.read_text().startswith("# Evaluation report")


@pytest.mark.parametrize("ci_low", [0.0, 0.2])
def test_privacy_flags_handles_no_and_some_successful_attacks(ci_low):
    from synthdata.evaluation.report import _privacy_flags

    attacks = pd.DataFrame(
        {
            "model": ["ctgan", "baseline_train_copy"],
            "attack": ["linkability", "linkability"],
            "secret": ["", ""],
            "ci_low": [ci_low, 0.5],
            "reliable": [True, True],
        }
    )
    flags = _privacy_flags(attacks)
    assert len(flags) == (1 if ci_low > 0 else 0)
    assert all("baseline_train_copy" not in flag for flag in flags)


def test_data_section_summarizes_the_readiness_findings(make_config, make_dataset):
    cfg = make_config()
    findings = pd.DataFrame(
        {
            "check": ["constant", "constant", "skewed"],
            "column": ["a", "b", "c"],
            "value": [1, 1, 3.1],
            "threshold": [1, 1, 2],
            "status": ["warn", "warn", "info"],
            "action": ["drop", "drop", "log"],
        }
    )
    text = build_evaluation_report(cfg, make_dataset(), _combined_table(), {})
    assert "_Not run" in _section(text, "6. Data and preparation")
    findings.to_csv(imputation_output_dir(cfg) / "data_readiness.csv", index=False)
    data = _section(
        build_evaluation_report(cfg, make_dataset(), _combined_table(), {}),
        "6. Data and preparation",
    )
    assert "| constant | warn | 2 | a, b |" in data
    assert "| skewed | info | 1 | c |" in data


def test_imputation_reports_are_found_after_the_experiment_nests_output_dir(
    make_config, make_dataset, tmp_path
):
    from synthdata.experiment import start_experiment

    cfg = make_config()
    readiness = imputation_output_dir(cfg) / "data_readiness.csv"
    pd.DataFrame(
        {
            "check": ["constant"],
            "column": ["a"],
            "value": [1],
            "threshold": [1],
            "status": ["warn"],
            "action": ["drop"],
        }
    ).to_csv(readiness, index=False)
    (imputation_output_dir(cfg) / "imputation_validation_report.csv").write_text("column\n")
    experiment = start_experiment(cfg)
    # As scripts/run_evaluation.py does before evaluating.
    cfg.generation.output_dir = str(experiment.generation_dir)
    data = _section(
        build_evaluation_report(cfg, make_dataset(), _combined_table(), {}, experiment=experiment),
        "6. Data and preparation",
    )
    assert "| constant | warn | 1 | a |" in data
    assert "imputation_validation_report.csv](" in data
    assert "HPO is off" in data or "imputation_drift.csv" in data


def test_replicate_rows_fold_into_a_mean_and_interval():
    from synthdata.evaluation.report import _collapse_replicates

    table = pd.DataFrame(
        {"score": [0.5, 0.6, 0.7, 0.9], "link": ["a", "b", "c", "d"]},
        index=pd.Index(["ctgan", "ctgan__rep1", "ctgan__rep2", "tvae"], name="model"),
    )
    folded = _collapse_replicates(table)
    assert list(folded.index) == ["ctgan", "tvae"]
    assert folded.loc["ctgan", "seeds"] == 3
    assert folded.loc["ctgan", "score"].startswith("0.600 [")
    assert folded.loc["ctgan", "link"] == "a"
    assert folded.loc["tvae", "score"] == "0.9" or folded.loc["tvae", "score"].startswith("0.9")
