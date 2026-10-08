"""Unit tests for synthdata.evaluation.report: Markdown evaluation report
generation from a combined table + extras dict.
"""

from pathlib import Path

import pandas as pd
import pytest

from synthdata.evaluation.report import build_evaluation_report, save_evaluation_report

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
