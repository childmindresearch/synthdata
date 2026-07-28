"""Unit tests for synthdata.evaluation.report: Markdown evaluation report
generation from a combined table + extras dict.
"""

import pandas as pd
import pytest

from synthdata.evaluation.report import build_evaluation_report, save_evaluation_report

pytestmark = pytest.mark.unit


def _combined_table(with_gate: bool = False, all_pass: bool = True):
    df = pd.DataFrame(index=["model_a", "model_b"])
    df[("syntheval", "utility", "ks_test")] = [0.9, 0.1]
    df[("__all__", "utility", "rank")] = [1.0, 0.0]
    df[("__all__", "privacy", "rank")] = [0.5, 0.5]
    df[("__all__", "fairness", "rank")] = [0.5, 0.5]
    df[("__all__", "overall", "rank")] = [2.0, 1.0]
    if with_gate:
        df[("__all__", "privacy_gate", "pass")] = [True, all_pass]
        df[("__all__", "privacy_gate", "violations")] = ["", "" if all_pass else "mia_recall=0.9"]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestBuildEvaluationReport:
    def test_contains_expected_section_headers(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        for header in (
            "# Evaluation report",
            "## Run metadata",
            "## Ranked summary",
            "## Privacy gate",
            "## Recommended model",
            "## Fairness highlights",
            "## Plots",
        ):
            assert header in text

    def test_recommends_top_overall_rank_model_without_gate(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table(with_gate=False)
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        assert "`model_a`" in text.split("## Recommended model")[1].split("##")[0]

    def test_gate_failing_model_excluded_from_recommendation(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        # model_a has the higher overall rank but FAILS the gate; model_b passes.
        combined = _combined_table(with_gate=True, all_pass=True)
        combined[("__all__", "privacy_gate", "pass")] = [False, True]
        combined[("__all__", "privacy_gate", "violations")] = ["mia_recall=0.9 (max limit 0.6)", ""]
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        recommendation_section = text.split("## Recommended model")[1].split("##")[0]
        assert "`model_b`" in recommendation_section
        assert "`model_a`" not in recommendation_section

    def test_zero_models_pass_gate_does_not_raise_and_says_so(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table(with_gate=True)
        combined[("__all__", "privacy_gate", "pass")] = [False, False]
        combined[("__all__", "privacy_gate", "violations")] = [
            "mia_recall=0.9 (max limit 0.6)",
            "hit_rate=0.5 (max limit 0.05)",
        ]
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        recommendation_section = text.split("## Recommended model")[1].split("##")[0]
        assert "No model passed the privacy gate" in recommendation_section

    def test_privacy_gate_not_run_notes_caveat(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table(with_gate=False)
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        gate_section = text.split("## Privacy gate")[1].split("##")[0]
        assert "not run" in gate_section.lower()

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
        fairness_section = text.split("## Fairness highlights")[1]
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
