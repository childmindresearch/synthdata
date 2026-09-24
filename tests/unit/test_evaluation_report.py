"""Unit tests for synthdata.evaluation.report: Markdown evaluation report
generation from a combined table + extras dict.
"""

import os
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from synthdata.evaluation.artifacts import _model_artifact_id
from synthdata.evaluation.report import build_evaluation_report, save_evaluation_report

pytestmark = pytest.mark.unit


def _dataframe(
    data: dict[object, list[object]], index: list[str] | pd.Index | None = None
) -> pd.DataFrame:
    return pd.DataFrame(
        cast("dict[str, list[object]]", data),
        index=pd.Index(index) if index is not None else None,
    )


def _combined_table(with_gate: bool = False, all_pass: bool = True):
    df = _dataframe({}, index=["model_a", "model_b"])
    df[("syntheval", "utility", "ks_test")] = [0.9, 0.1]
    df[("__all__", "utility", "rank")] = [1.0, 0.0]
    df[("__all__", "privacy", "rank")] = [0.5, 0.5]
    df[("__all__", "fairness", "rank")] = [0.5, 0.5]
    df[("__all__", "utility", "U_tuning")] = [0.8, 0.6]
    df[("__all__", "overall", "rank")] = [2.0, 1.0]
    if with_gate:
        df[("__all__", "privacy_gate", "pass")] = [True, all_pass]
        df[("__all__", "privacy_gate", "violations")] = ["", "" if all_pass else "mia_recall=0.9"]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _successful_log_disparity_report() -> dict:
    return {
        "state": "succeeded",
        "summary_stats": {
            "mean_abs_log_disparity": 0.1,
            "median_abs_log_disparity": 0.1,
            "share_significant_bh": 0.0,
        },
        **{
            table: pd.DataFrame({"value": [1]})
            for table in (
                "leaf_results",
                "hierarchy_results",
                "subgroup_table",
                "leaf_equity_table",
                "legend_table",
                "label_counts",
            )
        },
    }


class TestBuildEvaluationReport:
    def test_log_disparity_plot_link_requires_succeeded_state(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        html_path = (
            Path(cfg.plots.output_dir)
            / "evaluation"
            / "log_disparity"
            / f"{_model_artifact_id('good')}.html"
        )
        html_path.parent.mkdir(parents=True)
        html_path.write_text("<html></html>")

        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {
                "log_disparity_reports": {
                    "good": _successful_log_disparity_report(),
                    "failed": {"state": "failed", "error": "boom"},
                    "incomplete": {"state": "indeterminate", "reason": "missing tables"},
                }
            },
        )

        plots_section = text.split("## Plots", 1)[1]
        assert "Log disparity report (good)" in plots_section
        assert "Log disparity report (failed)" not in plots_section
        assert "Log disparity report (incomplete)" not in plots_section

    def test_plot_links_use_artifact_ids_and_include_existing_3d_plot(
        self, make_config, make_dataset
    ):
        cfg = make_config()
        dataset = make_dataset()
        plots_dir = Path(cfg.plots.output_dir) / "evaluation"
        plots_dir.mkdir(parents=True)
        (plots_dir / "rank_tradeoff_3d.html").write_text("<html></html>")
        model_name = "team/model: candidate"
        html_path = plots_dir / "log_disparity" / f"{_model_artifact_id(model_name)}.html"
        html_path.parent.mkdir(parents=True)
        html_path.write_text("<html></html>")

        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {"log_disparity_reports": {model_name: _successful_log_disparity_report()}},
        )

        plots_section = text.split("## Plots", 1)[1]
        assert "Utility, privacy, and fairness rank trade-off (3D)" in plots_section
        assert f"Log disparity report ({model_name})" in plots_section
        assert f"{_model_artifact_id(model_name)}.html" in plots_section

    def test_plot_links_are_relative_to_report_and_only_existing_files(
        self, make_config, make_dataset, tmp_path
    ):
        cfg = make_config()
        cfg.plots.formats = ("svg",)
        dataset = make_dataset()
        plots_dir = Path(cfg.plots.output_dir) / "evaluation"
        plots_dir.mkdir(parents=True)
        (plots_dir / "utility_vs_privacy.svg").write_text("svg")
        report_dir = tmp_path / "reports" / "nested"
        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {"log_disparity_reports": {}},
            report_dir=report_dir,
        )
        plots_section = text.split("## Plots", 1)[1]
        expected = os.path.relpath(plots_dir / "utility_vs_privacy.svg", report_dir).replace(
            os.sep, "/"
        )
        assert f"]({expected})" in plots_section
        assert "utility_vs_fairness" not in plots_section
        assert str(cfg.plots.output_dir) not in plots_section

    def test_rejects_combined_table_without_overall_rank(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table().drop(columns=[("__all__", "overall", "rank")])

        with pytest.raises(ValueError, match="overall rank"):
            build_evaluation_report(cfg, dataset, combined, {})

    def test_renders_blocked_legacy_report_without_candidates(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = pd.DataFrame(index=pd.Index([], name="model"))

        text = build_evaluation_report(cfg, dataset, combined, {})

        assert "No overall rank column was produced" in text

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

    def test_complete_coverage_keeps_existing_rank_and_recommendation_wording(
        self, make_config, make_dataset
    ):
        text = build_evaluation_report(
            make_config(),
            make_dataset(),
            _combined_table(),
            {"selected_datasets": {"model_a": None, "model_b": None}},
        )

        assert "## Ranked summary (higher = better)" in text
        assert "## Generation coverage" not in text
        assert "Selected using highest complete fixed-transform tuning utility only" in text
        assert "partial coverage" not in text.lower()

    def test_partial_coverage_labels_available_only_ranking_and_failed_outputs(
        self, make_config, make_dataset
    ):
        combined = _combined_table().loc[["model_a"]]
        text = build_evaluation_report(
            make_config(),
            make_dataset(),
            combined,
            {
                "selected_datasets": {"model_a": None},
                "evaluation_coverage": {
                    "status": "partial",
                    "requested_models": ["model_a", "stage_a_failed"],
                    "evaluated_models": ["model_a"],
                    "failed_outputs": ["stage_a_failed"],
                },
            },
        )

        assert "## Generation coverage" in text
        assert "Failed outputs: `stage_a_failed`" in text
        assert "not a complete-run ranking" in text
        assert "## Ranked summary (partial coverage; higher = better)" in text
        recommendation = text.split("## Recommended model")[1].split("##")[0]
        assert "available models only" in recommendation
        assert "coverage is partial" in recommendation

    def test_recommends_highest_complete_tuning_utility_without_gate(
        self, make_config, make_dataset
    ):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table(with_gate=False)
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        assert "`model_a`" in text.split("## Recommended model")[1].split("##")[0]

    def test_gate_failure_is_warning_not_selection_filter(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        # model_a has higher tuning utility but FAILS gate; selection still uses tuning utility.
        combined = _combined_table(with_gate=True, all_pass=True)
        combined[("__all__", "privacy_gate", "pass")] = [False, True]
        combined[("__all__", "privacy_gate", "violations")] = ["mia_recall=0.9 (max limit 0.6)", ""]
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        recommendation_section = text.split("## Recommended model")[1].split("##")[0]
        assert "`model_a`" in recommendation_section
        assert "failed; selection was not blocked" in recommendation_section

    def test_zero_models_pass_gate_does_not_block_recommendation(self, make_config, make_dataset):
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
        assert "`model_a`" in recommendation_section

    def test_privacy_gate_not_run_notes_caveat(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table(with_gate=False)
        extras = {"selected_datasets": {"model_a": None, "model_b": None}}
        text = build_evaluation_report(cfg, dataset, combined, extras)
        gate_section = text.split("## Privacy gate")[1].split("##")[0]
        assert "not run" in gate_section.lower()

    def test_incomplete_final_holdout_release_score_is_withheld(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        score = {
            "status": "succeeded",
            "score": 0.7,
            "audit_only": True,
            "dimensions": {
                "utility": {"score": 0.8},
                "privacy": {"score": 0.6, "identity": 0.5},
                "fairness": {"score": 0.9},
            },
        }
        text = build_evaluation_report(
            cfg,
            dataset,
            combined,
            {"final_holdout_evidence": {"release_score": score}},
        )
        section = text.split("## Final-holdout release score audit")[1].split("##")[0]
        assert "Status: `indeterminate`" in section
        assert "Claimed success withheld" in section
        assert "R_final" not in section
        assert "0.8" not in section

    def test_indeterminate_release_score_is_rendered(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {
                "release_score": {
                    "status": "indeterminate",
                    "score": None,
                    "indeterminate_dimensions": ["privacy"],
                }
            },
        )
        assert "Status: `indeterminate`" in text
        assert "Indeterminate dimensions: `privacy`" in text

    def test_fairness_highlights_include_log_disparity_error_rows(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        combined = _combined_table()
        extras = {
            "selected_datasets": {"model_a": None, "model_b": None},
            "log_disparity_reports": {
                "model_a": _successful_log_disparity_report(),
                "model_b": {"error": "boom", "error_type": "KeyError"},
            },
        }
        text = build_evaluation_report(cfg, dataset, combined, extras)
        fairness_section = text.split("## Fairness highlights")[1]
        assert "model_a" in fairness_section
        assert "model_b" in fairness_section

    def test_log_disparity_indeterminate_state_is_rendered_without_plot_guidance(
        self, make_config, make_dataset
    ):
        cfg = make_config()
        dataset = make_dataset()
        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {
                "log_disparity_reports": {
                    "good": {
                        **_successful_log_disparity_report(),
                    },
                    "incomplete": {
                        "state": "indeterminate",
                        "reason": "missing release provenance",
                    },
                    "failed": {"state": "failed", "error": "boom"},
                }
            },
        )

        fairness_section = text.split("## Fairness highlights")[1].split("## Plots")[0]
        assert "incomplete" in fairness_section
        assert "indeterminate" in fairness_section
        assert "missing release provenance" in fairness_section
        assert "failed" in fairness_section
        assert "See the per-model interactive sunburst reports" not in fairness_section

    def test_log_disparity_malformed_success_with_sentinel_is_withheld(
        self, make_config, make_dataset
    ):
        text = build_evaluation_report(
            make_config(),
            make_dataset(),
            _combined_table(),
            {
                "log_disparity_reports": {
                    "tampered": {
                        "state": "succeeded",
                        "reason": "sentinel-secret",
                        "summary_stats": {
                            "mean_abs_log_disparity": 999.0,
                            "median_abs_log_disparity": 999.0,
                            "share_significant_bh": 1.0,
                        },
                    }
                }
            },
        )
        section = text.split("## Fairness highlights")[1].split("## Plots")[0]
        assert "incomplete_report" in section
        assert "999" not in section
        assert "sentinel-secret" not in section

    def test_log_disparity_indeterminate_reason_is_preserved_safely(
        self, make_config, make_dataset
    ):
        text = build_evaluation_report(
            make_config(),
            make_dataset(),
            _combined_table(),
            {
                "log_disparity_reports": {
                    "model": {
                        "state": "indeterminate",
                        "reason": "missing release provenance",
                    }
                }
            },
        )
        assert "missing release provenance" in text

    def test_log_disparity_absent_or_unknown_state_is_indeterminate(
        self, make_config, make_dataset
    ):
        cfg = make_config()
        dataset = make_dataset()
        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {
                "log_disparity_reports": {
                    "absent": {"summary_stats": {"mean_abs_log_disparity": 0.1}},
                    "unknown": {
                        "state": "mystery",
                        "summary_stats": {"mean_abs_log_disparity": 0.2},
                    },
                }
            },
        )
        fairness_section = text.split("## Fairness highlights")[1].split("## Plots")[0]
        assert "report_state_missing_or_unknown" in fairness_section
        assert "0.1" not in fairness_section
        assert "0.2" not in fairness_section
        assert "See the per-model interactive sunburst reports" not in fairness_section

    def test_final_holdout_unknown_state_withholds_metrics(self, make_config, make_dataset):
        cfg = make_config()
        dataset = make_dataset()
        score = {"status": "succeeded", "score": 0.7, "dimensions": {"utility": {"score": 0.8}}}
        text = build_evaluation_report(
            cfg,
            dataset,
            _combined_table(),
            {"final_holdout_evidence": {"state": "mystery", "release_score": score}},
        )
        section = text.split("## Final-holdout release score audit")[1].split("##")[0]
        assert "Status: `indeterminate`" in section
        assert "R_final" not in section
        assert "0.8" not in section

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
