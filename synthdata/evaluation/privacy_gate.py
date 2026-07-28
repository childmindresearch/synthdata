"""Absolute (not merely relative-to-other-models) privacy safety floor.

The combined evaluation table's ranked/scaled columns (see
:mod:`synthdata.evaluation.combine`) only tell you which model looks *better
or worse than the other candidates in this run* -- a model can rank "best on
privacy" purely by comparison while still leaking an unacceptable absolute
amount, which is exactly the wrong thing to optimize for when working with
sensitive healthcare data. This module checks each model's RAW metric value
(from the combined table, before any min-max scaling) against a fixed
threshold configured in ``evaluation.privacy_gate`` (see
:class:`synthdata.config.PrivacyGateConfig`), producing a per-model
pass/fail verdict that is surfaced (never silently hidden, and never used to
silently drop a model from the ranked table).
"""

import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)

_PASS_COL = "pass"
_VIOLATIONS_COL = "violations"


def _find_metric_column(columns: pd.Index, metric_name: str) -> tuple | None:
    """Find the combined table's 3-level ``(framework, type, metric)`` column
    matching a bare metric name (e.g. ``"mia_recall"`` or
    ``"privacy.identifiability_score.score_OC"``), ignoring framework/type.
    Returns None if not present (metric not selected/computed this run).
    """
    matches = [c for c in columns if len(c) == 3 and c[2] == metric_name]
    if not matches:
        return None
    return matches[0]


def evaluate_privacy_gate(combined: pd.DataFrame, privacy_gate_cfg) -> pd.DataFrame | None:
    """Check every configured threshold against ``combined``'s raw metric
    values. Returns a ``models x {"pass", "violations"}`` DataFrame, or None
    if the gate is disabled or has no thresholds configured (nothing to
    merge into the combined table in that case).

    A metric listed in ``privacy_gate_cfg.thresholds`` that isn't present in
    ``combined`` (not selected this run, or its framework failed) is logged
    as a WARNING and excluded from the gate check for every model -- it is
    NEVER treated as a silent pass. Likewise, a NaN raw value for a specific
    model (e.g. that model's evaluation failed) fails that model's check for
    that metric rather than being silently skipped.
    """
    if not privacy_gate_cfg.enabled:
        logger.info("[privacy_gate] disabled via evaluation.privacy_gate.enabled; skipping")
        return None
    thresholds = privacy_gate_cfg.thresholds
    if not thresholds:
        logger.info("[privacy_gate] no thresholds configured; skipping")
        return None

    pass_mask = pd.Series(True, index=combined.index)
    violation_lists: dict = {model: [] for model in combined.index}

    checked_any = False
    for metric_name, spec in thresholds.items():
        column = _find_metric_column(combined.columns, metric_name)
        if column is None:
            logger.warning(
                "[privacy_gate] metric %r not found in the combined evaluation table (not "
                "selected/computed this run) -- excluded from the gate check, NOT treated as "
                "a pass for any model",
                metric_name,
            )
            continue
        checked_any = True

        bound = spec["bound"]
        limit = spec["value"]
        values = combined[column]
        metric_pass = values <= limit if bound == "max" else values >= limit
        # A NaN raw value (that model's evaluation for this metric failed/was
        # missing) can't be verified -- fail conservatively rather than
        # silently passing (mirrors this repo's "fail loudly" principle).
        metric_pass = metric_pass.where(values.notna(), other=False)

        for model in combined.index:
            if not metric_pass.loc[model]:
                val = values.loc[model]
                detail = f"{val:.4g}" if pd.notna(val) else "NaN (could not evaluate)"
                violation_lists[model].append(f"{metric_name}={detail} ({bound} limit {limit})")

        pass_mask &= metric_pass

    if not checked_any:
        logger.warning(
            "[privacy_gate] none of the configured threshold metrics were found in the "
            "combined table; the gate could not check anything this run"
        )
        return None

    result = pd.DataFrame(
        {
            _PASS_COL: pass_mask,
            _VIOLATIONS_COL: pd.Series(
                {m: "; ".join(v) for m, v in violation_lists.items()}, index=combined.index
            ),
        },
        index=combined.index,
    )

    for model in result.index[~result[_PASS_COL]]:
        logger.warning(
            "[privacy_gate] model %r FAILED the privacy gate: %s",
            model,
            result.loc[model, _VIOLATIONS_COL],
        )

    return result


def merge_privacy_gate_results(
    combined: pd.DataFrame, gate_result: pd.DataFrame | None
) -> pd.DataFrame:
    """Merge ``evaluate_privacy_gate``'s output into ``combined`` as
    ``("__all__", "privacy_gate", "pass")`` / ``("__all__", "privacy_gate",
    "violations")`` columns. A no-op (returns ``combined`` unchanged) if the
    gate was disabled/skipped (``gate_result is None``).
    """
    if gate_result is None:
        return combined
    combined = combined.copy()
    combined[("__all__", "privacy_gate", _PASS_COL)] = gate_result[_PASS_COL]
    combined[("__all__", "privacy_gate", _VIOLATIONS_COL)] = gate_result[_VIOLATIONS_COL]
    return combined
