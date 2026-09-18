"""Deprecated compatibility aliases for historical Task12 imports.

New code must import :mod:`synthdata.evaluation.release_evidence_eval`.
"""

from synthdata.evaluation import custom_eval
from synthdata.evaluation.release import release_privacy_evidence
from synthdata.evaluation.release_evidence_eval import (  # noqa: F401
    CANONICAL_RELEASE_EVIDENCE_KEYS,
    LEGACY_TASK12_PROTOCOL_VERSION,
    RELEASE_EVIDENCE_PROTOCOL_VERSION,
    _release_evidence_record,
    run_release_evidence_evaluation,
    validate_release_evidence_results,
)

TASK12_CUSTOM_KEYS = CANONICAL_RELEASE_EVIDENCE_KEYS
TASK12_PROTOCOL_VERSION = LEGACY_TASK12_PROTOCOL_VERSION
_task12_record = _release_evidence_record


def run_task12_custom_evaluation(*args, **kwargs):
    """Deprecated wrapper preserving monkeypatch points for old callers."""
    from dataclasses import replace

    import synthdata.evaluation.release_evidence_eval as implementation

    implementation.release_privacy_evidence = release_privacy_evidence
    implementation.custom_eval = custom_eval
    implementation._release_evidence_record = _task12_record
    result = run_release_evidence_evaluation(*args, **kwargs)
    legacy_result = {}
    for model, records in result.items():
        legacy_result[model] = [
            replace(
                record,
                error=(
                    record.error.replace(
                        "release_evidence_release_or_representation_error",
                        "task12_release_or_representation_error",
                    )
                    if record.error
                    else None
                ),
            )
            for record in records
        ]
    return legacy_result


validate_task12_custom_results = validate_release_evidence_results
