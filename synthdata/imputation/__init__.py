"""Missing data imputation pipeline.

Public API: :func:`run_imputation`, :func:`build_validation_report`,
:func:`apply_rounding`, :func:`validate_imputed_column`.

Canonical imputation uses fixed HyperImpute plugins. Legacy backends remain
available only for explicit two-role compatibility:

- ``"hyperimpute"`` -- fixed median/mean and most-frequent plugins.
- ``"tabimpute"`` and ``"refidiff"`` -- deferred canonical methods.
"""

from synthdata.imputation.benchmark import run_refidiff_benchmark
from synthdata.imputation.hyperimpute_backend import HyperImputeState
from synthdata.imputation.pipeline import (
    apply_rounding,
    build_validation_report,
    run_imputation,
    validate_imputed_column,
)

__all__ = [
    "apply_rounding",
    "HyperImputeState",
    "build_validation_report",
    "run_imputation",
    "run_refidiff_benchmark",
    "validate_imputed_column",
]
