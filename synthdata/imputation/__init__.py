"""Missing data imputation pipeline.

Public API: :func:`run_imputation`, :func:`build_validation_report`,
:func:`apply_rounding`, :func:`validate_imputed_column`.

Two backends are available via ``imputation.method`` in the config:

- ``"tabimpute"`` (default) -- :mod:`synthdata.imputation.tabimpute_backend`,
  a TabPFN-based imputer with one-hot categorical encoding.
- ``"refidiff"`` -- :mod:`synthdata.imputation.refidiff_backend`, a
  predictive+diffusion hybrid imputer (arXiv:2505.14451) with a more
  memory-efficient binary categorical encoding, better suited to wide
  datasets where tabimpute's one-hot encoding causes out-of-memory errors.
"""

from synthdata.imputation.pipeline import (
    apply_rounding,
    build_validation_report,
    run_imputation,
    validate_imputed_column,
)

__all__ = [
    "apply_rounding",
    "build_validation_report",
    "run_imputation",
    "validate_imputed_column",
]
