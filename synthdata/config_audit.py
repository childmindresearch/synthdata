"""Config, semantic-policy, and dataset/schema audit helpers."""

from dataclasses import dataclass
from pathlib import Path

from synthdata.config import load_config
from synthdata.data import load_dataset
from synthdata.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConfigAuditResult:
    """Summary of a successful config and dataset audit."""

    config_path: Path
    dataset_name: str
    dataset_version: str | None
    row_count: int
    feature_columns: tuple[str, ...]
    target_column: str
    schema_columns: tuple[str, ...]


def audit_config(path: str | Path) -> ConfigAuditResult:
    """Validate config and policies, then load configured data and schema.

    Dataset loading uses the normal project loader, including its established
    cleanup, declaration, schema, and split checks. This function does not run
    imputation, generation, evaluation, plotting, or model inference. An uncached
    UCI source can still trigger a dataset download.
    """
    config_path = Path(path).expanduser().resolve()
    logger.info("[config-audit] validating config=%s", config_path)
    cfg = load_config(config_path)

    logger.info(
        "[config-audit] checking configured dataset and schema dataset=%s version=%s",
        cfg.name,
        cfg.data.version or "unversioned",
    )
    dataset = load_dataset(cfg)
    result = ConfigAuditResult(
        config_path=config_path,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        row_count=len(dataset.full_df),
        feature_columns=tuple(dataset.feature_columns),
        target_column=dataset.target_column,
        schema_columns=tuple(dataset.variable_schema),
    )
    logger.info(
        "[config-audit] passed config=%s dataset=%s version=%s rows=%d features=%d schema_columns=%d",
        result.config_path,
        result.dataset_name,
        result.dataset_version or "unversioned",
        result.row_count,
        len(result.feature_columns),
        len(result.schema_columns),
    )
    return result
