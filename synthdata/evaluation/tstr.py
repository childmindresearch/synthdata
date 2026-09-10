"""Fixed-seed train-on-synthetic, test-on-real and holdout fairness metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

from .release import _validate_frame_provenance

PROTOCOL_VERSION = "tstr-v1"


def _require_release(frame: pd.DataFrame) -> None:
    try:
        provenance = _validate_frame_provenance(frame)
    except ValueError as error:
        raise ValueError("synthetic input must be a verified release-form frame") from error
    if (
        provenance.get("release_form") is not True
        or provenance.get("role") != "release"
        or provenance.get("source_role") != "synthetic"
    ):
        raise ValueError("synthetic input must have release/synthetic provenance")


def _features(
    train: pd.DataFrame, test: pd.DataFrame, excluded: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [c for c in train.columns if c not in excluded]
    train_encoded = pd.get_dummies(train[columns], dummy_na=True)
    test_columns = [c for c in columns if c in test.columns]
    test_encoded = pd.get_dummies(test[test_columns], dummy_na=True)
    # Reindexing test to train's columns prevents evaluation-only categories from
    # changing the representation learned by the release model.
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded.reset_index(drop=True), test_encoded.reset_index(drop=True)


def _prediction_identity(predictions: np.ndarray) -> str:
    """Return stable identity for model prediction values."""
    values = np.asarray(predictions).reshape(-1).tolist()
    return sha256(repr(values).encode("utf-8")).hexdigest()


def _artifact_digest(artifact: Mapping[str, object]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "artifact_digest"}
    return sha256(
        repr(sorted(payload.items(), key=lambda item: item[0])).encode("utf-8")
    ).hexdigest()


def _population_identity(
    actual: np.ndarray, protected: pd.DataFrame, columns: Sequence[str]
) -> str:
    """Return identity for target and protected rows used by fairness scoring."""
    values = (
        protected[list(columns)].astype(object).where(protected[list(columns)].notna(), "<missing>")
    )
    payload = {
        "target": actual.tolist(),
        "protected_columns": list(columns),
        "protected": values.to_dict("records"),
    }
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def _xgb(seed: int, classes: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="multi:softprob",
        num_class=classes,
        random_state=seed,
        n_jobs=1,
        eval_metric="mlogloss",
        tree_method="hist",
    )


@dataclass(frozen=True)
class TSTRResult:
    """Structured TSTR result, also convertible to its public report mapping."""

    report: dict
    model: object | None = None
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    envelope: dict | None = None

    def as_dict(self) -> dict:
        return self.report


def run_tstr_evaluation(
    synthetic_release: pd.DataFrame,
    real_data: pd.DataFrame,
    *,
    target_column: str,
    evaluation_role: str = "tuning",
    seed: int = 17,
    protected_columns: Sequence[str] = (),
    role_hashes: Mapping[str, str] | None = None,
) -> TSTRResult:
    """Fit only release synthetic rows and evaluate one model on real rows.

    ``evaluation_role`` is ``tuning`` or ``final_holdout``. No validation split
    is requested from XGBoost, so every synthetic release row is used for fit.
    """
    if evaluation_role not in {"tuning", "final_holdout"}:
        raise ValueError("evaluation_role must be tuning or final_holdout")
    _require_release(synthetic_release)
    real_provenance = _validate_frame_provenance(real_data)
    expected_source = "tuning" if evaluation_role == "tuning" else "final_holdout"
    if (
        not isinstance(real_provenance, Mapping)
        or real_provenance.get("source_role") != expected_source
    ):
        raise ValueError(f"real data must have {expected_source} provenance")
    synthetic_provenance = synthetic_release.attrs["release_provenance"]
    if synthetic_provenance.get("common_protocol_digest") != real_provenance.get(
        "common_protocol_digest"
    ):
        raise ValueError("synthetic and real data must share release provenance")
    if target_column not in synthetic_release or target_column not in real_data:
        raise ValueError("target column is missing")
    classes = sorted(pd.unique(real_data[target_column]).tolist(), key=lambda x: str(x))
    synthetic_classes = set(pd.unique(synthetic_release[target_column]))
    missing = [value for value in classes if value not in synthetic_classes]
    extra = sorted((value for value in synthetic_classes if value not in set(classes)), key=str)
    metadata = {
        "producer": "task10_tstr",
        "protocol_version": PROTOCOL_VERSION,
        "producer_protocol_version": PROTOCOL_VERSION,
        "evaluation_role": evaluation_role,
        "fit_roles": ["train"] if evaluation_role == "tuning" else ["train", "tuning"],
        "source_role": "synthetic",
        "release_form": True,
        "seed": seed,
        "common_protocol_digest": synthetic_provenance["common_protocol_digest"],
        "target_column": target_column,
        "protected_columns": list(protected_columns),
        "release_transform_digest": synthetic_provenance.get("release_transform_digest"),
        "role_hashes": {
            "synthetic": synthetic_provenance.get("role_hash"),
            expected_source: real_provenance.get("role_hash"),
        },
        "target_identity": sha256(repr(real_data[target_column].tolist()).encode()).hexdigest(),
        "protected_identity": sha256(
            repr(real_data[list(protected_columns)].to_dict("records")).encode()
        ).hexdigest(),
    }
    if role_hashes is not None:
        metadata["role_hashes"] = dict(role_hashes)
    metadata["target_population_identity"] = metadata["target_identity"]
    metadata["protected_population_identity"] = metadata["protected_identity"]
    supports = {str(c): int((real_data[target_column] == c).sum()) for c in classes}
    if (
        missing
        or extra
        or synthetic_release[target_column].isna().any()
        or real_data[target_column].isna().any()
    ):
        reason = (
            "missing_synthetic_target_classes"
            if missing
            else "synthetic_target_domain_mismatch"
            if extra
            else "invalid_target_values"
        )
        report = {
            "state": "indeterminate",
            "reason": reason,
            "missing_classes": missing,
            "unexpected_synthetic_classes": extra,
            "class_supports": supports,
            "primary_metric": "macro_f1",
            "result_metadata": metadata,
        }
        return TSTRResult(
            report,
            envelope={
                "producer": "task10_tstr",
                "protocol_version": PROTOCOL_VERSION,
                "result_metadata": metadata,
                "report": report,
            },
        )
    x_train, x_test = _features(synthetic_release, real_data, {target_column, *protected_columns})
    model = _xgb(seed, len(classes))
    # Label encoding is deterministic and independent of XGBoost's category handling.
    labels = {value: i for i, value in enumerate(classes)}
    y_train = synthetic_release[target_column].map(labels).to_numpy()
    y_test = real_data[target_column].map(labels).to_numpy()
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_test))
    raw_predictions = np.asarray(model.predict(x_test))
    predictions = (
        np.argmax(raw_predictions, axis=1)
        if raw_predictions.ndim > 1
        else raw_predictions.astype(int).reshape(-1)
    )
    y_onehot = np.column_stack([(y_test == i).astype(int) for i in range(len(classes))])
    auprc = float(average_precision_score(y_onehot, probabilities, average="macro"))
    per_class = {
        str(value): {
            "support": supports[str(value)],
            "f1": float(f1_score(y_test == i, predictions == i, zero_division=0)),
        }
        for i, value in enumerate(classes)
    }
    report = {
        "state": "complete",
        "primary_metric": "macro_f1",
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_ovr_auprc": auprc,
        "class_supports": supports,
        "per_class": per_class,
        "target_order": classes,
        "result_metadata": metadata,
    }
    if protected_columns and evaluation_role == "final_holdout":
        fairness_predictions = np.asarray(classes, dtype=object)[predictions]
        prediction_artifact = {
            "producer": "task10_tstr",
            "protocol_version": PROTOCOL_VERSION,
            "producer_protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "verified": True,
            "source_role": "final_holdout",
            "prediction_source": "tstr_model",
            "prediction_length": len(predictions),
            "prediction_identity": _prediction_identity(fairness_predictions),
            "population_role": real_provenance["source_role"],
            "population_digest": real_provenance["digest"],
            "common_protocol_digest": real_provenance["common_protocol_digest"],
            "target_column": target_column,
            "protected_columns": list(protected_columns),
            "population_length": len(real_data),
            "population_identity": _population_identity(
                real_data[target_column].to_numpy(), real_data, protected_columns
            ),
        }
        prediction_artifact["release_transform_digest"] = synthetic_provenance.get(
            "release_transform_digest"
        )
        prediction_artifact["role_hashes"] = metadata["role_hashes"]
        prediction_artifact["target_identity"] = metadata["target_identity"]
        prediction_artifact["protected_identity"] = metadata["protected_identity"]
        prediction_artifact["artifact_digest"] = _artifact_digest(prediction_artifact)
        fairness = compute_equalized_odds(
            real_data[target_column],
            fairness_predictions,
            real_data[list(protected_columns)],
            target_classes=classes,
            protected_columns=protected_columns,
            prediction_artifact=prediction_artifact,
        )
        report["equalized_odds"] = fairness
        report["prediction_artifact"] = prediction_artifact
    envelope = {
        "producer": "task10_tstr",
        "protocol_version": PROTOCOL_VERSION,
        "result_metadata": metadata,
        "report": report,
        "prediction_artifact": report.get("prediction_artifact"),
    }
    return TSTRResult(report, model, predictions, probabilities, envelope)


def run_tstr(*args, **kwargs) -> TSTRResult:
    """Compatibility alias for :func:`run_tstr_evaluation`."""
    return run_tstr_evaluation(*args, **kwargs)


def compute_equalized_odds(
    y_true: Iterable,
    y_pred: Iterable,
    protected: pd.DataFrame,
    *,
    target_classes: Sequence | None = None,
    protected_columns: Sequence[str] | None = None,
    min_support: int = 1,
    provenance: Mapping[str, object] | None = None,
    prediction_artifact: Mapping[str, object] | None = None,
) -> dict:
    """Compute nested OVR equalized-odds gaps from verified final predictions."""
    artifact = prediction_artifact if prediction_artifact is not None else provenance
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("source_role") != "final_holdout"
        or artifact.get("prediction_source") != "tstr_model"
        or artifact.get("verified") is not True
    ):
        raise ValueError("equalized-odds predictions require verified final_holdout model output")
    if "artifact_digest" in artifact and artifact.get("artifact_digest") != _artifact_digest(
        artifact
    ):
        raise ValueError("prediction artifact immutable content binding is invalid")
    actual = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    if len(actual) != len(predicted) or len(actual) != len(protected):
        raise ValueError("prediction and protected-column lengths must match")
    if artifact.get("prediction_length") != len(predicted) or artifact.get(
        "prediction_identity"
    ) != _prediction_identity(predicted):
        raise ValueError("prediction artifact is not bound to supplied predictions")
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    classes = (
        list(target_classes)
        if target_classes is not None
        else sorted(pd.unique(actual).tolist(), key=str)
    )
    columns = list(protected_columns or protected.columns)
    missing_columns = [column for column in columns if column not in protected.columns]
    if missing_columns:
        raise ValueError(f"protected columns are missing: {missing_columns}")
    if (
        artifact.get("population_role") != "final_holdout"
        or artifact.get("population_length") != len(actual)
        or artifact.get("target_column") is None
        or artifact.get("protected_columns") != columns
        or artifact.get("population_identity") != _population_identity(actual, protected, columns)
    ):
        raise ValueError("equalized-odds inputs are not bound to verified final_holdout population")
    slices = []
    target_scores: dict[object, list[float]] = {target: [] for target in classes}
    for column in columns:
        for target in classes:
            values = (
                protected[column]
                .astype(object)
                .where(protected[column].notna(), "<missing>")
                .to_numpy()
            )
            groups = sorted(pd.unique(values).tolist(), key=str)
            rates = []
            reasons = []
            for group in groups:
                mask = values == group
                pos = actual == target
                neg = ~pos
                total_support = int(mask.sum())
                positive_support = int((mask & pos).sum())
                negative_support = int((mask & neg).sum())
                if (
                    total_support < min_support
                    or positive_support < min_support
                    or negative_support < min_support
                ):
                    reasons.append(
                        {
                            "group": group,
                            "reason": "insufficient_positive_or_negative_support",
                            "support": total_support,
                            "positive_support": positive_support,
                            "negative_support": negative_support,
                        }
                    )
                    continue
                true_positive = int((mask & pos & (predicted == target)).sum())
                false_positive = int((mask & neg & (predicted == target)).sum())
                rates.append(
                    {
                        "group": group,
                        "support": total_support,
                        "positive_support": positive_support,
                        "negative_support": negative_support,
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "tpr": true_positive / positive_support,
                        "fpr": false_positive / negative_support,
                    }
                )
            if len(rates) < 2:
                item = {
                    "protected_column": column,
                    "target_class": target,
                    "state": "invalid",
                    "reason": "fewer_than_two_valid_groups",
                    "invalid_groups": reasons,
                    "valid_group_count": len(rates),
                }
            else:
                tpr_values = np.asarray([x["tpr"] for x in rates], dtype=float)
                fpr_values = np.asarray([x["fpr"] for x in rates], dtype=float)
                tpr_gap = max(tpr_values) - min(tpr_values)
                fpr_gap = max(fpr_values) - min(fpr_values)
                gap = (tpr_gap + fpr_gap) / 2
                item = {
                    "protected_column": column,
                    "target_class": target,
                    "state": "valid",
                    "gap": gap,
                    "tpr_gap": tpr_gap,
                    "fpr_gap": fpr_gap,
                    "valid_group_count": len(rates),
                    "group_rates": rates,
                }
                target_scores[target].append(float(gap))
            slices.append(item)
    valid_class_scores = [float(np.mean(scores)) for scores in target_scores.values() if scores]
    valid = valid_class_scores
    return {
        "state": "complete" if valid else "indeterminate",
        "metric": "equalized_odds",
        "macro_valid_slice_score": float(np.mean(valid)) if valid else None,
        "worst_valid_gap": float(max(valid)) if valid else None,
        "slices": slices,
        "aggregation": "macro valid protected slices within target class, then macro valid target classes",
    }


equalized_odds = compute_equalized_odds
