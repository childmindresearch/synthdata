"""TabPFGen-based synthetic data generation (two variants) and their HPO objectives.

Both variants operate on the *imputed* train split (unlike TabPFN, TabPFGen's
SGLD sampler needs fully-observed numeric inputs).
"""

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
import torch
from tabpfgen import TabPFGen

from synthdata.data import (
    decode_label_encoded_columns,
    label_encode_non_numeric_columns,
    validate_semantic_context,
)
from synthdata.generation.hpo import (
    StageAScreenContract,
    persist_stage_a_exception,
    persist_stage_a_trial_exception,
    prepare_stage_a_screen,
    screen_stage_a_trial,
)
from synthdata.utils import get_logger

logger = get_logger(__name__)


def validate_tabpfgen_target(
    target_column: str,
    target_is_categorical: bool | None,
) -> None:
    """Reject target schemas unsupported by the classification-only backend."""
    if not isinstance(target_is_categorical, bool):
        raise ValueError(
            f"TabPFGen generation requires an explicit schema decision for target column "
            f"{target_column!r}; target_is_categorical must be a boolean"
        )
    if not target_is_categorical:
        raise ValueError(
            f"TabPFGen generation requires a categorical target, but the variable schema "
            f"declares target column {target_column!r} as continuous. The current standard "
            "and custom TabPFGen variants are classification-only."
        )


def _balanced_generation_size(n_samples: int, target: np.ndarray) -> int:
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"TabPFGen n_samples must be an integer, got {n_samples!r}")
    if n_samples < 1:
        raise ValueError(f"TabPFGen n_samples must be positive, got {n_samples}")
    class_count = len(np.unique(target))
    if class_count < 1:
        raise ValueError("TabPFGen requires at least one target class")
    return int(np.ceil(n_samples / class_count) * class_count)


def _trim_classification_output(
    features: np.ndarray,
    labels: np.ndarray | None,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    features = np.asarray(features)
    if len(features) < n_samples:
        raise RuntimeError(
            "TabPFGen returned fewer rows than requested: "
            f"returned={len(features)}, requested={n_samples}"
        )
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(features):
            raise RuntimeError(
                "TabPFGen returned feature and label arrays with different lengths: "
                f"features={len(features)}, labels={len(labels)}"
            )
        labels = labels[:n_samples]
    return features[:n_samples], labels


def _proportional_class_counts(n_samples: int, proportions: pd.Series) -> dict:
    expected = proportions.to_numpy(dtype=float) * n_samples
    counts = np.floor(expected).astype(int)
    remainder = n_samples - int(counts.sum())
    if remainder < 0 or remainder > len(counts):
        raise RuntimeError(
            "TabPFGen class-count allocation could not produce the requested size: "
            f"requested={n_samples}, allocated={int(counts.sum())}"
        )
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    return {label: int(count) for label, count in zip(proportions.index, counts, strict=True)}


def _record_hpo_generator_metadata(
    trial: optuna.Trial,
    plugin_name: str,
    params: dict,
    n_samples: int,
    random_state: int,
    implementation_fingerprint: str,
) -> None:
    from synthdata.generation.synthcity_backend import (
        build_generator_metadata,
    )

    trial.set_user_attr("generator_plugin_name", plugin_name)
    trial.set_user_attr("generator_privacy_claim_type", "none")
    trial.set_user_attr("generator_implementation_fingerprint", implementation_fingerprint)
    trial.set_user_attr("generator_metadata_state", "pending")
    try:
        metadata = build_generator_metadata(
            plugin_name,
            params,
            n_samples,
            random_state,
            plugin_fqdn=f"synthdata.tabpfgen.{plugin_name}",
            implementation_fingerprint=implementation_fingerprint,
        )
    except (TypeError, ValueError, RuntimeError):
        trial.set_user_attr("generator_metadata_state", "missing")
        raise
    trial.set_user_attr("generator_metadata", metadata)
    trial.set_user_attr("generator_metadata_state", "present")


class TabPFGenSGLDLabels(TabPFGen):
    """TabPFGen variant that assigns labels from post-SGLD nearest-neighbor lookup.

    TabPFGen's built-in ``generate_classification`` re-assigns labels at the end
    using a TabPFN classifier. TabPFN is an in-context learner; when test points
    are close to (but not identical to) training points it produces unstable
    predictions and collapses all generated labels to a single class, regardless
    of SGLD parameters.

    This subclass instead:
      1. Runs SGLD guided by initialization labels (each sample starts near and
         is pulled toward a specific class).
      2. After SGLD converges, re-assigns each synthetic sample the label of its
         nearest neighbor in the scaled training set, reflecting where the
         sample actually drifted to rather than where it started.
    """

    def generate_classification(self, X_train, y_train, n_samples, balance_classes=True):
        x_scaled = self.scaler.fit_transform(X_train)
        x_train = torch.tensor(x_scaled, device=self.device, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, device=self.device)

        generation_size = (
            _balanced_generation_size(n_samples, np.asarray(y_train))
            if balance_classes
            else n_samples
        )
        if balance_classes:
            classes = np.unique(y_train)
            n_per_class = generation_size // len(classes)
            x_parts, y_parts = [], []
            for cls in classes:
                idx = np.where(y_train == cls)[0]
                sample_idx = np.random.choice(idx, size=n_per_class)
                x_parts.append(
                    x_train[sample_idx]
                    + torch.randn(n_per_class, X_train.shape[1], device=self.device) * 0.01
                )
                y_parts.append(torch.full((n_per_class,), cls, device=self.device))
            x_synth = torch.cat(x_parts)
            y_synth = torch.cat(y_parts)
        else:
            x_synth = torch.randn(n_samples, X_train.shape[1], device=self.device) * 0.01
            y_synth = torch.randint(0, len(np.unique(y_train)), (n_samples,), device=self.device)

        for _step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train_t)

        # Re-assign labels based on nearest neighbor in the scaled training set:
        # SGLD may drift samples across class boundaries, so this reflects final
        # positions rather than initialization assignments.
        x_synth_np = x_synth.detach().cpu().numpy()
        sq_dists = np.sum((x_synth_np[:, None, :] - x_scaled[None, :, :]) ** 2, axis=-1)
        nn_indices = np.argmin(sq_dists, axis=1)
        y_synth_drifted = y_train[nn_indices]

        n_relabelled = int((y_synth_drifted != y_synth.cpu().numpy()).sum())
        logger.info(
            "Post-drift NN relabelling: %d/%d samples (%.1f%%) changed class.",
            n_relabelled,
            n_samples,
            n_relabelled / n_samples * 100,
        )

        x_synth_out = self.scaler.inverse_transform(x_synth_np)
        return x_synth_out[:n_samples], y_synth_drifted[:n_samples]


def generate_tabpfgen_standard(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    tabpfgen_params: dict | None = None,
    relabel_with_classifier: bool = False,
    *,
    target_is_categorical: bool | None = None,
    variable_schema_fingerprint: str | None = None,
    semantic_context: dict | None = None,
) -> pd.DataFrame:
    """Default TabPFGen classification synthesis.

    ``relabel_with_classifier=True`` discards TabPFGen's own predicted labels
    and instead assigns labels via a freshly-fit ``TabPFNClassifier`` (used for
    the HPO-tuned variant in the notebook, since it was found empirically more
    stable across sampled hyperparameters).
    """
    validate_tabpfgen_target(target_column, target_is_categorical)
    validate_semantic_context(
        semantic_context,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        target_is_categorical=target_is_categorical,
        variable_schema_fingerprint=variable_schema_fingerprint,
        frame_columns=train_imputed_df.columns,
    )
    # TabPFGen's SGLD sampler requires a purely numeric input array. The
    # imputed train split can still have string-valued declared-categorical
    # columns (e.g. Pegboard__peg_dom_hand's "Left"/"Right"/"Ambidexterous"),
    # since imputation decodes categorical columns back to their original
    # labels. Mirror generate_tabpfn_standard: force-factorize every declared
    # categorical_columns entry (even if already numeric) to integer codes,
    # then decode the synthesized categorical columns back afterward via the
    # same category_maps -- see synthdata.data.label_encode_non_numeric_columns.
    encoded_features, category_maps = label_encode_non_numeric_columns(
        train_imputed_df, feature_columns, categorical_columns=categorical_columns
    )
    x_train = encoded_features.to_numpy(dtype=float)
    y_train = train_imputed_df[target_column].values

    generator = TabPFGen(**(tabpfgen_params or {}))
    generation_size = _balanced_generation_size(n_samples, y_train)
    x_synth, y_synth = generator.generate_classification(
        X_train=x_train, y_train=y_train, n_samples=generation_size, balance_classes=True
    )
    x_synth, y_synth = _trim_classification_output(x_synth, y_synth, n_samples)

    synthetic_encoded = pd.DataFrame(x_synth, columns=feature_columns)

    if relabel_with_classifier:
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier()
        clf.fit(x_train, y_train)
        target_values = clf.predict(synthetic_encoded.to_numpy(dtype=float))
    else:
        n_cls = train_imputed_df[target_column].nunique()
        target_values = pd.Series(y_synth.astype(int)).clip(0, n_cls - 1).values

    synthetic = decode_label_encoded_columns(synthetic_encoded, category_maps)
    synthetic[target_column] = target_values
    return synthetic


def generate_tabpfgen_custom(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    seed: int = 42,
    sgld_params: dict | None = None,
    *,
    target_is_categorical: bool | None = None,
    variable_schema_fingerprint: str | None = None,
    semantic_context: dict | None = None,
) -> pd.DataFrame:
    """TabPFGenSGLDLabels synthesis with oversample-then-subsample class balancing.

    ``balance_classes=True`` in TabPFGen always generates equal counts per class,
    so to recover the true training class distribution we over-generate and then
    subsample proportionally.
    """
    validate_tabpfgen_target(target_column, target_is_categorical)
    validate_semantic_context(
        semantic_context,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        target_is_categorical=target_is_categorical,
        variable_schema_fingerprint=variable_schema_fingerprint,
        frame_columns=train_imputed_df.columns,
    )
    # See the comment in generate_tabpfgen_standard for why declared
    # categorical_columns must be force-factorized before the float cast.
    encoded_features, category_maps = label_encode_non_numeric_columns(
        train_imputed_df, feature_columns, categorical_columns=categorical_columns
    )
    x_train = encoded_features.to_numpy(dtype=float)
    y_train = train_imputed_df[target_column].values

    train_proportions = train_imputed_df[target_column].value_counts(normalize=True)
    n_classes = len(train_proportions)
    n_per_class_needed = int(np.ceil(n_samples * train_proportions.max()))
    n_to_generate = n_per_class_needed * n_classes

    generator = TabPFGenSGLDLabels(
        **(sgld_params or {"n_sgld_steps": 1000, "sgld_noise_scale": 0.1})
    )
    x_synth_all, y_synth_all = generator.generate_classification(
        x_train, y_train, n_samples=n_to_generate, balance_classes=True
    )
    x_synth_all, y_synth_all = _trim_classification_output(
        x_synth_all,
        y_synth_all,
        n_to_generate,
    )

    synth_all_encoded = pd.DataFrame(x_synth_all, columns=feature_columns)
    synth_all = decode_label_encoded_columns(synth_all_encoded, category_maps)

    n_classes_enc = train_imputed_df[target_column].nunique()
    synth_all[target_column] = pd.Series(y_synth_all.astype(int)).clip(0, n_classes_enc - 1).values

    parts = []
    class_counts = _proportional_class_counts(n_samples, train_proportions)
    for orig_cls in train_proportions.index:
        n_needed = class_counts[orig_cls]
        cls_rows = synth_all[synth_all[target_column] == orig_cls]
        parts.append(
            cls_rows.sample(n=n_needed, replace=len(cls_rows) < n_needed, random_state=seed)
        )
    synthetic = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(synthetic) != n_samples:
        raise RuntimeError(
            f"TabPFGen custom generation returned {len(synthetic)} rows; expected {n_samples}"
        )
    return synthetic


def build_tabpfgen_standard_objective(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    sgld_step_cap: int,
    eval_fn: Callable[[pd.DataFrame], float],
    *,
    seed: int = 42,
    stage_a_contract: StageAScreenContract | None = None,
    stage_a_source_df: pd.DataFrame | None = None,
    stage_a_root: str | None = None,
    study_name: str | None = None,
    target_is_categorical: bool | None = None,
    variable_schema_fingerprint: str | None = None,
    semantic_context: dict | None = None,
):
    """Optuna objective searching TabPFGen's SGLD hyperparameters (standard variant)."""
    validate_tabpfgen_target(target_column, target_is_categorical)
    validate_semantic_context(
        semantic_context,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        target_is_categorical=target_is_categorical,
        variable_schema_fingerprint=variable_schema_fingerprint,
        frame_columns=train_imputed_df.columns,
    )
    from tabpfn import TabPFNClassifier

    prepare_stage_a_screen(stage_a_contract, stage_a_source_df, stage_a_root, study_name)

    # See the comment in generate_tabpfgen_standard for why declared
    # categorical_columns must be force-factorized before the float cast.
    try:
        encoded_features, category_maps = label_encode_non_numeric_columns(
            train_imputed_df, feature_columns, categorical_columns=categorical_columns
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        if stage_a_contract is not None:
            persist_stage_a_exception(stage_a_root, study_name, stage_a_contract, exc)
        raise
    x_feat = encoded_features.to_numpy(dtype=float)
    y_label = train_imputed_df[target_column].values
    generation_size = _balanced_generation_size(n_samples, y_label)
    plugin_name = "tabpfgen_standard"
    from synthdata.generation.synthcity_backend import generator_implementation_fingerprint

    implementation_fingerprint = generator_implementation_fingerprint(plugin_name)

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("generator_plugin_name", plugin_name)
        trial.set_user_attr("generator_privacy_claim_type", "none")
        trial.set_user_attr("generator_implementation_fingerprint", implementation_fingerprint)
        trial.set_user_attr("generator_metadata_state", "pending")
        n_steps = min(trial.suggest_int("n_sgld_steps", 100, 2000, step=100), sgld_step_cap)
        step_size = trial.suggest_float("sgld_step_size", 0.001, 0.1, log=True)
        noise_scale = trial.suggest_float("sgld_noise_scale", 0.001, 0.5, log=True)
        params = {
            "n_sgld_steps": n_steps,
            "sgld_step_size": step_size,
            "sgld_noise_scale": noise_scale,
        }
        try:
            gen = TabPFGen(**params)
            x_s, _ = gen.generate_classification(
                X_train=x_feat,
                y_train=y_label,
                n_samples=generation_size,
                balance_classes=True,
            )
            x_s, _ = _trim_classification_output(x_s, None, n_samples)
            syn_encoded = pd.DataFrame(x_s, columns=feature_columns)
            clf = TabPFNClassifier()
            clf.fit(x_feat, y_label)
            target_values = clf.predict(syn_encoded.to_numpy(dtype=float))
            syn = decode_label_encoded_columns(syn_encoded, category_maps)
            syn[target_column] = target_values
            _record_hpo_generator_metadata(
                trial,
                plugin_name,
                params,
                n_samples,
                seed,
                implementation_fingerprint,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            trial.set_user_attr("generator_metadata_state", "missing")
            if stage_a_contract is not None:
                persist_stage_a_trial_exception(
                    trial,
                    stage_a_root,
                    study_name,
                    stage_a_contract,
                    exc,
                )
            logger.warning("tabpfgen_standard trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc
        if stage_a_contract is not None:
            screen_stage_a_trial(
                trial,
                syn,
                stage_a_contract,
                stage_a_source_df,
                stage_a_root,
                study_name,
            )
        return eval_fn(syn)

    return objective


def build_tabpfgen_custom_objective(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    sgld_step_cap: int,
    eval_fn: Callable[[pd.DataFrame], float],
    seed: int = 42,
    *,
    stage_a_contract: StageAScreenContract | None = None,
    stage_a_source_df: pd.DataFrame | None = None,
    stage_a_root: str | None = None,
    study_name: str | None = None,
    target_is_categorical: bool | None = None,
    variable_schema_fingerprint: str | None = None,
    semantic_context: dict | None = None,
):
    """Optuna objective searching TabPFGenSGLDLabels's SGLD hyperparameters."""
    validate_tabpfgen_target(target_column, target_is_categorical)
    validate_semantic_context(
        semantic_context,
        target_column=target_column,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        target_is_categorical=target_is_categorical,
        variable_schema_fingerprint=variable_schema_fingerprint,
        frame_columns=train_imputed_df.columns,
    )
    prepare_stage_a_screen(stage_a_contract, stage_a_source_df, stage_a_root, study_name)
    # See the comment in generate_tabpfgen_standard for why declared
    # categorical_columns must be force-factorized before the float cast.
    try:
        encoded_features, category_maps = label_encode_non_numeric_columns(
            train_imputed_df, feature_columns, categorical_columns=categorical_columns
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        if stage_a_contract is not None:
            persist_stage_a_exception(stage_a_root, study_name, stage_a_contract, exc)
        raise
    x_feat = encoded_features.to_numpy(dtype=float)
    y_label = train_imputed_df[target_column].values
    proportions = train_imputed_df[target_column].value_counts(normalize=True)
    plugin_name = "tabpfgen_custom"
    from synthdata.generation.synthcity_backend import generator_implementation_fingerprint

    implementation_fingerprint = generator_implementation_fingerprint(plugin_name)

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("generator_plugin_name", plugin_name)
        trial.set_user_attr("generator_privacy_claim_type", "none")
        trial.set_user_attr("generator_implementation_fingerprint", implementation_fingerprint)
        trial.set_user_attr("generator_metadata_state", "pending")
        n_steps = min(trial.suggest_int("n_sgld_steps", 100, 2000, step=100), sgld_step_cap)
        step_size = trial.suggest_float("sgld_step_size", 0.001, 0.1, log=True)
        noise_scale = trial.suggest_float("sgld_noise_scale", 0.001, 0.5, log=True)
        params = {
            "n_sgld_steps": n_steps,
            "sgld_step_size": step_size,
            "sgld_noise_scale": noise_scale,
        }
        try:
            n_per = int(np.ceil(n_samples * proportions.max()))
            gen = TabPFGenSGLDLabels(**params)
            x_s, y_s = gen.generate_classification(
                x_feat,
                y_label,
                n_samples=n_per * len(proportions),
                balance_classes=True,
            )
            all_df_encoded = pd.DataFrame(x_s, columns=feature_columns)
            all_df = decode_label_encoded_columns(all_df_encoded, category_maps)
            all_df[target_column] = (
                pd.Series(y_s.astype(int))
                .clip(0, train_imputed_df[target_column].nunique() - 1)
                .values
            )
            parts = []
            class_counts = _proportional_class_counts(n_samples, proportions)
            for cls in proportions.index:
                n_need = class_counts[cls]
                rows = all_df[all_df[target_column] == cls]
                parts.append(rows.sample(n=n_need, replace=len(rows) < n_need, random_state=seed))
            syn = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
            if len(syn) != n_samples:
                raise RuntimeError(
                    f"TabPFGen custom HPO generation returned {len(syn)} rows; expected {n_samples}"
                )
            _record_hpo_generator_metadata(
                trial,
                plugin_name,
                params,
                n_samples,
                seed,
                implementation_fingerprint,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            trial.set_user_attr("generator_metadata_state", "missing")
            if stage_a_contract is not None:
                persist_stage_a_trial_exception(
                    trial,
                    stage_a_root,
                    study_name,
                    stage_a_contract,
                    exc,
                )
            logger.warning("tabpfgen_custom trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc
        if stage_a_contract is not None:
            screen_stage_a_trial(
                trial,
                syn,
                stage_a_contract,
                stage_a_source_df,
                stage_a_root,
                study_name,
            )
        return eval_fn(syn)

    return objective
