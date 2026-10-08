"""TabPFN-based synthetic data generation (two variants).

Both variants use TabPFN's unsupervised-experiment API (``tabpfn_extensions``).
TabPFN's foundation model handles missing values natively, so the caller may
pass either the original (pre-imputation) train split or the imputed one --
see ``generation.tabpfn.data_variants`` in the pipeline config, which drives
:func:`synthdata.generation.pipeline.run_generation`.
"""

import numpy as np
import pandas as pd
import torch

from synthdata.data import decode_label_encoded_columns, label_encode_non_numeric_columns
from synthdata.utils import get_logger

logger = get_logger(__name__)


def validate_tabpfn_target(target_column: str, target_is_categorical: bool) -> None:
    """Reject target schemas unsupported by the current TabPFN generators."""
    if not target_is_categorical:
        raise ValueError(
            f"TabPFN generation requires a categorical target, but the variable schema "
            f"declares target column {target_column!r} as continuous. The current standard "
            "and custom TabPFN variants are classification-only."
        )


def sample_labels(proba: np.ndarray, classes: np.ndarray, seed: int) -> np.ndarray:
    """Draw one label per row from its predicted class probabilities."""
    u = np.random.default_rng(seed).random((len(proba), 1))
    idx = (u > np.cumsum(proba, axis=1)).sum(axis=1)
    return np.asarray(classes)[np.minimum(idx, len(classes) - 1)]


def _patch_explicit_categorical_feature_inference():
    """Make TabPFN extensions honor the caller's categorical column list.

    ``TabPFNUnsupervisedModel.fit`` calls ``infer_categorical_features`` again,
    and that helper adds low-cardinality numeric columns to the list even when
    the caller supplied an explicit list. This is incorrect for this pipeline:
    the variable schema is the source of truth, so a continuous column remains
    continuous regardless of its observed cardinality.

    The extension imports this helper directly into
    ``tabpfn_extensions.unsupervised.unsupervised``. Patching only the public
    ``tabpfn_extensions.unsupervised`` package leaves that bound reference
    unchanged, so the cardinality heuristic continues to run. Replace the
    bound helper in the nested module instead.

    The explicit indices are still supplied through ``set_categorical_features``
    by the extension's experiment API, and the separate TabPFN class-count
    limit is retained by :func:`_patch_use_classifier_nan_bug` because it is a
    model capability constraint, not a type inference decision.

    Tracking: this works around the direct import of
    ``infer_categorical_features`` in the pinned ``tabpfn-extensions``
    git-main dependency. Re-check the upstream module and call binding before
    removing it if the dependency starts documenting an opt-out for cardinality
    inference.
    """
    from tabpfn_extensions.unsupervised import unsupervised as unsupervised_module

    if getattr(unsupervised_module, "_synthdata_explicit_types_patched", False):
        return

    def infer_categorical_features(
        X: np.ndarray, categorical_features: list[int] | None = None
    ) -> list[int]:
        del X
        return list(categorical_features or [])

    unsupervised_module.__dict__["infer_categorical_features"] = infer_categorical_features
    unsupervised_module.__dict__["_synthdata_explicit_types_patched"] = True
    logger.info("[tabpfn] patched unsupervised type inference to honor explicit schema roles")


def _patch_use_classifier_nan_bug():
    """Work around a tabpfn_extensions bug where a column's classifier-vs-
    regressor decision is inconsistent between ``density_`` (which picks the
    model) and ``sample_from_model_prediction_`` (which picks the predict
    API), causing e.g. ``TabPFNClassifier.predict() got an unexpected keyword
    argument 'output_type'``.

    Root cause: ``use_classifier_`` counts unique values via
    ``torch.unique``/``np.unique`` without dropping NaNs first. Since NaN !=
    NaN, every missing entry counts as its own "unique" value. ``density_``
    calls it on a target-observed-filtered (NaN-free) column, while
    ``sample_from_model_prediction_`` calls it again on the raw column
    (which, for imputation/synthesis, is mostly NaN) -- inflating the unique
    count past ``max_classes`` and flipping the decision for genuinely
    low-cardinality categorical columns. Filtering NaNs before counting
    makes both call sites agree. Safe/idempotent to call repeatedly; patches
    the class in place since there's no supported extension point.

    Tracking: distinct from the categorical-inference bug fixed by upstream
    PRs #326/#312 (see ``/memories/repo/synthdata-tabpfn-notes.md``) -- this
    is `TabPFNUnsupervisedModel.use_classifier_` in
    ``tabpfn_extensions/unsupervised/unsupervised.py``, still present as of
    the git-main commit this repo pins in ``pyproject.toml``
    (``tabpfn-extensions = { git = ..., branch = "main" }``). No upstream
    issue/PR has been filed for this specific bug yet -- if you file one,
    add the link here and re-check whether this patch is still needed before
    deleting it.
    """
    from tabpfn_extensions import unsupervised
    from tabpfn_extensions.utils import get_max_num_classes

    def use_classifier_(self, column_idx, y):
        is_categorical = column_idx in self.categorical_features
        if self.tabpfn_clf is None:
            return is_categorical
        max_classes = get_max_num_classes(self.tabpfn_clf)
        if torch.is_tensor(y):
            y_valid = y[~torch.isnan(y)] if torch.is_floating_point(y) else y
            n_unique = torch.unique(y_valid).numel()
        else:
            y_arr = np.asarray(y)
            if np.issubdtype(y_arr.dtype, np.floating):
                y_arr = y_arr[~np.isnan(y_arr)]
            n_unique = len(np.unique(y_arr))
        return is_categorical and (max_classes is None or n_unique <= max_classes)

    unsupervised.TabPFNUnsupervisedModel.use_classifier_ = use_classifier_


def _patch_regression_prediction_device():
    """Return the regressor's prediction and sample on the CPU, where
    tabpfn_extensions keeps the table it is filling in.

    tabpfn 9.x returns ``predict(output_type="full")`` logits and the
    ``criterion`` (a ``FullSupportBarDistribution``) on the fitted model's
    device; tabpfn 8.0.8 returned them on the CPU. On a GPU,
    ``impute_single_permutation_`` then writes a CUDA sample into a CPU tensor
    and fails with "Expected all tensors to be on the same device". ``impute_``
    hits the same mismatch when it averages the per-permutation criteria and
    samples from the merged distribution, so the dict itself is moved, not just
    the sample. The criterion is copied before moving it so the fitted model
    keeps its own on the GPU. Classifier predictions are numpy arrays and pass
    through unchanged. Idempotent.

    Tracking: tabpfn_extensions ``TabPFNUnsupervisedModel`` at the commit
    pinned in ``uv.lock``; not filed upstream yet.
    """
    import copy

    from tabpfn_extensions import unsupervised

    cls = unsupervised.TabPFNUnsupervisedModel
    if getattr(cls, "_synthdata_cpu_prediction_patched", False):
        return
    original = cls.sample_from_model_prediction_

    def sample_from_model_prediction_(self, column_idx, X_fit, model, X_predict, t):
        pred, pred_sampled = original(self, column_idx, X_fit, model, X_predict, t)
        if isinstance(pred, dict) and torch.is_tensor(pred.get("logits")):
            pred = {
                **pred,
                "logits": pred["logits"].cpu(),
                "criterion": copy.deepcopy(pred["criterion"]).cpu(),
            }
        return pred, pred_sampled.cpu()

    cls.sample_from_model_prediction_ = sample_from_model_prediction_
    cls._synthdata_cpu_prediction_patched = True


def _make_experiment():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions import unsupervised
    from tabpfn_extensions.unsupervised import experiments

    _patch_explicit_categorical_feature_inference()
    _patch_use_classifier_nan_bug()
    _patch_regression_prediction_device()

    model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(), tabpfn_reg=TabPFNRegressor()
    )
    experiment = experiments.GenerateSyntheticDataExperiment(task_type="unsupervised")
    # Disable the internal auto-plot: should_plot=False is not respected by this
    # version and self.data has duplicate indices after pd.concat, which breaks
    # seaborn reindex.
    experiment.plot = lambda **kwargs: None
    return experiment, model_unsupervised


def generate_tabpfn_standard(
    train_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    target_is_categorical: bool = True,
    variable_schema_fingerprint: str | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, object]:
    """Features-only synthesis; target label drawn post-hoc from a fresh classifier.

    Each label is sampled from the classifier's predicted class probabilities
    rather than taken as the most likely class, so minority classes appear at
    their conditional rate instead of being argmax-ed away.

    Returns ``(synthetic_df, experiment)`` -- ``experiment.data`` (the
    real+synthetic long frame) is useful for real-vs-synthetic plotting.
    """
    validate_tabpfn_target(target_column, target_is_categorical)

    from tabpfn import TabPFNClassifier

    # TabPFN requires a purely numeric array; the raw (pre-imputation) train
    # split may still have string-valued categorical columns (e.g. a plain CSV
    # source, unlike the pre-encoded UCI hepatitis example) -- those are
    # always factorized. Declared categorical_columns are ALSO force-
    # factorized even when already numeric (e.g. a real {1..5} ordinal or a
    # {0, 2} nominal): TabPFN's unsupervised experiment API's categorical
    # handling returns compact 0-indexed class predictions regardless of the
    # input's actual value domain, so an unencoded numeric-but-categorical
    # column comes back from generation silently shifted/relabeled (e.g.
    # {1..5} -> {0..4}, or {0, 2} -> {0, 1}) unless it goes through this same
    # factorize/decode round-trip first.
    encoded_features, category_maps = label_encode_non_numeric_columns(
        train_df, feature_columns, categorical_columns=categorical_columns
    )
    x = encoded_features.to_numpy(dtype=float)
    y = train_df[target_column].to_numpy()
    attribute_names = list(feature_columns)
    categorical_indices = [
        attribute_names.index(c) for c in categorical_columns if c in attribute_names
    ]

    experiment, model_unsupervised = _make_experiment()
    logger.info(
        "[tabpfn] standard experiment.run train_shape=%s n_samples=%d "
        "categorical_columns=%s categorical_indices=%s target=%r target_kind=%s "
        "schema_fingerprint=%s",
        x.shape,
        n_samples,
        list(categorical_columns),
        categorical_indices,
        target_column,
        "categorical" if target_is_categorical else "continuous",
        variable_schema_fingerprint or "unavailable",
    )
    experiment.run(
        tabpfn=model_unsupervised,
        X=x,
        y=y,
        attribute_names=attribute_names,
        indices=list(range(len(attribute_names))),
        categorical_features=categorical_indices,
        n_samples=n_samples,
        should_plot=False,
    )
    experiment.data = experiment.data.reset_index(drop=True)

    # Not experiment.data_synthetic: tabpfn_extensions unconditionally resamples
    # (with replacement) data_synthetic to match len(data_real) for its internal
    # pairplot, even with should_plot=False -- reading it back would silently
    # give ``len(train_df)`` rows instead of the requested n_samples.
    synthetic_values = np.asarray(experiment.synthetic_X.detach().cpu().numpy(), dtype=float)
    synthetic_encoded = pd.DataFrame(synthetic_values, columns=attribute_names)

    clf = TabPFNClassifier()
    clf.fit(x, y)
    proba = clf.predict_proba(synthetic_encoded.to_numpy(dtype=float))
    target_values = sample_labels(proba, clf.classes_, seed)

    synthetic_data = decode_label_encoded_columns(synthetic_encoded, category_maps)
    synthetic_data[target_column] = target_values

    return synthetic_data, experiment


def generate_tabpfn_custom(
    train_df: pd.DataFrame,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    target_is_categorical: bool = True,
    variable_schema_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, object]:
    """Features + target modeled jointly (target treated as just another column).

    Returns ``(synthetic_df, experiment)``.
    """
    validate_tabpfn_target(target_column, target_is_categorical)

    # See the comment in generate_tabpfn_standard for why categorical_columns
    # (+ target_column here, since it's modeled jointly with the features) is
    # passed through to force-factorize already-numeric categorical columns.
    modeled_categorical_columns = list(categorical_columns) + [target_column]
    encoded_train, category_maps = label_encode_non_numeric_columns(
        train_df,
        train_df.columns.tolist(),
        categorical_columns=modeled_categorical_columns,
    )
    train_array = encoded_train.to_numpy(dtype=float)
    attribute_names = train_df.columns.tolist()
    categorical_indices = [
        attribute_names.index(c) for c in modeled_categorical_columns if c in attribute_names
    ]

    experiment, model_unsupervised = _make_experiment()
    logger.info(
        "[tabpfn] custom experiment.run train_shape=%s n_samples=%d "
        "categorical_columns=%s categorical_indices=%s target=%r target_kind=%s "
        "schema_fingerprint=%s",
        train_array.shape,
        n_samples,
        modeled_categorical_columns,
        categorical_indices,
        target_column,
        "categorical" if target_is_categorical else "continuous",
        variable_schema_fingerprint or "unavailable",
    )
    experiment.run(
        tabpfn=model_unsupervised,
        X=train_array,
        y=np.array([]),
        attribute_names=attribute_names,
        indices=list(range(len(attribute_names))),
        categorical_features=categorical_indices,
        n_samples=n_samples,
        should_plot=False,
    )
    experiment.data = experiment.data.reset_index(drop=True)

    # See the comment in generate_tabpfn_standard: experiment.data_synthetic is
    # resampled to match len(data_real), not n_samples -- use the raw array instead.
    synthetic_values = np.asarray(experiment.synthetic_X.detach().cpu().numpy(), dtype=float)
    synthetic_encoded = pd.DataFrame(synthetic_values, columns=attribute_names)
    synthetic_data = decode_label_encoded_columns(synthetic_encoded, category_maps)
    return synthetic_data, experiment
