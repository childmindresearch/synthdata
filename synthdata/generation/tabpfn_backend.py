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


def _make_experiment():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions import unsupervised
    from tabpfn_extensions.unsupervised import experiments

    _patch_use_classifier_nan_bug()

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
) -> tuple[pd.DataFrame, object]:
    """Features-only synthesis; target label assigned post-hoc via a fresh classifier.

    Returns ``(synthetic_df, experiment)`` -- ``experiment.data`` (the
    real+synthetic long frame) is useful for real-vs-synthetic plotting.
    """
    from tabpfn import TabPFNClassifier

    # TabPFN requires a purely numeric array; the raw (pre-imputation) train
    # split may still have string-valued categorical columns (e.g. a plain CSV
    # source, unlike the pre-encoded UCI hepatitis example) that aren't caught
    # by ``categorical_columns`` alone -- encode every feature column, NaN
    # preserved so TabPFN's native missing-value handling still applies.
    encoded_features, category_maps = label_encode_non_numeric_columns(train_df, feature_columns)
    x = encoded_features.to_numpy(dtype=float)
    y = train_df[target_column].to_numpy()
    attribute_names = list(feature_columns)
    categorical_indices = [
        attribute_names.index(c) for c in categorical_columns if c in attribute_names
    ]

    experiment, model_unsupervised = _make_experiment()
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
    synthetic_encoded = pd.DataFrame(
        experiment.synthetic_X.detach().cpu().numpy(), columns=attribute_names
    )

    clf = TabPFNClassifier()
    clf.fit(x, y)
    target_values = clf.predict(synthetic_encoded.to_numpy(dtype=float))

    synthetic_data = decode_label_encoded_columns(synthetic_encoded, category_maps)
    synthetic_data[target_column] = target_values

    return synthetic_data, experiment


def generate_tabpfn_custom(
    train_df: pd.DataFrame,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
) -> tuple[pd.DataFrame, object]:
    """Features + target modeled jointly (target treated as just another column).

    Returns ``(synthetic_df, experiment)``.
    """
    encoded_train, category_maps = label_encode_non_numeric_columns(
        train_df, train_df.columns.tolist()
    )
    train_array = encoded_train.to_numpy(dtype=float)
    attribute_names = train_df.columns.tolist()
    categorical_indices = [
        attribute_names.index(c)
        for c in list(categorical_columns) + [target_column]
        if c in attribute_names
    ]

    experiment, model_unsupervised = _make_experiment()
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
    synthetic_encoded = pd.DataFrame(
        experiment.synthetic_X.detach().cpu().numpy(), columns=attribute_names
    )
    synthetic_data = decode_label_encoded_columns(synthetic_encoded, category_maps)
    return synthetic_data, experiment
