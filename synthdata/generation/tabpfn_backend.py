"""TabPFN-based synthetic data generation (two variants).

Both variants use TabPFN's unsupervised-experiment API (``tabpfn_extensions``)
and operate on the *original* (pre-imputation) train split, since TabPFN's
foundation model handles missing values natively.
"""

import numpy as np
import pandas as pd

from synthdata.utils import get_logger

logger = get_logger(__name__)


def _make_experiment():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn_extensions import unsupervised
    from tabpfn_extensions.unsupervised import experiments

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

    x = train_df[feature_columns].to_numpy()
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

    synthetic_data = experiment.data_synthetic[attribute_names].reset_index(drop=True)

    clf = TabPFNClassifier()
    clf.fit(x, y)
    synthetic_data[target_column] = clf.predict(synthetic_data.to_numpy())

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
    train_array = train_df.values
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

    synthetic_data = pd.DataFrame(
        experiment.data_synthetic.drop(columns=["real_or_synthetic"]),
        columns=train_df.columns,
    )
    return synthetic_data, experiment
