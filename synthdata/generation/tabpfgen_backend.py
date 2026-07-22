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

from synthdata.utils import get_logger

logger = get_logger(__name__)


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

        if balance_classes:
            classes = np.unique(y_train)
            n_per_class = n_samples // len(classes)
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
        return x_synth_out, y_synth_drifted


def generate_tabpfgen_standard(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    tabpfgen_params: dict | None = None,
    relabel_with_classifier: bool = False,
) -> pd.DataFrame:
    """Default TabPFGen classification synthesis.

    ``relabel_with_classifier=True`` discards TabPFGen's own predicted labels
    and instead assigns labels via a freshly-fit ``TabPFNClassifier`` (used for
    the HPO-tuned variant in the notebook, since it was found empirically more
    stable across sampled hyperparameters).
    """
    x_train = train_imputed_df[feature_columns].values.astype(float)
    y_train = train_imputed_df[target_column].values

    generator = TabPFGen(**(tabpfgen_params or {}))
    x_synth, y_synth = generator.generate_classification(
        X_train=x_train, y_train=y_train, n_samples=n_samples, balance_classes=True
    )

    synthetic = pd.DataFrame(x_synth, columns=feature_columns)
    for col in categorical_columns:
        n_cats = train_imputed_df[col].nunique()
        synthetic[col] = synthetic[col].round().clip(0, n_cats - 1).astype(int)

    if relabel_with_classifier:
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier()
        clf.fit(x_train, y_train)
        synthetic[target_column] = clf.predict(synthetic.to_numpy())
    else:
        n_cls = train_imputed_df[target_column].nunique()
        synthetic[target_column] = pd.Series(y_synth.astype(int)).clip(0, n_cls - 1).values

    return synthetic


def generate_tabpfgen_custom(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    seed: int = 42,
    sgld_params: dict | None = None,
) -> pd.DataFrame:
    """TabPFGenSGLDLabels synthesis with oversample-then-subsample class balancing.

    ``balance_classes=True`` in TabPFGen always generates equal counts per class,
    so to recover the true training class distribution we over-generate and then
    subsample proportionally.
    """
    x_train = train_imputed_df[feature_columns].values.astype(float)
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

    synth_all = pd.DataFrame(x_synth_all, columns=feature_columns)
    for col in categorical_columns:
        n_cats = train_imputed_df[col].nunique()
        synth_all[col] = synth_all[col].round().clip(0, n_cats - 1).astype(int)

    n_classes_enc = train_imputed_df[target_column].nunique()
    synth_all[target_column] = pd.Series(y_synth_all.astype(int)).clip(0, n_classes_enc - 1).values

    parts = []
    for orig_cls, proportion in train_proportions.items():
        n_needed = round(n_samples * proportion)
        cls_rows = synth_all[synth_all[target_column] == orig_cls]
        parts.append(
            cls_rows.sample(n=n_needed, replace=len(cls_rows) < n_needed, random_state=seed)
        )
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_tabpfgen_standard_objective(
    train_imputed_df: pd.DataFrame,
    feature_columns: list,
    categorical_columns: list,
    target_column: str,
    n_samples: int,
    sgld_step_cap: int,
    eval_fn: Callable[[pd.DataFrame], float],
):
    """Optuna objective searching TabPFGen's SGLD hyperparameters (standard variant)."""
    from tabpfn import TabPFNClassifier

    x_feat = train_imputed_df[feature_columns].values.astype(float)
    y_label = train_imputed_df[target_column].values

    def objective(trial: optuna.Trial) -> float:
        n_steps = min(trial.suggest_int("n_sgld_steps", 100, 2000, step=100), sgld_step_cap)
        step_size = trial.suggest_float("sgld_step_size", 0.001, 0.1, log=True)
        noise_scale = trial.suggest_float("sgld_noise_scale", 0.001, 0.5, log=True)
        try:
            gen = TabPFGen(
                n_sgld_steps=n_steps, sgld_step_size=step_size, sgld_noise_scale=noise_scale
            )
            x_s, _ = gen.generate_classification(
                X_train=x_feat, y_train=y_label, n_samples=n_samples, balance_classes=True
            )
            syn = pd.DataFrame(x_s, columns=feature_columns)
            for c in categorical_columns:
                syn[c] = syn[c].round().clip(0, train_imputed_df[c].nunique() - 1).astype(int)
            clf = TabPFNClassifier()
            clf.fit(x_feat, y_label)
            syn[target_column] = clf.predict(syn.to_numpy())
        except (ValueError, RuntimeError) as exc:
            logger.warning("tabpfgen_standard trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc
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
):
    """Optuna objective searching TabPFGenSGLDLabels's SGLD hyperparameters."""
    x_feat = train_imputed_df[feature_columns].values.astype(float)
    y_label = train_imputed_df[target_column].values
    proportions = train_imputed_df[target_column].value_counts(normalize=True)

    def objective(trial: optuna.Trial) -> float:
        n_steps = min(trial.suggest_int("n_sgld_steps", 100, 2000, step=100), sgld_step_cap)
        step_size = trial.suggest_float("sgld_step_size", 0.001, 0.1, log=True)
        noise_scale = trial.suggest_float("sgld_noise_scale", 0.001, 0.5, log=True)
        try:
            n_per = int(np.ceil(n_samples * proportions.max()))
            gen = TabPFGenSGLDLabels(
                n_sgld_steps=n_steps, sgld_step_size=step_size, sgld_noise_scale=noise_scale
            )
            x_s, y_s = gen.generate_classification(
                x_feat,
                y_label,
                n_samples=n_per * len(proportions),
                balance_classes=True,
            )
            all_df = pd.DataFrame(x_s, columns=feature_columns)
            for c in categorical_columns:
                all_df[c] = all_df[c].round().clip(0, train_imputed_df[c].nunique() - 1).astype(int)
            all_df[target_column] = (
                pd.Series(y_s.astype(int))
                .clip(0, train_imputed_df[target_column].nunique() - 1)
                .values
            )
            parts = []
            for cls, prop in proportions.items():
                n_need = round(n_samples * prop)
                rows = all_df[all_df[target_column] == cls]
                parts.append(rows.sample(n=n_need, replace=len(rows) < n_need, random_state=seed))
            syn = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        except (ValueError, RuntimeError) as exc:
            logger.warning("tabpfgen_custom trial %d failed: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc
        return eval_fn(syn)

    return objective
