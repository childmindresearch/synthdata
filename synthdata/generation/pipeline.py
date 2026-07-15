"""Orchestrates synthetic data generation across synthcity, TabPFN, and TabPFGen.

For each enabled model family, generates a default synthetic dataset and,
if ``generation.hpo.enabled``, an Optuna-tuned ``*_hpo`` variant. Everything is
cached to CSV under ``generation.output_dir`` (skip regeneration unless
``generation.force_retrain``), and best hyperparameters are cached to a shared
JSON file (see :class:`synthdata.generation.hpo.BestParamsCache`).
"""

from collections.abc import Callable

import pandas as pd

from synthdata.config import Config
from synthdata.data import Dataset
from synthdata.generation import hpo as hpo_mod
from synthdata.generation import synthcity_backend as sc
from synthdata.generation import tabpfgen_backend as tpfgen
from synthdata.generation import tabpfn_backend as tpfn
from synthdata.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def run_generation(
    cfg: Config, dataset: Dataset, plot_callback: Callable | None = None
) -> dict[str, pd.DataFrame]:
    """Generate (or load cached) synthetic datasets for every configured model.

    ``plot_callback(name, synthetic_df, extra)`` is invoked right after each
    dataset is (re)generated (not when loaded from cache), so callers can save
    real-vs-synthetic figures inline; see :mod:`synthdata.plotting.generation_plots`.
    """
    if dataset.train_imputed_df is None:
        raise RuntimeError(
            "Dataset must be imputed before generation (run synthdata.imputation.run_imputation first)"
        )

    gen_cfg = cfg.generation
    output_dir = ensure_dir(gen_cfg.output_dir)
    n_samples = gen_cfg.n_samples
    seed = cfg.seed

    best_params_path = gen_cfg.hpo.best_params_path or hpo_mod.default_best_params_path(
        output_dir
    )
    best_params = hpo_mod.BestParamsCache(best_params_path)

    synthetic_datasets: dict[str, pd.DataFrame] = {}

    def _cached_or_build(name, build_fn):
        path = output_dir / f"{name}.csv"
        if path.exists() and not gen_cfg.force_retrain:
            logger.info("[%s] using cached synthetic data at %s", name, path)
            df = pd.read_csv(path)
            synthetic_datasets[name] = df
            return df

        logger.info("[%s] generating synthetic data (n_samples=%d)", name, n_samples)
        result = build_fn()
        df, extra = result if isinstance(result, tuple) else (result, None)
        df.to_csv(path, index=False)
        synthetic_datasets[name] = df
        if plot_callback is not None:
            try:
                plot_callback(name, df, extra)
            except Exception as exc:  # noqa: BLE001 - plotting must never break generation
                logger.warning("[%s] plot callback failed: %s", name, exc)
        return df

    # ------------------------------------------------------------------
    # synthcity models
    # ------------------------------------------------------------------
    if gen_cfg.synthcity.enabled and gen_cfg.synthcity.names:
        fairness_column = dataset.sensitive_columns[0] if dataset.sensitive_columns else None
        train_loader = sc.make_loader(
            dataset.train_imputed_df,
            dataset.target_column,
            dataset.sensitive_columns,
            random_state=seed,
            fairness_column=fairness_column,
        )

        for name in gen_cfg.synthcity.names:
            _cached_or_build(
                name,
                lambda name=name: sc.fit_generate(
                    name, {}, train_loader, n_samples, seed, workspace=output_dir / "synthcity_workspace"
                ),
            )

            if gen_cfg.hpo.enabled:
                if not best_params.has("synthcity", name):
                    objective = sc.build_synthcity_objective(
                        name, train_loader, gen_cfg.hpo, seed, workspace=output_dir / "synthcity_workspace"
                    )
                    params = hpo_mod.run_study(
                        f"hpo_{name}", objective, gen_cfg.hpo, output_dir, seed
                    )
                    best_params.set("synthcity", name, params)
                params = dict(best_params.get("synthcity", name))

                override = gen_cfg.hpo.final_n_iter_override
                if override and sc.plugin_accepts(name, "n_iter"):
                    params["n_iter"] = override

                _cached_or_build(
                    f"{name}_hpo",
                    lambda name=name, params=params: sc.fit_generate(
                        name, params, train_loader, n_samples, seed, workspace=output_dir / "synthcity_workspace"
                    ),
                )

    # ------------------------------------------------------------------
    # TabPFN models (no HPO; use the original, pre-imputation train split)
    # ------------------------------------------------------------------
    if gen_cfg.tabpfn.enabled:
        if "standard" in gen_cfg.tabpfn.variants:
            _cached_or_build(
                "tabpfn_standard",
                lambda: tpfn.generate_tabpfn_standard(
                    dataset.train_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    n_samples,
                ),
            )
        if "custom" in gen_cfg.tabpfn.variants:
            _cached_or_build(
                "tabpfn_custom",
                lambda: tpfn.generate_tabpfn_custom(
                    dataset.train_df,
                    dataset.categorical_columns,
                    dataset.target_column,
                    n_samples,
                ),
            )

    # ------------------------------------------------------------------
    # TabPFGen models (use the imputed train split)
    # ------------------------------------------------------------------
    if gen_cfg.tabpfgen.enabled:
        eval_fn = None
        if gen_cfg.hpo.enabled:
            eval_fn = hpo_mod.build_synthetic_eval_fn(
                dataset.train_imputed_df,
                dataset.test_imputed_df,
                dataset.target_column,
                dataset.sensitive_columns,
                gen_cfg.hpo.metric_config,
                seed,
                workspace=output_dir / "synthcity_workspace",

            )

        if "standard" in gen_cfg.tabpfgen.variants:
            _cached_or_build(
                "tabpfgen_standard",
                lambda: tpfgen.generate_tabpfgen_standard(
                    dataset.train_imputed_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    n_samples,
                    tabpfgen_params=gen_cfg.tabpfgen.standard_params,
                ),
            )

            if gen_cfg.hpo.enabled:
                if not best_params.has("tabpfgen", "tabpfgen_standard"):
                    objective = tpfgen.build_tabpfgen_standard_objective(
                        dataset.train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        gen_cfg.hpo.sgld_step_cap,
                        eval_fn,
                    )
                    params = hpo_mod.run_study(
                        "hpo_tabpfgen_standard",
                        objective,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                        drop_keys=(),
                    )
                    best_params.set("tabpfgen", "tabpfgen_standard", params)
                params = best_params.get("tabpfgen", "tabpfgen_standard")

                _cached_or_build(
                    "tabpfgen_standard_hpo",
                    lambda params=params: tpfgen.generate_tabpfgen_standard(
                        dataset.train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        tabpfgen_params=params,
                        relabel_with_classifier=True,
                    ),
                )

        if "custom" in gen_cfg.tabpfgen.variants:
            _cached_or_build(
                "tabpfgen_custom",
                lambda: tpfgen.generate_tabpfgen_custom(
                    dataset.train_imputed_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    n_samples,
                    seed=seed,
                    sgld_params=gen_cfg.tabpfgen.custom_params,
                ),
            )

            if gen_cfg.hpo.enabled:
                if not best_params.has("tabpfgen", "tabpfgen_custom"):
                    objective = tpfgen.build_tabpfgen_custom_objective(
                        dataset.train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        gen_cfg.hpo.sgld_step_cap,
                        eval_fn,
                        seed=seed,
                    )
                    params = hpo_mod.run_study(
                        "hpo_tabpfgen_custom",
                        objective,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                        drop_keys=(),
                    )
                    best_params.set("tabpfgen", "tabpfgen_custom", params)
                params = best_params.get("tabpfgen", "tabpfgen_custom")

                _cached_or_build(
                    "tabpfgen_custom_hpo",
                    lambda params=params: tpfgen.generate_tabpfgen_custom(
                        dataset.train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        seed=seed,
                        sgld_params=params,
                    ),
                )

    logger.info("Generated/loaded %d synthetic datasets: %s", len(synthetic_datasets), sorted(synthetic_datasets))
    return synthetic_datasets
