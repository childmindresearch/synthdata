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
from synthdata.data import Dataset, remask_synthetic
from synthdata.generation import cart_fill, class_quota
from synthdata.generation import hpo as hpo_mod
from synthdata.generation import synthcity_backend as sc
from synthdata.generation import tabpfn_backend as tpfn
from synthdata.utils import (
    ensure_dir,
    get_logger,
    replicate_name,
    resolve_device,
    set_global_seed,
    split_replicate_name,
)

# tabpfgen_backend is imported lazily (see the `gen_cfg.tabpfgen.enabled` branch
# below) because it does `from tabpfgen import TabPFGen` at module scope (to
# subclass TabPFGenSGLDLabels), which requires the optional `tabpfn` extra.
# Keeping it out of this module's top-level imports lets `synthdata.generation`
# (and anything testing config/caching/other-backend logic) import cleanly with
# only the base dependencies installed.

logger = get_logger(__name__)


def needs_imputed_data(gen_cfg) -> bool:
    """Whether any configured model actually requires the imputed train split.

    synthcity and TabPFGen always fit on ``train_imputed_df``. TabPFN can fit on
    either split (see ``TabPFNConfig.data_variants``); it only needs imputed data
    if "imputed" is one of the requested variants.
    """
    return (
        (gen_cfg.synthcity.enabled and bool(gen_cfg.synthcity.names))
        or gen_cfg.tabpfgen.enabled
        or (gen_cfg.tabpfn.enabled and "imputed" in gen_cfg.tabpfn.data_variants)
    )


def run_generation(
    cfg: Config,
    dataset: Dataset,
    plot_callback: Callable | None = None,
    experiment=None,
) -> dict[str, pd.DataFrame]:
    """Generate (or load cached) synthetic datasets for every configured model.

    ``plot_callback(name, synthetic_df, extra)`` is invoked right after each
    dataset is (re)generated (not when loaded from cache), so callers can save
    real-vs-synthetic figures inline; see :mod:`synthdata.plotting.generation_plots`.
    If ``experiment`` (a :class:`synthdata.experiment.Experiment`) is given, a
    failed plot callback is recorded to its manifest (``stage="generation_plot_failed"``)
    in addition to the console warning, so the skip is visible from the
    output directory alone.
    """
    if dataset.train_imputed_df is None and needs_imputed_data(cfg.generation):
        raise RuntimeError(
            "Dataset must be imputed before generation (run synthdata.imputation.run_imputation first)"
        )
    if cfg.generation.hpo.enabled and dataset.tuning_index.empty:
        raise RuntimeError(
            "generation.hpo.enabled needs a tuning split to score candidates on; set "
            "data.tuning_fraction above 0"
        )

    gen_cfg = cfg.generation
    output_dir = ensure_dir(gen_cfg.output_dir)
    n_samples = gen_cfg.n_samples
    seed = cfg.seed
    device = resolve_device(cfg.device)

    best_params_path = gen_cfg.hpo.best_params_path or hpo_mod.default_best_params_path(output_dir)
    best_params = hpo_mod.BestParamsCache(best_params_path)

    synthetic_datasets: dict[str, pd.DataFrame] = {}

    # Every synthetic dataset is drawn to the real train class shares (per-class
    # quotas, see class_quota), so no generator gains macro-F1 by rebalancing
    # the classes. HPO candidates get the search-train shares.
    match_prior = gen_cfg.match_class_prior and dataset.target_is_categorical
    train_prior = dataset.train_df[dataset.target_column].value_counts(normalize=True)
    class_prior = train_prior if match_prior else None
    search_prior = (
        dataset.search_train_imputed_df[dataset.target_column].value_counts(normalize=True)
        if match_prior and gen_cfg.hpo.enabled and dataset.search_train_imputed_df is not None
        else None
    )
    # Not in output_dir itself: every CSV there is read as a synthetic dataset.
    class_sampling_path = output_dir / "diagnostics" / "class_sampling.csv"

    def _record_class_sampling(name, report):
        """Keep each model's quotas, rows kept and raw class shares in class_sampling.csv."""
        rows = report.to_frame(name)
        if class_sampling_path.exists():
            previous = pd.read_csv(class_sampling_path)
            rows = pd.concat([previous[previous["model"] != name], rows], ignore_index=True)
        ensure_dir(class_sampling_path.parent)
        rows.to_csv(class_sampling_path, index=False)

    # Hyperparameter search fits candidates on train minus tuning and scores
    # them on tuning. Final models (default and tuned) are fitted on all of
    # train, so the test split is only ever used by evaluation.
    hpo_eval_fn = None
    if gen_cfg.hpo.enabled and needs_imputed_data(gen_cfg):
        hpo_eval_fn = hpo_mod.build_hpo_eval_fn(
            dataset.search_train_imputed_df,
            dataset.tuning_imputed_df,
            dataset.target_column,
            dataset.nominal_columns,
            dataset.categorical_columns,
            dataset.target_is_categorical,
            dataset.sensitive_columns,
            gen_cfg.hpo,
            seed,
            match_prior=match_prior,
            workspace=output_dir / "synthcity_workspace",
        )

    def _cached_or_build(name, build_fn):
        """Load or build every replicate of ``name``; ``build_fn(seed, count)`` makes one.

        Replicate ``r`` is saved as ``replicate_name(name, r)`` and generated
        with seed ``cfg.seed + r`` (replicate 0 keeps the plain name and the
        base seed), so the spread across replicates measures how much a
        model's scores depend on its random seed.
        """
        for replicate in range(gen_cfg.n_replicates):
            _build_one(replicate_name(name, replicate), seed + replicate, build_fn)

    remask = cfg.imputation.missing_indicators.remask_synthetic and bool(
        dataset.missing_indicator_columns
    )

    cart = {}

    def _cart_fill(df, replicate_seed):
        """Values for indicator-only columns in "recorded" synthetic rows (CART leaf sampling)."""
        values = dataset.indicator_only_values
        if not cart:
            real = dataset.train_imputed_df.set_axis(dataset.train_df.index)
            cart["reference"] = real
            categorical = set(dataset.variable_schema) - {
                c for c, e in dataset.variable_schema.items() if e.get("kind") == "continuous"
            }
            cart["categorical"] = categorical
            cart["models"] = cart_fill.fit_cart_fills(
                cart_fill.encode_predictors(real, real),
                values.loc[dataset.train_df.index],
                categorical,
                seed,
            )
        filled = cart_fill.fill(
            df,
            cart["models"],
            dataset.missing_indicator_columns,
            cart_fill.encode_predictors(df, cart["reference"]),
            replicate_seed,
        )
        report = cart_fill.recorded_rows_report(
            filled, values.loc[dataset.train_df.index], cart["categorical"]
        )
        return filled, pd.DataFrame(report)

    def _write_released(name, df, overwrite, replicate_seed):
        """Save the re-masked copy that is shared (evaluation scores the filled one)."""
        released = output_dir / "released" / f"{name}.csv"
        if remask and (overwrite or not released.exists()):
            ensure_dir(released.parent)
            out = remask_synthetic(df, dataset.missing_indicator_columns)
            if not dataset.indicator_only_values.empty:
                filled, report = _cart_fill(df, replicate_seed)
                out = pd.concat([out, filled], axis=1)
                report.to_csv(released.with_name(f"{name}_recorded_values.csv"), index=False)
            out.to_csv(released, index=False)

    def _build_one(name, replicate_seed, build_fn):
        path = output_dir / f"{name}.csv"
        if path.exists() and not gen_cfg.force_retrain:
            logger.info("[%s] using cached synthetic data at %s", name, path)
            df = pd.read_csv(path)
            synthetic_datasets[name] = df
            _write_released(name, df, overwrite=False, replicate_seed=replicate_seed)
            return df

        logger.info(
            "[%s] generating synthetic data (n_samples=%d, seed=%d)",
            name,
            n_samples,
            replicate_seed,
        )
        # Backends without a seed argument draw from the global RNGs.
        set_global_seed(replicate_seed)
        result = build_fn(replicate_seed, n_samples)
        df, extra = result if isinstance(result, tuple) else (result, None)
        if match_prior and dataset.target_column in df:
            report = df.attrs.get("class_sampling")
            if report is None:
                # Backend without per-class sampling: draw whole new datasets
                # (rejection sampling) until every class has its quota.
                def sample(count, round_seed, labels):
                    set_global_seed(round_seed)
                    more = build_fn(round_seed, count)
                    return more[0] if isinstance(more, tuple) else more

                df, report = class_quota.sample_to_quota(
                    sample,
                    dataset.target_column,
                    train_prior,
                    n_samples,
                    replicate_seed,
                    first_batch=df,
                )
            df.attrs.pop("class_sampling", None)
            _record_class_sampling(name, report)
        df.to_csv(path, index=False)
        _write_released(name, df, overwrite=True, replicate_seed=replicate_seed)
        synthetic_datasets[name] = df
        base, replicate = split_replicate_name(name)
        if replicate and df.equals(synthetic_datasets.get(base)):
            logger.warning(
                "[%s] replicate is identical to %s: this backend ignores the seed, so its "
                "replicates add no uncertainty information",
                name,
                base,
            )
        if plot_callback is not None:
            try:
                plot_callback(name, df, extra)
            except (ValueError, TypeError, OSError, RuntimeError) as exc:
                # Plotting must never break generation: skip just this
                # figure, but persist the skip to the experiment manifest so
                # it's visible from the output directory, not just the console.
                logger.warning("[%s] plot callback failed: %s", name, exc)
                if experiment is not None:
                    experiment.record(
                        "generation_plot_failed",
                        model=name,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
        return df

    # ------------------------------------------------------------------
    # synthcity models
    # ------------------------------------------------------------------
    if gen_cfg.synthcity.enabled and gen_cfg.synthcity.names:
        # synthcity's sensitive_features are the secrets its attribute-inference
        # and l-diversity metrics target; the fairness column is a protected group.
        fairness_column = dataset.protected_columns[0] if dataset.protected_columns else None
        train_loader = sc.make_loader(
            dataset.train_imputed_df,
            dataset.target_column,
            dataset.sensitive_columns,
            random_state=seed,
            fairness_column=fairness_column,
        )
        search_loader = (
            sc.make_loader(
                dataset.search_train_imputed_df,
                dataset.target_column,
                dataset.sensitive_columns,
                random_state=seed,
                fairness_column=fairness_column,
            )
            if gen_cfg.hpo.enabled
            else None
        )

        for name in gen_cfg.synthcity.names:
            _cached_or_build(
                name,
                lambda seed, count, name=name: sc.fit_generate(
                    name,
                    {},
                    train_loader,
                    count,
                    seed,
                    workspace=output_dir / "synthcity_workspace",
                    device=device,
                    classification=dataset.target_is_categorical,
                    class_prior=class_prior,
                ),
            )

            if gen_cfg.hpo.enabled and hpo_mod.model_budget(gen_cfg.hpo, name)[0] > 0:
                if not best_params.has("synthcity", name):
                    objective = sc.build_synthcity_objective(
                        name,
                        search_loader,
                        gen_cfg.hpo,
                        seed,
                        hpo_eval_fn,
                        n_samples,
                        workspace=output_dir / "synthcity_workspace",
                        device=device,
                        classification=dataset.target_is_categorical,
                        class_prior=search_prior,
                    )
                    params = hpo_mod.run_study(
                        f"hpo_{name}",
                        objective,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                        checkpoint_workspace=output_dir / "synthcity_workspace",
                        checkpoint_plugin=name,
                    )
                    best_params.set("synthcity", name, params)
                params = dict(best_params.get("synthcity", name))

                override = gen_cfg.hpo.final_n_iter_override
                if override and sc.plugin_accepts(name, "n_iter"):
                    params["n_iter"] = override

                _cached_or_build(
                    f"{name}_hpo",
                    lambda seed, count, name=name, params=params: sc.fit_generate(
                        name,
                        params,
                        train_loader,
                        count,
                        seed,
                        workspace=output_dir / "synthcity_workspace",
                        device=device,
                        classification=dataset.target_is_categorical,
                        class_prior=class_prior,
                    ),
                )

    # ------------------------------------------------------------------
    # TabPFN models (no HPO; can fit on the original pre-imputation train
    # split and/or the imputed one -- see gen_cfg.tabpfn.data_variants)
    # ------------------------------------------------------------------
    if gen_cfg.tabpfn.enabled or gen_cfg.tabpfgen.enabled:
        tpfn.set_model_version(gen_cfg.tabpfn.model_version)
    if gen_cfg.tabpfn.enabled:
        tpfn.validate_tabpfn_target(dataset.target_column, dataset.target_is_categorical)
        for data_variant in gen_cfg.tabpfn.data_variants:
            if data_variant == "imputed":
                train_df_variant = dataset.train_imputed_df
                suffix = "_imputed"
            else:
                train_df_variant = dataset.train_df
                suffix = ""

            if "standard" in gen_cfg.tabpfn.variants:
                _cached_or_build(
                    f"tabpfn_standard{suffix}",
                    lambda seed, count, train_df_variant=train_df_variant: (
                        tpfn.generate_tabpfn_standard(
                            train_df_variant,
                            dataset.feature_columns,
                            dataset.categorical_columns,
                            dataset.target_column,
                            count,
                            target_is_categorical=dataset.target_is_categorical,
                            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                            seed=seed,
                        )
                    ),
                )
            if "custom" in gen_cfg.tabpfn.variants:
                _cached_or_build(
                    f"tabpfn_custom{suffix}",
                    lambda seed, count, train_df_variant=train_df_variant: (
                        tpfn.generate_tabpfn_custom(
                            train_df_variant,
                            dataset.categorical_columns,
                            dataset.target_column,
                            count,
                            target_is_categorical=dataset.target_is_categorical,
                            variable_schema_fingerprint=dataset.variable_schema_fingerprint,
                        )
                    ),
                )

    # ------------------------------------------------------------------
    # TabPFGen models (use the imputed train split)
    # ------------------------------------------------------------------
    if gen_cfg.tabpfgen.enabled:
        from synthdata.generation import tabpfgen_backend as tpfgen

        eval_fn = hpo_eval_fn

        if "standard" in gen_cfg.tabpfgen.variants:
            _cached_or_build(
                "tabpfgen_standard",
                lambda seed, count: tpfgen.generate_tabpfgen_standard(
                    dataset.train_imputed_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    count,
                    tabpfgen_params=gen_cfg.tabpfgen.standard_params,
                ),
            )

            if (
                gen_cfg.hpo.enabled
                and hpo_mod.model_budget(gen_cfg.hpo, "tabpfgen_standard")[0] > 0
            ):
                if not best_params.has("tabpfgen", "tabpfgen_standard"):
                    objective = tpfgen.build_tabpfgen_standard_objective(
                        dataset.search_train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        hpo_mod.epoch_range(
                            gen_cfg.hpo, "tabpfgen_standard", tpfgen.SGLD_STEP_RANGE
                        ),
                        eval_fn,
                    )
                    params = hpo_mod.run_study(
                        "hpo_tabpfgen_standard",
                        objective,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                    )
                    best_params.set("tabpfgen", "tabpfgen_standard", params)
                params = best_params.get("tabpfgen", "tabpfgen_standard")

                _cached_or_build(
                    "tabpfgen_standard_hpo",
                    lambda seed, count, params=params: tpfgen.generate_tabpfgen_standard(
                        dataset.train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        count,
                        tabpfgen_params=params,
                        relabel_with_classifier=True,
                    ),
                )

        if "custom" in gen_cfg.tabpfgen.variants:
            _cached_or_build(
                "tabpfgen_custom",
                lambda seed, count: tpfgen.generate_tabpfgen_custom(
                    dataset.train_imputed_df,
                    dataset.feature_columns,
                    dataset.categorical_columns,
                    dataset.target_column,
                    count,
                    seed=seed,
                    sgld_params=gen_cfg.tabpfgen.custom_params,
                ),
            )

            if gen_cfg.hpo.enabled and hpo_mod.model_budget(gen_cfg.hpo, "tabpfgen_custom")[0] > 0:
                if not best_params.has("tabpfgen", "tabpfgen_custom"):
                    objective = tpfgen.build_tabpfgen_custom_objective(
                        dataset.search_train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        n_samples,
                        hpo_mod.epoch_range(gen_cfg.hpo, "tabpfgen_custom", tpfgen.SGLD_STEP_RANGE),
                        eval_fn,
                        seed=seed,
                    )
                    params = hpo_mod.run_study(
                        "hpo_tabpfgen_custom",
                        objective,
                        gen_cfg.hpo,
                        output_dir,
                        seed,
                    )
                    best_params.set("tabpfgen", "tabpfgen_custom", params)
                params = best_params.get("tabpfgen", "tabpfgen_custom")

                _cached_or_build(
                    "tabpfgen_custom_hpo",
                    lambda seed, count, params=params: tpfgen.generate_tabpfgen_custom(
                        dataset.train_imputed_df,
                        dataset.feature_columns,
                        dataset.categorical_columns,
                        dataset.target_column,
                        count,
                        seed=seed,
                        sgld_params=params,
                    ),
                )

    logger.info(
        "Generated/loaded %d synthetic datasets: %s",
        len(synthetic_datasets),
        sorted(synthetic_datasets),
    )
    return synthetic_datasets
