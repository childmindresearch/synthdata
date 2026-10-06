"""Static metric catalogs and context-aware emitted-key manifests.

The framework selection names are not always the identities written by the
evaluators. This module is the single expansion point for selected emitted
keys; data-dependent qualified identities are added only from declared schema
and target context, never discovered from an observed result table.
"""

from collections.abc import Mapping, Sequence

# Canonical release-evidence identities. Adapters must
# validate against these manifests instead of inferring identities from rows.
CANONICAL_METRIC_MANIFEST = {
    "elastic_net_jsd": "elastic_net_jsd.v1",
    "mixed_mmd": "mixed_mmd.v1",
    "release_privacy": "release_privacy.v1",
    "tstr_macro_f1": "tstr_macro_f1.v1",
    "equalized_odds": "equalized_odds.final.v1",
    "representation_evidence": "representation_evidence.v1",
}

CANONICAL_EXPECTED_MANIFEST = tuple(CANONICAL_METRIC_MANIFEST.values())
CANONICAL_HPO_ALLOWLIST = frozenset(
    {
        CANONICAL_METRIC_MANIFEST["elastic_net_jsd"],
        CANONICAL_METRIC_MANIFEST["mixed_mmd"],
        CANONICAL_METRIC_MANIFEST["tstr_macro_f1"],
    }
)

LEGACY_AUDIT_MANIFEST = (
    "synthcity.privacy.k-anonymization.v1",
    "synthcity.privacy.k-map.v1",
    "synthcity.privacy.distinct-l-diversity.v1",
    "syntheval.median_DCR.legacy",
    "syntheval.eps_identif_risk.legacy",
    "syntheval.mia_recall.legacy",
    "syntheval.att_discl_risk.legacy",
    "synthcity.performance.hidden_split.legacy",
    "custom.fairness.synthetic_cv.legacy",
    "syntheval.avg_macro_F1_diff_v2.legacy",
    "syntheval.avg_F1_diff.legacy",
)

# Framework-facing manifests are intentionally separate so each adapter can
# bind only identities it owns.  A copy is returned by helper below; callers
# must not mutate these module constants.
SYNTHCITY_CANONICAL_MANIFEST = (
    CANONICAL_METRIC_MANIFEST["elastic_net_jsd"],
    CANONICAL_METRIC_MANIFEST["mixed_mmd"],
)
SYNTHEVAL_CANONICAL_MANIFEST = (CANONICAL_METRIC_MANIFEST["tstr_macro_f1"],)
CUSTOM_CANONICAL_MANIFEST = (
    CANONICAL_METRIC_MANIFEST["release_privacy"],
    CANONICAL_METRIC_MANIFEST["equalized_odds"],
    CANONICAL_METRIC_MANIFEST["representation_evidence"],
)

CANONICAL_MANIFEST_BY_FRAMEWORK = {
    "synthcity": SYNTHCITY_CANONICAL_MANIFEST,
    "syntheval": SYNTHEVAL_CANONICAL_MANIFEST,
    "custom": CUSTOM_CANONICAL_MANIFEST,
}


def canonical_expected_manifest() -> tuple[str, ...]:
    """Return immutable release metric identities in canonical order."""
    return CANONICAL_EXPECTED_MANIFEST


def canonical_manifest_for_framework(framework: str) -> tuple[str, ...]:
    """Return exact canonical identities owned by ``framework``."""
    try:
        return CANONICAL_MANIFEST_BY_FRAMEWORK[framework]
    except KeyError as exc:
        raise ValueError(f"Unknown canonical framework: {framework!r}") from exc


# ---------------------------------------------------------------------------
# synthcity
# ---------------------------------------------------------------------------

#: Full default metric set (mirrors the hepatitis notebook's synthcity_metric_config).
SYNTHCITY_METRIC_CONFIG = {
    "sanity": [
        "data_mismatch",
        "common_rows_proportion",
        "nearest_syn_neighbor_distance",
        "close_values_probability",
        "distant_values_probability",
    ],
    "stats": [
        "jensenshannon_dist",
        "chi_squared_test",
        "inv_kl_divergence",
        "ks_test",
        "max_mean_discrepancy",
        "wasserstein_dist",
        "prdc",
        "alpha_precision",
    ],
    "performance": [
        "linear_model",
        "mlp",
        "xgb",
        "feat_rank_distance",
        "linear_model_augmentation",
        "mlp_augmentation",
        "xgb_augmentation",
    ],
    "detection": [
        "detection_xgb",
        "detection_mlp",
        "detection_linear",
    ],
    "privacy": [
        "delta-presence",
        "k-anonymization",
        "k-map",
        "distinct l-diversity",
        "identifiability_score",
        "DomiasMIA_prior",
    ],
    "attack": [
        "data_leakage_mlp",
        "data_leakage_xgb",
        "data_leakage_linear",
    ],
}

SYNTHCITY_EMITTED_KEY_SUFFIXES = {
    "sanity.data_mismatch": ("score",),
    "sanity.common_rows_proportion": ("score",),
    "sanity.nearest_syn_neighbor_distance": ("mean",),
    "sanity.close_values_probability": ("score",),
    "sanity.distant_values_probability": ("score",),
    "stats.jensenshannon_dist": (
        "marginal",
        "source_table_macro_v2",
        "max_variable_v2",
    ),
    "stats.chi_squared_test": ("marginal",),
    "stats.inv_kl_divergence": ("marginal",),
    "stats.ks_test": ("marginal",),
    "stats.max_mean_discrepancy": ("joint",),
    "stats.wasserstein_dist": ("joint",),
    "stats.prdc": ("precision", "recall", "density", "coverage"),
    "stats.alpha_precision": (
        "delta_precision_alpha_OC",
        "delta_coverage_beta_OC",
        "authenticity_OC",
        "delta_precision_alpha_naive",
        "delta_coverage_beta_naive",
        "authenticity_naive",
    ),
    "performance.linear_model": ("gt", "syn_id", "syn_ood"),
    "performance.mlp": ("gt", "syn_id", "syn_ood"),
    "performance.xgb": ("gt", "syn_id", "syn_ood"),
    "performance.feat_rank_distance": ("corr", "pvalue"),
    "performance.linear_model_augmentation": ("gt", "aug_ood"),
    "performance.mlp_augmentation": ("gt", "aug_ood"),
    "performance.xgb_augmentation": ("gt", "aug_ood"),
    "detection.detection_xgb": ("mean", "raw_auc", "effective_auc_v2"),
    "detection.detection_mlp": ("mean", "raw_auc", "effective_auc_v2"),
    "detection.detection_linear": ("mean", "raw_auc", "effective_auc_v2"),
    "privacy.delta-presence": ("score",),
    "privacy.k-anonymization": ("gt", "syn"),
    "privacy.k-map": ("score",),
    "privacy.distinct l-diversity": ("gt", "syn"),
    "privacy.identifiability_score": (
        "score",
        "score_OC",
        "score_entropy_weighted",
        "score_OC_entropy_weighted",
    ),
    "privacy.DomiasMIA_prior": ("accuracy", "aucroc", "effective_auc_v2"),
    "attack.data_leakage_mlp": (
        "baseline_adjusted_advantage_v2",
        "legacy_accuracy",
        "mean",
    ),
    "attack.data_leakage_xgb": (
        "baseline_adjusted_advantage_v2",
        "legacy_accuracy",
        "mean",
    ),
    "attack.data_leakage_linear": (
        "baseline_adjusted_advantage_v2",
        "legacy_accuracy",
        "mean",
    ),
}


_SYNTHCITY_CATEGORICAL_ATTACK_SUFFIXES = (
    "raw_accuracy",
    "majority_baseline",
    "chance_baseline",
    "balanced_accuracy",
    "macro_f1",
    "classification_score",
    "classification_baseline",
    "baseline_adjusted_advantage_v2",
    "n_eval",
    "uncertainty_v2",
)
_SYNTHCITY_CONTINUOUS_ATTACK_SUFFIXES = (
    "normalized_mae_v2",
    "disclosure_risk_v2",
    "baseline_mae",
    "n_eval",
    "uncertainty_v2",
)


def _declared_key_components(values: Sequence[str] | None, label: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence, not a string")
    components = tuple(values)
    if any(not isinstance(value, str) or not value for value in components):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(components) != len(set(components)):
        raise ValueError(f"{label} must not contain duplicates")
    return components


def emitted_keys_for_synthcity_metrics(
    metric_config,
    *,
    variable_columns: Sequence[str] | None = None,
    attack_target_types: Mapping[str, str] | None = None,
) -> list[str]:
    """Translate selected SynthCity metrics into a context-bound key manifest.

    ``variable_columns`` and ``attack_target_types`` are optional for backward
    compatibility with direct base-key callers. Production evaluation passes
    them from the resolved dataset contract, which makes JSD variable rows and
    typed attribute-inference target rows required identities. A result cannot
    enlarge its own expected set by emitting an arbitrary suffix.
    """
    variable_columns = _declared_key_components(variable_columns, "variable_columns")
    if attack_target_types is not None:
        if not isinstance(attack_target_types, Mapping):
            raise ValueError("attack_target_types must be a mapping")
        if any(
            not isinstance(target, str) or not target or not isinstance(target_type, str)
            for target, target_type in attack_target_types.items()
        ):
            raise ValueError("attack_target_types must map non-empty target names to string types")
        unknown_types = sorted(
            {
                target_type
                for target_type in attack_target_types.values()
                if target_type not in {"categorical", "continuous"}
            }
        )
        if unknown_types:
            raise ValueError(
                "attack_target_types contains unsupported type(s): " + ", ".join(unknown_types)
            )

    emitted_keys = []

    # Canonical objectives are emitted by the adapter, not by a
    # native SynthCity category/name lookup.  Keep this branch explicit so a
    # framework report cannot silently rename an unrelated native metric.
    canonical = {
        CANONICAL_METRIC_MANIFEST["elastic_net_jsd"],
        CANONICAL_METRIC_MANIFEST["mixed_mmd"],
    }
    selected = [str(name) for names in metric_config.values() for name in names]
    if any(name in canonical for name in selected):
        if set(selected) - canonical:
            raise ValueError(
                "Canonical SynthCity HPO metrics must be selected without native metrics"
            )
        return list(selected)

    def add(key: str) -> None:
        if key not in emitted_keys:
            emitted_keys.append(key)

    for category, metric_names in metric_config.items():
        for metric_name in metric_names:
            metric_key = f"{category}.{metric_name}"
            try:
                suffixes = SYNTHCITY_EMITTED_KEY_SUFFIXES[metric_key]
            except KeyError as exc:
                raise ValueError(
                    f"SynthCity metric {metric_key!r} has no emitted-key contract"
                ) from exc
            for suffix in suffixes:
                add(f"{metric_key}.{suffix}")

            if metric_key == "stats.jensenshannon_dist" and variable_columns is not None:
                for column in variable_columns:
                    add(f"{metric_key}.variable_v2.{column}")

            if metric_key.startswith("attack.") and attack_target_types is not None:
                for target, target_type in attack_target_types.items():
                    suffixes_for_target = (
                        _SYNTHCITY_CATEGORICAL_ATTACK_SUFFIXES
                        if target_type == "categorical"
                        else _SYNTHCITY_CONTINUOUS_ATTACK_SUFFIXES
                    )
                    for suffix in suffixes_for_target:
                        add(f"{metric_key}.{suffix}.{target}")
    return emitted_keys


def is_known_synthcity_emitted_key(key: str) -> bool:
    """Return whether a fully qualified SynthCity key belongs to the catalog."""
    if not isinstance(key, str) or not key:
        return False
    for base_key, suffixes in SYNTHCITY_EMITTED_KEY_SUFFIXES.items():
        if key == base_key or any(key == f"{base_key}.{suffix}" for suffix in suffixes):
            return True
        if base_key == "stats.jensenshannon_dist" and key.startswith(f"{base_key}.variable_v2."):
            return bool(key.removeprefix(f"{base_key}.variable_v2."))
        if base_key.startswith("attack."):
            target_suffix = key.removeprefix(f"{base_key}.")
            if "." not in target_suffix:
                continue
            suffix, target = target_suffix.rsplit(".", 1)
            if target and suffix in (
                *_SYNTHCITY_CATEGORICAL_ATTACK_SUFFIXES,
                *_SYNTHCITY_CONTINUOUS_ATTACK_SUFFIXES,
            ):
                return True
    return False


#: synthcity's own "category" (sanity/stats/.../attack) rolled up to utility/privacy.
SYNTHCITY_CATEGORY_TO_TYPE = {
    "sanity": "utility",
    "stats": "utility",
    "performance": "utility",
    "detection": "privacy",
    "privacy": "privacy",
    "attack": "privacy",
}

#: ``stats.alpha_precision``'s "_naive" sub-metrics (delta_precision_alpha_naive,
#: delta_coverage_beta_naive, authenticity_naive) duplicate the "_OC"
#: (OneClass-embedding) sub-metrics' exact same 3 quantities computed in a
#: different (raw min-max normalized) feature space -- confirmed via
#: ``synthcity.metrics.eval_statistical.AlphaPrecision._normalize_covariates``'s
#: own docstring ("This is an internal method to replicate the old, naive
#: method for evaluating AlphaPrecision"). Left uncorrected this metric alone
#: silently contributes 6 utility columns instead of 3 genuinely distinct ones,
#: so the "_naive" duplicates are excluded from the combined table (see
#: combine.py's ``_synthcity_frames``) -- the OC variants are kept since they're
#: the currently-preferred/default computation.
SYNTHCITY_REDUNDANT_SUBMETRIC_SUFFIXES = ("_naive",)


def is_redundant_synthcity_submetric(metric_key: str) -> bool:
    """Whether a synthcity result column (e.g.
    ``"stats.alpha_precision.authenticity_naive"``) is a known-redundant
    duplicate sub-metric that should be excluded from the combined evaluation
    table -- see ``SYNTHCITY_REDUNDANT_SUBMETRIC_SUFFIXES``.
    """
    return metric_key.endswith(SYNTHCITY_REDUNDANT_SUBMETRIC_SUFFIXES)


# ---------------------------------------------------------------------------
# syntheval
# ---------------------------------------------------------------------------

#: Full evaluation preset (mirrors the hepatitis notebook's `complete_eval`).
#: Includes the two custom fairness metrics (equal_opportunity, equalized_odds)
#: added to this repo's syntheval fork; these get re-tagged to framework="custom"
#: downstream (see SYNTHEVAL_CUSTOM_FAIRNESS_KEYS below).
SYNTHEVAL_PRESET = {
    "dwm": {},
    "pca": {"preprocess": "std"},
    "cio": {"confidence": 95},
    "corr_diff": {"mixed_corr": True},
    "mi_diff": {},
    "ks_test": {"sig_lvl": 0.05, "n_perms": 1000},
    "h_dist": {},
    "p_mse": {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
    "q_mse": {"num_quants": 10, "cat_mse": False},
    "auroc_diff": {"model": "log_reg", "num_boots": 1},
    "cls_acc": {
        "cls_models": ["rf", "adaboost", "svm", "logreg"],
        "F1_type": "micro",
        "k_folds": 5,
        "full_output": False,
    },
    "nndr": {},
    "nnaa": {"n_resample": 30},
    "dcr": {},
    "hit_rate": {"thres_percent": 0.0333},
    "eps_risk": {},
    "mia": {"num_eval_iter": 5},
    "att_discl": {"numerical_dist_thresh": 1 / 30},
    "statistical_parity": {"positive_class": 1, "folds": 5, "full_output": True},
    "equalized_odds": {"positive_class": 1, "folds": 5, "full_output": True},
    "equal_opportunity": {"positive_class": 1, "folds": 5, "full_output": True},
}

#: The 3 fairness metrics above whose "positive_class" preset param is
#: config-driven (see synthdata.evaluation.syntheval_eval.build_preset) --
#: NOT applied to the separate binary_target pass, whose collapsed target is
#: already normalized to 1=positive/0=negative by construction.
FAIRNESS_METRICS_WITH_POSITIVE_CLASS = frozenset(
    {"statistical_parity", "equalized_odds", "equal_opportunity"}
)

SYNTHEVAL_METRIC_TYPE = {
    "dwm": "utility",
    "pca": "utility",
    "cio": "utility",
    "corr_diff": "utility",
    "mi_diff": "utility",
    "ks_test": "utility",
    "h_dist": "utility",
    "p_mse": "utility",
    "q_mse": "utility",
    "auroc_diff": "utility",
    "cls_acc": "utility",
    "nndr": "privacy",
    "nnaa": "privacy",
    "dcr": "privacy",
    "hit_rate": "privacy",
    "eps_risk": "privacy",
    "mia": "privacy",
    "att_discl": "privacy",
    "statistical_parity": "fairness",
    "equalized_odds": "fairness",
    "equal_opportunity": "fairness",
}

#: Some SynthEval metrics' RESULT COLUMN names differ from their preset/
#: selection key, or add extra per-(target_var[, protected_attribute])
#: breakdown columns when `full_output: True` is set (as SYNTHEVAL_PRESET
#: does for the 3 fairness metrics) -- none of these are literal keys in
#: SYNTHEVAL_METRIC_TYPE above, so a plain dict lookup would silently
#: misclassify them as "utility" (the fallback default). See:
#: - metric_auroc_difference.py: primary column is "auroc" (not "auroc_diff"),
#:   per-target sub-columns are "auroc_<target_var>".
#: - metric_statistical_parity.py / metric_equal_opportunity.py /
#:   metric_equalized_odds.py: per-(target_var, protected_attribute)
#:   sub-columns are "sp_"/"eo_"/"eqo_" + "<target_var>_<protected_attribute>".
_SYNTHEVAL_SUBMETRIC_PREFIX_TYPE = {
    "auroc_": "utility",
    "sp_": "fairness",
    "eo_": "fairness",
    "eqo_": "fairness",
}

#: Same idea, for which of these prefixed sub-columns belong to the custom
#: (fork-only) fairness metrics -- see SYNTHEVAL_CUSTOM_FAIRNESS_KEYS below.
_SYNTHEVAL_CUSTOM_SUBMETRIC_PREFIXES = ("eo_", "eqo_")


def classify_syntheval_metric(name: str) -> str:
    """Classify an emitted result key through the versioned contract registry.

    The custom-fairness fork metrics are evaluated through SynthEval but are
    registered under the root ``custom`` framework. Unknown emitted keys fail
    closed instead of being silently treated as utility.
    """
    if name in SYNTHEVAL_EMITTED_KEY_TYPE:
        return SYNTHEVAL_EMITTED_KEY_TYPE[name]

    from synthdata.evaluation.metric_contracts import (
        DEFAULT_METRIC_CONTRACT_REGISTRY,
        UnknownMetricContractError,
    )

    for framework in ("syntheval", "custom"):
        try:
            return DEFAULT_METRIC_CONTRACT_REGISTRY.resolve(
                framework=framework,
                emitted_key=name,
            ).semantic_family
        except UnknownMetricContractError:
            continue
    raise UnknownMetricContractError(f"Unknown emitted SynthEval metric key: {name!r}")


def is_custom_syntheval_metric(name: str) -> bool:
    """Whether a SynthEval RESULT COLUMN name belongs to one of this repo's
    fork-only fairness metrics (SYNTHEVAL_CUSTOM_FAIRNESS_KEYS), including
    their full_output=True per-(target_var, protected_attribute) sub-columns.
    """
    return name in SYNTHEVAL_CUSTOM_FAIRNESS_KEYS or name.startswith(
        _SYNTHEVAL_CUSTOM_SUBMETRIC_PREFIXES
    )


#: Custom additions to the syntheval fork (see submodules/syntheval fairness/): these
#: are computed via SynthEval's `evaluate()` call but re-tagged framework="custom"
#: in the combined evaluation table, since they are not part of upstream SynthEval.
SYNTHEVAL_CUSTOM_FAIRNESS_KEYS = {"equalized_odds", "equal_opportunity"}

# Exact identities emitted by multiclass one-vs-rest implementations. Binary
# passes continue to use identities in SYNTHEVAL_PRESET_EMITTED_KEYS.
SYNTHEVAL_MULTICLASS_EMITTED_KEYS = {
    "auroc_diff": ("auroc_macro_ovr_v3",),
    "statistical_parity": ("statistical_parity_macro_ovr_v1",),
    "equalized_odds": ("equalized_odds_macro_ovr_v1",),
    "equal_opportunity": ("equal_opportunity_macro_ovr_v1",),
}

SYNTHEVAL_EMITTED_KEY_TYPE = {
    "auroc_macro_ovr_v3": "utility",
    "statistical_parity_macro_ovr_v1": "fairness",
    "equalized_odds_macro_ovr_v1": "fairness",
    "equal_opportunity_macro_ovr_v1": "fairness",
}

# Directions match the registered metric contracts. OvR AUROC agreement is
# transformed with one_minus_absolute, so larger (closer to zero) is preferable;
# lower fairness disparity is preferable.
SYNTHEVAL_EMITTED_KEY_DIRECTION = {
    "auroc_macro_ovr_v3": "maximize",
    "statistical_parity_macro_ovr_v1": "minimize",
    "equalized_odds_macro_ovr_v1": "minimize",
    "equal_opportunity_macro_ovr_v1": "minimize",
}
SYNTHEVAL_CUSTOM_FAIRNESS_KEYS.update(
    {
        "equalized_odds_macro_ovr_v1",
        "equal_opportunity_macro_ovr_v1",
    }
)

# SynthEval's preset names identify evaluator classes, not the normalized
# metric identities written to benchmark results. Keep this translation
# explicit so selection completeness is checked against the rows that are
# actually emitted. Optional holdout rows are added only when a holdout table
# is part of the evaluation context.
SYNTHEVAL_PRESET_EMITTED_KEYS = {
    "dwm": ("avg_dwm_diff",),
    "pca": ("pca_eigval_diff", "pca_eigvec_ang"),
    "cio": ("avg_cio",),
    "corr_diff": ("corr_mat_diff",),
    "mi_diff": ("mutual_inf_diff",),
    "ks_test": ("ks_tvd_stat", "frac_ks_sigs"),
    "h_dist": ("avg_h_dist",),
    "p_mse": ("avg_pMSE",),
    "q_mse": ("avg_qMSE",),
    "auroc_diff": ("auroc",),
    "cls_acc": ("avg_F1_diff",),
    "nndr": ("avg_nndr", "priv_loss_nndr"),
    "nnaa": ("nnaa", "priv_loss_nnaa"),
    "dcr": ("median_DCR",),
    "hit_rate": ("hit_rate",),
    "eps_risk": ("eps_identif_risk", "priv_loss_eps"),
    "mia": ("mia_recall", "mia_precision"),
    "att_discl": ("att_discl_risk",),
    "statistical_parity": ("statistical_parity",),
    "equalized_odds": ("equalized_odds",),
    "equal_opportunity": ("equal_opportunity",),
}

# Structured SynthEval execution validates the versioned rows produced by the
# repaired metrics. Metrics without a v2 normalizer continue to use their
# historical normalized identities until their dedicated repair lands.
SYNTHEVAL_V2_PRESET_EMITTED_KEYS = {
    "corr_diff": ("corr_mat_diff_v2",),
    "mi_diff": ("mutual_inf_diff_v2",),
    "ks_test": ("ks_tvd_stat_v2", "frac_ks_sigs_v2"),
    "h_dist": ("avg_h_dist_v2",),
    "p_mse": ("avg_pMSE_v2",),
    "q_mse": ("avg_qMSE_v2",),
    "auroc_diff": ("auroc_v2",),
}


def _syntheval_key_component(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("SynthEval key components must be non-empty strings")
    return value.replace(" ", "_").lower()


def _syntheval_qualified_output_keys(
    preset_name: str,
    parameters: Mapping,
    *,
    target_columns: Sequence[str] | None,
    protected_columns: Sequence[str] | None,
    multiclass: bool = False,
) -> tuple[str, ...]:
    if not isinstance(parameters, Mapping):
        raise ValueError(f"SynthEval preset {preset_name!r} parameters must be a mapping")
    if not parameters.get("full_output", False):
        return ()
    targets = tuple(target_columns or ())
    protected = tuple(protected_columns or ())
    if preset_name == "auroc_diff":
        suffix = "macro_ovr_v3" if multiclass else "v2"
        return tuple(f"auroc_{_syntheval_key_component(target)}_{suffix}" for target in targets)
    fairness_prefix = {
        "statistical_parity": "sp_",
        "equalized_odds": "eqo_",
        "equal_opportunity": "eo_",
    }.get(preset_name)
    if fairness_prefix is None:
        return ()
    if multiclass:
        # Classwise OvR diagnostics include observed class values, which are
        # intentionally not inferred from the declared target column. Their
        # aggregate identity remains required; the executor accounts for the
        # classwise diagnostic prefixes as optional outputs.
        return ()
    return tuple(
        f"{fairness_prefix}{_syntheval_key_component(target)}_{protected_attribute}"
        for target in targets
        for protected_attribute in protected
    )


def syntheval_execution_manifest(
    preset: dict,
    *,
    include_holdout_outputs: bool,
    target_columns: Sequence[str] | None = None,
    protected_columns: Sequence[str] | None = None,
    target_is_binary: bool | None = None,
) -> dict[str, tuple[str, ...]]:
    """Build static per-method output expectations for structured SynthEval runs.

    The manifest is resolved from the selected preset and declared evaluation
    context before evaluation. Binary qualified diagnostics are required only
    when the selected preset explicitly requests full output and the
    corresponding target/protected columns are declared. Multiclass fairness
    diagnostics include observed classes, so only aggregate keys are required.
    """
    manifest = {}
    for preset_name, parameters in preset.items():
        if preset_name == "cls_acc":
            score_type = parameters.get("F1_type", "macro")
            metric_name = (
                "avg_balanced_accuracy_diff_v2"
                if score_type == "balanced_accuracy"
                else "avg_macro_F1_diff_v2"
            )
            expected = [metric_name]
            if include_holdout_outputs:
                expected.append(f"{metric_name}_hout")
        elif preset_name in SYNTHEVAL_V2_PRESET_EMITTED_KEYS:
            expected = list(SYNTHEVAL_V2_PRESET_EMITTED_KEYS[preset_name])
        else:
            try:
                expected = list(SYNTHEVAL_PRESET_EMITTED_KEYS[preset_name])
            except KeyError as exc:
                raise ValueError(
                    f"SynthEval preset {preset_name!r} has no execution manifest"
                ) from exc
            if include_holdout_outputs:
                expected.extend(SYNTHEVAL_HOLDOUT_EMITTED_KEYS.get(preset_name, ()))
        if target_is_binary is False:
            expected = list(SYNTHEVAL_MULTICLASS_EMITTED_KEYS.get(preset_name, expected))
        expected.extend(
            _syntheval_qualified_output_keys(
                preset_name,
                parameters,
                target_columns=target_columns,
                protected_columns=protected_columns,
                multiclass=target_is_binary is False,
            )
        )
        manifest[preset_name] = tuple(dict.fromkeys(expected))
    return manifest


def syntheval_execution_keys_by_framework(
    execution_manifest: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    """Group static execution identities by root framework ownership."""
    expected = {"syntheval": [], "custom": []}
    for emitted_keys in execution_manifest.values():
        for emitted_key in emitted_keys:
            framework = syntheval_framework_for_emitted_key(emitted_key)
            if emitted_key not in expected[framework]:
                expected[framework].append(emitted_key)
    return expected


SYNTHEVAL_HOLDOUT_EMITTED_KEYS = {
    "cls_acc": ("avg_F1_diff_hout",),
    "nndr": ("priv_loss_nndr",),
    "nnaa": ("priv_loss_nnaa",),
    "eps_risk": ("priv_loss_eps",),
}


def syntheval_framework_for_emitted_key(name: str) -> str:
    """Return the root framework label for a normalized SynthEval key."""
    if name in SYNTHEVAL_CUSTOM_FAIRNESS_KEYS or name.startswith(("eo_", "eqo_")):
        return "custom"
    return "syntheval"


def resolve_syntheval_emitted_keys(
    selection_cfg,
    *,
    include_holdout_outputs: bool,
) -> dict[str, list[str]]:
    """Resolve selected preset names into normalized keys grouped by framework."""
    selected = resolve_selection(
        selection_cfg.enabled,
        selection_cfg.categories,
        selection_cfg.metrics,
        list(SYNTHEVAL_PRESET),
        SYNTHEVAL_METRIC_TYPE,
    )
    return emitted_keys_for_syntheval_presets(
        selected,
        include_holdout_outputs=include_holdout_outputs,
    )


def emitted_keys_for_syntheval_presets(
    preset_names,
    *,
    include_holdout_outputs: bool,
) -> dict[str, list[str]]:
    """Translate preset names through the authoritative execution manifest."""
    preset = {}
    for preset_name in preset_names:
        try:
            preset[preset_name] = dict(SYNTHEVAL_PRESET[preset_name])
        except KeyError as exc:
            raise ValueError(
                f"SynthEval preset {preset_name!r} has no normalized emitted-key contract"
            ) from exc
    manifest = syntheval_execution_manifest(
        preset,
        include_holdout_outputs=include_holdout_outputs,
    )
    expected = {"syntheval": [], "custom": []}
    for emitted_keys in manifest.values():
        for emitted_key in emitted_keys:
            framework = syntheval_framework_for_emitted_key(emitted_key)
            if emitted_key not in expected[framework]:
                expected[framework].append(emitted_key)
    return expected


# ---------------------------------------------------------------------------
# custom (log disparity + the syntheval-fork-only fairness metrics above)
# ---------------------------------------------------------------------------

#: log-disparity's own summary metrics, and whether lower values are "better"
#: (used to orient them for ranking: True => minimize, False => maximize).
#:
#: NOTE: ``log_disparity_median_abs`` is deliberately NOT listed here (unlike
#: ``build_log_disparity_summary_table``'s raw output, which still includes
#: it) -- it's computed from the exact same per-subgroup value array as
#: ``log_disparity_mean_abs`` (see metric_log_disparity.py's ``summary_stats``
#: construction), so counting both as independent ranked metrics would double-
#: count one underlying signal. It still shows up in the combined table's raw
#: columns (informational) via ``_log_disparity_frames``, just excluded from
#: the fairness rank sum via this dict's key set.
LOG_DISPARITY_METRICS = {
    "log_disparity_mean_abs": True,
    "log_disparity_share_significant": True,
}

CUSTOM_METRIC_TYPE = {
    **{name: "fairness" for name in LOG_DISPARITY_METRICS},
    **{name: "fairness" for name in SYNTHEVAL_CUSTOM_FAIRNESS_KEYS},
}


def resolve_selection(
    enabled: bool,
    categories: list | None,
    metrics: list | None,
    all_metric_names: list,
    metric_type_map: dict,
) -> list:
    """Resolve partial-selection config into a concrete list of metric names.

    Precedence: disabled -> [] (callers must skip execution and validation);
    explicit `metrics` (validated against `all_metric_names`) -> that list ;
    `categories` (utility/privacy/fairness, matched via `metric_type_map`) ->
    matching metrics ; neither given -> all.
    """
    if not enabled:
        return []
    if metrics:
        unknown = [m for m in metrics if m not in all_metric_names]
        if unknown:
            raise ValueError(
                f"Unknown metric name(s) {unknown}; available: {sorted(all_metric_names)}"
            )
        return list(metrics)
    if categories:
        return [m for m in all_metric_names if metric_type_map.get(m) in categories]
    return list(all_metric_names)
