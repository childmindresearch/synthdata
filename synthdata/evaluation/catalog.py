"""Static metric catalogs: which metrics exist per framework, and their utility/
privacy/fairness type, used to resolve partial selection (by category or by
explicit metric name) in :mod:`synthdata.evaluation.combine`.
"""

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
        "detection_gmm",
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
    "attacks": [
        "data_leakage_mlp",
        "data_leakage_xgb",
        "data_leakage_linear",
    ],
}

#: synthcity's own "category" (sanity/stats/.../attacks) rolled up to utility/privacy.
SYNTHCITY_CATEGORY_TO_TYPE = {
    "sanity": "utility",
    "stats": "utility",
    "performance": "utility",
    "detection": "privacy",
    "privacy": "privacy",
    "attacks": "privacy",
}

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

#: Custom additions to the syntheval fork (see submodules/syntheval fairness/): these
#: are computed via SynthEval's `evaluate()` call but re-tagged framework="custom"
#: in the combined evaluation table, since they are not part of upstream SynthEval.
SYNTHEVAL_CUSTOM_FAIRNESS_KEYS = {"equalized_odds", "equal_opportunity"}

# ---------------------------------------------------------------------------
# custom (log disparity + the syntheval-fork-only fairness metrics above)
# ---------------------------------------------------------------------------

#: log-disparity's own summary metrics, and whether lower values are "better"
#: (used to orient them for ranking: True => minimize, False => maximize).
LOG_DISPARITY_METRICS = {
    "log_disparity_mean_abs": True,
    "log_disparity_median_abs": True,
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

    Precedence: disabled -> [] ; explicit `metrics` (validated against
    `all_metric_names`) -> that list ; `categories` (utility/privacy/fairness,
    matched via `metric_type_map`) -> matching metrics ; neither given -> all.
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
