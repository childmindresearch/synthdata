# Verification of generators and metrics

Every generator and metric the pipeline runs, checked against its paper or upstream reference implementation. Each row ends in one of four states:

- **Verified**: matches the reference, by code reading or a test against an independent computation.
- **Fixed**: our code or fork had a bug against the reference; the fix and a regression test are in the repo.
- **Deviation**: differs from the reference on purpose; the reason is given.
- **Open**: a known upstream problem we have not fixed; the impact on results is stated so you can judge how much to trust the number.

IDs such as SC-05 or SE-12 refer to the evaluation audit of 2026-10-07. Reference tests live in `tests/unit/test_synthcity_reference_metrics.py`, `tests/unit/test_syntheval_reference_metrics.py` and `tests/unit/test_log_disparity_metrics.py`; the end-to-end ordering checks ("copy of train" vs "column shuffle") are in `tests/integration/test_metric_known_answers.py`.

## Versions checked

| Component | Version | Relation to upstream |
| --- | --- | --- |
| synthcity | fork `alperkent-cmi/synthcity` | upstream `vanderschaarlab/synthcity@23f322f` plus metric fixes listed below, a joblib-free metric loop and quieter plugin loading. Generator plugins are unmodified. |
| SynthEval | fork `alperkent-cmi/syntheval` | upstream `schneiderkamplab/syntheval@42a3d34` plus two fairness metrics, parallel execution and the fixes listed below. |
| TabPFN | `tabpfn==8.0.8` | PyPI release, unmodified; three runtime patches in `synthdata/generation/tabpfn_backend.py` (below). |
| tabpfn-extensions | git `6ed13f6` (locked in `uv.lock`) | unsupervised generation API, unmodified. |
| TabPFGen | `tabpfgen==0.1.4` | PyPI release; the custom variant subclasses it. |
| pgmpy | `0.1.26` | used by synthcity's Bayesian network plugin. |
| Anonymeter | `anonymeter>=1.1.0` (1.1.0 checked) | PyPI release, unmodified. |

## Generators

| Generator | Reference | Status | Notes |
| --- | --- | --- | --- |
| ctgan | Xu et al. 2019; synthcity plugin | Verified | Unmodified plugin; HPO over synthcity's own search space. |
| tvae | Xu et al. 2019; synthcity plugin | Verified | Unmodified plugin. |
| rtvae | Akrami et al. 2020; synthcity plugin | Verified | Unmodified plugin. |
| adsgan | Yoon et al. 2020; synthcity plugin | Verified | Unmodified plugin. |
| pategan | Jordon et al. 2019; synthcity plugin | Deviation | Unmodified plugin, but HPO caps its iterations (`model_iter_caps: {pategan: 5}` for hepatitis, 30 for LORIS) because each iteration trains a teacher ensemble. Its differential-privacy budget is the plugin default (epsilon 1); the pipeline does not report the privacy actually spent. |
| ddpm | Kotelnikov et al. 2023 (TabDDPM); synthcity plugin | Verified | Unmodified plugin; fork fixes only a device error when the benchmark serialises it. |
| bayesian_network | synthcity plugin on pgmpy | Deviation | HPO searches `tree_search` structure learning only: pgmpy's PC and hill climbing are not deterministic even with every seed fixed, which broke reproducibility. |
| marginals baseline | synthcity `marginal_distributions` | Verified | Independent per-column sampling; a reference row that is never recommended. |
| train-copy baseline | none (sanity row) | Verified | Train rows sampled without replacement; scores worst on privacy in the integration tests. |
| tabpfn standard | Hollmann et al. 2025; tabpfn-extensions unsupervised API | Deviation | Generates features with TabPFN's unsupervised model, then labels each row with a fresh `TabPFNClassifier.predict` (the most likely class). That gives synthetic labels less noise than real ones, which can make TSTR look better than a generator that samples labels; TabPFGen's reference code labels the same way. |
| tabpfn custom | as above | Verified | Target modelled jointly as one more categorical column. |
| tabpfn runtime patches | tabpfn-extensions `6ed13f6` | Deviation | (1) honour the schema's categorical list instead of re-inferring it from cardinality; (2) drop NaNs before counting classes so the classifier/regressor choice agrees between fit and sample; (3) replace a non-finite sample from `BarDistribution.icdf` underflow with the distribution mean and log a warning. Each fixes a crash or a silent relabelling; none is filed upstream yet. |
| tabpfn replicates | | Open | TabPFN may pin its own seed, so seed replicates can be identical; the pipeline logs a warning when they are. To be confirmed on the GPU run. |
| tabpfgen standard | Haan 2025, `sebhaan/TabPFGen` | Deviation | Upstream `generate_classification` with its default `balance_classes=True`: the output is class-balanced, not the training prevalence, and returns `n_samples // n_classes` rows per class. Labels are TabPFN's argmax, as upstream. Class-prior matching (below) resamples it to the training prevalence. |
| tabpfgen custom | as above | Deviation | Subclass that labels each sample by its nearest scaled training row after SGLD (TabPFN relabelling collapsed to one class on small data), then over-generates and subsamples to the training class proportions. Assumes targets coded 0..k-1, which both shipped configs satisfy. |

## Imputation

| Method | Reference | Status | Notes |
| --- | --- | --- | --- |
| missforest (default) | Stekhoven & Buehlmann, *Bioinformatics* 2012; scikit-learn `IterativeImputer` + `RandomForestRegressor` | Deviation | scikit-learn's documented MissForest equivalent. Differences from the R package: nominal columns with more than two categories are one-hot encoded and the most likely category is kept (MissForest uses classification forests), binary and ordinal columns are imputed as codes and rounded, and the defaults are 50 trees and 5 rounds instead of 100 and 10. hyperimpute's `missforest` plugin was not used: it refits on whatever it transforms, so it cannot fit on train and fill the holdout. |
| simple | scikit-learn `SimpleImputer` (median / most frequent) | Verified | Unit test checks the fill equals the fit rows' median and mode and that transformed rows do not move it. Used by CI. |
| two-phase fit | refit after selection, as scikit-learn `GridSearchCV(refit=True)` | Verified | HPO fits and scores on data imputed by an imputer fitted on train minus tuning; the final models and the holdout use one refitted on all of train with the same settings and seed. `imputation_drift.csv` reports how much the two fits disagree on the cells both fill (Wasserstein / std for continuous, total variation for categorical). |
| missing indicators and re-masking | Sperrin et al., *Stat Med* 2020; scikit-learn `MissingIndicator` | Verified | Indicators are computed from the raw values for columns missing in at least 5% of the train rows outside tuning. Evaluation scores the filled data with indicators, because SynthEval and synthcity metrics need complete data; the indicators make the missingness pattern part of what fidelity metrics compare. `released/<model>.csv` has values blanked where the synthetic indicator is 1. |
| indicator-only columns | Nowok, Raab & Dibben, *J Stat Softw* 2016; `synthpop::syn.cart` and `syn.smooth` | Deviation | Columns missing in at least 80% of train-minus-tuning rows (`indicator_only_fraction`) are neither imputed nor generated; generators model only their indicator. In `released/`, rows marked recorded get a value by synthpop's CART method: a tree (`minbucket` 5) fitted on the real train rows where the value was observed, a random donor from the synthetic row's leaf, and density smoothing for continuous columns. Deviation: Silverman's bandwidth instead of R's Sheather-Jones. Target, stratification and role columns are never made indicator-only; `keep_values_columns` exempts others. Evaluation does not see these values; `released/<model>_recorded_values.csv` compares recorded values with the real observed ones (KS statistic or total variation). |

## HPO objective

| Component | Reference | Status | Notes |
| --- | --- | --- | --- |
| TSTR macro-F1 objective | Esteban et al. 2017 (TSTR); Kotelnikov et al., ICML 2023 (TabDDPM tunes every generator on validation ML efficiency) | Deviation | Custom glue around library code (`synthdata/evaluation/tstr.py`): a fixed `XGBClassifier` (80 trees, depth 4, learning rate 0.08, from the feat branch) trained on the candidate, scored on the real tuning rows with scikit-learn `f1_score(average="macro")`, averaged over `hpo.tstr_seeds` seeds. Nominal columns are one-hot encoded, categories seen only in the real rows get no column. Unit tests check it learns a planted signal, is deterministic, and gives 0 F1 to a class the candidate lacks. Privacy is not in the objective; evaluation reports it. |
| macro AUPRC (logged) | Saito & Rehmsmeier 2015; scikit-learn `average_precision_score` | Verified | Mean of one-vs-rest average precision over the classes present in tuning. Logged on every trial; `hpo.objective: tstr_macro_auprc` optimizes it instead. |
| TRTR ceiling (logged) | same classifier on real search-train rows | Verified | Logged once per run and stored on every trial as `trtr_macro_f1` / `trtr_macro_auprc`. |
| class-prior matching | none (sanity rule) | Deviation | Every candidate and every saved synthetic dataset is resampled to the real train class shares, keeping its size (without replacement while a class has enough rows). Stops a generator from raising macro-F1 by rebalancing classes. A class the generator never produced cannot be added; the screens reject such trials. |
| screens as constraints | Optuna `TPESampler(constraints_func=...)`; Watanabe & Hutter, IJCAI 2023 | Verified | Copies (exact-match rate vs search-train at most the tuning rows' rate + 0.02), every target class present, coverage of train categories with at least 1% frequency at least 0.9 per categorical column, numeric values outside the train range at most 1 point above the tuning rows' share (fresh real rows fall outside too, about 2/n per column). Infeasible trials keep their value but can't be best; crashed trials are marked failed. If no trial is feasible the model keeps its default settings. |

## Column roles

`data.quasi_identifier_columns`, `data.sensitive_columns` and `data.protected_columns` follow statistical disclosure control usage (Hundepool et al., *Statistical Disclosure Control*, 2012; sdcMicro). Quasi-identifiers are public, linkable attributes (age, sex, region); sensitive columns are the secrets an attacker tries to infer (a diagnosis, income); protected columns define fairness groups. Before step 8 one `sensitive_columns` list fed all three roles, so attribute-inference metrics tried to infer sex and age. The config now fails if `sensitive_columns` is set without `protected_columns`, so a config written for the old meaning cannot carry over silently.

| Consumer | Role it receives | Notes |
| --- | --- | --- |
| synthcity `sensitive_features` (data_leakage, distinct l-diversity) | sensitive | data_leakage predicts each sensitive column from all other columns; l-diversity counts distinct sensitive values per cluster. |
| synthcity k-anonymization, k-map, delta-presence | every non-sensitive column | Deviation: synthcity has no quasi-identifier argument and treats every non-sensitive column as one, a stronger adversary than the declared QIs. |
| synthcity `fairness_column` | first protected column | Used only by synthcity's augmentation benchmark. |
| SynthEval att_discl | sensitive | Each metric gets its own `AnalysisConfig`; the attacker knows every other column, not only the QIs (stronger than the QI model). |
| SynthEval statistical_parity, equal_opportunity, equalized_odds | protected | statistical_parity uses only binary protected columns, as upstream. |
| log disparity | protected (`evaluation.log_disparity.protected_columns` overrides) | |
| imputation benchmark | none of the three | Role columns are never masked or scored, as sensitive columns were before. |
| quasi-identifiers | recorded in the dataset manifest | No current metric takes them; the planned Anonymeter linkability and inference attacks (step 8, PR 6) will use them as the attacker's auxiliary columns. |

## synthcity metrics

Scored against the train split (members), except performance and DOMIAS, which use the held-out split. Detection AUC is filed as utility (fidelity), not privacy.

| Metric | Reference | Status | Notes |
| --- | --- | --- | --- |
| inv_kl_divergence | KL divergence of marginals | Fixed (SC-01) | Categories were paired by frequency rank, not by value, so a 70/30 vs 30/70 split scored like identical data. Now matches the hand-computed KL. |
| chi_squared_test | Pearson 1900; `scipy.stats.chisquare` | Fixed (SC-01, SC-04) | Ran on proportions, so the p-value was near 1 for any data; now runs on counts and matches scipy. NaN or an error still becomes p = 0 (Open). |
| alpha_precision: authenticity | Alaa et al. 2022 | Fixed (SC-02) | Real radii were indexed by synthetic-row index; a generator collapsed onto one real record scored about 0.7, now 0. Matches a brute-force implementation of the paper's definition. Note: this also departs from the authors' own released code, which has the same indexing. |
| alpha_precision: alpha-precision, beta-recall | Alaa et al. 2022 | Verified | `_naive` duplicates are excluded from the ranking. |
| feat_rank_distance | synthcity | Fixed (earlier, fork `492d5f8`) | Runs on xgboost 3, ranks the right axis, direction "maximize"; its p-value column is reported but not ranked. |
| jensenshannon_dist | Lin 1991; `scipy.spatial.distance.jensenshannon` | Open (SC-11) | Natural-log JSD (maximum about 0.83), bins span real and synthetic ranges so outliers lower the score. Monotone in the right direction; values are not comparable with base-2 JSD elsewhere. |
| ks_test | `scipy.stats.ks_2samp` | Open (SC-11) | Also runs on label-encoded nominal columns, where the result depends on code order. |
| max_mean_discrepancy | Gretton et al. 2012 | Open (SC-05) | RBF kernel with gamma fixed at 1 on unscaled columns, so wide-range columns contribute almost nothing and narrow ones dominate. It still separates shifted from unshifted data. |
| wasserstein_dist, prdc | synthcity | Open (SC-05) | Inputs are not scaled, so wide-range columns dominate. |
| performance (linear, mlp, xgb, augmentation) | train on synthetic, test on real | Open (SC-07) | Train-on-synthetic, test-on-held-out wiring verified. Upstream's "gt" comparison model and the XGB depth parameter have known bugs; the "gt" column is the same for every model, so it does not move the ranking. For 3+ classes the AUROC is micro-averaged with no parameter to change it, so the majority class dominates; holdout TSTR (custom, below) gives the macro view. |
| detection (xgb, mlp, gmm, linear) | real-vs-synthetic classifier AUC | Verified | Stratified, seeded cross-validation. Small folds bias the AUC slightly upward (SC-10). |
| sanity (common rows, NN distance, close/distant values) | synthcity | Open (SC-17) | Distances are min-max normalised within each run, so they rank models only roughly. |
| identifiability_score | Yoon et al. 2020 | Open (SC-09) | A column constant in the real data gets weight 1e16 and dominates the distance (fixed for SynthEval's version below, not yet here). |
| k-anonymization, distinct l-diversity, k-map, delta-presence | Sweeney 2002; Machanavajjhala 2007 | Open (SC-13) | Computed on the synthetic table; below about 20 rows they return placeholder values (999 or 0). Treat as rough indicators on small data. |
| DomiasMIA_prior | van Breugel et al. 2023 | Open (SC-08) | The prior density includes members, which biases toward "no leakage"; reported as max(AUC, 1-AUC). |
| data_leakage (mlp, xgb, linear) | attribute-inference attack | Open (SC-12) | The attacker's baseline comes from the candidate's own synthetic data, so the baseline moves with the model being judged. |
| Subsampling | | Deviation (LK-09) | synthcity scores `min(len(real), len(synthetic))` rows, drawn once with seed 0. Seed replicates (`generation.n_replicates`) give the spread instead. |
| Result cache | | Fixed | synthcity cached metric results on disk keyed by data and metric name, not code, so a metric fix could be masked by an old result. The pipeline now runs with `use_cache=False`. |

## SynthEval metrics

| Metric | Reference | Status | Notes |
| --- | --- | --- | --- |
| h_dist (Hellinger) | Scott 1979 bins | Fixed (SE-02) | The bin-width formula was inverted, so every [0,1]-scaled numeric column had one bin and a distance of exactly 0; categorical columns were binned over each table's own range. Now matches `numpy.histogram_bin_edges(bins="scott")` on shared bins. |
| eps_identif_risk, priv_loss_eps | Yoon et al. 2020 | Fixed (SE-05) | The holdout risk was divided by the train size, inflating `priv_loss_eps`; a constant column got weight 1e16. Both fixed. The loss still depends on the train/holdout size ratio (SE-06, Open). |
| nnaa | Yale et al. 2020 | Fixed (SE-04) | Normalised score was 1-AA, so a copy of train (AA = 0) scored best; now 1-2\|AA-0.5\|. Formula of AA itself verified. |
| priv_loss_nnaa | Yale et al. 2020 | Verified | Holdout AA minus train AA. |
| avg_nndr, priv_loss_nndr | Yale et al. 2020 | Verified | Higher NNDR is more private, as SynthEval orients it. |
| statistical_parity | demographic parity difference (fairlearn) | Fixed (SE-03) | Signed gaps were averaged across protected attributes and could cancel; now absolute. Computed on a classifier trained and tested on synthetic data only, with unshuffled folds (SE-14, Open). |
| equal_opportunity (fork) | Hardt et al. 2016 | Fixed (SE-03) | Same cancellation fix. |
| equalized_odds (fork) | Hardt et al. 2016 | Deviation | Mean of the absolute TPR and FPR gaps (fairlearn `agg="mean"`); fairlearn's default reports the larger of the two. |
| p_mse | Snoke et al. 2018 | Open (SE-12) | Formula verified; nominal codes enter the logistic model as numbers, which understates pMSE. |
| corr_diff, mi_diff, ks_test, cio, dwm, pca, q_mse | SynthEval | Verified | Cramér's V, NMI and KS verified in the audit. `corr_diff` drops pairs that are invalid only in synthetic data (SE-11, Open). |
| auroc_diff, cls_acc | SynthEval; sklearn `roc_auc_score(multi_class="ovr", average="macro")` | Verified | SynthEval runs `auroc_diff` and the subgroup gap metrics only on a 2-class target. For 3+ classes the pipeline runs the unmodified metrics once per class against the rest and averages with equal weight (`evaluation.class_averaging: ovr_macro`, per-class values in `ovr_per_class.csv`), or once on the `binary_target` collapse (`binary`). `cls_acc` uses macro F1 (`F1_type="macro"`); it was micro, which equals accuracy and hides a failing minority class. |
| dcr, hit_rate | SynthEval | Verified | DCR is tanh-normalised (SE-20). |
| mia | SynthEval | Open (SE-16) | When the attacker predicts no members, precision and recall disagree on direction; seeded by the pipeline before each metric. |
| att_discl | SynthEval | Open (SE-15) | Scores train and holdout together with no member/non-member contrast, so it measures utility as much as risk. |
| Seeding | | Verified | Each metric runs in its own `evaluate` call after reseeding, so reruns match exactly. |
| Checkpoints | | Fixed | Checkpoint schema bumped to v4 so results computed before these fixes are recomputed. |

## Custom metrics and scoring

| Item | Reference | Status | Notes |
| --- | --- | --- | --- |
| log disparity value | Bhanot et al. 2021, Eq. 3 | Verified | Log odds ratio matches a hand computation; chi-squared without continuity correction equals the two-proportion z-test; Benjamini-Hochberg matches statsmodels. Cells under 5 counts are marked "Insufficient Data". |
| log_disparity_share_significant | Bhanot et al. 2021 | Fixed | Subgroups missing from the synthetic data (infinite disparity, no p-value) counted as fine, rewarding a generator that drops a minority. They now count as misrepresented, matching the paper's "Absent" label. |
| log_disparity_mean_abs | Bhanot et al. 2021 | Deviation | Averages finite disparities only; absent subgroups are captured by the share above instead of making the mean infinite. |
| holdout TSTR (macro-F1, balanced accuracy, macro AUPRC, per-class F1) | sklearn `f1_score(average="macro")`, `balanced_accuracy_score`, `average_precision_score` | Verified | The fixed XGBoost of the HPO objective, fitted on each dataset and scored on the test split; the same classifier fitted on the real train rows is the ceiling (`tstr_holdout.csv`). The three macro scores are ranked as custom utility; per-class F1 is reported, not ranked. |
| Anonymeter singling out, linkability, inference | Giomi et al., PoPETs 2023; `anonymeter` evaluators | Verified | Library code, unmodified. Risk = (attack rate - control rate) / (1 - control rate), with the test split as the control. Targets are one encounter per patient (`privacy_attacks.unit: patient`), and the train targets are subsampled to the control size, because whether a predicate singles out one record depends on how many records there are. Linkability links the QI columns to the sensitive columns; inference guesses each sensitive column from the QIs and the worst one is ranked. Singling out stops after `singling_out_max_attempts` candidate predicates (upstream: 10 million), which can underestimate the risk and is logged. Attacks Anonymeter flags as no better than random guessing are marked `reliable: false` in `privacy_attacks.csv`. Unit tests check a copy of train scores higher than fresh data and that a seed reproduces the risks; the random linkability baseline is unseeded upstream, so only its `baseline_rate` varies. |
| holdout-referenced DCR and NNDR | Platzer & Reutterer 2021 (holdout reference); heuristics per Ganev & De Cristofaro 2023 | Deviation | Custom code (`synthdata/evaluation/privacy_attacks.py`) on scikit-learn `NearestNeighbors`, Euclidean distance over min-max scaled numeric and one-hot nominal columns. `dcr_closer_to_train_share`: share of synthetic rows closer to a training sample (as many patients as the test split) than to the test rows, 0.5 expected. `distance_mia_auc`: AUC of telling those training patients from test patients by each patient's closest encounter to the synthetic data, 0.5 expected. `dcr_holdout_ratio`, `nndr_holdout_ratio`: 5th percentile of the synthetic rows' DCR (NNDR) to train over the test rows' own, 1 expected. No privacy guarantee; they are labelled heuristics in the report. Ranking gives no credit past the reference (share and AUC clipped at 0.5, ratios at 1). Unit tests check a train copy scores 0 ratios and AUC above 0.6, and fresh data scores near the references. |
| Ranking | | Verified | Per-metric direction, min-max scaling across models, mean per framework and type, weighted geometric mean across types (floor 0.01). Seed replicates give 95% Student-t intervals and Welch-test ties. Baselines are scored but never recommended. |
