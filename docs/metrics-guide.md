# What the metrics mean

A plain-language guide to every number in `report.md` and `combined_evaluation.csv`. For how each metric was checked against its paper or library, and where ours differs, see [verification.md](verification.md).

## How to read any metric

- **Compare with the two baselines first.** Every run scores two fixed reference datasets next to the generators. `baseline_train_copy` is the real training rows released as is: the best utility and fairness you can get and the worst privacy. `baseline_marginals` samples each column on its own, so it keeps every column's distribution but no relationship between columns: the utility a generator must beat. A metric is only informative if those two land far apart.
- **Look at the direction.** Each panel of the `metrics_<type>.png` plots says whether higher, lower or closer to 0 is better, and draws a green dashed line at the ideal value where one exists. `metric_directions.csv` lists the same for every metric.
- **Look at the interval.** With several seeds (`generation.n_replicates`), each model shows a mean and a 95% interval. Two models whose intervals overlap a lot are not clearly different.
- **Ranks are relative.** The utility, privacy, fairness and overall scores rescale every metric between the worst and the best model of this run. A score of 1 means "best here", not "perfect".

## Utility: is the synthetic data as useful as the real data?

### Prediction (custom)

| Metric | What it measures | Read it as |
| --- | --- | --- |
| `tstr_macro_f1`, `tstr_balanced_accuracy`, `tstr_macro_auprc` | **Train on synthetic, test on real.** A fixed XGBoost classifier learns the target from the synthetic rows and predicts the real holdout. Macro and balanced scores weigh every class equally. | Higher is better. The `baseline_train_copy` row is the ceiling (train on real, test on real). Chance level is about 1 / number of classes for macro-F1 and balanced accuracy. |
| `tstr_kappa` | Agreement between predictions and truth beyond chance (Cohen's kappa). | 0 = chance, 1 = perfect. Reported, not ranked. |
| `f1_<class>` | F1 of one class. | Shows which class the synthetic data gets wrong. |
| `tsts_*` | **Train on synthetic, test on synthetic**: the same classifier fitted and scored inside the synthetic data (80/20 split). | Not ranked. TSTS far above TSTR means the synthetic data holds a pattern that real rows don't share; TSTS near chance means the generator lost the target. |

### Distributions (SynthEval)

| Metric | What it measures | Better |
| --- | --- | --- |
| `avg_dwm_diff` | Dimension-wise means: average difference between real and synthetic column means (scaled). | Lower |
| `pca_eigval_diff`, `pca_eigvec_ang` | Whether the main directions of variation (principal components) match. | Lower |
| `avg_cio` | Confidence-interval overlap of column means. | Higher |
| `corr_mat_diff` | Difference between the real and synthetic correlation matrices (mixed: numeric and categorical). | Lower |
| `mutual_inf_diff` | Same, with mutual information instead of correlation. | Lower |
| `ks_tvd_stat`, `frac_ks_sigs` | Kolmogorov-Smirnov distance per column (TVD for categories) and the share of columns that differ significantly. | Lower |
| `avg_h_dist` | Hellinger distance between column distributions (0 = identical, 1 = no overlap). | Lower |
| `avg_pMSE` | Propensity score: how well a logistic regression tells real rows from synthetic ones. 0 = indistinguishable. | Lower |
| `avg_qMSE` | Difference between real and synthetic quantiles (and category shares). | Lower |
| `avg_F1_diff`, `avg_F1_diff_hout` | Classifiers (random forest, AdaBoost, SVM, logistic regression) trained on real vs on synthetic: difference in macro-F1, on cross-validation folds and on the holdout. | Lower |
| `auroc` | Difference in AUROC of a model trained on real vs on synthetic data. Run per class against the rest for 3+ classes. | Lower |

### Distributions and detection (synthcity)

| Metric | What it measures | Better |
| --- | --- | --- |
| `sanity.data_mismatch` | Share of columns whose data type differs. | Lower |
| `sanity.common_rows_proportion` | Share of synthetic rows that are exact copies of real rows. | Lower (a privacy warning sign) |
| `sanity.nearest_syn_neighbor_distance` | Typical distance from a real row to its nearest synthetic row. | Lower for utility, but 0 means copies |
| `sanity.close_values_probability`, `distant_values_probability` | Share of synthetic rows very close to / very far from any real row. | Close: higher; distant: lower |
| `stats.jensenshannon_dist`, `inv_kl_divergence`, `chi_squared_test`, `ks_test` | Column-by-column distribution similarity (Jensen-Shannon distance, inverse KL divergence, chi-squared and KS p-values). | Distance: lower; inverse KL and p-values: higher |
| `stats.max_mean_discrepancy` | Kernel distance between the whole real and synthetic distributions. | Lower |
| `stats.wasserstein_dist` | Sinkhorn divergence (about half the squared earth mover's distance) on min-max scaled columns. Runs in plain PyTorch, without KeOps. | Lower |
| `stats.prdc.precision`, `recall`, `density`, `coverage` | Precision: synthetic rows look like real ones. Recall and coverage: the synthetic data covers all kinds of real rows. Density: like precision, counting how many real neighbours each synthetic row has. | Higher |
| `stats.alpha_precision.*` | Alpha-precision (realistic rows), beta-recall (diversity) and authenticity (rows not copied from train). | Higher |
| `performance.linear_model`, `mlp`, `xgb` | Train-on-synthetic, test-on-real scores of three model types. | Higher |
| `performance.feat_rank_distance.corr` | Whether a model trained on synthetic data ranks the features as important in the same order as one trained on real data. | Higher |
| `performance.*_augmentation` | Gain from adding synthetic rows to the real training rows. Unranked. | Higher |
| `detection.detection_xgb`, `mlp`, `linear` | AUC of a classifier trained to tell real rows from synthetic ones. 0.5 = cannot tell them apart. | Lower (towards 0.5) |
| `detection.detection_gmm` | Not a real detector (it ignores the labels). Unranked. | - |

## Privacy: could the synthetic data reveal a real patient?

Privacy numbers are evidence from specific attacks, not guarantees. Compare every model with `baseline_train_copy`, the worst case.

### Attacks and distance checks (custom)

| Metric | What it measures | Read it as |
| --- | --- | --- |
| `anonymeter_singling_out_risk` | Can an attacker write a rule (for example "born 1950, female, region 4") that matches exactly one real training patient, using only the synthetic data? | 0 = no better than an attacker who never saw the synthetic data; 1 = always succeeds. Lower is better. A lower bound: even a copy of train scores low here (see the user manual). |
| `anonymeter_linkability_risk` | Given two partial records of the same person (some columns each), can the attacker link them through the synthetic data? | 0 to 1, lower is better. Catches copies well. |
| `anonymeter_inference_risk` | Knowing a person's quasi-identifiers, can the attacker guess their sensitive columns better than from unseen real rows? Worst sensitive column. | 0 to 1, lower is better. |
| `dcr_closer_to_train_share` | For each synthetic row, is its nearest training row closer than its nearest unseen (test) row? Share of rows where it is. | 0.5 is ideal: synthetic rows are as close to unseen people as to training ones. Above 0.5 means they hug the training rows. |
| `distance_mia_auc` | A membership attack that guesses "was this person in training?" from their distance to the synthetic data. | 0.5 is ideal (a coin flip). Higher means membership leaks. |
| `dcr_holdout_ratio`, `nndr_holdout_ratio` | Distance from synthetic rows to their closest training row, divided by the same for test rows, at the closest 5% (DCR: distance to closest record; NNDR: nearest-neighbour distance ratio). | 1 or above is ideal. Below 1 means synthetic rows sit closer to training rows than to unseen ones. |

### SynthEval

| Metric | What it measures | Better |
| --- | --- | --- |
| `avg_nndr`, `priv_loss_nndr` | Ratio of the distances to the closest and second-closest real row. Near 1 = no single real row stands out as the source. The privacy loss compares it on training vs holdout rows. | Higher; privacy loss lower |
| `nnaa`, `priv_loss_nnaa` | Nearest-neighbour adversarial accuracy: can nearest neighbours tell real from synthetic? Privacy loss compares it on training vs holdout rows. | NNAA near 0.5 for utility; privacy loss lower |
| `median_DCR` | Distance to closest record (median, normalised). 0 = copies. | Higher |
| `hit_rate` | Share of synthetic rows that match a real row within a small tolerance. | Lower |
| `eps_identif_risk`, `priv_loss_eps` | Epsilon-identifiability: share of real rows whose closest synthetic row is closer than their closest real row; the privacy loss compares training with holdout rows. | Lower |
| `mia_recall`, `mia_precision` | A membership-inference classifier (LightGBM) trying to tell training rows from holdout rows using the synthetic data. | Near the no-skill level is best; lower is better |
| `att_discl_risk` | Attribute disclosure: can an attacker infer a sensitive column of a training row by matching on the others? | Lower |

### synthcity

| Metric | What it measures | Better |
| --- | --- | --- |
| `privacy.identifiability_score` | Like epsilon-identifiability above. | Lower |
| `privacy.k-anonymization`, `k-map`, `distinct l-diversity`, `delta-presence` | Cluster-size proxies named after the classic definitions. They score a collapsed generator as very private, so they are unranked. | - |
| `attack.data_leakage_xgb` | Can a classifier trained on synthetic data predict the sensitive columns of real rows? (linear and mlp versions are unranked). | Lower |
| `privacy.DomiasMIA_prior` | Density-based membership attack. Opt-in only: it fails on wide data. | Lower |

## Fairness: does the synthetic data treat groups like the real data does?

Protected columns are set in `data.protected_columns`, and the groups of a numeric one (such as age) in `data.protected_bins`.

| Metric | What it measures | Read it as |
| --- | --- | --- |
| `demographic_parity_gap_error` | The holdout TSTR classifier, trained once on the synthetic and once on the real data, predicts the holdout. For each class and protected group: the group's share of predictions in that class minus everyone else's. The score is the average distance between the synthetic-trained and the real-trained gap. | 0 is ideal (the real disparity kept). Blank when `fairness_signal` is 0. |
| `equal_opportunity_gap_error` | The same for the true-positive rate (recall). | 0 is ideal. |
| `equalized_odds_gap_error` | The same, taking the larger of the true- and false-positive rate differences. | 0 is ideal. |
| `fairness_signal` | 1 when the synthetic-trained classifier beats chance on the holdout (bootstrap 95% interval of balanced accuracy above 1 / classes); otherwise the three scores above are blank and not ranked. | Not ranked. |
| `statistical_parity` | A classifier trained and tested on the synthetic data: difference in the share of positive predictions between a protected group and everyone else. Exactly 0 when the synthetic data has no class signal (the classifier predicts one class), so it is not ranked. | 0 means equal rates. Not ranked. |
| `equal_opportunity` | Difference in true-positive rate (recall) between a group and everyone else, as above. | Not ranked. |
| `equalized_odds` | Mean of the differences in true-positive and false-positive rates, as above. | Not ranked. |
| `log_disparity_mean_abs` | How far each protected subgroup x outcome share drifts from its real share, in log-odds, averaged (Bhanot et al. 2021). | 0 is ideal. |
| `log_disparity_share_significant` | Share of subgroups whose drift is significant after Benjamini-Hochberg correction. | 0 is ideal. |
| `subgroup_distance_mia_auc_gap` | Fairness of privacy: `distance_mia_auc` computed inside each protected group, then the largest difference between two groups of the same protected column. | 0 is ideal (every group equally exposed). Not ranked. |
| `subgroup_distance_mia_auc_max` | The most exposed protected group's `distance_mia_auc`. | 0.5 is ideal. Not ranked. |

The three gap metrics need a two-class target and a two-group protected column. For a target with 3 or more classes they run once per class against the rest. A protected column with more than two groups is compared one group against the rest (each group with at least 5% of the rows) and the gaps are averaged; a numeric protected column such as birth year is cut into the bins you set in `data.protected_bins`, each bin then being a group (left out, with a warning, when it has no bins). The `sp_*`, `eo_*` and `eqo_*` columns hold the gap of each group.

The two subgroup privacy metrics use the same groups (two-valued columns contribute both values). They are reported but not ranked: a copy of train leaks in every group alike, so its gap is near 0, just like an independent draw's, and the worst group mostly repeats the overall `distance_mia_auc`, which is already ranked under privacy. Read them next to `distance_mia_auc`: a model with a low overall AUC but a large gap leaks mainly for one group. `privacy_subgroups.csv` has each group's AUC with its standard error and patient counts; small groups give noisy AUCs, and groups under 10 patients on either side are left out.
