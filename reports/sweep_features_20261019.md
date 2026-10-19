# Feature ablation/selection for nn_pca on events 321-324

## Frozen validation

nn_pca 6.71% MAPE (bias +0.80%, final24h 2.78%); ridge 7.61%. Protocol: 315-event supervised bank (incl. post-rule 311-320), test on 321-324, focal tiers 500/1000/1500/2000, event-balanced MAPE.

pca layout: 16 PCs + 96 explicit + 552 daily + 504 rule = 1168 columns. Custom matrices are exact column subsets of the finished (masked) layout; train/val counts asserted equal (see audit in evaluation.json).

## Main table: feature set | columns | MAPE | bias | final24h

| feature set | columns | MAPE | bias | final24h |
|---|---:|---:|---:|---:|
| nn_pca_full | 1168 | 6.71% | +0.80% | 2.78% |
| drop_pcs | 1152 | 7.64% | +1.98% | 3.30% |
| drop_explicit | 1072 | 8.45% | +1.70% | 3.58% |
| drop_daily | 616 | 6.60% | -1.33% | 2.69% |
| drop_rule | 664 | 6.66% | +0.92% | 2.87% |
| nn_minimal | 1224 | 7.01% | +0.78% | 2.84% |
| nn_standard | 1440 | 9.18% | +3.11% | 3.47% |
| nn_rich | 2976 | 9.45% | +3.89% | 3.23% |
| rule_only_plus_explicit | 600 | 6.28% | -0.81% | 2.59% |
| drop_rule_types | 904 | 6.69% | +1.25% | 2.92% |
| drop_rule_era | 856 | 6.70% | -0.27% | 2.83% |
| explicit_only | 96 | 6.08% | +0.59% | 2.95% |
| pc_plus_explicit | 112 | 6.51% | -0.12% | 3.06% |
| expl_plus_daily | 648 | 7.19% | +2.40% | 3.01% |
| rule_only | 504 | 6.23% | -1.81% | 2.60% |
| daily_pace_only | 976 | 7.66% | +0.52% | 3.04% |
| daily_harmonic_only | 808 | 6.44% | -0.04% | 2.75% |
| ridge_selected | 504 | 7.18% | +0.31% | 2.94% |

## 1. Block ablation (pca layout)

| feature set | columns | MAPE | bias | final24h |
|---|---:|---:|---:|---:|
| nn_pca_full | 1168 | 6.71% | +0.80% | 2.78% |
| drop_pcs | 1152 | 7.64% | +1.98% | 3.30% |
| drop_explicit | 1072 | 8.45% | +1.70% | 3.58% |
| drop_daily | 616 | 6.60% | -1.33% | 2.69% |
| drop_rule | 664 | 6.66% | +0.92% | 2.87% |

## 2. Family comparison

| feature set | columns | MAPE | bias | final24h |
|---|---:|---:|---:|---:|
| nn_minimal | 1224 | 7.01% | +0.78% | 2.84% |
| nn_standard | 1440 | 9.18% | +3.11% | 3.47% |
| nn_rich | 2976 | 9.45% | +3.89% | 3.23% |
| nn_pca_full | 1168 | 6.71% | +0.80% | 2.78% |

## 3. Rule-metadata deep dive

| feature set | columns | MAPE | bias | final24h |
|---|---:|---:|---:|---:|
| nn_pca_full | 1168 | 6.71% | +0.80% | 2.78% |
| drop_rule | 664 | 6.66% | +0.92% | 2.87% |
| rule_only_plus_explicit | 600 | 6.28% | -0.81% | 2.59% |
| drop_rule_types | 904 | 6.69% | +1.25% | 2.92% |
| drop_rule_era | 856 | 6.70% | -0.27% | 2.83% |
| explicit_only | 96 | 6.08% | +0.59% | 2.95% |
| pc_plus_explicit | 112 | 6.51% | -0.12% | 3.06% |
| expl_plus_daily | 648 | 7.19% | +2.40% | 3.01% |
| rule_only | 504 | 6.23% | -1.81% | 2.60% |

## 4. Daily-summary deep dive

| feature set | columns | MAPE | bias | final24h |
|---|---:|---:|---:|---:|
| nn_pca_full | 1168 | 6.71% | +0.80% | 2.78% |
| drop_daily | 616 | 6.60% | -1.33% | 2.69% |
| daily_pace_only | 976 | 7.66% | +0.52% | 3.04% |
| daily_harmonic_only | 808 | 6.44% | -0.04% | 2.75% |

## 5. Permutation importance (fitted full pca, val rows)

| block | cols | MAPE (permuted) | ΔMAPE |
|---|---:|---:|---:|
| expl | 96 | 6.99% | +0.28pp |
| pc | 16 | 6.72% | +0.01pp |
| daily_harmonic | 192 | 6.72% | +0.01pp |
| rule_era | 240 | 6.43% | -0.27pp |
| daily | 552 | 6.38% | -0.33pp |
| daily_pace | 360 | 6.35% | -0.36pp |
| rule_types | 192 | 6.34% | -0.37pp |
| rule | 504 | 6.04% | -0.67pp |

## 6. LOO selection on post-rule train events 311-320

| holdout event | full pca MAPE | selected MAPE |
|---|---:|---:|
| 311 | 16.32% | 15.16% |
| 312 | 8.95% | 6.37% |
| 313 | 6.55% | 4.86% |
| 314 | 7.88% | 6.73% |
| 315 | 8.67% | 8.81% |
| 316 | 9.73% | 8.19% |
| 317 | 10.94% | 11.24% |
| 318 | 9.09% | 7.63% |
| 319 | 16.70% | 15.75% |
| 320 | 7.40% | 8.02% |
| **mean** | **10.22%** | **9.27%** |

| feature set | LOO mean MAPE (311-320) | 321-324 (tuned) |
|---|---:|---:|
| rule_only_plus_explicit | 9.26% | 6.28% |
| rule_only **(selected)** | 9.27% | 6.23% |
| drop_daily | 9.48% | 6.60% |
| daily_harmonic_only | 9.59% | 6.44% |
| nn_minimal | 9.88% | 7.01% |
| nn_pca_full | 10.22% | 6.71% |
| drop_rule_era | 10.30% | 6.70% |
| drop_rule_types | 10.68% | 6.69% |
| drop_rule | 11.70% | 6.66% |
| expl_plus_daily | 12.01% | 7.19% |
| pc_plus_explicit | 12.18% | 6.51% |
| explicit_only | 13.17% | 6.08% |

## 7. Ridge on selected feature set

ridge on rule_only: MAPE 7.18% (frozen ridge on full pca 7.61%).

## Key per-event detail

| key config | MAPE | per-event (321/322/323/324) |
|---|---:|---|
| nn_pca_full | 6.71% | 321:5.67%  322:9.34%  323:4.02%  324:7.80% |
| drop_daily | 6.60% | 321:5.08%  322:10.35%  323:4.32%  324:6.65% |
| drop_rule | 6.66% | 321:6.78%  322:9.22%  323:3.79%  324:6.84% |
| drop_pcs | 7.64% | 321:5.86%  322:9.04%  323:3.89%  324:11.76% |
| rule_only | 6.23% | 321:5.07%  322:10.26%  323:4.33%  324:5.25% |

## Caveats

- The 321-324 grid is descriptive (4 events); the tuned headline is the LOO-selected set scored once on 321-324.
- LOO over 311-320 removes labels only (Protocol A features keep the full reference pool), same convention as the harness.
- Permutation importance runs on 78 validation rows and can understate blocks whose information is duplicated by another block.
