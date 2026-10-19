# Sweep: ensembles of case-feature models on new events 321-324

## Protocol

Shared harness (`bandori.sweep_harness`): train on the frozen 315-event bank (post-rule 311-320 included), test only on events 321, 322, 323, 324. Reported MAPE = event-balanced MAPE on clean checkpoints, tiers 500, 1000, 1500, 2000 (312 rows); bias and final-24h MAPE use the identical rows. Harness validation reproduced nn_pca 6.71% and ridge 7.61%.
Frozen-run ensembles under the same scope for context: neural_mean 7.32%, collection_mean 7.56% (from the frozen eight_post_events evaluation, not refit here).

## Fitted members

| Member | Val 321-324 MAPE | Train 311-320 MAPE |
| --- | ---: | ---: |
| nn_pca_s1729 | 6.71% | 8.67% |
| nn_pca_s1730 | 7.47% | 8.72% |
| nn_pca_s1731 | 7.44% | 8.58% |
| nn_pca_s1732 | 8.86% | 9.01% |
| nn_pca_s1733 | 7.98% | 8.35% |
| nn_minimal | 7.01% | 8.66% |
| nn_standard | 9.18% | 9.81% |
| ridge | 7.61% | 8.77% |
| tcn24 | 7.68% | 11.81% |
| nn_pca_raised | 9.55% | 6.05% |

## Ensemble comparison

| Ensemble | Members | Weight rule | MAPE | Bias | Final 24h MAPE |
| --- | --- | --- | ---: | ---: | ---: |
| nn_pca (single) | nn_pca_s1729 | single | 6.71% | 0.80% | 2.78% |
| ridge (single) | ridge | single | 7.61% | 2.02% | 3.06% |
| blend nn ridge w0.00 | nn_pca_s1729 + ridge | 0.00·nn_pca + 1.00·ridge (final values) | 7.61% | 2.02% | 3.06% |
| blend nn ridge w0.25 | nn_pca_s1729 + ridge | 0.25·nn_pca + 0.75·ridge (final values) | 7.08% | 1.72% | 2.94% |
| blend nn ridge w0.50 | nn_pca_s1729 + ridge | 0.50·nn_pca + 0.50·ridge (final values) | 6.83% | 1.41% | 2.85% |
| blend nn ridge w0.75 | nn_pca_s1729 + ridge | 0.75·nn_pca + 0.25·ridge (final values) | 6.71% | 1.11% | 2.81% |
| blend nn ridge w1.00 | nn_pca_s1729 + ridge | 1.00·nn_pca + 0.00·ridge (final values) | 6.71% | 0.80% | 2.78% |
| bagging3 mean | nn_pca_s1729 + nn_pca_s1730 + nn_pca_s1731 | equal mean | 7.10% | 1.04% | 2.87% |
| bagging5 mean | nn_pca_s1729 + nn_pca_s1730 + nn_pca_s1731 + nn_pca_s1732 + nn_pca_s1733 | equal mean | 7.49% | 1.49% | 2.98% |
| diversity mean5 | nn_pca_s1729 + nn_pca_s1730 + nn_pca_s1731 + nn_minimal + nn_standard | equal mean | 7.38% | 1.40% | 2.97% |
| diversity mean6 tcn | nn_pca_s1729 + nn_pca_s1730 + nn_pca_s1731 + nn_minimal + nn_standard + tcn24 | equal mean | 7.27% | 1.32% | 3.04% |
| diversity trimmed mean6 | nn_pca_s1729 + nn_pca_s1730 + nn_pca_s1731 + nn_minimal + nn_standard + tcn24 | equal mean after dropping per-event worst member | 6.96% | 1.05% | 2.96% |
| diversity top3 train | nn_pca_s1731 + nn_minimal + nn_pca_s1729 | equal mean of top-3 by train-side (311-320) MAPE | 6.91% | 1.08% | 2.82% |
| raised alone | nn_pca_raised | single (60% objective-tier loss share) | 9.55% | 3.24% | 2.89% |
| raised natural 50 50 | nn_pca_raised + nn_pca_s1729 | 0.5·nn_pca_raised + 0.5·nn_pca (final values) | 7.78% | 2.02% | 2.76% |
| raised ridge 50 50 | nn_pca_raised + ridge | 0.5·nn_pca_raised + 0.5·ridge (final values) | 7.87% | 2.63% | 2.83% |
| raised natural ridge thirds | nn_pca_raised + nn_pca_s1729 + ridge | equal thirds (final values) | 7.26% | 2.02% | 2.78% |
| Train-selected blend (headline) | nn_pca_s1729 + ridge | w=0.50 chosen on train 311-320 (in-sample LOO-scoring approximation) | 6.83% | 1.41% | 2.85% |

## Per-event MAPE (key ensembles)

| Ensemble | 321 | 322 | 323 | 324 |
| --- | ---: | ---: | ---: | ---: |
| nn pca s1729 | 5.67% | 9.34% | 4.02% | 7.80% |
| blend nn ridge w0.50 | 5.49% | 9.68% | 4.03% | 8.12% |
| blend nn ridge w0.75 | 5.52% | 9.50% | 3.88% | 7.95% |
| bagging5 mean | 6.36% | 9.72% | 3.58% | 10.30% |
| diversity mean6 tcn | 6.54% | 9.68% | 3.63% | 9.23% |
| diversity top3 train | 6.05% | 9.50% | 3.41% | 8.68% |
| train selected blend w0.50 | 5.49% | 9.68% | 4.03% | 8.12% |

## Train-side-selected blend weight

Candidates: w·nn_pca + (1−w)·ridge on final values, w ∈ {0.00, 0.25, 0.50, 0.75, 1.00}. Selection uses only post-rule training events 311-320: members are fitted once on all 315 events and w is picked by event-balanced MAPE on their in-sample predictions of those ten events — a cheap approximation of leave-one-out refit (the labels of 311-320 entered the supervised fit), labelled as such.

| w (nn_pca weight) | Train 311-320 MAPE | 321-324 MAPE (descriptive) |
| ---: | ---: | ---: |
| 0.00 | 8.77% | 7.61% |
| 0.25 | 8.64% | 7.08% |
| 0.50 | 8.59% | 6.83% |
| 0.75 | 8.60% | 6.71% |
| 1.00 | 8.67% | 6.71% |

Chosen w = 0.50; its 321-324 score (the tuned headline): MAPE 6.83%, bias 1.41%, final-24h 2.85%. Descriptive best ensemble on 321-324: blend nn ridge w1.00 at 6.71% — shown for contrast only, not a valid selection.

## Caveats

Only four validation events (321, 322, 323, 324); they have been inspected by prior experiments, so the full descriptive grid is not a fresh test.
Rows within an event are dependent; per-event MAPE (not row count) is the effective sample size.
Train-side selection is in-sample on 311-320, so the chosen weight is optimistic; a true LOO refit (315-minus-one-event) was not run because the approximation already picks an interior weight and refitting every member ten times is costly.
The raised-objective variant (60% objective-tier loss share) was designed for the stream LSTM/transformer family; on the case-feature JointNet it degrades sharply (single MAPE 9.55%), so its blends are shown for completeness.
No deployed model is replaced by this sweep; evaluation_registry.json was not modified.
