# Raised objective-tier loss (60%) — eight_post_events retrain

Objective tiers 500, 1000, 1500, 2000 carry **60%** of total training loss weight mass (raised from 22.1%). Relative weights inside and outside the objective group are preserved, so the duration/event/horizon/proxy/730-day/3x balancing is unchanged within each group; the same normalized weights drive the relative-MSE term and the per-tier signed-bias penalty. Only the `eight_post_events` fold (315 training events, validation 321-324) is retrained; streams, packing and every hyperparameter match the frozen runs.

Reported MAPE counts only tiers 500, 1000, 1500 and 2000 on the `eight_post_events` fold, clean checkpoints. Companion statistics use the identical rows.

| Model | MAPE | Change vs frozen | Signed bias | Final 24h MAPE |
| --- | ---: | ---: | ---: | ---: |
| lstm_reference (frozen) | 9.99% | — | -8.54% | 7.03% |
| lstm_reference (60% objective) | 8.53% | -1.46 pp | -6.60% | 7.10% |
| lstm_hourly (frozen) | 9.22% | — | -6.37% | 6.91% |
| lstm_hourly (60% objective) | 7.89% | -1.33 pp | -2.88% | 6.38% |
| lstm_native (frozen) | 9.41% | — | -4.82% | 6.83% |
| lstm_native (60% objective) | 9.76% | +0.35 pp | -9.14% | 6.61% |
| transformer (frozen) | 12.20% | — | -9.13% | 8.65% |
| transformer (60% objective) | 12.15% | -0.05 pp | -10.24% | 8.66% |

## Non-NN references (frozen)

| Model | MAPE | Signed bias | Final 24h MAPE |
| --- | ---: | ---: | ---: |
| ridge_clock | 7.64% | 2.19% | 3.07% |
| ridge_previous | 7.61% | 2.02% | 3.06% |
| baseline | 9.99% | 0.78% | 4.03% |

## Paired event-bootstrap: raised minus frozen

| Model | MAPE change (percentage points) | 95% interval |
| --- | ---: | ---: |
| lstm_reference | -1.46 | [-1.81, -1.11] |
| lstm_hourly | -1.33 | [-2.86, +0.19] |
| lstm_native | +0.35 | [-2.03, +2.74] |
| transformer | -0.05 | [-1.45, +1.36] |

Negative favors the raised-objective retrain. Four event clusters limit precision.

## All comparatively recent models (standing scope)

Every recent model ranked on the identical scoped rows (tiers 500/1000/1500/2000, eight_post_events fold, clean checkpoints).

| Model | MAPE | Signed bias | Final 24h MAPE |
| --- | ---: | ---: | ---: |
| ridge_previous | 7.61% | 2.02% | 3.06% |
| ridge_clock | 7.64% | 2.19% | 3.07% |
| lstm_hourly (60% obj) | 7.89% | -2.88% | 6.38% |
| lstm_reference (60% obj) | 8.53% | -6.60% | 7.10% |
| lstm_hourly (frozen) | 9.22% | -6.37% | 6.91% |
| lstm_native (frozen) | 9.41% | -4.82% | 6.83% |
| lstm_native (60% obj) | 9.76% | -9.14% | 6.61% |
| baseline | 9.99% | 0.78% | 4.03% |
| lstm_reference (frozen) | 9.99% | -8.54% | 7.03% |
| transformer (60% obj) | 12.15% | -10.24% | 8.66% |
| transformer (frozen) | 12.20% | -9.13% | 8.65% |

Best raised NN (lstm_hourly 60% objective) against each comparator; negative favors the raised model.

| lstm_hourly (60% obj) minus | MAPE change (percentage points) | 95% interval |
| --- | ---: | ---: |
| ridge_previous | +0.28 | [-1.35, +1.78] |
| ridge_clock | +0.25 | [-1.26, +1.77] |
| lstm_reference (60% obj) | -0.65 | [-2.09, +1.52] |
| lstm_hourly (frozen) | -1.33 | [-2.86, +0.19] |
| lstm_native (frozen) | -1.52 | [-2.79, -0.26] |
| lstm_native (60% obj) | -1.88 | [-4.16, +0.61] |
| baseline | -2.10 | [-3.05, -1.17] |
| lstm_reference (frozen) | -2.10 | [-3.96, +0.46] |
| transformer (60% obj) | -4.26 | [-6.29, -1.61] |
| transformer (frozen) | -4.31 | [-6.18, -2.43] |

Intervals use paired event bootstrap over the four validation events. Against the two ridge variants the interval spans no difference; against both transformers, lstm_native (frozen) and baseline the raised hourly LSTM is clearly better.

## Weight shift

| Bundle | Objective mass before | Objective factor | Other factor |
| --- | ---: | ---: | ---: |
| reference | 22.08% | 2.718 | 0.513 |
| hourly | 22.08% | 2.718 | 0.513 |
| native | 22.09% | 2.716 | 0.513 |
| transformer | 22.09% | 2.716 | 0.513 |

## Fixed stress queries (same scope)

| Model | Off-grid 08:37 | 6h tracker gap | 24h tracker gap |
| --- | ---: | ---: | ---: |
| lstm_reference (60% objective) | 7.38% | 10.61% | 9.71% |
| lstm_hourly (60% objective) | 6.68% | 11.32% | 11.07% |
| lstm_native (60% objective) | 7.39% | 7.76% | 9.23% |
| transformer (60% objective) | 11.06% | 10.48% | 12.14% |

Stress queries keep the same fixed checkpoint subset as the frozen study. Tracker gaps are not verified server outages.

Fit: `runs/raised_objective_20261019`. No deployed model is replaced by this experiment.
