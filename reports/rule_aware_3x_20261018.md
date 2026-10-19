# Rule-aware models: 3x post-change weighting and robustness

**Historical cross-validation on twelve previously inspected post-change events; not a fresh test.** Every fold trains only on outcomes available before that validation block starts. Events 325/326 never enter fitting or selection. Earlier protected events are reused under the explicit all-except-325/326 instruction; the ledger is retained.

## Changes applied to every approach

The age function stays `exp(-age_days / 730)`. Events **starting on or after 2026-05-26 receive exactly 3x weight**, multiplicatively, in supervised fitting, weighted normalizers/PCA, historical analogs, daily activity priors and train-only statistical corrections. Event IDs do not determine rule era: events 312/313 are pre-change. Saved older fits retain their original configuration and numerical forecasts.

All approaches receive the actual event type and dated reward metadata. The voiced-stamp boundary moved T1000→T1500; T500 gained an animated voiced stamp and T1000 a flashing title. [Official May announcement](https://www.sina.cn/news/detail/5300754143514601.html). T2000 gained a profile decoration from April 7. [Official April announcement](https://www.sina.cn/news/detail/5285051785020053.html). These are changes in rewards and incentives; no point-formula change is asserted.

Sparse new T1500 histories may borrow completion/pace *shapes* from recent legacy T1000 reward-boundary histories. Those T1000 scores remain T1000 labels and never become artificial T1500 supervised targets. The direct same-tier reference count and role-transfer support are explicit. Input columns that never vary in training are masked, so unseen tiers or rule flags do not activate never-trained random connections.

The raw LSTM keeps its raw sequence, small growth summaries, event type and joint output design. Reward metadata is added alongside those inputs. Independent fractional-time branches and output heads are batched without changing the recurrent computation. Values and parameter gradients are checked against the reference implementation. It trains for the same fixed 120 epochs, with the same bias objective and 730-day decay, using CPU for this replay.

## Chronological protocol

| Fold | Post-change training events | Validation events |
| --- | ---: | --- |
| cold_start | 0 | 311, 314, 315, 316 |
| four_post_events | 4 | 317, 318, 319, 320 |
| eight_post_events | 8 | 321, 322, 323, 324 |

No architecture, gain or ensemble membership is selected using a validation-block outcome. The final family is chosen from these cross-validation results using the predeclared error-plus-bias score, a fold-stability penalty, and paired baseline bias/final-day guards. Stress scenarios are diagnostics only and cannot change that choice.

## Clean replay results

| Approach | All-tier MAPE | T500/1000/1500/2000 MAPE | Signed bias | Worst fold MAPE | Final 24h MAPE | Bias / late guards |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 11.54% | 14.37% | -3.65% | 13.65% | 3.36% | pass |
| nn_minimal | 10.67% | 12.69% | -2.29% | 12.90% | 3.15% | pass |
| nn_standard | 10.90% | 13.24% | -2.40% | 13.13% | 3.20% | pass |
| nn_rich | 11.49% | 13.90% | 0.21% | 13.49% | 3.21% | fail |
| nn_pca | 10.42% | 12.31% | -0.99% | 12.65% | 3.02% | fail |
| bagged | 11.01% | 12.87% | -1.49% | 13.21% | 3.16% | fail |
| ridge **selected** | 9.15% | 10.89% | -1.38% | 11.25% | 2.68% | pass |
| hierarchical | 9.67% | 10.68% | 0.41% | 11.19% | 2.61% | fail |
| daily_baseline | 10.76% | 13.70% | -0.50% | 11.91% | 2.44% | fail |
| calibrated_stat | 10.74% | 13.39% | -0.31% | 12.17% | 2.54% | fail |
| trajectory | 10.94% | 13.57% | 0.13% | 12.09% | 2.47% | fail |
| tcn24 | 10.78% | 12.79% | -2.08% | 13.05% | 3.16% | pass |
| raw_lstm | 9.61% | 12.06% | -2.81% | 11.13% | 4.24% | fail |
| neural_mean | 10.25% | 12.05% | -1.69% | 12.44% | 3.12% | pass |
| statistical_mean | 9.73% | 11.70% | -0.29% | 11.25% | 2.46% | fail |
| collection_mean | 9.45% | 11.39% | -1.27% | 11.22% | 2.66% | fail |
| collection_median | 9.70% | 11.81% | -1.10% | 11.51% | 2.60% | fail |
| collection_trimmed | 9.54% | 11.61% | -1.14% | 11.30% | 2.59% | fail |

Each arm uses identical scored cases. Errors are event-balanced. The four focal tiers are shown separately because averages over many deeper tiers can conceal a T1000/T1500 problem.

The selected family is **ridge**. Guard failures for the raw LSTM: cold_start: final-24h MAPE; four_post_events: final-24h MAPE; eight_post_events: bias; eight_post_events: row_bias; eight_post_events: final-24h MAPE. A lower overall error does not override a failed signed-bias or final-day guard; the tolerance is 0.25 percentage points relative to the completion baseline in each fold.

## Adaptation across the rule change

| Approach | No new-rule examples | Four new-rule examples | Eight new-rule examples |
| --- | ---: | ---: | ---: |
| baseline | 13.65% | 12.95% | 8.01% |
| nn_minimal | 12.90% | 11.63% | 7.47% |
| nn_standard | 13.13% | 11.82% | 7.76% |
| nn_rich | 13.49% | 11.93% | 9.06% |
| nn_pca | 12.65% | 10.62% | 8.00% |
| bagged | 13.21% | 11.65% | 8.16% |
| ridge | 11.25% | 9.23% | 6.97% |
| hierarchical | 11.19% | 9.79% | 8.04% |
| daily_baseline | 11.71% | 11.91% | 8.66% |
| calibrated_stat | 12.17% | 11.81% | 8.23% |
| trajectory | 12.09% | 11.81% | 8.91% |
| tcn24 | 13.05% | 11.82% | 7.47% |
| raw_lstm | 11.13% | 9.33% | 8.38% |
| neural_mean | 12.44% | 10.86% | 7.46% |
| statistical_mean | 11.25% | 10.24% | 7.70% |
| collection_mean | 11.22% | 10.04% | 7.09% |
| collection_median | 11.51% | 10.29% | 7.30% |
| collection_trimmed | 11.30% | 10.16% | 7.17% |

Three-times weighting cannot supply evidence before the first new-rule outcome. A large cold-start error should not be hidden by averaging it with later adaptation.

## Missingness and arbitrary-time robustness

| Approach | 08:37 queries | 6h tracker gap | 24h tracker gap |
| --- | ---: | ---: | ---: |
| baseline | 10.56% | 10.88% | 13.01% |
| nn_minimal | 9.62% | 10.25% | 12.04% |
| nn_standard | 9.79% | 11.08% | 12.46% |
| nn_rich | 10.00% | 12.10% | 12.91% |
| nn_pca | 9.22% | 10.25% | 11.81% |
| bagged | 9.96% | 11.24% | 12.57% |
| ridge | 8.07% | 8.58% | 10.51% |
| hierarchical | 8.40% | 9.30% | 10.71% |
| daily_baseline | 9.44% | 10.00% | 12.85% |
| calibrated_stat | 9.28% | 10.19% | 12.70% |
| trajectory | 9.67% | 10.30% | 13.47% |
| tcn24 | 9.70% | 10.86% | 12.25% |
| raw_lstm | 8.60% | 10.61% | 12.09% |
| neural_mean | 9.21% | 10.46% | 11.83% |
| statistical_mean | 8.47% | 9.26% | 11.41% |
| collection_mean | 8.31% | 9.25% | 11.23% |
| collection_median | 8.49% | 9.41% | 11.50% |
| collection_trimmed | 8.37% | 9.31% | 11.34% |

The primary clean table includes all daily checkpoints; stress rows use three selected checkpoints per event. To measure degradation, compare each gap with its identical tier/time clean intersection:

| Approach | Clean → 6h gap MAPE | Clean → 24h gap MAPE |
| --- | ---: | ---: |
| baseline | 10.48% → 10.88% | 9.58% → 13.01% |
| ridge | 8.00% → 8.58% | 7.29% → 10.51% |
| raw_lstm | 8.53% → 10.61% | 7.61% → 12.09% |

### Dense raw-LSTM timestamp check

| Inputs / model | Focal-tier MAPE | Signed bias | Events | Scored rows |
| --- | ---: | ---: | ---: | ---: |
| Raw LSTM, all query times | 13.83% | -6.72% | 12 | 32842 |
| Raw LSTM, scores at most 1h old | 13.72% | -6.61% | 12 | 32482 |
| Linear remaining-score anchor | 18.95% | -10.52% | 12 | 32842 |
| Raw LSTM, supported heads only | 13.51% | -5.54% | 12 | 30463 |
| Linear anchor, same supported heads | 18.43% | -9.73% | 12 | 30463 |

Query times include original focal-tier tracker timestamps and the hourly grid. 2,379 rows use a T1500 head with no previous supervised examples, so that head retains its linear anchor before joint rank reconciliation. Dense observations are correlated within twelve events, and their time distribution differs from the primary matched-clock comparison. This diagnostic cannot be compared as an identical-case gain over the other families.

Dense-query WAPE is 27.83% for the LSTM versus 26.36% for the linear anchor; log RMSE is 0.376 versus 0.302. Lower mean relative error therefore does not establish improvement under every loss or on high-score outliers.


The gap experiments remove the target event’s observations after the declared last-known time; they do not fabricate a game-server outage or fill the gap with later scores. Freshness limits are explicitly relaxed for the stale-input diagnostic, with score age retained as an input. 08:37 tests are branch-time checks, not a claim of validation at every possible hour. The raw LSTM supports any issue timestamp by construction and trains on hourly plus fractional-hour queries.

## Event-type behavior

| Type | Events | Selected MAPE | Selected bias | Raw LSTM MAPE | Raw LSTM bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| challenge | 2 | 14.31% | -13.35% | 12.19% | -7.57% |
| versus | 2 | 7.27% | 0.10% | 7.93% | -0.87% |
| mission_live | 3 | 8.16% | 0.50% | 9.93% | -1.18% |
| live_try | 3 | 8.84% | 2.57% | 8.94% | -0.52% |
| medley | 1 | 6.91% | 0.26% | 10.10% | -6.45% |
| festival | 1 | 8.74% | 0.49% | 8.43% | -5.33% |

Medley and Festival have one post-change validation event each. Their apparent performance is not a stable estimate of an entire event type. Story events have no recent samples. Group differences also reflect event bonuses and popularity.

## Selected model by tier

| Tier | MAPE | Signed bias | Events |
| --- | ---: | ---: | ---: |
| T1 | — | — | 0 |
| T10 | — | — | 0 |
| T20 | 15.59% | 1.01% | 12 |
| T30 | 12.65% | -0.23% | 12 |
| T40 | 10.13% | 0.52% | 12 |
| T50 | 11.57% | 0.24% | 12 |
| T100 | 9.94% | 0.43% | 12 |
| T200 | 7.99% | -0.49% | 12 |
| T300 | 8.33% | -3.66% | 12 |
| T400 | 9.30% | -5.72% | 12 |
| T500 | 10.37% | -8.50% | 12 |
| T1000 | 17.19% | 14.92% | 12 |
| T1500 | 10.31% | 3.03% | 12 |
| T2000 | 5.63% | -3.96% | 12 |
| T3000 | 4.57% | -2.22% | 12 |
| T4000 | 4.61% | -1.70% | 12 |
| T5000 | 5.17% | -2.05% | 12 |
| T10000 | 5.12% | -1.38% | 12 |
| T20000 | 5.61% | -1.55% | 12 |
| T30000 | 7.29% | -3.26% | 12 |
| T40000 | 9.26% | -4.64% | 12 |
| T50000 | 8.54% | -2.87% | 11 |
| T70000 | 13.66% | -8.44% | 11 |
| T100000 | 18.67% | -17.30% | 9 |

T1000 remains a material weakness: 17.19% MAPE and 14.92% signed bias. The selection guards operate on the declared all-tier objective; they do not establish unbiasedness at every individual tier. The challenge-type sample is also small and runs low. These limitations remain visible rather than being repaired using the validation outcomes.


## Frozen event-325 comparison

The family choice was fixed before this retrospective case study. The 373-hour event exceeds every training-event duration, and two issue times have stale tracker inputs. This event cannot establish general performance. [Exact single-point forecasts and matched comparisons](rule_aware_3x_event325.md).

| Approach | All 32 tier/time points | Same 24 fresh-input points |
| --- | ---: | ---: |
| Ours | 4.69% | 4.31% |
| MASHIRO | 12.43% | 11.73% |
| RUI | 14.62% | 14.36% |

Nanami is evaluated only at its two supplied issue times in the linked report; its smaller sample is not mixed with the eight-time comparison.


## Artifact and leakage audit

[Saved-model audit](rule_aware_3x_audit.json): 27/27 checks passed. All eighteen approaches retain identical forecasts when future target-event scores and its own final labels are multiplied by 1000. Fold chronology, reference prefixes, genuine T1500 targets, numerical artifact hashes and the unchanged protected-event ledger are checked. Unit tests cover recurrent branch values/gradients and train-only support masks.


Event-clustered paired bootstrap intervals against the completion baseline are in the JSON. Twelve weekly event clusters, several overlapping reward changes and repeated historical inspection limit the strength of any robustness claim. No prediction intervals are fitted on validation errors. The unusually long event 325 remains a separate retrospective case study.

The combined follow-up changes reward metadata, reference support and training weights together. It does not isolate a causal accuracy gain from the 3x multiplier alone.

Production weights: `runs/rule_aware_3x_20261018/full_refit`. Selection: `runs/rule_aware_3x_20261018/selection.json`. Per-checkpoint audit rows remain in each fold’s `predictions.csv`.
