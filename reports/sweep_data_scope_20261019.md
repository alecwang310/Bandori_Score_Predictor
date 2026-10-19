# Training-data scope sweep (new post-rule events 321-324)

Scope: event-balanced MAPE on clean checkpoints of events 321-324, tiers 500/1000/1500/2000 (312 rows), identical for every row below. Both models are fit with frozen hyperparameters (JointNet hidden 8, seed 1729, gain 1.0; ridge strength 0.01, gain 0.5) — no tuning was performed, so every 321-324 number is descriptive.

Protocol A (table below): validation features are the frozen full-history features for every variant (reference pool = all 315 training events); only which labeled rows enter the supervised fit — and the compression/masks fit on those rows — changes. `n_supervised cells (sum w)` is `np.count_nonzero(prep.weights) (prep.weights.sum())` after any scaling.

Frozen baselines reproduced by `validate()`: nn_pca MAPE 6.71%, ridge 7.61%.

## Ranked results — Protocol A

| variant | model | MAPE | bias | final24h | n_supervised cells (sum w) |
| --- | --- | --- | --- | --- | --- |
| post_rule+20recent | nn_pca | 6.41% | 0.19% | 2.40% | 15626 (0.7500) |
| post_rule+50recent | nn_pca | 6.67% | 1.68% | 2.51% | 31465 (0.7500) |
| full_315 | nn_pca | 6.71% | 0.80% | 2.78% | 50232 (0.7699) |
| post_rule_only | nn_pca | 6.92% | 2.26% | 2.34% | 5348 (0.7500) |
| post_rule+100recent | nn_pca | 6.96% | 1.16% | 2.78% | 44242 (0.7500) |
| weight_pre2026_x0.5 | nn_pca | 7.10% | 1.93% | 2.66% | 50232 (0.5286) |
| weight_beyond50recent_x0.25 | nn_pca | 7.28% | 1.72% | 2.65% | 50232 (0.5164) |
| post_rule_only | ridge | 7.29% | 2.42% | 3.23% | 5348 (0.7500) |
| post_rule+20recent | ridge | 7.30% | 1.72% | 3.16% | 15626 (0.7500) |
| weight_pre2026_x0.25 | nn_pca | 7.31% | 1.80% | 2.52% | 50232 (0.4080) |
| weight_beyond50recent_x0.5 | nn_pca | 7.35% | 1.49% | 3.12% | 50232 (0.6009) |
| weight_pre2026_x0.25 | ridge | 7.59% | 1.87% | 3.09% | 50232 (0.4080) |
| weight_pre2026_x0.5 | ridge | 7.61% | 1.95% | 3.08% | 50232 (0.5286) |
| full_315 | ridge | 7.61% | 2.02% | 3.06% | 50232 (0.7699) |
| weight_beyond50recent_x0.5 | ridge | 7.70% | 2.16% | 3.08% | 50232 (0.6009) |
| weight_beyond50recent_x0.25 | ridge | 7.80% | 2.31% | 3.10% | 50232 (0.5164) |
| post_rule+100recent | ridge | 7.81% | 2.21% | 3.10% | 44242 (0.7500) |
| post_rule+50recent | ridge | 8.23% | 2.81% | 3.29% | 31465 (0.7500) |

## Per-event MAPE — Protocol A

| variant | model | 321 | 322 | 323 | 324 | pooled |
| --- | --- | --- | --- | --- | --- | --- |
| post_rule+20recent | nn_pca | 5.56% | 9.08% | 3.72% | 7.27% | 6.41% |
| post_rule+50recent | nn_pca | 6.08% | 8.12% | 3.08% | 9.40% | 6.67% |
| full_315 | nn_pca | 5.67% | 9.34% | 4.02% | 7.80% | 6.71% |
| post_rule_only | nn_pca | 5.62% | 8.23% | 5.61% | 8.23% | 6.92% |
| post_rule+100recent | nn_pca | 6.22% | 9.63% | 3.22% | 8.80% | 6.96% |
| weight_pre2026_x0.5 | nn_pca | 6.11% | 8.81% | 3.92% | 9.55% | 7.10% |
| weight_beyond50recent_x0.25 | nn_pca | 6.17% | 9.34% | 3.64% | 9.95% | 7.28% |
| post_rule_only | ridge | 5.78% | 9.23% | 6.15% | 8.00% | 7.29% |
| post_rule+20recent | ridge | 5.90% | 9.90% | 5.77% | 7.63% | 7.30% |
| weight_pre2026_x0.25 | nn_pca | 6.12% | 9.25% | 5.00% | 8.87% | 7.31% |
| weight_beyond50recent_x0.5 | nn_pca | 6.14% | 9.80% | 4.43% | 9.01% | 7.35% |
| weight_pre2026_x0.25 | ridge | 6.06% | 10.07% | 5.71% | 8.53% | 7.59% |
| weight_pre2026_x0.5 | ridge | 6.22% | 10.08% | 5.54% | 8.60% | 7.61% |
| full_315 | ridge | 6.37% | 10.05% | 5.35% | 8.67% | 7.61% |
| weight_beyond50recent_x0.5 | ridge | 6.30% | 10.08% | 5.47% | 8.95% | 7.70% |
| weight_beyond50recent_x0.25 | ridge | 6.24% | 10.09% | 5.58% | 9.27% | 7.80% |
| post_rule+100recent | ridge | 6.50% | 10.05% | 5.49% | 9.20% | 7.81% |
| post_rule+50recent | ridge | 6.40% | 9.92% | 5.70% | 10.89% | 8.23% |

## nn_pca: best Protocol A variant `post_rule+20recent` vs full_315 baseline (negative delta = variant wins)

| event | full_315 | post_rule+20recent | delta |
| --- | --- | --- | --- |
| 321 | 5.67% | 5.56% | -0.11 pp |
| 322 | 9.34% | 9.08% | -0.25 pp |
| 323 | 4.02% | 3.72% | -0.30 pp |
| 324 | 7.80% | 7.27% | -0.52 pp |
| pooled | 6.71% | 6.41% | -0.30 pp |

## ridge: best Protocol A variant `post_rule_only` vs full_315 baseline (negative delta = variant wins)

| event | full_315 | post_rule_only | delta |
| --- | --- | --- | --- |
| 321 | 6.37% | 5.78% | -0.59 pp |
| 322 | 10.05% | 9.23% | -0.82 pp |
| 323 | 5.35% | 6.15% | +0.80 pp |
| 324 | 8.67% | 8.00% | -0.67 pp |
| pooled | 7.61% | 7.29% | -0.32 pp |

## Protocol B — restricted analog reference pool [311..320]

**These rows are NOT matched to Protocol A**: the reference pool change rebuilds the validation case features as well. Never mix this table with the Protocol A table.

| variant | model | MAPE | bias | final24h | n_supervised cells (sum w) |
| --- | --- | --- | --- | --- | --- |
| post_rule_only+restricted_refs | ridge | 9.18% | 5.85% | 2.71% | 5348 (0.7500) |
| post_rule_only+restricted_refs | nn_pca | 10.19% | 2.77% | 3.63% | 5348 (0.7500) |

| variant | model | 321 | 322 | 323 | 324 | pooled |
| --- | --- | --- | --- | --- | --- | --- |
| post_rule_only+restricted_refs | ridge | 5.86% | 6.46% | 11.22% | 13.19% | 9.18% |
| post_rule_only+restricted_refs | nn_pca | 10.78% | 9.84% | 9.66% | 10.50% | 10.19% |

## Supervision retained per scope

| variant | events | cells | sum w | scaled cells |
| --- | --- | --- | --- | --- |
| full_315 | 310 | 50232 | 0.7699 | — |
| post_rule+100recent | 110 | 44242 | 0.7500 | — |
| post_rule+20recent | 30 | 15626 | 0.7500 | — |
| post_rule+50recent | 60 | 31465 | 0.7500 | — |
| post_rule_only | 10 | 5348 | 0.7500 | — |
| weight_beyond50recent_x0.25 | 310 | 50232 | 0.5164 | 24060 |
| weight_beyond50recent_x0.5 | 310 | 50232 | 0.6009 | 24060 |
| weight_pre2026_x0.25 | 310 | 50232 | 0.4080 | 38573 |
| weight_pre2026_x0.5 | 310 | 50232 | 0.5286 | 38573 |

## Caveats

- `post_rule_only` supervises ids 311-320 only — 10 events, of which 8 (311, 314-320) start on/after the 2026-05-26 rule change; 312/313 are pre-rule events that the id range of the spec includes.
- Only 8-10 events of supervision (vs 315) in the restricted scopes; compression/masks are fit on the restricted rows only.
- The frozen training context holds cases for 310 of the 315 training events (ids 1, 2, 3, 4, 10 have none) — identical for every Protocol A variant including the full_315 baseline, so scope comparisons stay matched.
- Recency-weighted variants scale weights without renormalizing total mass (fixed decay/strength acts relatively stronger) and keep full-bank-fitted compression/masks — only loss weights change.
- `sum w < 1` everywhere: proxy labels carry quality weight < 1.
- Protocol B's restricted reference pool logged case-build failures while rebuilding validation context (tiny tiers 1/10 lost all references, plus stale-score checkpoints); the scored focal set nonetheless stayed at exactly 312 rows, but the features differ from Protocol A so the tables must never be mixed.
- Four test events only; deltas are descriptive, do not overclaim.
