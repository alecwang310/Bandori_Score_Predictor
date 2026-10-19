# Consolidated sweep comparison — new post-rule events only

Protocol (identical for every number below, via `bandori/sweep_harness.py`, validated
to reproduce the frozen nn_pca 6.71% / ridge 7.61% exactly): train on the
`eight_post_events` bank (315 events incl. post-rule block 311-320), test only on
events **321-324**, clean checkpoints, tiers **500/1000/1500/2000**, event-balanced
MAPE over 312 matched rows. Everything on 321-324 is descriptive (4 events); the
LOO discipline (selection on post-rule training events 311-320 only) is applied
where a config is crowned.

## Master ranking (best config per method family)

| Rank | Config | Family (sweep) | MAPE | Bias | Final 24h |
|---:|---|---|---:|---:|---:|
| 1 | **rule_plus_explicit x post_rule+20** | stack | **5.64%** | -0.45% | 2.26% |
| 2 | rule_only x post_rule_only | stack | 5.72% | -0.52% | 2.19% |
| 2 | rule_only x post_rule+20 | stack | 5.72% | -0.75% | 2.34% |
| 4 | rule_only, full bank (LOO-selected) | features | 6.23% | -1.81% | 2.60% |
| 5 | rule_plus_explicit, full bank | features | 6.28% | -0.81% | 2.59% |
| 6 | post_rule+20recent, full pca | data-scope | 6.41% | +0.19% | 2.40% |
| 7 | nn_pca natural (frozen baseline) | all | 6.71% | +0.80% | 2.78% |
| 8 | tcn24 @ 0.6 objective share | objective | 7.03% | better | better |
| 9 | ridge s=0.1 | capacity | 7.19% | +0.72% | 3.15% |
| 10 | ridge post_rule_only | data-scope | 7.29% | +2.42% | 3.23% |
| 11 | ridge natural (frozen baseline) | all | 7.61% | +2.02% | 3.06% |
| - | best ensemble (top-3 mean) | ensemble | 6.91% | +1.08% | 2.82% |
| - | raised-objective nn_pca @ 0.6 | objective | 9.55% | +3.24% | 2.89% |
| - | nn_pca hidden 64 / deep 32x16 | capacity | 9.19% / 9.34% | - | - |

## What each sweep established

1. **Data scope** (sweep_data_scope): trimming supervision to the post-rule block + ~20
   recent older events beats the full 315 bank (-0.30pp, all four events) and nearly
   kills the positive bias. Removing ALL old data is mildly harmful for the NN (6.92%),
   helpful for ridge (7.29%). Soft recency down-weighting consistently hurts - trimming
   beats reweighting. Protocol B (rebuilding features on a post-rule-only analog
   reference pool) collapses to 9-10%: the reference pool needs old events.

2. **Features** (sweep_features): rule metadata is the signal on post-rule events, but
   only when the matrix is lean - rule_only (504 cols) 6.23% LOO-selected, vs full pca
   6.71%. Explicit columns are load-bearing (drop = 8.45%). Daily summaries do not
   earn their keep (permuting them improves the fitted model); pace/rate summaries are
   noise, clock harmonics mildly useful. LOO caught the explicit_only trap (6.08%
   descriptive, 13.17% LOO). Ridge on rule_only 7.18% confirms the finding is
   model-agnostic.

3. **Capacity** (sweep_capacity): U-shaped, hidden 8 is optimal; 16/32/64 degrade
   monotonically, two hidden layers never help, PCA rank 16 is the sweet spot, default
   decay/epochs near-optimal. Train-side LOO was too noisy (10 events) to confirm h8
   over h64 - the h8 advantage is descriptive-only per the selection discipline.

4. **Objective-tier loss** (sweep_objective): the 60% reweighting that helped the
   hourly LSTM (+1.3pp) is actively harmful for JointNet (+2.84pp at share 0.6); the
   response is within seed noise (~1pp spread) except at high share. Ridge insensitive
   (control confirmed). tcn24 benefits (-0.65pp) - temporal/shared encoders respond to
   tier reweighting; tiny MLPs do not.

5. **Ensembles** (sweep_ensemble): nothing beats the single nn_pca. Seed variance
   dominates (seeds: 6.71/7.47/7.44/8.86/7.98% - seed 1729 is an outlier-good draw);
   averaging regresses toward worse seeds, and nn_pca already has the lowest bias of
   every member. Honest train-side blend selection lands at 6.83% - worse than the
   untuned single. Raised-objective blending also fails (9.55% alone).

## The stack (sweep_stack): effects combine

| config | full bank | post_rule+20 | post_rule_only |
|---|---:|---:|---:|
| rule_plus_explicit (600 cols) | 6.28% | **5.64%** | 6.23% |
| rule_only (504 cols) | 6.23% | 5.72% | 5.72% |
| full pca (1168 cols) | 6.71% | 6.41% | 6.92% |

The feature win and the scope win are complementary: together they take 6.71% to
**5.64%** (-1.07pp), with bias -0.45% and final-24h 2.26%. rule_only x post_rule_only
(5.72%, 2.19% final24h) needs supervision from only 10 events and is within noise of
the best. ridge on the stack improves too (6.92%) but stays behind the NN.

## Reproducibility note (found during consolidation)

JointNet.fit is layout-sensitive at the 1e-15 level: a strided column view
(e.g. `X[:, np.arange(1168)]`) perturbs the mean/scale summation order vs a contiguous
copy, and 240 Adam steps amplify that into different models (observed 6.41% vs 6.70%
for bitwise-equal inputs). Every number above uses contiguous copies; the harness
`validate()` reproduces the frozen scores exactly. Any future sweep must copy column
subsets to contiguous arrays before fitting.

## Caveats

- 4 test events, previously inspected; all deltas descriptive. Rows within an event are
  dependent; effective n = 4.
- Seed spread ~1pp on the JointNet family; the grid is single-seed (1729) except the
  ensemble/objective seed checks. Stack gains (-1.07pp) are larger than the seed spread
  of the full-pca model but were not multi-seed confirmed.
- LOO selection uses labels-only holdout (Protocol A); LOO levels are higher than
  321-324 levels because training events carry more checkpoints.
- Event 322 is the hard event for every config (~8.4-10.3%); event 323 the easiest.
- Nothing committed; frozen runs and evaluation_registry.json untouched.
