# Raised objective-tier loss weighting — JointNet sweep (321-324)

## Summary

Same harness, same rows as every sweep: 315-event bank train, events 321-324 test, event-balanced MAPE on tiers 500/1000/1500/2000 (312 clean rows). Natural objective weight mass is 20.2%. Rescale via raise_objective_share preserves within-group ratios; only the weights array changes.

| Config | Share | Objective set | Hidden | MAPE | Bias | Final 24h | Mass before → after |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| nn_pca natural | natural | — | 8 | 6.71% | 0.80% | 2.78% | 20.2% → 20.2% |
| nn_pca share sweep | 0.22 | 500,1000,1500,2000 | 8 | 8.39% | 2.37% | 3.24% | 20.2% → 22.0% |
| nn_pca share sweep | 0.40 | 500,1000,1500,2000 | 8 | 9.50% | 4.11% | 2.90% | 20.2% → 40.0% |
| nn_pca share sweep | 0.60 | 500,1000,1500,2000 | 8 | 9.55% | 3.24% | 2.89% | 20.2% → 60.0% |
| nn_pca share sweep | 0.80 | 500,1000,1500,2000 | 8 | 6.65% | -0.26% | 2.47% | 20.2% → 80.0% |
| nn_pca objective set @0.6 | 0.60 | 500,1000 | 8 | 7.32% | 0.34% | 2.57% | 12.8% → 60.0% |
| nn_pca objective set @0.6 | 0.60 | 500,1000,1500 | 8 | 7.80% | 2.29% | 2.37% | 14.0% → 60.0% |
| nn_pca objective set @0.6 | 0.60 | 500,1000,1500,2000 | 8 | 9.55% | 3.24% | 2.89% | 20.2% → 60.0% |
| nn_pca objective set @0.6 | 0.60 | 1000,2000 | 8 | 6.59% | -0.63% | 2.62% | 12.9% → 60.0% |
| nn_pca width @0.6 | 0.60 | 500,1000,1500,2000 | 4 | 10.83% | 4.51% | 3.68% | 20.2% → 60.0% |
| nn_pca width @0.6 | 0.60 | 500,1000,1500,2000 | 8 | 9.55% | 3.24% | 2.89% | 20.2% → 60.0% |
| nn_pca width @0.6 | 0.60 | 500,1000,1500,2000 | 16 | 8.51% | 2.99% | 2.59% | 20.2% → 60.0% |
| ridge natural | natural | — | — | 7.61% | 2.02% | 3.06% | 20.2% → 20.2% |
| ridge control @0.6 | 0.60 | 500,1000,1500,2000 | — | 7.85% | 2.61% | 3.05% | 20.2% → 60.0% |
| nn minimal natural | natural | — | — | 7.01% | 0.78% | 2.84% | 20.2% → 20.2% |
| nn minimal share 0.6 | 0.60 | 500,1000,1500,2000 | — | 7.81% | 1.64% | 2.69% | 20.2% → 60.0% |
| tcn24 natural | natural | — | — | 7.68% | 0.93% | 3.48% | 20.2% → 20.2% |
| tcn24 share 0.6 | 0.60 | 500,1000,1500,2000 | — | 7.03% | -0.17% | 3.21% | 20.2% → 60.0% |
| nn_pca TUNED (LOO-selected) | 0.80 | 500,1000,1500,2000 | 8 | 6.65% | -0.26% | 2.47% | 20.2% → 80.0% |

## Per-event MAPE (key configs)

| Config | 321 | 322 | 323 | 324 |
| --- | ---: | ---: | ---: | ---: |
| natural | 5.67% | 9.34% | 4.02% | 7.80% |
| share 0.22 | 6.28% | 9.73% | 4.68% | 12.89% |
| share 0.4 | 7.98% | 9.63% | 5.20% | 15.19% |
| share 0.6 | 7.56% | 10.90% | 6.93% | 12.82% |
| share 0.8 | 5.40% | 11.00% | 4.53% | 5.68% |
| ridge natural | 6.37% | 10.05% | 5.35% | 8.67% |
| ridge @0.6 | 6.56% | 10.20% | 5.54% | 9.09% |
| tuned (LOO, share 0.8) | 5.40% | 11.00% | 4.53% | 5.68% |

## LOO selection on 311-320

Selection on post-rule TRAINING events 311-320 only (leave-one-out); mean focal-tier MAPE across the 10 held-out events.

| Share | LOO mean MAPE | LOO mean bias |
| ---: | ---: | ---: |
| natural | 10.45% | 1.07% |
| 0.22 | 9.66% | -0.84% |
| 0.4 | 9.34% | -0.31% |
| 0.6 | 8.62% | -0.40% |
| 0.8 | 7.10% | -1.12% |

Selected share: **0.8**. The tuned row above refits the full 315-event bank at that share and scores 321-324.

## Per-tier ablation check

Per-tier MAPE on 321-324 across ALL tiers (ablation honesty: do non-objective tiers degrade?).

| Tier | Natural MAPE | Tuned-share MAPE | Δ pp | Objective? |
| ---: | ---: | ---: | ---: | --- |
| 20 | 14.45% | 13.31% | -1.14 | no |
| 30 | 9.15% | 9.86% | +0.72 | no |
| 40 | 7.58% | 8.39% | +0.81 | no |
| 50 | 9.48% | 10.94% | +1.45 | no |
| 100 | 7.65% | 6.65% | -1.00 | no |
| 200 | 8.95% | 7.32% | -1.63 | no |
| 300 | 5.38% | 4.10% | -1.28 | no |
| 400 | 6.31% | 4.52% | -1.78 | no |
| 500 | 4.30% | 5.52% | +1.22 | yes |
| 1000 | 7.66% | 5.13% | -2.53 | yes |
| 1500 | 9.56% | 9.88% | +0.32 | yes |
| 2000 | 5.30% | 6.08% | +0.78 | yes |
| 3000 | 4.25% | 3.87% | -0.38 | no |
| 4000 | 5.27% | 4.79% | -0.48 | no |
| 5000 | 6.39% | 5.75% | -0.65 | no |
| 10000 | 5.55% | 5.36% | -0.18 | no |
| 20000 | 6.67% | 5.91% | -0.76 | no |
| 30000 | 11.36% | 11.64% | +0.27 | no |
| 40000 | 18.13% | 18.33% | +0.20 | no |
| 50000 | 12.10% | 12.38% | +0.28 | no |
| 70000 | 17.45% | 18.26% | +0.82 | no |
| 100000 | 10.85% | 11.41% | +0.56 | no |

## Seed sensitivity (nn_pca, 321-324)

3 seeds x {natural, 0.22, 0.8} on nn_pca 321-324; seed spread swamps reweighting effect

| Seed | Share | MAPE | Bias |
| ---: | ---: | ---: | ---: |
| 1729 | natural | 6.71% | 0.80% |
| 1729 | 0.22 | 8.39% | 2.37% |
| 1729 | 0.8 | 6.65% | -0.26% |
| 7 | natural | 7.69% | 1.88% |
| 7 | 0.22 | 6.85% | -0.15% |
| 7 | 0.8 | 7.31% | -1.52% |
| 99 | natural | 7.01% | 0.74% |
| 99 | 0.22 | 7.10% | 0.75% |
| 99 | 0.8 | 6.89% | 0.60% |

| Share | Mean MAPE over 3 seeds |
| ---: | ---: |
| natural | 7.14% |
| 0.22 | 7.45% |
| 0.8 | 6.95% |

## Caveats

Share grid on 321-324 is descriptive; only the LOO-selected share on 311-320 is a tuned pick.
Four validation events limit precision; weight mass is exact but realized loss share also depends on per-tier error magnitudes.
Seed spread on 321-324 is ~1 pp — larger than most share effects; single-seed share differences at 321-324 (e.g. 0.22 at seed 1729) are not reliable.
Compression and masks are refit inside each LOO fold (train-only preprocessing).
Run finished in 712s. No deployed model is replaced.
