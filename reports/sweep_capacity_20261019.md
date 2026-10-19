# Capacity sweep: small case-feature NN family (2026-10-19)

## Protocol

Frozen harness (bandori.sweep_harness): train on the 315-event bank (10 post-rule events 311-320 inside), test only on new post-rule events 321-324. Event-balanced MAPE on clean checkpoints of focal tiers 500/1000/1500/2000 (312 rows). Same rows, same scoring for every config.

Harness reproduction: nn_pca 6.71% MAPE, bias 0.80%, final24h 2.78% (validated live against frozen 6.71%); ridge 7.61% (validated live).

## 1. Hidden-width sweep (pca matrix, full 315-event bank)

| config | model | params | MAPE | bias | final24h | train | per-event |
|---|---|---|---|---|---|---|---|
| nn_pca_h8 | jointnet | 9,568 | 6.71% | +0.80% | 2.78% | 8.88% | 321:5.67%  322:9.34%  323:4.02%  324:7.80% |
| nn_pca_h16 | jointnet | 19,112 | 7.54% | +1.03% | 2.80% | 9.24% | 321:6.20%  322:9.73%  323:4.54%  324:9.68% |
| nn_pca_h2 | jointnet | 2,410 | 7.56% | +1.68% | 3.34% | 9.78% | 321:7.39%  322:10.14%  323:3.54%  324:9.17% |
| nn_pca_h32 | jointnet | 38,200 | 7.70% | +1.60% | 3.05% | 9.59% | 321:7.54%  322:9.06%  323:4.35%  324:9.84% |
| nn_pca_h4 | jointnet | 4,796 | 8.54% | +2.47% | 3.17% | 9.03% | 321:6.49%  322:9.81%  323:5.01%  324:12.85% |
| nn_pca_h64 | jointnet | 76,376 | 9.19% | +1.94% | 3.60% | 12.81% | 321:8.25%  322:11.27%  323:3.98%  324:13.26% |

## 2. PCA-rank sweep (JointNet hidden 8)

| config | model | params | MAPE | bias | final24h | train | per-event |
|---|---|---|---|---|---|---|---|
| nn_pca_k16_h8 | jointnet | 9,568 | 6.71% | +0.80% | 2.78% | — | 321:5.67%  322:9.34%  323:4.02%  324:7.80% |
| nn_pca_k32_h8 | jointnet | 9,696 | 6.94% | +0.82% | 2.69% | — | 321:5.60%  322:8.76%  323:4.00%  324:9.42% |
| nn_pca_k8_h8 | jointnet | 9,504 | 7.28% | +1.47% | 2.87% | — | 321:6.92%  322:9.71%  323:3.67%  324:8.81% |
| nn_pca_k4_h8 | jointnet | 9,472 | 7.64% | +1.43% | 2.89% | — | 321:5.84%  322:9.39%  323:3.95%  324:11.36% |

## 3. Ridge-strength sweep (pca matrix, gain .5)

| config | model | params | MAPE | bias | final24h | train | per-event |
|---|---|---|---|---|---|---|---|
| ridge_s0.1 | ridge | 28,056 | 7.19% | +0.72% | 3.15% | — | 321:6.37%  322:9.79%  323:5.38%  324:7.21% |
| ridge_s0.01 | ridge | 28,056 | 7.61% | +2.02% | 3.06% | — | 321:6.37%  322:10.05%  323:5.35%  324:8.67% |
| ridge_s1.0 | ridge | 28,056 | 8.02% | -0.06% | 3.53% | — | 321:7.99%  322:10.18%  323:6.87%  324:7.04% |
| ridge_s0.001 | ridge | 28,056 | 8.20% | +3.32% | 3.08% | — | 321:6.80%  322:10.27%  323:6.01%  324:9.71% |

## 4. Depth variant (2 hidden tanh layers, this module only)

| config | model | params | MAPE | bias | final24h | train | per-event |
|---|---|---|---|---|---|---|---|
| nn_deep_16x8 | deep-jointnet | 19,056 | 8.15% | +1.27% | 3.44% | 9.93% | 321:7.59%  322:11.39%  323:5.28%  324:8.35% |
| nn_deep_32x16 | deep-jointnet | 38,344 | 9.34% | +4.67% | 3.80% | 10.15% | 321:9.57%  322:8.77%  323:6.81%  324:12.21% |

## 5. Decay x epochs grid at the best hidden width

| config | model | params | MAPE | bias | final24h | train | per-event |
|---|---|---|---|---|---|---|---|
| nn_pca_h8_d0.01_e120 | jointnet | 9,568 | 7.12% | +0.91% | 3.19% | — | 321:7.37%  322:9.81%  323:4.28%  324:7.04% |
| nn_pca_h8_d0.003_e480 | jointnet | 9,568 | 7.16% | -0.07% | 2.92% | — | 321:6.35%  322:10.14%  323:3.46%  324:8.69% |
| nn_pca_h8_d0.003_e120 | jointnet | 9,568 | 7.30% | +1.75% | 2.96% | — | 321:6.26%  322:8.66%  323:4.34%  324:9.95% |
| nn_pca_h8_d0.01_e240 | jointnet | 9,568 | 7.39% | +1.03% | 3.35% | — | 321:7.42%  322:9.77%  323:4.69%  324:7.69% |
| nn_pca_h8_d0.01_e480 | jointnet | 9,568 | 7.72% | +1.69% | 3.36% | — | 321:7.81%  322:9.93%  323:4.72%  324:8.40% |
| nn_pca_h8_d0.001_e480 | jointnet | 9,568 | 7.93% | +1.34% | 2.85% | — | 321:5.52%  322:10.69%  323:5.80%  324:9.70% |
| nn_pca_h8_d0.001_e120 | jointnet | 9,568 | 8.75% | +2.22% | 3.09% | — | 321:6.01%  322:10.30%  323:5.52%  324:13.16% |
| nn_pca_h8_d0.001_e240 | jointnet | 9,568 | 8.86% | +3.03% | 3.17% | — | 321:8.76%  322:8.66%  323:4.96%  324:13.05% |

## All configs ranked by 321-324 MAPE

| config | model | params | MAPE | bias | final24h | train | per-event |
|---|---|---|---|---|---|---|---|
| nn_pca_h8 | jointnet | 9,568 | 6.71% | +0.80% | 2.78% | 8.88% | 321:5.67%  322:9.34%  323:4.02%  324:7.80% |
| nn_pca_k16_h8 | jointnet | 9,568 | 6.71% | +0.80% | 2.78% | — | 321:5.67%  322:9.34%  323:4.02%  324:7.80% |
| nn_pca_k32_h8 | jointnet | 9,696 | 6.94% | +0.82% | 2.69% | — | 321:5.60%  322:8.76%  323:4.00%  324:9.42% |
| nn_pca_h8_d0.01_e120 | jointnet | 9,568 | 7.12% | +0.91% | 3.19% | — | 321:7.37%  322:9.81%  323:4.28%  324:7.04% |
| nn_pca_h8_d0.003_e480 | jointnet | 9,568 | 7.16% | -0.07% | 2.92% | — | 321:6.35%  322:10.14%  323:3.46%  324:8.69% |
| ridge_s0.1 | ridge | 28,056 | 7.19% | +0.72% | 3.15% | — | 321:6.37%  322:9.79%  323:5.38%  324:7.21% |
| nn_pca_k8_h8 | jointnet | 9,504 | 7.28% | +1.47% | 2.87% | — | 321:6.92%  322:9.71%  323:3.67%  324:8.81% |
| nn_pca_h8_d0.003_e120 | jointnet | 9,568 | 7.30% | +1.75% | 2.96% | — | 321:6.26%  322:8.66%  323:4.34%  324:9.95% |
| nn_pca_h8_d0.01_e240 | jointnet | 9,568 | 7.39% | +1.03% | 3.35% | — | 321:7.42%  322:9.77%  323:4.69%  324:7.69% |
| nn_pca_h16 | jointnet | 19,112 | 7.54% | +1.03% | 2.80% | 9.24% | 321:6.20%  322:9.73%  323:4.54%  324:9.68% |
| nn_pca_h2 | jointnet | 2,410 | 7.56% | +1.68% | 3.34% | 9.78% | 321:7.39%  322:10.14%  323:3.54%  324:9.17% |
| ridge_s0.01 | ridge | 28,056 | 7.61% | +2.02% | 3.06% | — | 321:6.37%  322:10.05%  323:5.35%  324:8.67% |
| nn_pca_k4_h8 | jointnet | 9,472 | 7.64% | +1.43% | 2.89% | — | 321:5.84%  322:9.39%  323:3.95%  324:11.36% |
| nn_pca_h32 | jointnet | 38,200 | 7.70% | +1.60% | 3.05% | 9.59% | 321:7.54%  322:9.06%  323:4.35%  324:9.84% |
| nn_pca_h8_d0.01_e480 | jointnet | 9,568 | 7.72% | +1.69% | 3.36% | — | 321:7.81%  322:9.93%  323:4.72%  324:8.40% |
| nn_pca_h8_d0.001_e480 | jointnet | 9,568 | 7.93% | +1.34% | 2.85% | — | 321:5.52%  322:10.69%  323:5.80%  324:9.70% |
| ridge_s1.0 | ridge | 28,056 | 8.02% | -0.06% | 3.53% | — | 321:7.99%  322:10.18%  323:6.87%  324:7.04% |
| nn_deep_16x8 | deep-jointnet | 19,056 | 8.15% | +1.27% | 3.44% | 9.93% | 321:7.59%  322:11.39%  323:5.28%  324:8.35% |
| ridge_s0.001 | ridge | 28,056 | 8.20% | +3.32% | 3.08% | — | 321:6.80%  322:10.27%  323:6.01%  324:9.71% |
| nn_pca_h4 | jointnet | 4,796 | 8.54% | +2.47% | 3.17% | 9.03% | 321:6.49%  322:9.81%  323:5.01%  324:12.85% |
| nn_pca_h8_d0.001_e120 | jointnet | 9,568 | 8.75% | +2.22% | 3.09% | — | 321:6.01%  322:10.30%  323:5.52%  324:13.16% |
| nn_pca_h8_d0.001_e240 | jointnet | 9,568 | 8.86% | +3.03% | 3.17% | — | 321:8.76%  322:8.66%  323:4.96%  324:13.05% |
| nn_pca_h64 | jointnet | 76,376 | 9.19% | +1.94% | 3.60% | 12.81% | 321:8.25%  322:11.27%  323:3.98%  324:13.26% |
| nn_deep_32x16 | deep-jointnet | 38,344 | 9.34% | +4.67% | 3.80% | 10.15% | 321:9.57%  322:8.77%  323:6.81%  324:12.21% |

## Train-side selection (LOO on post-rule training events 311-320)

Primary rule: fit on the other 9 post-rule events, score the held-out one; pick = lowest mean of the 10 held-out MAPEs.

| config | LOO mean MAPE | LOO std | per-fold |
|---|---|---|---|
| nn_pca_h64 | 9.45% | 4.14% | 18.57 10.73 7.14 9.95 6.01 4.84 8.06 6.10 15.14 7.98 |
| nn_pca_h32 | 9.68% | 4.07% | 17.43 13.11 7.42 10.50 6.37 4.53 8.27 5.80 15.28 8.13 |
| ridge_s0.01 | 9.79% | 3.48% | 17.32 7.16 5.85 10.92 9.44 6.87 8.63 7.94 14.92 8.84 |
| ridge_s0.001 | 9.85% | 3.23% | 16.83 7.81 6.43 11.05 9.47 6.60 8.45 7.92 14.49 9.49 |
| nn_pca_h8_d0.001_e480 | 9.94% | 4.47% | 15.03 18.12 11.83 9.88 5.96 4.53 6.15 6.51 14.58 6.76 |
| nn_pca_h8_d0.001_e240 | 10.31% | 3.90% | 16.58 15.16 11.84 10.80 6.28 4.91 8.63 5.83 14.33 8.77 |
| nn_pca_h16 | 10.37% | 4.18% | 17.81 14.17 12.00 7.91 6.06 5.19 8.64 7.14 16.19 8.55 |
| nn_pca_h8_d0.001_e120 | 10.48% | 4.31% | 17.14 12.87 8.18 13.47 6.22 4.54 7.93 6.03 16.82 11.61 |
| ridge_s0.1 | 10.62% | 4.21% | 19.64 6.55 4.75 11.38 10.02 8.74 11.32 8.33 16.29 9.15 |
| nn_pca_h8_d0.003_e120 | 11.15% | 4.54% | 17.36 17.76 13.54 9.94 5.77 7.57 9.43 5.96 16.71 7.45 |
| nn_pca_k4_h8 | 11.15% | 5.00% | 17.77 16.54 17.74 9.26 5.48 5.85 7.36 7.95 16.49 7.07 |
| nn_pca_h8_d0.003_e480 | 11.21% | 4.66% | 18.15 15.90 15.92 10.71 5.40 6.36 8.10 6.63 16.55 8.43 |
| nn_pca_k8_h8 | 11.24% | 4.83% | 18.02 18.36 14.94 8.58 5.64 6.18 9.83 6.56 16.23 8.05 |
| nn_pca_h8 | 11.31% | 4.94% | 17.55 17.86 15.60 9.49 5.67 5.96 8.17 6.70 17.66 8.43 |
| nn_pca_k32_h8 | 11.33% | 4.74% | 16.80 18.91 15.16 10.90 5.70 5.81 9.35 6.27 16.02 8.36 |
| nn_deep_32x16 | 12.14% | 4.30% | 17.72 15.73 15.80 12.31 6.21 8.24 10.10 7.54 18.65 9.09 |
| nn_deep_16x8 | 12.18% | 5.40% | 20.34 16.92 16.26 7.41 6.93 8.40 11.68 6.89 20.19 6.79 |
| nn_pca_h4 | 12.22% | 4.71% | 18.51 18.41 15.61 10.46 6.38 8.37 9.84 7.52 18.54 8.61 |
| nn_pca_h2 | 12.50% | 4.88% | 19.86 15.59 15.49 11.21 6.74 9.78 11.47 7.51 20.54 6.79 |
| nn_pca_h8_d0.01_e480 | 12.56% | 5.51% | 21.85 17.09 16.71 7.33 6.65 10.55 12.06 7.18 19.55 6.60 |
| nn_pca_h8_d0.01_e120 | 12.67% | 5.56% | 22.33 16.87 15.26 9.24 6.45 9.36 12.29 7.25 20.82 6.87 |
| ridge_s1.0 | 12.73% | 4.79% | 22.56 7.32 6.48 12.97 11.53 12.00 15.28 10.02 19.13 10.02 |
| nn_pca_h8_d0.01_e240 | 12.75% | 5.66% | 22.38 17.08 16.26 8.62 6.43 10.65 11.89 7.24 20.64 6.29 |

**Primary pick: nn_pca_h64** — refit on the full 315-event bank, score on 321-324: MAPE 9.19%  bias 1.94%  final24h 3.60%  rows 312 (per-event 321:8.25%  322:11.27%  323:3.98%  324:13.26%).

Secondary robustness rule: fit on the full bank minus the held-out event (314 events), same 10-fold LOO:

| config | LOO mean MAPE | LOO std | per-fold |
|---|---|---|---|
| ridge_s0.001 | 9.13% | 2.62% | 13.72 6.67 5.39 10.25 9.57 7.10 8.26 7.47 13.39 9.45 |
| nn_pca_h8_d0.001_e480 | 9.16% | 3.73% | 14.17 7.69 6.64 6.16 8.14 7.70 8.48 7.80 18.34 6.50 |
| nn_pca_h8_d0.001_e120 | 9.18% | 4.22% | 12.80 6.56 5.15 6.00 7.87 7.62 9.06 9.15 20.31 7.28 |
| nn_pca_h32 | 9.22% | 3.15% | 14.21 8.54 4.80 8.12 7.65 8.70 8.49 8.84 15.89 6.95 |
| nn_pca_h16 | 9.53% | 3.36% | 16.58 7.11 4.59 8.02 9.67 7.95 9.62 9.59 14.50 7.63 |
| ridge_s0.01 | 9.70% | 2.99% | 15.63 6.75 6.00 11.06 9.82 7.80 8.46 8.30 14.40 8.77 |
| nn_pca_h8_d0.001_e240 | 9.75% | 2.97% | 13.73 6.21 9.40 7.89 8.60 7.84 8.66 11.28 16.33 7.59 |
| nn_pca_h8_d0.003_e120 | 10.03% | 3.37% | 16.04 6.86 6.96 8.76 8.86 8.35 11.06 9.31 16.66 7.42 |
| nn_pca_h64 | 10.05% | 3.51% | 16.42 7.81 5.43 7.91 7.46 14.47 10.34 7.91 14.35 8.44 |
| nn_pca_k4_h8 | 10.15% | 3.24% | 14.98 8.04 6.02 8.14 7.90 10.36 10.62 10.79 16.85 7.84 |
| nn_pca_k8_h8 | 10.19% | 4.06% | 18.84 8.92 5.77 7.50 8.62 8.72 10.13 8.33 17.14 7.89 |
| nn_pca_k32_h8 | 10.36% | 3.91% | 15.20 8.97 5.18 8.15 8.63 8.85 11.09 11.02 19.30 7.21 |
| nn_pca_h8 | 10.45% | 3.65% | 14.33 6.64 7.02 8.11 13.08 10.20 10.24 9.43 18.57 6.92 |
| nn_pca_h8_d0.003_e480 | 10.70% | 4.42% | 18.41 6.99 5.16 8.46 9.47 13.23 10.42 8.71 18.69 7.42 |
| nn_pca_h4 | 10.85% | 3.75% | 16.74 8.05 6.07 8.17 10.36 9.84 12.78 10.34 18.23 7.89 |
| ridge_s0.1 | 10.99% | 3.71% | 19.31 7.28 6.62 11.47 10.38 9.43 11.10 9.35 16.01 8.93 |
| nn_deep_32x16 | 12.24% | 5.27% | 25.55 8.49 7.32 9.99 9.77 9.92 11.41 9.97 18.49 11.47 |
| nn_pca_h8_d0.01_e120 | 12.55% | 4.64% | 23.36 8.69 7.87 11.11 10.64 10.95 13.49 10.76 18.94 9.65 |
| nn_pca_h8_d0.01_e480 | 12.73% | 4.23% | 21.37 8.61 9.01 10.34 10.62 12.43 14.46 10.84 19.70 9.91 |
| nn_pca_h2 | 12.91% | 3.79% | 19.54 10.02 8.84 11.23 10.31 11.91 14.97 11.03 20.18 11.05 |
| nn_pca_h8_d0.01_e240 | 13.02% | 4.76% | 22.24 9.05 7.43 10.94 10.12 12.57 15.53 11.26 20.98 10.06 |
| nn_deep_16x8 | 13.17% | 4.79% | 23.53 9.45 6.85 11.48 10.79 11.33 12.78 11.96 20.57 12.99 |
| ridge_s1.0 | 13.36% | 4.48% | 23.37 8.93 7.67 13.34 12.11 12.22 15.28 11.35 18.79 10.52 |

**Secondary pick: ridge_s0.001** — refit on the full bank, score on 321-324: MAPE 8.20%  bias 3.32%  final24h 3.08%  rows 312 (per-event 321:6.80%  322:10.27%  323:6.01%  324:9.71%).

## Descriptive best on 321-324 (study object, not a tuned claim)

**nn_pca_h8**: MAPE 6.71%  bias 0.80%  final24h 2.78%  rows 312 (per-event 321:5.67%  322:9.34%  323:4.02%  324:7.80%).

## Overfitting signs (in-sample vs 321-324)

- h=2: train 9.78% vs val 7.56% (gap -2.22pp)
- h=4: train 9.03% vs val 8.54% (gap -0.49pp)
- h=8: train 8.88% vs val 6.71% (gap -2.17pp)
- h=16: train 9.24% vs val 7.54% (gap -1.70pp)
- h=32: train 9.59% vs val 7.70% (gap -1.89pp)
- h=64: train 12.81% vs val 9.19% (gap -3.62pp)

In-sample MAPE is non-monotonic but the frozen 321-324 score bottoms out at hidden 8 and degrades steadily past 16 (6.71 -> 7.54 -> 7.70 -> 9.19%); at hidden 64 even the *in-sample* score jumps to 12.81%, so the largest model fails to fit the training bank under the default 240-epoch budget as well as smaller ones — capacity is wasted, not useful.

## Capacity story

- Descriptive minimum on new events sits at hidden 8 (~6.71% MAPE); both smaller (2,4) and larger (16,32,64) widths are worse, and 64 is markedly worse — bigger is not better for post-rule generalization with this frozen feature bank.
- Two hidden layers (deep variants) do not beat the single-layer winner at matched-ish parameter counts (8.15% / 9.34% vs 6.71%).
- PCA rank 16 remains the sweet spot; compression below 8 or above 32 loses either signal or noise control (k4: 7.64%, k8: 7.28%, k32: 6.94%).
- Ridge strength 0.1 beats the frozen default 0.01 descriptively (7.19% vs 7.61%) but still loses to the NN at its best.
- Default decay .003 / epochs 240 is already near the grid optimum at h8; decay .001 is clearly too weak (7.9-8.9%) and longer training at decay .01 degrades.
- Neither train-side LOO rule recovers the descriptive winner: the post-rule-only LOO picks **nn_pca_h64** (321-324: 9.19%) and the 314-event LOO picks **ridge_s0.001** (321-324: 8.20%), while nn_pca_h8 scores 6.71% — both LOO rules rank h8 mid-pack on 311-320. Selection on 10 post-rule events is too noisy to confirm the h8 advantage; the honest headline is the train-side pick score, and the descriptive h8 number should be labeled as such.

## Caveats

- 321-324 is only 4 events (312 rows); per-event swings are a few MAPE points, so the width ranking among {8,16} is within noise.
- The primary LOO trains on 9/10 of the post-rule labels (~9/315 of the bank; same frozen features, Protocol A), so held-out scores are noisier and level-shifted vs the full-bank refit; it is used only for *ranking* configs.
- The depth variant is implemented only in this module; it is not part of the frozen pipeline and should not be promoted without a rerun here.
- All numbers from one harness, one seed (1729); no ensembling.
- Total wall time 717s.
