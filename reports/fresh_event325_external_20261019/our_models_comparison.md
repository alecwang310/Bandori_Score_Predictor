Event 325: Fresh versus saved earlier models

Exact same eight supplied external timestamps; T500/T1000/T1500/T2000. No retraining, tuning, or selection for this comparison. Each point forecast is rounded upward to an integer consistently with its runtime. Closing tracker proxies are identical to the previous external comparison.

| Model | All 32 points MAPE | Fresh-only 24 points MAPE | Fresh-only signed bias |
|---|---:|---:|---:|
| Rule-aware ridge, post-rule 3x | 4.69% | 4.31% | -1.50% |
| Earlier PCA NN + 24h observer | 7.32% | 7.21% | -5.79% |
| Earlier rich-feature NN | 7.64% | 7.36% | -5.40% |
| Earlier ridge | 7.69% | 7.51% | -6.69% |
| Fresh | 8.69% | 7.99% | +6.55% |
| Earlier PCA NN | 9.26% | 9.04% | -6.65% |
| Earlier calibrated statistical + 24h observer | 9.56% | 9.33% | -7.53% |
| Earlier calibrated statistical | 10.43% | 10.62% | -8.46% |
| Earlier simple baseline | 10.84% | 10.33% | -7.38% |

All points include two tracker-gap checkpoints with observations approximately 23 and 46 hours old. Fresh-only excludes them for every model. All matched observation ages were checked against Fresh. Earlier control and observer models are different revisions from the rule-aware 3x suite; these rows compare saved deployed approaches, not isolated architecture effects. All saved earlier fits exclude 325/326 and use 319 eligible earlier events. Fresh supervises only the 12 actual post-rule events and uses 319 curve-reference events.

| Hours remaining | Rule-aware ridge 3x | Earlier ridge | PCA NN + 24h | Fresh |
|---:|---:|---:|---:|---:|
| 219 | 5.04% | 11.79% | 9.64% | 4.50% |
| 195 | 5.55% | 11.06% | 9.44% | 7.90% |
| 171 | 5.36% | 8.47% | 8.08% | 11.58% |
| 146 | 5.80% | 8.27% | 7.65% | 10.88% |
| 123 | 5.84% | 8.25% | 7.63% | 10.69% |
| 99 | 4.34% | 7.07% | 7.96% | 12.48% |
| 27 | 4.02% | 4.16% | 5.00% | 9.34% |
| 3 | 1.55% | 2.49% | 3.13% | 2.15% |

All saved variants (descriptive, including ensembles; no new winner selected):

| Saved variant | All-points MAPE | Fresh-only MAPE |
|---|---:|---:|
| Fresh | 8.69% | 7.99% |
| control/bagged | 9.15% | 8.82% |
| control/baseline | 10.84% | 10.33% |
| control/calibrated_stat | 10.43% | 10.62% |
| control/collection_mean | 8.59% | 8.53% |
| control/collection_median | 8.63% | 8.58% |
| control/collection_trimmed | 8.64% | 8.57% |
| control/hierarchical | 10.09% | 10.09% |
| control/neural_mean | 8.73% | 8.52% |
| control/neural_winner | 7.64% | 7.36% |
| control/nn_minimal | 9.59% | 9.37% |
| control/nn_pca | 9.26% | 9.04% |
| control/nn_rich | 7.64% | 7.36% |
| control/nn_standard | 9.04% | 8.85% |
| control/ridge | 7.69% | 7.51% |
| control/statistical_mean | 8.48% | 8.58% |
| control/trajectory | 8.94% | 8.98% |
| observer/bagged | 9.15% | 8.82% |
| observer/baseline | 9.59% | 9.06% |
| observer/calibrated_stat | 9.56% | 9.33% |
| observer/collection_mean | 8.39% | 8.17% |
| observer/collection_median | 8.88% | 8.58% |
| observer/collection_trimmed | 8.67% | 8.37% |
| observer/control_champion | 10.43% | 10.62% |
| observer/hierarchical | 9.41% | 9.46% |
| observer/neural_mean | 8.31% | 8.14% |
| observer/neural_winner | 7.64% | 7.36% |
| observer/nn_minimal | 9.59% | 9.37% |
| observer/nn_pca | 7.32% | 7.21% |
| observer/nn_rich | 7.64% | 7.36% |
| observer/nn_standard | 9.04% | 8.85% |
| observer/pace24_flat | 9.77% | 9.45% |
| observer/ridge | 7.69% | 7.51% |
| observer/statistical_mean | 8.39% | 8.29% |
| observer/tcn24 | 7.64% | 7.36% |
| observer/trajectory | 9.03% | 8.49% |
| rule_aware_3x/ridge | 4.69% | 4.31% |

No raw LSTM/transformer event-325 matched forecasts were present in these saved prediction files; their event-321–324 scores must not be inserted into this comparison. This is one previously examined, unusually long event, not evidence of a universal ordering. No fitted assets or sealed Fresh settings changed.
