# Tuned nn_pca production model (2026-10-19)

## Frozen parity

The runtime feature path (`ridge_inputs` with `clock=None`) was asserted elementwise identical to the frozen harness `finish_context` pca layout on the matched validation cases, and the frozen nn_pca baseline was reproduced (6.71% MAPE).

## LOO hyperparameter search (events 311-320 only)

Grid: hidden [4, 8, 16] x epochs [120, 240, 480] x lr [0.003, 0.01, 0.03] x weight_decay [0.001, 0.003, 0.01] = 81 configs x 10 leave-one-out folds; gain chosen post-hoc from [0.5, 0.75, 1.0, 1.25] per config (no refit).
Best: hidden=16 epochs=480 lr=0.003 decay=0.001 gain=1.25 — LOO MAPE 7.05% (bias -0.23%).
Frozen config (h8/e240/lr.01/decay.003/g1) LOO MAPE 9.43%.

| hidden | epochs | lr | decay | gain | LOO MAPE | LOO bias |
|---|---|---|---|---|---|---|
| 16 | 480 | 0.003 | 0.001 | 1.25 | 7.05% | -0.23% |
| 16 | 240 | 0.003 | 0.001 | 1.25 | 7.09% | -0.44% |
| 16 | 240 | 0.01 | 0.001 | 1.25 | 7.14% | -0.63% |
| 8 | 480 | 0.003 | 0.001 | 1.25 | 7.52% | -0.13% |
| 16 | 120 | 0.003 | 0.001 | 1.25 | 7.58% | -0.60% |
| 16 | 120 | 0.01 | 0.001 | 1.25 | 7.58% | -0.27% |
| 16 | 480 | 0.003 | 0.001 | 1.0 | 7.61% | -0.90% |
| 16 | 240 | 0.01 | 0.001 | 1.0 | 7.64% | -1.21% |
| 8 | 480 | 0.01 | 0.001 | 1.25 | 7.68% | +0.47% |
| 16 | 240 | 0.003 | 0.001 | 1.0 | 7.73% | -1.04% |

Selection uses LOO event-balanced focal MAPE only; events 321-324 never enter selection.

## Final fits on 321-324 (descriptive)

Production scope post_rule+20recent (30 supervised events, full 315 reference pool): MAPE 9.56%  bias 3.31%  final24h 2.59%  rows 312  321:7.98%  322:10.45%  323:4.37%  324:15.45%
Full-315 supervised scope, same tuned config: MAPE 8.59%  bias 1.40%  final24h 2.33%  rows 312  321:6.12%  322:10.59%  323:6.91%  324:10.73%

| config | scope | h/ep/lr/dec/gain | LOO MAPE | 321-324 MAPE | bias | final24h |
|---|---|---|---|---|---|---|
| loo_best | post_rule+20recent | 16/480/0.003/0.001/g1.25 | 7.1% | 9.6% | 3.3% | 2.6% |
| loo_best | full_315 | 16/480/0.003/0.001/g1.25 | 7.1% | 8.6% | 1.4% | 2.3% |
| loo_rank2 | post_rule+20recent | 16/240/0.003/0.001/g1.25 | 7.1% | 9.6% | 4.0% | 2.6% |
| loo_rank2 | full_315 | 16/240/0.003/0.001/g1.25 | 7.1% | 9.1% | 1.7% | 2.5% |
| loo_rank3 | post_rule+20recent | 16/240/0.01/0.001/g1.25 | 7.1% | 8.1% | 1.6% | 2.7% |
| loo_rank3 | full_315 | 16/240/0.01/0.001/g1.25 | 7.1% | 8.8% | 2.8% | 2.4% |
| frozen_config | post_rule+20recent | 8/240/0.01/0.003/g1.0 | 9.4% | 6.4% | 0.2% | 2.4% |
| frozen_config | full_315 | 8/240/0.01/0.003/g1.0 | 9.4% | 6.7% | 0.8% | 2.8% |

The LOO winner does **not** transfer to the fresh events: LOO strongly favored it (7.05% vs 9.43% for the frozen config) but on 321-324 the frozen config is clearly better (6.41% at the sweep-winning scope). With only 10 LOO clusters the selection signal is noisy; this is disclosed, not corrected on the holdout.

Packaged model = LOO-selected config with scope **full_315** (basis: descriptive 321-324 comparison of the tuned config; hyperparameters from LOO only).

## Sigma approximation (stdv)

Calibration = out-of-fold residual RMS on the 10 LOO predictions of events 311-320 (event-balanced, 730-day decayed, horizon buckets and clean/gap scenarios exactly per `forecast_uncertainty` conventions; sigma = forecast x RMS; T1/T10 borrow T20).
In-sample OOF coverage of +/-1 sigma: 72.7% (5348 rows; focal 72.7% on 969).
Same sigma on 321-324 point forecasts: 62.7% (1532 rows; focal 59.0% on 312). Coverage below the nominal 68% on fresh events means the sigma is somewhat narrow there.
Seed-ensemble epistemic spread (8 seeds, production scope): median relative spread 1.05% of the point; this is reported as a secondary diagnostic — combined stdv would be sqrt((forecast*RMS)^2 + ensemble_std^2).

## Packaging and demo

Model directory: `runs/joint_production_20261019/model` — kind `bandori_cn_joint_deployment_v1`, reference pool = full 315 fold-train events, supervised scope = 315 events, gain=1.25, `JointPredictor` loads only NumPy artifacts (joint_model.npz, compression.npz, input_masks.npz, reference bank, calibration).
Demo event 326 at 2026-10-17 20:00 CN (snapshot): T1=55541460±20098156; T10=42501675±15379598; T20=42501675±15379598; T30=25058152±5186341; T40=16983237±3289470; T50=13400633±2715829; T100=8299796±1822502; T200=4351205±632735; T300=3829073±586903; T400=3829073±603773; T500=3587391±597311; T1000=1808654±208437; T1500=1808654±268359; T2000=1502171±103337; T3000=991668±82347; T4000=797662±61864; T5000=714523±59562; T10000=370166±33515; T20000=195055±39748; T30000=115784±25497; T40000=66811±18278; T50000=17455±38967

## Caveats

Hyperparameters were selected by LOO on 311-320; the packaged scope (30 vs 315 events) was chosen by the descriptive 321-324 comparison and is disclosed as such.
Sigma calibration events (10) overlap model-family selection; uncertainty is descriptive, not independently calibrated, and carries no coverage guarantee.
Only 8 of events 311-320 start on/after the 2026-05-26 rule change; 312-313 are pre-rule ids inside the spec range.
The LOO OOF sigma is built from prequential cases at issue hours (8, 20, 21); production checkpoints at other hours reuse the same horizon/scenario cells without extra smoothing.
