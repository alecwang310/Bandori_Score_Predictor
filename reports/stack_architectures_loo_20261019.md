Stack-like architecture study: isolated pre-321 LOO selection

Frozen global winner: **mlp32_standard**. Selection was sealed before loading 321–324 outcomes.

LOO holds out actual post-rule events 311, 314–320, one event at a time. The fixed 30-event supervised scope becomes 29 per fold. All held curves, labels and metadata are removed from training references; affected later training priors are rebuilt. Training-only scaling/masks are refit per fold. Older reference history remains available, with 730-day exponential decay and post-rule 3x training weights. LOO can fit on later pre-321 events, so its error is not a live chronological forecast estimate.

Ten architectures × three regularization/training settings × eight folds × two prespecified seeds. The selection metric scores the mean final forecast of the two seeds, with equal event weight, at T500/T1000/T1500/T2000. All models output the full 24 tiers jointly.

| Preselected family winner | Parameters | LOO MAPE | Test MAPE, 321–324 | Test bias | Final 24h MAPE |
|---|---:|---:|---:|---:|---:|
| mlp32_standard **GLOBAL PICK** | 20024 | 8.02% | 6.02% | -0.34% | 2.23% |
| mlp4_short | 2524 | 9.95% | 6.04% | +0.35% | 2.37% |
| mlp8_standard | 5024 | 9.04% | 5.77% | -0.86% | 2.27% |
| mlp16_standard | 10024 | 8.32% | 5.99% | -0.98% | 2.25% |
| deep16_8_short | 9968 | 10.52% | 5.97% | +0.36% | 2.60% |
| deep32_8_short | 19712 | 10.08% | 5.66% | -0.35% | 2.48% |
| local_skip8_short | 5120 | 8.92% | 6.00% | -0.00% | 2.27% |
| local_skip16_standard | 10120 | 8.22% | 6.15% | -0.02% | 2.23% |
| tier_shared8_standard | 448 | 9.03% | 6.18% | -2.06% | 2.64% |
| tier_shared16_standard | 872 | 9.18% | 6.20% | -2.03% | 2.67% |

Fixed original-style stack, seed 1729: 5.64% test MAPE. Frozen Fresh research reference: 5.96% on the identical 312 rows.

| Event | Selected model MAPE |
|---|---:|
| 321 | 5.32% |
| 322 | 9.12% |
| 323 | 3.37% |
| 324 | 6.27% |

Selected minus Fresh: +0.057 percentage points; four-event paired bootstrap 95% interval [-0.577, +0.899] pp. This interval is descriptive and does not erase prior inspection of 321–324.

The ten per-family choices and the global winner were chosen using LOO only. No settings or deployed assets were changed after the final test. This study does not establish an untouched future generalization result; the architecture family and representation were informed by earlier experiments.

Event 318 has no eligible final-24h observations in the LOO bank. LOO closing-day diagnostics therefore cover seven events; the overall selection metric still covers all eight. A diagnostic-only aggregation fix was recorded before selection and before opening test data, preserving all fitted results and the original protocol.

Training source contains only 315 events before 321. Test 321–324 and excluded 325/326 are absent from all tuning data views. The held-event poisoning audit is in preparation_audit.json; source/code hashes are in protocol.json. Selection and final assessment are separate files; selected weights are saved under models/.

Reproduction:

```sh
python -m bandori.stack_loo_experiment prepare --run runs/stack_architectures_loo_20261019_v1
python -m bandori.stack_loo_experiment tune --run runs/stack_architectures_loo_20261019_v1
python -m bandori.stack_loo_experiment test --run runs/stack_architectures_loo_20261019_v1
```

Outcome: the honest LOO-selected model did not improve overall test accuracy. Its 6.02% is slightly worse than Fresh (5.96%), the original-width two-seed model (5.77%), and the existing single-seed stack reference (5.64%). The deeper 32×8 family scored 5.66%, but had 10.08% LOO error and was not the global selection; it must not be retroactively promoted using these test results. The selected model improves final-day error only marginally against the old stack (2.23% versus 2.26%).

[Plots](stack_architectures_loo_20261019.png) · [Selection](../runs/stack_architectures_loo_20261019_v1/selection.json) · [Final audit](../runs/stack_architectures_loo_20261019_v1/final_audit.json)
