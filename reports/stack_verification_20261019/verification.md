Stack verification after the workspace refactor

Conclusion: the current stack is a stronger historical point predictor than Fresh overall. The gain on event 325 reproduces in the isolated standalone package and persists over five prespecified initialization seeds. It is not uniformly better by tier, and the four-event 321–324 advantage is small, selected after inspecting that development block, and statistically inconclusive. Ridge retains an advantage at the supplied external checkpoints and near closing.

Independent runtime replay: event 325

Every comparison below uses exactly matching T500/T1000/T1500/T2000 rows, with observations at most one hour old. There are 47 scheduled 08/20/21 CN checkpoints; six have no eligible fresh observations, leaving 41 checkpoints × four tiers = 164 points. External-24 is the subset of six supplied issue times with fresh input. The isolated runtime imported only the shipped distribution, explicitly blocking PyTorch. Final targets are closing tracker proxies, not independently verified official finals.

| Matched subset | Points per model | Fresh MAPE | Ridge MAPE | Stack MAPE |
|---|---:|---:|---:|---:|
| Full event | 164 | 10.63% | 8.79% | 7.55% |
| 08:00 only | 52 | 12.24% | 9.14% | 8.12% |
| Supplied external timestamps | 24 | 7.99% | 4.31% | 4.50% |
| Final 24 hours | 12 | 3.95% | 2.19% | 2.43% |

| Full-event statistic | Fresh | Ridge | Stack |
|---|---:|---:|---:|
| Mean signed relative error | +4.59% | -6.30% | -2.84% |
| 90th percentile absolute relative error | 20.98% | 18.76% | 12.64% |
| Worst absolute relative error | 41.57% | 50.36% | 48.62% |

Stack wins 96/164 paired focal observations against Fresh. The lower average and 90th percentile do not imply a lower worst-case error: all models struggle with T2000 at the first issue, where Fresh is better.

| Tier | Fresh MAPE | Ridge MAPE | Stack MAPE |
|---|---:|---:|---:|
| T500 | 6.90% | 6.34% | 6.94% |
| T1000 | 9.79% | 9.17% | 5.63% |
| T1500 | 17.47% | 6.94% | 7.11% |
| T2000 | 8.38% | 12.71% | 10.50% |

| Hours remaining | Fresh MAPE | Ridge MAPE | Stack MAPE |
|---|---:|---:|---:|
| 0-24h | 3.95% | 2.19% | 2.43% |
| 24-72h | 10.06% | 4.65% | 5.23% |
| 72-144h | 12.44% | 5.23% | 5.83% |
| 144-240h | 10.47% | 5.41% | 4.99% |
| >240h | 11.47% | 14.46% | 11.23% |

Supplementary all-labeled-tier comparison (not the standing headline): matched 870 observations per model, Fresh 13.30%, Ridge 12.39%, Stack 11.69% MAPE.

Recent historical research comparison, 321–324

These are earlier fits stopped before event 321, not the current production models. The production stack supervises on 321–324 and cannot honestly test on them. We reproduced the original rule+explicit × post20 research recipe and matched all 312 observations to the frozen Fresh research CSV.

| Event | Fresh MAPE | Stack MAPE |
|---|---:|---:|
| 321 | 6.00% | 5.31% |
| 322 | 9.09% | 8.42% |
| 323 | 3.84% | 3.09% |
| 324 | 4.91% | 5.73% |
| Equal-event average | 5.96% | 5.64% |

Stack minus Fresh: -0.327 percentage points; exhaustive four-event paired bootstrap 95% interval [-0.733, +0.436] pp. It includes zero. This is descriptive, not adjusted for the feature/scope search. Stack wins three events; Fresh wins 324.

Fixed-recipe robustness, not another hyperparameter selection

Prespecified seeds 1729–1733 were all reported. Compression, features, scope, architecture, epochs and loss were fixed; no seed or ensemble was promoted.

| Seed | Stack research MAPE, 321–324 | Stack production MAPE, 325 |
|---|---:|---:|
| 1729 | 5.636% | 7.545% |
| 1730 | 6.004% | 8.072% |
| 1731 | 6.003% | 7.974% |
| 1732 | 5.677% | 7.298% |
| 1733 | 5.767% | 7.236% |

Research seed mean 5.818% (range 5.636–6.004%) versus Fresh 5.963%. Three of five seeds beat Fresh on this inspected four-event block. Production seed mean 7.625% (range 7.236–8.072%) versus Fresh 10.635%; all five beat Fresh on 325. This establishes local initialization stability, not robustness across many independent future events.

Fitting, features and isolation checks

- Current stack: exactly 30 supervised events, 12 actual post-rule (311, 314–324) plus 18 most recent older events. Reference pool is 319 events through 324. Fresh supervises only the 12 post-rule events, so the comparison changes both learning method and training scope.
- Retraining the documented current recipe reproduced mean/scale/a/b/v/d/trained exactly: maximum absolute difference zero for every saved parameter array. 15,132 supervised tier/checkpoint cells; 600 selected inputs, hidden width 8, 5,024 trainable parameters.
- Training cache excludes event 325/326 cases. All inspected same-rank training references have outcomes available before the target event starts. All 319 reference labels were available before the production fit cutoff.
- Poisoning 325/326 final labels leaves training targets, weights and all fitted parameters unchanged. Runtime future-tail, late-publication and own-final poisoning leaves forecasts unchanged for all three models on both 325 and 326.
- The actual isolated packaged predictions match the independent harness and padded JointPredictor paths on every available tier: 24/24 for 325 and 22/22 for 326.
- The 600 selected columns are explicit normalized progress/remaining/current-fraction/presence inputs (96) plus rule/type/reference-support inputs (504). The 16 PCs and 552 direct daily-summary columns are excluded. The feature builder still computes them, but they do not enter the stack net. Historical completion priors remain part of the forecast.
- Stack has no calibrated sigma; this verifies point accuracy, not improved probabilistic calibration. Fresh and Ridge retain their own small-sample uncertainty estimates.
- The model assets and protected evaluation ledger were unchanged. The current package checksum verifier passes all 71 files.

Pipeline and evidence issues found (left unchanged by this verification)

1. `dist/bandori/stack_runtime.py:59`: any explicit `--max-stale-hours` value calls `config.replace(...)`, but Config has no replace method. Verified failure: AttributeError. Use dataclasses.replace(config, ...) and add a real CLI regression check before advertising stale-input support.
2. `README.md:63`: the documented retraining command uses train_postrule30_test325.py, which actually fits 32 events. The current package was reproduced from train_post12_30_test325.py (30 events); the displayed eval_325.py command also evaluates an earlier scope, not that exact production recipe.
3. `README.md:32–33`: “325 never seen in selection” overstates the evidence. Several scopes were compared on 325 (original post20, all pre325, 32-event version, corrected 30-event version), and the deployment/export includes its outcomes and asserts their published errors before writing assets. Targets do not leak into the coefficient fit, but 325 is already a development/assessment event, not an untouched final test.
4. Small 321–324 gains carry model-search inflation. Do not assign the production package the 5.64% research score or call its improvement statistically established. Event 326 remains the next closing outcome for forecasts frozen beforehand; one more event will still give limited evidence.

Reproduction

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -B reports/stack_verification_20261019/runtime_audit.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B reports/stack_verification_20261019/research_audit.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B dist/tools/verify_stack_parity.py --out-dir reports/stack_verification_20261019/runtime_outputs --padded-dir /tmp/bandori_stack_review_parity_20261019
.venv/bin/python -B dist/verify_package.py
```

No deployment routing, weights, hyperparameters, or existing reports were changed. Only this verification directory was added.
