Stack production sigma approximation — 2026-10-19

Production now defaults to the existing Stack h8/seed1729 model and the concise five-tier report. The network weights, reference curves and point forecasts are unchanged. Explicit `--model auto` retains the >240h Ridge / otherwise Fresh router.

The new sigma is an approximate RMS prediction-error scale, not a centered standard deviation or a demonstrated 68% interval. It includes systematic bias. Physics notation is `1.23(45) * 10^6 = (1.23 ± 0.45) * 10^6`.

Method and separation

- Calibration events: 311, 314, 315, 316, 317, 318, 319, 320: eight actual post-rule events. 312/313 are excluded because their actual starts precede the rule change.
- Each fold fits 29 of the fixed 30 pre-321 supervised events. The held event is absent from all training curves, metadata, labels and references. Later training priors that could have incorporated its outcome are rebuilt. All scaling and feature masks use training rows only.
- Fixed production recipe: 600 rule/explicit features, JointNet h8, seed 1729, 240 epochs, lr .01, weight decay .003, bias penalty 3, gain 1. Training keeps 730-day decay, post-rule 3x weights, and proxy-label weights. No architecture or seed selection is performed here.
- The deployment network is still trained on 30 events through 324, including 321–324. Events 321–326 never enter sigma residual calibration. No training-set fitted errors are used as residuals.
- There are 4,385 labeled residual rows covering 22 tiers, but only eight event clusters. T1/T10 have no same-tier labels, so receive no sigma; T1500 uses its own outcomes.
- Checkpoints are 08:00, 20:00 and 21:00 CN. Other issue times use the same remaining-hour cells. No special time-of-day, outage or type calibration is claimed.

For each tier and remaining-hour bin, let `r = (forecast - truth) / forecast`. Average `r²` within an event, then average those event values with `exp(-age_days/730)` weights. Set `sigma = current_point_forecast * sqrt(weighted_mean_event_r²)`. This forecast denominator permits applying the learned relative scale to a forecast without knowing its final outcome. The common 3x factor for post-rule calibration events cancels during normalization; it is retained in the fold network fits.

Bins are [0,24), [24,48), [48,72), [72,96), [96,144), [144,192), [192,∞). A sparse cell expands symmetrically to neighboring bins until at least four event clusters are available; it never borrows another tier. JSON records the bins, event count, effective count, bias and centered spread. Input older than one hour gets no sigma, even when a caller relaxes point-forecast staleness limits.

Relative sigma by remaining hours

| Remaining hours | T100 | T500 | T1000 | T1500 | T2000 |
|---|---:|---:|---:|---:|---:|
| 0–24 | 3.90% | 5.38% | 4.42% | 3.63% | 4.00% |
| 24–48 | 7.87% | 8.12% | 8.19% | 5.65% | 6.93% |
| 48–72 | 10.67% | 10.15% | 8.93% | 6.78% | 7.82% |
| 72–96 | 13.38% | 12.43% | 9.65% | 7.92% | 8.18% |
| 96–144 | 18.97% | 16.29% | 10.34% | 9.58% | 9.03% |
| 144–192 | 30.69% | 22.23% | 12.97% | 12.75% | 11.63% |
| 192+ | 91.23% | 65.94% | 40.78% | 42.48% | 46.48% |

These are horizon-bin RMS values, not errors at each exact hour. At 144–192h, all five cells contain eight events. The last-day cells contain seven; the 192+h cells contain six to seven.

Saved example: event 326, 2026-10-17 20:00 CN, 171 hours remaining

```text
T100：8.9(27) * 10^6
T500：3.97(88) * 10^6
T1000：1.95(25) * 10^6
T1500：1.95(25) * 10^6
T2000：1.58(18) * 10^6
```

Limits

LOO permits fitting on later pre-321 events and therefore estimates event transfer, not live chronological performance. These events have already influenced model-family experiments, so this is not independent uncertainty validation. Labels include closing-tracker proxies. The production network has more recent training events than the LOO network, but sigma is not reduced on that basis.

Calibration event durations range from about 178 to 226 hours. Event 325 (373h) is outside this support, even near closing time, and its sigma is flagged as extrapolation. No unsupported numeric inflation is invented for long events. Very early, 192+h scales are large; the eight-event sample is too small to fit reliable tail or event-type distributions.

Reproduction and validation

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B -m bandori.stack_sigma_calibration
OPENBLAS_NUM_THREADS=1 PYTHONPATH=tests:. .venv/bin/python -B -m unittest test_stack_sigma -v
python3 dist/verify_package.py --regen
```

The builder checks the frozen LOO protocol and source hashes, verifies disjoint fold membership, compares its fixed-seed focal replay to the archived replay (maximum difference <1e-8), and binds sigma to the deployment weight SHA256 and availability time. The inherited held-outcome poisoning audit is recorded in audit.json. Tests cover event weighting, bias, recency, sparse horizons, unavailable tiers, stale inputs, identity/time guards, default/Enter selection, the opt-in duration router, full/concise reports, and T1500 independence. Production replay compares all 24 tier records on events 325/326 against the earlier package; finals and current scores match exactly.

[Concise example](stack_sigma_20261019/stack_326.txt) · [Unrounded JSON](stack_sigma_20261019/stack_326.json) · [Residuals](../runs/stack_sigma_20261019/residuals.csv) · [Calibration audit](../runs/stack_sigma_20261019/audit.json) · [Production verification](stack_sigma_20261019/production_verification.json)
