Matched-time point forecast comparison

Event 325; single preselected system: **ridge**. Refit on 319 eligible earlier events, excluding [325, 326]. Two-year exponential age decay retained.

Duration context: this event lasts 373.00h (15.54 days); the latest thirty training events have median 202.00h, and the longest training event lasts 346.00h. Only 2 training events fall within 20% of its duration, including 1 of the same event type. Matched timestamps permit an event-specific comparison; this single event does not establish performance on ordinary-duration events.

Earlier protected events were reused only under the user’s explicit instruction. The original holdout ledger remains unchanged. Choices, gains and ensemble memberships were frozen before this refit.

Recorded closing scores (tracker proxies): T500 = 12,647,536, T1000 = 9,543,389, T1500 = 8,516,669, T2000 = 7,002,130.

| Hours remaining | CN issue time | T500 | T1000 | T1500 | T2000 |
|---:|---|---:|---:|---:|---:|
| 219 | 2026-10-05T20:00:00+08:00 | 12,442,858 | 8,828,658 | 8,560,907 | 6,265,320 |
| 195 | 2026-10-06T20:00:00+08:00 | 12,725,416 | 8,711,703 | 8,711,703 | 6,262,316 |
| 171 | 2026-10-07T20:00:00+08:00 | 13,009,701 | 9,096,062 | 9,096,062 | 6,506,471 |
| 146 † | 2026-10-08T21:00:00+08:00 | 13,235,573 | 9,252,274 | 9,252,274 | 6,522,819 |
| 123 † | 2026-10-09T20:00:00+08:00 | 13,266,544 | 9,256,505 | 9,256,505 | 6,526,906 |
| 99 | 2026-10-10T20:00:00+08:00 | 12,929,372 | 9,288,255 | 8,867,223 | 6,417,879 |
| 27 | 2026-10-13T20:00:00+08:00 | 13,134,698 | 9,742,558 | 9,223,599 | 6,874,076 |
| 3 | 2026-10-14T20:00:00+08:00 | 12,226,083 | 9,412,293 | 8,466,880 | 6,939,788 |

† Last-known-score forecasts across a tracker gap, not fresh observations: 146h: up to 22.75h old, 123h: up to 45.75h old. No later score was interpolated or used. This gap does not establish a game-server outage.

| Hours remaining | Our MAPE | MASHIRO MAPE | RUI MAPE |
|---:|---:|---:|---:|
| 219 | 5.04% | 18.79% | 24.81% |
| 195 | 5.55% | 19.31% | 20.45% |
| 171 | 5.36% | 15.23% | 19.29% |
| 146 | 5.80% | 15.07% | 16.21% |
| 123 | 5.84% | 14.01% | 14.58% |
| 99 | 4.34% | 11.25% | 13.86% |
| 27 | 4.02% | 2.83% | 3.91% |
| 3 | 1.55% | 2.96% | 3.84% |

| Model | Scored points | MAPE | Signed bias |
|---|---:|---:|---:|
| Ours | 32 | 4.69% | -0.90% |
| MASHIRO | 32 | 12.43% | -11.67% |
| RUI | 32 | 14.62% | -14.58% |
| Nanami | 8 | 4.02% | +1.89% |

Restricting every model to timestamps/tiers with fresh inputs for our forecast: Ours 4.31% (24 points), MASHIRO 11.73% (24 points), RUI 14.36% (24 points) MAPE.

Nanami has 2 supplied timestamps. On the same 8 tier/timestamp points: Ours 2.78%, MASHIRO 2.89%, RUI 3.87%, Nanami 4.02% MAPE.

Retrospective comparison on a previously inspected event, not a fresh test
Only observations known by each exact issue time enter prediction; missing historical publication times are assumed equal to observation times
Recorded closing tracker proxies are not independently verified official finals
