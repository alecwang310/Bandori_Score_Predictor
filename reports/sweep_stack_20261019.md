# Stacked sweep winners — feature set x training scope

## Results

Scope: post_rule+20recent means ids 311-320 plus the 20 most recent older training events (event end time). rule_only = 504 rule-metadata columns; rule_plus_explicit = 600 columns. Compression and masks refit on each scope (train-only). Seed 1729.

| config | cols | MAPE | bias | final24h | per-event 321/322/323/324 |
| --- | ---: | ---: | ---: | ---: | --- |
| rule_plus_explicit@post20 | 600 | 5.64% | -0.45% | 2.26% | 321:5.31%  322:8.42%  323:3.09%  324:5.73% |
| rule_only@post20 | 504 | 5.72% | -0.75% | 2.34% | 321:5.35%  322:9.29%  323:3.16%  324:5.10% |
| rule_only@post_only | 504 | 5.72% | -0.52% | 2.19% | 321:5.23%  322:8.89%  323:3.38%  324:5.39% |
| rule_only@full | 504 | 6.23% | -1.81% | 2.60% | 321:5.07%  322:10.26%  323:4.33%  324:5.25% |
| rule_plus_explicit@post_only | 600 | 6.23% | +0.31% | 2.20% | 321:5.25%  322:8.54%  323:4.51%  324:6.63% |
| rule_plus_explicit@full | 600 | 6.28% | -0.81% | 2.59% | 321:4.90%  322:9.78%  323:3.67%  324:6.79% |
| full_pca@post20 | 1168 | 6.41% | +0.19% | 2.40% | 321:5.56%  322:9.08%  323:3.72%  324:7.27% |
| rule_only@post20_ridge | 504 | 6.92% | -0.19% | 2.88% | 321:5.21%  322:9.86%  323:6.29%  324:6.31% |

Reference points (frozen protocol, full bank, full pca): nn_pca 6.71%, ridge 7.61%. Single wins: rule_only@full 6.23% (features sweep), full_pca@post20 6.41% (data-scope sweep).
All scores descriptive on 4 test events; rows within an event are dependent.

