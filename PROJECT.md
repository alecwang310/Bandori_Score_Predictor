# Bandori CN score-line forecasting — fresh / ridge / stack

Three forecast models for BanGDream! 国服 event tier score lines, sharing one data
infrastructure and one distribution format. Everything historical is archived
under `old/`; this workspace contains only the active models and their
infrastructure.

## Layout

| Path | What it is |
|---|---|
| `dist/` | **The** distribution folder — fresh + ridge + stack, standardized input/output, swappable model packages (`dist/README.md` is the user manual) |
| `bandori/` | Shared code library: data structures, feature builders, training/evaluation for ridge and stack, sweep harness |
| `fresh_predictor/` | Fresh model source (physics anchors + 120-coef correction) |
| `tests/` | Test suite (`PYTHONPATH=tests:. .venv/bin/python -m unittest discover -s tests -t tests`) |
| `collect_bandori_data.py` | CN data collector (infrastructure) |
| `data/` | Frozen experiment bank used for training/evaluation |
| `bandori_cn_data/` | Live downloaded CN data (runtime input) |
| `runs/` | Active model fits and experiments (fresh, ridge, stack, sweeps) |
| `reports/` | Active evaluation reports (standing convention: MAPE on tiers 500/1000/1500/2000) |
| `old/` | Archive: superseded dists, zips, one-off scripts, older runs and reports |
| `PLAN.md` | The reorganization/design spec this workspace follows |

## The three models

| Model | Shape | Trained on | Held-out event 325 (standing / late-checkpoint rows) |
|---|---|---|---:|
| **stack** | 600 rule+progress features → 5,024-param tanh net, `exp(δ)` on the analog gap | 30 events: 12 strictly-post-rule (311, 314–324) + 18 most recent older | **7.55% / 4.50%** |
| **ridge** | PCA features → per-tier linear correction, `1+0.5·δ` | 319 events through 324 | 8.79% / 4.31% |
| **fresh** | physics anchors (completion/pace) × `(1+δ)`, 120 partially-pooled coefs | 12 strictly-post-rule events (311, 314–324) | 10.63% / 7.99% |

(analog baseline: 14.73% / 8.65%. Event 325 is the only labeled event never seen
in fitting or selection; 321–324 was the earlier comparison set, now inspected.
Event 326 closes 2026-10-24 — the next honest test.)

Known evidence limits: 4-event and 1-event comparisons are descriptive; the
321–324 numbers carry selection inflation; label finals are closing tracker
proxies. Details in `reports/sweep_consolidated_20261019.md` and
`reports/fresh_event325_external_20261019/`.

## Quick start (distribution)

See `dist/README.md` for the full manual. In short:

```sh
cd dist
python3 predict.py --event latest --as-of now            # production Stack + concise report
python3 predict.py --model stack --event 326 --as-of "2026-10-17 20:00"
python3 predict.py --snapshot examples/event_325_20261014_2000.json.gz \
    --event 325 --as-of "2026-10-14 20:00"
```

Input (same for every model): snapshot json.gz or `--data` live dir + event +
as-of time (CN time for naive timestamps). Output (same for every model): one
JSON schema + a text bulletin (focus tiers T100/T500/T1000/T1500/T2000;
`--report full` for all). Swapping a model = one package + one registry line.
Omitting `--model` or pressing Enter selects the registry default, **stack**.
Explicit `--model auto` retains the 240-hour Ridge/Fresh switch.
Stack now reports approximate same-tier/horizon sigma from eight isolated
pre-321 LOO events (730-day decay, bias included), with no 68% coverage guarantee.
See `reports/stack_sigma_20261019.md` for the method and support limits.

## Development

```sh
python3 -m pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -t tests
PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/train_postrule30_test325.py   # retrain stack
PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/eval_325.py                    # 325 holdout eval
```

Standing reporting convention (from 2026-10-19): reported MAPE is event-balanced
on clean checkpoints of the evaluation events, counting **only tiers 500, 1000,
1500, 2000**; companion statistics use identical rows. `evaluation_registry.json`
is the protected holdout ledger — holdouts never enter fitting or selection.
