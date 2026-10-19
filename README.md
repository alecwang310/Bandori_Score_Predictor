# BANGDREAM

**BANGDREAM Applies Neural Guidance, Delivering Rank Estimates And Margins.**

A recursive acronym: the first term, **BANGDREAM** (BanG Dream!), refers to this
project itself. The initials are **B A N G D R E A M**.

A local research and production repository for BanG Dream! Chinese-server event
score-line forecasts. Predict the final scores jointly from the scores known at
the issue time; report approximate uncertainty by tier and hours remaining.

Production defaults to the frozen **Stack** model and a concise report covering
T100/T500/T1000/T1500/T2000. Use `--report full` for every available tier.
Explicit `--model auto` retains the >240-hour Ridge / otherwise Fresh switch.
Stack sigma uses eight isolated pre-321 post-rule event holdouts with 730-day
age decay; it includes bias and does not guarantee 68% coverage.

## Run an offline example

```sh
git clone https://github.com/alecwang310/Bandori_Score_Predictor.git
cd Bandori_Score_Predictor
python3 -m pip install -r dist/requirements.txt
python3 dist/predict.py --event 326 --as-of '2026-10-17 20:00' \
    --snapshot dist/examples/event_326_20261017_2000.json.gz --no-save
```

For the latest event using an existing downloaded data directory:

```sh
python3 dist/predict.py --event latest --data /path/to/bandori_cn_data
```

The default issue time is the latest 08:00 within the event. Use `--as-of now`
or an explicit China time to change it. See [the distribution manual](dist/README.md)
and [the uncertainty report](reports/stack_sigma_20261019.md).

## What is versioned

- `bandori/`, `fresh_predictor/`, `tests/`: active research and runtime source.
- `dist/`: a self-contained snapshot of `../bandori_predictions_dist`, including
  the frozen production models, calibration and example inputs. Its internal
  paths remain `dist/` to match the existing training tools and tests.
- `reports/`: current results, audits and figures.
- `runs/`: experiment scripts, configuration, selection records and tabular
  results. Large research matrices, model checkpoints and case caches are omitted.
- `evaluation_registry.json`: protected holdout ledger.
- [PROJECT.md](PROJECT.md) and [PLAN.md](PLAN.md): preserved project documentation.
- `SOURCE_SNAPSHOT.json`: source-to-repository paths and hashes of copied files.

This repository starts with the current active snapshot. Downloads, archived
experiments, virtual environments and caches stay in the parent workspace and
are not tracked. The production package runs independently; reproducing research
fits also requires the omitted datasets and experiment arrays. This is a copy,
not a file synchronization link: later parent-workspace edits must be copied in
before committing them here.

## Verification

Check the frozen package and the focused production/uncertainty regression tests:

```sh
python3 dist/verify_package.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=tests:. python3 -B -m unittest test_stack_sigma -v
```

The upstream repository is
[alecwang310/Bandori_Score_Predictor](https://github.com/alecwang310/Bandori_Score_Predictor).
Licensed under the [MIT License](LICENSE).
