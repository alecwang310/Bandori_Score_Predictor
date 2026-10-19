# Reorganization plan — unified fresh/ridge/stack workspace (2026-10-19)

## Layout (after this reorg)

```
old/                      # archive: everything else from before (old dists, zips, one-off
                          #   audit/build/compare scripts, superseded runs/reports, docs)
dist/                     # THE distribution folder (built from old/dist_fresh) —
                          #   fresh + ridge + stack, standardized input/output, swappable models
bandori/                  # shared code library (training/eval for the 3 models + infra)
fresh_predictor/          # Fresh model source (research side; dist carries its own copy)
tests/                    # test suite
collect_bandori_data.py   # data collection infrastructure
requirements.txt
evaluation_registry.json  # protected holdout ledger (read-only)
data/                     # frozen experiment bank (training/eval)
bandori_cn_data/          # live CN data (runtime input)
runs/                     # active fresh/ridge/stack runs only (10 dirs)
reports/                  # active fresh/ridge/stack reports only
```

Tooling that never moves: `.venv`, `.runtime`, `.git`, `.gitattributes`, `.gitignore`.
All commands run from the repo root with `PYTHONPATH=.` (or as modules).

## The three models

| model | code | assets | fit |
|---|---|---|---|
| fresh | `fresh_predictor/` | `runs/fresh_post_rule_v1_20261019/production_model/` | 12 strictly-post-rule events (311, 314-324), 120 coefs, strength 0.01 |
| ridge | `bandori/advanced_statistics.py` (JointRidge) + `bandori/joint_runtime.py` | `runs/joint_production_20261019/model/` (or old/dist_fresh/long_event_model) | 319 events through 324, PCA features, gain 0.5 |
| stack | `bandori/joint_runtime.py` (JointPredictor) | TRAIN via harness (below) | post12_30: **30 events exactly** = 12 strictly-post-rule (311, 314-324; 312/313 are pre-rule starts) + 18 most recent older; rule+explicit 600 cols; JointNet h8 seed 1729, 240 epochs, gain 1.0; fit_cutoff = 325 start; weighted cells 15,132. Use `runs/sweep_stack_20261019/train_post12_30_test325.py` (NOT train_postrule30_*, which fits 32 events). Verified on held-out 325: 4.50% external-24 / 7.55% standing-164 |

Stack training recipe (already validated — `runs/sweep_stack_20261019/train_post12_30_test325.py`
scores 4.50% external-24 / 7.55% standing-164 on held-out 325): use
`bandori/sweep_harness.py` (`load_bank`, `prepare_fit`) + `block_cols({'rule','expl'})`
from `bandori/sweep_features.py` + `fit_jointnet`; export weights to the dist
package (mean/scale/a/b/v/d + 600-col layout descriptor). Runtime must be
NumPy-only (no torch at inference).

## Unified dist design (`dist/`)

Entry point `dist/predict.py` (keep `run.sh`/`run.command`/`run.bat` working):

```
python3 predict.py [--model auto|fresh|ridge|stack] [--event N|latest]
  [--as-of TIME|now] [--snapshot path.json.gz | --data /path/to/bandori_cn_data]
  [--report focus|full] [--output-prefix /path/name] [--no-save] [--max-stale-hours H]
```

* **Input (standardized)**: identical for every model — a snapshot json.gz OR a
  live data dir, plus event id and as-of time (CN time for naive timestamps).
  `--model auto` keeps the user-specified routing rule (recorded in
  `ROUTING.json`): total event duration > 240h -> ridge, <= 240h -> fresh;
  the rule must be registry-driven so swapping the auto choice is one line.
* **Output (standardized)**: one JSON schema for all models —
  `{api_version, model:{name,kind,selection,reason}, event_id, event_name,
  event_type, checkpoint_cn, closes_cn, hours_remaining, total_duration_hours,
  routing:{rule,threshold,selected}, predictions:[{tier,status,final,current,
  observed_at,score_age_hours,sigma:{value,relative,method}|null,detail:{...}}],
  limitations:[...]}` and the matching TXT bulletin (focus tiers
  T100/T500/T1000/T1500/T2000 by default; `--report full` = all tiers).
  `detail` may carry model-specific fields (anchor/pace for fresh, multiplier
  for stack/ridge).
* **Model packages** `dist/models/<name>/` with `manifest.json`
  (`{name, kind, api_version, fit_cutoff, supervised_events, assets:{file:sha256},
  training_provenance, limitations}`) + asset files. `dist/models/registry.json`
  lists models + the auto-routing choice. **Swapping a model** = drop a package
  + one registry line; nothing else changes.
* Fresh prediction semantics unchanged (`fresh_predictor` runtime copied into
  dist); ridge semantics unchanged (`1+gain·delta`, gain 0.5); stack = `exp(delta)`,
  gain 1.0 (use `bandori/joint_runtime.py` semantics and its feature-parity path).
* Sigma: fresh keeps its calibrated sigma; ridge keeps its calibration; stack
  may emit `sigma: null` with an explicit `uncalibrated` note (schema allows it).

## Acceptance criteria

1. `dist/` is self-contained (only stdlib + numpy; `requirements.txt` = numpy).
2. Regression: fresh on `examples/event_326_20261017_2000.json.gz` and
   `examples/event_325_20261014_2000.json.gz` reproduces the old dist outputs
   (bit-identical finals, modulo timestamp/generation fields).
3. Ridge and stack both run end-to-end on both examples and on the live-data
   path (`--data bandori_cn_data`); stack finals must match
   `bandori/joint_runtime`/harness predictions for the same cutoff.
4. `python3 -m unittest discover -s tests` green from the workspace root
   (fix path references if the move broke any; do not weaken assertions).
5. `dist/README.md` documents install, all flags, the unified JSON schema, the
   routing rule with its provenance ("specified by the user after viewing 325
   results; 325/326 excluded from fitting"), and the swap procedure.
6. PACKAGE_SHA256.json regenerated over the final dist contents and verified.
7. Nothing under `old/` is modified except by copying files out of it.
8. The protected ledger `evaluation_registry.json` is unchanged (verify digest).
