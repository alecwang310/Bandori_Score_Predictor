"""Build-time export of the stack model package into dist/models/stack/.

Inference never needs this script: dist/ runs NumPy-only from the frozen
package. Reproduce from the repo root (research library required):

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -u -B dist/tools/export_stack_package.py

Recipe (validated by runs/sweep_stack_20261019/train_post12_30_test325.py):
  * supervised scope = 30 events: 12 strictly-post-rule (311, 314-324; 312/313
    start pre-rule) + the 18 most recent older events;
  * reference/analog pool = 319 events through 324 (fold-train 315 + 321-324);
  * features = finished pca layout (1168 cols), block_cols({'rule','expl'})
    -> 600 cols = explicit block 16..111 + rule block 664..1167;
  * JointNet hidden 8, seed 1729, config epochs 240 / lr 0.01 / decay 0.003,
    gain 1.0 (exp multiplier);
  * fit_cutoff = start of event 325 = 1790647200; 325/326 excluded from fitting.

Before writing anything the script re-runs the held-out 325 test path and
asserts the published scores (external-24 ~4.50%, standing-164 ~7.55%) and the
runtime/harness feature parity (ridge_inputs == finished pca, elementwise).
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bandori import sweep_harness as H
from bandori.experiments import correction_forecast
from bandori.pipeline import digest, save_bank
from bandori.regime_suite import finish_context, make_context
from bandori.ridge_runtime import ridge_inputs
from bandori.role_analog import build_cases
from bandori.sweep_features import block_cols
from collect_bandori_data import atomic_json

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'dist' / 'models' / 'stack'
POST = [311] + list(range(314, 325))          # 12 strictly-post-rule
FIT_CUTOFF = 1790647200.0                     # start of event 325
FOCAL = (500, 1000, 1500, 2000)
PUBLISHED = dict(external_24pts=0.04496268238970286, standing_164rows=0.07545191815010358)
EXPECTED_COLS = np.concatenate([np.arange(16, 112), np.arange(664, 1168)])
GAIN = 1.0

LIMITATIONS = [
    'Single held-out validation: event 325 scored 4.50% (external 24 fresh-input points) and '
    '7.55% (standing 164 focal-tier rows) MAPE; descriptive, not an independent fresh test',
    'Events 325/326 are excluded from fitting (fit_cutoff = start of event 325)',
    'Sigma is uncalibrated: no uncertainty interval is produced for stack forecasts',
    'Hyperparameters (hidden 8, seed 1729, 240 epochs, gain 1.0) are inherited from the frozen '
    'sweep recipe and were not re-selected for this exact 30-event scope',
    'Supervised labels include closing-tracker proxies; RMS-like errors contain systematic bias '
    'and carry no coverage guarantee',
    'Reference/analog pool stops at event 324; forecasts for events inside that pool are rejected',
]


def mape(points):
    errs = [v / label - 1 for _, v, label in points]
    return dict(rows=len(errs), mape=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true', help='overwrite an existing package directory')
    args = ap.parse_args()

    bank = H.load_bank()
    data = bank.data
    assert len(POST) == 12, POST
    older = [e for e in sorted(bank.train_ids, key=lambda e: data.events[e].end, reverse=True)
             if e not in POST]
    supervised = sorted(set(POST) | set(older[:18]))
    assert len(supervised) == 30, supervised
    assert len(set(supervised)) == 30
    bank.fit_cutoff = data.events[325].start
    assert bank.fit_cutoff == FIT_CUTOFF, bank.fit_cutoff
    refs = list(bank.train_ids) + [321, 322, 323, 324]
    assert len(refs) == 319 and len(set(refs)) == 319

    cols = block_cols({'rule', 'expl'})
    assert len(cols) == 600 and np.array_equal(cols, EXPECTED_COLS), \
        (len(cols), cols[:5], cols[-5:])
    print(f'supervised: {len(supervised)} events (POST {POST}); older {sorted(e for e in supervised if e not in POST)}',
          flush=True)

    prep = H.prepare_fit(bank, supervised)
    weighted = int(np.count_nonzero(prep.weights))
    assert weighted == 15132, weighted
    tr = np.ascontiguousarray(prep.matrix['pca'][:, cols])
    model = H.fit_jointnet(tr, prep.targets, prep.weights, prep.bins, bank.config, hidden=8, seed=1729)
    print(f'trained JointNet: input {tr.shape}, weighted cells {weighted}', flush=True)

    # ------------------------------------------------ held-out 325 validation
    cases, failures, _ = build_cases(data, bank.config, [325], refs)
    if failures:
        print(f'325 case failures: {len(failures)}', flush=True)
    ctx = finish_context(make_context(cases, data, bank.config, refs), prep.compression, prep.masks)[0]
    x_rt, index_rt = ridge_inputs(cases, data, bank.config, refs, prep.compression, prep.masks['pca'])
    assert (index_rt == ctx['index']).all(), 'index parity failed'
    assert np.allclose(x_rt, ctx['matrices']['pca'], atol=1e-10), 'runtime/harness feature parity failed'
    print('runtime/harness feature parity ok', flush=True)
    x = np.ascontiguousarray(ctx['matrices']['pca'][:, cols])
    values = correction_forecast(ctx['cases'], ctx['index'], model.predict(x), GAIN, 'nn')

    import csv
    from collect_bandori_data import timestamp as ts
    ext = list(csv.DictReader(open(ROOT / 'reports' / 'fresh_event325_external_20261019' / 'comparison.csv',
                                   encoding='utf-8-sig')))
    ext_pts = {(ts(r['issued_at']), int(r['tier'])) for r in ext if r['fresh_input'] == 'True'}
    idx = {}
    for i, c in enumerate(cases):
        idx.setdefault((int(c.cutoff), c.tier), i)
    sets = {}
    for name, ii in (('external_24pts', [i for k, i in idx.items() if k in ext_pts]),
                     ('standing_164rows', [i for i, c in enumerate(cases) if c.tier in FOCAL])):
        pts = [(ctx['cases'][i], float(values[i]), data.labels[(325, ctx['cases'][i].tier)]) for i in ii]
        sets[name] = mape(pts)
        expected = PUBLISHED[name]
        assert abs(sets[name]['mape'] - expected) < 1e-6, (name, sets[name], expected)
        print(f"325 {name}: MAPE {100*sets[name]['mape']:.2f}% bias {100*sets[name]['bias']:+.2f}% "
              f"rows {sets[name]['rows']}", flush=True)

    # ------------------------------------------------------------ write assets
    if PKG.exists() and any(PKG.iterdir()):
        if not args.force:
            raise SystemExit(f'{PKG} exists; rerun with --force to replace it')
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True, exist_ok=True)
    atomic_json(PKG / 'config.json', bank.config.to_dict())
    prep.compression.save(PKG / 'compression.npz')
    np.savez_compressed(PKG / 'input_masks.npz', pca=prep.masks['pca'])
    model.save(PKG / 'joint_model.npz')
    save_bank(PKG, data, refs, bank.config.tiers)  # reference/analog pool = frozen 319
    bank_json = json.loads((PKG / 'reference_bank.json').read_text())
    assets = ['config.json', 'reference_bank.json', 'reference_bank.npz', 'joint_model.npz',
              'compression.npz', 'input_masks.npz']
    source_sha = digest(PKG / 'joint_model.npz')
    tiers = bank.config.tiers
    deployment = dict(
        kind='bandori_cn_joint_deployment_v1', feature_version='daily_v1', server=3,
        champion='stack_post12_30', gain=GAIN, hidden=8, seed=1729, epochs=240,
        learning_rate=bank.config.learning_rate, weight_decay=bank.config.weight_decay,
        bias_penalty=bank.config.bias_penalty, source_model_sha256=source_sha,
        pca_layout=dict(matrix='pca', total_columns=1168,
                        blocks=dict(explicit=[16, 112], rule=[664, 1168]),
                        selector="bandori.sweep_features.block_cols({'rule','expl'})",
                        selected_columns=600),
        pca_columns=[int(c) for c in cols],
        training_event_ids=refs,           # reference/analog pool, events through 324
        supervised_event_ids=supervised,   # exactly 30: 12 strictly post-rule + 18 recent older
        fit_cutoff=bank.fit_cutoff,
        supervised_targets_by_tier={str(t): int(np.count_nonzero(prep.weights[:, i]))
                                    for i, t in enumerate(tiers)},
        training_targets_by_tier={str(t): int(np.count_nonzero(prep.weights[:, i]))
                                  for i, t in enumerate(tiers)},
        training_durations_hours=[(e['end'] - e['start']) / 3600 for e in bank_json['events']],
        weighted_cells=weighted,
        validation_heldout_event_325=dict(
            scope='held-out event 325 (excluded from fitting); descriptive single holdout',
            external_24pts=sets['external_24pts'], standing_164rows=sets['standing_164rows'],
            ridge_reference=dict(external_24pts=0.04307469034481867, standing_164rows=0.08790822825824854),
            fresh_reference=dict(external_24pts=0.07991951604650417, standing_164rows=0.10634931606797905)),
        recipe_source='runs/sweep_stack_20261019/train_post12_30_test325.py',
        hashes={name: digest(PKG / name) for name in assets})
    atomic_json(PKG / 'deployment.json', deployment)

    manifest = dict(
        name='stack', kind='bandori_cn_joint_deployment_v1', api_version=1,
        fit_cutoff=bank.fit_cutoff, supervised_events=len(supervised),
        assets={name: digest(PKG / name) for name in assets + ['deployment.json']},
        training_provenance=dict(
            recipe='bandori.sweep_harness.prepare_fit + fit_jointnet; exported by '
                   'dist/tools/export_stack_package.py; validated by '
                   'runs/sweep_stack_20261019/train_post12_30_test325.py',
            scope='30 supervised events: 12 strictly-post-rule (311, 314-324) + 18 most recent older '
                  'events; 312/313 are pre-rule starts and are not part of the post-rule block',
            supervised_event_ids=supervised,
            reference_pool='319 events through 324 (fold-train 315 + 321-324); analog/daily features '
                           'use this pool, supervised labels only the 30',
            features='finished pca layout (1168 columns); selected 600 = explicit block cols 16..111 '
                     "+ rule block cols 664..1167 (block_cols({'rule','expl'}))",
            architecture='JointNet hidden 8, seed 1729, 240 epochs, learning_rate 0.01, weight_decay '
                         '0.003, bias_penalty 3, weights mean/scale/a/b/v/d',
            multiplier='exp(gain*delta) with gain 1.0',
            weighted_cells=weighted,
            fit_cutoff_basis='start of event 325 = 1790647200; labels of 321-324 verified available '
                             'before 325; 325/326 excluded from fitting',
            validation=dict(scope='held-out event 325',
                            external_24pts=sets['external_24pts'],
                            standing_164rows=sets['standing_164rows']),
            exported_at=datetime.now(timezone.utc).isoformat()),
        limitations=LIMITATIONS)
    atomic_json(PKG / 'manifest.json', manifest)
    print('wrote package:', PKG, flush=True)
    for name in sorted(p.name for p in PKG.iterdir()):
        print(' ', name, flush=True)


if __name__ == '__main__':
    sys.exit(main())
