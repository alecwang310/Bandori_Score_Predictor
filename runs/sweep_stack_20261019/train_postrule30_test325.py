"""30-event scope with the COMPLETE post-rule block: 311-324 + 18 most recent older.

The sweep's best scope was 30 events (10 post-rule 311-320 + 20 recent older) but
that predates 321-324 being available. For a pre-325 production fit, the natural
30-event scope is all 12 post-rule events (311-324) + the 18 most recent older
events. Same stack recipe (rule+explicit 600 cols, JointNet h8 seed 1729,
240 epochs, gain 1.0, same weights). Fit cutoff = 325 start (causal; 321-324
labels verified available before 325). Test: held-out 325, same two row sets.

Run:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/train_postrule30_test325.py
"""
import csv
import json
from pathlib import Path

import numpy as np

from bandori import sweep_harness as H
from bandori.advanced_statistics import JointRidge
from bandori.analog import reconcile
from bandori.experiments import correction_forecast
from bandori.regime_suite import finish_context, make_context
from bandori.representations import Compression
from bandori.role_analog import build_cases
from bandori.sweep_features import block_cols
from collect_bandori_data import timestamp

FOCAL = (500, 1000, 1500, 2000)
POST = list(range(311, 325))  # all 12 post-rule events
OUT = Path('runs/sweep_stack_20261019/train_postrule30_test325.json')


def mape(points):
    errs = [p[1] / p[2] - 1 for p in points]
    return dict(rows=len(errs), mape=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)))


def main():
    bank = H.load_bank()
    data = bank.data
    start325 = data.events[325].start
    older = [e for e in sorted(bank.train_ids, key=lambda e: data.events[e].end, reverse=True) if e not in POST]
    supervised = sorted(set(POST) | set(older[:18]))  # 30 events: 12 post-rule + 18 recent
    print(f'supervised: {len(supervised)} events; post-rule {sorted(e for e in supervised if e >= 311)}', flush=True)
    print(f'older included: {sorted(e for e in supervised if e < 311)}', flush=True)
    bank.fit_cutoff = start325

    prep = H.prepare_fit(bank, supervised)
    print(f'weighted cells: {int(np.count_nonzero(prep.weights))}, weight mass {float(prep.weights.sum()):.3f}', flush=True)
    cols = block_cols({'rule', 'expl'})
    tr = np.ascontiguousarray(prep.matrix['pca'][:, cols])
    model = H.fit_jointnet(tr, prep.targets, prep.weights, prep.bins, bank.config, hidden=8, seed=1729)

    refs = list(bank.train_ids) + [321, 322, 323, 324]  # 319: causal inference pool
    cases, failures, _ = build_cases(data, bank.config, [325], refs)
    print(f'325 cases: {len(cases)}, failures: {len(failures)}', flush=True)

    ctx = finish_context(make_context(cases, data, bank.config, refs), prep.compression, prep.masks)[0]
    x = np.ascontiguousarray(ctx['matrices']['pca'][:, cols])
    stack_vals = correction_forecast(ctx['cases'], ctx['index'], model.predict(x), 1.0, 'nn')

    m = Path('runs/rule_aware_3x_20261018/full_refit/models')
    ridge = JointRidge.load(str(m / 'ridge.npz'))
    comp = Compression.load(str(m / 'compression.npz'))
    with np.load(m / 'input_masks.npz', allow_pickle=False) as a:
        masks = {k: a[k].copy() for k in a.files}
    ctx_r = finish_context(make_context(cases, data, bank.config, refs), comp, masks)[0]
    ridge_vals = correction_forecast(ctx_r['cases'], ctx_r['index'], ridge.predict(ctx_r['matrices']['pca']), 0.5, 'ridge')
    baseline = reconcile(cases, [c.baseline for c in cases])

    ext = list(csv.DictReader(open('reports/fresh_event325_external_20261019/comparison.csv', encoding='utf-8-sig')))
    ext_pts = {(timestamp(r['issued_at']), int(r['tier'])) for r in ext if r['fresh_input'] == 'True'}
    idx = {}
    for i, c in enumerate(cases):
        idx.setdefault((int(c.cutoff), c.tier), i)
    A = [i for k, i in idx.items() if k in ext_pts]
    B = [i for i, c in enumerate(cases) if c.tier in FOCAL]

    def pick(vals, ii):
        return [(cases[i], float(vals[i]), data.labels[(325, cases[i].tier)]) for i in ii]

    result = dict(model='stack recipe (rule+explicit 600, JointNet h8 seed1729, gain 1.0)',
        scope='30 events: post-rule 311-324 + 18 most recent older', supervised_events=sorted(supervised),
        fit_cutoff=start325, weighted_cells=int(np.count_nonzero(prep.weights)), sets={})
    known = {'stack_post20(311-320+20)': dict(external_24pts=0.0435, standing_164rows=0.0748),
             'stack_all_pre325(319)': dict(external_24pts=0.0614, standing_164rows=0.0885),
             'ridge_full_refit': dict(external_24pts=0.0431, standing_164rows=0.0879),
             'baseline': dict(external_24pts=0.0865, standing_164rows=0.1473)}
    for set_name, ii in (('external_24pts', A), ('standing_164rows', B)):
        res = {'stack_postrule30': mape(pick(stack_vals, ii)),
               'ridge_full_refit': mape(pick(ridge_vals, ii)),
               'baseline': mape(pick(baseline, ii))}
        result['sets'][set_name] = res
        print(f'--- {set_name}', flush=True)
        for k, v in res.items():
            print(f'  {k:18s} MAPE {100*v["mape"]:6.2f}%  bias {100*v["bias"]:+6.2f}%  rows {v["rows"]}', flush=True)
        for k, v in known.items():
            print(f'  {k:26s} MAPE {100*v[set_name]:6.2f}%  (prior runs)', flush=True)
    result['known_comparison'] = known
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
