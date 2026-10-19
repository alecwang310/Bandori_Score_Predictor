"""Production-style refit: stack recipe on ALL data before event 325, tested on 325.

Same algorithm as the sweep stack winner (rule+explicit 600-col features, JointNet
hidden 8, seed 1729, 240 epochs, gain 1.0, same weights recipe), but supervised on
every event labeled before 325 (all 319 bank events through 324, including the
post-rule block 311-324). Fit cutoff = event 325's start so event-age weights and
label eligibility are causally correct at deployment time.

Test: event 325 only (never trained on), two row sets, same as disentangle_325:
  external_24pts   - 6x20:00 issues, 219h->3h remaining (the 4.31% comparison set)
  standing_164rows - 08/20/21 checkpoints across the full ~373h event
Reference pool at inference = 319 (complete before 325). Contexts finished with
this model's train-only compression/masks.

Run:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/train_all_pre325_test325.py
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
OUT = Path('runs/sweep_stack_20261019/train_all_pre325_test325.json')


def mape(points):
    errs = [p[1] / p[2] - 1 for p in points]
    return dict(rows=len(errs), mape=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)))


def main():
    bank = H.load_bank()
    data = bank.data
    start325 = data.events[325].start
    supervised = sorted(set(bank.train_ids) | {321, 322, 323, 324})  # all 319 before 325
    # causally correct cutoffs: everything trained on must be labeled before 325
    late = [e for e in supervised if data.label_available(e, bank.config) > start325]
    print(f'supervised events: {len(supervised)}; labels first available after 325 start: {late}', flush=True)
    bank.fit_cutoff = start325  # ages + eligibility measured at 325 deployment

    prep = H.prepare_fit(bank, supervised)
    print(f'weighted cells: {int(np.count_nonzero(prep.weights))}, weight mass {float(prep.weights.sum()):.3f}', flush=True)
    cols = block_cols({'rule', 'expl'})
    tr = np.ascontiguousarray(prep.matrix['pca'][:, cols])
    model = H.fit_jointnet(tr, prep.targets, prep.weights, prep.bins, bank.config, hidden=8, seed=1729)

    refs = list(supervised)  # 319: complete before 325
    cases, failures, _ = build_cases(data, bank.config, [325], refs)
    print(f'325 cases: {len(cases)}, failures: {len(failures)}', flush=True)

    ctx = finish_context(make_context(cases, data, bank.config, refs), prep.compression, prep.masks)[0]
    x = np.ascontiguousarray(ctx['matrices']['pca'][:, cols])
    stack_vals = correction_forecast(ctx['cases'], ctx['index'], model.predict(x), 1.0, 'nn')

    # reference models, same 325 rows
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
    print(f'row sets: external {len(A)}/{len(ext_pts)}, standing {len(B)}', flush=True)

    def pick(vals, ii):
        return [(cases[i], float(vals[i]), data.labels[(325, cases[i].tier)]) for i in ii]

    known = {  # from disentangle_325.json / eval_325.json
        'stack_post20': dict(external_24pts=0.0435, standing_164rows=0.0748),
        'ridge_full_refit': dict(external_24pts=0.0431, standing_164rows=0.0879),
        'baseline': dict(external_24pts=0.0865, standing_164rows=0.1473),
    }
    result = dict(model='stack recipe (rule+explicit 600, JointNet h8 seed1729, gain 1.0) on ALL pre-325 data',
        supervised_events=len(supervised), fit_cutoff=start325,
        weighted_cells=int(np.count_nonzero(prep.weights)), known_comparison=known, sets={})
    for set_name, ii in (('external_24pts', A), ('standing_164rows', B)):
        res = {
            'stack_all_pre325': mape(pick(stack_vals, ii)),
            'ridge_full_refit': mape(pick(ridge_vals, ii)),
            'baseline': mape(pick(baseline, ii)),
        }
        result['sets'][set_name] = res
        print(f'--- {set_name}', flush=True)
        for k, v in res.items():
            print(f'  {k:18s} MAPE {100*v["mape"]:6.2f}%  bias {100*v["bias"]:+6.2f}%  rows {v["rows"]}', flush=True)
        kn = known['stack_post20'][set_name]
        print(f'  {"stack_post20(old)":18s} MAPE {100*kn:6.2f}%  (previous run)', flush=True)
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
