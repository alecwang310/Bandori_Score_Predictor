"""Disentangle why event-325 MAPE is 4.31% (external report) vs 7.44% (our stack).

2x2: models {stack JointNet (rule+explicit x post_rule+20), full-refit rule-aware
ridge (the 4.31% model)} x row sets {24 external fresh points (6x20:00 issues,
219h->3h remaining), our 164 standing-convention rows (08/20/21, full 373h event)}.

Both models are trained/frozen before 325; 325 labels are read only for scoring.
Reference pool at inference = 319 (321-324 were complete before 325's first
checkpoint; deployment-realistic). Each model uses its own train-only
compression/masks. Run:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/disentangle_325.py
"""
import copy
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
from bandori.sweep_data_scope import post_plus_recent
from bandori.sweep_features import block_cols
from collect_bandori_data import timestamp

FOCAL = (500, 1000, 1500, 2000)
OUT = Path('runs/sweep_stack_20261019/disentangle_325.json')


def mape(points):
    errs = [p[1] / p[2] - 1 for p in points]
    return dict(rows=len(errs), mape=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)))


def main():
    bank = H.load_bank()
    refs = list(bank.train_ids) + [321, 322, 323, 324]  # 319: causal inference pool
    cases, failures, _ = build_cases(bank.data, bank.config, [325], refs)
    print(f'325 cases: {len(cases)}, failures: {len(failures)}', flush=True)

    # --- stack model (refit exactly as sweep_stack; self-check first)
    ids = post_plus_recent(bank, 20)
    prep = H.prepare_fit(bank, ids)
    cols = block_cols({'rule', 'expl'})
    tr = np.ascontiguousarray(prep.matrix['pca'][:, cols])
    stack = H.fit_jointnet(tr, prep.targets, prep.weights, prep.bins, bank.config, hidden=8, seed=1729)
    va = np.ascontiguousarray(prep.val['matrices']['pca'][:, cols])
    check = H.score(prep.val['cases'], correction_forecast(prep.val['cases'], prep.val['index'],
                    stack.predict(va), 1.0, 'nn'), bank.data)
    print(f'self-check 321-324 stack: {H.fmt(check)}', flush=True)
    assert abs(check['mape'] - 0.0564) < 5e-4

    ctx_s = finish_context(make_context(cases, bank.data, bank.config, refs), prep.compression, prep.masks)[0]
    xs = np.ascontiguousarray(ctx_s['matrices']['pca'][:, cols])
    stack_vals = correction_forecast(ctx_s['cases'], ctx_s['index'], stack.predict(xs), 1.0, 'nn')

    # --- full-refit rule-aware ridge (the 4.31% model)
    m = Path('runs/rule_aware_3x_20261018/full_refit/models')
    ridge = JointRidge.load(str(m / 'ridge.npz'))
    comp = Compression.load(str(m / 'compression.npz'))
    with np.load(m / 'input_masks.npz', allow_pickle=False) as a:
        masks = {k: a[k].copy() for k in a.files}
    ctx_r = finish_context(make_context(cases, bank.data, bank.config, refs), comp, masks)[0]
    ridge_vals = correction_forecast(ctx_r['cases'], ctx_r['index'], ridge.predict(ctx_r['matrices']['pca']), 0.5, 'ridge')

    baseline = reconcile(cases, [c.baseline for c in cases])

    # --- row sets
    ext = list(csv.DictReader(open('reports/fresh_event325_external_20261019/comparison.csv', encoding='utf-8-sig')))
    ext_pts = {(timestamp(r['issued_at']), int(r['tier'])) for r in ext if r['fresh_input'] == 'True'}
    idx = {}
    for i, c in enumerate(cases):
        idx.setdefault((int(c.cutoff), c.tier), i)
    A = [i for (k, t), i in idx.items() if (k, t) in ext_pts]
    B = [i for i, c in enumerate(cases) if c.tier in FOCAL]
    print(f'row sets: external-24 matched {len(A)}/{len(ext_pts)}, standing {len(B)}', flush=True)

    def pick(vals, ii):
        return [(cases[i], float(vals[i]), bank.data.labels[(325, cases[i].tier)]) for i in ii]

    result = {}
    for set_name, ii in (('external_24pts', A), ('standing_164rows', B)):
        result[set_name] = {
            'stack': mape(pick(stack_vals, ii)),
            'ridge_full_refit': mape(pick(ridge_vals, ii)),
            'baseline': mape(pick(baseline, ii)),
        }
        print(f'--- {set_name}', flush=True)
        for k, v in result[set_name].items():
            print(f'  {k:16s} MAPE {100*v["mape"]:6.2f}%  bias {100*v["bias"]:+6.2f}%  rows {v["rows"]}', flush=True)
    OUT.write_text(json.dumps(dict(result, matched_external_points=len(A),
        unmatched_external_points=sorted(str(p) for p in ext_pts if p not in idx)), indent=1, default=float))
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
