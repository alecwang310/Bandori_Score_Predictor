"""One-shot evaluation of the stack winner (rule+explicit x post_rule+20) on held-out event 325.

Event 325 never entered fitting, preprocessing or selection for this model. The
registry policy allows scoring holdouts; 325 labels are read only here. Two
reference-pool variants:
  matched_refs315 - identical protocol to the 321-324 evaluation (references = 315)
  causal_refs319  - references also include 321-324, which were labeled before 325 ran
Contexts are finished with the stack scope's train-only compression/masks.
Run: PYTHONPATH=. .venv/bin/python runs/sweep_stack_20261019/eval_325.py
"""
import json
from pathlib import Path

import numpy as np

from bandori import sweep_harness as H
from bandori.advanced_statistics import JointRidge
from bandori.analog import reconcile
from bandori.evaluation import summary
from bandori.experiments import correction_forecast
from bandori.regime_suite import finish_context, make_context
from bandori.role_analog import build_cases
from bandori.sweep_data_scope import post_plus_recent
from bandori.sweep_features import block_cols

FOCAL = (500, 1000, 1500, 2000)
OUT = Path('runs/sweep_stack_20261019/eval_325.json')


def score_event(cases, values, data, label):
    rows = [(c, v) for c, v in zip(cases, values)
            if data.labels.get((c.event_id, c.tier)) is not None and data.labels[(c.event_id, c.tier)] > 0]
    focal = [(c, v) for c, v in rows if c.tier in FOCAL]
    errs = [v / data.labels[(c.event_id, c.tier)] - 1 for c, v in focal]
    per_tier = {}
    for t in sorted({c.tier for c in cases}):
        sub = [(c, v) for c, v in rows if c.tier == t]
        e = [v / data.labels[(c.event_id, c.tier)] - 1 for c, v in sub]
        if e:
            per_tier[t] = dict(rows=len(e), mape=float(np.mean(np.abs(e))), bias=float(np.mean(e)))
    hours = {}
    import datetime
    cn = datetime.timezone(datetime.timedelta(hours=8))
    for hset, name in (((8,), '08h-only'), ((8, 20, 21), '08/20/21')):
        sub = [(c, v) for c, v in focal
               if datetime.datetime.fromtimestamp(c.cutoff, cn).hour in hset]
        if sub:
            e = [v / data.labels[(c.event_id, c.tier)] - 1 for c, v in sub]
            hours[name] = dict(rows=len(e), mape=float(np.mean(np.abs(e))), bias=float(np.mean(e)))
    out = dict(label=label, focal=dict(rows=len(focal), mape=float(np.mean(np.abs(errs))),
        bias=float(np.mean(errs)), events=1), per_tier=per_tier, checkpoints=hours)
    print(f"[{label}] focal MAPE {100*out['focal']['mape']:.2f}%  bias {100*out['focal']['bias']:+.2f}%  rows {out['focal']['rows']}", flush=True)
    for t, v in per_tier.items():
        mark = '*' if t in FOCAL else ' '
        print(f"   {mark}T{t:<6d} rows={v['rows']:3d} mape={100*v['mape']:6.2f}% bias={100*v['bias']:+6.2f}%", flush=True)
    for k, v in hours.items():
        print(f"   checkpoints {k}: rows={v['rows']} mape={100*v['mape']:.2f}% bias={100*v['bias']:+.2f}%", flush=True)
    return out


def main():
    bank = H.load_bank()
    ids = post_plus_recent(bank, 20)
    prep = H.prepare_fit(bank, ids)
    cols = block_cols({'rule', 'expl'})
    tr = np.ascontiguousarray(prep.matrix['pca'][:, cols])
    model = H.fit_jointnet(tr, prep.targets, prep.weights, prep.bins, bank.config, hidden=8, seed=1729)
    # self-check: must reproduce the sweep_stack number on 321-324 before trusting 325
    va = np.ascontiguousarray(prep.val['matrices']['pca'][:, cols])
    delta = model.predict(va)
    check = H.score(prep.val['cases'], correction_forecast(prep.val['cases'], prep.val['index'], delta, 1.0, 'nn'), bank.data)
    print(f'refit check 321-324: {H.fmt(check)} (expect MAPE 5.64%)', flush=True)
    assert abs(check['mape'] - 0.0564) < 5e-4, 'refit does not reproduce the sweep stack winner'

    result = dict(model='stack rule+explicit x post_rule+20 (JointNet h8 seed1729, gain 1.0)',
        fit=dict(scope=sorted(ids), supervised_cells=int(np.count_nonzero(prep.weights)),
                 note='compression+masks fit on the 30 scope events only; 325 unseen'),
        label_sources={str(t): bank.data.label_sources.get((325, t)) for t in FOCAL},
        labels={str(t): bank.data.labels.get((325, t)) for t in FOCAL}, variants={})
    for name, refs in (('matched_refs315', bank.train_ids), ('causal_refs319', bank.train_ids + [321, 322, 323, 324])):
        cases, failures, roles = build_cases(bank.data, bank.config, [325], refs)
        print(f'--- {name}: {len(cases)} cases, {len(failures)} case failures', flush=True)
        ctx = make_context(cases, bank.data, bank.config, refs)
        ctx, _ = finish_context(ctx, prep.compression, prep.masks)
        x = np.ascontiguousarray(ctx['matrices']['pca'][:, cols])
        delta = model.predict(x)
        values = correction_forecast(ctx['cases'], ctx['index'], delta, 1.0, 'nn')
        res = dict(stack=score_event(ctx['cases'], values, bank.data, f'stack {name}'))
        base = reconcile(ctx['cases'], [c.baseline for c in ctx['cases']])
        res['baseline'] = score_event(ctx['cases'], base, bank.data, f'analog-baseline {name}')
        if name == 'matched_refs315':
            m = 'runs/rule_aware_3x_20261018/eight_post_events/models'
            ridge = JointRidge.load(m + '/ridge.npz')
            from bandori.representations import Compression
            comp = Compression.load(m + '/compression.npz')
            with np.load(m + '/input_masks.npz', allow_pickle=False) as a:
                masks = {k: a[k].copy() for k in a.files}
            ctx2, _ = finish_context(make_context(cases, bank.data, bank.config, refs), comp, masks)
            d2 = ridge.predict(ctx2['matrices']['pca'])
            v2 = correction_forecast(ctx2['cases'], ctx2['index'], d2, 0.5, 'ridge')
            res['ridge_frozen'] = score_event(ctx2['cases'], v2, bank.data, 'ridge eight_post frozen')
        result['variants'][name] = res
        result.setdefault('case_failures', {})[name] = [dict(tier=f['tier'], checkpoint=f['checkpoint'], reason=f['reason']) for f in failures[:10]]
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print('wrote', OUT, flush=True)


if __name__ == '__main__':
    main()
