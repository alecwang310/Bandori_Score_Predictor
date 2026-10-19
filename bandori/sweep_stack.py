"""Stack the two sweep winners: rule_only / rule+explicit features x trimmed training scope.

From the 2026-10 sweep:
- features sweep: rule_only (504 cols) 6.23%, rule+explicit (600) 6.28% on the full bank
- data-scope sweep: post_rule+20recent scope 6.41% with the full pca matrix

This module re-runs both feature sets under the trimmed scope (and the reverse
pairing) so the two effects can be read together. Same frozen protocol:
train on the eight_post_events bank, test on events 321-324, focal tiers,
event-balanced MAPE. Compression and masks are refit on the supervised rows of
each scope (train-only preprocessing).
"""
import copy
import json
import time
from pathlib import Path

import numpy as np

from . import sweep_harness as H
from .experiments import correction_forecast
from .sweep_data_scope import POST_IDS, post_plus_recent
from .sweep_features import block_cols

RUN_DIR=Path('runs/sweep_stack_20261019')
OUT_REPORT=Path('reports/sweep_stack_20261019.md')


def fit_eval(bank,prep,name,cols,gain=1.0,kind='nn',ridge=False):
    t0=time.time()
    # ascontiguousarray: column fancy-indexing can hand back a strided view
    # (e.g. X[:, np.arange(1168)]), and fit results are layout-sensitive at the
    # 1e-15 level, which 240 Adam steps amplify into different models.
    tr=np.ascontiguousarray(prep.matrix['pca'][:,cols])
    va=np.ascontiguousarray(prep.val['matrices']['pca'][:,cols])
    assert tr.shape[1]==va.shape[1]==len(cols)
    if ridge:
        model=H.fit_ridge(tr,prep.targets,prep.weights,prep.bins)
        delta=model.predict(va);values=correction_forecast(prep.val['cases'],prep.val['index'],delta,.5,'ridge')
    else:
        model=H.fit_jointnet(tr,prep.targets,prep.weights,prep.bins,bank.config,hidden=8,seed=1729)
        delta=model.predict(va);values=correction_forecast(prep.val['cases'],prep.val['index'],delta,gain,'nn')
    res=H.score(prep.val['cases'],values,bank.data)
    res.update(columns=int(tr.shape[1]),ridge=ridge,seconds=time.time()-t0)
    print(f'{name:36s} cols={len(cols):5d} {"ridge" if ridge else "nn   "} {H.fmt(res)}  [{time.time()-t0:.1f}s] {H.per_event_line(res)}',flush=True)
    return res


def main():
    bank=H.load_bank()
    H.validate(bank)
    rules=block_cols({'rule'});rule_expl=block_cols({'rule','expl'})
    scopes=dict(full=bank.train_ids,post20=post_plus_recent(bank,20),post_only=sorted(POST_IDS))
    results={}
    for scope_name,ids in scopes.items():
        print(f'--- scope {scope_name}: {len(ids)} supervised events',flush=True)
        prep=H.prepare_fit(bank,ids)
        for feat_name,cols in (('rule_only',rules),('rule_plus_explicit',rule_expl)):
            key=f'{feat_name}@{scope_name}'
            results[key]=fit_eval(bank,prep,key,cols)
        if scope_name=='post20':
            results['rule_only@post20_ridge']=fit_eval(bank,prep,'rule_only@post20_ridge',rules,ridge=True)
            results['full_pca@post20']=fit_eval(bank,prep,'full_pca@post20',np.arange(1168))
    H.save_results(RUN_DIR/'evaluation.json',
        dict(title='Stacked sweep winners: feature set x training scope',
            scope_note=H.__doc__.splitlines()[2].strip() if H.__doc__ else '',
            results={k:{kk:vv for kk,vv in v.items() if kk!='full'} for k,v in results.items()},
            per_event={k:{str(e):vv['mape'] for e,vv in v['per_event'].items()} for k,v in results.items()}))
    ranked=sorted(results.items(),key=lambda kv:kv[1]['mape'])
    lines=['Scope: post_rule+20recent means ids 311-320 plus the 20 most recent older training events '
        '(event end time). rule_only = 504 rule-metadata columns; rule_plus_explicit = 600 columns. '
        'Compression and masks refit on each scope (train-only). Seed 1729.','',
        '| config | cols | MAPE | bias | final24h | per-event 321/322/323/324 |','| --- | ---: | ---: | ---: | ---: | --- |']
    for k,v in ranked:
        pe=H.per_event_line(v)
        lines.append(f"| {k} | {v['columns']} | {100*v['mape']:.2f}% | {100*v['bias']:+.2f}% | "
            f"{100*v['final24h']:.2f}% | {pe} |")
    lines+=['','Reference points (frozen protocol, full bank, full pca): nn_pca 6.71%, ridge 7.61%. '
        'Single wins: rule_only@full 6.23% (features sweep), full_pca@post20 6.41% (data-scope sweep).',
        'All scores descriptive on 4 test events; rows within an event are dependent.','']
    H.write_report(OUT_REPORT,'Stacked sweep winners — feature set x training scope',
        [('Results',lines)])


if __name__=='__main__':
    main()
