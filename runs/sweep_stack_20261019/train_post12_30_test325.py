"""Corrected 30-event scope: 12 strictly-post-rule (311, 314-324) + 18 recent older.

Fixes train_postrule30_test325.py, which silently fit 32 events (range(311,325)
includes pre-rule-start 312/313, and 14+18=32). This is the user's spec:
"all post rule, with 30 all in all". Same stack recipe otherwise.
"""
import csv, json
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

FOCAL=(500,1000,1500,2000)
POST=[311]+list(range(314,325))  # 12 strictly-post-rule (312/313 start pre-rule)
OUT=Path('runs/sweep_stack_20261019/train_post12_30_test325.json')

def mape(pts):
    e=[v/l-1 for _,v,l in pts]
    return dict(rows=len(e), mape=float(np.mean(np.abs(e))), bias=float(np.mean(e)))

def main():
    bank=H.load_bank(); data=bank.data
    assert len(POST)==12, POST
    older=[e for e in sorted(bank.train_ids,key=lambda e:data.events[e].end,reverse=True) if e not in POST]
    supervised=sorted(set(POST)|set(older[:18]))
    print(f'supervised: {len(supervised)} events (want 30): post-rule {POST}', flush=True)
    print(f'older: {sorted(e for e in supervised if e not in POST)}', flush=True)
    assert len(supervised)==30
    bank.fit_cutoff=data.events[325].start
    prep=H.prepare_fit(bank,supervised)
    print(f'weighted cells: {int(np.count_nonzero(prep.weights))}, mass {float(prep.weights.sum()):.3f}', flush=True)
    cols=block_cols({'rule','expl'})
    tr=np.ascontiguousarray(prep.matrix['pca'][:,cols])
    model=H.fit_jointnet(tr,prep.targets,prep.weights,prep.bins,bank.config,hidden=8,seed=1729)
    refs=list(bank.train_ids)+[321,322,323,324]
    cases,_,_=build_cases(data,bank.config,[325],refs)
    ctx=finish_context(make_context(cases,data,bank.config,refs),prep.compression,prep.masks)[0]
    x=np.ascontiguousarray(ctx['matrices']['pca'][:,cols])
    stack=correction_forecast(ctx['cases'],ctx['index'],model.predict(x),1.0,'nn')
    m=Path('runs/rule_aware_3x_20261018/full_refit/models')
    ridge=JointRidge.load(str(m/'ridge.npz')); comp=Compression.load(str(m/'compression.npz'))
    with np.load(m/'input_masks.npz',allow_pickle=False) as a: masks={k:a[k].copy() for k in a.files}
    ctx_r=finish_context(make_context(cases,data,bank.config,refs),comp,masks)[0]
    ridge_v=correction_forecast(ctx_r['cases'],ctx_r['index'],ridge.predict(ctx_r['matrices']['pca']),0.5,'ridge')
    base=reconcile(cases,[c.baseline for c in cases])
    ext=list(csv.DictReader(open('reports/fresh_event325_external_20261019/comparison.csv',encoding='utf-8-sig')))
    ext_pts={(timestamp(r['issued_at']),int(r['tier'])) for r in ext if r['fresh_input']=='True'}
    idx={}
    for i,c in enumerate(cases): idx.setdefault((int(c.cutoff),c.tier),i)
    A=[i for k,i in idx.items() if k in ext_pts]
    B=[i for i,c in enumerate(cases) if c.tier in FOCAL]
    result=dict(model='stack recipe (rule+explicit 600, JointNet h8 seed1729, gain 1.0)',
        scope='30 events: strictly-post-rule 311,314-324 + 18 most recent older',
        supervised_events=supervised, fit_cutoff=bank.fit_cutoff,
        weighted_cells=int(np.count_nonzero(prep.weights)), sets={})
    for name,ii in (('external_24pts',A),('standing_164rows',B)):
        pick=lambda vals:[(cases[i],float(vals[i]),data.labels[(325,cases[i].tier)]) for i in ii]
        res=dict(stack_post12_30=mape(pick(stack)),ridge_full_refit=mape(pick(ridge_v)),baseline=mape(pick(base)))
        result['sets'][name]=res
        print(f'--- {name}',flush=True)
        for k,v in res.items():
            print(f'  {k:18s} MAPE {100*v["mape"]:6.2f}%  bias {100*v["bias"]:+6.2f}%  rows {v["rows"]}',flush=True)
    OUT.write_text(json.dumps(result,indent=1,default=float))
    print('wrote',OUT,flush=True)

if __name__=='__main__': main()
