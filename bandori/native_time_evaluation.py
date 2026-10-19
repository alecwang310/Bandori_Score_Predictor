"""Matched tracker/hour query evaluation, including wall-time-weighted errors."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from collect_bandori_data import atomic_json
from .config import Config
from .dataset import Dataset
from .evaluation import paired_bootstrap, summary
from .native_lstm import NativeJointLSTM
from .native_sequence import forecast_event


def integrated_metrics(rows,values,data):
    groups={}
    for i,row in enumerate(rows):groups.setdefault((int(row[0]),int(row[1])),[]).append(i)
    weights=np.zeros(len(rows))
    for (eid,tier),ii in groups.items():
        ii=sorted(ii,key=lambda i:rows[i,2]);t=rows[ii,2]
        w=np.minimum(86400,np.maximum(0,np.diff(np.r_[t,data.events[eid].end])))
        weights[ii]=w/max(w.sum(),1)
    events={}
    for eid in np.unique(rows[:,0]).astype(int):
        keep=rows[:,0]==eid;error=values[keep]/rows[keep,6]-1;w=weights[keep]
        events[eid]=(float(np.average(abs(error),weights=w)),float(np.average(error,weights=w)))
    return dict(mape=float(np.mean([a for a,b in events.values()])),bias=float(np.mean([b for a,b in events.values()])),
        events=len(events),rows=len(rows),scope='Equal events and tiers; piecewise-constant query errors integrated over elapsed wall time')


def evaluate(data,root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());source=Path(plan['source'])
    config=Config.load(source/'config.json');data.add_terminal_labels(config);torch.set_num_threads(1)
    with np.load(source/'all_timestamp_predictions.npz',allow_pickle=False) as a:rows=a['rows'].copy()
    values={'reference':rows[:,5].copy(),'hourly':np.zeros(len(rows)),'native':np.zeros(len(rows))}
    for fold in plan['folds']:
        models={c:NativeJointLSTM.load(root/c/fold['name']/'model.pt') for c in ('hourly','native')}
        for eid in fold['validation']:
            ii=np.flatnonzero(rows[:,0]==eid);times=np.unique(rows[ii,2]);positions={t:i for i,t in enumerate(config.tiers)}
            slots={t:i for i,t in enumerate(times)}
            print('Evaluate dense queries',eid,len(times),flush=True)
            for cadence,model in models.items():
                predicted,_=forecast_event(model,data,data.events[eid],config.tiers,times,cadence)
                values[cadence][ii]=[predicted[slots[rows[i,2]],positions[int(rows[i,1])]] for i in ii]
    cases=[SimpleNamespace(event_id=int(r[0]),tier=int(r[1]),remaining_hours=float(r[3])) for r in rows]
    result=dict(scope='Identical original tracker/hourly focal-tier queries as the frozen reference; no new timestamps selected using forecast errors',
        scores={n:summary(cases,p,data) for n,p in values.items()},
        time_integrated={n:integrated_metrics(rows,p,data) for n,p in values.items()},
        fresh={n:summary([c for c,r in zip(cases,rows) if r[7]<=1],p[rows[:,7]<=1],data) for n,p in values.items()},
        native_vs_hourly=paired_bootstrap(cases,values['hourly'],values['native'],data,1729),
        initially_unsupported_head_rows=int(np.count_nonzero(rows[:,8]==0)),
        limitations=['Only twelve previously inspected event clusters','Tracker timestamps are correlated and unevenly distributed in wall time'])
    np.savez_compressed(root/'matched_timestamp_predictions.npz',rows=rows,**values)
    atomic_json(root/'matched_timestamp_evaluation.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--data',default='data/full_approaches_20261017')
    args=p.parse_args();plan=json.loads((Path(args.model)/'plan.json').read_text());cfg=Config.load(Path(plan['source'])/'config.json')
    result=evaluate(Dataset.load(args.data,cfg.tiers),args.model);print(json.dumps(result['time_integrated'],indent=2))


if __name__=='__main__':main()
