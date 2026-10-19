"""Frozen-cadence ablation: packed raw LSTM, hourly versus every observation."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import torch

from collect_bandori_data import atomic_json, write_csv
from .config import Config
from .dataset import Dataset
from .evaluation import summary
from .native_lstm import NativeJointLSTM
from .native_sequence import NativeStream, forecast_event, native_times, pack, stream
from .pipeline import digest
from .regime_suite import load_context, records, stress_dataset
from .role_analog import build_cases


# Administrative resume fix only: this runner version wrote tuple-valued
# Config fields, then compared them directly with their JSON list roundtrip.
LEGACY_RUNNER_SHA='060e14907d07df88e6bcaaf8f9aa039cdfaa12c5d67a5592bb427c2ac8e4f287'


def compatible_plan(old,new):
    old,new=json.loads(json.dumps(old)),json.loads(json.dumps(new))
    if old.get('code',{}).get('native_experiment.py')==LEGACY_RUNNER_SHA:
        new['code']['native_experiment.py']=LEGACY_RUNNER_SHA
    return old==new


def cached_stream(data,eid,tiers,cadence,output):
    path=output/'streams'/f'{cadence}_{eid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as a:return NativeStream(**{f.name:(int(a[f.name]) if f.name=='event_id' else a[f.name].copy()) for f in fields(NativeStream)})
    result=stream(data,data.events[eid],tiers,cadence=cadence)
    np.savez_compressed(path,**{f.name:getattr(result,f.name) for f in fields(NativeStream)})
    return result


def predict_cases(model,data,cases,tiers,cadence):
    result=np.zeros(len(cases));positions={t:i for i,t in enumerate(tiers)};groups={}
    for i,c in enumerate(cases):groups.setdefault(c.event_id,[]).append(i)
    for eid,ii in groups.items():
        cutoffs=sorted({cases[i].cutoff for i in ii});values,_=forecast_event(model,data,data.events[eid],tiers,cutoffs,cadence)
        slots={t:i for i,t in enumerate(cutoffs)}
        for i in ii:result[i]=values[slots[cases[i].cutoff],positions[cases[i].tier]]
    return result


def train(data,source,output,epochs=120):
    source,output=Path(source),Path(output);config=Config.load(source/'config.json');data.add_terminal_labels(config)
    original=json.loads((source/'plan.json').read_text());full=json.loads((source/'full_refit/model.json').read_text())
    ids=full['training_event_ids'];registry=digest('evaluation_registry.json')
    if set(ids)&{325,326}:raise ValueError('Explicit holdout in native training')
    plan=dict(kind='native_tracker_lstm_v2',source=str(source.resolve()),source_sha256=digest(source/'model.json'),
        config=config.to_dict(),folds=original['folds'],training_event_ids=ids,epochs=epochs,hidden=32,decoder=16,cpu_threads=1,
        learning_rate=.005,decay=.001,bias_penalty=3,cadences=['hourly','native'],
        architecture='Same packed stock LSTM and clock/freshness-aware small query decoder in both arms; all native updates retained',
        semantics='Queries read the last recurrent state without an artificial recurrent step; elapsed time and clock are explicit; absent-tier age is zero',
        weighting='Duration-integrated within each event/horizon, capped at 24h across long gaps, then original recency/3x/proxy/event balancing',
        uncertainty='Previously inspected chronological folds; no fresh-test or Gaussian-coverage claim',
        code={p:digest(Path('bandori')/p) for p in ('native_sequence.py','native_lstm.py','native_experiment.py')})
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        old=json.loads((output/'plan.json').read_text())
        if not compatible_plan(old,plan):raise ValueError('Native resume plan differs')
        if old['code']!=plan['code']:
            atomic_json(output/'resume_execution.json',dict(reason='JSON tuple/list resume comparison fix; no numerical changes',
                frozen_plan_sha256=digest(output/'plan.json'),frozen_runner_sha256=old['code']['native_experiment.py'],
                current_runner_sha256=plan['code']['native_experiment.py']))
        plan=old
    else:atomic_json(output/'plan.json',plan)
    torch.set_num_threads(1);evaluations={};cadence_stats={}
    for cadence in plan['cadences']:
        print('Build',cadence,'causal raw streams',flush=True)
        streams={eid:cached_stream(data,eid,config.tiers,cadence,output) for eid in ids}
        lengths=[len(s.times) for s in streams.values()]
        cadence_stats[cadence]=dict(total_steps=sum(lengths),median_steps=float(np.median(lengths)),max_steps=max(lengths))
        for fold in [*plan['folds'],dict(name='full_refit',train=ids,fit_cutoff=full['fit_cutoff'],validation=[])]:
            directory=output/cadence/fold['name'];directory.mkdir(parents=True,exist_ok=True);path=directory/'model.pt'
            if not path.exists():
                bundle=pack([streams[eid] for eid in fold['train']],data,config,fold['fit_cutoff'])
                print('Fit',cadence,fold['name'],'steps',bundle['native_steps'],'targets',int(np.count_nonzero(bundle['weights'])),flush=True)
                model=NativeJointLSTM().fit(bundle,epochs,progress=lambda step,loss:print(cadence,fold['name'],step,round(loss,6),flush=True))
                model.save(path);atomic_json(directory/'model.json',dict(training_event_ids=fold['train'],fit_cutoff=fold['fit_cutoff'],
                    cadence=cadence,steps=bundle['native_steps'],targets=int(np.count_nonzero(bundle['weights'])),model_sha256=digest(path),
                    trained_tiers={str(t):bool(model.trained[j]) for j,t in enumerate(config.tiers)}))
                del bundle
            else:
                if digest(path)!=json.loads((directory/'model.json').read_text())['model_sha256']:raise ValueError('Native model checksum mismatch')
                model=NativeJointLSTM.load(path)
            if not fold['validation']:continue
            if (directory/'evaluation.json').exists():
                evaluations[f'{cadence}:{fold["name"]}']=json.loads((directory/'evaluation.json').read_text());continue
            print('Evaluate',cadence,fold['name'],flush=True)
            cases,_,_=build_cases(data,config,fold['validation'],fold['train']);values=predict_cases(model,data,cases,config.tiers,cadence)
            rows=records(dict(cases=cases),{cadence:values},data,fold['name'],'clean')
            evaluation=dict(clean=summary(cases,values,data),stress={})
            for scenario,gap,offset in (('off_grid_37m',0,37*60),('gap_6h',6,0),('gap_24h',24,0)):
                cc=[];vv=[]
                # Use the exact issue/tier intersections of the previously frozen
                # stress rows, keeping the comparison independent of new errors.
                import csv
                with (source/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:
                    prior=[r for r in csv.DictReader(f) if r['scenario']==scenario]
                from collect_bandori_data import timestamp
                from dataclasses import replace
                from .role_analog import RoleAwareAnalog
                groups={}
                for r in prior:groups.setdefault((int(r['event_id']),timestamp(r['checkpoint'])),[]).append(int(r['tier']))
                for (eid,cutoff),tiers in groups.items():
                    sparse=stress_dataset(data,eid,cutoff,gap);cfg=replace(config,max_stale_minutes=48*60 if gap else config.max_stale_minutes)
                    analog=RoleAwareAnalog(sparse,cfg,fold['train']);current=[]
                    for tier in tiers:
                        try:current.append(analog.build(sparse.events[eid],tier,cutoff))
                        except ValueError:pass
                    predictions=predict_cases(model,sparse,current,config.tiers,cadence);cc.extend(current);vv.extend(predictions)
                    rows.extend(records(dict(cases=current),{cadence:predictions},data,fold['name'],scenario))
                evaluation['stress'][scenario]=summary(cc,np.asarray(vv),data)
            atomic_json(directory/'evaluation.json',evaluation);write_csv(directory/'predictions.csv',list(rows[0]),rows)
            evaluations[f'{cadence}:{fold["name"]}']=evaluation
        del streams
    atomic_json(output/'evaluation.json',dict(folds=evaluations,cadence=cadence_stats,source_sha256=plan['source_sha256'],
        limitations=['Only twelve previously inspected event clusters','Every target and preprocessor stays within its fold training IDs',
                      'Native update frequency is irregular; no synthetic high-frequency information is created']))
    if digest('evaluation_registry.json')!=registry:raise ValueError('Holdout ledger changed')
    print('Native and hourly-control fits complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/full_approaches_20261017')
    p.add_argument('--source',default='runs/rule_aware_3x_20261018');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=120)
    args=p.parse_args();cfg=Config.load(Path(args.source)/'config.json');train(Dataset.load(args.data,cfg.tiers),args.source,args.output,args.epochs)


if __name__=='__main__':main()
