"""Frozen native raw+small-stat transformer fits and matched historical replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch

from collect_bandori_data import atomic_json,timestamp,write_csv
from .config import Config
from .dataset import Dataset
from .native_experiment import cached_stream
from .native_sequence import forecast_event,pack
from .pipeline import digest
from .raw_transformer import RawJointTransformer
from .regime_suite import stress_dataset


def verify_plan(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    for name,h in plan['code'].items():
        if digest(Path('bandori')/name)!=h:raise ValueError('Frozen transformer code changed: '+name)
    if digest(Path(plan['native_source'])/'plan.json')!=plan['native_plan_sha256']:raise ValueError('Native source plan changed')
    if digest(Path(plan['source'])/'model.json')!=plan['source_model_sha256']:raise ValueError('Frozen reference source changed')
    if digest('evaluation_registry.json')!=plan['ledger_sha256']:raise ValueError('Protected evaluation ledger changed')
    return plan


def fit_fold(root,name):
    root=Path(root);p=verify_plan(root);native=Path(p['native_source']);source=Path(p['source'])
    config=Config.load(source/'config.json');full=json.loads((source/'full_refit/model.json').read_text())
    folds={f['name']:f for f in p['folds']};folds['full_refit']=dict(train=p['training_event_ids'],fit_cutoff=full['fit_cutoff'])
    f=folds[name];directory=root/name;directory.mkdir(parents=True,exist_ok=True);path=directory/'model.pt'
    if path.exists() and (directory/'model.json').exists():
        m=json.loads((directory/'model.json').read_text())
        if m['model_sha256']!=digest(path) or m['training_event_ids']!=f['train'] or m['plan_sha256']!=digest(root/'plan.json'):raise ValueError('Saved transformer fit differs')
        return name
    torch.set_num_threads(1);data=Dataset.load(p['data'],config.tiers);data.add_terminal_labels(config)
    for eid in f['train']:
        if digest(native/'streams'/f'native_{eid}.npz')!=p['stream_hashes'][str(eid)]:raise ValueError('Raw stream cache changed')
    streams=[cached_stream(data,e,config.tiers,'native',native) for e in f['train']]
    bundle=pack(streams,data,config,f['fit_cutoff']);del streams
    print('Fit transformer',name,'steps',bundle['native_steps'],flush=True)
    model=RawJointTransformer(**p['architecture']).fit(bundle,**p['optimization'],
        progress=lambda epoch,loss:print(name,epoch,round(loss,7),flush=True))
    temporary=path.with_suffix('.tmp');model.save(temporary);os.replace(temporary,path)
    atomic_json(directory/'model.json',dict(training_event_ids=f['train'],fit_cutoff=f['fit_cutoff'],
        model_sha256=digest(path),plan_sha256=digest(root/'plan.json'),native_steps=bundle['native_steps'],
        parameter_count=sum(v.numel() for v in model.parameters()),
        trained_tiers={str(t):bool(model.trained[j]) for j,t in enumerate(config.tiers)}))
    return name


def evaluate_fold(data,root,fold,tiers):
    root=Path(root);p=verify_plan(root);directory=root/fold['name'];meta=json.loads((directory/'model.json').read_text())
    if digest(directory/'model.pt')!=meta['model_sha256']:raise ValueError('Transformer weights changed')
    model=RawJointTransformer.load(directory/'model.pt');positions={t:i for i,t in enumerate(tiers)}
    source=Path(p['native_source'])/'native'/fold['name']/'predictions.csv'
    with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r['scenario'],int(r['event_id'])),[]).append(i)
    for (scenario,eid),ii in groups.items():
        times=sorted({timestamp(rows[i]['checkpoint']) for i in ii})
        print('Replay transformer',fold['name'],scenario,eid,flush=True)
        if scenario in ('clean','off_grid_37m'):
            pred,_=forecast_event(model,data,data.events[eid],tiers,times)
            slots={t:i for i,t in enumerate(times)}
            for i in ii:rows[i]['transformer']=float(pred[slots[timestamp(rows[i]['checkpoint'])],positions[int(rows[i]['tier'])]])
        else:
            gap={'gap_6h':6,'gap_24h':24}[scenario]
            for t in times:
                sparse=stress_dataset(data,eid,t,gap);pred,_=forecast_event(model,sparse,sparse.events[eid],tiers,[t])
                for i in ii:
                    if timestamp(rows[i]['checkpoint'])==t:rows[i]['transformer']=float(pred[0,positions[int(rows[i]['tier'])]])
    write_csv(directory/'predictions.csv',list(rows[0]),rows)
    return len(rows)


def run(data_path,native_source,output,workers):
    native_source,output=Path(native_source),Path(output);native=json.loads((native_source/'plan.json').read_text());source=Path(native['source'])
    config=Config.load(source/'config.json');ids=native['training_event_ids']
    if set(ids)&{325,326}:raise ValueError('325/326 cannot enter transformer training')
    architecture=dict(width=32,heads=4,layers=2,feedforward=64,decoder=16,seed=1729)
    optimization=dict(epochs=120,learning_rate=.003,bias_penalty=3,decay=.001)
    plan=dict(kind='native_raw_small_stat_causal_transformer_v1',data=str(Path(data_path).resolve()),
        native_source=str(native_source.resolve()),native_plan_sha256=digest(native_source/'plan.json'),source=str(source.resolve()),
        source_model_sha256=digest(source/'model.json'),config=config.to_dict(),training_event_ids=ids,folds=native['folds'],
        excluded_event_ids=[325,326],architecture=architecture,optimization=optimization,
        attention='Two pre-norm full-history causal self-attention blocks; no history downsampling, cropping or bidirectional attention',
        representation='Exactly the native LSTM raw stream and small 6h/24h growth, clock, freshness, type and reward side inputs',
        position='Sinusoidal record position; physical clock, elapsed progress and gap duration are already explicit raw inputs',
        objective='Identical duration/event/horizon/proxy/730-day/3x weights to native LSTM; exact full-bank relative MSE plus per-tier/horizon signed-bias penalty',
        batching='Length buckets, right padding discarded from loss; deterministic two-pass full-bank gradient accumulation and one Adam update per epoch',
        selection='One architecture and fixed training duration; no validation stopping, architecture search or promotion',
        cpu_threads=1,ledger_sha256=digest('evaluation_registry.json'),
        stream_hashes={str(e):digest(native_source/'streams'/f'native_{e}.npz') for e in ids},
        code={n:digest(Path('bandori')/n) for n in ('raw_transformer.py','transformer_experiment.py','native_sequence.py','raw_sequence.py','rules.py')})
    plan=json.loads(json.dumps(plan));output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Transformer resume plan differs')
    else:atomic_json(output/'plan.json',plan)
    names=[f['name'] for f in plan['folds']]+['full_refit']
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fit_fold,str(output),n) for n in names]
        for future in as_completed(futures):print('Transformer saved',future.result(),flush=True)
    torch.set_num_threads(1);data=Dataset.load(data_path,config.tiers);data.add_terminal_labels(config)
    for fold in plan['folds']:
        if not (output/fold['name']/'predictions.csv').exists():evaluate_fold(data,output,fold,config.tiers)
    verify_plan(output)
    print('Transformer fits and matched replay complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/full_approaches_20261017')
    p.add_argument('--native-source',default='runs/native_lstm_v2_20261018');p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.data,a.native_source,a.output,a.workers)


if __name__=='__main__':main()
