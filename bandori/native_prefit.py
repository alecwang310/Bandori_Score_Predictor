"""Fit independent frozen native folds concurrently, then resume evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path


def fit_fold(root,name,engine):
    import numpy as np
    import torch
    from collect_bandori_data import atomic_json
    from .config import Config
    from .dataset import Dataset
    from .native_experiment import LEGACY_RUNNER_SHA,cached_stream
    from .native_sequence import pack
    from .native_lstm import NativeJointLSTM
    from .native_cpu import CPUNativeLSTM
    from .pipeline import digest
    root=Path(root);plan=json.loads((root/'plan.json').read_text());source=Path(plan['source'])
    config=Config.load(source/'config.json');full=json.loads((source/'full_refit/model.json').read_text())
    folds={f['name']:f for f in plan['folds']};folds['full_refit']=dict(train=plan['training_event_ids'],fit_cutoff=full['fit_cutoff'])
    fold=folds[name];directory=root/'native'/name;directory.mkdir(parents=True,exist_ok=True);path=directory/'model.pt'
    if path.exists() and (directory/'model.json').exists():
        meta=json.loads((directory/'model.json').read_text())
        if digest(path)!=meta['model_sha256'] or meta['training_event_ids']!=fold['train']:raise ValueError('Existing native model differs')
        return name
    for filename,expected in plan['code'].items():
        if filename=='native_experiment.py' and expected==LEGACY_RUNNER_SHA:continue
        if digest(Path(__file__).parent/filename)!=expected:raise ValueError('Native numerical code differs from frozen plan')
    torch.set_num_threads(plan['cpu_threads']);data=Dataset.load('data/full_approaches_20261017',config.tiers);data.add_terminal_labels(config)
    streams=[cached_stream(data,eid,config.tiers,'native',root) for eid in fold['train']]
    bundle=pack(streams,data,config,fold['fit_cutoff']);del streams
    print('Fit parallel native',name,'steps',bundle['native_steps'],flush=True)
    model_class=CPUNativeLSTM if engine=='cpu' else NativeJointLSTM
    model=model_class(plan['hidden'],plan['decoder'],config.seed).fit(bundle,plan['epochs'],
        plan['learning_rate'],plan['bias_penalty'],plan['decay'],
        progress=lambda epoch,loss:print('native',name,epoch,round(loss,6),flush=True))
    temporary=path.with_suffix('.pt.tmp');model.save(temporary);os.replace(temporary,path)
    atomic_json(directory/'model.json',dict(training_event_ids=fold['train'],fit_cutoff=fold['fit_cutoff'],cadence='native',
        steps=bundle['native_steps'],targets=int(np.count_nonzero(bundle['weights'])),model_sha256=digest(path),
        trained_tiers={str(t):bool(model.trained[j]) for j,t in enumerate(config.tiers)},execution='Independent parallel fit; identical frozen equations and training plan',
        execution_engine=engine,execution_sha256=digest(Path(__file__).parent/'native_cpu.py') if engine=='cpu' else None))
    return name


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--engine',choices=('cpu','torch'),default='cpu')
    args=p.parse_args();root=Path(args.model);plan=json.loads((root/'plan.json').read_text())
    names=[f['name'] for f in plan['folds']]+['full_refit']
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fit_fold,str(root),name,args.engine) for name in names]
        for f in as_completed(futures):print('Native fit saved:',f.result(),flush=True)


if __name__=='__main__':main()
