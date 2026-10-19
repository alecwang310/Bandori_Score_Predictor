"""Saved point forecasts from clock-feature and native-cadence experiments."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from collect_bandori_data import CN, atomic_bytes, atomic_json, iso, timestamp
from .clock_experiment import context, finish, load_fit, predictions
from .config import Config
from .dataset import Dataset
from .native_lstm import NativeJointLSTM
from .native_sequence import forecast_event
from .pipeline import attach_bank, digest
from .role_analog import RoleAwareAnalog


def finalize_features(root):
    root=Path(root);meta=json.loads((root/'full_refit/model.json').read_text())
    config=(root/'full_refit/config.json').read_bytes()
    if (root/'config.json').exists() and (root/'config.json').read_bytes()!=config:raise ValueError('Frozen config differs')
    if not (root/'config.json').exists():atomic_bytes(root/'config.json',config)
    files=('config.json','plan.json','selection.json','evaluation.json','full_refit/model.json')
    manifest=dict(kind='clock_feature_suite_v2',training_event_ids=meta['training_event_ids'],excluded_event_ids=[325,326],
        fit_cutoff=meta['fit_cutoff'],training_targets_by_tier=meta['training_targets_by_tier'],
        hashes={f:digest(root/f) for f in files})
    path=root/'model.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Frozen clock manifest differs')
    if not path.exists():atomic_json(path,manifest)
    return manifest


def predict_features(live,root,eid,cutoff,approach=None):
    root=Path(root);fit=root/'full_refit';meta=json.loads((fit/'model.json').read_text());cfg=Config.load(fit/'config.json')
    if eid in meta['training_event_ids'] or cutoff<meta['fit_cutoff']:raise ValueError('Training event or pre-fit forecast')
    data=attach_bank(fit,live);event=data.known_event(data.events[eid],cutoff)[0]
    analog=RoleAwareAnalog(data,cfg,meta['training_event_ids']);cases=[];unavailable=[]
    for tier in cfg.tiers:
        try:cases.append(analog.build(event,tier,cutoff))
        except ValueError as error:unavailable.append(dict(tier=tier,status='unavailable',reason=str(error)))
    if not cases:raise ValueError('No usable score lines')
    cores,specs,pc,clock_pc,masks=load_fit(fit);ctx=context(cases,data,cfg,meta['training_event_ids']);ctx,_,_=finish(ctx,pc,clock_pc,masks)
    all_values=predictions(ctx,specs,cores,cfg.tiers);name=approach or json.loads((root/'selection.json').read_text())['champion']
    if name not in all_values:raise ValueError('Unknown clock model')
    rows=[dict(tier=c.tier,status='ok',final=math.ceil(all_values[name][i]),current=math.ceil(c.current),observed_at=iso(c.observed_at),
        score_age_hours=(cutoff-c.observed_at)/3600) for i,c in enumerate(cases)]
    return dict(event_id=eid,checkpoint_cn=datetime.fromtimestamp(cutoff,CN).isoformat(),hours_remaining=(event.end-cutoff)/3600,
        approach=name,feature_version='clock_v2',model_sha256=digest(fit/'model.json'),predictions=sorted(rows+unavailable,key=lambda r:r['tier']))


def predict_native(live,root,eid,cutoff,cadence='native'):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());source=Path(plan['source']);cfg=Config.load(source/'config.json')
    directory=root/cadence/'full_refit';meta=json.loads((directory/'model.json').read_text())
    if eid in meta['training_event_ids'] or cutoff<meta['fit_cutoff']:raise ValueError('Training event or pre-fit forecast')
    if digest(directory/'model.pt')!=meta['model_sha256']:raise ValueError('Native model checksum mismatch')
    if digest(source/'model.json')!=plan['source_sha256']:raise ValueError('Native source checksum mismatch')
    source_meta=json.loads((source/'full_refit/model.json').read_text())
    for name,expected in source_meta['hashes'].items():
        if digest(source/'full_refit'/name)!=expected:raise ValueError('Source bank checksum mismatch')
    data=attach_bank(source/'full_refit',live);event=data.known_event(data.events[eid],cutoff)[0]
    torch.set_num_threads(2);model=NativeJointLSTM.load(directory/'model.pt');values,details=forecast_event(model,data,event,cfg.tiers,[cutoff],cadence)
    rows=[]
    for j,tier in enumerate(cfg.tiers):
        if not details['present'][0,j]:rows.append(dict(tier=tier,status='unavailable',reason='No positive score known'));continue
        rows.append(dict(tier=tier,status='ok',final=math.ceil(values[0,j]),current=math.ceil(details['current'][0,j]),
            observed_at=iso(details['observed'][0,j]),score_age_hours=(cutoff-details['observed'][0,j])/3600,
            stale_input=bool((cutoff-details['observed'][0,j])>3600),trained=meta['trained_tiers'][str(tier)]))
    return dict(event_id=eid,checkpoint_cn=datetime.fromtimestamp(cutoff,CN).isoformat(),hours_remaining=(event.end-cutoff)/3600,
        approach='lstm_'+cadence,native_steps=details['native_steps'],model_sha256=meta['model_sha256'],predictions=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=('features','native'))
    p.add_argument('--model',required=True);p.add_argument('--data',default='bandori_cn_data');p.add_argument('--event',type=int,required=True)
    p.add_argument('--as-of',required=True);p.add_argument('--output',required=True);p.add_argument('--approach');p.add_argument('--cadence',choices=('native','hourly'),default='native')
    args=p.parse_args();cutoff=timestamp(args.as_of)
    if cutoff is None:raise ValueError('Timezone-qualified --as-of required')
    if Path(args.output).exists():raise ValueError('Fixed forecast already exists')
    if args.kind=='features':cfg=Config.load(Path(args.model)/'full_refit/config.json')
    else:cfg=Config(**{**json.loads((Path(args.model)/'plan.json').read_text())['config']})
    live=Dataset.load(args.data,cfg.tiers)
    result=predict_features(live,args.model,args.event,cutoff,args.approach) if args.kind=='features' else predict_native(live,args.model,args.event,cutoff,args.cadence)
    atomic_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
