"""Refit frozen family choices on all eligible events before explicit holdouts."""
import argparse
from dataclasses import replace
from datetime import datetime,timedelta
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN,atomic_json,iso,timestamp
from .advanced_statistics import CachedAnalog,HierarchicalSurface,JointRidge
from .config import Config
from .dataset import Dataset
from .experiments import component_prediction,context,ensemble_predictions
from .joint import JointNet
from .pipeline import attach_bank,digest,save_bank,training_arrays
from .representations import Compression
from .statistical import fit_calibration


CLASSES={'JointNet':JointNet,'JointRidge':JointRidge,'HierarchicalSurface':HierarchicalSurface}


def issue_times(event,hours):
    day=datetime.fromtimestamp(event.start,CN).replace(hour=0,minute=0,second=0,microsecond=0)
    while day.timestamp()<event.end:
        for hour in sorted(set(hours)):
            cutoff=day.replace(hour=hour).timestamp()
            if event.start<cutoff<event.end:yield cutoff
        day+=timedelta(days=1)


def refit(data,source,output,excluded=(325,326),hours=(8,20,21),registry='evaluation_registry.json',reuse_earlier_holdouts=False):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Choose a new immutable refit directory')
    if not hours or any(not isinstance(h,int) or not 0<=h<24 for h in hours):raise ValueError('Invalid checkpoint hours')
    cfg=Config.load(source/'config.json');data.add_terminal_labels(cfg)
    original=json.loads((source/'model.json').read_text())
    for file,expected in original['hashes'].items():
        if digest(source/file)!=expected:raise ValueError('Source model checksum mismatch: '+file)
    if not excluded or any(e not in data.events for e in excluded):raise ValueError('Explicit holdout events must be present')
    fit_cutoff=min(data.events[e].start for e in excluded)
    ids=[e.event_id for e in data.eligible_events(cfg) if e.event_id not in excluded and data.label_available(e.event_id,cfg)<=fit_cutoff]
    if len(ids)<cfg.min_train_events:raise ValueError('Not enough eligible completed events')
    ledger=json.loads(Path(registry).read_text()) if Path(registry).exists() else {'protected_event_ids':[]}
    reused=sorted(set(ids)&set(ledger['protected_event_ids']))
    if reused and not reuse_earlier_holdouts:raise ValueError('Explicit authorization is required to reuse earlier holdouts')
    selection=json.loads((source/'selection.json').read_text());source_plan=json.loads((source/'plan.json').read_text())
    output.mkdir(parents=True)
    plan=dict(role='full_refit_with_explicit_event_holdouts',source_model=str(source.resolve()),
        frozen_selection_sha256=digest(source/'selection.json'),config=cfg.to_dict(),training_event_ids=ids,
        excluded_event_ids=list(excluded),fit_cutoff=fit_cutoff,checkpoint_hours=list(hours),
        reused_earlier_protected_events=reused,authorization='User explicitly requested all events except 325 and 326' if reused else 'No earlier protected events reused',
        selection_policy='Family choices, gains, ensemble memberships and champion frozen before this refit; no held-out target selects a model',
        evaluation_policy='Retrospective event comparison, not a fresh test; original holdout ledger remains unchanged',
        implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    atomic_json(output/'plan.json',plan);atomic_json(output/'config.json',cfg.to_dict())
    predictor=CachedAnalog(data,cfg,ids);cases=[];excluded_cases=[]
    for number,eid in enumerate(ids):
        if number%25==0:print(f'Build full-refit training cases: {number}/{len(ids)} events',flush=True)
        event=data.events[eid]
        for cutoff in issue_times(event,hours):
            for tier in cfg.tiers:
                if (eid,tier) not in data.curves:continue
                try:cases.append(predictor.build(event,tier,cutoff))
                except ValueError as error:excluded_cases.append(dict(event_id=eid,tier=tier,checkpoint=iso(cutoff),reason=str(error)))
    if not cases:raise ValueError('No usable prequential training cases')
    print('Build full-refit representations:',len(cases),'tier/checkpoint cases',flush=True)
    ctx=context(cases,data,cfg,ids)
    target,weights=training_arrays(cases,ctx['index'],data,cfg,fit_cutoff)
    compression=Compression().fit(ctx['matrices']['rich'],weights,cfg.tiers,seed=cfg.seed)
    ctx['matrices']['pca']=compression.transform(ctx['matrices']['rich'])
    cores={};specs=selection['components']
    for key,kind in original['core_kinds'].items():
        print('Refit frozen core:',key,flush=True)
        if kind=='JointNet':
            representation=next(s['representation'] for s in specs.values() if key in s.get('cores',[]))
            hidden=JointNet.load(source/'models'/(key+'.npz')).hidden
            seed=int(key.rsplit('_s',1)[1]) if key.startswith('bag_') else cfg.seed
            cores[key]=JointNet(hidden,seed).fit(ctx['matrices'][representation],target,weights,ctx['bins'],cfg)
        elif kind=='JointRidge':
            representation=next(s['representation'] for s in specs.values() if key in s.get('cores',[]))
            strength=float(key.rsplit('_a',1)[1])
            cores[key]=JointRidge().fit(ctx['matrices'][representation],target,weights,ctx['bins'],strength,cfg.bias_penalty)
        elif kind=='HierarchicalSurface':
            strength=float(key.rsplit('_a',1)[1])
            tier_penalty=next(p[1] for p in source_plan['statistical']['hierarchical_penalties'] if p[0]==strength)
            case_weights=np.zeros(len(cases))
            for g,t in zip(*np.where(ctx['index']>=0)):case_weights[ctx['index'][g,t]]=weights[g,t]
            cores[key]=HierarchicalSurface().fit(ctx['surface'],cases,data,case_weights,cfg.tiers,strength,tier_penalty,cfg.bias_penalty)
        else:raise ValueError('Unknown frozen core: '+kind)
    for spec in specs.values():
        if spec['kind']=='statistical':spec['factors']=fit_calibration(cases,data,cfg,spec['alpha'],fit_cutoff)
        elif spec['kind']=='trajectory':
            changed=[replace(c,pace=float(v)) for c,v in zip(cases,ctx['shape'][spec['bandwidth']])]
            spec['factors']=fit_calibration(changed,data,cfg,1,fit_cutoff)
    selection['rule']='Original model choices retained; all fitted weights and statistical correction factors refitted on authorized earlier events'
    atomic_json(output/'selection.json',selection);atomic_json(output/'exclusions.json',excluded_cases)
    (output/'models').mkdir();compression.save(output/'models/compression.npz')
    for key,core in cores.items():core.save(output/'models'/(key+'.npz'))
    save_bank(output,data,ids,cfg.tiers)
    files=['plan.json','config.json','selection.json','reference_bank.json','reference_bank.npz','models/compression.npz']+['models/'+k+'.npz' for k in cores]
    metadata=dict(version=3,role='point_only_full_refit',output_tiers=cfg.tiers,training_event_ids=ids,
        excluded_event_ids=list(excluded),fit_cutoff=fit_cutoff,checkpoint_hours=list(hours),core_kinds=original['core_kinds'],
        training_joint_checkpoints=len(target),supervised_training_targets=int(np.count_nonzero(weights)),
        training_targets_by_tier={str(t):int(np.count_nonzero(weights[:,i])) for i,t in enumerate(cfg.tiers)},
        hashes={f:digest(output/f) for f in files},limitations=['Historical publication times and schedule revisions may be unknown',
            'Retrospective comparison; event 325 was previously inspected','Closing tracker proxies are not independently verified official finals'])
    atomic_json(output/'model.json',metadata)
    print('Full refit complete:',len(ids),'training events;',metadata['supervised_training_targets'],'supervised targets; champion',selection['champion'],flush=True)
    return metadata


def predict_point(live,model,event_id,cutoff,output=None,max_stale_hours=None):
    model=Path(model);metadata=json.loads((model/'model.json').read_text())
    for file,expected in metadata['hashes'].items():
        if digest(model/file)!=expected:raise ValueError('Model checksum mismatch: '+file)
    local=datetime.fromtimestamp(cutoff,CN)
    if local.hour not in metadata['checkpoint_hours'] or (local.minute,local.second,local.microsecond)!=(0,0,0):raise ValueError('Checkpoint hour was not included in this fit')
    if event_id in metadata['training_event_ids'] or cutoff<metadata['fit_cutoff']:raise ValueError('Prediction reuses training event or precedes fitting')
    cfg=Config.load(model/'config.json');freshness_hours=cfg.max_stale_minutes/60
    if max_stale_hours is not None:
        if not math.isfinite(max_stale_hours) or max_stale_hours<=0:raise ValueError('Score age limit must be finite and positive')
        cfg=replace(cfg,max_stale_minutes=max_stale_hours*60)
    data=attach_bank(model,live)
    predictor=CachedAnalog(data,cfg,metadata['training_event_ids']);cases=[];failures=[]
    for tier in cfg.tiers:
        try:cases.append(predictor.build(data.events[event_id],tier,cutoff))
        except ValueError as error:failures.append(dict(tier=tier,status='unavailable',reason=str(error)))
    ctx=context(cases,data,cfg,metadata['training_event_ids'],Compression.load(model/'models/compression.npz'))
    cores={key:CLASSES[kind].load(model/'models'/(key+'.npz')) for key,kind in metadata['core_kinds'].items()}
    selection=json.loads((model/'selection.json').read_text())
    predictions=ensemble_predictions({name:component_prediction(spec,ctx,cores,cfg.tiers) for name,spec in selection['components'].items()},selection['ensembles'])
    points=predictions[selection['champion']]
    rows=[dict(tier=c.tier,status='ok',current=math.ceil(c.current),observed_at=iso(c.observed_at),final=math.ceil(value),reference_events=len(c.references),
               score_age_hours=(cutoff-c.observed_at)/3600,stale_input=cutoff-c.observed_at>freshness_hours*3600) for c,value in zip(cases,points)]
    result=dict(event_id=event_id,checkpoint_cn=local.isoformat(),hours_remaining=(data.events[event_id].end-cutoff)/3600,
        model=str(model.resolve()),champion=selection['champion'],predictions=sorted(rows+failures,key=lambda r:r['tier']),limitations=metadata['limitations'],
        maximum_score_age_hours=cfg.max_stale_minutes/60,prediction_implementation_sha256=digest(__file__))
    if output:
        if Path(output).exists():raise ValueError('Fixed prediction already exists')
        atomic_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--data',required=True);fit.add_argument('--source-model',required=True);fit.add_argument('--output',required=True)
    fit.add_argument('--exclude',nargs='+',type=int,default=[325,326]);fit.add_argument('--hours',nargs='+',type=int,default=[8,20,21])
    fit.add_argument('--registry',default='evaluation_registry.json');fit.add_argument('--reuse-earlier-holdouts',action='store_true')
    pred=commands.add_parser('predict');pred.add_argument('--data',required=True);pred.add_argument('--model',required=True)
    pred.add_argument('--event',required=True,type=int);pred.add_argument('--as-of',required=True);pred.add_argument('--output',required=True)
    pred.add_argument('--max-stale-hours',type=float,help='Permit an explicitly marked last-known-score forecast across a tracker gap; never use future observations')
    args=p.parse_args()
    if args.command=='train':
        cfg=Config.load(Path(args.source_model)/'config.json');data=Dataset.load(args.data,cfg.tiers)
        refit(data,args.source_model,args.output,args.exclude,args.hours,args.registry,args.reuse_earlier_holdouts)
    else:
        cfg=Config.load(Path(args.model)/'config.json');data=Dataset.load(args.data,cfg.tiers);cutoff=timestamp(args.as_of)
        if cutoff is None:raise ValueError('Use a timezone-qualified timestamp')
        print(json.dumps(predict_point(data,args.model,args.event,cutoff,args.output,args.max_stale_hours),indent=2))


if __name__=='__main__':main()
