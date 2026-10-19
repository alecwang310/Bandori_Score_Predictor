"""Refit frozen observer choices on the already authorized full training bank."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np

from collect_bandori_data import CN, atomic_json, iso, timestamp, write_csv
from .advanced_statistics import CachedAnalog, HierarchicalSurface, JointRidge
from .config import Config
from .dataset import Dataset
from .experiments import context, ensemble_predictions
from .full_refit import issue_times
from .joint import JointNet
from .observer_experiment import (CLASSES, augment, controls_at, daily_prediction,
                                  flat_pace, raw_stat)
from .pipeline import attach_bank, digest, save_bank, training_arrays
from .representations import Compression
from .statistical import fit_calibration
from .temporal import JointTemporalNet


def verified_metadata(directory):
    directory=Path(directory);metadata=json.loads((directory/'model.json').read_text())
    for path,expected in metadata['hashes'].items():
        if digest(directory/path)!=expected:raise ValueError('Model checksum mismatch: '+path)
    return metadata


def refit(data,control,observer,output,registry='evaluation_registry.json',reuse_earlier_holdouts=False):
    control,observer,output=Path(control),Path(observer),Path(output)
    if output.exists():raise ValueError('Use a new immutable refit directory')
    old=verified_metadata(control);original=verified_metadata(observer)
    config=Config.load(control/'config.json')
    if config.to_dict()!=Config.load(observer/'config.json').to_dict():raise ValueError('Refit and observer configurations differ')
    data.add_terminal_labels(config)
    ids=old['training_event_ids'];excluded=old.get('excluded_event_ids',[])
    if old.get('role')!='point_only_full_refit' or not excluded:raise ValueError('Use the explicit-holdout full control refit')
    if set(ids)&set(excluded):raise ValueError('Training overlaps excluded events')
    if any(data.label_available(e,config)>old['fit_cutoff'] for e in ids):raise ValueError('Training label not available at fit time')
    ledger_before=digest(registry) if Path(registry).exists() else None
    protected=json.loads(Path(registry).read_text())['protected_event_ids'] if ledger_before else []
    reused=sorted(set(ids)&set(protected))
    if reused and not reuse_earlier_holdouts:raise ValueError('Earlier holdout reuse needs the explicit authorized-refit flag')
    selection=json.loads((observer/'selection.json').read_text())
    controls=json.loads((control/'selection.json').read_text())
    if selection['control_champion']!=controls['champion']:raise ValueError('Frozen control champions differ')
    output.mkdir(parents=True);(output/'models').mkdir()
    atomic_json(output/'plan.json',dict(role='full_observer_refit',control=str(control.resolve()),observer_source=str(observer.resolve()),
        config=config.to_dict(),training_event_ids=ids,excluded_event_ids=excluded,fit_cutoff=old['fit_cutoff'],
        checkpoint_hours=old['checkpoint_hours'],frozen_selection_sha256=digest(observer/'selection.json'),
        reused_earlier_protected_events=reused,authorization='Earlier user instruction: train all eligible events except 325 and 326',
        evaluation_policy='No selection against 325/326; previously inspected retrospective cases only',
        implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    atomic_json(output/'config.json',config.to_dict());atomic_json(output/'control_selection.json',controls)
    predictor=CachedAnalog(data,config,ids);cases=[];failures=[]
    for number,eid in enumerate(ids):
        if number%25==0:print('Build full observer refit:',number,'/',len(ids),flush=True)
        event=data.events[eid]
        for cutoff in issue_times(event,old['checkpoint_hours']):
            for tier in config.tiers:
                if (eid,tier) not in data.curves:continue
                try:cases.append(predictor.build(event,tier,cutoff))
                except ValueError as error:failures.append(dict(event_id=eid,tier=tier,checkpoint=iso(cutoff),reason=str(error)))
    print('Build full observer representations:',len(cases),'cases',flush=True)
    compression=Compression.load(control/'models/compression.npz')
    ctx=augment(context(cases,data,config,ids,compression),data,config,ids)
    target,weights=training_arrays(cases,ctx['index'],data,config,old['fit_cutoff'])
    print('Supervised targets:',int(np.count_nonzero(weights)),flush=True)
    specs=[*selection['components'].values(),*selection['paired_raw'].values()];cores={}
    case_weights=np.zeros(len(cases))
    for g,t in zip(*np.where(ctx['index']>=0)):case_weights[ctx['index'][g,t]]=weights[g,t]
    for key,kind in original['core_kinds'].items():
        print('Refit observer core:',key,flush=True)
        source_core=CLASSES[kind].load(observer/'models'/(key+'.npz'))
        if kind=='JointNet':
            representation=next(s['representation'] for s in specs if key in s.get('cores',[]))
            seed=int(key.rsplit('_s',1)[1]) if key.startswith('daily_bag_') else config.seed
            core=JointNet(source_core.hidden,seed).fit(ctx['augmented'][representation],target,weights,ctx['bins'],config)
        elif kind=='JointRidge':
            representation=next(s['representation'] for s in specs if key in s.get('cores',[]))
            strength=float(key.rsplit('_a',1)[1])
            core=JointRidge().fit(ctx['augmented'][representation],target,weights,ctx['bins'],strength,config.bias_penalty)
        elif kind=='HierarchicalSurface':
            strength=float(key.rsplit('_a',1)[1])
            core=HierarchicalSurface().fit(ctx['daily_surface'],cases,data,case_weights,config.tiers,strength,10*strength,config.bias_penalty)
        elif kind=='JointTemporalNet':
            core=JointTemporalNet(source_core.channels,source_core.hidden,config.seed).fit(ctx['augmented']['standard'],ctx['daily']['sequence'],target,weights,ctx['bins'],config)
        else:raise ValueError('Unknown observer core')
        cores[key]=core;core.save(output/'models'/(key+'.npz'))
    need_flat=any(s.get('flat_clock') for s in specs)
    if need_flat:ctx['flat_pace']=flat_pace(ctx,data,config,ids)
    factor_cache={}
    for spec in specs:
        if spec['kind']!='daily_stat' or spec['family']=='baseline':continue
        cache_key=(spec['family'],spec['blend'],spec.get('flat_clock',False))
        if cache_key not in factor_cache:
            pace=ctx['flat_pace'] if spec.get('flat_clock') else ctx['daily']['pace']
            values=(1-spec['blend'])*raw_stat(spec['family'],ctx,controls['components'])+spec['blend']*pace
            changed=[replace(c,baseline=float(p),pace=float(p)) for c,p in zip(cases,values)]
            factor_cache[cache_key]=fit_calibration(changed,data,config,0,old['fit_cutoff'])
        spec['factors']=factor_cache[cache_key]
    selection['rule']='Observer architecture, gains and champion frozen on earlier tuning; only coefficients and training-only factors refitted'
    atomic_json(output/'selection.json',selection);atomic_json(output/'exclusions.json',failures)
    for key in old['core_kinds']:shutil.copy2(control/'models'/(key+'.npz'),output/'models'/('control_'+key+'.npz'))
    compression.save(output/'models/compression.npz');save_bank(output,data,ids,config.tiers)
    files=['plan.json','config.json','control_selection.json','selection.json','reference_bank.json','reference_bank.npz']
    files+=['models/'+p.name for p in (output/'models').glob('*.npz')]
    metadata=dict(version=1,kind='full_daily_observer',fit_cutoff=old['fit_cutoff'],training_event_ids=ids,
        excluded_event_ids=excluded,checkpoint_hours=old['checkpoint_hours'],core_kinds=original['core_kinds'],
        control_core_kinds=old['core_kinds'],training_joint_checkpoints=len(target),
        supervised_training_targets=int(np.count_nonzero(weights)),
        training_targets_by_tier={str(t):int(np.count_nonzero(weights[:,i])) for i,t in enumerate(config.tiers)},
        hashes={f:digest(output/f) for f in files},limitations=[
            'Frozen observer choices refitted on the authorized full bank; earlier research holdouts reused explicitly',
            'Event 325 is previously inspected and unusually long; it is not a fresh generalization test',
            'Closing tracker proxies are not independently verified official finals',
            'Historical publication and schedule revision timing may be incomplete',
            'No new probability intervals have been calibrated'])
    atomic_json(output/'model.json',metadata)
    if ledger_before and digest(registry)!=ledger_before:raise ValueError('Holdout ledger changed unexpectedly')
    print('Full observer refit complete:',len(ids),'events;',metadata['supervised_training_targets'],'targets',flush=True)
    return metadata


def predict(live,model,event_id,cutoff,output=None,max_stale_hours=None,all_models=False):
    model=Path(model);metadata=verified_metadata(model);config=Config.load(model/'config.json')
    local=datetime.fromtimestamp(cutoff,CN)
    if local.hour not in metadata['checkpoint_hours'] or (local.minute,local.second,local.microsecond)!=(0,0,0):
        raise ValueError('Checkpoint hour was not fitted')
    if event_id in metadata['training_event_ids'] or cutoff<metadata['fit_cutoff']:raise ValueError('Training event or pre-fit forecast')
    freshness=config.max_stale_minutes/60
    if max_stale_hours is not None:
        if not math.isfinite(max_stale_hours) or max_stale_hours<=0:raise ValueError('Invalid score-age limit')
        config=replace(config,max_stale_minutes=max_stale_hours*60)
    data=attach_bank(model,live);cases=[];failures=[];ids=metadata['training_event_ids']
    predictor=CachedAnalog(data,config,ids)
    for tier in config.tiers:
        try:cases.append(predictor.build(data.events[event_id],tier,cutoff))
        except ValueError as error:failures.append(dict(tier=tier,status='unavailable',reason=str(error)))
    if not cases:raise ValueError('No available tiers')
    ctx=augment(context(cases,data,config,ids,Compression.load(model/'models/compression.npz')),data,config,ids)
    ctx['flat_pace']=flat_pace(ctx,data,config,ids)
    selection=json.loads((model/'selection.json').read_text());control_selection=json.loads((model/'control_selection.json').read_text())
    cores={key:CLASSES[kind].load(model/'models'/(key+'.npz')) for key,kind in metadata['core_kinds'].items()}
    control_cores={key:CLASSES[kind].load(model/'models'/('control_'+key+'.npz')) for key,kind in metadata['control_core_kinds'].items()}
    control=controls_at(control_selection,ctx,control_cores,config.tiers)
    predictions=ensemble_predictions({n:daily_prediction(s,ctx,cores,control,control_selection['components'],config.tiers)
                                      for n,s in selection['components'].items()},selection['ensembles'])
    predictions['control_champion']=control[selection['control_champion']]
    raw={n:daily_prediction(s,ctx,cores,control,control_selection['components'],config.tiers) for n,s in selection['paired_raw'].items()} if all_models else {}
    rows=[]
    for i,c in enumerate(cases):
        row=dict(tier=c.tier,status='ok',current=math.ceil(c.current),final=math.ceil(predictions[selection['champion']][i]),
            observed_at=iso(c.observed_at),score_age_hours=(cutoff-c.observed_at)/3600,
            stale_input=cutoff-c.observed_at>freshness*3600,hourly_coverage=float(ctx['daily']['coverage'][i]),
            supervised_training_targets=metadata['training_targets_by_tier'][str(c.tier)])
        if all_models:row.update(control={n:float(p[i]) for n,p in control.items()},observer={n:float(p[i]) for n,p in predictions.items()},observer_raw={n:float(p[i]) for n,p in raw.items()})
        rows.append(row)
    result=dict(event_id=event_id,checkpoint_cn=local.isoformat(),champion=selection['champion'],predictions=sorted(rows+failures,key=lambda r:r['tier']),limitations=metadata['limitations'])
    if output:
        if Path(output).exists():raise ValueError('Fixed prediction already exists')
        atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--data',required=True);fit.add_argument('--control',required=True)
    fit.add_argument('--observer',required=True);fit.add_argument('--output',required=True);fit.add_argument('--registry',default='evaluation_registry.json')
    fit.add_argument('--reuse-earlier-holdouts',action='store_true')
    pred=commands.add_parser('predict');pred.add_argument('--data',required=True);pred.add_argument('--model',required=True)
    pred.add_argument('--event',required=True,type=int);pred.add_argument('--as-of',required=True);pred.add_argument('--output',required=True)
    pred.add_argument('--max-stale-hours',type=float)
    args=parser.parse_args()
    if args.command=='train':
        config=Config.load(Path(args.control)/'config.json');data=Dataset.load(args.data,config.tiers)
        refit(data,args.control,args.observer,args.output,args.registry,args.reuse_earlier_holdouts)
    else:
        config=Config.load(Path(args.model)/'config.json');data=Dataset.load(args.data,config.tiers);cutoff=timestamp(args.as_of)
        if cutoff is None:raise ValueError('Timezone-qualified --as-of required')
        print(json.dumps(predict(data,args.model,args.event,cutoff,args.output,args.max_stale_hours),indent=2))


if __name__=='__main__':main()
