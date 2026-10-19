"""Run one predeclared comparison, then forecast both systems from frozen fits."""
from collections import defaultdict
from dataclasses import asdict,replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN,atomic_json,checkpoints,iso,write_csv
from .analog import AnalogPredictor,FEATURES,horizon,reconcile
from .config import Config
from .dataset import Curve,Dataset,Event
from .evaluation import interval_radius,paired_bootstrap,summary
from .joint import JointNet,pack
from .statistical import fit_calibration,predict as statistical_forecast


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_events(data,config,protected=()):
    events=data.eligible_events(config)
    n=config.tuning_events+config.calibration_events+config.test_events
    if len(events)<config.min_train_events+n:raise ValueError('Not enough labeled events for the predeclared chronological split')
    test=events[-config.test_events:]
    if set(protected)&{e.event_id for e in test}:
        raise ValueError('Newest test block includes previously inspected events; wait for new labeled events')
    calibration=events[-config.test_events-config.calibration_events:-config.test_events]
    tune=events[-n:-config.test_events-config.calibration_events]
    blocked=set(protected)|{e.event_id for e in tune+calibration+test}
    training=[e for e in events[:-n] if e.event_id not in blocked and data.label_available(e.event_id,config)<=tune[0].start]
    if len(training)<config.min_train_events:raise ValueError('Too few training events after permanent holdout exclusions')
    return {'train':[e.event_id for e in training],'tune':[e.event_id for e in tune],
            'calibration':[e.event_id for e in calibration],'test':[e.event_id for e in test],
            'fit_cutoff':tune[0].start}


def build_cases(data,config,event_ids,reference_ids):
    predictor=AnalogPredictor(data,config,reference_ids)
    cases=[];exclusions=[]
    for eid in event_ids:
        event=data.events[eid]
        for cutoff in checkpoints(event.start,event.end,event.end):
            for tier in config.tiers:
                if (eid,tier) not in data.curves:continue
                try:cases.append(predictor.build(event,tier,cutoff))
                except ValueError as error:
                    exclusions.append(dict(event_id=eid,tier=tier,checkpoint=iso(cutoff),reason=str(error)))
    return cases,exclusions


def training_arrays(cases,indices,data,config,fit_cutoff):
    targets=np.zeros((*indices.shape,3));targets[...,2]=1
    weights=np.zeros(indices.shape);cells=defaultdict(list)
    for g,t in zip(*np.where(indices>=0)):
        c=cases[indices[g,t]];truth=data.labels.get((c.event_id,c.tier))
        if truth is None or truth<c.current:continue
        targets[g,t]=[c.current,max(0,c.baseline-c.current),truth]
        cells[(c.event_id,horizon(c.remaining_hours))].append((g,t,c))
    ages={eid:config.event_weight((fit_cutoff-data.events[eid].end)/86400,data.events[eid].start) for eid,_ in cells}
    totals=defaultdict(float)
    for eid,h in cells:totals[h]+=ages[eid]
    for (eid,h),positions in cells.items():
        for g,t,c in positions:
            quality=config.proxy_label_weight if data.label_sources.get((eid,c.tier))=='post_close_tracker_proxy' else 1
            weights[g,t]=quality*ages[eid]/totals[h]/len(positions)/len(totals)
    return targets,weights


def neural_forecast(cases,net,gain,tiers):
    if not cases:return np.array([])
    x,index,_=pack(cases,tiers)
    corrections=net.predict(x) if gain else np.zeros(index.shape)
    predictions=np.array([c.baseline for c in cases])
    for g,t in zip(*np.where(index>=0)):
        i=index[g,t];c=cases[i]
        predictions[i]=c.current+max(0,c.baseline-c.current)*math.exp(gain*corrections[g,t])
    return reconcile(cases,predictions)


def admissible(result,baseline):
    # These rules and tolerances are fixed before opening tuning/test results.
    if result['selection_score'] is None:return False
    if abs(result['overall']['bias'])>abs(baseline['overall']['bias'])+.0025:return False
    late=result['by_horizon'].get('0-24h',{})
    base=baseline['by_horizon'].get('0-24h',{})
    return not base.get('rows') or (late.get('rows',0)>0 and late['mape']<=base['mape']+.0025)


def save_bank(output,data,training,tiers):
    ids=set(training)
    atomic_json(output/'reference_bank.json',{'events':[asdict(data.events[i]) for i in training],
        'labels':[[e,t,y,data.label_sources.get((e,t),'provided_verified')] for (e,t),y in data.labels.items() if e in ids and t in tiers],
        'outages':[r for r in data.outages if r[0] in ids],'revisions':[r for r in data.revisions if r[0] in ids],
        'final_available':{str(e):t for e,t in data.final_available.items() if e in ids}})
    np.savez_compressed(output/'reference_bank.npz',**{f'e{e}_t{t}':np.column_stack((c.times,c.scores,c.available))
        for (e,t),c in data.curves.items() if e in ids and t in tiers})


def attach_bank(output,live):
    bank=json.loads((output/'reference_bank.json').read_text())
    events=dict(live.events);ids={e['event_id'] for e in bank['events']}
    events.update({e['event_id']:Event(**e) for e in bank['events']})
    curves={k:c for k,c in live.curves.items() if k[0] not in ids}
    with np.load(output/'reference_bank.npz',allow_pickle=False) as arrays:
        for name in arrays.files:
            e,t=(int(part[1:]) for part in name.split('_'));a=arrays[name]
            curves[(e,t)]=Curve(a[:,0],a[:,1],a[:,2])
    labels={(e,t):y for e,t,y,_ in bank['labels']};sources={(e,t):s for e,t,_,s in bank['labels']}
    return Dataset(list(events.values()),curves,labels,
                   [r for r in live.outages if r[0] not in ids]+bank['outages'],
                   [r for r in live.revisions if r[0] not in ids]+bank['revisions'],live.provenance,sources,
                   {int(e):t for e,t in bank.get('final_available',{}).items()})


def train(data,config,output,registry='evaluation_registry.json'):
    if not config.tiers:config=replace(config,tiers=tuple(sorted({t for _,t in data.curves})))
    data.add_terminal_labels(config)
    output,registry=Path(output),Path(registry)
    if output.exists() and any(output.iterdir()):raise ValueError('Experiment directories are immutable; choose a new path')
    ledger=json.loads(registry.read_text()) if registry.exists() else {'protected_event_ids':[]}
    split=split_events(data,config,ledger['protected_event_ids'])
    protected=sorted(set(ledger['protected_event_ids'])|set(split['tune']+split['calibration']+split['test']))
    atomic_json(registry,{'protected_event_ids':protected,'policy':'Inspected holdouts never enter fitting or a fresh final test.'})
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'plan.json',{'config':config.to_dict(),'split':split,'input':data.provenance,
        'implementation_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'systems':['completion baseline + joint NN','completion/pace statistical mixture'],
        'features':FEATURES,'neural_correction_bound':.5,'neural_bias_penalty':config.bias_penalty,
        'selection_score':'Mean over tier/horizon cells of event-balanced MAPE + absolute event-balanced bias',
        'guards':'Tuning absolute overall bias and last-24h MAPE may worsen by at most 0.25 percentage points',
        'test_policy':'Select both systems before creating final-test cases; final test must be previously uninspected',
        'label_policy':'Archived/manual labels by default; opt-in closing proxies identified separately'})
    atomic_json(output/'config.json',config.to_dict());atomic_json(output/'split.json',split)
    partitions={};exclusions={}
    for part in ('train','tune','calibration'):
        print('Building causal cases:',part,flush=True)
        partitions[part],exclusions[part]=build_cases(data,config,split[part],split['train'])
    training,tune=partitions['train'],partitions['tune']
    if not training or not tune:raise ValueError('No usable training/tuning cases')
    x,indices,bins=pack(training,config.tiers)
    targets,weights=training_arrays(training,indices,data,config,split['fit_cutoff'])
    if weights.sum()<=0:raise ValueError('No supervised training targets')
    base_tune=reconcile(tune,[c.baseline for c in tune]);base_summary=summary(tune,base_tune,data)
    if base_summary['selection_score'] is None:raise ValueError('Tuning has no labels')
    candidates=[];nets={};tune_outputs={'baseline':base_tune}
    selected_nn={'hidden':config.hidden_sizes[0],'gain':0.,'score':base_summary['selection_score']}
    selected_stat={'alpha':0.,'calibrated':False,'score':base_summary['selection_score'],'factors':{}}
    print('Joint training checkpoints:',len(x),'supervised targets:',int(np.count_nonzero(weights)),flush=True)
    for hidden in config.hidden_sizes:
        print('Fit fresh joint NN:',hidden,'hidden units',flush=True)
        net=JointNet(hidden,config.seed).fit(x,targets,weights,bins,config);nets[hidden]=net
        for gain in config.neural_gains:
            prediction=neural_forecast(tune,net,gain,config.tiers);result=summary(tune,prediction,data)
            name=f'nn_h{hidden}_g{gain:g}';tune_outputs[name]=prediction
            candidate={'system':'baseline_nn','name':name,'hidden':hidden,'gain':gain,
                       'score':result['selection_score'],'admissible':admissible(result,base_summary),'metrics':result}
            candidates.append(candidate)
            if candidate['admissible'] and candidate['score']<selected_nn['score']-1e-12:
                selected_nn={k:candidate[k] for k in ('hidden','gain','score')}
    for alpha in config.statistical_blends:
        factors=fit_calibration(training,data,config,alpha,split['fit_cutoff'])
        prediction=statistical_forecast(tune,alpha,factors);result=summary(tune,prediction,data)
        name=f'stat_alpha{alpha:g}';tune_outputs[name]=prediction
        candidate={'system':'statistical','name':name,'alpha':alpha,'score':result['selection_score'],
                   'admissible':admissible(result,base_summary),'metrics':result}
        candidates.append(candidate)
        if candidate['admissible'] and candidate['score']<selected_stat['score']-1e-12:
            selected_stat={'alpha':alpha,'calibrated':True,'score':candidate['score'],'factors':factors}
    selection={'baseline_nn':selected_nn,'statistical':selected_stat}
    atomic_json(output/'selection.json',selection)
    print('Both selections locked:',{k:{a:b for a,b in v.items() if a!='factors'} for k,v in selection.items()},flush=True)
    partitions['test'],exclusions['test']=build_cases(data,config,split['test'],split['train'])
    net=nets[selected_nn['hidden']];net.save(output/'joint_model.npz');save_bank(output,data,split['train'],config.tiers)
    evaluation={'selection':selection,'candidates':candidates,'partitions':{},'uncertainty':{},'bootstrap':{},
                'joint_training_checkpoints':len(x),'supervised_training_targets':int(np.count_nonzero(weights)),
                'training_targets_by_tier':{str(t):int(np.count_nonzero(weights[:,i])) for i,t in enumerate(config.tiers)},
                'snapshot':data.provenance,'split_event_counts':{k:len(v) for k,v in split.items() if isinstance(v,list)}}
    audit=[]
    for part,cases in partitions.items():
        predictions={'baseline':reconcile(cases,[c.baseline for c in cases]),
                     'baseline_nn':neural_forecast(cases,net,selected_nn['gain'],config.tiers),
                     'statistical':statistical_forecast(cases,selected_stat['alpha'],selected_stat['factors'])}
        evaluation['partitions'][part]={name:summary(cases,p,data) for name,p in predictions.items()}
        if part=='calibration':
            evaluation['uncertainty']={name:interval_radius(cases,p,data,config.interval_alpha) for name,p in predictions.items()}
        if part=='test':
            evaluation['bootstrap']={name:paired_bootstrap(cases,predictions['baseline'],predictions[name],data,config.seed)
                                     for name in ('baseline_nn','statistical')}
        for i,c in enumerate(cases):
            audit.append(dict(partition=part,event_id=c.event_id,tier=c.tier,checkpoint=iso(c.cutoff),
                observed_at=iso(c.observed_at),hours_remaining=c.remaining_hours,current=c.current,
                baseline=float(predictions['baseline'][i]),baseline_nn=float(predictions['baseline_nn'][i]),
                statistical=float(predictions['statistical'][i]),actual=data.labels.get((c.event_id,c.tier),''),
                label_source=data.label_sources.get((c.event_id,c.tier),'missing'),pace_estimate=c.pace,
                pace_references=c.pace_references,reference_event_ids=','.join(map(str,c.references))))
    write_csv(output/'predictions.csv',list(audit[0]),audit)
    atomic_json(output/'evaluation.json',evaluation);atomic_json(output/'exclusions.json',exclusions)
    files=('config.json','split.json','selection.json','joint_model.npz','reference_bank.json','reference_bank.npz')
    atomic_json(output/'model.json',{'version':1,'output_tiers':config.tiers,'fit_cutoff':split['fit_cutoff'],
        'training_event_ids':split['train'],'training_targets_by_tier':evaluation['training_targets_by_tier'],
        'uncertainty':evaluation['uncertainty'],'hashes':{f:digest(output/f) for f in files},
        'limitations':['Historical publication/schedule revision times may be unknown',
                       'Tracker gaps are not proof of server outages',
                       'No accuracy claim for unevaluated tiers or modern events without labels']})
    return evaluation


def predict(live,model,event_id,cutoff,output=None):
    directory=Path(model);metadata=json.loads((directory/'model.json').read_text())
    for name,expected in metadata['hashes'].items():
        if digest(directory/name)!=expected:raise ValueError('Model checksum mismatch: '+name)
    local=datetime.fromtimestamp(cutoff,CN)
    if (local.hour,local.minute,local.second,local.microsecond)!=(8,0,0,0):raise ValueError('Checkpoint must be exactly 08:00 CN')
    if cutoff<metadata['fit_cutoff'] or event_id in metadata['training_event_ids']:raise ValueError('Prediction would precede fitting or reuse a training event')
    config=Config.load(directory/'config.json');data=attach_bank(directory,live)
    if event_id not in data.events:raise ValueError('Event absent from input data')
    predictor=AnalogPredictor(data,config,metadata['training_event_ids']);cases=[];failures=[]
    for tier in config.tiers:
        try:cases.append(predictor.build(data.events[event_id],tier,cutoff))
        except ValueError as error:failures.append(dict(tier=tier,status='unavailable',reason=str(error)))
    selection=json.loads((directory/'selection.json').read_text());stat=selection['statistical']
    predictions={'baseline':reconcile(cases,[c.baseline for c in cases]),
                 'baseline_nn':neural_forecast(cases,JointNet.load(directory/'joint_model.npz'),selection['baseline_nn']['gain'],config.tiers),
                 'statistical':statistical_forecast(cases,stat['alpha'],stat['factors'])}
    results=[]
    for i,c in enumerate(cases):
        estimates={}
        for name,values in predictions.items():
            radius=metadata['uncertainty'].get(name,{}).get('by_tier',{}).get(str(c.tier),{}).get('log_radius');value=values[i]
            estimates[name]={'final':math.ceil(value),
                'lower':max(math.ceil(c.current),math.floor(value*math.exp(-radius))) if radius is not None else None,
                'upper':math.ceil(value*math.exp(radius)) if radius is not None else None}
        results.append(dict(tier=c.tier,status='ok',current=math.ceil(c.current),observed_at=iso(c.observed_at),
            forecasts=estimates,reference_events=len(c.references),pace_reference_events=c.pace_references,
            neural_supervised_targets=metadata['training_targets_by_tier'][str(c.tier)]))
    result={'event_id':event_id,'checkpoint_cn':local.isoformat(),'model':str(directory.resolve()),
            'predictions':sorted(results+failures,key=lambda r:r['tier']),'limitations':metadata['limitations']}
    if output:
        if Path(output).exists():raise ValueError('Fixed prediction already exists; choose a new path')
        atomic_json(output,result)
    return result
