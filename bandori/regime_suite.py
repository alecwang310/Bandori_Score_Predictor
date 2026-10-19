"""Dated-rule, 3x-weighted all-family refits and chronological robustness."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN, atomic_bytes, atomic_json, iso, timestamp, write_csv
from .advanced_statistics import HierarchicalSurface, JointRidge
from .analog import Case, HORIZONS, horizon, reconcile
from .config import Config
from .dataset import Dataset
from .evaluation import breakdown, paired_bootstrap, summary
from .experiments import FAMILIES, NEURAL, admissible, context, correction_forecast, ensemble_predictions
from .joint import JointNet
from .lstm_variants import SupportedInputLSTM
from .observer_experiment import augment
from .pipeline import attach_bank, digest, save_bank, training_arrays
from .raw_sequence import pack_streams
from .regime_sequence import forecast_event, training_streams
from .representations import Compression
from .role_analog import RoleAwareAnalog, build_cases
from .rules import FEATURES, RULE_DATE, RULE_START, SOURCES, era, packed_metadata
from .statistical import fit_calibration, predict as statistical_predict
from .temporal import JointTemporalNet


CLASSES={'JointNet':JointNet,'JointRidge':JointRidge,'HierarchicalSurface':HierarchicalSurface,
         'JointTemporalNet':JointTemporalNet,'SupportedInputLSTM':SupportedInputLSTM}
ENSEMBLES={
    'neural_mean':dict(method='mean',members=[*NEURAL,'tcn24','raw_lstm']),
    'statistical_mean':dict(method='mean',members=['ridge','calibrated_stat','trajectory','hierarchical']),
    **{name:dict(method=method,members=['baseline','raw_lstm','ridge','calibrated_stat','trajectory','hierarchical'])
       for name,method in (('collection_mean','mean'),('collection_median','median'),('collection_trimmed','trimmed_mean'))}}


def hyperparameters_only(value):
    if isinstance(value,dict):return {k:hyperparameters_only(v) for k,v in value.items() if k!='factors'}
    if isinstance(value,list):return [hyperparameters_only(v) for v in value]
    return value


def metadata(ctx,data):
    n,k=ctx['index'].shape;values=packed_metadata(ctx['cases'],ctx['index'],data)
    support=np.zeros((n,k,3))
    for g,t in zip(*np.where(ctx['index']>=0)):
        c=ctx['cases'][ctx['index'][g,t]]
        support[g,t,:2]=[math.log1p(c.features.get('role_transfer_support',0)),math.log1p(c.features.get('direct_tier_references',len(c.references)))]
        support[g,t,2]=sum(era(data.events[e])==era(data.events[c.event_id]) for e in c.references)/max(1,len(c.references))
    ctx['rule_metadata']=np.concatenate((values,support),axis=2)
    return ctx


def make_context(cases,data,config,ids):
    return metadata(augment(context(cases,data,config,ids),data,config,ids),data)


def subset(ctx,ids):
    allowed=set(ids);case_mask=np.array([c.event_id in allowed for c in ctx['cases']]);case_indices=np.flatnonzero(case_mask)
    mapping=np.full(len(case_mask),-1,int);mapping[case_indices]=np.arange(len(case_indices))
    rows=np.array([any(i>=0 and case_mask[i] for i in row) for row in ctx['index']])
    index=ctx['index'][rows].copy();valid=index>=0;index[valid]=mapping[index[valid]]
    result=dict(cases=[ctx['cases'][i] for i in case_indices],index=index,bins=ctx['bins'][rows],
        matrices={n:x[rows].copy() for n,x in ctx['matrices'].items()},shape={b:x[case_mask].copy() for b,x in ctx['shape'].items()},
        surface=ctx['surface'][case_mask].copy(),rule_metadata=ctx['rule_metadata'][rows].copy())
    result['daily']={n:(x[rows].copy() if n in ('summaries','sequence') else x[case_mask].copy()) for n,x in ctx['daily'].items()}
    if any(e not in allowed for c in result['cases'] for e in c.references):raise ValueError('Cached prefix references include a fold holdout')
    return result


def save_context(output,ctx):
    atomic_bytes(output/'cases.json.gz',gzip.compress(json.dumps([asdict(c) for c in ctx['cases']],separators=(',',':')).encode(),compresslevel=1))
    arrays=dict(index=ctx['index'],bins=ctx['bins'],surface=ctx['surface'],rule_metadata=ctx['rule_metadata'])
    arrays.update({'matrix_'+n:x for n,x in ctx['matrices'].items()})
    arrays.update({'shape_'+str(b):x for b,x in ctx['shape'].items()});arrays.update({'daily_'+n:x for n,x in ctx['daily'].items()})
    np.savez_compressed(output/'contexts.npz',**arrays)
    atomic_json(output/'cache.json',dict(hashes={f:digest(output/f) for f in ('cases.json.gz','contexts.npz')}))


def load_context(output):
    manifest=json.loads((output/'cache.json').read_text())
    for name,expected in manifest['hashes'].items():
        if digest(output/name)!=expected:raise ValueError('Context cache checksum mismatch')
    cases=[Case(**c) for c in json.loads(gzip.decompress((output/'cases.json.gz').read_bytes()))]
    ctx=dict(cases=cases,matrices={},shape={},daily={})
    with np.load(output/'contexts.npz',allow_pickle=False) as arrays:
        for name in arrays.files:
            x=arrays[name].copy()
            if name.startswith('matrix_'):ctx['matrices'][name[7:]]=x
            elif name.startswith('shape_'):ctx['shape'][float(name[6:])]=x
            elif name.startswith('daily_'):ctx['daily'][name[6:]]=x
            else:ctx[name]=x
    return ctx


def finish_context(ctx,compression,masks=None,weights=None):
    ctx['matrices']['pca']=compression.transform(ctx['matrices']['rich'])
    n,k=ctx['index'].shape
    extra=np.column_stack((ctx['daily']['summaries'].reshape(n,-1),ctx['rule_metadata'].reshape(n,-1)))
    ctx['matrices']={name:np.column_stack((x,extra)) for name,x in ctx['matrices'].items()}
    case_extra=np.zeros((len(ctx['cases']),len(FEATURES)+3+ctx['daily']['summaries'].shape[-1]))
    for g,t in zip(*np.where(ctx['index']>=0)):
        case_extra[ctx['index'][g,t]]=np.r_[ctx['daily']['summaries'][g,t],ctx['rule_metadata'][g,t]]
    ctx['surface']=np.column_stack((ctx['surface'],case_extra))
    if masks is None:
        positive=weights.sum(axis=1)>0
        masks={name:(np.ptp(x[positive],axis=0)>1e-8).astype(float) for name,x in ctx['matrices'].items()}
    ctx['matrices']={name:x*masks[name] for name,x in ctx['matrices'].items()}
    return ctx,masks


def raw_stat(family,ctx,base,observer):
    cases=ctx['cases'];b=np.array([c.baseline for c in cases])
    if family=='baseline':return b
    if family=='daily_baseline':return ctx['daily']['pace']
    spec=base['components'][family]
    if spec['kind']=='baseline':p=b
    elif family=='calibrated_stat':p=(1-spec['alpha'])*b+spec['alpha']*np.array([c.pace for c in cases])
    elif family=='trajectory':p=ctx['shape'][spec['bandwidth']]
    else:raise ValueError('Unknown statistical family')
    updated=observer['components'][family]
    blend=updated.get('blend',0) if updated['kind']=='daily_stat' else 0
    return (1-blend)*p+blend*ctx['daily']['pace']


def predictions(specs,cores,ctx,data,tiers):
    if not ctx['cases']:return {name:np.array([]) for name in [*specs,*ENSEMBLES]}
    result={}
    for name,spec in specs.items():
        kind=spec['kind']
        if kind=='baseline':result[name]=reconcile(ctx['cases'],[c.baseline for c in ctx['cases']])
        elif kind=='nn':
            delta=np.mean([cores[k].predict(ctx['matrices'][spec['representation']]) for k in spec['cores']],axis=0)
            result[name]=correction_forecast(ctx['cases'],ctx['index'],delta,spec['gain'])
        elif kind=='ridge':
            delta=cores[spec['core']].predict(ctx['matrices'][spec['representation']])
            result[name]=correction_forecast(ctx['cases'],ctx['index'],delta,spec['gain'],'ridge')
        elif kind=='temporal':
            delta=cores[spec['core']].predict(ctx['matrices']['standard'],ctx['daily']['sequence'])
            result[name]=correction_forecast(ctx['cases'],ctx['index'],delta,spec['gain'])
        elif kind=='hierarchical':
            delta=cores[spec['core']].predict(ctx['surface'],[c.tier for c in ctx['cases']])
            result[name]=reconcile(ctx['cases'],[c.current+max(0,c.baseline-c.current)*(1+spec['gain']*d) for c,d in zip(ctx['cases'],delta)])
        elif kind=='statistical':
            p=raw_stat(name,ctx,spec['base'],spec['observer'])
            changed=[replace(c,baseline=float(v),pace=float(v)) for c,v in zip(ctx['cases'],p)]
            result[name]=statistical_predict(changed,0,spec['factors'])
        elif kind=='lstm':
            values=np.zeros(len(ctx['cases']));positions={t:i for i,t in enumerate(tiers)};groups={}
            for i,c in enumerate(ctx['cases']):groups.setdefault(c.event_id,[]).append(i)
            for eid,indices in groups.items():
                cutoffs=sorted({ctx['cases'][i].cutoff for i in indices})
                output,_=forecast_event(cores[spec['core']],data,data.events[eid],tiers,cutoffs);slots={t:i for i,t in enumerate(cutoffs)}
                for i in indices:
                    c=ctx['cases'][i];values[i]=output[slots[c.cutoff],positions[c.tier]]
            result[name]=reconcile(ctx['cases'],values)
        else:raise ValueError('Unknown model kind')
    return ensemble_predictions(result,ENSEMBLES)


def fit_models(ctx,data,config,ids,fit_cutoff,base,observer,source,output,lstm_epochs=120):
    output.mkdir(parents=True,exist_ok=True);(output/'models').mkdir(exist_ok=True)
    targets,weights=training_arrays(ctx['cases'],ctx['index'],data,config,fit_cutoff)
    compression=Compression().fit(ctx['matrices']['rich'],weights,config.tiers,seed=config.seed)
    ctx,masks=finish_context(ctx,compression,weights=weights);cores={};specs={'baseline':dict(kind='baseline')}
    for family in NEURAL:
        old=base['components'][family]
        if old['kind']=='baseline':specs[family]=dict(kind='baseline');continue
        rep=old['representation'];keys=[];gain=observer['components'][family].get('gain',old['gain'])
        for old_key in old['cores']:
            key='rule_'+old_key;seed=int(old_key.rsplit('_s',1)[1]) if old_key.startswith('bag_') else config.seed
            hidden=JointNet.load(source/'models'/(old_key+'.npz')).hidden
            print('Fit',output.name,key,flush=True)
            cores[key]=JointNet(hidden,seed).fit(ctx['matrices'][rep],targets,weights,ctx['bins'],config);keys.append(key)
        specs[family]=dict(kind='nn',representation=rep,cores=keys,gain=gain)
    old=base['components']['ridge'];key='ridge';rep=old.get('representation','pca')
    old_key=old.get('cores',['ridge_pca_a0.01'])[0];strength=float(old_key.rsplit('_a',1)[1])
    print('Fit',output.name,'ridge',flush=True)
    cores[key]=JointRidge().fit(ctx['matrices'][rep],targets,weights,ctx['bins'],strength,config.bias_penalty)
    specs['ridge']=dict(kind='ridge',core=key,representation=rep,gain=observer['components']['ridge'].get('gain',old.get('gain',.5)))
    case_weights=np.zeros(len(ctx['cases']))
    for g,t in zip(*np.where(ctx['index']>=0)):case_weights[ctx['index'][g,t]]=weights[g,t]
    key='hierarchical';old=base['components'][key];strength=float(old.get('core','hierarchical_a0.01').rsplit('_a',1)[1])
    print('Fit',output.name,key,flush=True)
    cores[key]=HierarchicalSurface().fit(ctx['surface'],ctx['cases'],data,case_weights,config.tiers,strength,10*strength,config.bias_penalty)
    specs[key]=dict(kind='hierarchical',core=key,gain=observer['components'][key].get('gain',old.get('gain',1)))
    for family in ('daily_baseline','calibrated_stat','trajectory'):
        values=raw_stat(family,ctx,base,observer)
        changed=[replace(c,baseline=float(v),pace=float(v)) for c,v in zip(ctx['cases'],values)]
        factors=fit_calibration(changed,data,config,0,fit_cutoff) if family!='daily_baseline' else {}
        specs[family]=dict(kind='statistical',factors=factors,base=base,observer=observer)
    print('Fit',output.name,'tcn24',flush=True)
    cores['tcn24']=JointTemporalNet(4,8,config.seed).fit(ctx['matrices']['standard'],ctx['daily']['sequence'],targets,weights,ctx['bins'],config)
    specs['tcn24']=dict(kind='temporal',core='tcn24',gain=.5)
    print('Build',output.name,'raw LSTM streams',flush=True)
    streams=training_streams(data,ids,config.tiers,config.seed)
    bundle,lstm_targets,lstm_weights,lstm_bins=pack_streams(streams,data,config,fit_cutoff)
    import torch
    torch.set_num_threads(2)
    print('Fit',output.name,'raw LSTM; targets',int(np.count_nonzero(lstm_weights)),flush=True)
    cores['raw_lstm']=SupportedInputLSTM(32,16,config.seed,'cpu').fit(bundle,lstm_targets,lstm_weights,lstm_bins,lstm_epochs,.005,config.bias_penalty,.001,
        progress=lambda step,loss:print(output.name,'LSTM',step,'loss',round(loss,6),flush=True))
    specs['raw_lstm']=dict(kind='lstm',core='raw_lstm')
    compression.save(output/'models/compression.npz');np.savez_compressed(output/'models/input_masks.npz',**masks)
    kinds={}
    for key,core in cores.items():
        kinds[key]=type(core).__name__;core.save(output/'models'/(key+('.pt' if key=='raw_lstm' else '.npz')))
    atomic_json(output/'config.json',config.to_dict());atomic_json(output/'components.json',specs);save_bank(output,data,ids,config.tiers)
    files=['config.json','components.json','reference_bank.json','reference_bank.npz']+['models/'+p.name for p in (output/'models').iterdir()]
    model=dict(kind='rule_aware_3x',training_event_ids=ids,fit_cutoff=fit_cutoff,core_kinds=kinds,
        supervised_training_targets=int(np.count_nonzero(weights)),lstm_training_targets=int(np.count_nonzero(lstm_weights)),
        training_targets_by_tier={str(t):int(np.count_nonzero(weights[:,j])) for j,t in enumerate(config.tiers)},
        lstm_targets_by_tier={str(t):int(np.count_nonzero(lstm_weights[...,j])) for j,t in enumerate(config.tiers)},
        hashes={f:digest(output/f) for f in files})
    atomic_json(output/'model.json',model)
    return cores,specs,compression,masks,model


def load_fit(directory):
    meta=json.loads((directory/'model.json').read_text())
    for f,h in meta['hashes'].items():
        if digest(directory/f)!=h:raise ValueError('Model checksum mismatch: '+f)
    cores={key:CLASSES[kind].load(directory/'models'/(key+('.pt' if kind=='SupportedInputLSTM' else '.npz')),
                                   **({'device':'cpu'} if kind=='SupportedInputLSTM' else {})) for key,kind in meta['core_kinds'].items()}
    specs=json.loads((directory/'components.json').read_text());compression=Compression.load(directory/'models/compression.npz')
    with np.load(directory/'models/input_masks.npz',allow_pickle=False) as arrays:masks={k:arrays[k].copy() for k in arrays.files}
    return cores,specs,compression,masks,meta


def records(ctx,values,data,fold,scenario):
    return [dict(fold=fold,scenario=scenario,event_id=c.event_id,event_type=data.events[c.event_id].event_type,tier=c.tier,
        checkpoint=iso(c.cutoff),hours_remaining=c.remaining_hours,current=c.current,score_age_hours=(c.cutoff-c.observed_at)/3600,
        actual=data.labels.get((c.event_id,c.tier),''),label_source=data.label_sources.get((c.event_id,c.tier),'missing'),
        **{n:float(p[i]) for n,p in values.items()}) for i,c in enumerate(ctx['cases'])]


def stress_dataset(data,eid,cutoff,gap):
    from .dataset import Dataset
    curves=dict(data.curves)
    for key,curve in list(curves.items()):
        if key[0]==eid:curves[key]=curve.prefix(cutoff-gap*3600,data.events[eid].start)
    return Dataset(list(data.events.values()),curves,data.labels,data.outages,data.revisions,data.provenance,data.label_sources,data.final_available)


def run(data,source,observer_source,output,registry='evaluation_registry.json',lstm_epochs=120):
    source,observer_source,output=Path(source),Path(observer_source),Path(output)
    original=json.loads((source/'model.json').read_text());ids=original['training_event_ids']
    for name,expected in original['hashes'].items():
        if digest(source/name)!=expected:raise ValueError('Source checksum mismatch: '+name)
    config=replace(Config.load(source/'config.json'),post_rule_change_at=RULE_DATE,post_rule_change_multiplier=3)
    data.add_terminal_labels(config);ledger_before=digest(registry)
    if set(ids)&{325,326}:raise ValueError('Explicit holdout in training bank')
    ordered=sorted(ids,key=lambda e:data.events[e].start);new=[e for e in ordered if data.events[e].start>=RULE_START]
    if len(new)!=12:raise ValueError('Expected the frozen pre-325 bank with twelve post-rule events')
    folds=[]
    for i,label in enumerate(('cold_start','four_post_events','eight_post_events')):
        validation=new[4*i:4*i+4];fit_cutoff=min(data.events[e].start for e in validation)
        train_ids=[e for e in ordered if data.label_available(e,config)<=fit_cutoff and e not in validation]
        folds.append(dict(name=label,train=train_ids,validation=validation,fit_cutoff=fit_cutoff,
                          post_rule_training_events=sum(era(data.events[e]) for e in train_ids)))
    plan=dict(config=config.to_dict(),training_event_ids=ids,excluded=[325,326],folds=folds,
        source=str(source.resolve()),observer_source=str(observer_source.resolve()),lstm_epochs=lstm_epochs,sources=SOURCES,
        hyperparameters='Frozen prior family choices; no per-fold or held-out architecture tuning',
        neural='All selected representations plus 24h summaries, actual event type and dated reward metadata; train-only input support masks',
        lstm='Raw LSTM 32/16, same raw inputs and small summaries plus reward metadata; batched query branches, train-only input masks',
        role_transfer='T1500 may borrow recent legacy T1000 completion/pace priors only; no surrogate T1500 supervised labels',
        clocks=[8,20,21],stress='Fixed 6h/24h tracker gaps at an early, middle and late 08:00 checkpoint per validation event',
        ensemble=ENSEMBLES,selection='Three equal chronological fold scores (MAPE + absolute bias by tier/horizon) plus 0.25*fold-score std; paired baseline bias and final-day guards',
        test_policy='Historical replay on previously inspected events, not fresh validation; 325/326 remain excluded from every fit and selection',
        authorization='Reuse earlier protected events under the explicit all-except-325/326 refit instruction; keep the ledger unchanged',
        implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    if output.exists():
        old=json.loads((output/'plan.json').read_text())
        for key in ('config','training_event_ids','folds','lstm_epochs'):
            if old[key]!=plan[key]:raise ValueError('Resume plan differs: '+key)
    else:output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
    base=hyperparameters_only(json.loads((source/'selection.json').read_text()))
    observer=hyperparameters_only(json.loads((observer_source/'selection.json').read_text()))
    if (output/'cache.json').exists():ctx_all=load_context(output)
    else:
        print('Build one causal prequential training bank for all chronological prefixes',flush=True)
        cases,exclusions,roles=build_cases(data,config,ordered,ordered)
        atomic_json(output/'role_transfer_audit.json',roles);atomic_json(output/'case_exclusions.json',exclusions)
        print('Build representations and 24h observer:',len(cases),'cases',flush=True)
        ctx_all=make_context(cases,data,config,ordered);save_context(output,ctx_all)
    all_rows=[];fold_evaluations={};fold_summaries={}
    for fold in folds:
        directory=output/fold['name']
        if (directory/'evaluation.json').exists():
            fold_evaluations[fold['name']]=json.loads((directory/'evaluation.json').read_text())
            continue
        if (directory/'model.json').exists():cores,specs,compression,masks,meta=load_fit(directory)
        else:
            ctx=subset(ctx_all,fold['train'])
            cores,specs,compression,masks,meta=fit_models(ctx,data,config,fold['train'],fold['fit_cutoff'],base,observer,source,directory,lstm_epochs)
            del ctx
        print('Evaluate frozen fold:',fold['name'],flush=True)
        cases,excluded,roles=build_cases(data,config,fold['validation'],fold['train'])
        ctx=make_context(cases,data,config,fold['train']);ctx,_=finish_context(ctx,compression,masks)
        values=predictions(specs,cores,ctx,data,config.tiers)
        evaluation=dict(name=fold['name'],train_events=len(fold['train']),post_rule_training_events=fold['post_rule_training_events'],
            validation_events=fold['validation'],clean={n:summary(cases,p,data) for n,p in values.items()},
            by_type={n:breakdown(cases,p,data,lambda c:data.events[c.event_id].event_type) for n,p in values.items()},stress={},
            coverage=dict(cases=len(cases),events=len({c.event_id for c in cases}),T1500=sum(c.tier==1500 for c in cases)),
            training_targets_by_tier=meta['training_targets_by_tier'])
        scenarios=('off_grid_37m','gap_6h','gap_24h')
        rows=records(ctx,values,data,fold['name'],'clean');stress_cases={n:[] for n in scenarios};stress_values={s:{n:[] for n in values} for s in scenarios}
        for eid in fold['validation']:
            checkpoints=sorted({c.cutoff for c in cases if c.event_id==eid and datetime.fromtimestamp(c.cutoff,CN).hour==8})
            if not checkpoints:continue
            chosen=sorted({checkpoints[min(len(checkpoints)-1,1)],checkpoints[len(checkpoints)//2],checkpoints[-1]})
            for cutoff in chosen:
                for scenario,gap,offset in (('off_grid_37m',0,37*60),('gap_6h',6,0),('gap_24h',24,0)):
                    query=cutoff+offset
                    if query>=data.events[eid].end:continue
                    sparse=stress_dataset(data,eid,query,gap);cfg=replace(config,max_stale_minutes=48*60 if gap else config.max_stale_minutes)
                    predictor=RoleAwareAnalog(sparse,cfg,fold['train']);cc=[]
                    for tier in config.tiers:
                        if (eid,tier) not in sparse.curves:continue
                        try:cc.append(predictor.build(sparse.events[eid],tier,query))
                        except ValueError:pass
                    if not cc:continue
                    sc=make_context(cc,sparse,cfg,fold['train']);sc,_=finish_context(sc,compression,masks)
                    vv=predictions(specs,cores,sc,sparse,config.tiers)
                    rows.extend(records(sc,vv,data,fold['name'],scenario));stress_cases[scenario].extend(cc)
                    for n,p in vv.items():stress_values[scenario][n].extend(p)
        for scenario in scenarios:evaluation['stress'][scenario]={n:summary(stress_cases[scenario],np.array(v),data) for n,v in stress_values[scenario].items()}
        atomic_json(directory/'evaluation.json',evaluation);write_csv(directory/'predictions.csv',list(rows[0]),rows)
        fold_evaluations[fold['name']]=evaluation
        del cores,ctx
    # Selection uses the completed clean cross-validation only. Stress outcomes
    # are diagnostics and never alter architectures, gains or membership.
    names=list(next(iter(fold_evaluations.values()))['clean']);scores={}
    for name in names:
        rr=[f['clean'][name] for f in fold_evaluations.values()]
        raw=np.array([r['selection_score'] for r in rr]);safe=all(admissible(dict(r,stable_score=r['selection_score']),dict(f['clean']['baseline'],stable_score=f['clean']['baseline']['selection_score'])) for r,f in zip(rr,fold_evaluations.values()))
        scores[name]=dict(score=float(raw.mean()+.25*raw.std()),admissible=safe,fold_scores=list(raw))
    champion=min((n for n in names if scores[n]['admissible']),key=lambda n:scores[n]['score'])
    atomic_json(output/'selection.json',dict(champion=champion,scores=scores,rule=plan['selection']))
    print('Robustness selection frozen:',champion,flush=True)
    production=output/'full_refit'
    if not (production/'model.json').exists():
        ctx=subset(ctx_all,ordered)
        fit_models(ctx,data,config,ordered,original['fit_cutoff'],base,observer,source,production,lstm_epochs)
        del ctx
    atomic_json(output/'evaluation.json',dict(champion=champion,folds=fold_evaluations,scores=scores,config=config.to_dict(),
        limitations=['Twelve historical post-rule events; already inspected, not a fresh test',
                     'Cold-start has no post-rule training example; upweighting cannot manufacture one',
                     'Bootstrap/per-type evidence is limited by event count and serial dependence'],
        training_event_ids=ids,excluded=[325,326],sources=SOURCES))
    if digest(registry)!=ledger_before:raise ValueError('Protected ledger changed')
    return json.loads((output/'evaluation.json').read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True)
    parser.add_argument('--source',default='runs/all_except_325_326_20261017');parser.add_argument('--observer',default='runs/observer_24h_20261017')
    parser.add_argument('--output',required=True);parser.add_argument('--lstm-epochs',type=int,default=120)
    args=parser.parse_args();config=Config.load(Path(args.source)/'config.json');data=Dataset.load(args.data,config.tiers)
    result=run(data,args.source,args.observer,args.output,lstm_epochs=args.lstm_epochs)
    from .regime_reporting import publish
    publish(args.output,Path('reports')/(Path(args.output).name+'.md'))
    print('Completed:',result['champion'])


if __name__=='__main__':main()
