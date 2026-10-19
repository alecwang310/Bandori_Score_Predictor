"""Predeclared representation ablations, statistical models and fixed ensembles."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN,atomic_json,checkpoints,iso,timestamp,write_csv
from .advanced_statistics import CachedAnalog,HierarchicalSurface,JointRidge,surface_features,trajectory_forecasts
from .analog import HORIZONS,horizon,reconcile
from .config import Config
from .dataset import Dataset
from .evaluation import interval_radius,paired_bootstrap,summary
from .joint import JointNet
from .pipeline import attach_bank,digest,save_bank,split_events,training_arrays
from .representations import Compression,FEATURES,MINIMAL,RICH,matrices
from .statistical import fit_calibration,predict as statistical_forecast


FAMILIES=('nn_minimal','nn_standard','nn_rich','nn_pca','bagged','ridge','calibrated_stat','trajectory','hierarchical')
NEURAL=FAMILIES[:5]
STATISTICAL=FAMILIES[5:]


def build_cases(data,config,ids,references):
    predictor=CachedAnalog(data,config,references);cases=[];excluded=[]
    for eid in ids:
        event=data.events[eid]
        for cutoff in checkpoints(event.start,event.end,event.end):
            for tier in config.tiers:
                if (eid,tier) not in data.curves:continue
                try:cases.append(predictor.build(event,tier,cutoff))
                except ValueError as error:excluded.append(dict(event_id=eid,tier=tier,checkpoint=iso(cutoff),reason=str(error)))
    return cases,excluded


def correction_forecast(cases,index,corrections,gain,kind='nn'):
    result=np.array([c.baseline for c in cases])
    for g,t in zip(*np.where(index>=0)):
        i=index[g,t];c=cases[i];delta=gain*corrections[g,t]
        multiplier=math.exp(delta) if kind=='nn' else 1+delta
        result[i]=c.current+max(0,c.baseline-c.current)*multiplier
    return reconcile(cases,result)


def score_tuning(cases,predictions,data,blocks):
    result=summary(cases,predictions,data);scores=[]
    for ids in blocks:
        ii=[i for i,c in enumerate(cases) if c.event_id in ids]
        value=summary([cases[i] for i in ii],np.asarray(predictions)[ii],data)['selection_score']
        if value is not None:scores.append(value)
    result['block_scores']=scores
    result['stable_score']=result['selection_score']+.25*float(np.std(scores)) if scores else None
    return result


def admissible(result,baseline):
    if result['stable_score'] is None:return False
    for key in ('bias','row_bias'):
        if abs(result['overall'][key])>abs(baseline['overall'][key])+.0025:return False
    late=result['by_horizon'].get('0-24h',{});base=baseline['by_horizon'].get('0-24h',{})
    return not base.get('rows') or (late.get('rows',0)>0 and late['mape']<=base['mape']+.0025)


def context(cases,data,config,references,compression=None):
    if not cases:return {'cases':cases,'matrices':{},'index':np.empty((0,len(config.tiers)),int),'shape':{},'surface':np.empty((0,10))}
    arrays,index,bins=matrices(cases,data,config.tiers)
    shape=trajectory_forecasts(cases,data,config,references)
    if compression is not None:arrays['pca']=compression.transform(arrays['rich'])
    return dict(cases=cases,matrices=arrays,index=index,bins=bins,shape=shape,surface=surface_features(cases,shape[.3]))


def component_prediction(spec,ctx,cores,tiers):
    cases=ctx['cases'];kind=spec['kind']
    if not cases:return np.array([])
    if kind=='baseline':return reconcile(cases,[c.baseline for c in cases])
    if kind in ('nn','ridge','bagged'):
        corrections=np.mean([cores[key].predict(ctx['matrices'][spec['representation']]) for key in spec['cores']],axis=0)
        return correction_forecast(cases,ctx['index'],corrections,spec['gain'],'ridge' if kind=='ridge' else 'nn')
    if kind=='statistical':return statistical_forecast(cases,spec['alpha'],spec['factors'])
    if kind=='trajectory':
        changed=[replace(c,pace=float(p)) for c,p in zip(cases,ctx['shape'][spec['bandwidth']])]
        return statistical_forecast(changed,1,spec['factors'])
    if kind=='hierarchical':
        corrections=cores[spec['core']].predict(ctx['surface'],[c.tier for c in cases])
        return reconcile(cases,[c.current+max(0,c.baseline-c.current)*(1+spec['gain']*d) for c,d in zip(cases,corrections)])
    raise ValueError('Unknown model kind: '+kind)


def ensemble_predictions(predictions,members):
    result=dict(predictions)
    for name,spec in members.items():
        matrix=np.asarray([result[m] for m in spec['members']])
        if spec['method']=='median':value=np.median(matrix,axis=0)
        elif spec['method']=='trimmed_mean':value=np.mean(np.sort(matrix,axis=0)[1:-1],axis=0)
        else:value=np.mean(matrix,axis=0)
        result[name]=value
    return result


def run_suite(data,config,output,registry='evaluation_registry.json',replay=None):
    if not config.tiers:config=replace(config,tiers=tuple(sorted({t for _,t in data.curves})))
    data.add_terminal_labels(config);output=Path(output);registry=Path(registry)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new immutable experiment directory')
    protected=json.loads(registry.read_text())['protected_event_ids'] if registry.exists() else []
    if replay is None:
        split=split_events(data,config,protected)
    else:
        previous=Path(replay);metadata=json.loads((previous/'model.json').read_text())
        if digest(previous/'split.json')!=metadata['hashes']['split.json']:raise ValueError('Replay split checksum mismatch')
        split=json.loads((previous/'split.json').read_text())
        if set(split['train'])&set(protected):raise ValueError('Replay training events overlap protected holdouts')
        partitions=[split[name] for name in ('train','tune','calibration','test')]
        if len(set(sum(partitions,[])))!=sum(map(len,partitions)):raise ValueError('Replay partitions overlap')
        if any(e not in data.events for e in sum(partitions,[])):raise ValueError('Replay events absent from data')
        if any(data.label_available(e,config)>split['fit_cutoff'] for e in split['train']):raise ValueError('Replay training labels were not available at fit cutoff')
        boundaries=[max(data.events[e].end for e in ids) for ids in partitions[:-1]]
        starts=[min(data.events[e].start for e in ids) for ids in partitions[1:]]
        if any(a>=b for a,b in zip(boundaries,starts)):raise ValueError('Replay partitions are not chronological')
    protected=sorted(set(protected)|set(split['tune']+split['calibration']+split['test']))
    atomic_json(registry,dict(protected_event_ids=protected,policy='Inspected holdouts never enter fitting or a fresh final test.'))
    output.mkdir(parents=True,exist_ok=True);blocks=[set(map(int,b)) for b in np.array_split(split['tune'],3)]
    atomic_json(output/'plan.json',dict(config=config.to_dict(),split=split,input=data.provenance,
        implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        representations={'minimal':MINIMAL,'standard':FEATURES,'rich':RICH,'pca':'16 train-only components plus explicit progress, remaining time, score fraction and presence'},
        neural={'hidden_sizes':config.hidden_sizes,'gains':[.5,1],'seed_bag':[config.seed+i for i in range(3)]},
        ridge={'representations':['minimal','pca'],'strengths':[.01,.1],'gains':[.5,1]},
        statistical={'completion_pace_weights':[0,.5,1],'trajectory_bandwidths':[.1,.3],
                     'hierarchical_penalties':[[.01,.1],[.1,1.0]],'hierarchical_gains':[.5,1]},
        selection='Mean tier/horizon MAPE + absolute bias, plus 0.25 times standard deviation across three chronological tuning blocks',
        guards='Overall event-balanced and row-mean absolute tuning bias and final-24h tuning MAPE may worsen at most 0.25 percentage points',
        ensemble='Fixed equal family means, median and trimmed mean; no fitted ensemble weights',
        replay_of=str(Path(replay).resolve()) if replay is not None else None,
        test_policy='Follow-up on an already inspected test block; no fresh generalization claim' if replay is not None else 'Lock family winners and all ensemble memberships before generating fresh test cases'))
    atomic_json(output/'config.json',config.to_dict());atomic_json(output/'split.json',split)
    partitions={};excluded={}
    for part in ('train','tune','calibration'):
        print('Build causal cases:',part,flush=True)
        partitions[part],excluded[part]=build_cases(data,config,split[part],split['train'])
    if not partitions['train'] or not partitions['tune']:raise ValueError('No usable training or tuning cases')
    print('Build causal representations and trajectory analogs: train',flush=True)
    train_ctx=context(partitions['train'],data,config,split['train'])
    target,weights=training_arrays(partitions['train'],train_ctx['index'],data,config,split['fit_cutoff'])
    if weights.sum()<=0:raise ValueError('No supervised targets')
    compression=Compression().fit(train_ctx['matrices']['rich'],weights,config.tiers,seed=config.seed)
    train_ctx['matrices']['pca']=compression.transform(train_ctx['matrices']['rich'])
    print('Build causal representations and trajectory analogs: tune',flush=True)
    tune_ctx=context(partitions['tune'],data,config,split['train'],compression)
    baseline_spec={'kind':'baseline'};cores={};specs={'baseline':baseline_spec};results={}
    results['baseline']=score_tuning(partitions['tune'],component_prediction(baseline_spec,tune_ctx,cores,config.tiers),data,blocks)
    if results['baseline']['stable_score'] is None:raise ValueError('No usable tuning labels')
    best={family:'baseline' for family in FAMILIES};candidates=[]

    def add(name,family,spec):
        specs[name]=spec;prediction=component_prediction(spec,tune_ctx,cores,config.tiers)
        result=score_tuning(partitions['tune'],prediction,data,blocks);results[name]=result
        safe=admissible(result,results['baseline'])
        candidates.append(dict(name=name,family=family,score=result['stable_score'],admissible=safe,metrics=result))
        if safe and result['stable_score']<results[best[family]]['stable_score']-1e-12:best[family]=name

    for rep in ('minimal','standard','rich','pca'):
        for hidden in config.hidden_sizes:
            key=f'nn_{rep}_h{hidden}';print('Fit',key,train_ctx['matrices'][rep].shape,flush=True)
            cores[key]=JointNet(hidden,config.seed).fit(train_ctx['matrices'][rep],target,weights,train_ctx['bins'],config)
            for gain in (.5,1.):add(f'{key}_g{gain:g}','nn_'+rep,dict(kind='nn',representation=rep,cores=[key],gain=gain))
    bag_keys=[]
    for seed in (config.seed,config.seed+1,config.seed+2):
        key=f'bag_standard_s{seed}';print('Fit',key,flush=True)
        cores[key]=JointNet(config.hidden_sizes[0],seed).fit(train_ctx['matrices']['standard'],target,weights,train_ctx['bins'],config);bag_keys.append(key)
    for gain in (.5,1.):add(f'bagged_g{gain:g}','bagged',dict(kind='bagged',representation='standard',cores=bag_keys,gain=gain))
    for rep in ('minimal','pca'):
        for strength in (.01,.1):
            key=f'ridge_{rep}_a{strength:g}';print('Fit',key,flush=True)
            cores[key]=JointRidge().fit(train_ctx['matrices'][rep],target,weights,train_ctx['bins'],strength,config.bias_penalty)
            for gain in (.5,1.):add(f'{key}_g{gain:g}','ridge',dict(kind='ridge',representation=rep,cores=[key],gain=gain))
    for alpha in (0.,.5,1.):
        factors=fit_calibration(partitions['train'],data,config,alpha,split['fit_cutoff'])
        add(f'calibrated_stat_a{alpha:g}','calibrated_stat',dict(kind='statistical',alpha=alpha,factors=factors))
    for bandwidth in (.1,.3):
        changed=[replace(c,pace=float(p)) for c,p in zip(partitions['train'],train_ctx['shape'][bandwidth])]
        factors=fit_calibration(changed,data,config,1,split['fit_cutoff'])
        add(f'trajectory_b{bandwidth:g}','trajectory',dict(kind='trajectory',bandwidth=bandwidth,factors=factors))
    case_weights=np.zeros(len(partitions['train']))
    for g,t in zip(*np.where(train_ctx['index']>=0)):case_weights[train_ctx['index'][g,t]]=weights[g,t]
    for global_penalty,tier_penalty in ((.01,.1),(.1,1.)):
        key=f'hierarchical_a{global_penalty:g}';print('Fit',key,flush=True)
        cores[key]=HierarchicalSurface().fit(train_ctx['surface'],partitions['train'],data,case_weights,config.tiers,global_penalty,tier_penalty,config.bias_penalty)
        for gain in (.5,1.):add(f'{key}_g{gain:g}','hierarchical',dict(kind='hierarchical',core=key,gain=gain))
    neural_winner=min(NEURAL,key=lambda family:results[best[family]]['stable_score'])
    selected={'baseline':baseline_spec,**{family:specs[name] for family,name in best.items()}}
    selected['neural_winner']=selected[neural_winner]
    ensembles={'neural_mean':dict(method='mean',members=list(NEURAL)),
               'statistical_mean':dict(method='mean',members=list(STATISTICAL))}
    mixed=['baseline','neural_winner',*STATISTICAL]
    ensembles.update({name:dict(method=method,members=mixed) for name,method in
        (('collection_mean','mean'),('collection_median','median'),('collection_trimmed','trimmed_mean'))})
    selected_tune={name:component_prediction(spec,tune_ctx,cores,config.tiers) for name,spec in selected.items()}
    selected_tune=ensemble_predictions(selected_tune,ensembles)
    selected_scores={name:score_tuning(partitions['tune'],p,data,blocks) for name,p in selected_tune.items()}
    safe=[name for name,r in selected_scores.items() if admissible(r,selected_scores['baseline'])]
    champion=min(safe,key=lambda name:selected_scores[name]['stable_score'])
    atomic_json(output/'selection.json',dict(family_winners=best,components=selected,ensembles=ensembles,champion=champion,
                                            rule='All selections frozen using tuning only; tests cannot change champion'))
    print('Selection locked. Champion:',champion,'family winners:',best,flush=True)
    # First access to test-derived features occurs after every selection above.
    partitions['test'],excluded['test']=build_cases(data,config,split['test'],split['train'])
    (output/'models').mkdir();compression.save(output/'models/compression.npz')
    needed=set()
    for spec in selected.values():
        needed.update(spec.get('cores',[]))
        if 'core' in spec:needed.add(spec['core'])
    core_kinds={}
    for key in sorted(needed):
        core=cores[key];core.save(output/'models'/(key+'.npz'));core_kinds[key]=type(core).__name__
    save_bank(output,data,split['train'],config.tiers)
    evaluation=dict(champion=champion,family_winners=best,candidates=candidates,partitions={},uncertainty={},bootstrap={},
        snapshot=data.provenance,split_event_counts={k:len(v) for k,v in split.items() if isinstance(v,list)},
        training_joint_checkpoints=len(target),supervised_training_targets=int(np.count_nonzero(weights)),
        training_targets_by_tier={str(t):int(np.count_nonzero(weights[:,i])) for i,t in enumerate(config.tiers)},
        representation_widths={name:x.shape[1] for name,x in train_ctx['matrices'].items()},
        ensemble_members=ensembles,test_status='previously_inspected_follow_up' if replay is not None else 'fresh',
        test_policy='Follow-up on an already inspected test block; champion selected on tuning only' if replay is not None else 'All reported test systems preselected; champion never chosen from test')
    audit=[]
    for part in ('tune','calibration','test'):
        print('Evaluate locked systems:',part,flush=True)
        ctx=tune_ctx if part=='tune' else context(partitions[part],data,config,split['train'],compression)
        predictions=ensemble_predictions({n:component_prediction(s,ctx,cores,config.tiers) for n,s in selected.items()},ensembles)
        evaluation['partitions'][part]={name:summary(ctx['cases'],p,data) for name,p in predictions.items()}
        if part=='tune':
            for name in predictions:evaluation['partitions'][part][name].update(stable_score=selected_scores[name]['stable_score'],block_scores=selected_scores[name]['block_scores'])
        if part=='calibration':
            for name,p in predictions.items():
                bands={'pooled':interval_radius(ctx['cases'],p,data,config.interval_alpha),'by_horizon':{}}
                for h,label in enumerate(HORIZONS):
                    ii=[i for i,c in enumerate(ctx['cases']) if horizon(c.remaining_hours)==h]
                    bands['by_horizon'][label]=interval_radius([ctx['cases'][i] for i in ii],p[ii],data,config.interval_alpha)
                evaluation['uncertainty'][name]=bands
        if part=='test':
            evaluation['bootstrap']={name:paired_bootstrap(ctx['cases'],predictions['baseline'],p,data,config.seed) for name,p in predictions.items() if name!='baseline'}
        for i,c in enumerate(ctx['cases']):
            audit.append(dict(partition=part,event_id=c.event_id,tier=c.tier,checkpoint=iso(c.cutoff),
                hours_remaining=c.remaining_hours,current=c.current,actual=data.labels.get((c.event_id,c.tier),''),
                label_source=data.label_sources.get((c.event_id,c.tier),'missing'),
                reference_event_ids=','.join(map(str,c.references)),**{name:float(p[i]) for name,p in predictions.items()}))
    write_csv(output/'predictions.csv',list(audit[0]),audit)
    atomic_json(output/'evaluation.json',evaluation);atomic_json(output/'exclusions.json',excluded)
    files=['config.json','split.json','selection.json','reference_bank.json','reference_bank.npz','models/compression.npz']+['models/'+k+'.npz' for k in needed]
    atomic_json(output/'model.json',dict(version=2,output_tiers=config.tiers,training_event_ids=split['train'],fit_cutoff=split['fit_cutoff'],
        core_kinds=core_kinds,uncertainty=evaluation['uncertainty'],training_targets_by_tier=evaluation['training_targets_by_tier'],
        hashes={f:digest(output/f) for f in files},limitations=['Historical publication times and schedule revisions may be unknown',
            'Closing proxies are not independently verified official finals','Only 08:00 forecasts have been fitted and evaluated',
            'Unseen tiers without labels lack direct validation; ensemble spread alone is not a calibrated interval']))
    return evaluation


def predict_suite(live,model,event_id,cutoff,output=None):
    directory=Path(model);metadata=json.loads((directory/'model.json').read_text())
    for path,expected in metadata['hashes'].items():
        if digest(directory/path)!=expected:raise ValueError('Model checksum mismatch: '+path)
    local=datetime.fromtimestamp(cutoff,CN)
    if (local.hour,local.minute,local.second,local.microsecond)!=(8,0,0,0):raise ValueError('Checkpoint must be exactly 08:00 CN')
    if cutoff<metadata['fit_cutoff'] or event_id in metadata['training_event_ids']:raise ValueError('Checkpoint precedes fitting or reuses a training event')
    config=Config.load(directory/'config.json');data=attach_bank(directory,live)
    if event_id not in data.events:raise ValueError('Event absent from the input bundle')
    predictor=CachedAnalog(data,config,metadata['training_event_ids']);cases=[];failures=[]
    for tier in config.tiers:
        try:cases.append(predictor.build(data.events[event_id],tier,cutoff))
        except ValueError as error:failures.append(dict(tier=tier,status='unavailable',reason=str(error)))
    ctx=context(cases,data,config,metadata['training_event_ids'],Compression.load(directory/'models/compression.npz'))
    classes={'JointNet':JointNet,'JointRidge':JointRidge,'HierarchicalSurface':HierarchicalSurface}
    cores={key:classes[kind].load(directory/'models'/(key+'.npz')) for key,kind in metadata['core_kinds'].items()}
    selection=json.loads((directory/'selection.json').read_text())
    predictions=ensemble_predictions({name:component_prediction(spec,ctx,cores,config.tiers) for name,spec in selection['components'].items()},selection['ensembles'])
    rows=[]
    for i,c in enumerate(cases):
        forecasts={}
        for name,p in predictions.items():
            bands=metadata['uncertainty'][name];label=HORIZONS[horizon(c.remaining_hours)]
            interval=bands['by_horizon'][label]['by_tier'].get(str(c.tier),{});source='tier_and_horizon'
            if interval.get('log_radius') is None:interval=bands['pooled']['by_tier'].get(str(c.tier),{});source='tier_pooled_horizons'
            radius=interval.get('log_radius');value=p[i]
            forecasts[name]=dict(final=math.ceil(value),lower=max(math.ceil(c.current),math.floor(value*math.exp(-radius))) if radius is not None else None,
                upper=math.ceil(value*math.exp(radius)) if radius is not None else None,interval_scope=source if radius is not None else 'insufficient_calibration',
                calibration_events=interval.get('events',0),nominal_coverage=1-config.interval_alpha)
        rows.append(dict(tier=c.tier,status='ok',current=math.ceil(c.current),forecasts=forecasts,
                         supervised_training_targets=metadata['training_targets_by_tier'][str(c.tier)]))
    result=dict(event_id=event_id,checkpoint_cn=local.isoformat(),champion=selection['champion'],model=str(directory.resolve()),
                predictions=sorted(rows+failures,key=lambda r:r['tier']),limitations=metadata['limitations'])
    if output:
        if Path(output).exists():raise ValueError('Fixed prediction already exists')
        atomic_json(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--data',required=True);fit.add_argument('--config',default='experiment_config.json');fit.add_argument('--output',required=True)
    fit.add_argument('--registry',default='evaluation_registry.json');fit.add_argument('--report')
    fit.add_argument('--replay',help='Reuse a frozen split for a clearly labelled follow-up, never a fresh final test')
    pred=commands.add_parser('predict');pred.add_argument('--data',required=True);pred.add_argument('--model',required=True);pred.add_argument('--event',required=True,type=int)
    pred.add_argument('--as-of',required=True);pred.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='train':
        config=Config.load(args.config);data=Dataset.load(args.data,config.tiers);result=run_suite(data,config,args.output,args.registry,args.replay)
        from .suite_reporting import publish
        publish(args.output,args.report or str(Path('reports')/(Path(args.output).name+'.md')))
        print(json.dumps(dict(champion=result['champion'],test={n:v['overall'] for n,v in result['partitions']['test'].items()}),indent=2))
    else:
        config=Config.load(Path(args.model)/'config.json');data=Dataset.load(args.data,config.tiers);cutoff=timestamp(args.as_of)
        if cutoff is None:raise ValueError('Use a timezone-qualified --as-of')
        print(json.dumps(predict_suite(data,args.model,args.event,cutoff,args.output),indent=2))


if __name__=='__main__':main()
