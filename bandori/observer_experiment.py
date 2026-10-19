"""Paired 24-hour observer ablation on a frozen chronological split."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import numpy as np

from collect_bandori_data import CN, atomic_json, iso, timestamp, write_csv
from .advanced_statistics import HierarchicalSurface, JointRidge
from .analog import HORIZONS, Case, horizon, reconcile
from .config import Config
from .daily_observer import CHANNELS, SUMMARIES, DailyObserver, observer_arrays
from .dataset import Dataset
from .evaluation import paired_bootstrap, summary
from .experiments import (FAMILIES, NEURAL, admissible, build_cases, component_prediction,
                          context, correction_forecast, ensemble_predictions, score_tuning)
from .joint import JointNet
from .pipeline import attach_bank, digest, save_bank, training_arrays
from .representations import Compression
from .statistical import fit_calibration, predict as statistical_predict
from .temporal import JointTemporalNet


CLASSES={'JointNet':JointNet,'JointRidge':JointRidge,'HierarchicalSurface':HierarchicalSurface,
         'JointTemporalNet':JointTemporalNet}


def augment(ctx,data,config,references):
    daily=observer_arrays(ctx['cases'],ctx['index'],data,config,references)
    n,k=ctx['index'].shape
    ctx['daily']=daily
    ctx['augmented']={name:np.column_stack((x,daily['summaries'].reshape(n,k*len(SUMMARIES))))
                      for name,x in ctx['matrices'].items()}
    rows=np.zeros((len(ctx['cases']),len(SUMMARIES)))
    for g,t in zip(*np.where(ctx['index']>=0)):rows[ctx['index'][g,t]]=daily['summaries'][g,t]
    ctx['daily_surface']=np.column_stack((ctx['surface'],rows))
    return ctx


def raw_stat(family,ctx,controls):
    cases=ctx['cases']
    if family=='baseline':return np.array([c.baseline for c in cases])
    if family=='calibrated_stat':
        alpha=controls[family]['alpha']
        return np.array([(1-alpha)*c.baseline+alpha*c.pace for c in cases])
    if family=='trajectory':return ctx['shape'][controls[family]['bandwidth']]
    raise ValueError('Unexpected statistical family')


def daily_prediction(spec,ctx,cores,control_predictions,controls,tiers):
    cases=ctx['cases'];kind=spec['kind']
    if not cases:return np.array([])
    if kind=='control':return control_predictions[spec['family']]
    if kind in ('daily_nn','daily_ridge'):
        corrections=np.mean([cores[key].predict(ctx['augmented'][spec['representation']]) for key in spec['cores']],axis=0)
        return correction_forecast(cases,ctx['index'],corrections,spec['gain'],'ridge' if kind=='daily_ridge' else 'nn')
    if kind=='temporal':
        corrections=cores[spec['core']].predict(ctx['augmented']['standard'],ctx['daily']['sequence'])
        return correction_forecast(cases,ctx['index'],corrections,spec['gain'])
    if kind=='daily_hierarchical':
        corrections=cores[spec['core']].predict(ctx['daily_surface'],[c.tier for c in cases])
        return reconcile(cases,[c.current+max(0,c.baseline-c.current)*(1+spec['gain']*d) for c,d in zip(cases,corrections)])
    if kind=='daily_stat':
        pace=ctx['flat_pace'] if spec.get('flat_clock') else ctx['daily']['pace']
        raw=(1-spec['blend'])*raw_stat(spec['family'],ctx,controls)+spec['blend']*pace
        changed=[replace(c,baseline=float(p),pace=float(p)) for c,p in zip(cases,raw)]
        return statistical_predict(changed,0,spec.get('factors',{}))
    raise ValueError('Unknown observer model kind: '+kind)


def controls_at(selection,ctx,cores,tiers):
    return ensemble_predictions({n:component_prediction(s,ctx,cores,tiers)
                                 for n,s in selection['components'].items()},selection['ensembles'])


def flat_pace(ctx,data,config,references):
    observer=DailyObserver(data,config,references);values=[]
    for case in ctx['cases']:
        event=data.known_event(data.events[case.event_id],case.cutoff)[0]
        curve=data.curves[(case.event_id,case.tier)].prefix(case.cutoff,event.start)
        _,rate,valid,active,_=observer.window(case,event,curve)
        total=float(np.sum(rate[valid]*active[valid]));exposure=float(active[valid].sum())
        values.append(observer.pace(case,event,np.ones((2,24)),total,exposure,float(valid.mean())))
    return np.asarray(values)


def validate_source(source,data,registry):
    source=Path(source);metadata=json.loads((source/'model.json').read_text())
    for path,expected in metadata['hashes'].items():
        if digest(source/path)!=expected:raise ValueError('Source checksum mismatch: '+path)
    config=Config.load(source/'config.json');split=json.loads((source/'split.json').read_text())
    protected=set(json.loads(Path(registry).read_text())['protected_event_ids']) if Path(registry).exists() else set()
    groups=[split[k] for k in ('train','tune','calibration','test')]
    if set(split['train'])&protected:raise ValueError('Source training overlaps protected holdouts')
    if len(set(sum(groups,[])))!=sum(map(len,groups)):raise ValueError('Overlapping partitions')
    if any(data.label_available(e,config)>split['fit_cutoff'] for e in split['train']):raise ValueError('Training label unavailable')
    if any(max(data.events[e].end for e in a)>=min(data.events[e].start for e in b) for a,b in zip(groups,groups[1:])):
        raise ValueError('Split is not chronological')
    selection=json.loads((source/'selection.json').read_text())
    return config,split,selection,metadata


def train(data,source,output,registry='evaluation_registry.json',temporal_channels=(4,8)):
    source=Path(source);output=Path(output)
    config,split,control_selection,metadata=validate_source(source,data,registry)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new immutable experiment directory')
    data.add_terminal_labels(config)
    output.mkdir(parents=True);(output/'models').mkdir()
    ledger_before=digest(registry) if Path(registry).exists() else None
    blocks=[set(map(int,b)) for b in np.array_split(split['tune'],3)]
    atomic_json(output/'plan.json',dict(source=str(source.resolve()),split=split,config=config.to_dict(),
        implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        observer=dict(window_hours=24,previous_day_hours=24,max_interpolation_gap_hours=2,
                      summaries=SUMMARIES,channels=CHANNELS,calendar_prior_events=5,
                      historical_opening_exclusion_hours=12,historical_closing_exclusion_hours=24),
        paired_design='Frozen control fits and family hyperparameters; observer refits use the identical training cases and weights',
        fixed_raw_ablation='Original frozen NN/ridge/hierarchical gains; fixed 0.5 statistical blends; first temporal width at gain 0.5, regardless of tuning guards',
        neural_gains=[0,.5,1],statistical_blends=[0,.25,.5,1],temporal_channels=list(temporal_channels),
        temporal=dict(hidden=8,kernel=3,dilations=[1,3],pooling=['24h mean','first 6h mean','last 6h mean'],shared_across_tiers=True),
        selection='Existing three-block stable score and bias/late-error guards against each paired control; zero gain retains that control',
        test_policy='Previously inspected chronological test; follow-up evidence, not fresh validation',
        calibration_policy='Calibration outcomes are reported only; they do not fit weights or select models',
        primary_comparison='Per-family observer minus paired frozen control on exactly the same cases',
        ablation='Flat-clock 24h pace uses identical data and matching but a constant activity profile'))
    atomic_json(output/'config.json',config.to_dict());atomic_json(output/'split.json',split)
    atomic_json(output/'control_selection.json',control_selection)
    compression=Compression.load(source/'models/compression.npz')
    control_cores={key:CLASSES[kind].load(source/'models'/(key+'.npz')) for key,kind in metadata['core_kinds'].items()}
    contexts={};excluded={}
    for partition in ('train','tune'):
        print('Build causal observer:',partition,flush=True)
        cases,excluded[partition]=build_cases(data,config,split[partition],split['train'])
        ctx=context(cases,data,config,split['train'],compression)
        contexts[partition]=augment(ctx,data,config,split['train'])
        contexts[partition]['flat_pace']=flat_pace(ctx,data,config,split['train'])
        print(partition,'cases',len(cases),'mean hourly coverage',float(ctx['daily']['coverage'].mean()),flush=True)
    train_ctx,tune_ctx=contexts['train'],contexts['tune']
    target,weights=training_arrays(train_ctx['cases'],train_ctx['index'],data,config,split['fit_cutoff'])
    train_controls=controls_at(control_selection,train_ctx,control_cores,config.tiers)
    tune_controls=controls_at(control_selection,tune_ctx,control_cores,config.tiers)
    families=('baseline',*FAMILIES,'tcn24','pace24_flat')
    parents={family:family for family in ('baseline',*FAMILIES)}
    parents.update(tcn24='neural_winner',pace24_flat='baseline')
    selected={name:dict(kind='control',family=parents[name]) for name in families}
    paired_raw={name:dict(spec) for name,spec in selected.items()}
    scores={name:score_tuning(tune_ctx['cases'],tune_controls[parents[name]],data,blocks) for name in families}
    candidates=[];cores={}

    def add(family,name,spec):
        prediction=daily_prediction(spec,tune_ctx,cores,tune_controls,control_selection['components'],config.tiers)
        result=score_tuning(tune_ctx['cases'],prediction,data,blocks)
        base=score_tuning(tune_ctx['cases'],tune_controls[parents[family]],data,blocks)
        safe=admissible(result,base)
        candidates.append(dict(family=family,name=name,spec=spec,admissible=safe,metrics=result))
        if safe and result['stable_score']<scores[family]['stable_score']-1e-12:
            selected[family]=dict(spec,candidate_name=name);scores[family]=result

    for family in NEURAL:
        old=control_selection['components'][family]
        if old['kind']=='baseline':continue
        representation=old['representation'];keys=[]
        for old_key in old['cores']:
            old_core=control_cores[old_key];seed=config.seed
            if old_key.startswith('bag_standard_s'):seed=int(old_key.split('_s')[-1])
            key='daily_'+old_key;print('Fit',key,flush=True)
            cores[key]=JointNet(old_core.hidden,seed).fit(train_ctx['augmented'][representation],target,weights,train_ctx['bins'],config)
            keys.append(key)
        for gain in (.5,1.):add(family,f'{family}_daily_g{gain:g}',dict(kind='daily_nn',representation=representation,cores=keys,gain=gain))
        paired_raw[family]=dict(kind='daily_nn',representation=representation,cores=keys,gain=old['gain'])
    old=control_selection['components']['ridge']
    if old['kind']!='baseline':
        representation=old['representation'];old_key=old['cores'][0];key='daily_'+old_key
        strength=float(old_key.rsplit('_a',1)[1]);print('Fit',key,flush=True)
        cores[key]=JointRidge().fit(train_ctx['augmented'][representation],target,weights,train_ctx['bins'],strength,config.bias_penalty)
        for gain in (.5,1.):add('ridge',f'ridge_daily_g{gain:g}',dict(kind='daily_ridge',representation=representation,cores=[key],gain=gain))
        paired_raw['ridge']=dict(kind='daily_ridge',representation=representation,cores=[key],gain=old['gain'])
    case_weights=np.zeros(len(train_ctx['cases']))
    for g,t in zip(*np.where(train_ctx['index']>=0)):case_weights[train_ctx['index'][g,t]]=weights[g,t]
    old=control_selection['components']['hierarchical']
    if old['kind']!='baseline':
        key='daily_'+old['core'];strength=float(old['core'].rsplit('_a',1)[1]);print('Fit',key,flush=True)
        cores[key]=HierarchicalSurface().fit(train_ctx['daily_surface'],train_ctx['cases'],data,case_weights,config.tiers,strength,10*strength,config.bias_penalty)
        for gain in (.5,1.):add('hierarchical',f'hierarchical_daily_g{gain:g}',dict(kind='daily_hierarchical',core=key,gain=gain))
        paired_raw['hierarchical']=dict(kind='daily_hierarchical',core=key,gain=old['gain'])
    for family in ('baseline','calibrated_stat','trajectory','pace24_flat'):
        parent=parents[family]
        if parent!='baseline' and control_selection['components'][parent]['kind']=='baseline':continue
        raw=raw_stat(parent,train_ctx,control_selection['components'])
        pace=train_ctx['flat_pace'] if family=='pace24_flat' else train_ctx['daily']['pace']
        for blend in (.25,.5,1.):
            changed=[replace(c,baseline=float(p),pace=float(p)) for c,p in zip(train_ctx['cases'],(1-blend)*raw+blend*pace)]
            factors=fit_calibration(changed,data,config,0,split['fit_cutoff']) if parent!='baseline' else {}
            add(family,f'{family}_daily_b{blend:g}',dict(kind='daily_stat',family=parent,blend=blend,factors=factors,flat_clock=family=='pace24_flat'))
            if blend==.5:paired_raw[family]=dict(kind='daily_stat',family=parent,blend=blend,factors=factors,flat_clock=family=='pace24_flat')
    for channels in temporal_channels:
        key=f'tcn24_c{channels}';print('Fit',key,flush=True)
        cores[key]=JointTemporalNet(channels,8,config.seed).fit(train_ctx['augmented']['standard'],train_ctx['daily']['sequence'],target,weights,train_ctx['bins'],config)
        for gain in (.5,1.):add('tcn24',f'{key}_g{gain:g}',dict(kind='temporal',core=key,gain=gain))
        if channels==temporal_channels[0]:paired_raw['tcn24']=dict(kind='temporal',core=key,gain=.5)
    neural_winner=min((*NEURAL,'tcn24'),key=lambda name:scores[name]['stable_score'])
    selected['neural_winner']=selected[neural_winner]
    ensembles=control_selection['ensembles']
    tune_predictions=ensemble_predictions({name:daily_prediction(s,tune_ctx,cores,tune_controls,control_selection['components'],config.tiers)
                                          for name,s in selected.items()},ensembles)
    ensemble_scores={n:score_tuning(tune_ctx['cases'],p,data,blocks) for n,p in tune_predictions.items()}
    parent_champion=control_selection['champion'];champion='control_champion'
    best_score=score_tuning(tune_ctx['cases'],tune_controls[parent_champion],data,blocks)
    champion_guard=best_score
    for name,result in ensemble_scores.items():
        if admissible(result,champion_guard) and result['stable_score']<best_score['stable_score']-1e-12:
            champion=name;best_score=result
    selection=dict(components=selected,paired_raw=paired_raw,ensembles=ensembles,champion=champion,control_champion=parent_champion,
                   neural_winner=neural_winner,rule='Frozen before calibration and test observer features are generated')
    atomic_json(output/'selection.json',selection)
    atomic_json(output/'tuning_candidates.json',candidates)
    print('Selection locked:',champion,'neural:',neural_winner,flush=True)
    # All choices above are immutable from this point; neither later partition can change them.
    evaluation=dict(test_status='previously_inspected_follow_up',champion=champion,control_champion=parent_champion,
        neural_winner=neural_winner,split_event_counts={k:len(split[k]) for k in ('train','tune','calibration','test')},
        training_joint_checkpoints=len(target),supervised_training_targets=int(np.count_nonzero(weights)),
        training_targets_by_tier={str(t):int(np.count_nonzero(weights[:,i])) for i,t in enumerate(config.tiers)},
        partitions={},paired_bootstrap={},coverage={},snapshot=data.provenance)
    audit=[]
    for partition in ('tune','calibration','test'):
        print('Evaluate paired models:',partition,flush=True)
        if partition=='tune':ctx=tune_ctx;control_predictions=tune_controls
        else:
            cases,excluded[partition]=build_cases(data,config,split[partition],split['train'])
            ctx=augment(context(cases,data,config,split['train'],compression),data,config,split['train'])
            ctx['flat_pace']=flat_pace(ctx,data,config,split['train'])
            control_predictions=controls_at(control_selection,ctx,control_cores,config.tiers)
        predictions=ensemble_predictions({name:daily_prediction(s,ctx,cores,control_predictions,control_selection['components'],config.tiers)
                                         for name,s in selected.items()},ensembles)
        raw_predictions={name:daily_prediction(s,ctx,cores,control_predictions,control_selection['components'],config.tiers) for name,s in paired_raw.items()}
        predictions['control_champion']=control_predictions[parent_champion]
        evaluation['partitions'][partition]=dict(
            control={n:summary(ctx['cases'],p,data) for n,p in control_predictions.items()},
            observer_raw={n:summary(ctx['cases'],p,data) for n,p in raw_predictions.items()},
            observer={n:summary(ctx['cases'],p,data) for n,p in predictions.items()})
        coverage=ctx['daily']['coverage']
        evaluation['coverage'][partition]=dict(cases=len(coverage),mean=float(coverage.mean()) if len(coverage) else None,
            full_day_fraction=float(np.mean(coverage==1)),at_least_8h_fraction=float(np.mean(coverage>=1/3)),
            events=len({c.event_id for c in ctx['cases']}))
        if partition=='test':
            evaluation['paired_bootstrap']={name:paired_bootstrap(ctx['cases'],control_predictions[name] if name in control_predictions else control_predictions[parents[name]],p,data,config.seed)
                for name,p in predictions.items() if name!='control_champion'}
            evaluation['raw_paired_bootstrap']={name:paired_bootstrap(ctx['cases'],control_predictions[parents[name]],p,data,config.seed) for name,p in raw_predictions.items()}
        for i,c in enumerate(ctx['cases']):
            audit.append(dict(partition=partition,event_id=c.event_id,tier=c.tier,checkpoint=iso(c.cutoff),hours_remaining=c.remaining_hours,
                current=c.current,actual=data.labels.get((c.event_id,c.tier),''),label_source=data.label_sources.get((c.event_id,c.tier),'missing'),
                hourly_coverage=coverage[i],**{'control_'+n:float(p[i]) for n,p in control_predictions.items()},
                **{'raw_'+n:float(p[i]) for n,p in raw_predictions.items()},
                **{'observer_'+n:float(p[i]) for n,p in predictions.items()}))
    write_csv(output/'predictions.csv',list(audit[0]),audit)
    needed=set()
    for spec in [*selected.values(),*paired_raw.values()]:
        needed.update(spec.get('cores',[]))
        if 'core' in spec:needed.add(spec['core'])
    kinds={}
    for key in sorted(needed):cores[key].save(output/'models'/(key+'.npz'));kinds[key]=type(cores[key]).__name__
    for key in control_cores:shutil.copy2(source/'models'/(key+'.npz'),output/'models'/('control_'+key+'.npz'))
    compression.save(output/'models/compression.npz');save_bank(output,data,split['train'],config.tiers)
    atomic_json(output/'evaluation.json',evaluation);atomic_json(output/'exclusions.json',excluded)
    files=['config.json','split.json','selection.json','control_selection.json','reference_bank.json','reference_bank.npz']
    files+=['models/'+p.name for p in (output/'models').glob('*.npz')]
    model=dict(version=1,kind='daily_observer',fit_cutoff=split['fit_cutoff'],training_event_ids=split['train'],
        core_kinds=kinds,control_core_kinds=metadata['core_kinds'],training_targets_by_tier=evaluation['training_targets_by_tier'],
        hashes={f:digest(output/f) for f in files},limitations=[
            'Previously inspected test block: descriptive follow-up, not a fresh test',
            'No new calibrated probability intervals; final labels include closing-tracker proxies',
            'Historical publication and schedule revision timing may be incomplete',
            'Only 08:00 CN has been evaluated; unsupported tiers cannot gain validated heads'])
    atomic_json(output/'model.json',model)
    if ledger_before is not None and digest(registry)!=ledger_before:raise ValueError('Holdout ledger changed unexpectedly')
    return evaluation


def predict(live,model,event_id,cutoff,output=None):
    model=Path(model);metadata=json.loads((model/'model.json').read_text())
    for path,expected in metadata['hashes'].items():
        if digest(model/path)!=expected:raise ValueError('Model checksum mismatch: '+path)
    if event_id in metadata['training_event_ids'] or cutoff<metadata['fit_cutoff']:raise ValueError('Training event or pre-fit forecast')
    local=datetime.fromtimestamp(cutoff,CN)
    if (local.hour,local.minute,local.second,local.microsecond)!=(8,0,0,0):raise ValueError('Checkpoint must be exactly 08:00 CN')
    config=Config.load(model/'config.json');data=attach_bank(model,live)
    from .advanced_statistics import CachedAnalog
    predictor=CachedAnalog(data,config,metadata['training_event_ids']);cases=[];failures=[]
    for tier in config.tiers:
        try:cases.append(predictor.build(data.events[event_id],tier,cutoff))
        except ValueError as error:failures.append(dict(tier=tier,status='unavailable',reason=str(error)))
    if not cases:raise ValueError('No available tiers')
    ctx=augment(context(cases,data,config,metadata['training_event_ids'],Compression.load(model/'models/compression.npz')),data,config,metadata['training_event_ids'])
    ctx['flat_pace']=flat_pace(ctx,data,config,metadata['training_event_ids'])
    control_selection=json.loads((model/'control_selection.json').read_text());selection=json.loads((model/'selection.json').read_text())
    control_cores={key:CLASSES[kind].load(model/'models'/('control_'+key+'.npz')) for key,kind in metadata['control_core_kinds'].items()}
    cores={key:CLASSES[kind].load(model/'models'/(key+'.npz')) for key,kind in metadata['core_kinds'].items()}
    control=controls_at(control_selection,ctx,control_cores,config.tiers)
    predictions=ensemble_predictions({n:daily_prediction(s,ctx,cores,control,control_selection['components'],config.tiers) for n,s in selection['components'].items()},selection['ensembles'])
    predictions['control_champion']=control[selection['control_champion']]
    rows=[dict(tier=c.tier,status='ok',current=math.ceil(c.current),hourly_coverage=float(ctx['daily']['coverage'][i]),
               final=math.ceil(predictions[selection['champion']][i]),
               supervised_training_targets=metadata['training_targets_by_tier'][str(c.tier)]) for i,c in enumerate(cases)]
    result=dict(event_id=event_id,checkpoint_cn=local.isoformat(),champion=selection['champion'],
                predictions=sorted(rows+failures,key=lambda r:r['tier']),limitations=metadata['limitations'])
    if output:
        if Path(output).exists():raise ValueError('Fixed prediction already exists')
        atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--data',required=True);fit.add_argument('--source-model',required=True)
    fit.add_argument('--output',required=True);fit.add_argument('--registry',default='evaluation_registry.json');fit.add_argument('--report')
    pred=commands.add_parser('predict');pred.add_argument('--data',required=True);pred.add_argument('--model',required=True)
    pred.add_argument('--event',type=int,required=True);pred.add_argument('--as-of',required=True);pred.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='train':
        config=Config.load(Path(args.source_model)/'config.json');data=Dataset.load(args.data,config.tiers)
        evaluation=train(data,args.source_model,args.output,args.registry)
        from .observer_reporting import publish
        publish(args.output,args.report or str(Path('reports')/(Path(args.output).name+'.md')))
        print(json.dumps(dict(champion=evaluation['champion'],neural_winner=evaluation['neural_winner']),indent=2))
    else:
        config=Config.load(Path(args.model)/'config.json');data=Dataset.load(args.data,config.tiers);cutoff=timestamp(args.as_of)
        if cutoff is None:raise ValueError('Timezone-qualified --as-of required')
        print(json.dumps(predict(data,args.model,args.event,cutoff,args.output),indent=2))


if __name__=='__main__':main()
