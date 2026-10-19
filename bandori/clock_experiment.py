"""Paired clock-feature ablation for statistical and small joint models."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from collect_bandori_data import atomic_json, timestamp, write_csv
from .advanced_statistics import HierarchicalSurface, JointRidge
from .analog import reconcile
from .clock_features import COMPACT_INDICES, FEATURES, ClockCompression, arrays
from .config import Config
from .dataset import Dataset
from .evaluation import paired_bootstrap, summary
from .experiments import NEURAL, admissible, correction_forecast
from .joint import JointNet
from .pipeline import attach_bank, digest, save_bank, training_arrays
from .regime_suite import (finish_context as original_finish, load_context, make_context as original_context,
                           raw_stat, records, stress_dataset, subset as original_subset)
from .representations import Compression
from .role_analog import RoleAwareAnalog, build_cases
from .statistical import fit_calibration, predict as statistical_predict


CLASSES={'JointNet':JointNet,'JointRidge':JointRidge,'HierarchicalSurface':HierarchicalSurface}
ENSEMBLES={'neural_mean':list(NEURAL),'statistical_mean':['ridge','calibrated_stat','trajectory','hierarchical'],
           'collection_mean':['baseline','ridge','calibrated_stat','trajectory','hierarchical',*NEURAL]}


def context(cases,data,config,ids):
    ctx=original_context(cases,data,config,ids);ctx['clock']=arrays(ctx,data);return ctx


def subset(ctx,ids):
    result=original_subset(ctx,ids);allowed=set(ids)
    keep=np.array([c.event_id in allowed for c in ctx['cases']]);rows=np.array([any(i>=0 and keep[i] for i in row) for row in ctx['index']])
    result['clock']=dict(features=ctx['clock']['features'][rows].copy(),pace=ctx['clock']['pace'][keep].copy())
    return result


def finish(ctx,compression,clock_compression=None,masks=None,weights=None):
    base_masks={k[5:]:v for k,v in masks.items() if k.startswith('base_')} if masks else None
    ctx,base_masks=original_finish(ctx,compression,base_masks,weights)
    n,k=ctx['index'].shape;full=ctx['clock']['features'].reshape(n,-1)
    if clock_compression is None:clock_compression=ClockCompression().fit(full,weights)
    compact=ctx['clock']['features'][:,:,COMPACT_INDICES].reshape(n,-1)
    compressed=np.column_stack((compact,clock_compression.transform(full)))
    for name,x in ctx['matrices'].items():ctx['matrices'][name]=np.column_stack((x,compressed if name in ('minimal','pca') else full))
    extra=np.zeros((len(ctx['cases']),len(COMPACT_INDICES)))
    for g,t in zip(*np.where(ctx['index']>=0)):extra[ctx['index'][g,t]]=ctx['clock']['features'][g,t,COMPACT_INDICES]
    ctx['surface']=np.column_stack((ctx['surface'],extra))
    if masks is None:
        masks={'base_'+name:v for name,v in base_masks.items()}
        masks.update({'new_'+name:(np.ptp(x[weights.sum(axis=1)>0],axis=0)>1e-8).astype(float) for name,x in ctx['matrices'].items()})
    ctx['matrices']={name:x*masks['new_'+name] for name,x in ctx['matrices'].items()}
    return ctx,clock_compression,masks


def statistical_base(name,ctx,spec):
    b=np.array([c.baseline for c in ctx['cases']]);pace=ctx['clock']['pace']
    if name=='daily_baseline':return pace
    if name=='clock_baseline':return .5*b+.5*pace
    original=raw_stat(name,ctx,spec['base'],spec['observer'])
    return .5*original+.5*pace


def predictions(ctx,specs,cores,tiers):
    result={}
    for name,spec in specs.items():
        kind=spec['kind']
        if kind=='baseline':result[name]=reconcile(ctx['cases'],[c.baseline for c in ctx['cases']])
        elif kind=='nn':
            delta=np.mean([cores[key].predict(ctx['matrices'][spec['representation']]) for key in spec['cores']],axis=0)
            result[name]=correction_forecast(ctx['cases'],ctx['index'],delta,spec['gain'])
        elif kind=='ridge':result[name]=correction_forecast(ctx['cases'],ctx['index'],cores[name].predict(ctx['matrices']['pca']),spec['gain'],'ridge')
        elif kind=='hierarchical':
            delta=cores[name].predict(ctx['surface'],[c.tier for c in ctx['cases']])
            result[name]=reconcile(ctx['cases'],[c.current+max(0,c.baseline-c.current)*(1+spec['gain']*d) for c,d in zip(ctx['cases'],delta)])
        elif kind=='statistical':
            p=statistical_base(name,ctx,spec)
            changed=[replace(c,baseline=float(v),pace=float(v)) for c,v in zip(ctx['cases'],p)]
            result[name]=statistical_predict(changed,0,spec.get('factors',{}))
        else:raise ValueError('Unknown clock model kind')
    for name,members in ENSEMBLES.items():result[name]=reconcile(ctx['cases'],np.mean([result[n] for n in members],axis=0))
    return result


def fit_models(ctx,data,config,ids,cutoff,source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'models').mkdir(exist_ok=True)
    old=json.loads((source/'components.json').read_text());targets,weights=training_arrays(ctx['cases'],ctx['index'],data,config,cutoff)
    compression=Compression().fit(ctx['matrices']['rich'],weights,config.tiers,seed=config.seed)
    ctx,clock_compression,masks=finish(ctx,compression,weights=weights);cores={};specs={'baseline':dict(kind='baseline')}
    for name in NEURAL:
        previous=old[name];keys=[]
        if previous['kind']=='baseline':specs[name]=dict(kind='baseline');continue
        for i,key in enumerate(previous['cores']):
            architecture=JointNet.load(source/'models'/(key+'.npz'));newkey=name+'_'+str(i)
            print('Fit',output.name,newkey,flush=True)
            suffix=key.rsplit('_s',1)[-1]
            seed=int(suffix) if suffix.isdigit() else config.seed
            cores[newkey]=JointNet(architecture.hidden,seed).fit(ctx['matrices'][previous['representation']],targets,weights,ctx['bins'],config);keys.append(newkey)
        specs[name]=dict(previous,cores=keys)
    print('Fit',output.name,'ridge',flush=True)
    cores['ridge']=JointRidge().fit(ctx['matrices']['pca'],targets,weights,ctx['bins'],.01,config.bias_penalty)
    specs['ridge']=dict(kind='ridge',gain=old['ridge']['gain'])
    cw=np.zeros(len(ctx['cases']))
    for g,t in zip(*np.where(ctx['index']>=0)):cw[ctx['index'][g,t]]=weights[g,t]
    print('Fit',output.name,'hierarchical',flush=True)
    cores['hierarchical']=HierarchicalSurface().fit(ctx['surface'],ctx['cases'],data,cw,config.tiers,.01,.1,config.bias_penalty)
    specs['hierarchical']=dict(kind='hierarchical',gain=old['hierarchical']['gain'])
    for name in ('daily_baseline','clock_baseline','calibrated_stat','trajectory'):
        if name in ('calibrated_stat','trajectory'):
            spec=dict(kind='statistical',base=old[name]['base'],observer=old[name]['observer'])
        else:spec=dict(kind='statistical')
        values=statistical_base(name,ctx,spec);changed=[replace(c,baseline=float(v),pace=float(v)) for c,v in zip(ctx['cases'],values)]
        spec['factors']=fit_calibration(changed,data,config,0,cutoff) if name in ('calibrated_stat','trajectory') else {};specs[name]=spec
    compression.save(output/'models/compression.npz');clock_compression.save(output/'models/clock_compression.npz')
    np.savez_compressed(output/'models/input_masks.npz',**masks)
    for name,model in cores.items():model.save(output/'models'/(name+'.npz'))
    atomic_json(output/'components.json',specs);atomic_json(output/'config.json',config.to_dict());save_bank(output,data,ids,config.tiers)
    files=['components.json','config.json','reference_bank.json','reference_bank.npz']+['models/'+p.name for p in (output/'models').iterdir()]
    atomic_json(output/'model.json',dict(kind='clock_features_v2',training_event_ids=ids,fit_cutoff=cutoff,
        feature_names=list(FEATURES),core_kinds={k:type(v).__name__ for k,v in cores.items()},
        training_targets_by_tier={str(t):int(np.count_nonzero(weights[:,j])) for j,t in enumerate(config.tiers)},
        hashes={f:digest(output/f) for f in files}))
    return cores,specs,compression,clock_compression,masks


def load_fit(output):
    output=Path(output);meta=json.loads((output/'model.json').read_text())
    for f,h in meta['hashes'].items():
        if digest(output/f)!=h:raise ValueError('Clock model checksum mismatch: '+f)
    cores={k:CLASSES[v].load(output/'models'/(k+'.npz')) for k,v in meta['core_kinds'].items()}
    with np.load(output/'models/input_masks.npz',allow_pickle=False) as a:masks={k:a[k].copy() for k in a.files}
    return cores,json.loads((output/'components.json').read_text()),Compression.load(output/'models/compression.npz'),ClockCompression.load(output/'models/clock_compression.npz'),masks


def run(data,source,output):
    source,output=Path(source),Path(output);config=Config.load(source/'config.json');data.add_terminal_labels(config)
    original=json.loads((source/'plan.json').read_text());full=json.loads((source/'full_refit/model.json').read_text());ids=full['training_event_ids']
    ledger=digest('evaluation_registry.json')
    if set(ids)&{325,326}:raise ValueError('Holdout in clock training bank')
    plan=dict(kind='clock_feature_ablation_v2',source=str(source.resolve()),source_sha256=digest(source/'model.json'),
        config=config.to_dict(),folds=original['folds'],training_event_ids=ids,clock_features=list(FEATURES),
        representations='Raw 83 clock statistics in standard/rich models; train-only 16 clock PCs plus 12 explicit summaries in minimal/PCA/ridge; 12 explicit summaries in hierarchical surface',
        settings='Existing hyperparameters/gains/seed choices fixed; statistical pace blend fixed at 0.5; 319 training events excluding 325/326',
        selection='Original equal-fold error+bias+0.25fold std with paired overall/row bias and final-24h guards; stress never selects',
        code={name:digest(Path('bandori')/name) for name in ('clock_features.py','clock_experiment.py')})
    output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Clock resume plan differs')
    else:atomic_json(output/'plan.json',plan)
    ctx=load_context(source);cache=output/'clock_cache.npz'
    if cache.exists():
        expected=json.loads((output/'clock_cache.json').read_text())
        if digest(cache)!=expected['sha256']:raise ValueError('Clock cache checksum mismatch')
        with np.load(cache,allow_pickle=False) as a:ctx['clock']={k:a[k].copy() for k in a.files}
    else:
        print('Build clock features for',len(ctx['cases']),'prequential training cases',flush=True)
        ctx['clock']=arrays(ctx,data);np.savez_compressed(cache,**ctx['clock']);atomic_json(output/'clock_cache.json',dict(sha256=digest(cache)))
    evaluations={}
    for fold in plan['folds']:
        directory=output/fold['name']
        if (directory/'evaluation.json').exists():evaluations[fold['name']]=json.loads((directory/'evaluation.json').read_text());continue
        if (directory/'model.json').exists():cores,specs,compression,clock_compression,masks=load_fit(directory)
        else:cores,specs,compression,clock_compression,masks=fit_models(subset(ctx,fold['train']),data,config,fold['train'],fold['fit_cutoff'],source/'full_refit',directory)
        print('Evaluate clock features:',fold['name'],flush=True)
        cases,_,_=build_cases(data,config,fold['validation'],fold['train']);validation=context(cases,data,config,fold['train'])
        validation,_,_=finish(validation,compression,clock_compression,masks);values=predictions(validation,specs,cores,config.tiers)
        evaluation=dict(clean={n:summary(cases,p,data) for n,p in values.items()},stress={})
        rows=records(validation,values,data,fold['name'],'clean')
        with (source/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:old_rows=list(csv.DictReader(f))
        lookup={(r['scenario'],r['event_id'],r['tier'],r['checkpoint']):r for r in old_rows}
        for scenario,gap in (('off_grid_37m',0),('gap_6h',6),('gap_24h',24)):
            cc=[];vv={n:[] for n in values};groups={}
            for r in old_rows:
                if r['scenario']==scenario:groups.setdefault((int(r['event_id']),timestamp(r['checkpoint'])),[]).append(int(r['tier']))
            for (eid,cutoff),tiers in groups.items():
                sparse=stress_dataset(data,eid,cutoff,gap);cfg=replace(config,max_stale_minutes=48*60 if gap else config.max_stale_minutes)
                analog=RoleAwareAnalog(sparse,cfg,fold['train']);current=[]
                for tier in tiers:
                    try:current.append(analog.build(sparse.events[eid],tier,cutoff))
                    except ValueError:pass
                if not current:continue
                current_ctx=context(current,sparse,cfg,fold['train']);current_ctx,_,_=finish(current_ctx,compression,clock_compression,masks)
                pp=predictions(current_ctx,specs,cores,cfg.tiers);rows.extend(records(current_ctx,pp,data,fold['name'],scenario));cc.extend(current)
                for n,p in pp.items():vv[n].extend(p)
            evaluation['stress'][scenario]={n:summary(cc,np.asarray(p),data) for n,p in vv.items()}
        # Store paired old forecasts on precisely the same rows for reporting.
        for row in rows:
            old=lookup[(row['scenario'],str(row['event_id']),str(row['tier']),row['checkpoint'])]
            if abs(float(old['baseline'])-row['baseline'])>1e-6:raise ValueError('Paired baseline changed')
            for name in specs:
                if name in old:row['control_'+name]=float(old[name])
        write_csv(directory/'predictions.csv',list(rows[0]),rows);atomic_json(directory/'evaluation.json',evaluation);evaluations[fold['name']]=evaluation
    scores={}
    for name in next(iter(evaluations.values()))['clean']:
        rr=[f['clean'][name] for f in evaluations.values()];raw=np.array([r['selection_score'] for r in rr])
        safe=all(admissible(dict(r,stable_score=r['selection_score']),dict(f['clean']['baseline'],stable_score=f['clean']['baseline']['selection_score'])) for r,f in zip(rr,evaluations.values()))
        scores[name]=dict(score=float(raw.mean()+.25*raw.std()),admissible=safe)
    champion=min((n for n in scores if scores[n]['admissible']),key=lambda n:scores[n]['score'])
    atomic_json(output/'selection.json',dict(champion=champion,scores=scores));print('Clock selection frozen:',champion,flush=True)
    if not (output/'full_refit/model.json').exists():fit_models(subset(ctx,ids),data,config,ids,full['fit_cutoff'],source/'full_refit',output/'full_refit')
    atomic_json(output/'evaluation.json',dict(champion=champion,folds=evaluations,features=list(FEATURES),
        limitations=['Twelve previously inspected chronological event clusters; not fresh validation','325/326 excluded from every fit and model choice']))
    if digest('evaluation_registry.json')!=ledger:raise ValueError('Protected ledger changed')
    print('Clock-feature fitting complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/full_approaches_20261017')
    p.add_argument('--source',default='runs/rule_aware_3x_20261018');p.add_argument('--output',required=True)
    args=p.parse_args();config=Config.load(Path(args.source)/'config.json');run(Dataset.load(args.data,config.tiers),args.source,args.output)


if __name__=='__main__':main()
