"""Sealed pre-321 LOO architecture selection, followed by one 321–324 assessment.

python -m bandori.stack_loo_experiment prepare|tune|test|report --run PATH
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np

from collect_bandori_data import atomic_json, timestamp
from fresh_predictor.data import fingerprint, restrict
from fresh_predictor.runtime import load_history
from .config import Config
from .dataset import Dataset
from .pipeline import digest
from .role_analog import build_cases
from .rules import RULE_START
from .stack_architectures import ARCHITECTURES, StackNetwork
from .stack_loo_data import (FeatureScale, combine, load_cases, prediction, prepare,
                             raw_features, save_cases, train_cases)


SOURCE = Path('runs/rule_aware_3x_20261018')
FOCAL = (500, 1000, 1500, 2000)
TEST_IDS = (321, 322, 323, 324)
SEEDS = (1729, 1730)
SETTINGS = (dict(name='standard', epochs=240, decay=.003),
            dict(name='strong', epochs=240, decay=.01),
            dict(name='short', epochs=120, decay=.003))
DEPENDENCIES = ('bandori/stack_architectures.py', 'bandori/stack_loo_data.py',
    'bandori/stack_loo_experiment.py', 'bandori/analog.py', 'bandori/advanced_statistics.py',
    'bandori/role_analog.py', 'bandori/rules.py', 'bandori/representations.py',
    'bandori/pipeline.py', 'bandori/joint.py', 'bandori/config.py', 'bandori/dataset.py',
    'fresh_predictor/data.py')


def now():return datetime.now(timezone.utc).isoformat()


def candidates():
    return [dict(name=a['name']+'_'+s['name'], architecture=a, setting=s)
            for a in ARCHITECTURES for s in SETTINGS]


def protocol(root, create=False):
    root=Path(root);path=root/'protocol.json'
    if path.exists():
        plan=json.loads(path.read_text())
        assert plan['code']=={p:digest(p) for p in DEPENDENCIES},'Frozen implementation changed'
        assert plan['source_hashes']=={p:digest(p) for p in plan['source_hashes']},'Frozen training bank changed'
        assert plan['ledger_sha256']==digest('evaluation_registry.json'),'Protected ledger changed'
        return plan
    if not create:raise ValueError('Run prepare first')
    bank=SOURCE/'eight_post_events'
    data=load_history(bank)
    meta=json.loads((bank/'model.json').read_text());cutoff=meta['fit_cutoff']
    assert not set(data.events)&set(TEST_IDS+(325,326))
    post=sorted(e for e in data.events if data.events[e].start>=RULE_START)
    assert post==[311,314,315,316,317,318,319,320],post
    older=sorted(set(data.events)-set(post),key=lambda e:(data.events[e].end,e),reverse=True)
    supervised=sorted(post+older[:30-len(post)],key=lambda e:(data.events[e].start,e))
    assert len(supervised)==30
    paths=[str(bank/'reference_bank.json'),str(bank/'reference_bank.npz'),str(SOURCE/'config.json')]
    plan=dict(version=1,created_at_utc=now(),training_bank=str(bank),training_config=str(SOURCE/'config.json'),
        source_hashes={p:digest(p) for p in paths},training_data_fingerprint=fingerprint(data),
        reference_event_ids=list(data.events),supervised_event_ids=supervised,loo_event_ids=post,
        test_event_ids=list(TEST_IDS),excluded_event_ids=[325,326],fit_cutoff=cutoff,
        candidates=candidates(),seeds=list(SEEDS),feature_columns=600,checkpoint_hours=[8,20,21],
        selection='Lowest equal-event focal MAPE of the two-seed arithmetic-mean final forecast across all eight LOO folds',
        tiebreak='Within 1e-12: fewer parameters, stronger decay, fewer epochs, then name',
        final_forecast='Mean of final scores from the two fixed seeds, after each joint rank reconciliation',
        test_policy='Freeze global and per-architecture winners before opening 321–324 labels. Report all ten preselected family winners; never change picks after test.',
        leakage_policy='Physically remove held event curves/labels/metadata from training references. Rebuild all supervised-event priors whose start follows held-label availability. Fit all scaling/masks on fold training rows only.',
        loo_temporal_limit='LOO may train on later pre-321 events; it estimates event transfer, not a chronological live forecast. Final test is strictly chronological.',
        evidence_limit='Architectural ideas informed by prior inspected experiments on 321–324; this is a new selection-isolated historical comparison, not an untouched test.',
        weighting=dict(characteristic_days=730,post_rule_multiplier=3,proxy_weight=.75),
        code={p:digest(p) for p in DEPENDENCIES},ledger_sha256=digest('evaluation_registry.json'))
    root.mkdir(parents=True,exist_ok=True);atomic_json(path,plan);return plan


def config_for(plan):
    cfg=Config.load(plan['training_config'])
    assert cfg.characteristic_time_days==730 and cfg.post_rule_change_multiplier==3
    return cfg


def cache_fold(root,label,prep):
    folder=root/'features'/label
    if folder.exists():raise ValueError('Fold cache already exists')
    folder.mkdir(parents=True)
    save_cases(folder/'train_cases.json.gz',prep.cases)
    values=dict(x=prep.x,targets=prep.targets,weights=prep.weights,bins=prep.bins,index=prep.index,
        explicit_mean=prep.scale.mean,explicit_scale=prep.scale.scale,feature_mask=prep.scale.mask)
    if hasattr(prep,'valid_cases'):
        save_cases(folder/'valid_cases.json.gz',prep.valid_cases)
        values.update(valid_x=prep.valid_x,valid_index=prep.valid_index,valid_bins=prep.valid_bins)
    np.savez_compressed(folder/'arrays.npz',**values)
    paths=sorted(p for p in folder.iterdir() if p.is_file())
    atomic_json(folder/'manifest.json',dict(rebuilt_event_ids=prep.rebuilt_event_ids,
        reference_event_ids=prep.reference_event_ids,train_case_events=sorted({c.event_id for c in prep.cases}),
        hashes={p.name:digest(p) for p in paths}))


def load_fold(root,label):
    folder=root/'features'/label;meta=json.loads((folder/'manifest.json').read_text())
    assert all(digest(folder/p)==h for p,h in meta['hashes'].items())
    with np.load(folder/'arrays.npz',allow_pickle=False) as a:arrays={n:a[n].copy() for n in a.files}
    train=load_cases(folder/'train_cases.json.gz')
    valid=load_cases(folder/'valid_cases.json.gz') if (folder/'valid_cases.json.gz').exists() else None
    return arrays,train,valid,meta


def score(cases,values,data):
    rows=[]
    for c,p in zip(cases,values):
        if c.tier in FOCAL and (c.event_id,c.tier) in data.labels:
            y=data.labels[(c.event_id,c.tier)]
            rows.append(dict(event_id=c.event_id,tier=c.tier,checkpoint=c.cutoff,
                hours_remaining=c.remaining_hours,prediction=float(p),actual=y,error=float(p/y-1)))
    def metric(rr):
        groups={}
        for r in rr:groups.setdefault(r['event_id'],[]).append(r['error'])
        if not groups:return dict(rows=0,events=0,mape=None,bias=None)
        return dict(rows=len(rr),events=len(groups),mape=float(np.mean([np.mean(np.abs(v)) for v in groups.values()])),
            bias=float(np.mean([np.mean(v) for v in groups.values()])))
    return dict(overall=metric(rows),last24h=metric([r for r in rows if r['hours_remaining']<=24]),
        morning=metric([r for r in rows if datetime.fromtimestamp(r['checkpoint'],__import__('collect_bandori_data').CN).hour==8]),
        by_event={str(e):metric([r for r in rows if r['event_id']==e]) for e in sorted({r['event_id'] for r in rows})}),rows


def prepare_phase(root):
    plan=protocol(root,True);data=load_history(plan['training_bank']);cfg=config_for(plan)
    assert fingerprint(data)==plan['training_data_fingerprint']
    base_path=root/'base_cases.json.gz'
    if base_path.exists():base=load_cases(base_path)
    else:
        print('Building causal priors from the pre-321 bank only',flush=True)
        base,failed,_=build_cases(data,cfg,plan['supervised_event_ids'],plan['reference_event_ids'])
        save_cases(base_path,base);atomic_json(root/'base_case_failures.json',failed)
    for held in plan['loo_event_ids']:
        folder=root/'features'/f'loo_{held}'
        if folder.exists():
            load_fold(root,f'loo_{held}');continue
        t=time.monotonic()
        prep=prepare(data,cfg,plan['supervised_event_ids'],plan['reference_event_ids'],base,plan['fit_cutoff'],held)
        assert held not in prep.reference_event_ids
        assert not any(c.event_id==held or held in c.references for c in prep.cases)
        cache_fold(root,f'loo_{held}',prep)
        print(f'LOO {held}: {len(prep.cases)} train cases; rebuilt events {prep.rebuilt_event_ids}; {time.monotonic()-t:.1f}s',flush=True)
    if not (root/'features/full').exists():
        prep=prepare(data,cfg,plan['supervised_event_ids'],plan['reference_event_ids'],base,plan['fit_cutoff'])
        cache_fold(root,'full',prep)
    # Actual poisoning audit for an early held event: later priors must not retain its outcomes.
    held=plan['loo_event_ids'][0]
    before,_,_,_=load_fold(root,f'loo_{held}')
    for key,c in data.curves.items():
        if key[0]==held:c.scores=c.scores*1000
    for key in list(data.labels):
        if key[0]==held:data.labels[key]*=1000
    history,cases,_=train_cases(data,cfg,plan['supervised_event_ids'],plan['reference_event_ids'],base,held)
    raw,index,bins=raw_features(cases,history,cfg)
    targets,weights=__import__('bandori.pipeline',fromlist=['training_arrays']).training_arrays(cases,index,history,cfg,plan['fit_cutoff'])
    scale=FeatureScale().fit(raw,weights,cfg.tiers)
    for a,b in [(before['x'],scale.transform(raw)),(before['targets'],targets),(before['weights'],weights)]:np.testing.assert_array_equal(a,b)
    atomic_json(root/'preparation_audit.json',dict(held_event_curve_and_label_poisoning_invariant=True,
        test_events_absent_from_training_bank=True,fold_count=len(plan['loo_event_ids']),
        base_cases_sha256=digest(base_path),completed_at_utc=now()))
    protocol(root)


def construct(candidate,seed):
    a=dict(candidate['architecture']);a.pop('name')
    return StackNetwork(seed=seed,**a)


def fit(candidate,seed,arrays,cfg):
    setting=candidate['setting']
    cfg=replace(cfg,seed=seed,epochs=setting['epochs'],weight_decay=setting['decay'])
    return construct(candidate,seed).fit(arrays['x'],arrays['targets'],arrays['weights'],arrays['bins'],cfg)


def tune_phase(root):
    plan=protocol(root);cfg=config_for(plan);data=load_history(plan['training_bank'])
    if (root/'selection.json').exists():raise ValueError('Selection already sealed; choose a new run')
    (root/'loo').mkdir(exist_ok=True)
    for held in plan['loo_event_ids']:
        arrays,train,valid,meta=load_fold(root,f'loo_{held}')
        assert held not in meta['train_case_events'] and held not in meta['reference_event_ids']
        for candidate in plan['candidates']:
            path=root/'loo'/f"{candidate['name']}_{held}.json"
            if path.exists():
                cached=json.loads(path.read_text())
                assert cached['protocol_sha256'] in [digest(root/'protocol.json'),*plan.get('accepted_prior_protocol_hashes',[])], 'Cached fit belongs to another protocol'
                continue
            started=time.monotonic();values=[];seed_scores=[];count=None
            for seed in plan['seeds']:
                model=fit(candidate,seed,arrays,cfg)
                count=sum(p.size for p in model.parameters())
                pred=prediction(valid,arrays['valid_index'],model.predict(arrays['valid_x']))
                seed_scores.append(score(valid,pred,data)[0]);values.append(pred)
            ensemble=np.mean(values,axis=0);metrics,_=score(valid,ensemble,data)
            atomic_json(path,dict(candidate=candidate['name'],held=held,params=count,metrics=metrics,
                seed_metrics=seed_scores,seconds=time.monotonic()-started,protocol_sha256=digest(root/'protocol.json')))
        print(f'Completed all 30 candidates for LOO event {held}',flush=True)
    rankings=[]
    for c in plan['candidates']:
        folds=[json.loads((root/'loo'/f"{c['name']}_{h}.json").read_text()) for h in plan['loo_event_ids']]
        rankings.append(dict(candidate=c['name'],architecture=c['architecture']['name'],params=folds[0]['params'],
            loo_mape=float(np.mean([f['metrics']['overall']['mape'] for f in folds])),
            loo_bias=float(np.mean([f['metrics']['overall']['bias'] for f in folds])),
            loo_final24h=float(np.mean([f['metrics']['last24h']['mape'] for f in folds if f['metrics']['last24h']['mape'] is not None])),
            loo_final24h_events=sum(f['metrics']['last24h']['mape'] is not None for f in folds),
            fold_mape={str(f['held']):f['metrics']['overall']['mape'] for f in folds}))
    by_name={c['name']:c for c in plan['candidates']}
    def order(r):
        s=by_name[r['candidate']]['setting']
        return (round(r['loo_mape'],12),r['params'],-s['decay'],s['epochs'],r['candidate'])
    rankings.sort(key=order)
    family={a['name']:min([r for r in rankings if r['architecture']==a['name']],key=order)['candidate'] for a in ARCHITECTURES}
    selection=dict(sealed_at_utc=now(),winner=rankings[0]['candidate'],family_winners=family,rankings=rankings,
        protocol_sha256=digest(root/'protocol.json'),evidence='Selected using pre-321 LOO only; no 321–324 labels or predictions loaded by this phase')
    atomic_json(root/'selection.json',selection)
    print('SEALED LOO WINNER:',selection['winner'],rankings[0]['loo_mape'],flush=True)


def test_phase(root,data_path):
    plan=protocol(root);selection=json.loads((root/'selection.json').read_text())
    if (root/'assessment.json').exists():raise ValueError('Test already assessed; do not retune this run')
    assert selection['protocol_sha256']==digest(root/'protocol.json')
    selection_hash=digest(root/'selection.json')
    # Opening held-out outcomes happens only here, after immutable selection.
    source=Dataset.load(data_path,config_for(plan).tiers);source.add_terminal_labels(config_for(plan))
    target=restrict(source,plan['test_event_ids'],labels=False)
    history=load_history(plan['training_bank']);data=combine(history,target);cfg=config_for(plan)
    arrays,train,_,_=load_fold(root,'full')
    scale=FeatureScale();scale.mean=arrays['explicit_mean'];scale.scale=arrays['explicit_scale'];scale.mask=arrays['feature_mask'];scale.explicit_width=4*len(cfg.tiers)
    cases,failed,_=build_cases(data,cfg,plan['test_event_ids'],plan['reference_event_ids'])
    raw,index,bins=raw_features(cases,data,cfg);x=scale.transform(raw)
    by_name={c['name']:c for c in plan['candidates']}
    finalists=list(dict.fromkeys([selection['winner'],*selection['family_winners'].values(),'mlp8_standard']))
    results={};all_rows=[];(root/'models').mkdir(exist_ok=True)
    for name in finalists:
        values=[];seed_metrics=[]
        for seed in plan['seeds']:
            model=fit(by_name[name],seed,arrays,cfg)
            model.save(root/'models'/f'{name}_seed{seed}.npz')
            pred=prediction(cases,index,model.predict(x));values.append(pred)
            seed_metrics.append(score(cases,pred,source)[0])
            if seed==1729 and name=='mlp8_standard':
                results['legacy_stack_seed1729']=score(cases,pred,source)[0]
        metrics,rows=score(cases,np.mean(values,axis=0),source)
        results[name]=dict(metrics=metrics,seed_metrics=seed_metrics)
        all_rows.extend([dict(model=name,**r) for r in rows])
        print('TEST (settings already frozen):',name,metrics['overall']['mape'],flush=True)
    # The historical Fresh research forecast is a reference only, never a selector input.
    fresh={}
    with Path('runs/fresh_post_rule_v1_20261019/historical_predictions.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['tier']) in FOCAL and r['actual']:
                fresh[(int(r['event_id']),int(r['tier']),timestamp(r['checkpoint']))]=r
    chosen=[r for r in all_rows if r['model']==selection['winner']]
    assert {(r['event_id'],r['tier'],r['checkpoint']) for r in chosen}==set(fresh)
    fresh_rows=[]
    for r in chosen:
        old=fresh[(r['event_id'],r['tier'],r['checkpoint'])]
        assert float(old['actual'])==r['actual']
        fresh_rows.append(dict(r,model='fresh_reference',prediction=float(old['final']),error=float(old['final'])/r['actual']-1))
    all_rows+=fresh_rows
    def simple(rr):
        vals={e:[r['error'] for r in rr if r['event_id']==e] for e in TEST_IDS}
        return dict(mape=float(np.mean([np.mean(np.abs(v)) for v in vals.values()])),
            bias=float(np.mean([np.mean(v) for v in vals.values()])),rows=len(rr))
    differences=[]
    for e in TEST_IDS:
        a=[r for r in chosen if r['event_id']==e];b=[r for r in fresh_rows if r['event_id']==e]
        differences.append(np.mean([abs(r['error']) for r in a])-np.mean([abs(r['error']) for r in b]))
    draws=np.array([np.mean([differences[i] for i in indices]) for indices in itertools.product(range(4),repeat=4)])
    assessment=dict(assessed_at_utc=now(),selected=selection['winner'],selection_sha256=selection_hash,
        results=results,fresh_reference=simple(fresh_rows),
        selected_minus_fresh=dict(mape_change=float(np.mean(differences)),event_bootstrap_95=np.quantile(draws,[.025,.975]).tolist()),
        failures=failed,matched_focal_rows=len(chosen),
        limitations=['Four previously inspected events, not an untouched future test',
            'LOO folds include later pre-321 events; final assessment remains chronological',
            'Closing tracker proxies, not independently verified official final scores',
            'Per-architecture test rows are descriptive; only the presealed global winner is selected'])
    assert selection_hash==digest(root/'selection.json');protocol(root)
    with (root/'test_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    atomic_json(root/'assessment.json',assessment)
    report_phase(root)


def report_phase(root):
    plan=protocol(root);s=json.loads((root/'selection.json').read_text());a=json.loads((root/'assessment.json').read_text())
    ranking={r['candidate']:r for r in s['rankings']}
    lines=['Stack-like architecture study: isolated pre-321 LOO selection','',
        f"Frozen global winner: **{s['winner']}**. Selection was sealed before loading 321–324 outcomes.",
        '', 'LOO holds out actual post-rule events 311, 314–320, one event at a time. The fixed 30-event supervised scope becomes 29 per fold. All held curves, labels and metadata are removed from training references; affected later training priors are rebuilt. Training-only scaling/masks are refit per fold. Older reference history remains available, with 730-day exponential decay and post-rule 3x training weights. LOO can fit on later pre-321 events, so its error is not a live chronological forecast estimate.','',
        'Ten architectures × three regularization/training settings × eight folds × two prespecified seeds. The selection metric scores the mean final forecast of the two seeds, with equal event weight, at T500/T1000/T1500/T2000. All models output the full 24 tiers jointly.','',
        '| Preselected family winner | Parameters | LOO MAPE | Test MAPE, 321–324 | Test bias | Final 24h MAPE |',
        '|---|---:|---:|---:|---:|---:|']
    for name in list(dict.fromkeys([s['winner'],*s['family_winners'].values()])):
        row=ranking[name];m=a['results'][name]['metrics'];tag=' **GLOBAL PICK**' if name==s['winner'] else ''
        lines.append(f"| {name}{tag} | {row['params']} | {100*row['loo_mape']:.2f}% | {100*m['overall']['mape']:.2f}% | {100*m['overall']['bias']:+.2f}% | {100*m['last24h']['mape']:.2f}% |")
    legacy=a['results']['legacy_stack_seed1729'];fresh=a['fresh_reference']
    lines += ['',f"Fixed original-style stack, seed 1729: {100*legacy['overall']['mape']:.2f}% test MAPE. Frozen Fresh research reference: {100*fresh['mape']:.2f}% on the identical {fresh['rows']} rows.",'',
        '| Event | Selected model MAPE |','|---|---:|']
    for e,m in a['results'][s['winner']]['metrics']['by_event'].items():lines.append(f"| {e} | {100*m['mape']:.2f}% |")
    change=a['selected_minus_fresh'];lines += ['',f"Selected minus Fresh: {100*change['mape_change']:+.3f} percentage points; four-event paired bootstrap 95% interval [{100*change['event_bootstrap_95'][0]:+.3f}, {100*change['event_bootstrap_95'][1]:+.3f}] pp. This interval is descriptive and does not erase prior inspection of 321–324.",'',
        'The ten per-family choices and the global winner were chosen using LOO only. No settings or deployed assets were changed after the final test. This study does not establish an untouched future generalization result; the architecture family and representation were informed by earlier experiments.','',
        'Event 318 has no eligible final-24h observations in the LOO bank. LOO closing-day diagnostics therefore cover seven events; the overall selection metric still covers all eight. A diagnostic-only aggregation fix was recorded before selection and before opening test data, preserving all fitted results and the original protocol.','',
        'Training source contains only 315 events before 321. Test 321–324 and excluded 325/326 are absent from all tuning data views. The held-event poisoning audit is in preparation_audit.json; source/code hashes are in protocol.json. Selection and final assessment are separate files; selected weights are saved under models/.','',
        'Reproduction:','', '```sh',
        f'python -m bandori.stack_loo_experiment prepare --run {root}',
        f'python -m bandori.stack_loo_experiment tune --run {root}',
        f'python -m bandori.stack_loo_experiment test --run {root}', '```','']
    (root/'report.md').write_text('\n'.join(lines))
    Path('reports/stack_architectures_loo_20261019.md').write_text('\n'.join(lines))


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','tune','test','report'])
    p.add_argument('--run',default='runs/stack_architectures_loo_20261019_v1');p.add_argument('--data',default='data/full_approaches_20261017')
    args=p.parse_args();root=Path(args.run)
    if args.phase=='prepare':prepare_phase(root)
    elif args.phase=='tune':tune_phase(root)
    elif args.phase=='test':test_phase(root,args.data)
    else:report_phase(root)


if __name__=='__main__':main()
