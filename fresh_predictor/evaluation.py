"""One frozen post-rule assessment, never a hyperparameter selection step."""
import csv
import json
from pathlib import Path

import numpy as np

from bandori.evaluation import paired_bootstrap,summary
from bandori.pipeline import digest
from collect_bandori_data import CN,atomic_json,iso,timestamp,write_csv
from .anchors import Anchors,build_cases
from .data import FOCAL,Policy,restrict
from .pipeline import cached_cases,load_source,verify
from .regression import reconcile
from .runtime import Predictor


def policy_of(p):
    values=dict(p['policy']);values['tiers']=tuple(values['tiers']);values['checkpoint_hours']=tuple(values['checkpoint_hours'])
    return Policy(**values)


def assess(root):
    root=Path(root);p=verify(root);policy=policy_of(p);seal=digest(root/'selection.json')
    if (root/'historical_assessment.json').exists():raise ValueError('Assessment already exists; do not iterate the frozen protocol against it')
    data=load_source(p['data_path'],policy);history=restrict(data,p['selection_reference_ids']);target=restrict(data,p['audit_event_ids'],labels=False)
    cases=cached_cases(root,'audit',history,target,p['audit_event_ids'],policy);model=Predictor(root/'research_model')
    if model.meta['selection_sha256']!=seal:raise ValueError('Research model does not match sealed selection')
    predictions=model.model.predict(cases);anchor=reconcile(cases,np.asarray([c.anchor for c in cases]));ii=[i for i,c in enumerate(cases) if c.tier in FOCAL]
    focal=[cases[i] for i in ii];values=predictions[ii];baseline=anchor[ii]
    result=dict(scope=dict(event_ids=p['audit_event_ids'],tiers=list(FOCAL),checkpoints=list(policy.checkpoint_hours)),
        new=summary(focal,values,data),anchor=summary(focal,baseline,data),
        paired_anchor=paired_bootstrap(focal,baseline,values,data,1729),
        selection_sha256=seal,protocol_sha256=digest(root/'protocol.json'),
        limitations=['Previously inspected historical audit, not a fresh test','No change to features, hyperparameters or family may follow this result'])
    # Read old forecasts only after predictions and hyperparameters are frozen.
    oldpath=Path('runs/rule_aware_3x_20261018/eight_post_events/predictions.csv')
    with oldpath.open(encoding='utf-8-sig') as f:
        old={(int(r['event_id']),int(r['tier']),timestamp(r['checkpoint'])):r for r in csv.DictReader(f) if r['scenario']=='clean' and int(r['tier']) in FOCAL}
    matched=[i for i,c in enumerate(focal) if (c.event_id,c.tier,c.cutoff) in old];cc=[focal[i] for i in matched]
    comparisons={'fresh':values[matched],'new_anchor':baseline[matched]}
    for name in ('baseline','ridge','nn_pca'):
        comparisons[name]=np.asarray([float(old[(c.event_id,c.tier,c.cutoff)][name]) for c in cc])
    result['matched']={n:summary(cc,v,data) for n,v in comparisons.items()}
    result['matched_rows']=len(cc);result['old_benchmark_rows']=len(old)
    result['by_event']={str(e):summary([c for c in focal if c.event_id==e],values[[i for i,c in enumerate(focal) if c.event_id==e]],data)['overall'] for e in p['audit_event_ids']}
    morning=[i for i,c in enumerate(focal) if int((c.cutoff/3600+8)%24)==8]
    result['at_08_cn']=summary([focal[i] for i in morning],values[morning],data)
    rows=[dict(event_id=c.event_id,tier=c.tier,checkpoint=iso(c.cutoff),hours_remaining=c.hours_remaining,current=c.current,
        final=float(v),anchor=float(a),actual=data.labels.get((c.event_id,c.tier),''),
        reference_event_ids=','.join(str(e) for e,t in c.references)) for c,v,a in zip(cases,predictions,anchor)]
    write_csv(root/'historical_predictions.csv',list(rows[0]),rows)
    if digest(root/'selection.json')!=seal:raise ValueError('Selection changed during assessment')
    atomic_json(root/'historical_assessment.json',result)
    return result
