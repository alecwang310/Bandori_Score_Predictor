"""Training phases with a sealed historical audit block and no LOO shortcuts."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np

from bandori.dataset import Dataset
from bandori.evaluation import summary
from bandori.pipeline import digest,save_bank
from collect_bandori_data import atomic_bytes,atomic_json,iso,write_csv
from .anchors import FEATURES,Point,build_cases
from .data import FOCAL,Policy,fingerprint,restrict
from .regression import JointCorrection,reconcile


STRENGTHS=('anchor',.001,.01,.1,1.)
AUDIT_IDS=(321,322,323,324)
EXCLUDED=(325,326)


def frozen_codes():
    paths=[Path(__file__).parent/name for name in ('data.py','anchors.py','regression.py','pipeline.py')]
    return {str(p):digest(p) for p in paths}


def load_source(path,policy):
    data=Dataset.load(path,policy.tiers);data.add_terminal_labels(policy.label_config())
    return data


def permitted_ids(data,policy):
    cutoff=min(data.events[e].start for e in EXCLUDED)
    ids=[e.event_id for e in data.eligible_events(policy.label_config())
         if e.event_id not in EXCLUDED and data.label_available(e.event_id,policy.label_config())<=cutoff]
    return ids,cutoff


def plan_for(data,data_path,policy):
    ids,cutoff=permitted_ids(data,policy);audit_start=min(data.events[e].start for e in AUDIT_IDS)
    selection_ids=[e for e in ids if e not in AUDIT_IDS and data.label_available(e,policy.label_config())<=audit_start]
    post=[e for e in selection_ids if data.events[e].start>=policy.rule_at]
    if set(post)&set(AUDIT_IDS+EXCLUDED):raise ValueError('Audit/future event in selection')
    inner=[]
    for i,eid in enumerate(post):
        if i<4:continue
        at=data.events[eid].start;refs=[e for e in selection_ids if data.label_available(e,policy.label_config())<=at]
        inner.append(dict(validation=eid,fit_cutoff=at,reference_ids=refs,supervised_ids=[e for e in refs if data.events[e].start>=policy.rule_at]))
    if len(inner)<2:raise ValueError('Need at least two chronological inner validation events')
    return dict(version=1,kind='fresh_post_rule_joint_remaining_score',policy=policy.to_dict(),data_path=str(Path(data_path).resolve()),
        strengths=list(STRENGTHS),selection_reference_ids=selection_ids,selection_supervised_ids=post,inner_folds=inner,
        audit_event_ids=list(AUDIT_IDS),audit_fit_cutoff=audit_start,excluded_event_ids=list(EXCLUDED),
        production_reference_ids=ids,production_supervised_ids=[e for e in ids if data.events[e].start>=policy.rule_at],production_fit_cutoff=cutoff,
        selection_fingerprint=fingerprint(restrict(data,selection_ids)),code=frozen_codes(),
        feature_names=list(FEATURES),coefficient_count=len(FEATURES)+4*len(policy.tiers),
        selection_metric='Mean of focal-tier event-balanced MAPE across chronological inner validation events; ties prefer stronger shrinkage',
        objective='Relative MSE plus 3x squared bias by tier/horizon; global/type/tier shrinkage; only post-rule target labels',
        archive='experiment_archive/20261019_frozen_v2/manifest.json',
        evidence_policy='Design informed by prior experiments; audit 321-324 already inspected and is not a fresh test. No audit result may alter hyperparameters.',
        production_authorization='Existing user authorization to refit all eligible earlier events except 325/326; hyperparameters stay sealed before audit reuse',
        protected_ledger_sha256=digest('evaluation_registry.json'))


def verify(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    if p['code']!=frozen_codes():raise ValueError('Frozen predictor implementation changed')
    if digest('evaluation_registry.json')!=p['protected_ledger_sha256']:raise ValueError('Protected ledger changed')
    return p


def cached_cases(root,label,history,live,ids,policy):
    root=Path(root);path=root/(label+'_cases.json.gz');receipt=root/(label+'_cases_manifest.json')
    expected=dict(history_fingerprint=fingerprint(history),target_fingerprint=fingerprint(restrict(live,ids,labels=False)),
        ids=list(ids),policy=policy.to_dict(),code={str(Path(__file__).parent/n):digest(Path(__file__).parent/n) for n in ('data.py','anchors.py')})
    if path.exists():
        old=json.loads(receipt.read_text())
        if old['inputs']!=expected or digest(path)!=old['sha256']:raise ValueError('New feature cache inputs differ')
        rows=json.loads(gzip.decompress(path.read_bytes()))
        return [Point(**dict(r,features=np.asarray(r['features']),references=tuple(map(tuple,r['references'])))) for r in rows]
    points=build_cases(history,live,ids,policy,progress=lambda eid,n:print(label,'event',eid,'points',n,flush=True))
    for c in points:
        for eid,tier in c.references:
            if eid==c.event_id or history.label_available(eid,policy.label_config())>live.events[c.event_id].start:raise ValueError('Noncausal reference')
    rows=[dict(asdict(c),features=c.features.tolist()) for c in points]
    atomic_bytes(path,gzip.compress(json.dumps(rows,separators=(',',':')).encode(),compresslevel=1))
    atomic_json(receipt,dict(inputs=expected,sha256=digest(path),points=len(points)))
    return points


def fit(cases,data,policy,cutoff,strength):
    if strength=='anchor':return None
    return JointCorrection().fit(cases,data,policy,cutoff,float(strength))


def predict(model,cases):
    return reconcile(cases,np.array([c.anchor for c in cases])) if model is None else model.predict(cases)


def focal_metric(cases,values,data):
    ii=[i for i,c in enumerate(cases) if c.tier in FOCAL]
    return summary([cases[i] for i in ii],np.asarray(values)[ii],data)


def choose(scores):
    # Candidates are in ascending complexity: anchor, then increasing strength.
    candidates=range(len(STRENGTHS))
    return min(candidates,key=lambda i:(float(np.mean(scores[i])),0 if i==0 else 1,0 if i==0 else -float(STRENGTHS[i])))


def select(data_path,root):
    root=Path(root);policy=Policy();data=load_source(data_path,policy);p=plan_for(data,data_path,policy)
    root.mkdir(parents=True,exist_ok=True);path=root/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=p:raise ValueError('Protocol already exists with different inputs')
    else:atomic_json(path,p)
    verify(root)
    if (root/'selection.json').exists():raise ValueError('Selection is already sealed; use saved artifacts')
    # Exclude audit curves, outcomes and metadata from the selection data view.
    safe=restrict(data,p['selection_reference_ids']);points=cached_cases(root,'selection',safe,safe,p['selection_supervised_ids'],policy)
    scores=[[] for _ in STRENGTHS];fold_results=[];calibration_rows=[];forecast_rows=[]
    for fold in p['inner_folds']:
        eid=fold['validation'];training=[c for c in points if c.event_id in fold['supervised_ids']];validation=[c for c in points if c.event_id==eid]
        train_data=restrict(safe,fold['reference_ids']);previous_choice=choose(scores) if scores[0] else len(STRENGTHS)-1
        result={};predictions=[]
        for i,strength in enumerate(STRENGTHS):
            model=fit(training,train_data,policy,fold['fit_cutoff'],strength);values=predict(model,validation);predictions.append(values)
            score=focal_metric(validation,values,safe);scores[i].append(score['overall']['mape']);result[str(strength)]=score['overall']
        # Adaptive prequential calibration: this choice cannot inspect the
        # current validation outcome, even though final hyper-selection can.
        chosen=predictions[previous_choice]
        for i,c in enumerate(validation):
            truth=safe.labels.get((c.event_id,c.tier))
            row=dict(event_id=c.event_id,tier=c.tier,checkpoint=iso(c.cutoff),hours_remaining=c.hours_remaining,
                current=c.current,actual='' if truth is None else truth,prediction=float(chosen[i]),
                strength=STRENGTHS[previous_choice],choice_uses_only_earlier_validation_events=True)
            if truth is not None:calibration_rows.append(row)
            forecast_rows.append(dict(row,**{'candidate_'+str(s):float(v[i]) for s,v in zip(STRENGTHS,predictions)}))
        fold_results.append(dict(validation=eid,training_event_ids=fold['supervised_ids'],metrics=result,adaptive_strength=STRENGTHS[previous_choice]))
        print('Inner validation',eid,result,flush=True)
    chosen=choose(scores);selection=dict(strength=STRENGTHS[chosen],protocol_sha256=digest(path),
        selection_data_fingerprint=p['selection_fingerprint'],validation_event_ids=[f['validation'] for f in p['inner_folds']],
        candidates=[dict(strength=s,mean_mape=float(np.mean(scores[i])),event_mapes=scores[i]) for i,s in enumerate(STRENGTHS)],
        folds=fold_results,no_audit_labels_used=True)
    atomic_json(root/'selection.json',selection)
    write_csv(root/'inner_forecasts.csv',list(forecast_rows[0]),forecast_rows)
    write_csv(root/'calibration_forecasts.csv',list(calibration_rows[0]),calibration_rows)
    # The sealed research fit is identical for every event in the audit block.
    research=root/'research_model';research.mkdir(exist_ok=True)
    model=fit(points,safe,policy,p['audit_fit_cutoff'],selection['strength'])
    write_model(research,model,points,safe,policy,p['audit_fit_cutoff'],selection,root)
    print('SEALED selection',selection['strength'],'before audit features/outcomes',flush=True)
    return selection


def write_model(directory,model,points,data,policy,cutoff,selection,root):
    if model is None:
        model=JointCorrection().fit(points,data,policy,cutoff,1.);model.coefficients[:]=0.
    model.save(directory/'correction.npz');ids=sorted(data.events,key=lambda e:data.events[e].start)
    save_bank(directory,data,ids,policy.tiers);atomic_json(directory/'policy.json',policy.to_dict())
    calibration=Path(root)/'calibration_forecasts.csv';atomic_bytes(directory/'calibration_forecasts.csv',calibration.read_bytes())
    names=('correction.npz','reference_bank.json','reference_bank.npz','policy.json','calibration_forecasts.csv')
    atomic_json(directory/'model.json',dict(kind='fresh_post_rule_joint_v1',fit_cutoff=cutoff,strength=selection['strength'],
        selection_sha256=digest(Path(root)/'selection.json'),protocol_sha256=digest(Path(root)/'protocol.json'),
        reference_event_ids=ids,supervised_event_ids=sorted({c.event_id for c in points},key=lambda e:data.events[e].start),
        coefficient_count=len(model.coefficients),trained_tiers={str(t):bool(v) for t,v in zip(model.tiers,model.trained)},
        hashes={n:digest(directory/n) for n in names}))


def refit(root):
    root=Path(root);p=verify(root);policy=Policy(**{**p['policy'],'tiers':tuple(p['policy']['tiers']),'checkpoint_hours':tuple(p['policy']['checkpoint_hours'])})
    selection=json.loads((root/'selection.json').read_text())
    if selection['protocol_sha256']!=digest(root/'protocol.json'):raise ValueError('Sealed selection does not match protocol')
    data=load_source(p['data_path'],policy);safe=restrict(data,p['production_reference_ids'])
    points=cached_cases(root,'production',safe,safe,p['production_supervised_ids'],policy)
    directory=root/'production_model'
    if directory.exists():raise ValueError('Production model already exists')
    directory.mkdir();model=fit(points,safe,policy,p['production_fit_cutoff'],selection['strength'])
    write_model(directory,model,points,safe,policy,p['production_fit_cutoff'],selection,root)
    return json.loads((directory/'model.json').read_text())
