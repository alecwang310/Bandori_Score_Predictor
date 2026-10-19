"""Descriptive, event-balanced out-of-fold RMS error scales; no point refit."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import timestamp
from .pipeline import digest


EDGES=(0,24,48,72,96,144,192,float('inf'))


def bucket(hours):
    if not math.isfinite(hours) or hours<0:raise ValueError('Invalid forecast horizon')
    return min(len(EDGES)-2,int(np.searchsorted(EDGES,hours,side='right')-1))


def cell(records,fit_cutoff,tau=730):
    """Average within an event first; dense tracking cannot create more events."""
    groups={}
    for r in records:groups.setdefault(int(r['event_id']),[]).append(r)
    if not groups:return None
    first=[];second=[];weights=[]
    for eid,rr in groups.items():
        errors=np.array([(float(r['ridge'])-float(r['actual']))/float(r['ridge']) for r in rr])
        first.append(float(errors.mean()));second.append(float(np.mean(errors**2)))
        weights.append(math.exp(-max(0,fit_cutoff-rr[0]['event_end'])/86400/tau))
    mean=float(np.average(first,weights=weights));mse=float(np.average(second,weights=weights))
    w=np.asarray(weights)
    return dict(relative_rmse=math.sqrt(max(0,mse)),relative_bias=mean,
        centered_relative_sd=math.sqrt(max(0,mse-mean**2)),events=len(groups),rows=sum(map(len,groups.values())),
        effective_events=float(w.sum()**2/np.dot(w,w)),event_ids=sorted(groups))


def build_calibration(directory):
    root=Path(directory);plan=json.loads((root/'plan.json').read_text())
    model=json.loads((root/'full_refit/model.json').read_text());bank=json.loads((root/'full_refit/reference_bank.json').read_text())
    ends={int(e['event_id']):e['end'] for e in bank['events']};cutoff=model['fit_cutoff'];rows=[]
    allowed={e for fold in plan['folds'] for e in fold['validation']}
    for fold in plan['folds']:
        if set(fold['train'])&set(fold['validation']):raise ValueError('Calibration fold leakage')
        with (root/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                eid=int(r['event_id'])
                if eid not in fold['validation'] or eid not in allowed or eid in (325,326):raise ValueError('Unexpected calibration event')
                if not r['actual'] or float(r['actual'])<=0 or float(r['ridge'])<=0:continue
                if timestamp(r['checkpoint'])>=ends[eid]:raise ValueError('Calibration query is after event close')
                if ends[eid]>cutoff:raise ValueError('Calibration outcome postdates the deployment fit')
                r['event_end']=ends[eid];rows.append(r)
    cells={};tiers=plan['config']['tiers']
    for scenario in ('clean','gap_6h','gap_24h'):
        for tier in tiers:
            base=[r for r in rows if r['scenario']==scenario and int(r['tier'])==tier]
            for h in range(len(EDGES)-1):
                selected=[r for r in base if bucket(float(r['hours_remaining']))==h]
                value=cell(selected,cutoff)
                if value:cells[f'{scenario}:{tier}:{h}']=value
            value=cell(base,cutoff)
            if value:cells[f'{scenario}:{tier}:all']=value
    return dict(version=1,point_model='ridge',source_selection_sha256=digest(root/'selection.json'),
        source_model_sha256=digest(root/'model.json'),calibration_event_ids=sorted(allowed),excluded_event_ids=[325,326],
        available_at=cutoff,characteristic_days=730,minimum_direct_events=6,minimum_fallback_events=4,
        horizon_edges_hours=[v if math.isfinite(v) else None for v in EDGES],cells=cells,
        formula='r=(forecast-truth)/forecast; RMS=sqrt(weighted event mean(r^2)); sigma=forecast*RMS',
        top_rank_fallback={'1':20,'10':20},
        limitations=['Previously inspected out-of-fold outcomes also selected the model family; uncertainty is not independently calibrated',
            'Twelve post-rule event clusters; repeated timestamps are not independent events',
            'RMS includes systematic bias; it is not a centered standard deviation or guaranteed 68% coverage',
            'Closing-tracker proxies are not verified official finals',
            'T1/T10 have no direct recent labels; any supplied sigma explicitly borrows T20',
            'No event-type-specific calibration is fitted with only one to three events per type',
            'Production has more training events; no unsupported reduction of sigma is applied'])


def estimate(calibration,tier,hours,point,score_age_hours=0):
    h=bucket(hours);scenario='gap_24h' if score_age_hours>12 else 'gap_6h' if score_age_hours>2 else 'clean'
    cells=calibration['cells'];direct=calibration['minimum_direct_events'];fallback=calibration['minimum_fallback_events']
    candidates=[(tier,h,direct),(tier,'all',direct),(tier,h,fallback),(tier,'all',fallback)]
    borrowed=calibration.get('top_rank_fallback',{}).get(str(tier))
    if borrowed is not None:candidates.extend(((borrowed,h,direct),(borrowed,'all',direct),(borrowed,h,fallback),(borrowed,'all',fallback)))
    chosen=None
    # Stale-score scale must never shrink below a clean scale on the same tier.
    for source_tier,source_horizon,minimum in candidates:
        key=f'{scenario}:{source_tier}:{source_horizon}';c=cells.get(key)
        if c and c['events']>=minimum:
            chosen=(source_tier,source_horizon,c);break
    if chosen is None and scenario!='clean':
        clean=estimate(calibration,tier,hours,point,0)
        if clean:clean.update(stale_calibration_unavailable=True,requested_scenario=scenario)
        return clean
    if chosen is None:return None
    source_tier,source_horizon,c=chosen;relative=c['relative_rmse']
    clean_scale=estimate(calibration,tier,hours,point,0) if scenario!='clean' else None
    if clean_scale:relative=max(relative,clean_scale['relative_rmse'])
    return dict(value=point*relative,relative_rmse=relative,relative_bias=c['relative_bias'],
        centered_relative_sd=c['centered_relative_sd'],events=c['events'],effective_events=c['effective_events'],rows=c['rows'],
        source_tier=source_tier,borrowed_from_tier=source_tier if source_tier!=tier else None,
        horizon_bucket=source_horizon,horizon_fallback=source_horizon=='all',scenario=scenario,
        approximate=True,coverage_guaranteed=False)


def add_uncertainty(forecast,calibration):
    if timestamp(forecast['checkpoint_cn'])<calibration['available_at']:raise ValueError('Calibration was unavailable at this issue time')
    if forecast['source_model_sha256']!=calibration['source_model_sha256']:raise ValueError('Uncertainty belongs to a different frozen model')
    for row in forecast['predictions']:
        if row['status']=='ok':row['sigma']=estimate(calibration,row['tier'],forecast['hours_remaining'],row['final'],row['score_age_hours'])
    forecast['uncertainty']=dict(formula=calibration['formula'],calibration_event_ids=calibration['calibration_event_ids'],
        limitations=calibration['limitations'])
    return forecast
