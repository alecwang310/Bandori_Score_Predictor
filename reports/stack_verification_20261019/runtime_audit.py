"""Replay the shipped models in isolation; do not fit or modify any model."""
import csv
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIST=ROOT/'dist'
sys.path.insert(0,str(DIST))
sys.modules['torch']=None
from unified import Distribution
from bandori.dataset import Dataset
from bandori.live_data import load_snapshot
from collect_bandori_data import CN,timestamp

OUT=Path(__file__).resolve().parent
MODEL_NAMES=('fresh','ridge','stack')
FOCAL=(500,1000,1500,2000)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    ledger=digest(ROOT/'evaluation_registry.json')
    distribution=Distribution(DIST)
    asset_before={str(p.relative_to(DIST)):digest(p) for p in (DIST/'models').rglob('*') if p.is_file()}
    for name in MODEL_NAMES:
        manifest=json.loads((DIST/'models'/name/'manifest.json').read_text())
        for file,h in manifest['assets'].items():assert digest(DIST/'models'/name/file)==h
    data=Dataset.load(ROOT/'bandori_cn_data',distribution.policy.tiers)
    data.add_terminal_labels(distribution.policy.label_config())
    truth={tier:data.labels[(325,tier)] for tier in distribution.policy.tiers if (325,tier) in data.labels}
    event=data.events[325]
    cutoffs=[]
    day=datetime.fromtimestamp(event.start,CN).replace(hour=0,minute=0,second=0,microsecond=0)
    while day.timestamp()<event.end:
        for hour in (8,20,21):
            value=day.replace(hour=hour).timestamp()
            if event.start<value<event.end:cutoffs.append(value)
        day+=timedelta(days=1)
    rows=[];failed=[];availability={};runtime_outputs=OUT/'runtime_outputs';runtime_outputs.mkdir(exist_ok=True)
    for i,cutoff in enumerate(cutoffs):
        for name in MODEL_NAMES:
            try:result=distribution.predict(name,data,325,cutoff)
            except ValueError as error:
                failed.append(dict(model=name,cutoff=cutoff,reason=str(error)));continue
            valid=[r for r in result['predictions'] if r['status']=='ok']
            assert all(r['final']>=r['current'] for r in valid)
            assert all(a['final']>=b['final'] for a,b in zip(valid,valid[1:]))
            assert all(timestamp(r['observed_at'])<=cutoff and r['score_age_hours']<=1 for r in valid)
            availability[(name,cutoff)]={r['tier'] for r in valid}
            for row in valid:
                if row['tier'] in truth:
                    error=(row['final']/truth[row['tier']]-1)*100
                    rows.append(dict(model=name,event_id=325,cutoff=cutoff,checkpoint_cn=result['checkpoint_cn'],
                        hours_remaining=result['hours_remaining'],tier=row['tier'],prediction=row['final'],actual=truth[row['tier']],
                        observed_at=row['observed_at'],score_age_hours=row['score_age_hours'],
                        signed_error_percent=error,absolute_error_percent=abs(error)))
            if cutoff==timestamp('2026-10-14T20:00:00+08:00'):
                (runtime_outputs/f'{name}_325.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        if i%8==0:print(f'Replayed {i+1}/{len(cutoffs)} scheduled checkpoints',flush=True)
    common=set.intersection(*[{(r['cutoff'],r['tier']) for r in rows if r['model']==name and r['tier'] in FOCAL} for name in MODEL_NAMES])
    union=set.union(*[{(r['cutoff'],r['tier']) for r in rows if r['model']==name and r['tier'] in FOCAL} for name in MODEL_NAMES])
    assert common==union, 'Unmatched focal coverage between models'
    external=list(csv.DictReader((ROOT/'reports/fresh_event325_external_20261019/comparison.csv').open()))
    external_keys={(timestamp(r['issued_at']),int(r['tier'])) for r in external if r['fresh_input']=='True'}
    assert external_keys<=common and len(external_keys)==24
    def metrics(selected):
        return {name:dict(rows=len(rr),mape_percent=float(np.mean([r['absolute_error_percent'] for r in rr])),
            bias_percent=float(np.mean([r['signed_error_percent'] for r in rr])),
            p90_absolute_error_percent=float(np.quantile([r['absolute_error_percent'] for r in rr],.9)),
            worst_absolute_error_percent=max(r['absolute_error_percent'] for r in rr))
            for name in MODEL_NAMES if (rr:=[r for r in selected if r['model']==name])}
    focal=[r for r in rows if (r['cutoff'],r['tier']) in common]
    summary=dict(standing=metrics(focal),external_24=metrics([r for r in focal if (r['cutoff'],r['tier']) in external_keys]),
        only_0800=metrics([r for r in focal if datetime.fromtimestamp(r['cutoff'],CN).hour==8]),
        last_24h=metrics([r for r in focal if r['hours_remaining']<=24]),
        by_tier={str(t):metrics([r for r in focal if r['tier']==t]) for t in FOCAL},
        by_horizon={label:metrics([r for r in focal if low<r['hours_remaining']<=high]) for label,low,high in
                    [('0-24h',0,24),('24-72h',24,72),('72-144h',72,144),('144-240h',144,240),('>240h',240,float('inf'))]},
        scheduled_checkpoints=len(cutoffs),scored_checkpoints=len(common)//4,failed_checkpoints=failed,
        all_labeled_tiers=metrics(rows),matched_focal_rows_per_model=len(common),finals=truth,checks={})
    assert len(common)==164
    for name,expected in [('stack',7.545191815010358),('ridge',8.790822825824854),('fresh',10.634931606797905)]:
        assert abs(summary['standing'][name]['mape_percent']-expected)<1e-4,(name,summary['standing'][name])
    # Real exported-model checks: unseen outcomes, future observations and late publications.
    for eid,date in [(325,'2026-10-14T20:00:00+08:00'),(326,'2026-10-17T20:00:00+08:00')]:
        cutoff=timestamp(date)
        snapshot=DIST/f'examples/event_{eid}_{"20261014" if eid==325 else "20261017"}_2000.json.gz'
        live=load_snapshot(snapshot)
        before={name:distribution.predict(name,live,eid,cutoff) for name in MODEL_NAMES}
        for name,value in before.items():
            (runtime_outputs/f'{name}_{eid}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        for key,c in live.curves.items():
            c.times=np.r_[c.times,cutoff+60,cutoff-1]
            c.scores=np.r_[c.scores,1e14,1e14]
            c.available=np.r_[c.available,cutoff+60,cutoff+60]
            order=np.argsort(c.times,kind='stable');c.times=c.times[order];c.scores=c.scores[order];c.available=c.available[order]
            live.labels[key]=1e15
        for name in MODEL_NAMES:
            after=distribution.predict(name,live,eid,cutoff)
            assert before[name]['predictions']==after['predictions'],(eid,name,'causality failed')
            summary['checks'][f'{name}_{eid}_future_tail_late_publication_own_label_invariant']=True
    # Verify advertised stale-input option without changing or silently bypassing it.
    try:distribution.predict('stack',data,325,timestamp('2026-10-08T21:00:00+08:00'),48)
    except Exception as error:summary['stack_stale_option']=dict(ok=False,error_type=type(error).__name__,error=str(error))
    else:summary['stack_stale_option']=dict(ok=True)
    summary['package_manifest_present']=(DIST/'PACKAGE_SHA256.json').exists()
    assert asset_before=={str(p.relative_to(DIST)):digest(p) for p in (DIST/'models').rglob('*') if p.is_file()}
    assert ledger==digest(ROOT/'evaluation_registry.json')
    summary['checks'].update(model_assets_unchanged=True,protected_ledger_unchanged=True,matched_focal_coverage=True,
        asset_hashes_valid=True,numpy_only_isolated_import=True)
    summary['asset_sha256']=asset_before
    save('runtime_metrics.json',summary)
    with (OUT/'runtime_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:summary[k] for k in ('standing','external_24','only_0800','last_24h','stack_stale_option')},indent=2),flush=True)

if __name__=='__main__':main()
