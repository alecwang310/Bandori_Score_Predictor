"""Point forecasts from the native raw+small-stat causal transformer."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from collect_bandori_data import CN,atomic_json,iso,timestamp
from .config import Config
from .dataset import Dataset
from .native_sequence import forecast_event
from .pipeline import attach_bank,digest
from .raw_transformer import RawJointTransformer
from .transformer_experiment import verify_plan


def predict(live,root,eid,cutoff):
    root=Path(root);p=verify_plan(root);source=Path(p['source']);directory=root/'full_refit'
    m=json.loads((directory/'model.json').read_text());cfg=Config.load(source/'config.json')
    if eid in m['training_event_ids'] or cutoff<m['fit_cutoff']:raise ValueError('Training event or pre-fit forecast')
    if digest(directory/'model.pt')!=m['model_sha256'] or m['plan_sha256']!=digest(root/'plan.json'):raise ValueError('Transformer weights/plan mismatch')
    bank=json.loads((source/'full_refit/model.json').read_text())
    for name,h in bank['hashes'].items():
        if digest(source/'full_refit'/name)!=h:raise ValueError('Reference bank changed')
    data=attach_bank(source/'full_refit',live);event=data.known_event(data.events[eid],cutoff)[0]
    torch.set_num_threads(1);model=RawJointTransformer.load(directory/'model.pt');values,details=forecast_event(model,data,event,cfg.tiers,[cutoff])
    rows=[]
    for j,tier in enumerate(cfg.tiers):
        if not details['present'][0,j]:rows.append(dict(tier=tier,status='unavailable',reason='No positive score known'));continue
        rows.append(dict(tier=tier,status='ok',final=math.ceil(values[0,j]),current=math.ceil(details['current'][0,j]),
            observed_at=iso(details['observed'][0,j]),score_age_hours=(cutoff-details['observed'][0,j])/3600,
            stale_input=bool(cutoff-details['observed'][0,j]>3600),trained=m['trained_tiers'][str(tier)]))
    return dict(event_id=eid,checkpoint_cn=datetime.fromtimestamp(cutoff,CN).isoformat(),hours_remaining=(event.end-cutoff)/3600,
        approach='raw_small_stat_transformer',parameter_count=m['parameter_count'],native_steps=details['native_steps'],
        model_sha256=m['model_sha256'],predictions=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--data',default='bandori_cn_data')
    p.add_argument('--event',required=True,type=int);p.add_argument('--as-of',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();t=timestamp(a.as_of)
    if t is None:raise ValueError('Timezone-qualified --as-of required')
    if Path(a.output).exists():raise ValueError('Fixed forecast already exists')
    plan=verify_plan(a.model);cfg=Config.load(Path(plan['source'])/'config.json')
    result=predict(Dataset.load(a.data,cfg.tiers),a.model,a.event,t);atomic_json(a.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
