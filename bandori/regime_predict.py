"""Freeze and use a chosen rule-aware joint point predictor."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path

from collect_bandori_data import CN, atomic_bytes, atomic_json, iso, timestamp
from .config import Config
from .dataset import Dataset
from .pipeline import attach_bank, digest
from .regime_suite import finish_context, load_fit, make_context, predictions
from .role_analog import RoleAwareAnalog


def finalize(directory):
    directory=Path(directory);model=json.loads((directory/'full_refit/model.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    content=(directory/'full_refit/config.json').read_bytes();config_path=directory/'config.json'
    if config_path.exists() and config_path.read_bytes()!=content:raise ValueError('Frozen configuration differs')
    if not config_path.exists():atomic_bytes(config_path,content)
    files=('plan.json','config.json','selection.json','evaluation.json','full_refit/model.json')
    manifest=dict(kind='rule_aware_suite',champion=selection['champion'],training_event_ids=model['training_event_ids'],
        excluded_event_ids=[325,326],fit_cutoff=model['fit_cutoff'],training_targets_by_tier=model['training_targets_by_tier'],
        hashes={name:digest(directory/name) for name in files},
        limitations=['Historical post-rule validation was previously inspected; only twelve event clusters',
            'Closing tracker proxies are not independently verified official finals',
            'No probability intervals are calibrated',
            'Model selection uses frozen chronological folds; events 325/326 excluded from all fits and selection',
            'Historical publication/schedule metadata may be incomplete'])
    path=directory/'model.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Frozen manifest differs')
    if not path.exists():atomic_json(path,manifest)
    return manifest


def predict(live,directory,event_id,cutoff,output=None,max_stale_hours=None,approach=None):
    directory=Path(directory);manifest=json.loads((directory/'model.json').read_text())
    for name,expected in manifest['hashes'].items():
        if digest(directory/name)!=expected:raise ValueError('Suite checksum mismatch: '+name)
    if event_id in manifest['training_event_ids'] or cutoff<manifest['fit_cutoff']:raise ValueError('Training event or pre-fit forecast')
    fit=directory/'full_refit';config=Config.load(fit/'config.json');freshness=config.max_stale_minutes/60
    if max_stale_hours is not None:
        if not math.isfinite(max_stale_hours) or max_stale_hours<=0:raise ValueError('Invalid score-age allowance')
        config=replace(config,max_stale_minutes=max_stale_hours*60)
    data=attach_bank(fit,live);event=data.known_event(data.events[event_id],cutoff)[0]
    if not event.start<cutoff<event.end:raise ValueError('Checkpoint outside the known event schedule')
    predictor=RoleAwareAnalog(data,config,manifest['training_event_ids']);cases=[];failed=[]
    for tier in config.tiers:
        try:cases.append(predictor.build(event,tier,cutoff))
        except ValueError as error:failed.append(dict(tier=tier,status='unavailable',reason=str(error)))
    if not cases:raise ValueError('No usable score lines at this checkpoint')
    cores,specs,compression,masks,metadata=load_fit(fit)
    ctx=make_context(cases,data,config,manifest['training_event_ids']);ctx,_=finish_context(ctx,compression,masks)
    forecasts=predictions(specs,cores,ctx,data,config.tiers);name=approach or manifest['champion']
    if name not in forecasts:raise ValueError('Unknown approach: '+name)
    rows=[dict(tier=c.tier,status='ok',current=math.ceil(c.current),final=math.ceil(forecasts[name][i]),
               observed_at=iso(c.observed_at),score_age_hours=(cutoff-c.observed_at)/3600,
               stale_input=(cutoff-c.observed_at)>freshness*3600,
               role_transfer_references=int(c.features.get('role_transfer_support',0)),
               supervised_training_targets=metadata['lstm_targets_by_tier'][str(c.tier)] if name=='raw_lstm' else metadata['training_targets_by_tier'][str(c.tier)]) for i,c in enumerate(cases)]
    result=dict(event_id=event_id,checkpoint_cn=datetime.fromtimestamp(cutoff,CN).isoformat(),champion=name,
        model=str(directory.resolve()),hours_remaining=(event.end-cutoff)/3600,post_rule_change_multiplier=3,
        predictions=sorted(rows+failed,key=lambda r:r['tier']),limitations=manifest['limitations'])
    if output:
        if Path(output).exists():raise ValueError('Fixed prediction already exists')
        atomic_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    freeze=commands.add_parser('finalize');freeze.add_argument('--model',required=True)
    pred=commands.add_parser('predict');pred.add_argument('--data',required=True);pred.add_argument('--model',required=True)
    pred.add_argument('--event',required=True,type=int);pred.add_argument('--as-of',required=True);pred.add_argument('--output',required=True)
    pred.add_argument('--approach');pred.add_argument('--max-stale-hours',type=float)
    args=parser.parse_args()
    if args.command=='finalize':print(json.dumps(finalize(args.model),indent=2))
    else:
        config=Config.load(Path(args.model)/'full_refit/config.json');data=Dataset.load(args.data,config.tiers);cutoff=timestamp(args.as_of)
        if cutoff is None:raise ValueError('Timezone-qualified --as-of required')
        print(json.dumps(predict(data,args.model,args.event,cutoff,args.output,args.max_stale_hours,args.approach),indent=2))


if __name__=='__main__':main()
