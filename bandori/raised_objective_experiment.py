"""Raised objective-tier loss: retrain the recent NN models with tiers 500/1000/1500/2000 at 60% of loss weight.

The four NN models behind the recent comparison (lstm_reference, lstm_hourly,
lstm_native, transformer) are retrained on the **eight_post_events** fold only.
Streams, packing rules and every hyperparameter match the frozen runs exactly;
the sole change is a global per-tier rescale of the training weights so the four
objective tiers carry exactly 60% of total loss weight mass. Relative weights
inside and outside the objective group are preserved, so the existing
duration/event/horizon/proxy/730-day/3x balancing is untouched within each
group. The rescale governs both the relative-MSE term and the per-tier
signed-bias penalty mass, which read the same normalized weights; realized
loss contribution additionally depends on per-tier error magnitudes.

Comparison follows the standing reporting convention: MAPE counts only tiers
500, 1000, 1500 and 2000 on the eight_post_events fold (validation events
321-324), with companion statistics on the identical rows. No frozen run or
frozen training module is modified.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from collect_bandori_data import atomic_json, timestamp, write_csv
from .analog import reconcile
from .config import Config
from .dataset import Dataset
from .evaluation import paired_bootstrap
from .lstm_variants import SupportedInputLSTM
from .native_experiment import cached_stream
from .native_lstm import NativeJointLSTM
from .native_sequence import pack
from .native_sequence import forecast_event as native_forecast
from .pipeline import digest
from .raw_sequence import pack_streams
from .raw_transformer import RawJointTransformer
from .regime_sequence import forecast_event as reference_forecast
from .regime_sequence import training_streams
from .regime_suite import stress_dataset
from .transformer_reporting import FOCAL_TIERS, REPORT_FOLD, in_scope, score

OBJECTIVE=FOCAL_TIERS
SHARE=.6
MILESTONES=(1,20,40,60,80,100,120)


def raise_objective_share(weights,tiers,objective=OBJECTIVE,share=SHARE):
    """Rescale per-tier training weights so `objective` tiers carry `share` of total mass.

    weights has the tier axis last, e.g. (steps,2,tiers) from pack() or
    (length,2,events,tiers) from pack_streams(). Positive group factors leave
    zero weights (unsupported tiers) at zero and keep within-group ratios.
    """
    w=np.array(weights,dtype=np.float64,copy=True)
    columns=np.array([t in tuple(objective) for t in tiers])
    if not columns.any():raise ValueError('No objective tier present')
    total=float(w.sum())
    if total<=0:raise ValueError('No training weight')
    objective_mass=float(w[...,columns].sum());rest=total-objective_mass
    if objective_mass<=0 or rest<=0:raise ValueError('Need positive weight mass inside and outside the objective tiers')
    objective_factor=share*total/objective_mass;rest_factor=(1-share)*total/rest
    w[...,columns]*=objective_factor;w[...,~columns]*=rest_factor
    info=dict(objective_tiers=[t for t in tiers if t in tuple(objective)],share=share,
        objective_mass_before=objective_mass/total,objective_factor=float(objective_factor),rest_factor=float(rest_factor))
    return w.astype(weights.dtype,copy=False),info


def _fit_reference(data,config,fold,progress):
    streams=training_streams(data,fold['train'],config.tiers,config.seed)
    bundle,target,weights,bins=pack_streams(streams,data,config,fold['fit_cutoff']);del streams
    raised,info=raise_objective_share(weights,config.tiers)
    print('Fit lstm_reference; hourly cells',bundle['x'].shape[0],'objective mass before',
        round(info['objective_mass_before'],4),flush=True)
    model=SupportedInputLSTM(32,16,config.seed,'cpu').fit(bundle,target,raised,bins,120,.005,config.bias_penalty,.001,
        progress=lambda step,loss:progress('lstm_reference',step,loss))
    return model,info


def _fit_native_like(data,config,fold,cadence,native_cache,progress,transformer=False):
    streams=[cached_stream(data,e,config.tiers,cadence,native_cache) for e in fold['train']]
    bundle=pack(streams,data,config,fold['fit_cutoff']);del streams
    raised,info=raise_objective_share(bundle['weights'],config.tiers)
    training=dict(bundle);training['weights']=raised
    name='transformer' if transformer else 'lstm_'+cadence
    print('Fit',name,'steps',bundle['native_steps'],'objective mass before',
        round(info['objective_mass_before'],4),flush=True)
    if transformer:
        model=RawJointTransformer(width=32,heads=4,layers=2,feedforward=64,decoder=16,seed=1729).fit(
            training,epochs=120,learning_rate=.003,bias_penalty=config.bias_penalty,decay=.001,
            progress=lambda step,loss:progress(name,step,loss))
    else:
        model=NativeJointLSTM().fit(training,epochs=120,learning_rate=.005,bias_penalty=config.bias_penalty,decay=.001,
            progress=lambda step,loss:progress(name,step,loss))
    return model,info


def replay(model,kind,data,tiers,rows,cadence=None):
    """Predict template rows exactly as each frozen pipeline did.

    native/transformer predictions come straight from native_sequence.forecast_event
    (which applies observed floors and rank order). The reference LSTM's frozen
    column additionally ran analog.reconcile over the evaluation cases at each
    checkpoint, so it is reproduced here over the template rows of that checkpoint.
    """
    positions={t:i for i,t in enumerate(tiers)};values=np.zeros(len(rows));groups={}
    for i,r in enumerate(rows):groups.setdefault((r['scenario'],int(r['event_id'])),[]).append(i)
    for (scenario,eid),ii in sorted(groups.items()):
        times=sorted({timestamp(rows[i]['checkpoint']) for i in ii})
        def predict(source,times):
            if kind=='reference':return reference_forecast(model,source,source.events[eid],tiers,times)[0]
            return native_forecast(model,source,source.events[eid],tiers,times,cadence)[0]
        if scenario in ('clean','off_grid_37m'):
            output=predict(data,times);slots={t:j for j,t in enumerate(times)}
            for i in ii:values[i]=output[slots[timestamp(rows[i]['checkpoint'])],positions[int(rows[i]['tier'])]]
        else:
            gap={'gap_6h':6,'gap_24h':24}[scenario]
            for t in times:
                sparse=stress_dataset(data,eid,t,gap);output=predict(sparse,[t])
                for i in ii:
                    if timestamp(rows[i]['checkpoint'])==t:values[i]=output[0,positions[int(rows[i]['tier'])]]
    if kind=='reference':
        # The frozen rule-aware pipeline reconciled raw LSTM outputs over the
        # evaluation cases at each (scenario,event,checkpoint); the template
        # rows are exactly those cases.
        checkpoints={}
        for i,r in enumerate(rows):checkpoints.setdefault((r['scenario'],r['event_id'],r['checkpoint']),[]).append(i)
        for ii in checkpoints.values():
            cases=[SimpleNamespace(event_id=int(rows[i]['event_id']),cutoff=timestamp(rows[i]['checkpoint']),
                tier=int(rows[i]['tier']),current=float(rows[i]['current'])) for i in ii]
            for i,v in zip(ii,reconcile(cases,values[ii])):values[i]=float(v)
    return values


def _read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def run(data_path,native_source,source,transformer_source,features,output,share=SHARE):
    native_source,source,transformer_source,features,output=map(Path,(native_source,source,transformer_source,features,output))
    plan_source=json.loads((transformer_source/'plan.json').read_text())
    config=Config.load(source/'config.json');tiers=list(config.tiers)
    fold=dict(next(f for f in plan_source['folds'] if f['name']==REPORT_FOLD))
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new immutable experiment directory')
    output.mkdir(parents=True)
    ledger_before=digest('evaluation_registry.json')
    plan=dict(kind='raised_objective_tier_loss_v1',share=share,objective_tiers=list(OBJECTIVE),
        data=str(Path(data_path).resolve()),native_source=str(native_source.resolve()),
        source=str(source.resolve()),transformer_source=str(transformer_source.resolve()),
        features=str(features.resolve()),config=config.to_dict(),
        fold=dict(name=REPORT_FOLD,train=fold['train'],validation=fold['validation'],fit_cutoff=fold['fit_cutoff']),
        excluded_event_ids=[325,326],
        weighting='Frozen pack()/pack_streams() weights rescaled by one factor per tier group; objective tiers carry `share` of total mass',
        reference=dict(model='SupportedInputLSTM(32,16)',epochs=120,learning_rate=.005,bias_penalty=config.bias_penalty,decay=.001),
        native=dict(model='NativeJointLSTM(32,16)',epochs=120,learning_rate=.005,bias_penalty=config.bias_penalty,decay=.001),
        transformer=dict(model='RawJointTransformer(width=32,heads=4,layers=2,feedforward=64,decoder=16)',epochs=120,learning_rate=.003),
        reporting='Standing convention: MAPE on tiers 500/1000/1500/2000, eight_post_events fold, clean rows',
        code={n:digest(Path('bandori')/n) for n in ('raised_objective_experiment.py','native_sequence.py','raw_sequence.py',
            'regime_sequence.py','native_lstm.py','lstm_variants.py','raw_transformer.py','rules.py')},
        stream_hashes={cadence:{str(e):digest(native_source/'streams'/f'{cadence}_{e}.npz') for e in fold['train']}
            for cadence in ('hourly','native')},
        ledger_sha256=ledger_before)
    atomic_json(output/'plan.json',plan)
    torch.set_num_threads(1)
    data=Dataset.load(data_path,config.tiers);data.add_terminal_labels(config)
    progress=lambda name,step,loss:print(name,step,round(loss,7),flush=True) if step in MILESTONES else None
    fitted={}
    fitted['reference']=_fit_reference(data,config,fold,progress)
    for cadence in ('hourly','native'):
        fitted[cadence]=_fit_native_like(data,config,fold,cadence,native_source,progress)
    fitted['transformer']=_fit_native_like(data,config,fold,'native',native_source,progress,transformer=True)
    templates=dict(reference=source/REPORT_FOLD/'predictions.csv',
        hourly=native_source/'hourly'/REPORT_FOLD/'predictions.csv',
        native=native_source/'native'/REPORT_FOLD/'predictions.csv',
        transformer=transformer_source/REPORT_FOLD/'predictions.csv')
    rows=_read_rows(templates['native'])
    def key(r):return r['scenario'],r['event_id'],r['tier'],r['checkpoint']
    joined={name:{key(r):r for r in _read_rows(path)} for name,path in templates.items()}
    if not all(set(joined['native'])==set(joined[n]) for n in joined):raise ValueError('Frozen template row coverage differs')
    clock={key(r):r for r in _read_rows(features/REPORT_FOLD/'predictions.csv')}
    for r in rows:
        k=key(r)
        r['frozen_reference']=joined['reference'][k]['raw_lstm'];r['frozen_hourly']=joined['hourly'][k]['hourly']
        r['frozen_native']=joined['native'][k]['native'];r['frozen_transformer']=joined['transformer'][k]['transformer']
        r['baseline']=joined['reference'][k]['baseline'];r['ridge_previous']=joined['reference'][k]['ridge']
        r['ridge_clock']=clock[k]['ridge']
    kinds=dict(reference='reference',hourly='native',native='native',transformer='native')
    cadences=dict(reference=None,hourly='hourly',native='native',transformer='native')
    for name,(model,_) in fitted.items():
        print('Replay raised-objective',name,flush=True)
        for r,v in zip(rows,replay(model,kinds[name],data,tiers,rows,cadences[name])):r['raised_'+name]=float(v)
    write_csv(output/'predictions.csv',list(rows[0]),rows)
    scoped=[r for r in rows if in_scope(r)]
    names=['frozen_reference','frozen_hourly','frozen_native','frozen_transformer',
        'raised_reference','raised_hourly','raised_native','raised_transformer','ridge_clock','ridge_previous','baseline']
    def summarize(rr,n):
        s=score(rr,n)[0];return dict(s['overall'],by_horizon={h:v for h,v in s['by_horizon'].items() if v['rows']})
    reported={n:summarize(scoped,n) for n in names}
    paired={}
    for name in ('reference','hourly','native','transformer'):
        _,cases,labels,frozen_values=score(scoped,'frozen_'+name)
        raised_values=np.asarray([float(r['raised_'+name]) for r in scoped])
        paired[name]=paired_bootstrap(cases,frozen_values,raised_values,labels,1729)
    stress={s:{n:score([r for r in rows if in_scope(r,s)],n)[0]['overall'] for n in names}
        for s in ('off_grid_37m','gap_6h','gap_24h')}
    result=dict(plan_sha256=digest(output/'plan.json'),share=share,objective_tiers=list(OBJECTIVE),
        scope=dict(fold=REPORT_FOLD,tiers=list(FOCAL_TIERS),scenario='clean',validation_events=fold['validation']),
        weight_shift={n:info for n,(model,info) in fitted.items()},
        reported=reported,paired=paired,stress=stress,
        trained_parameters={n:sum(v.numel() for v in model.parameters()) for n,(model,_) in fitted.items()},
        limitations=['Four previously inspected validation events (321-324), not a fresh test',
            'Only the eight_post_events fold is retrained; no full_refit or other fold',
            'Single share setting; no tuning against validation outcomes',
            'Weight mass is exactly 60% on the objective tiers; realized loss share also depends on per-tier errors',
            'Proxy final labels and initially unsupported tiers retain the original caveats'])
    atomic_json(output/'evaluation.json',result)
    if digest('evaluation_registry.json')!=ledger_before:raise ValueError('Holdout ledger changed unexpectedly')
    def pct(v):return '—' if v is None else f'{100*v:.2f}%'
    order=[('lstm_reference','reference'),('lstm_hourly','hourly'),('lstm_native','native'),('transformer','transformer')]
    lines=['# Raised objective-tier loss (60%) — eight_post_events retrain','',
        f"Objective tiers {', '.join(str(t) for t in OBJECTIVE)} carry **{100*share:.0f}%** of total training loss weight mass "
        f"(raised from {100*result['weight_shift']['transformer']['objective_mass_before']:.1f}%). "
        'Relative weights inside and outside the objective group are preserved, so the duration/event/horizon/proxy/730-day/3x '
        'balancing is unchanged within each group; the same normalized weights drive the relative-MSE term and the per-tier '
        'signed-bias penalty. Only the `eight_post_events` fold (315 training events, validation 321-324) is retrained; '
        'streams, packing and every hyperparameter match the frozen runs.','',
        'Reported MAPE counts only tiers 500, 1000, 1500 and 2000 on the `eight_post_events` fold, clean checkpoints. '
        'Companion statistics use the identical rows.','',
        '| Model | MAPE | Change vs frozen | Signed bias | Final 24h MAPE |','| --- | ---: | ---: | ---: | ---: |']
    for label,name in order:
        frozen=reported['frozen_'+name];raised=reported['raised_'+name]
        change=100*((raised.get('mape') or float('nan'))-(frozen.get('mape') or float('nan')))
        lines.append(f"| {label} (frozen) | {pct(frozen.get('mape'))} | — | {pct(frozen.get('bias'))} | "
            f"{pct(frozen.get('by_horizon',{}).get('0-24h',{}).get('mape'))} |")
        lines.append(f"| {label} (60% objective) | {pct(raised.get('mape'))} | {change:+.2f} pp | {pct(raised.get('bias'))} | "
            f"{pct(raised.get('by_horizon',{}).get('0-24h',{}).get('mape'))} |")
    lines+=['','## Non-NN references (frozen)','',
        '| Model | MAPE | Signed bias | Final 24h MAPE |','| --- | ---: | ---: | ---: |']
    for label in ('ridge_clock','ridge_previous','baseline'):
        a=reported[label]
        lines.append(f"| {label} | {pct(a.get('mape'))} | {pct(a.get('bias'))} | {pct(a.get('by_horizon',{}).get('0-24h',{}).get('mape'))} |")
    lines+=['','## Paired event-bootstrap: raised minus frozen','',
        '| Model | MAPE change (percentage points) | 95% interval |','| --- | ---: | ---: |']
    for label,name in order:
        a=paired[name]
        lines.append(f"| {label} | {100*a['mape_change']:+.2f} | [{100*a['interval_95'][0]:+.2f}, {100*a['interval_95'][1]:+.2f}] |")
    lines+=['','Negative favors the raised-objective retrain. Four event clusters limit precision.','',
        '## Weight shift','',
        '| Bundle | Objective mass before | Objective factor | Other factor |','| --- | ---: | ---: | ---: |']
    for n,info in result['weight_shift'].items():
        lines.append(f"| {n} | {100*info['objective_mass_before']:.2f}% | {info['objective_factor']:.3f} | {info['rest_factor']:.3f} |")
    lines+=['','## Fixed stress queries (same scope)','',
        '| Model | Off-grid 08:37 | 6h tracker gap | 24h tracker gap |','| --- | ---: | ---: | ---: |']
    for label,name in order:
        lines.append(f"| {label} (60% objective) | " +
            ' | '.join(pct(stress[s]['raised_'+name].get('mape')) for s in ('off_grid_37m','gap_6h','gap_24h')) + ' |')
    lines+=['','Stress queries keep the same fixed checkpoint subset as the frozen study. Tracker gaps are not verified server outages.','',
        f"Fit: `{output}`. No deployed model is replaced by this experiment.",'']
    destination=Path('reports')/(output.name+'.md');destination.parent.mkdir(exist_ok=True)
    destination.write_text('\n'.join(lines))
    print('Raised-objective retrain and comparison complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/full_approaches_20261017')
    p.add_argument('--native-source',default='runs/native_lstm_v2_20261018');p.add_argument('--source',default='runs/rule_aware_3x_20261018')
    p.add_argument('--transformer-source',default='runs/raw_transformer_20261018');p.add_argument('--features',default='runs/clock_features_v2_20261018')
    p.add_argument('--output',required=True);p.add_argument('--share',type=float,default=SHARE)
    a=p.parse_args();run(a.data,a.native_source,a.source,a.transformer_source,a.features,a.output,a.share)


if __name__=='__main__':main()
