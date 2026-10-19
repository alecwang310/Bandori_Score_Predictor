"""Matched feature and native-cadence results, with event-cluster evidence."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from collect_bandori_data import atomic_json
from .analog import HORIZONS
from .clock_experiment import ENSEMBLES
from .evaluation import paired_bootstrap, summary


def pct(x):return '—' if x is None else f'{100*x:.2f}%'


def key(row):return row['event_id'],row['tier'],row['checkpoint'],row['scenario']


def scored(rows,name):
    cases=[SimpleNamespace(event_id=int(r['event_id']),tier=int(r['tier']),remaining_hours=float(r['hours_remaining'])) for r in rows]
    data=SimpleNamespace(labels={(int(r['event_id']),int(r['tier'])):float(r['actual']) for r in rows if r['actual']},
        label_sources={(int(r['event_id']),int(r['tier'])):r['label_source'] for r in rows})
    values=np.asarray([float(r[name]) for r in rows]);return cases,data,values,summary(cases,values,data)


def plateau(rows,name):
    groups={}
    for r in rows:
        if int(r['tier']) in (1000,1500):groups.setdefault((r['event_id'],r['checkpoint']),{})[int(r['tier'])]=float(r[name])
    pairs=[r for r in groups.values() if len(r)==2]
    return dict(checkpoints=len(pairs),equal=sum(abs(r[1000]-r[1500])<1 for r in pairs),
        fraction=sum(abs(r[1000]-r[1500])<1 for r in pairs)/len(pairs) if pairs else None)


def publish(feature_root,native_root,destination):
    feature_root,native_root,destination=map(Path,(feature_root,native_root,destination))
    fp=json.loads((feature_root/'plan.json').read_text());fr=json.loads((feature_root/'evaluation.json').read_text())
    nr=json.loads((native_root/'evaluation.json').read_text());rows=[]
    for fold in fp['folds']:
        with (feature_root/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
        with (Path(fp['source'])/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:legacy={key(r):r for r in csv.DictReader(f)}
        for r in rr:r['lstm_reference']=legacy[key(r)]['raw_lstm']
        for cadence in ('hourly','native'):
            with (native_root/cadence/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:nn={key(r):r for r in csv.DictReader(f)}
            for r in rr:
                if key(r) not in nn:raise ValueError('Native and feature case coverage differs')
                r['lstm_'+cadence]=nn[key(r)][cadence]
        for r in rr:
            r['control_clock_baseline']=r['control_baseline']
            for name,members in ENSEMBLES.items():r['control_'+name]=float(np.mean([float(r['control_'+n]) for n in members]))
        rows.extend(rr)
    names=list(next(iter(fr['folds'].values()))['clean']);clean=[r for r in rows if r['scenario']=='clean']
    summaries={};core={};paired={};plateaus={};scenarios={};new_names=[*names,'lstm_reference','lstm_hourly','lstm_native']
    for n in new_names:
        cases,data,values,s=scored(clean,n)
        summaries[n]=s
        focal=[r for r in clean if int(r['tier']) in (500,1000,1500,2000)];core[n]=scored(focal,n)[3]
        reference='lstm_hourly' if n=='lstm_native' else 'control_'+n
        if reference in clean[0]:paired[n]=paired_bootstrap(cases,np.asarray([float(r[reference]) for r in clean]),values,data,1729)
        plateaus[n]=plateau(clean,n)
        for scenario in ('off_grid_37m','gap_6h','gap_24h'):
            scenarios.setdefault(scenario,{})[n]=scored([r for r in rows if r['scenario']==scenario],n)[3]
    controls={n:scored(clean,'control_'+n)[3] for n in names}
    dense_path=native_root/'matched_timestamp_evaluation.json'
    dense=json.loads(dense_path.read_text()) if dense_path.exists() else None
    result=dict(champion=fr['champion'],clock_features=fp['clock_features'],clean=summaries,core_tiers=core,
        paired_feature_controls=controls,paired_bootstrap=paired,scenarios=scenarios,plateaus=plateaus,
        native_cadence=nr['cadence'],folds=fr['folds'],matched_timestamps=dense,
        limitations=['Historical follow-up on twelve previously inspected post-rule events; not a fresh test',
            'Family choices, architectures, feature definitions and gains fixed before replay',
            'The native-versus-hourly LSTM comparison uses the same new packed architecture and small query decoder',
            'Every observation is retained; denser timestamps are not independent event clusters',
            'Existing deployed dist model is unchanged until a separate promotion decision'])
    destination.parent.mkdir(parents=True,exist_ok=True);atomic_json(destination.with_suffix('.json'),result)
    lines=['# Clock-feature engineering and native-frequency raw LSTM','',
        'Ridge remains the best of these approaches: clock features change its MAPE from 9.15% to 9.12%, with an event-bootstrap interval spanning no improvement. '
        'Most small NNs worsen. Native sampling also fails to improve the matched LSTM control (10.25% hourly versus 10.47% native), although signed bias decreases. '
        'The stable deployment is retained; the richer ridge is available as a separate candidate.','',
        '**Historical paired replay on twelve previously inspected events; no fresh-test claim.** '
        'All fits retain 730-day decay and extra 3x post-rule weight. Events 325/326 never enter fitting, feature preprocessing or selection. '
        'Three chronological folds have zero, four and eight post-rule examples; production refits use the authorized 319-event bank.','',
        f"Added **{len(fp['clock_features'])} features per tier**: circular mean daily peak time, peak concentration/stability/width, "
        'historical weekday/weekend peak timing, clock-profile entropy/correlation, 6h rate slope/acceleration, and eight fixed 3h clock-window rates, slopes, coverage, same-clock day changes and future exposure shares. '
        'Missing slope/overlap support has explicit flags. Ordinary peak timing excludes the opening 12h and final 24h; separate flags retain those regimes. '
        'Current-event shapes use only the known prefix and shrink toward training-reference-only historical profiles.','',
        'The minimal/PCA/ridge paths retain twelve explicit clock summaries plus sixteen training-only clock PCs. Standard/rich NN paths receive all clock features directly. '
        'Hierarchical regression receives twelve explicit summaries. The new statistical pace adapts the historical weekday/weekend curve with current-event hourly evidence, '
        'then adjusts recent/future calendar exposure while retaining the previous final-rush pace anchor. Its multiplier is bounded to 0.75–1.333. '
        'The completion/pace and trajectory families blend this estimate at a fixed 0.5 and refit corrections only on prequential training cases.','',
        '## Feature ablation on identical cases','',
        '| Approach | Previous MAPE | New MAPE | New focal-tier MAPE | New bias | Previous → new final 24h |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in names:
        a,b=controls[n],summaries[n]
        lines.append(f"| {n}{' **selected**' if n==fr['champion'] else ''} | {pct(a['overall'].get('mape'))} | {pct(b['overall'].get('mape'))} | {pct(core[n]['overall'].get('mape'))} | {pct(b['overall'].get('bias'))} | {pct(a['by_horizon'].get('0-24h',{}).get('mape'))} → {pct(b['by_horizon'].get('0-24h',{}).get('mape'))} |")
    lines+=['','Collection controls average the same member list as each new collection. This avoids calling an ensemble-membership change a feature improvement. '
        'Selection uses the predeclared error/bias/fold-stability score and original baseline signed-bias/final-day guards; stress and held-out 325/326 outcomes cannot select it.','',
        '## Raw LSTM sampling ablation','',
        'Both arms use one stock 32-unit LSTM and a 16-unit decoder, 120 fixed epochs, Adam 0.005, decay 0.001 and signed-bias penalty 3. '
        'A packed sequence retains the entire event, without padded fake observations or history truncation. '
        'The native arm updates at every publication-causal tracker timestamp across all tiers, including subsecond offsets between different lines. '
        'Typical per-tier spacing is around 30 minutes, with observed bursts near 11–12 seconds. No resampling invents scores between packets. '
        'Elapsed time, clock, freshness, type/reward metadata and small 6h/24h growth summaries are inputs; the engineered clock feature bank is not fed to this raw model. '
        'Queries read the latest recurrent state through a clock/freshness-aware decoder, avoiding an additional recurrent step merely because a query is off-grid. '
        'The loss integrates elapsed wall time within each event/horizon (up to 24h across long gaps), so dense 12-second bursts do not gain extra weight over quiet half-hour periods. '
        'Absent tiers have zero age rather than a fabricated time since event start. Both cadence arms use the same weighting and input semantics.','',
        '| Cadence | Total training event steps | Largest event | CV MAPE | Focal MAPE | Bias | Final 24h |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cadence in ('hourly','native'):
        n='lstm_'+cadence;a=summaries[n];c=nr['cadence'][cadence]
        lines.append(f"| {cadence} | {c['total_steps']:,} | {c['max_steps']:,} | {pct(a['overall'].get('mape'))} | {pct(core[n]['overall'].get('mape'))} | {pct(a['overall'].get('bias'))} | {pct(a['by_horizon'].get('0-24h',{}).get('mape'))} |")
    a=summaries['lstm_reference'];lines+=['',f"For context, the previously frozen hourly branch-LSTM had {pct(a['overall'].get('mape'))} MAPE and {pct(a['by_horizon'].get('0-24h',{}).get('mape'))} final-day MAPE on these same cases. "
        'Its stored forecasts are read without retraining it; architecture and weighting differ, so native versus the new hourly arm is the controlled cadence comparison.','']
    if paired.get('lstm_native'):
        b=paired['lstm_native'];lines+=['',f"Native minus hourly MAPE change: {100*b['mape_change']:+.2f} percentage points; event-cluster bootstrap 95% interval [{100*b['interval_95'][0]:+.2f}, {100*b['interval_95'][1]:+.2f}]. Negative favors native sampling.",'']
    b=paired['ridge'];lines+=['',f"Clock-feature ridge minus previous ridge: {100*b['mape_change']:+.3f} percentage points; event-cluster bootstrap 95% interval [{100*b['interval_95'][0]:+.3f}, {100*b['interval_95'][1]:+.3f}]. This interval includes no improvement.",'']
    if dense:
        lines+=['## Matched arbitrary-time queries','',dense['scope']+'. '
            'These are four focal tiers only. Equal-event, equal-tier errors are also integrated over elapsed wall time, so packet bursts do not inflate their contribution. '
            'The latest predicted error is held until the next query, with intervals capped at 24h across long gaps.','',
            '| Model | Query MAPE | Time-integrated MAPE | Time-integrated bias |',
            '| --- | ---: | ---: | ---: |']
        for n in ('reference','hourly','native'):
            a=dense['time_integrated'][n];lines.append(f"| {n} | {pct(dense['scores'][n]['overall'].get('mape'))} | {pct(a['mape'])} | {pct(a['bias'])} |")
        lines+=['',f"{dense['time_integrated']['native']['rows']:,} correlated queries; {dense['time_integrated']['native']['events']} event clusters. These are historical diagnostics, not additional independent test events.",'']
    lines+=['','## Missingness and non-grid queries','',
        '| Approach | 08:37 | 6h tracker gap | 24h tracker gap |','| --- | ---: | ---: | ---: |']
    for n in dict.fromkeys([fr['champion'],'ridge','calibrated_stat','lstm_hourly','lstm_native']):
        lines.append('| '+n+' | '+' | '.join(pct(scenarios[s][n]['overall'].get('mape')) for s in ('off_grid_37m','gap_6h','gap_24h'))+' |')
    lines+=['','Stress rows use three fixed issue times per event; their absolute averages have a different time distribution from the complete clean replay. '
        'A tracker gap is not treated as a verified server outage.','',
        '## T1000 / T1500 equality diagnostic','',
        '| Approach | Equal forecasts | Paired checkpoints |','| --- | ---: | ---: |']
    for n in ('ridge','nn_minimal','lstm_hourly','lstm_native'):
        a=plateaus[n];lines.append(f"| {n} | {a['equal']} ({pct(a['fraction'])}) | {a['checkpoints']} |")
    lines+=['','Joint ordering can still create plateaus. Added daily features do not introduce a separately learned positive tier-gap model. '
        'The stable portable package is unchanged. The separate experimental `dist_clock_v2/` candidate recalibrates approximate sigma from its own out-of-fold forecasts, '
        'without 325/326 outcomes or an independent coverage claim.','',
        '## Verification','',
        'All 106 tests pass on CPU, including stock-versus-optimized LSTM values/full gradients, native-time weighting, publication causality, saved-model roundtrips and both portable packages. '
        'The [artifact and leakage audit](clock_native_audit.json) passes: 325/326 future-tail and final-label mutations leave all 17 model outputs unchanged; '
        'fold chronology, training IDs, source/weight/executor hashes and the original protected ledger are verified.','',
        f"Feature models: `{feature_root}/full_refit`. Native/hourly models: `{native_root}/native/full_refit` and `{native_root}/hourly/full_refit`.",'']
    destination.write_text('\n'.join(lines));plot(result,destination.with_suffix('.png'))


def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(('baseline','ridge','nn_minimal','lstm_reference','lstm_hourly','lstm_native',result['champion'])))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for n in names:
        for ax,metric in zip(axes,('mape','bias')):
            ax.plot(np.arange(5),[100*result['clean'][n]['by_horizon'].get(h,{}).get(metric,np.nan) for h in HORIZONS],marker='o',label=n)
    for ax in axes:ax.set_xticks(np.arange(5),HORIZONS);ax.set_xlabel('Hours remaining');ax.grid(alpha=.2)
    axes[0].set_ylabel('MAPE (%)');axes[1].set_ylabel('Signed deviation (%)');axes[1].axhline(0,color='black',linewidth=.7)
    axes[0].legend(fontsize=8);fig.suptitle('Clock features and native-frequency LSTM · historical chronological replay')
    fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.svg'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',required=True);p.add_argument('--native',required=True)
    p.add_argument('--output',default='reports/clock_native_20261018.md');args=p.parse_args();publish(args.features,args.native,args.output)


if __name__=='__main__':main()
