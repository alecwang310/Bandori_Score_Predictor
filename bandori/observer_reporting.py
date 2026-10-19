"""Matched observer ablation tables and standalone horizon plots."""
import json
from pathlib import Path

import numpy as np

from collect_bandori_data import atomic_json
from .analog import HORIZONS
from .experiments import FAMILIES


def pct(value):return '—' if value is None else f'{100*value:.2f}%'


def publish(artifact,destination):
    artifact=Path(artifact);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    result=json.loads((artifact/'evaluation.json').read_text());selection=json.loads((artifact/'selection.json').read_text())
    plan=json.loads((artifact/'plan.json').read_text());test=result['partitions']['test'];tune=result['partitions']['tune']
    parent=result['control_champion'];champion=result['champion'];neural=result['neural_winner']
    lines=['# Causal 24-hour observer: paired model comparison','',
        '**Previously inspected chronological test. These are follow-up results, not fresh validation.**','',
        f"Training uses {result['split_event_counts']['train']} events, tuning {result['split_event_counts']['tune']}, independent calibration {result['split_event_counts']['calibration']}, and test {result['split_event_counts']['test']}. "
        f"There are {result['supervised_training_targets']:,} supervised training targets at {result['training_joint_checkpoints']:,} joint 08:00 checkpoints. "
        'Both arms use the same frozen split, training-only historical references, target masks, 730-day exponential age weighting, and scored cases. The protected-event ledger is unchanged.','',
        '## What changed','',
        'The observer reads the last 24 hourly score increments and the corresponding previous-day hours. Its channels expose normalized rate, within-day shape, day-to-day changes, availability, known active exposure, negative corrections, clock phase, and a historical activity prior. '
        'It supplies 23 summary features per tier to every existing neural representation, the seed bag, joint ridge and hierarchical regression. Completion/pace and trajectory models receive a clock-adjusted historical 24-hour pace estimate. A flat-clock variant isolates that adjustment from simply observing a full day.','',
        'The new joint temporal network shares two causal width-three convolutions across tiers (dilations 1 and 3; 4 or 8 channels). '
        'It pools the full day and first/last six hours, concatenates these embeddings with the standard inputs and observer summaries, and emits all tier corrections together through an eight-unit head. '
        'Corrections remain bounded changes to remaining points with the existing signed-bias loss. Missing target heads have no supervised weight and emit zero correction.','',
        'Hourly interpolation uses only observations published by issue time and refuses gaps over two hours. Negative increments and known downtime are exposed separately. '
        'The historical activity prior uses earlier training references only, excludes their opening 12h and closing 24h, and shrinks sparse hourly estimates toward a constant rate. '
        'This helps separate ordinary daily cycles from opening and closing surges. Future outages, schedule revisions, observations and the current event final never enter inputs.','',
        '## Frozen selection','',
        f"Tuning selected **{champion}**, compared with the prior champion **{parent}**. The tuning-selected neural family is **{neural}**. "
        'All choices were saved before calibration/test observer features were generated. Selection uses the existing three-block stable score, signed-bias guards, and a final-24h error guard against each paired control. Zero observer gain retains that control.','',
        'The direct ablation below retains each old NN/ridge/hierarchical gain, uses a fixed 0.5 blend for statistical models, and uses the first declared temporal width at gain 0.5. '
        'These arms are reported even if tuning rejects them; they are not selected using test outcomes. The guarded column reports the independently tuning-selected observer arm.','',
        '## Direct paired ablation','',
        '| Family | Tuning old | Tuning +24h, fixed | Test old | Test +24h, fixed | Test change (pp) | Test guarded | Fixed bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for family in ('baseline',*FAMILIES,'tcn24','pace24_flat'):
        paired='neural_winner' if family=='tcn24' else 'baseline' if family=='pace24_flat' else family
        old=test['control'][paired]['overall'];raw=test['observer_raw'][family]['overall'];guarded=test['observer'][family]['overall']
        old_t=tune['control'][paired]['overall'];raw_t=tune['observer_raw'][family]['overall']
        lines.append(f"| {family} | {pct(old_t.get('mape'))} | {pct(raw_t.get('mape'))} | {pct(old.get('mape'))} | {pct(raw.get('mape'))} | {100*(raw['mape']-old['mape']):+.2f} | {pct(guarded.get('mape'))} | {pct(raw.get('bias'))} |")
    lines+=['','For tcn24 the paired control is the old tuning-selected neural winner; this is an architecture-plus-input comparison. '
        'For pace24_flat the control is the old baseline. Negative changes favor the observer.','',
        '## Fixed collections','',
        '| Collection | Test old MAPE | Test observer MAPE | Old bias | Observer bias |','| --- | ---: | ---: | ---: | ---: |']
    for name in selection['ensembles']:
        a=test['control'][name]['overall'];b=test['observer'][name]['overall']
        lines.append(f"| {name} | {pct(a['mape'])} | {pct(b['mape'])} | {pct(a['bias'])} | {pct(b['bias'])} |")
    lines+=['','Collection memberships and equal weighting rules are unchanged. The mixed collection uses the neural winner selected on tuning, which may now be temporal.','',
        '## Selected champion by hours remaining','',
        '| Hours remaining | Old champion MAPE | New champion MAPE | Old bias | New bias | Scored targets |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for h in HORIZONS:
        a=test['control'][parent]['by_horizon'].get(h,{});b=test['observer'][champion]['by_horizon'].get(h,{})
        lines.append(f"| {h} | {pct(a.get('mape'))} | {pct(b.get('mape'))} | {pct(a.get('bias'))} | {pct(b.get('bias'))} | {b.get('rows',0)} |")
    lines+=['','## Tier coverage','',
        '| Tier | Supervised training targets | Test targets | Old champion MAPE | New champion MAPE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for tier,count in result['training_targets_by_tier'].items():
        a=test['control'][parent]['by_tier'].get(tier,{});b=test['observer'][champion]['by_tier'].get(tier,{})
        lines.append(f"| T{tier} | {count} | {b.get('rows',0)} | {pct(a.get('mape'))} | {pct(b.get('mape'))} |")
    lines+=['','T1500 has no supervised training targets in this frozen strict split; it cannot be claimed to improve here. '
        'The later full refit supports that tier but is unsuitable for testing events already in its training set.','',
        '## Data and uncertainty limits','',
        'Per-event clustered paired bootstrap intervals are descriptive and available in the JSON. The repeated test, two-event comparisons, changing tier coverage, and closing-tracker proxy labels limit generalization. '
        'No intervals were fitted to test errors, and no probability-band claims are made. The interval-calibration partition is reported only; its outcomes do not train or select the observer. '
        'An unrecorded tracker gap does not establish a server outage. At fixed 08:00 issue time, much clock phase is constant; improvements from within-day velocity shape should not automatically be attributed to calendar seasonality.','',
        'Hourly coverage by partition:','',
        '| Partition | Cases | Full 24h available | Mean hourly coverage | At least 8h available |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,v in result['coverage'].items():lines.append(f"| {name} | {v['cases']} | {pct(v['full_day_fraction'])} | {pct(v['mean'])} | {pct(v['at_least_8h_fraction'])} |")
    lines+=['','## Reproduce','',
        '```sh',f"python3 -m bandori.observer_experiment train --data {result['snapshot'].get('bundle','DATA')} --source-model {plan['source']} --output runs/observer_24h_new",'```','',
        f"Artifact: `{artifact}`. Per-checkpoint paired forecasts: `{artifact/'predictions.csv'}`. "
        'Exact source hashes, predeclared design, split and locked choices are saved with the fit.','']
    destination.write_text('\n'.join(lines));atomic_json(destination.with_suffix('.json'),result)
    plot(result,selection,destination.with_suffix('.png'))
    return result


def plot(result,selection,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test=result['partitions']['test'];parent=result['control_champion'];champion=result['champion']
    x=np.arange(5);labels=['0–24','24–48','48–72','72–96','96+']
    def values(system,key):return [100*system['by_horizon'].get(h,{}).get(key,np.nan) for h in HORIZONS]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    systems=[('Old champion',test['control'][parent],'#444444','--'),
             ('Observer champion',test['observer'][champion],'#1b9e77','-'),
             ('Old neural winner',test['control']['neural_winner'],'#7570b3','--'),
             ('Fixed 24h temporal net',test['observer_raw']['tcn24'],'#e67e22','-')]
    for row,key in enumerate(('mape','bias')):
        for label,system,color,style in systems:axes[row,0].plot(x,values(system,key),style,marker='o',color=color,label=label)
        axes[row,0].set_title('Error by remaining hours' if key=='mape' else 'Signed deviation by remaining hours')
        axes[row,0].set_ylabel('MAPE (%)' if key=='mape' else 'Mean signed error (%)')
        axes[row,0].set_xticks(x,labels);axes[row,0].set_xlabel('Hours remaining');axes[row,0].grid(alpha=.2)
        if key=='bias':axes[row,0].axhline(0,color='black',linewidth=.7)
        families=('baseline','nn_minimal','nn_standard','nn_rich','nn_pca','bagged','ridge','calibrated_stat','trajectory','hierarchical')
        delta=[100*(test['observer_raw'][n]['overall'][key]-test['control'][n]['overall'][key]) for n in families]
        axes[row,1].barh(families,delta,color=['#1b9e77' if v<0 else '#d95f02' for v in delta]);axes[row,1].axvline(0,color='black',linewidth=.7)
        axes[row,1].set_title('Fixed observer minus paired control');axes[row,1].set_xlabel('MAPE change (percentage points)' if key=='mape' else 'Signed bias change (percentage points)')
    axes[0,0].legend(fontsize=8);fig.suptitle('24-hour observer · previously inspected chronological test')
    fig.savefig(destination,dpi=160);fig.savefig(Path(destination).with_suffix('.svg'));plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for ax,name in zip(axes.flat,(*families,'tcn24','pace24_flat')):
        paired='neural_winner' if name=='tcn24' else 'baseline' if name=='pace24_flat' else name
        ax.plot(x,values(test['control'][paired],'mape'),'o--',color='#666666',label='Control')
        ax.plot(x,values(test['observer_raw'][name],'mape'),'o-',color='#377eb8',label='Fixed +24h')
        ax.plot(x,values(test['observer'][name],'mape'),':',color='#1b9e77',label='Tuning guarded')
        ax.set_title(name);ax.set_xticks(x,labels);ax.set_xlabel('Hours remaining');ax.set_ylabel('MAPE (%)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Paired family ablations · repeated test, not fresh validation')
    p=Path(destination);fig.savefig(p.with_name(p.stem+'_families.png'),dpi=160);plt.close(fig)


def publish_refit(data,model,destination,forecast_dir,benchmark='benchmarks/event_325_points.json'):
    """Descriptive paired full-bank comparison at the user's exact issue times."""
    from collect_bandori_data import timestamp, write_csv
    from .observer_refit import predict
    from .pipeline import digest
    model=Path(model);destination=Path(destination);forecast_dir=Path(forecast_dir)
    benchmark=json.loads(Path(benchmark).read_text());metadata=json.loads((model/'model.json').read_text())
    selection=json.loads((model/'selection.json').read_text());eid=benchmark['event_id'];tiers=benchmark['tier_order']
    if eid in metadata['training_event_ids'] or eid not in metadata['excluded_event_ids']:raise ValueError('Event must be explicitly excluded')
    if abs(data.events[eid].end-timestamp(benchmark['target_end']))>2:raise ValueError('Closing timestamp differs')
    rows=[];points=[]
    for issue in benchmark['issues']:
        cutoff=timestamp(issue['issued_at'])
        if abs((data.events[eid].end-cutoff)/3600-issue['hours_remaining'])>2/3600:raise ValueError('Hours remaining mismatch')
        print('Paired full-bank observer replay:',issue['hours_remaining'],'hours remaining',flush=True)
        result=predict(data,model,eid,cutoff,max_stale_hours=48,all_models=True)
        path=forecast_dir/f"event_{eid}_h{issue['hours_remaining']}.json"
        if path.exists():
            if json.loads(path.read_text())!=result:raise ValueError('Fixed replay differs')
        else:atomic_json(path,result)
        by_tier={r['tier']:r for r in result['predictions']}
        for tier in tiers:
            row=by_tier[tier]
            if row['status']!='ok':raise ValueError('Required tier unavailable')
            truth=data.labels.get((eid,tier))
            if not truth:raise ValueError('Comparison outcome unavailable')
            for arm in ('control','observer_raw','observer'):
                for name,value in row[arm].items():
                    rows.append(dict(event_id=eid,tier=tier,issued_at=issue['issued_at'],hours_remaining=issue['hours_remaining'],
                        arm=arm,model=name,prediction=value,actual=truth,stale_input=row['stale_input'],
                        signed_error=value/truth-1,absolute_error=abs(value/truth-1)))
        points.append(dict(hours_remaining=issue['hours_remaining'],issued_at=issue['issued_at'],
            predictions={str(t):by_tier[t]['final'] for t in tiers},stale=any(by_tier[t]['stale_input'] for t in tiers),sha256=digest(path)))
    def metrics(arm,name,fresh=False):
        rr=[r for r in rows if r['arm']==arm and r['model']==name and (not fresh or not r['stale_input'])]
        return dict(rows=len(rr),mape=float(np.mean([r['absolute_error'] for r in rr])),bias=float(np.mean([r['signed_error'] for r in rr])))
    names=('baseline',*FAMILIES,'tcn24','pace24_flat',*selection['ensembles'])
    comparison={}
    for name in names:
        parent='neural_winner' if name=='tcn24' else 'baseline' if name=='pace24_flat' else name
        comparison[name]={}
        for scope,fresh in (('all',False),('fresh',True)):
            comparison[name][scope]=dict(control=metrics('control',parent,fresh),observer=metrics('observer',name,fresh))
            if name in selection['paired_raw']:comparison[name][scope]['observer_raw']=metrics('observer_raw',name,fresh)
    champion=selection['champion'];parent=selection['control_champion']
    report=dict(event_id=eid,training_events=len(metadata['training_event_ids']),excluded_event_ids=metadata['excluded_event_ids'],
        supervised_training_targets=metadata['supervised_training_targets'],training_targets_by_tier=metadata['training_targets_by_tier'],
        champion=champion,control_champion=parent,comparison=comparison,points=points,
        champion_metrics={scope:dict(control=metrics('control',parent,fresh),observer=metrics('observer',champion,fresh))
                          for scope,fresh in (('all',False),('fresh',True))},
        limitation='Previously inspected, extra-long event 325; retrospective case study. Two stale timestamps excluded identically in fresh-only comparison.',
        model_sha256=digest(model/'model.json'))
    destination.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(destination.with_suffix('.json'),report);write_csv(destination.with_suffix('.csv'),list(rows[0]),rows)
    lines=['# Full-bank 24-hour observer refit and event 325 case study','',
        f"Refitted on **{report['training_events']} events**, excluding **{metadata['excluded_event_ids']}**, with **{metadata['supervised_training_targets']:,} supervised targets**. "
        'Two-year characteristic age weighting is unchanged. All architecture choices, gains, ensembles and the champion were frozen from the chronological observer experiment. '
        'No event-325 outcome selected or tuned this refit. Earlier protected events were reused under the prior explicit all-except-325/326 instruction; the ledger remains unchanged.','',
        '**This is a retrospective case study on a previously inspected, unusually long event. It is not a fresh test.** '
        'The exact quoted issue times are used. At 146h and 123h remaining the last score is stale; no future observation fills those gaps. '
        'The fresh-only table removes the same eight tier/time points from every arm. Closing labels are tracker proxies.','',
        '## Paired errors on fresh inputs only','',
        '| Family | Old MAPE | Fixed +24h MAPE | Guarded +24h MAPE | Old bias | Guarded bias |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        c=comparison[name]['fresh'];a=c['control'];b=c['observer'];raw=c.get('observer_raw',{})
        lines.append(f"| {name} | {pct(a['mape'])} | {pct(raw.get('mape'))} | {pct(b['mape'])} | {pct(a['bias'])} | {pct(b['bias'])} |")
    lines+=['','tcn24 is paired with the old neural winner; pace24_flat is paired with the baseline. '
        'Fixed observer arms retain old gains (and use a predeclared 0.5 statistical blend); guarded arms reflect only the earlier tuning selection. '
        'All errors here are equally weighted across the same four tiers and exact matched timestamps.','',
        '## Single selected point forecast per tier','',
        '| Hours remaining | T500 | T1000 | T1500 | T2000 |', '| ---: | ---: | ---: | ---: | ---: |']
    for row in points:lines.append(f"| {row['hours_remaining']}{' †' if row['stale'] else ''} | "+' | '.join(f"{row['predictions'][str(t)]:,}" for t in tiers)+' |')
    lines+=['','† Last-known-score forecast, marked stale.','',
        'The selected champion remains **'+champion+'**. On fresh inputs, its MAPE changes from '+pct(report['champion_metrics']['fresh']['control']['mape'])+' to '+pct(report['champion_metrics']['fresh']['observer']['mape'])+'. '
        'Including the two stale checkpoints changes MAPE from '+pct(report['champion_metrics']['all']['control']['mape'])+' to '+pct(report['champion_metrics']['all']['observer']['mape'])+'.','',
        'T1500 has '+str(metadata['training_targets_by_tier']['1500'])+' supervised training targets in this full refit. '
        'Support is still sparse, especially for events as long as 325. Event 326 is ongoing and cannot supply an accuracy score.','']
    destination.write_text('\n'.join(lines))
    return report
