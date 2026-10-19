"""Readable robustness results from the fixed rule-aware replay protocol."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from collect_bandori_data import atomic_json
from .analog import HORIZONS
from .evaluation import breakdown, paired_bootstrap, summary
from .pipeline import digest


def pct(v):return '—' if v is None else f'{100*v:.2f}%'


def publish(directory,destination):
    directory=Path(directory);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    result=json.loads((directory/'evaluation.json').read_text());plan=json.loads((directory/'plan.json').read_text())
    rows=[]
    for fold in plan['folds']:
        with (directory/fold['name']/'predictions.csv').open(encoding='utf-8-sig') as f:rows.extend(csv.DictReader(f))
    labels={};sources={}
    for r in rows:
        if r['actual']:
            key=(int(r['event_id']),int(r['tier']));labels[key]=float(r['actual']);sources[key]=r['label_source']
    data=SimpleNamespace(labels=labels,label_sources=sources)
    names=list(result['scores']);scores={};scenarios=sorted({r['scenario'] for r in rows})
    for scenario in scenarios:
        rr=[r for r in rows if r['scenario']==scenario]
        cases=[SimpleNamespace(event_id=int(r['event_id']),tier=int(r['tier']),remaining_hours=float(r['hours_remaining'])) for r in rr]
        scores[scenario]={n:summary(cases,np.array([float(r[n]) for r in rr]),data) for n in names}
    clean=[r for r in rows if r['scenario']=='clean'];cases=[SimpleNamespace(event_id=int(r['event_id']),tier=int(r['tier']),remaining_hours=float(r['hours_remaining'])) for r in clean]
    clean_map={(r['event_id'],r['tier'],r['checkpoint']):r for r in clean}
    matched_gaps={}
    for scenario in ('gap_6h','gap_24h'):
        stress=[r for r in rows if r['scenario']==scenario and (r['event_id'],r['tier'],r['checkpoint']) in clean_map]
        controls=[clean_map[(r['event_id'],r['tier'],r['checkpoint'])] for r in stress]
        cc=[SimpleNamespace(event_id=int(r['event_id']),tier=int(r['tier']),remaining_hours=float(r['hours_remaining'])) for r in stress]
        matched_gaps[scenario]={n:dict(clean=summary(cc,np.array([float(r[n]) for r in controls]),data),
                                       gap=summary(cc,np.array([float(r[n]) for r in stress]),data)) for n in names}
    predictions={n:np.array([float(r[n]) for r in clean]) for n in names}
    core_indices=[i for i,c in enumerate(cases) if c.tier in (500,1000,1500,2000)]
    core={n:summary([cases[i] for i in core_indices],p[core_indices],data) for n,p in predictions.items()}
    bootstrap={n:paired_bootstrap(cases,predictions['baseline'],p,data,1729) for n,p in predictions.items() if n!='baseline'}
    types={r['event_id']:r['event_type'] for r in clean}
    by_type={n:breakdown(cases,p,data,lambda c:types[str(c.event_id)]) for n,p in predictions.items()}
    champion=result['champion']
    guard_failures={}
    for name in names:
        failures=[]
        for fold_name,fold in result['folds'].items():
            a,b=fold['clean'][name],fold['clean']['baseline']
            for key in ('bias','row_bias'):
                if abs(a['overall'][key])>abs(b['overall'][key])+.0025:failures.append(fold_name+': '+key)
            late,base=a['by_horizon'].get('0-24h',{}),b['by_horizon'].get('0-24h',{})
            if base.get('rows') and (not late.get('rows') or late['mape']>base['mape']+.0025):failures.append(fold_name+': final-24h MAPE')
        guard_failures[name]=failures
    all_times_path=directory/'all_timestamp_evaluation.json'
    all_times=json.loads(all_times_path.read_text()) if all_times_path.exists() else None
    compact=dict(champion=champion,protocol=plan,clean=scores['clean'],core_tiers=core,scenarios={k:v for k,v in scores.items() if k!='clean'},
                 by_type=by_type,paired_bootstrap=bootstrap,folds={n:dict(post_rule_training_events=f['post_rule_training_events'],validation_events=f['validation_events'],coverage=f['coverage']) for n,f in result['folds'].items()},
                 guard_failures=guard_failures,matched_gap_clean=matched_gaps,all_timestamp_lstm=all_times,limitations=result['limitations'])
    atomic_json(destination.with_suffix('.json'),compact)
    lines=['# Rule-aware models: 3x post-change weighting and robustness','',
        '**Historical cross-validation on twelve previously inspected post-change events; not a fresh test.** '
        'Every fold trains only on outcomes available before that validation block starts. Events 325/326 never enter fitting or selection. '
        'Earlier protected events are reused under the explicit all-except-325/326 instruction; the ledger is retained.','',
        '## Changes applied to every approach','',
        'The age function stays `exp(-age_days / 730)`. Events **starting on or after 2026-05-26 receive exactly 3x weight**, multiplicatively, in supervised fitting, weighted normalizers/PCA, historical analogs, daily activity priors and train-only statistical corrections. '
        'Event IDs do not determine rule era: events 312/313 are pre-change. Saved older fits retain their original configuration and numerical forecasts.','',
        'All approaches receive the actual event type and dated reward metadata. The voiced-stamp boundary moved T1000→T1500; T500 gained an animated voiced stamp and T1000 a flashing title. '
        '[Official May announcement](https://www.sina.cn/news/detail/5300754143514601.html). '
        'T2000 gained a profile decoration from April 7. [Official April announcement](https://www.sina.cn/news/detail/5285051785020053.html). '
        'These are changes in rewards and incentives; no point-formula change is asserted.','',
        'Sparse new T1500 histories may borrow completion/pace *shapes* from recent legacy T1000 reward-boundary histories. '
        'Those T1000 scores remain T1000 labels and never become artificial T1500 supervised targets. '
        'The direct same-tier reference count and role-transfer support are explicit. Input columns that never vary in training are masked, so unseen tiers or rule flags do not activate never-trained random connections.','',
        'The raw LSTM keeps its raw sequence, small growth summaries, event type and joint output design. Reward metadata is added alongside those inputs. '
        'Independent fractional-time branches and output heads are batched without changing the recurrent computation. Values and parameter gradients are checked against the reference implementation. '
        'It trains for the same fixed 120 epochs, with the same bias objective and 730-day decay, using CPU for this replay.','',
        '## Chronological protocol','',
        '| Fold | Post-change training events | Validation events |','| --- | ---: | --- |']
    for name,f in result['folds'].items():lines.append(f"| {name} | {f['post_rule_training_events']} | {', '.join(map(str,f['validation_events']))} |")
    lines+=['','No architecture, gain or ensemble membership is selected using a validation-block outcome. '
        'The final family is chosen from these cross-validation results using the predeclared error-plus-bias score, a fold-stability penalty, and paired baseline bias/final-day guards. '
        'Stress scenarios are diagnostics only and cannot change that choice.','',
        '## Clean replay results','',
        '| Approach | All-tier MAPE | T500/1000/1500/2000 MAPE | Signed bias | Worst fold MAPE | Final 24h MAPE | Bias / late guards |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for n in names:
        a=scores['clean'][n];worst=max(f['clean'][n]['overall']['mape'] for f in result['folds'].values())
        guard='pass' if result['scores'][n]['admissible'] else 'fail'
        lines.append(f"| {n}{' **selected**' if n==champion else ''} | {pct(a['overall'].get('mape'))} | {pct(core[n]['overall'].get('mape'))} | {pct(a['overall'].get('bias'))} | {pct(worst)} | {pct(a['by_horizon'].get('0-24h',{}).get('mape'))} | {guard} |")
    lines+=['','Each arm uses identical scored cases. Errors are event-balanced. The four focal tiers are shown separately because averages over many deeper tiers can conceal a T1000/T1500 problem.','',
        f"The selected family is **{champion}**. Guard failures for the raw LSTM: "+('; '.join(guard_failures['raw_lstm']) or 'none')+'. '
        'A lower overall error does not override a failed signed-bias or final-day guard; the tolerance is 0.25 percentage points relative to the completion baseline in each fold.','',
        '## Adaptation across the rule change','',
        '| Approach | No new-rule examples | Four new-rule examples | Eight new-rule examples |','| --- | ---: | ---: | ---: |']
    order=[f['name'] for f in plan['folds']]
    for n in names:lines.append('| '+n+' | '+' | '.join(pct(result['folds'][f]['clean'][n]['overall'].get('mape')) for f in order)+' |')
    lines+=['','Three-times weighting cannot supply evidence before the first new-rule outcome. A large cold-start error should not be hidden by averaging it with later adaptation.','',
        '## Missingness and arbitrary-time robustness','',
        '| Approach | 08:37 queries | 6h tracker gap | 24h tracker gap |','| --- | ---: | ---: | ---: |']
    for n in names:lines.append('| '+n+' | '+' | '.join(pct(scores.get(s,{}).get(n,{}).get('overall',{}).get('mape')) for s in ('off_grid_37m','gap_6h','gap_24h'))+' |')
    lines+=['','The primary clean table includes all daily checkpoints; stress rows use three selected checkpoints per event. To measure degradation, compare each gap with its identical tier/time clean intersection:','',
        '| Approach | Clean → 6h gap MAPE | Clean → 24h gap MAPE |','| --- | ---: | ---: |']
    for n in dict.fromkeys(('baseline',champion,'raw_lstm')):
        cells=[]
        for scenario in ('gap_6h','gap_24h'):
            pair=matched_gaps[scenario][n]
            cells.append(pct(pair['clean']['overall'].get('mape'))+' → '+pct(pair['gap']['overall'].get('mape')))
        lines.append('| '+n+' | '+' | '.join(cells)+' |')
    if all_times:
        lines+=['','### Dense raw-LSTM timestamp check','',
            '| Inputs / model | Focal-tier MAPE | Signed bias | Events | Scored rows |',
            '| --- | ---: | ---: | ---: | ---: |']
        for title,key in (('Raw LSTM, all query times','lstm'),('Raw LSTM, scores at most 1h old','fresh_lstm'),('Linear remaining-score anchor','linear_anchor'),('Raw LSTM, supported heads only','supported_heads_lstm'),('Linear anchor, same supported heads','supported_heads_linear')):
            a=all_times[key]['overall']
            lines.append(f"| {title} | {pct(a.get('mape'))} | {pct(a.get('bias'))} | {a.get('events',0)} | {a.get('rows',0)} |")
        lines+=['',f"Query times include original focal-tier tracker timestamps and the hourly grid. {all_times['initially_unsupported_head_rows']:,} rows use a T1500 head with no previous supervised examples, so that head retains its linear anchor before joint rank reconciliation. "
            'Dense observations are correlated within twelve events, and their time distribution differs from the primary matched-clock comparison. This diagnostic cannot be compared as an identical-case gain over the other families.','']
        a,b=all_times['lstm']['overall'],all_times['linear_anchor']['overall']
        lines+=[f"Dense-query WAPE is {pct(a['wape'])} for the LSTM versus {pct(b['wape'])} for the linear anchor; log RMSE is {a['log_rmse']:.3f} versus {b['log_rmse']:.3f}. "
            'Lower mean relative error therefore does not establish improvement under every loss or on high-score outliers.','']
    lines+=['','The gap experiments remove the target event’s observations after the declared last-known time; they do not fabricate a game-server outage or fill the gap with later scores. '
        'Freshness limits are explicitly relaxed for the stale-input diagnostic, with score age retained as an input. '
        '08:37 tests are branch-time checks, not a claim of validation at every possible hour. The raw LSTM supports any issue timestamp by construction and trains on hourly plus fractional-hour queries.','',
        '## Event-type behavior','',
        '| Type | Events | Selected MAPE | Selected bias | Raw LSTM MAPE | Raw LSTM bias |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind,a in by_type[champion].items():
        b=by_type['raw_lstm'].get(kind,{})
        lines.append(f"| {kind} | {a.get('events',0)} | {pct(a.get('mape'))} | {pct(a.get('bias'))} | {pct(b.get('mape'))} | {pct(b.get('bias'))} |")
    lines+=['','Medley and Festival have one post-change validation event each. Their apparent performance is not a stable estimate of an entire event type. '
        'Story events have no recent samples. Group differences also reflect event bonuses and popularity.','',
        '## Selected model by tier','',
        '| Tier | MAPE | Signed bias | Events |','| --- | ---: | ---: | ---: |']
    for tier,a in scores['clean'][champion]['by_tier'].items():lines.append(f"| T{tier} | {pct(a.get('mape'))} | {pct(a.get('bias'))} | {a.get('events',0)} |")
    focal1000=scores['clean'][champion]['by_tier'].get('1000',{})
    lines+=['',f"T1000 remains a material weakness: {pct(focal1000.get('mape'))} MAPE and {pct(focal1000.get('bias'))} signed bias. "
        'The selection guards operate on the declared all-tier objective; they do not establish unbiasedness at every individual tier. '
        'The challenge-type sample is also small and runs low. These limitations remain visible rather than being repaired using the validation outcomes.','']
    comparison_path=destination.parent/'rule_aware_3x_event325.json'
    if comparison_path.exists() and (directory/'model.json').exists() and json.loads(comparison_path.read_text()).get('source_model_sha256')==digest(directory/'model.json'):
        comparison=json.loads(comparison_path.read_text())
        lines+=['','## Frozen event-325 comparison','',
            'The family choice was fixed before this retrospective case study. The 373-hour event exceeds every training-event duration, and two issue times have stale tracker inputs. '
            'This event cannot establish general performance. [Exact single-point forecasts and matched comparisons](rule_aware_3x_event325.md).','',
            '| Approach | All 32 tier/time points | Same 24 fresh-input points |','| --- | ---: | ---: |']
        for n in ('Ours','MASHIRO','RUI'):
            a=comparison['accuracy'][n]['mape_percent'];b=comparison['accuracy_fresh_inputs_only'][n]['mape_percent']
            lines.append(f'| {n} | {a:.2f}% | {b:.2f}% |')
        lines+=['','Nanami is evaluated only at its two supplied issue times in the linked report; its smaller sample is not mixed with the eight-time comparison.','']
    audit_path=destination.parent/'rule_aware_3x_audit.json'
    if audit_path.exists() and (directory/'model.json').exists() and json.loads(audit_path.read_text()).get('model_sha256')==digest(directory/'model.json'):
        audit=json.loads(audit_path.read_text())
        lines+=['','## Artifact and leakage audit','',
            f"[Saved-model audit](rule_aware_3x_audit.json): {sum(audit['checks'].values())}/{len(audit['checks'])} checks passed. "
            'All eighteen approaches retain identical forecasts when future target-event scores and its own final labels are multiplied by 1000. '
            'Fold chronology, reference prefixes, genuine T1500 targets, numerical artifact hashes and the unchanged protected-event ledger are checked. '
            'Unit tests cover recurrent branch values/gradients and train-only support masks.','']
    lines+=['','Event-clustered paired bootstrap intervals against the completion baseline are in the JSON. Twelve weekly event clusters, several overlapping reward changes and repeated historical inspection limit the strength of any robustness claim. '
        'No prediction intervals are fitted on validation errors. The unusually long event 325 remains a separate retrospective case study.','',
        'The combined follow-up changes reward metadata, reference support and training weights together. It does not isolate a causal accuracy gain from the 3x multiplier alone.','',
        f"Production weights: `{directory/'full_refit'}`. Selection: `{directory/'selection.json'}`. Per-checkpoint audit rows remain in each fold’s `predictions.csv`.",'']
    destination.write_text('\n'.join(lines));plot(compact,result,destination.with_suffix('.png'))


def plot(compact,result,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['baseline','calibrated_stat','trajectory','raw_lstm',compact['champion']];names=list(dict.fromkeys(names))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');x=np.arange(5)
    for n in names:
        for ax,key in zip(axes[0],('mape','bias')):
            ax.plot(x,[100*compact['clean'][n]['by_horizon'].get(h,{}).get(key,np.nan) for h in HORIZONS],marker='o',label=n)
            ax.set_xticks(x,HORIZONS);ax.set_xlabel('Hours remaining');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('MAPE (%)');axes[0,1].set_ylabel('Signed deviation (%)');axes[0,1].axhline(0,color='black',linewidth=.7)
    for n in names:
        axes[1,0].plot(np.arange(3),[100*result['folds'][f]["clean"][n]['overall']['mape'] for f in result['folds']],marker='o',label=n)
        scenarios=('off_grid_37m','gap_6h','gap_24h')
        axes[1,1].plot(np.arange(3),[100*compact['scenarios'][s][n]['overall']['mape'] for s in scenarios],marker='o',label=n)
    axes[1,0].set_xticks(np.arange(3),['0 post-rule','4 post-rule','8 post-rule']);axes[1,0].set_xlabel('New-rule training support');axes[1,0].set_ylabel('MAPE (%)')
    axes[1,1].set_xticks(np.arange(3),['08:37','6h gap','24h gap']);axes[1,1].set_ylabel('MAPE (%)')
    for ax in axes[1]:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('CN rule-aware models · 3× post-May-26 weight · historical cross-validation')
    fig.savefig(destination,dpi=160);fig.savefig(Path(destination).with_suffix('.svg'));plt.close(fig)
