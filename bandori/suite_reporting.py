"""Reports for locked representation/statistical/ensemble experiments."""
import json
import csv
from collections import defaultdict
import math
import os
from pathlib import Path

import numpy as np

from collect_bandori_data import atomic_bytes,atomic_json
from .analog import HORIZONS,horizon


LABELS={'baseline':'Baseline','nn_minimal':'NN: 6 features','nn_standard':'NN: 15 features','nn_rich':'NN: 79 features',
        'nn_pca':'NN: compressed','bagged':'NN: seed average','ridge':'Regularized linear','calibrated_stat':'Completion/pace',
        'trajectory':'Trajectory analogs','hierarchical':'Hierarchical surface','neural_winner':'Best NN on tuning',
        'neural_mean':'NN collection mean','statistical_mean':'Statistical collection mean','collection_mean':'Mixed mean',
        'collection_median':'Mixed median','collection_trimmed':'Mixed trimmed mean'}


def interval_diagnostics(artifact,result):
    """Audit already-frozen bands against test outcomes without fitting anything."""
    rows=[]
    with (Path(artifact)/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['partition']=='test' and row['actual']:rows.append(row)
    diagnostics={}
    for name,bands in result['uncertainty'].items():
        groups=defaultdict(list)
        for row in rows:
            tier=row['tier'];label=HORIZONS[horizon(float(row['hours_remaining']))]
            interval=bands['by_horizon'][label]['by_tier'].get(tier,{})
            fallback=interval.get('log_radius') is None
            if fallback:interval=bands['pooled']['by_tier'].get(tier,{})
            radius=interval.get('log_radius');value=float(row[name]);actual=float(row['actual'])
            groups[tier].append(dict(event_id=int(row['event_id']),horizon=label,supported=radius is not None,
                covered=abs(math.log(value/actual))<=radius if radius is not None else False,
                relative_width=2*math.sinh(radius) if radius is not None else None,pooled_fallback=fallback))
        def aggregate(records):
            supported=[r for r in records if r['supported']];clusters=defaultdict(list)
            for r in supported:clusters[(r['event_id'],r['horizon'])].append(r['covered'])
            return dict(rows=len(records),supported_rows=len(supported),unsupported_rows=len(records)-len(supported),
                row_coverage=sum(r['covered'] for r in supported)/len(supported) if supported else None,
                tier_horizon_event_groups=len(clusters),
                group_coverage=sum(all(v) for v in clusters.values())/len(clusters) if clusters else None,
                mean_relative_width=sum(r['relative_width'] for r in supported)/len(supported) if supported else None,
                pooled_fallback_rows=sum(r['pooled_fallback'] for r in supported))
        by_tier={tier:aggregate(records) for tier,records in groups.items()}
        diagnostics[name]={'by_tier':by_tier,'scope':'Descriptive test coverage; groups within one event are dependent. Bands and predictions were frozen before test inspection.'}
    return diagnostics


def publish(artifact,destination):
    artifact,destination=Path(artifact),Path(destination)
    result=json.loads((artifact/'evaluation.json').read_text());plan=json.loads((artifact/'plan.json').read_text())
    selection=json.loads((artifact/'selection.json').read_text())
    coverage=interval_diagnostics(artifact,result)
    lines=['Representation, statistical and ensemble comparison','',
        f"Tuning-selected champion: **{LABELS[result['champion']]}**. The test cannot change this selection.",'',
        f"Training: {result['training_joint_checkpoints']} joint checkpoints and {result['supervised_training_targets']} supervised tier targets. "+
        'Chronological events: '+', '.join(f'{k}={v}' for k,v in result['split_event_counts'].items())+'.','',
        'All inspected holdouts remain excluded from fitting, normalizers, PCA, calibration-factor fitting and historical references. Three chronological tuning blocks measure stability. Ensemble means/medians use fixed equal family weights; no test-fitted weights.','',
        '| Representation | Total joint input width |','|---|---:|']
    cfg=plan['config'];tau=cfg.get('characteristic_time_days')
    decay=f"exp(-age_days / {tau:g}); {tau/365:g}-year characteristic time" if tau else f"2 ** (-age_days / {cfg['half_life_days']:g}); half-life in days"
    lines[4:4]=[f'Recency weighting: `{decay}`. Applied to historical references, supervised fitting, normalization/PCA and fitted statistical correction factors.','',
        '**Previously inspected test block: this is a follow-up experiment, not a new final test.**' if result.get('test_status')=='previously_inspected_follow_up' else 'Test status: fresh when this experiment was first run.','']
    for rep,width in result['representation_widths'].items():lines.append(f'| {rep} | {width} |')
    lines+=['','Widths include all fixed tiers and explicit missing-tier indicators. The compressed view uses training-only PCA plus direct physical features.','',
            '| Model | Tune MAPE | Test MAPE | Test event bias | Test row bias | Test log RMSE | Test events |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,values in result['partitions']['test'].items():
        m=values['overall'];t=result['partitions']['tune'][name]['overall']
        if m['rows']:lines.append(f"| {LABELS[name]} | {100*t['mape']:.2f}% | {100*m['mape']:.2f}% | {100*m['bias']:+.2f}% | {100*m['row_bias']:+.2f}% | {m['log_rmse']:.3f} | {m['events']} |")
    lines+=['','These are fixed, preselected comparisons. A lower test score does not retroactively select a model. Multiple comparisons and few independent events limit the evidence. MAPE gives events equal weight; row bias weights every tracked tier/checkpoint equally.','',
            '| Family | Winner chosen on tuning |','|---|---|']
    for family,name in result['family_winners'].items():lines.append(f'| {LABELS[family]} | {name} |')
    lines+=['','A baseline winner means all tested corrections failed the declared tuning score/guards. It does not mean that family was omitted.','',
            '| Test horizon | Baseline MAPE | Champion MAPE | Champion row bias | Events |','|---|---:|---:|---:|---:|']
    for h in reversed(HORIZONS):
        b=result['partitions']['test']['baseline']['by_horizon'].get(h,{})
        c=result['partitions']['test'][result['champion']]['by_horizon'].get(h,{})
        if b.get('rows'):lines.append(f"| {h} | {100*b['mape']:.2f}% | {100*c['mape']:.2f}% | {100*c['row_bias']:+.2f}% | {c['events']} |")
    lines+=['','Systems use identical cases within each horizon; tier/event coverage can change across horizons.','',
            '| Champion test label source | Scores | Events | MAPE |','|---|---:|---:|---:|']
    for source,m in result['partitions']['test'][result['champion']]['by_label_source'].items():
        if m['rows']:lines.append(f"| {source} | {m['rows']} | {m['events']} | {100*m['mape']:.2f}% |")
    lines+=['','Post-close proxies are separately identified tracker observations, not independently verified official finals. Reported performance against proxies measures agreement with them.','',
            '| Tier | Training targets | Champion test scores | Champion test events | MAPE |','|---|---:|---:|---:|---:|']
    for tier,count in result['training_targets_by_tier'].items():
        m=result['partitions']['test'][result['champion']]['by_tier'].get(tier,{'rows':0,'events':0})
        error=f"{100*m['mape']:.2f}%" if m['rows'] else 'unavailable'
        lines.append(f"| T{tier} | {count} | {m['rows']} | {m['events']} | {error} |")
    interval=100*(1-plan['config']['interval_alpha'])
    lines+=['',f'Nominal {interval:g}% calibration bands are computed separately per tier and horizon, with a pooled-horizon fallback only when that tier has enough calibration events. Model disagreement alone is never presented as a calibrated band. Exchangeability, missing publication evidence and proxy labels limit coverage claims.','',
            '| Champion test tier | Checkpoint coverage | Whole event/horizon group coverage | Mean width / prediction | Supported checkpoints |',
            '|---|---:|---:|---:|---:|']
    for tier in ('500','1000','1500','2000'):
        value=coverage[result['champion']]['by_tier'].get(tier,{})
        if value.get('supported_rows'):
            lines.append(f"| T{tier} | {100*value['row_coverage']:.1f}% | {100*value['group_coverage']:.1f}% | {100*value['mean_relative_width']:.1f}% | {value['supported_rows']} |")
        else:lines.append(f'| T{tier} | unavailable | unavailable | unavailable | 0 |')
    lines+=['','These are descriptive coverage checks on frozen bands, not new calibration. Whole-group coverage requires every evaluated checkpoint within one tier/event/horizon to be covered; groups from the same event are dependent. Widths and checkpoint coverage alone do not establish a better uncertainty forecast.','',
            '| Model | MAPE change vs baseline (percentage points) | Event-bootstrap 95% interval |','|---|---:|---:|']
    for name in dict.fromkeys(('neural_mean','statistical_mean','collection_mean',result['champion'])):
        value=result['bootstrap'].get(name)
        if value:
            lo,hi=value['interval_95'];lines.append(f"| {LABELS[name]} | {100*value['mape_change']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['','Bootstrap ranges are descriptive and unadjusted for multiple comparisons. Serial dependence also limits interpretation. No further tuning used these test results.','',
            f"Snapshot: `{result['snapshot'].get('bundle')}`.",f"Full candidates and audit forecasts: `{artifact.resolve()}`."]
    atomic_bytes(destination,('\n'.join(lines)+'\n').encode())
    compact={k:v for k,v in result.items() if k!='candidates'}
    compact['candidate_scores']=[{k:v for k,v in c.items() if k!='metrics'} for c in result['candidates']]
    compact['test_interval_diagnostics']=coverage
    atomic_json(destination.with_suffix('.json'),dict(plan=plan,evaluation=compact,selection=selection))
    plot(result,destination.with_suffix('.png'))


def plot(result,destination):
    os.environ.setdefault('MPLCONFIGDIR',str((Path('runs')/'matplotlib_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names=list(dict.fromkeys(['baseline','nn_minimal','nn_standard','nn_rich','nn_pca','neural_mean','statistical_mean',result['champion']]))
    order=list(reversed(HORIZONS));fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col')
    for col,part in enumerate(('tune','test')):
        for row,metric in enumerate(('mape','row_bias')):
            ax=axes[row,col]
            for name in names:
                values=result['partitions'][part][name]['by_horizon']
                y=[100*values[h][metric] if values.get(h,{}).get('rows') else np.nan for h in order]
                ax.plot(range(5),y,marker='o',label=LABELS[name],linewidth=2 if name in ('baseline',result['champion']) else 1.1)
            baseline=result['partitions'][part]['baseline']['by_horizon']
            ax.set_xticks(range(5),[f"{h}\n{baseline.get(h,{}).get('events',0)} events" for h in order],fontsize=8)
            ax.grid(axis='y',alpha=.2);ax.set_ylabel('Event-balanced MAPE (%)' if row==0 else 'Row-mean signed error (%)')
            if row==0:
                test_title='Previously inspected test: follow-up' if result.get('test_status')=='previously_inspected_follow_up' else 'Fresh test: selections already locked'
                ax.set_title('Tuning: three chronological blocks' if col==0 else test_title);ax.legend(frameon=False,fontsize=7,ncol=2)
            else:ax.axhline(0,color='grey',linewidth=.7);ax.set_xlabel('Hours remaining at 08:00 CN')
    fig.suptitle('Representation ablations and model collections',fontweight='bold')
    fig.text(.5,.015,'Within-horizon cases are identical; between-horizon coverage can differ. See label-source and sample-count tables.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(destination,dpi=175);fig.savefig(destination.with_suffix('.svg'));plt.close(fig)
    models=[n for n in result['partitions']['test'] if n!='neural_winner']
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    for ax,metric,title in zip(axes,('mape','row_bias'),('Event-balanced MAPE (%)','Row-mean signed error (%)')):
        values=[100*result['partitions']['test'][n]['overall'][metric] for n in models]
        colors=['#df8f27' if n==result['champion'] else '#4679a6' for n in models]
        ax.barh(range(len(models)),values,color=colors);ax.set_yticks(range(len(models)),[LABELS[n] for n in models])
        ax.set_title(title);ax.axvline(0,color='grey',linewidth=.6);ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    test_label='previously inspected' if result.get('test_status')=='previously_inspected_follow_up' else 'fresh'
    fig.suptitle(f'Preselected systems on the same {test_label} test cases; orange = tuning champion')
    fig.tight_layout();fig.savefig(Path(destination).with_name(Path(destination).stem+'_models.png'),dpi=175);plt.close(fig)
