"""Publish frozen results; reporting never fits or selects a model."""
import json
import os
from pathlib import Path

from collect_bandori_data import atomic_bytes,atomic_json
from .analog import HORIZONS


NAMES={'baseline':'Simple baseline','baseline_nn':'Baseline + NN','statistical':'Statistical model'}


def write_report(artifact,destination):
    artifact,destination=Path(artifact),Path(destination)
    result=json.loads((artifact/'evaluation.json').read_text())
    plan=json.loads((artifact/'plan.json').read_text())
    split=plan['split'];nn=result['selection']['baseline_nn'];stat=result['selection']['statistical']
    lines=['Two competing CN cutoff predictors','',
        f"NN: {nn['hidden']} hidden units, correction gain {nn['gain']}. Statistical model: pace weight {stat['alpha']}, training-only calibration {'enabled' if stat['calibrated'] else 'disabled'}.",'',
        f"Training: {result['joint_training_checkpoints']} joint checkpoints, {result['supervised_training_targets']} supervised tier targets. Event splits: "+', '.join(f'{p}={len(split[p])}' for p in ('train','tune','calibration','test'))+'.','',
        'All tiers from an event share a split. Previous holdouts stay excluded from training and the frozen reference bank. Both systems were selected before opening a fresh final-test block.','',
        '| Partition | System | MAPE | Event-balanced bias | Row-mean bias | Log RMSE | Scores | Events |','|---|---|---:|---:|---:|---:|---:|---:|']
    for part in ('tune','test'):
        for name,values in result['partitions'][part].items():
            m=values['overall']
            if not m['rows']:continue
            lines.append(f"| {part} | {NAMES[name]} | {100*m['mape']:.2f}% | {100*m['bias']:+.2f}% | {100*m['row_bias']:+.2f}% | {m['log_rmse']:.3f} | {m['rows']} | {m['events']} |")
    lines+=['','MAPE and event-balanced bias give every event equal weight. Row-mean bias instead gives every available tier/checkpoint equal weight; uneven tracking can reverse their signs. Selection averages MAPE plus absolute bias over tier/time-horizon cells; event-balanced tuning bias and last-day error also have explicit guards. A training fit or a tuning win is not independent evidence of predictive accuracy.','',
            '| Test hours remaining | Baseline MAPE / bias | NN MAPE / bias | Statistical MAPE / bias | Events |','|---|---:|---:|---:|---:|']
    test=result['partitions']['test']
    for h in reversed(HORIZONS):
        cells=[test[name]['by_horizon'].get(h,{}) for name in NAMES]
        if not cells[0].get('rows'):continue
        errors=[f"{100*c['mape']:.2f}% / {100*c['bias']:+.2f}%" for c in cells]
        lines.append(f"| {h} | "+' | '.join(errors)+f" | {cells[0]['events']} |")
    lines+=['','The curves compare identical cases within each horizon. Coverage can differ between horizons.','',
            '| Test label source | Scores | Events | Baseline MAPE | NN MAPE | Statistical MAPE |','|---|---:|---:|---:|---:|---:|']
    for source,m in test['baseline']['by_label_source'].items():
        if not m['rows']:continue
        errors=[f"{100*test[n]['by_label_source'][source]['mape']:.2f}%" for n in NAMES]
        lines.append(f"| {source} | {m['rows']} | {m['events']} | "+' | '.join(errors)+' |')
    lines+=['','Closing tracker proxies are not independently verified official finals. They require an observation at/after closure, within five minutes, after the configured publication delay. Archives and manual verified labels take precedence. Proxy agreement must not be presented as verified-final accuracy.','',
            '| Tier | NN training targets | Test scores | Test events | NN MAPE | Statistical MAPE |','|---|---:|---:|---:|---:|---:|']
    tiers=sorted(result['training_targets_by_tier'],key=int)
    for tier in tiers:
        m=test['baseline_nn']['by_tier'].get(tier,{'rows':0,'events':0})
        s=test['statistical']['by_tier'].get(tier,{})
        a=f"{100*m['mape']:.2f}%" if m['rows'] else 'unavailable'
        b=f"{100*s['mape']:.2f}%" if m['rows'] else 'unavailable'
        lines.append(f"| T{tier} | {result['training_targets_by_tier'][tier]} | {m['rows']} | {m['events']} | {a} | {b} |")
    lines+=['','A head with zero supervised targets applies no neural correction. A tier without sufficient fresh observations and earlier training references remains unavailable.','']
    for name,values in result['bootstrap'].items():
        if values:
            lo,hi=values['interval_95']
            lines.append(f"{NAMES[name]} minus baseline MAPE: {100*values['mape_change']:+.2f} percentage points; event-clustered bootstrap 95% interval [{100*lo:+.2f}, {100*hi:+.2f}]. Negative favors the competitor. A range spanning zero is inconclusive.")
            lines.append('')
    lines+=['Calibration diagnostics use maximum log error per event. Prediction bands require enough independent calibration events for that individual tier; a band is never borrowed from other tiers with labels. Historical drift and proxy labels prevent an unconditional coverage claim.','',
            'Observation timestamps usually lack publication evidence, and historical schedule revisions may be missing. Confirmed outages are used only after their known-at time; tracker gaps alone are never interpreted as outages.','',
            f"Source snapshot: `{result['snapshot'].get('bundle')}`.",
            f"Frozen audit predictions: `{artifact.resolve()/'predictions.csv'}`."]
    atomic_bytes(destination,('\n'.join(lines)+'\n').encode())
    atomic_json(destination.with_suffix('.json'),{'plan':plan,'evaluation':result})
    return destination


def plot_report(artifact,destination):
    os.environ.setdefault('MPLCONFIGDIR',str((Path('runs')/'matplotlib_cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result=json.loads((Path(artifact)/'evaluation.json').read_text());order=list(reversed(HORIZONS))
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col')
    for column,part in enumerate(('tune','test')):
        models=result['partitions'][part]
        for row,metric in enumerate(('mape','bias')):
            ax=axes[row,column]
            for name,label in NAMES.items():
                values=models[name]['by_horizon']
                y=[100*values[h][metric] if values.get(h,{}).get('rows') else float('nan') for h in order]
                ax.plot(range(5),y,marker='o',label=label,linewidth=2)
            base=models['baseline']['by_horizon']
            ax.set_xticks(range(5),[f"{h}\n{base.get(h,{}).get('events',0)} events" for h in order],fontsize=9)
            ax.grid(axis='y',alpha=.2)
            ax.set_ylabel('Event-balanced MAPE (%)' if row==0 else 'Event-balanced signed bias (%)')
            if row==0:
                ax.set_title('Tuning: selected systems' if part=='tune' else 'Fresh test: selections frozen')
                ax.legend(frameon=False,fontsize=9)
            else:
                ax.axhline(0,color='grey',linewidth=.7)
                ax.set_xlabel('Hours remaining at 08:00 CN; earlier predictions on the left')
    fig.suptitle('Simple baseline + joint NN versus statistical predictor',fontweight='bold')
    fig.text(.5,.015,'Same cases for all systems within each horizon. Check the report for label sources and changing tier coverage.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=170);fig.savefig(destination.with_suffix('.svg'));plt.close(fig)
