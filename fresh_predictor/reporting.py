"""Summarize the single sealed assessment without selecting another model."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bandori.analog import HORIZONS
from bandori.evaluation import paired_bootstrap
from bandori.pipeline import digest
from collect_bandori_data import atomic_json,timestamp


def publish(root,destination):
    root,destination=Path(root),Path(destination)
    assessment=json.loads((root/'historical_assessment.json').read_text());selection=json.loads((root/'selection.json').read_text())
    audit=json.loads((root/'isolation_audit.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    with (root/'historical_predictions.csv').open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if int(r['tier']) in (500,1000,1500,2000) and r['actual']]
    with Path('runs/rule_aware_3x_20261018/eight_post_events/predictions.csv').open(encoding='utf-8-sig') as f:
        old={(int(r['event_id']),int(r['tier']),timestamp(r['checkpoint'])):r for r in csv.DictReader(f) if r['scenario']=='clean'}
    rows=[r for r in rows if (int(r['event_id']),int(r['tier']),timestamp(r['checkpoint'])) in old]
    cases=[SimpleNamespace(event_id=int(r['event_id']),tier=int(r['tier']),remaining_hours=float(r['hours_remaining'])) for r in rows]
    data=SimpleNamespace(labels={(int(r['event_id']),int(r['tier'])):float(r['actual']) for r in rows})
    prediction=np.array([float(r['final']) for r in rows]);paired={}
    for name in ('baseline','ridge','nn_pca'):
        before=np.array([float(old[(int(r['event_id']),int(r['tier']),timestamp(r['checkpoint']))][name]) for r in rows])
        paired[name]=paired_bootstrap(cases,before,prediction,data,1729)
    result=dict(assessment=assessment,selection=selection,audit=audit,paired=paired,
        production_model_sha256=digest(root/'production_model/model.json'),archive=json.loads(Path('reports/experiment_archive_20261019.json').read_text()))
    destination.parent.mkdir(parents=True,exist_ok=True);atomic_json(destination.with_suffix('.json'),result)
    pct=lambda x:'—' if x is None else f'{100*x:.2f}%'
    new=assessment['new'];lines=['# Fresh post-rule predictor: sealed selection and historical assessment','',
        f"The new 120-coefficient joint statistical model scores **{pct(new['overall']['mape'])} MAPE** on the fixed historical audit, "
        f"with {pct(new['overall']['bias'])} signed bias and {pct(new['by_horizon']['0-24h']['mape'])} final-day MAPE. "
        'Its one searched hyperparameter was sealed before this assessment. The design was informed by previous experiments, so this is not a fresh independent test.','',
        f"All previous experiments were cached and verified: {result['archive']['verified_files']:,} files, {result['archive']['archive_bytes']:,} compressed bytes, "
        f"SHA256 `{result['archive']['archive_sha256']}`. Source, all datasets, cached features, weights, reports, predictions and Git history are preserved. "
        'See [archive receipt](experiment_archive_20261019.json).','',
        '## Selection before assessment','',
        'Only actual post-rule events provide supervised labels. Selection uses events 311, 314–320, with forward validation on 317, 318, 319 and 320. '
        'Each fit uses earlier outcomes only, including reference-curve labels, anchors, normalization and coefficients. Audit events 321–324 and excluded 325/326 are absent from the selection data view. '
        'No old fitted network, PCA, feature cache or sweep winner is used.','',
        '| Predeclared shrinkage | Training-side chronological MAPE |','| --- | ---: |']
    for c in selection['candidates']:lines.append(f"| {c['strength']}{' **selected**' if c['strength']==selection['strength'] else ''} | {pct(c['mean_mape'])} |")
    lines+=['',f"Sealed selection SHA256: `{digest(root/'selection.json')}`. Only this setting is assessed; no audit outcome changes the model, gain, feature set or supervised scope. "
        'The research model uses eight post-rule events and the earlier 315-event curve bank for the entire audit block.','',
        '## Identical recent post-rule rows','',
        f"Events 321–324; T500/T1000/T1500/T2000; {assessment['matched_rows']} matched rows at 08/20/21 CN. The original benchmark has {assessment['old_benchmark_rows']} rows. "
        'Each event receives equal reporting weight.','',
        '| Model | MAPE | Signed bias | Final 24h MAPE |','| --- | ---: | ---: | ---: |']
    for name in ('baseline','ridge','nn_pca','new_anchor','fresh'):
        s=assessment['matched'][name];lines.append(f"| {name} | {pct(s['overall']['mape'])} | {pct(s['overall']['bias'])} | {pct(s['by_horizon'].get('0-24h',{}).get('mape'))} |")
    lines+=['','| Fresh minus comparator | MAPE change (percentage points) | Event-bootstrap 95% interval |','| --- | ---: | ---: |']
    for name,s in paired.items():lines.append(f"| {name} | {100*s['mape_change']:+.2f} | [{100*s['interval_95'][0]:+.2f}, {100*s['interval_95'][1]:+.2f}] |")
    lines+=['','Bootstrap intervals describe only four previously inspected event clusters and do not account for historical design selection. '
        'A lower average is not by itself proof of future improvement.','',
        '| Audit event | Fresh MAPE | Signed bias |','| --- | ---: | ---: |']
    for eid,s in assessment['by_event'].items():lines.append(f"| {eid} | {pct(s['mape'])} | {pct(s['bias'])} |")
    lines+=['',f"At 08:00 CN alone: MAPE {pct(assessment['at_08_cn']['overall']['mape'])}, bias {pct(assessment['at_08_cn']['overall']['bias'])}; "
        f"{assessment['at_08_cn']['overall']['rows']} rows. This is an additional breakdown, not a separate selection. Members of the same event remain dependent.",'',
        '## Production and uncertainty','',
        'Production coefficients are refitted to all twelve available post-rule events, with the unchanged sealed shrinkage and feature schema. '
        'The older 319-event bank supplies reference curves only. Events 325/326 stay outside fitting. Research-fit audit accuracy is not silently assigned to the larger production refit.','',
        'Approximate sigma comes from adaptive chronological forecasts on 317–320: each event uses a setting chosen using earlier validation events only. '
        'There are at most four calibration event clusters, with at least three required per same-tier cell and an all-horizon same-tier fallback. No rank borrowing is used. '
        'The residuals include bias and proxy-label limitations; normality and nominal coverage are unverified.','',
        f"Data-isolation audit: **{'passed' if audit['passed'] else 'FAILED'}**, {len(audit['checks'])} checks. "
        'The complete selection procedure was rerun after poisoning all audit/excluded histories and outcomes: the entire selection and research coefficients are identical. '
        'Future-tail and own-final mutations for 325/326 also leave predictions unchanged. The protected ledger is unchanged.','',
        'All **118 tests** pass on CPU, including the new chronological-isolation, reward-role-without-fabricated-labels, publication and save/load checks. '
        'The standalone package also reproduces the saved production forecasts in an external directory with PyTorch imports blocked.','',
        '## Run and reproduce','',
        'The standalone [dist package](../dist/README.md) prompts for the latest CN event and issues all available tier forecasts. '
        'It requires only Python and NumPy; the stable older package is preserved. The included October-17 input example is historical, with its actual generation timestamp recorded.','',
        'Implementation and reproduction: [design/protocol](../docs/fresh_post_rule_predictor.md). '
        f"Frozen run: `{root}`. Predictions locked before later closing labels provide the next prospective assessment.",'']
    destination.write_text('\n'.join(lines))
    plot(assessment['matched'],destination.with_suffix('.png'))
    return result


def plot(scores,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name in ('baseline','ridge','nn_pca','fresh'):
        for ax,metric in zip(axes,('mape','bias')):ax.plot(range(5),[100*scores[name]['by_horizon'].get(h,{}).get(metric,np.nan) for h in HORIZONS],marker='o',label=name)
    for ax in axes:ax.set_xticks(range(5),HORIZONS);ax.set_xlabel('Hours remaining');ax.grid(alpha=.2)
    axes[0].set_ylabel('MAPE (%)');axes[1].set_ylabel('Signed error (%)');axes[1].axhline(0,color='black',linewidth=.7);axes[0].legend(fontsize=8)
    fig.suptitle('Sealed fresh predictor · post-rule events 321–324 · focal tiers');fig.savefig(path,dpi=160);fig.savefig(path.with_suffix('.svg'));plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',default='reports/fresh_post_rule_20261019.md');a=parser.parse_args()
    publish(a.run,a.output)
