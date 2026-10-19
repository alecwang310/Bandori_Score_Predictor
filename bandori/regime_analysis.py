"""Training-bank-only diagnostics of CN reward changes and event types."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from collect_bandori_data import atomic_json, iso, write_csv
from .config import Config
from .daily_observer import sample
from .dataset import Dataset
from .rules import EVENT_TYPES, RULE_START, SOURCES, TYPE_NAMES, era


def design(rows,include_step=False):
    day=np.array([(r['start']-RULE_START)/86400/365 for r in rows])
    x=np.column_stack((np.ones(len(rows)),day,
        np.log([r['duration_hours']/202 for r in rows]),
        *[[float(r['event_type']==t) for r in rows] for t in EVENT_TYPES if t not in ('mission_live','unknown')]))
    if include_step:x=np.column_stack((x,[r['post_rule'] for r in rows]))
    return x


def adjusted_step(rows,key,seed=1729):
    rows=[r for r in rows if key in r and r['start']>=RULE_START-730*86400]
    if len(rows)<25 or sum(r['post_rule'] for r in rows)<4:return None
    x=design(rows,True);y=np.log([r[key] for r in rows]);w=np.array([r['age_weight'] for r in rows])
    penalty=np.diag([.001,.05,.05,*([.1]*(x.shape[1]-4)),.001])
    def fit(indices):
        a=x[indices];b=y[indices];weights=w[indices]
        return np.linalg.solve(a.T@(weights[:,None]*a)+penalty,a.T@(weights*b))[-1]
    coefficient=fit(np.arange(len(rows)));rng=np.random.default_rng(seed)
    samples=np.array([fit(rng.integers(0,len(rows),len(rows))) for _ in range(1000)])
    return dict(events=len(rows),post_events=sum(r['post_rule'] for r in rows),ratio=float(np.exp(coefficient)),
                ratio_interval_95=list(np.exp(np.quantile(samples,[.025,.975]))),
                controls='Event type, log duration, linear calendar trend; ridge-regularized nuisance effects',
                uncertainty='Descriptive event bootstrap, not proof of a causal rule effect')


def analyze(data,ids,config):
    latest=max(data.events[e].end for e in ids);rows=[]
    for eid in sorted(ids,key=lambda e:data.events[e].start):
        event=data.events[eid];duration=(event.end-event.start)/3600
        row=dict(event_id=eid,start=event.start,start_cn=iso(event.start),event_type=event.event_type,
            duration_hours=duration,post_rule=era(event),age_weight=config.recency_weight((latest-event.end)/86400))
        for tier in (500,1000,1500,2000,5000):
            truth=data.labels.get((eid,tier));curve=data.curves.get((eid,tier))
            if truth is None or curve is None:continue
            row[f'final_T{tier}']=truth;row[f'rate_T{tier}']=truth/duration
            prefix=curve.prefix(data.label_available(eid,config),event.start)
            for hours in (1,3,6,24):
                values,valid=sample(prefix,np.array([event.end-hours*3600]))
                if valid[0] and 0<=values[0]<=truth:row[f'rush_{hours}h_T{tier}']=1-values[0]/truth
        for a,b in ((500,1000),(1000,2000),(500,2000),(1000,1500)):
            if f'final_T{a}' in row and f'final_T{b}' in row:row[f'ratio_T{a}_T{b}']=row[f'final_T{a}']/row[f'final_T{b}']
        rows.append(row)
    def describe(rr,key):
        values=[r[key] for r in rr if key in r]
        return dict(events=len(values),median=float(np.median(values)) if values else None,
                    q25=float(np.quantile(values,.25)) if values else None,q75=float(np.quantile(values,.75)) if values else None)
    recent=[r for r in rows if r['start']>=RULE_START-365*86400]
    ratios={}
    for key in ('ratio_T500_T1000','ratio_T1000_T2000','ratio_T500_T2000','ratio_T1000_T1500'):
        ratios[key]=dict(before=describe([r for r in recent if not r['post_rule']],key),
                         after=describe([r for r in recent if r['post_rule']],key),adjusted=adjusted_step(rows,key))
    types={}
    for t in EVENT_TYPES[:-1]:
        rr=[r for r in rows if r['event_type']==t]
        modern=[r for r in rr if r['start']>=latest-730*86400]
        types[t]=dict(all_events=len(rr),recent_events=len(modern),post_rule_events=sum(r['post_rule'] for r in rr),
            duration=describe(modern or rr,'duration_hours'),
            final_T500=describe(modern,'final_T500'),rate_T500=describe(modern,'rate_T500'),
            final_T1000=describe(modern,'final_T1000'),rate_T1000=describe(modern,'rate_T1000'),
            rush24_T500=describe(modern,'rush_24h_T500'),rush24_T1000=describe(modern,'rush_24h_T1000'))
    return dict(training_events=len(ids),event_type_counts=dict(Counter(data.events[e].event_type for e in ids)),
        post_rule_events=[r['event_id'] for r in rows if r['post_rule']],ratios=ratios,types=types,events=rows,
        sources=SOURCES,limitations=['Uses only the authorized pre-325 training bank; 325 and 326 excluded',
          'Rank-ratio changes are observational, not proof of changed point-scoring arithmetic',
          'Unadjusted type medians mix popularity, event bonuses, chronology and duration',
          'Sparse recent types and T1500 pre-change absence prevent some comparisons'])


def publish(result,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(destination.with_suffix('.json'),result)
    fields=sorted({key for row in result['events'] for key in row})
    write_csv(destination.with_suffix('.csv'),fields,result['events'])
    lines=['# CN reward-rule changes and event-type diagnostics','',
        f"Analysis uses {result['training_events']} authorized training events before event 325. Events 325/326 are excluded. "
        'These diagnostics do not move historical validation outcomes into the old strict fits.','',
        '## Confirmed rules','',
        'From May 26, 2026 (event 311), the voiced stamp moves from T1000 to T1500, an animated voiced stamp is added at T500, and a flashing title is added at T1000. '
        '[Official account announcement](https://www.sina.cn/news/detail/5300754143514601.html). '
        'From April 7, 2026 (event 305), T2000 gains a profile decoration. '
        '[Official account announcement](https://www.sina.cn/news/detail/5285051785020053.html). '
        'This establishes reward changes, not a change to the event-point formula.','',
        'The pre-325 bank has '+str(len(result['post_rule_events']))+' completed, labelled events starting under the May rules. '
        'Event IDs are not chronological: 312/313 occurred in November 2025 and must not receive the post-May-2026 multiplier.','',
        '## Rank separation','',
        '| Score ratio | Before median (events) | After median (events) | Adjusted post/before ratio [95% bootstrap] |',
        '| --- | ---: | ---: | ---: |']
    for name,value in result['ratios'].items():
        def fmt(v):return f"{v['median']:.3f} ({v['events']})" if v['events'] else 'unavailable'
        a=value['adjusted'];adjusted=f"{a['ratio']:.3f} [{a['ratio_interval_95'][0]:.3f}, {a['ratio_interval_95'][1]:.3f}]" if a else 'insufficient data'
        lines.append(f"| {name} | {fmt(value['before'])} | {fmt(value['after'])} | {adjusted} |")
    lines+=['','Raw before/after medians use events beginning within one year before May 26 or after it. '
        'The adjusted comparison uses two years before the boundary, controls event type, log duration and a linear date trend, and keeps the 730-day age weights. '
        'Its intervals describe observational sampling uncertainty and do not account for every popularity or event-bonus change.','',
        '## Event types','',
        '| Type | All events | Recent 2y events | Post-change events | Median duration | Median T500 | Median T1000 | Final-24h fraction T500 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in result['types'].items():
        def number(v):return f"{v['median']:,.0f}" if v['events'] else 'unavailable'
        rush=r['rush24_T500'];fraction=f"{100*rush['median']:.1f}% (n={rush['events']})" if rush['events'] else 'unavailable'
        lines.append(f"| {TYPE_NAMES[name]} ({name}) | {r['all_events']} | {r['recent_events']} | {r['post_rule_events']} | {number(r['duration'])}h | {number(r['final_T500'])} | {number(r['final_T1000'])} | {fraction} |")
    lines+=['','Recent medians use the two years ending at the final training event; story events have no recent samples. '
        'Absolute points differ with the event mechanics and player participation. They should not be treated as interchangeable training examples just because they last a similar number of days. '
        'The final-24h fraction uses only curves with an actual nearby observation and interpolation gaps of at most two hours.','',
        '## Modelling implications','',
        'Use the actual event type and dated reward metadata in every approach. Preserve 730-day decay and multiply post-May-26 event weights by exactly 3 in supervision, normalizers/compression, analog matching, and train-only fitted corrections. '
        'A current T1500 voice-reward boundary may borrow completion-shape information from the legacy T1000 boundary, but those older T1000 finals must never become synthetic T1500 training targets. '
        'Report cold-start, partial-adaptation and later-adaptation periods separately, plus per-tier/type errors, bias, missingness, and long-event support.','']
    destination.write_text('\n'.join(lines));plot(result,destination.with_suffix('.png'))


def plot(result,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    palette=dict(zip(EVENT_TYPES,plt.cm.tab10.colors))
    for name in EVENT_TYPES[:-1]:
        rr=[r for r in result['events'] if r['event_type']==name and r['start']>=RULE_START-730*86400]
        for ax,key in zip(axes,('ratio_T500_T1000','ratio_T1000_T2000')):
            subset=[r for r in rr if key in r]
            ax.scatter([datetime.fromtimestamp(r['start']) for r in subset],[r[key] for r in subset],s=22,label=name,color=palette[name],alpha=.8)
            ax.set_ylabel(key.replace('ratio_','').replace('_',' / '));ax.grid(alpha=.2)
    for ax in axes:ax.axvline(datetime.fromtimestamp(RULE_START),color='black',linestyle='--',label='May 26 reward change')
    axes[0].legend(ncol=4,fontsize=8);axes[1].set_xlabel('Actual CN event start date')
    fig.suptitle('CN rank separation by event type · pre-325 training bank only')
    fig.savefig(destination,dpi=160);fig.savefig(Path(destination).with_suffix('.svg'));plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True)
    parser.add_argument('--bank',default='runs/all_except_325_326_20261017');parser.add_argument('--output',default='reports/rule_change_diagnostics.md')
    args=parser.parse_args();config=Config.load('experiment_config.json');data=Dataset.load(args.data,config.tiers);data.add_terminal_labels(config)
    ids=json.loads((Path(args.bank)/'model.json').read_text())['training_event_ids'];result=analyze(data,ids,config)
    publish(result,args.output);print(json.dumps(dict(post_rule=result['post_rule_events'],ratios=result['ratios'],types=result['event_type_counts']),indent=2))


if __name__=='__main__':main()
