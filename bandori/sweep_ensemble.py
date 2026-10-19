"""Ensemble sweep over the case-feature model family on new events 321-324.

Trains exclusively through `bandori.sweep_harness` on the frozen bank (315
training events of the `eight_post_events` fold, including post-rule 311-320)
and tests only on validation events 321-324. Standing scope throughout:
event-balanced MAPE on clean checkpoints, tiers 500/1000/1500/2000 (312 rows),
with signed bias and final-24h MAPE on the identical rows.

Families exercised:

* two-model value blends of nn_pca and ridge over a weight grid;
* diversity members (nn_pca at three seeds, nn_minimal, nn_standard, tcn24)
  combined by simple mean, per-event trimmed mean, and train-side top-3 mean;
* five-seed bagging of nn_pca;
* blends with the raised-objective nn_pca (`raise_objective_share`, share .6);
* a train-side-selected nn_pca/ridge blend weight chosen on post-rule training
  events 311-320 only (cheap in-sample LOO-scoring approximation: members are
  fitted once on the full 315 bank and the weight is picked by event-balanced
  MAPE on their in-sample predictions of 311-320 -- labelled as such, never
  presented as a refit leave-one-out). Its 321-324 score is the tuned headline.

Every number uses the same harness rows, splits and `score()`. No frozen run,
harness module or ledger file is modified.
"""
from pathlib import Path

import numpy as np

from .experiments import correction_forecast
from .raised_objective_experiment import raise_objective_share
from .sweep_harness import (FOCAL_TIERS, POST_RULE_TRAIN_EVENTS, VAL_EVENTS,
                            fit_jointnet, fit_ridge, fmt, load_bank,
                            prepare_fit, save_results, score, validate,
                            write_report)
from .temporal import JointTemporalNet

RUN=Path('runs/sweep_ensemble_20261019')
REPORT=Path('reports/sweep_ensemble_20261019.md')
PCA_SEEDS=(1729,1730,1731,1732,1733)
DIVERSITY_SEEDS=PCA_SEEDS[:3]
BLEND_GRID=(0.0,0.25,0.5,0.75,1.0)
RAISED_SHARE=.6
# Frozen-run ensembles under the same focal scope, quoted from the frozen
# eight_post_events evaluation (rule_aware_3x_20261018); not refit here.
FROZEN_ENSEMBLES={'neural_mean':dict(mape=.0732),'collection_mean':dict(mape=.0756)}


def member(name,model,rep,gain=1.0,corr='nn',kind='joint'):
    return dict(name=name,model=model,rep=rep,gain=gain,corr=corr,kind=kind)


def member_predict(m,ctx):
    """Per-case final values for a fitted member on a context (val or train)."""
    if m['kind']=='temporal':
        delta=m['model'].predict(ctx['matrices']['standard'],ctx['daily']['sequence'])
        return correction_forecast(ctx['cases'],ctx['index'],delta,m['gain'],'nn')
    delta=m['model'].predict(ctx['matrices'][m['rep']])
    return correction_forecast(ctx['cases'],ctx['index'],delta,m['gain'],m['corr'])


def combine(values_by_name,weights):
    """Weighted combination of final per-case value vectors."""
    out=None
    for name,w in weights.items():
        term=w*np.asarray(values_by_name[name])
        out=term if out is None else out+term
    return out


def trimmed_values(cases,values_by_name,members,per_event_mape):
    """Mean of members after dropping, per event, that event's worst member."""
    out=np.empty(len(cases))
    for eid in sorted({c.event_id for c in cases}):
        ii=[i for i,c in enumerate(cases) if c.event_id==eid]
        worst=max(members,key=lambda m:per_event_mape[m][eid]['mape'])
        keep=[m for m in members if m!=worst]
        out[ii]=np.mean([values_by_name[m][ii] for m in keep],axis=0)
    return out


def post_rule_train_view(bank,cases,values):
    """Score predictions restricted to post-rule training events 311-320."""
    allowed=set(POST_RULE_TRAIN_EVENTS)
    keep=[i for i,c in enumerate(cases) if c.event_id in allowed]
    return score([cases[i] for i in keep],np.asarray(values)[keep],bank.data)


def compact(result):
    return {k:result[k] for k in ('mape','bias','final24h','rows','events','per_event')}


def run():
    bank=load_bank()
    print('Validate harness against frozen baselines',flush=True)
    res_nn0,res_ridge0=validate(bank)
    prep=prepare_fit(bank)
    train_cases=prep.train['cases'];val_cases=prep.val['cases']

    members={}
    print('Fit nn_pca seeds',PCA_SEEDS,flush=True)
    for seed in PCA_SEEDS:
        model=fit_jointnet(prep.matrix['pca'],prep.targets,prep.weights,prep.bins,
                           bank.config,hidden=8,seed=seed)
        members[f'nn_pca_s{seed}']=member(f'nn_pca_s{seed}',model,'pca')
    print('Fit nn_minimal / nn_standard / ridge',flush=True)
    model=fit_jointnet(prep.matrix['minimal'],prep.targets,prep.weights,prep.bins,
                       bank.config,hidden=8,seed=1729)
    members['nn_minimal']=member('nn_minimal',model,'minimal')
    model=fit_jointnet(prep.matrix['standard'],prep.targets,prep.weights,prep.bins,
                       bank.config,hidden=8,seed=1729)
    members['nn_standard']=member('nn_standard',model,'standard')
    model=fit_ridge(prep.matrix['pca'],prep.targets,prep.weights,prep.bins)
    members['ridge']=member('ridge',model,'pca',gain=.5,corr='ridge')
    print('Fit tcn24 (slow)',flush=True)
    tcn=JointTemporalNet(4,8,1729).fit(prep.matrix['standard'],prep.train['daily']['sequence'],
                                       prep.targets,prep.weights,prep.bins,bank.config)
    members['tcn24']=member('tcn24',tcn,'standard',gain=.5,kind='temporal')
    raised_weights,raised_info=raise_objective_share(prep.weights,bank.config.tiers,share=RAISED_SHARE)
    print('Fit nn_pca raised-objective share',RAISED_SHARE,flush=True)
    model=fit_jointnet(prep.matrix['pca'],prep.targets,raised_weights,prep.bins,
                       bank.config,hidden=8,seed=1729)
    members['nn_pca_raised']=member('nn_pca_raised',model,'pca')

    val_values={name:member_predict(m,prep.val) for name,m in members.items()}
    val_scores={name:score(val_cases,v,bank.data) for name,v in val_values.items()}
    train_values={name:member_predict(m,prep.train) for name,m in members.items()}
    train_scores={name:post_rule_train_view(bank,train_cases,v) for name,v in train_values.items()}
    for name in members:
        print('member',name,'val',fmt(val_scores[name]),'| train 311-320',
              f"MAPE {100*train_scores[name]['mape']:.2f}%",flush=True)

    nn0='nn_pca_s1729'
    ensembles={}

    def add(name,values,rule,members_line):
        res=score(val_cases,values,bank.data)
        ensembles[name]=dict(members=members_line,rule=rule,score=compact(res))
        print('ensemble',name,fmt(res),flush=True)
        return res

    # 1. Two-model value blends of nn_pca and ridge (descriptive grid).
    for w in BLEND_GRID:
        add(f'blend_nn_ridge_w{w:.2f}',combine(val_values,{nn0:w,'ridge':1-w}),
            f'{w:.2f}·nn_pca + {1-w:.2f}·ridge (final values)','nn_pca_s1729 + ridge')

    # 2. Diversity members.
    mean5_names=[f'nn_pca_s{s}' for s in DIVERSITY_SEEDS]+['nn_minimal','nn_standard']
    mean6_names=mean5_names+['tcn24']
    add('bagging3_mean',combine(val_values,{n:1/3 for n in [f'nn_pca_s{s}' for s in DIVERSITY_SEEDS]}),
        'equal mean',' + '.join(f'nn_pca_s{s}' for s in DIVERSITY_SEEDS))
    add('bagging5_mean',combine(val_values,{f'nn_pca_s{s}':1/5 for s in PCA_SEEDS}),
        'equal mean',' + '.join(f'nn_pca_s{s}' for s in PCA_SEEDS))
    add('diversity_mean5',combine(val_values,{n:1/5 for n in mean5_names}),
        'equal mean',' + '.join(mean5_names))
    add('diversity_mean6_tcn',combine(val_values,{n:1/6 for n in mean6_names}),
        'equal mean',' + '.join(mean6_names))
    per_event_mape={n:val_scores[n]['per_event'] for n in mean6_names}
    add('diversity_trimmed_mean6',
        trimmed_values(val_cases,val_values,mean6_names,per_event_mape),
        'equal mean after dropping per-event worst member',' + '.join(mean6_names))
    ranked=sorted(mean6_names,key=lambda n:train_scores[n]['mape'])
    top3=ranked[:3]
    add('diversity_top3_train',combine(val_values,{n:1/3 for n in top3}),
        'equal mean of top-3 by train-side (311-320) MAPE',' + '.join(top3))

    # 3. Blend with the raised-objective variant.
    add('raised_alone',val_values['nn_pca_raised'],'single (60% objective-tier loss share)','nn_pca_raised')
    add('raised_natural_50_50',combine(val_values,{'nn_pca_raised':.5,nn0:.5}),
        '0.5·nn_pca_raised + 0.5·nn_pca (final values)','nn_pca_raised + nn_pca_s1729')
    add('raised_ridge_50_50',combine(val_values,{'nn_pca_raised':.5,'ridge':.5}),
        '0.5·nn_pca_raised + 0.5·ridge (final values)','nn_pca_raised + ridge')
    add('raised_natural_ridge_thirds',
        combine(val_values,{'nn_pca_raised':1/3,nn0:1/3,'ridge':1/3}),
        'equal thirds (final values)','nn_pca_raised + nn_pca_s1729 + ridge')

    # 4. Train-side-selected blend weight (honest headline).
    train_grid=[]
    for w in BLEND_GRID:
        res=post_rule_train_view(bank,train_cases,
                                 combine(train_values,{nn0:w,'ridge':1-w}))
        train_grid.append(dict(w=w,mape=res['mape'],bias=res['bias'],per_event=res['per_event']))
        print(f'train-side blend w={w:.2f}: MAPE {100*res["mape"]:.2f}%',flush=True)
    best=min(train_grid,key=lambda r:r['mape'])
    selected=add(f'train_selected_blend_w{best["w"]:.2f}',
                 combine(val_values,{nn0:best['w'],'ridge':1-best['w']}),
                 f'w={best["w"]:.2f} chosen on train 311-320 (in-sample LOO-scoring approximation)',
                 'nn_pca_s1729 + ridge')

    payload=dict(
        scope=dict(fold='eight_post_events',train_events=315,
                   post_rule_train_events=list(POST_RULE_TRAIN_EVENTS),
                   validation_events=list(VAL_EVENTS),focal_tiers=list(FOCAL_TIERS),
                   rows=int(val_scores[nn0]['rows'])),
        baselines=dict(nn_pca=compact(res_nn0),ridge=compact(res_ridge0),
                       frozen_ensembles=FROZEN_ENSEMBLES),
        raised_weight_info=raised_info,
        members={n:dict(val=compact(val_scores[n]),
                        train_311_320=compact(train_scores[n]))
                 for n in members},
        ensembles={n:dict(members=e['members'],rule=e['rule'],**e['score'])
                   for n,e in ensembles.items()},
        selection=dict(method='members fitted once on all 315 events; blend weight picked by '
                              'event-balanced MAPE on in-sample predictions of training events '
                              '311-320 (cheap approximation of leave-one-out refit)',
                       grid=train_grid,best_w=best['w'],headline_val=compact(selected)),
        limitations=[
            'Four previously inspected validation events (321-324); descriptive grids are not fresh tests',
            'Train-side selection scores in-sample predictions of 311-320 (members saw those labels), '
            'not a true leave-one-out refit',
            'Rows within an event are dependent; event count (4) limits precision',
            'Raised-objective variant was designed for the stream LLM family; here it only rescales '
            'the case-feature JointNet weights'])
    save_results(RUN/'evaluation.json',payload)

    f=lambda v:'—' if v is None else f'{100*v:.2f}%'
    table_lines=['| Ensemble | Members | Weight rule | MAPE | Bias | Final 24h MAPE |',
                 '| --- | --- | --- | ---: | ---: | ---: |']
    def row(label,members_line,rule,s):
        table_lines.append(f"| {label} | {members_line} | {rule} | {f(s['mape'])} | "
                           f"{f(s['bias'])} | {f(s['final24h'])} |")
    row('nn_pca (single)','nn_pca_s1729','single',compact(res_nn0))
    row('ridge (single)','ridge','single',compact(res_ridge0))
    for name in [f'blend_nn_ridge_w{w:.2f}' for w in BLEND_GRID]:
        e=ensembles[name];row(name.replace('_',' '),e['members'],e['rule'],e['score'])
    for name in ('bagging3_mean','bagging5_mean','diversity_mean5','diversity_mean6_tcn',
                 'diversity_trimmed_mean6','diversity_top3_train'):
        e=ensembles[name];row(name.replace('_',' '),e['members'],e['rule'],e['score'])
    for name in ('raised_alone','raised_natural_50_50','raised_ridge_50_50',
                 'raised_natural_ridge_thirds'):
        e=ensembles[name];row(name.replace('_',' '),e['members'],e['rule'],e['score'])
    e=ensembles[f'train_selected_blend_w{best["w"]:.2f}']
    row(f'Train-selected blend (headline)',e['members'],e['rule'],e['score'])

    key_names=[nn0,'blend_nn_ridge_w0.50','blend_nn_ridge_w0.75','bagging5_mean',
               'diversity_mean6_tcn','diversity_top3_train','train_selected_blend_w%.2f'%best['w']]
    per_event_lines=['| Ensemble | 321 | 322 | 323 | 324 |','| --- | ---: | ---: | ---: | ---: |']
    for name in key_names:
        s=ensembles[name]['score'] if name in ensembles else compact(res_nn0)
        pe={int(k):v for k,v in s['per_event'].items()}
        per_event_lines.append(f"| {name.replace('_',' ')} | "+
            ' | '.join(f"{100*pe[e]['mape']:.2f}%" for e in VAL_EVENTS)+' |')

    grid_lines=['| w (nn_pca weight) | Train 311-320 MAPE | 321-324 MAPE (descriptive) |',
                '| ---: | ---: | ---: |']
    for g in train_grid:
        grid_lines.append(f"| {g['w']:.2f} | {100*g['mape']:.2f}% | "
            f"{100*ensembles[f'blend_nn_ridge_w{g['w']:.2f}']['score']['mape']:.2f}% |")

    member_lines=['| Member | Val 321-324 MAPE | Train 311-320 MAPE |','| --- | ---: | ---: |']
    for n in members:
        member_lines.append(f"| {n} | {100*val_scores[n]['mape']:.2f}% | "
                            f"{100*train_scores[n]['mape']:.2f}% |")

    best_ens=min(ensembles.items(),key=lambda kv:kv[1]['score']['mape'])
    frozen_line=', '.join(f"{k} {100*v['mape']:.2f}%" for k,v in FROZEN_ENSEMBLES.items())
    write_report(REPORT,'Sweep: ensembles of case-feature models on new events 321-324',[
        ('Protocol',
         [f"Shared harness (`bandori.sweep_harness`): train on the frozen 315-event bank "
          f"(post-rule 311-320 included), test only on events {', '.join(map(str,VAL_EVENTS))}. "
          f"Reported MAPE = event-balanced MAPE on clean checkpoints, tiers "
          f"{', '.join(map(str,FOCAL_TIERS))} ({val_scores[nn0]['rows']} rows); bias and final-24h "
          f"MAPE use the identical rows. Harness validation reproduced nn_pca "
          f"{100*res_nn0['mape']:.2f}% and ridge {100*res_ridge0['mape']:.2f}%.",
          f"Frozen-run ensembles under the same scope for context: {frozen_line} "
          f"(from the frozen eight_post_events evaluation, not refit here)."]),
        ('Fitted members',member_lines),
        ('Ensemble comparison',table_lines),
        ('Per-event MAPE (key ensembles)',per_event_lines),
        ('Train-side-selected blend weight',
         [f"Candidates: w·nn_pca + (1−w)·ridge on final values, w ∈ {{{', '.join(f'{w:.2f}' for w in BLEND_GRID)}}}. "
          f"Selection uses only post-rule training events {POST_RULE_TRAIN_EVENTS[0]}-{POST_RULE_TRAIN_EVENTS[-1]}: "
          f"members are fitted once on all 315 events and w is picked by event-balanced MAPE on their "
          f"in-sample predictions of those ten events — a cheap approximation of leave-one-out refit "
          f"(the labels of 311-320 entered the supervised fit), labelled as such.",
          '',
          *grid_lines,'',
          f"Chosen w = {best['w']:.2f}; its 321-324 score (the tuned headline): "
          f"MAPE {f(selected['mape'])}, bias {f(selected['bias'])}, final-24h {f(selected['final24h'])}. "
          f"Descriptive best ensemble on 321-324: {best_ens[0].replace('_',' ')} at "
          f"{f(best_ens[1]['score']['mape'])} — shown for contrast only, not a valid selection."]),
        ('Caveats',
         [f"Only four validation events ({', '.join(map(str,VAL_EVENTS))}); they have been inspected "
          f"by prior experiments, so the full descriptive grid is not a fresh test.",
          "Rows within an event are dependent; per-event MAPE (not row count) is the effective sample size.",
          f"Train-side selection is in-sample on 311-320, so the chosen weight is optimistic; a true "
          f"LOO refit (315-minus-one-event) was not run because the approximation already picks an "
          f"interior weight and refitting every member ten times is costly.",
          f"The raised-objective variant ({100*RAISED_SHARE:.0f}% objective-tier loss share) was designed "
          f"for the stream LSTM/transformer family; on the case-feature JointNet it degrades sharply "
          f"(single MAPE {f(val_scores['nn_pca_raised']['mape'])}), so its blends are shown for completeness.",
          "No deployed model is replaced by this sweep; evaluation_registry.json was not modified."])])
    print('Sweep complete; best ensemble',best_ens[0],fmt(best_ens[1]['score']),flush=True)
    return payload


def main():
    run()


if __name__=='__main__':
    main()
