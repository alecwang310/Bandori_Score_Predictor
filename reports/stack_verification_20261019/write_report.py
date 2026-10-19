"""Render the verification results without touching any model package."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
runtime=json.loads((OUT/'runtime_metrics.json').read_text())
research=json.loads((OUT/'research_metrics.json').read_text())
rows=list(csv.DictReader((OUT/'runtime_predictions.csv').open()))
for r in rows:
    for key in ('hours_remaining','cutoff','absolute_error_percent','signed_error_percent'):r[key]=float(r[key])
    r['tier']=int(r['tier'])
focal=[r for r in rows if r['tier'] in (500,1000,1500,2000)]
colors=dict(fresh='#bc6a45',ridge='#6883b5',stack='#398f72')
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax=axes[0,0];events=list(research['research']['per_event']);x=np.arange(len(events));width=.34
for offset,name in [(-width/2,'fresh'),(width/2,'stack')]:
    values=[research['research']['per_event'][e][name] for e in events]
    bars=ax.bar(x+offset,values,width,label=name,color=colors[name]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
ax.set_xticks(x,events);ax.set_xlabel('Event ID');ax.set_ylabel('Event MAPE (%)')
ax.set_title('321–324: earlier research fits, inspected development data',fontsize=11);ax.legend(frameon=False)
ax=axes[0,1]
for name in ('fresh','ridge','stack'):
    means=[];xs=[]
    for lo in range(0,384,24):
        rr=[r for r in focal if r['model']==name and lo<=r['hours_remaining']<lo+24]
        if rr:xs.append((lo+12)/24);means.append(np.mean([r['absolute_error_percent'] for r in rr]))
    ax.plot(xs,means,'o-',label=name,color=colors[name],ms=4)
ax.invert_xaxis();ax.set_xlabel('Days remaining (24-hour bins)');ax.set_ylabel('MAPE (%)')
ax.set_title('325: packaged models across the full event',fontsize=11);ax.legend(frameon=False)
ax=axes[1,0];tiers=list(runtime['by_tier']);x=np.arange(4);width=.25
for offset,name in [(-width,'fresh'),(0,'ridge'),(width,'stack')]:
    values=[runtime['by_tier'][t][name]['mape_percent'] for t in tiers]
    bars=ax.bar(x+offset,values,width,label=name,color=colors[name]);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
ax.set_xticks(x,['T'+t for t in tiers]);ax.set_ylabel('MAPE (%)');ax.set_title('325: improvement differs by tier',fontsize=11)
ax=axes[1,1];seeds=list(research['production_fixed_seeds']);a=[research['production_fixed_seeds'][s]['standing']['mape_percent'] for s in seeds]
ax.plot(seeds,a,'o-',color=colors['stack'],label='stack: fixed seeds, no selection')
ax.axhline(runtime['standing']['fresh']['mape_percent'],color=colors['fresh'],ls='--',label='Fresh')
ax.axhline(runtime['standing']['ridge']['mape_percent'],color=colors['ridge'],ls=':',label='Ridge')
ax.set_xlabel('Initialization seed');ax.set_ylabel('325 full-event MAPE (%)');ax.set_title('325: fixed-recipe seed sensitivity',fontsize=11);ax.legend(frameon=False,fontsize=9)
for ax in axes.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Stack verification: historical gains, with limits',fontsize=16,fontweight='bold')
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.svg');plt.close(fig)
lines=['Stack verification after the workspace refactor','',
'Conclusion: the current stack is a stronger historical point predictor than Fresh overall. The gain on event 325 reproduces in the isolated standalone package and persists over five prespecified initialization seeds. It is not uniformly better by tier, and the four-event 321–324 advantage is small, selected after inspecting that development block, and statistically inconclusive. Ridge retains an advantage at the supplied external checkpoints and near closing.','',
'Independent runtime replay: event 325','',
'Every comparison below uses exactly matching T500/T1000/T1500/T2000 rows, with observations at most one hour old. There are 47 scheduled 08/20/21 CN checkpoints; six have no eligible fresh observations, leaving 41 checkpoints × four tiers = 164 points. External-24 is the subset of six supplied issue times with fresh input. The isolated runtime imported only the shipped distribution, explicitly blocking PyTorch. Final targets are closing tracker proxies, not independently verified official finals.','',
'| Matched subset | Points per model | Fresh MAPE | Ridge MAPE | Stack MAPE |',
'|---|---:|---:|---:|---:|']
for key,label in [('standing','Full event'),('only_0800','08:00 only'),('external_24','Supplied external timestamps'),('last_24h','Final 24 hours')]:
    d=runtime[key];lines.append(f"| {label} | {d['fresh']['rows']} | "+' | '.join(f"{d[n]['mape_percent']:.2f}%" for n in ('fresh','ridge','stack'))+' |')
lines+=['','| Full-event statistic | Fresh | Ridge | Stack |','|---|---:|---:|---:|']
for metric,label in [('bias_percent','Mean signed relative error'),('p90_absolute_error_percent','90th percentile absolute relative error'),('worst_absolute_error_percent','Worst absolute relative error')]:
    lines.append('| '+label+' | '+' | '.join(f"{runtime['standing'][n][metric]:+.2f}%" if metric=='bias_percent' else f"{runtime['standing'][n][metric]:.2f}%" for n in ('fresh','ridge','stack'))+' |')
lines+=['','Stack wins 96/164 paired focal observations against Fresh. The lower average and 90th percentile do not imply a lower worst-case error: all models struggle with T2000 at the first issue, where Fresh is better.','',
'| Tier | Fresh MAPE | Ridge MAPE | Stack MAPE |','|---|---:|---:|---:|']
for t,group in runtime['by_tier'].items():lines.append('| T'+t+' | '+' | '.join(f"{group[n]['mape_percent']:.2f}%" for n in ('fresh','ridge','stack'))+' |')
lines+=['','| Hours remaining | Fresh MAPE | Ridge MAPE | Stack MAPE |','|---|---:|---:|---:|']
for h,group in runtime['by_horizon'].items():lines.append('| '+h+' | '+' | '.join(f"{group[n]['mape_percent']:.2f}%" for n in ('fresh','ridge','stack'))+' |')
lines+=['','Supplementary all-labeled-tier comparison (not the standing headline): matched 870 observations per model, Fresh 13.30%, Ridge 12.39%, Stack 11.69% MAPE.','',
'Recent historical research comparison, 321–324','',
'These are earlier fits stopped before event 321, not the current production models. The production stack supervises on 321–324 and cannot honestly test on them. We reproduced the original rule+explicit × post20 research recipe and matched all 312 observations to the frozen Fresh research CSV.','',
'| Event | Fresh MAPE | Stack MAPE |','|---|---:|---:|']
for e,d in research['research']['per_event'].items():lines.append(f"| {e} | {d['fresh']:.2f}% | {d['stack']:.2f}% |")
d=research['research'];lines += [f"| Equal-event average | {d['fresh_mape_percent']:.2f}% | {d['stack_mape_percent']:.2f}% |",'',
    f"Stack minus Fresh: {d['paired_mape_change_pp']:+.3f} percentage points; exhaustive four-event paired bootstrap 95% interval [{d['paired_event_bootstrap_95_pp'][0]:+.3f}, {d['paired_event_bootstrap_95_pp'][1]:+.3f}] pp. It includes zero. This is descriptive, not adjusted for the feature/scope search. Stack wins three events; Fresh wins 324.",'',
'Fixed-recipe robustness, not another hyperparameter selection','',
'Prespecified seeds 1729–1733 were all reported. Compression, features, scope, architecture, epochs and loss were fixed; no seed or ensemble was promoted.','',
'| Seed | Stack research MAPE, 321–324 | Stack production MAPE, 325 |','|---|---:|---:|']
for seed in research['research_fixed_seeds']:
    a=research['research_fixed_seeds'][seed]['mape']*100;b=research['production_fixed_seeds'][seed]['standing']['mape_percent']
    lines.append(f'| {seed} | {a:.3f}% | {b:.3f}% |')
a=[v['mape']*100 for v in research['research_fixed_seeds'].values()];b=[v['standing']['mape_percent'] for v in research['production_fixed_seeds'].values()]
lines += ['',f'Research seed mean {np.mean(a):.3f}% (range {min(a):.3f}–{max(a):.3f}%) versus Fresh 5.963%. Three of five seeds beat Fresh on this inspected four-event block. Production seed mean {np.mean(b):.3f}% (range {min(b):.3f}–{max(b):.3f}%) versus Fresh 10.635%; all five beat Fresh on 325. This establishes local initialization stability, not robustness across many independent future events.','',
'Fitting, features and isolation checks','',
'- Current stack: exactly 30 supervised events, 12 actual post-rule (311, 314–324) plus 18 most recent older events. Reference pool is 319 events through 324. Fresh supervises only the 12 post-rule events, so the comparison changes both learning method and training scope.',
'- Retraining the documented current recipe reproduced mean/scale/a/b/v/d/trained exactly: maximum absolute difference zero for every saved parameter array. 15,132 supervised tier/checkpoint cells; 600 selected inputs, hidden width 8, 5,024 trainable parameters.',
'- Training cache excludes event 325/326 cases. All inspected same-rank training references have outcomes available before the target event starts. All 319 reference labels were available before the production fit cutoff.',
'- Poisoning 325/326 final labels leaves training targets, weights and all fitted parameters unchanged. Runtime future-tail, late-publication and own-final poisoning leaves forecasts unchanged for all three models on both 325 and 326.',
'- The actual isolated packaged predictions match the independent harness and padded JointPredictor paths on every available tier: 24/24 for 325 and 22/22 for 326.',
'- The 600 selected columns are explicit normalized progress/remaining/current-fraction/presence inputs (96) plus rule/type/reference-support inputs (504). The 16 PCs and 552 direct daily-summary columns are excluded. The feature builder still computes them, but they do not enter the stack net. Historical completion priors remain part of the forecast.',
'- Stack has no calibrated sigma; this verifies point accuracy, not improved probabilistic calibration. Fresh and Ridge retain their own small-sample uncertainty estimates.',
'- The model assets and protected evaluation ledger were unchanged. The current package checksum verifier passes all 71 files.','',
'Pipeline and evidence issues found (left unchanged by this verification)','',
'1. `dist/bandori/stack_runtime.py:59`: any explicit `--max-stale-hours` value calls `config.replace(...)`, but Config has no replace method. Verified failure: AttributeError. Use dataclasses.replace(config, ...) and add a real CLI regression check before advertising stale-input support.',
'2. `README.md:63`: the documented retraining command uses train_postrule30_test325.py, which actually fits 32 events. The current package was reproduced from train_post12_30_test325.py (30 events); the displayed eval_325.py command also evaluates an earlier scope, not that exact production recipe.',
'3. `README.md:32–33`: “325 never seen in selection” overstates the evidence. Several scopes were compared on 325 (original post20, all pre325, 32-event version, corrected 30-event version), and the deployment/export includes its outcomes and asserts their published errors before writing assets. Targets do not leak into the coefficient fit, but 325 is already a development/assessment event, not an untouched final test.',
'4. Small 321–324 gains carry model-search inflation. Do not assign the production package the 5.64% research score or call its improvement statistically established. Event 326 remains the next closing outcome for forecasts frozen beforehand; one more event will still give limited evidence.','',
'Reproduction','',
'```sh',
'OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -B reports/stack_verification_20261019/runtime_audit.py',
'OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B reports/stack_verification_20261019/research_audit.py',
'OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B dist/tools/verify_stack_parity.py --out-dir reports/stack_verification_20261019/runtime_outputs --padded-dir /tmp/bandori_stack_review_parity_20261019',
'.venv/bin/python -B dist/verify_package.py',
'```','',
'No deployment routing, weights, hyperparameters, or existing reports were changed. Only this verification directory was added.','']
(OUT/'verification.md').write_text('\n'.join(lines))
print('Wrote verification.md and comparison.png/.svg')
