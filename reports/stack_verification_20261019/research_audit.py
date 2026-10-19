"""Fixed-recipe reproduction and seed sensitivity; no choice or assets updated."""
import csv,itertools,json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from bandori import sweep_harness as H
from bandori.sweep_data_scope import post_plus_recent
from bandori.sweep_features import block_cols
from bandori.experiments import correction_forecast
from bandori.joint import JointNet
from bandori.pipeline import digest,training_arrays
from bandori.regime_suite import finish_context,make_context
from bandori.role_analog import build_cases
from collect_bandori_data import timestamp

OUT=Path(__file__).resolve().parent
SEEDS=(1729,1730,1731,1732,1733)
FIELDS=('mean','scale','a','b','v','d','trained')

def compare_models(a,b):return {name:float(np.max(np.abs(getattr(a,name).astype(float)-getattr(b,name).astype(float)))) for name in FIELDS}
def main():
    protected=digest(ROOT/'evaluation_registry.json')
    bank=H.load_bank();cols=block_cols({'rule','expl'})
    deployment=json.loads((ROOT/'dist/models/stack/deployment.json').read_text())
    package=JointNet.load(ROOT/'dist/models/stack/joint_model.npz')
    cached_ids={c.event_id for c in bank.ctx_all['cases']}
    assert not {325,326}&cached_ids
    bad=[]
    for case in bank.ctx_all['cases']:
        for ref in case.references:
            if ref==case.event_id or bank.data.label_available(ref,bank.config)>bank.data.events[case.event_id].start:
                bad.append((case.event_id,ref))
    assert not bad
    fresh={}
    with (ROOT/'runs/fresh_post_rule_v1_20261019/historical_predictions.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['tier']) in H.FOCAL_TIERS and r['actual']:
                fresh[(int(r['event_id']),int(r['tier']),timestamp(r['checkpoint']))]=r
    reference_summary={}
    raw_rows=[];seed_results={}
    # Research reproduction: fit before event 321; the production stack cannot score its training events honestly.
    ids=post_plus_recent(bank,20);prep=H.prepare_fit(bank,ids)
    assert not set(ids)&set(H.VAL_EVENTS)
    train=np.ascontiguousarray(prep.matrix['pca'][:,cols]);val=np.ascontiguousarray(prep.val['matrices']['pca'][:,cols])
    for seed in SEEDS:
        net=H.fit_jointnet(train,prep.targets,prep.weights,prep.bins,bank.config,hidden=8,seed=seed)
        preds=correction_forecast(prep.val['cases'],prep.val['index'],net.predict(val),1.0,'nn')
        score=H.score(prep.val['cases'],preds,bank.data)
        seed_results[str(seed)]={k:score[k] for k in ('mape','bias','final24h','rows','events','per_event')}
        print(f'Research seed {seed}: {100*score["mape"]:.3f}%',flush=True)
        if seed==1729:
            assert abs(score['mape']-.0564)<.0005
            for c,p in zip(prep.val['cases'],preds):
                key=(c.event_id,c.tier,c.cutoff)
                if key not in fresh:continue
                r=fresh[key];actual=float(r['actual'])
                assert actual==bank.data.labels[(c.event_id,c.tier)]
                raw_rows.append(dict(event_id=c.event_id,tier=c.tier,cutoff=c.cutoff,hours_remaining=c.remaining_hours,
                    actual=actual,stack=float(p),fresh=float(r['final']),
                    stack_absolute_error_percent=abs(p/actual-1)*100,
                    fresh_absolute_error_percent=abs(float(r['final'])/actual-1)*100))
    assert len(raw_rows)==len(fresh)==312
    per_event={}
    for eid in H.VAL_EVENTS:
        rr=[r for r in raw_rows if r['event_id']==eid]
        per_event[eid]={name:float(np.mean([r[name+'_absolute_error_percent'] for r in rr])) for name in ('stack','fresh')}
    changes=np.array([per_event[e]['stack']-per_event[e]['fresh'] for e in H.VAL_EVENTS])
    # Enumerate the entire four-event bootstrap, rather than treating 312 rows as independent.
    bootstrap=np.array([changes[list(idx)].mean() for idx in itertools.product(range(4),repeat=4)])
    reference_summary=dict(per_event=per_event,stack_mape_percent=float(np.mean([v['stack'] for v in per_event.values()])),
        fresh_mape_percent=float(np.mean([v['fresh'] for v in per_event.values()])),
        paired_mape_change_pp=float(changes.mean()),paired_event_bootstrap_95_pp=np.quantile(bootstrap,[.025,.975]).tolist(),
        warning='Four inspected events, winner selected after comparing this block; interval is descriptive and not selection-adjusted',
        supervised_event_ids=ids)
    # Reproduce the exact currently packaged production recipe, with no use of 325 for choosing seeds.
    bank.fit_cutoff=deployment['fit_cutoff'];ids=deployment['supervised_event_ids']
    assert len(ids)==30 and not {325,326}&set(ids)
    assert all(bank.data.label_available(e,bank.config)<=bank.fit_cutoff for e in deployment['training_event_ids'])
    prod=H.prepare_fit(bank,ids);train=np.ascontiguousarray(prod.matrix['pca'][:,cols])
    assert np.count_nonzero(prod.weights)==deployment['weighted_cells']==15132
    cases,failures,_=build_cases(bank.data,bank.config,[325],deployment['training_event_ids'])
    ctx=finish_context(make_context(cases,bank.data,bank.config,deployment['training_event_ids']),prod.compression,prod.masks)[0]
    val=np.ascontiguousarray(ctx['matrices']['pca'][:,cols])
    ext=list(csv.DictReader((ROOT/'reports/fresh_event325_external_20261019/comparison.csv').open()))
    ext_keys={(timestamp(r['issued_at']),int(r['tier'])) for r in ext if r['fresh_input']=='True'}
    prod_seed_results={};reproduction=None;first_net=None
    for seed in SEEDS:
        net=H.fit_jointnet(train,prod.targets,prod.weights,prod.bins,bank.config,hidden=8,seed=seed)
        preds=correction_forecast(ctx['cases'],ctx['index'],net.predict(val),1.0,'nn')
        def metric(keep):
            ee=[preds[i]/bank.data.labels[(325,c.tier)]-1 for i,c in enumerate(cases) if keep(c)]
            return dict(rows=len(ee),mape_percent=float(np.mean(np.abs(ee))*100),bias_percent=float(np.mean(ee)*100))
        prod_seed_results[str(seed)]=dict(standing=metric(lambda c:c.tier in H.FOCAL_TIERS),
            external_24=metric(lambda c:(c.cutoff,c.tier) in ext_keys))
        print(f'Production seed {seed}: standing {prod_seed_results[str(seed)]["standing"]["mape_percent"]:.3f}%',flush=True)
        if seed==1729:
            reproduction=compare_models(net,package);first_net=net
            assert all(v<1e-10 for v in reproduction.values()),reproduction
    # Training-stage target poisoning: 325/326 labels cannot change targets or the fitted network.
    for key in list(bank.data.labels):
        if key[0] in (325,326):bank.data.labels[key]*=1e9
    targets,weights=training_arrays(prod.train['cases'],prod.train['index'],bank.data,bank.config,bank.fit_cutoff)
    np.testing.assert_array_equal(targets,prod.targets);np.testing.assert_array_equal(weights,prod.weights)
    poisoned=H.fit_jointnet(train,targets,weights,prod.bins,bank.config,hidden=8,seed=1729)
    assert all(v==0 for v in compare_models(first_net,poisoned).values())
    assert protected==digest(ROOT/'evaluation_registry.json')
    report=dict(research=reference_summary,research_fixed_seeds=seed_results,production_fixed_seeds=prod_seed_results,
        package_parameter_reproduction_max_abs_difference=reproduction,
        checks=dict(training_reference_dates_causal=True,training_cache_excludes_325_326=True,
            training_targets_and_weights_unchanged_by_325_326_label_poisoning=True,
            fitted_parameters_unchanged_by_label_poisoning=True,protected_ledger_unchanged=True),
        seed_protocol='Prespecified seeds 1729–1733; no winning seed or ensemble selected, all packaged assets unchanged')
    (OUT/'research_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'research_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw_rows[0]));w.writeheader();w.writerows(raw_rows)
    print(json.dumps(reference_summary,indent=2),flush=True)

if __name__=='__main__':main()
