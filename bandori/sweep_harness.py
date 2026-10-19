"""Frozen-bank sweep harness: identical rows, splits and scoring for every sweep experiment.

Protocol (mandated for all sweep agents):

* **Train** only with data from the `eight_post_events` fold: 315 training events,
  of which 8 are post-rule (311-320). Fit cutoff 1787720400 (start of event 321).
* **Test** only on the four new post-rule events **321-324** (the fold's validation
  events). The user cares about new events only.
* **Reported MAPE** = event-balanced MAPE on clean checkpoints of events 321-324
  restricted to tiers 500, 1000, 1500, 2000 (the standing convention). Companion
  statistics: signed bias, final-24h MAPE, per-event MAPE.

Fixed across every experiment (the "matched rows"): validation case features are
built exactly as the frozen protocol — `build_cases(data, config, [321-324],
fold_train)` + `make_context`, i.e. the analog reference pool is always the full
315 training events. Experiments vary only (a) which *labeled* rows enter the
supervised fit, (b) the compression/masks fit on those rows, (c) architecture,
loss weights and ensemble rules. Removing old data means removing old *labels*
from the supervised fit; the input features stay the frozen causal features
(Protocol A). Protocol B (rebuilding features with a restricted reference pool)
must be labelled as such and never mixed into the same table as Protocol A.

Selection discipline: any hyperparameter or blend weight chosen by validation
must be chosen on the post-rule *training* events 311-320 only (rolling-origin or
leave-one-out). Scores on 321-324 for a full grid are descriptive; the headline
number is the score of the config that training-side selection picked.

The harness reproduces the frozen run's own numbers when fitting with the full
315-event supervised bank: nn_pca 6.71% MAPE, ridge 7.61% (see validate()).
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .analog import Case
from .config import Config
from .dataset import Dataset
from .evaluation import summary
from .experiments import correction_forecast
from .joint import JointNet, pack  # noqa: F401  (pack re-exported for convenience)
from .advanced_statistics import JointRidge
from .pipeline import digest, training_arrays
from .regime_suite import finish_context, load_context, make_context
from .role_analog import build_cases

FOCAL_TIERS=(500,1000,1500,2000)
VAL_EVENTS=(321,322,323,324)
POST_RULE_TRAIN_EVENTS=tuple(range(311,321))  # 8 post-rule events inside training
SOURCE=Path('runs/rule_aware_3x_20261018')
CACHE=Path('runs/sweep_harness_cache_20261019')

FROZEN_REFERENCE={'nn_pca':dict(mape=.0671,bias=.0080,final24h=.0278),
    'ridge':dict(mape=.0761,bias=.0202,final24h=.0306),
    'nn_minimal':dict(mape=.0761,bias=.0060,final24h=.0335),
    'tcn24':dict(mape=.0768,bias=.0093,final24h=.0348),
    'baseline':dict(mape=.0999,bias=.0078,final24h=.0403),
    'raw_lstm':dict(mape=.0999,bias=-.0854,final24h=.0703),
    'transformer':dict(mape=.1220,bias=-.0913,final24h=.0865),
    'lstm_hourly_60':dict(mape=.0789,bias=-.0288,final24h=.0638)}


def _focal(cases,values):
    keep=np.array([c.tier in FOCAL_TIERS for c in cases])
    return [c for c,k in zip(cases,keep) if k],np.asarray(values)[keep]


def score(cases,values,data):
    """Standing-scope metrics: focal tiers, event-balanced MAPE, clean cases only."""
    fc,fv=_focal(cases,values)
    s=summary(fc,fv,data)
    per_event={}
    for eid in sorted({c.event_id for c in fc}):
        ii=[i for i,c in enumerate(fc) if c.event_id==eid]
        per_event[eid]=summary([fc[i] for i in ii],fv[ii],data)['overall']
    return dict(mape=s['overall']['mape'],bias=s['overall']['bias'],
        final24h=s['by_horizon'].get('0-24h',{}).get('mape'),rows=s['overall']['rows'],
        events=s['overall']['events'],per_event=per_event,full=s)


class Bank:
    """Frozen matched rows plus the prequential training bank."""


def load_bank(data_path='data/full_approaches_20261017',source=SOURCE):
    source=Path(source)
    config=Config.load(source/'config.json')
    data=Dataset.load(data_path,config.tiers);data.add_terminal_labels(config)
    plan=json.loads((source/'plan.json').read_text())
    fold=dict(next(f for f in plan['folds'] if f['name']=='eight_post_events'))
    ctx_all=load_context(source)
    bank=SimpleNamespace(config=config,data=data,fold=fold,ctx_all=ctx_all,
        train_ids=list(fold['train']),val_ids=list(fold['validation']),fit_cutoff=fold['fit_cutoff'],
        reference_pool=list(fold['train']),source=source)
    bank.val=val_context(bank)
    return bank


def _val_cache_key(bank):
    payload=json.dumps(dict(pool=sorted(bank.reference_pool),fit_cutoff=bank.fit_cutoff,
        val=sorted(bank.val_ids),source=str(bank.source.resolve())))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def val_context(bank,rebuild=False):
    """Raw validation context for events 321-324; cached, protocol-frozen."""
    CACHE.mkdir(exist_ok=True)
    key=_val_cache_key(bank);marker=CACHE/f'val_{key}.json'
    if marker.exists() and not rebuild:
        saved=json.loads(marker.read_text())
        cases=[Case(**c) for c in saved['cases']]
        arrays=dict(np.load(CACHE/f'val_{key}.npz',allow_pickle=False))
        ctx=dict(cases=cases,matrices={},shape={},daily={})
        for name,x in arrays.items():
            x=x.copy()
            if name.startswith('matrix_'):ctx['matrices'][name[7:]]=x
            elif name.startswith('daily_'):ctx['daily'][name[6:]]=x
            else:ctx[name]=x
        return ctx
    cases,failures,roles=build_cases(bank.data,bank.config,list(bank.val_ids),bank.reference_pool)
    if failures:print('Validation case failures:',failures,flush=True)
    ctx=make_context(cases,bank.data,bank.config,bank.reference_pool)
    from dataclasses import asdict
    import gzip
    marker.write_text(json.dumps(dict(cases=[asdict(c) for c in ctx['cases']],roles=roles)))
    arrays={k:v for k,v in ctx.items() if isinstance(v,np.ndarray)}
    arrays.update({'matrix_'+n:x for n,x in ctx['matrices'].items()})
    arrays.update({'daily_'+n:x for n,x in ctx['daily'].items()})
    np.savez_compressed(CACHE/f'val_{key}.npz',**arrays)
    return ctx


def supervised_slice(ctx_all,ids):
    """Training rows for `ids`, keeping the frozen full-history case features.

    Unlike regime_suite.subset this does not enforce reference containment:
    features are fixed inputs (Protocol A). Only labels of excluded events are
    kept out of the supervised fit.
    """
    allowed=set(ids)
    case_mask=np.array([c.event_id in allowed for c in ctx_all['cases']])
    case_indices=np.flatnonzero(case_mask)
    mapping=np.full(len(case_mask),-1,int);mapping[case_indices]=np.arange(len(case_indices))
    rows=np.array([any(i>=0 and case_mask[i] for i in row) for row in ctx_all['index']])
    index=ctx_all['index'][rows].copy();valid=index>=0;index[valid]=mapping[index[valid]]
    out=dict(cases=[ctx_all['cases'][i] for i in case_indices],index=index,bins=ctx_all['bins'][rows],
        matrices={n:x[rows].copy() for n,x in ctx_all['matrices'].items()},
        surface=ctx_all['surface'][case_mask].copy(),rule_metadata=ctx_all['rule_metadata'][rows].copy())
    out['daily']={n:(x[rows].copy() if n in ('summaries','sequence') else x[case_mask].copy())
        for n,x in ctx_all['daily'].items()}
    return out


def prepare_fit(bank,supervised_ids=None):
    """Supervised training matrices for `supervised_ids` (default: all 315).

    Returns fitted compression and masks — fit only on the supervised rows, as
    train-only preprocessing must be. Validation finishing uses these.
    """
    from .representations import Compression
    if supervised_ids is None:supervised_ids=bank.train_ids
    train=supervised_slice(bank.ctx_all,supervised_ids)
    targets,weights=training_arrays(train['cases'],train['index'],bank.data,bank.config,bank.fit_cutoff)
    rich=train['matrices']['rich'].copy()
    compression=Compression().fit(rich,weights,bank.config.tiers,seed=bank.config.seed)
    train,masked=finish_context(train,compression,weights=weights)
    prep=SimpleNamespace(train=train,targets=targets,weights=weights,bins=train['bins'],
        compression=compression,masks=masked,supervised_ids=list(supervised_ids),
        matrix={n:train['matrices'][n] for n in train['matrices']})
    prep.val=val_matrices(bank,compression,masked)
    return prep


def val_matrices(bank,compression,masks):
    ctx=copy.deepcopy(bank.val)
    ctx,_=finish_context(ctx,compression,masks)
    return ctx


def predict(cases,index,values_by_case,gain=1.0,kind='nn'):
    """Map per-cell corrections to case predictions, same as the frozen pipeline."""
    return correction_forecast(cases,index,values_by_case,gain,kind)


def fit_jointnet(matrix,targets,weights,bins,config,hidden=8,seed=None,epochs=None,
                 learning_rate=None,decay=None):
    from dataclasses import replace
    changes={}
    if seed is not None:changes['seed']=seed
    if epochs is not None:changes['epochs']=epochs
    if learning_rate is not None:changes['learning_rate']=learning_rate
    if decay is not None:changes['weight_decay']=decay
    cfg=replace(config,**changes) if changes else config
    return JointNet(hidden,cfg.seed).fit(matrix,targets,weights,bins,cfg)


def fit_ridge(matrix,targets,weights,bins,strength=.01,bias_penalty=3):
    return JointRidge().fit(matrix,targets,weights,bins,strength,bias_penalty)


def evaluate(bank,prep,model,matrix_name='pca',gain=1.0,kind='nn',predict_fn=None):
    val=prep.val
    if predict_fn is not None:
        values=predict_fn(val)
    else:
        delta=model.predict(val['matrices'][matrix_name])
        values=correction_forecast(val['cases'],val['index'],delta,gain,kind)
    return score(val['cases'],values,bank.data),values


def evaluate_raw(bank,values):
    """Score an explicit per-case prediction vector on the frozen validation cases."""
    return score(bank.val['cases'],values,bank.data)


def fmt(result):
    f=lambda v:'—' if v is None else f'{100*v:.2f}%'
    return f"MAPE {f(result['mape'])}  bias {f(result['bias'])}  final24h {f(result['final24h'])}  rows {result['rows']}"


def validate(bank=None):
    """Reproduce frozen nn_pca and ridge on the matched rows; returns both results."""
    if bank is None:bank=load_bank()
    prep=prepare_fit(bank)
    nn=fit_jointnet(prep.matrix['pca'],prep.targets,prep.weights,prep.bins,bank.config,hidden=8,seed=1729)
    res_nn,_=evaluate(bank,prep,nn,'pca',gain=1.0,kind='nn')
    ridge=fit_ridge(prep.matrix['pca'],prep.targets,prep.weights,prep.bins)
    res_ridge,_=evaluate(bank,prep,ridge,'pca',gain=.5,kind='ridge')
    expected_nn=FROZEN_REFERENCE['nn_pca'];expected_ridge=FROZEN_REFERENCE['ridge']
    for name,res,expected in (('nn_pca',res_nn,expected_nn),('ridge',res_ridge,expected_ridge)):
        if abs(res['mape']-expected['mape'])>5e-4:
            raise ValueError(f'{name} reproduction failed: {res["mape"]:.4f} vs frozen {expected["mape"]:.4f}')
        print(f'validated {name}: {fmt(res)} (frozen {expected["mape"]*100:.2f}%)',flush=True)
    return res_nn,res_ridge


def per_event_line(result):
    return '  '.join(f'{e}:{100*v["mape"]:.2f}%' for e,v in sorted(result['per_event'].items()))


def write_report(path,title,sections):
    lines=[f'# {title}','']
    for head,body in sections:
        if head:lines+=[f'## {head}','']
        lines+=body+['']
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text('\n'.join(lines))
    print('wrote',path,flush=True)


def save_results(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(payload,plan_sha256=digest(str(SOURCE/'plan.json')) if (SOURCE/'plan.json').exists() else None)
    path.write_text(json.dumps(payload,indent=1,default=float))
    print('wrote',path,flush=True)
