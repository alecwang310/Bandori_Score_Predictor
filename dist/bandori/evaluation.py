"""Event-balanced error measures and explicitly grouped uncertainty estimates."""
from collections import defaultdict
import math

import numpy as np

from .analog import HORIZONS,horizon


def metrics(cases,predictions,data):
    groups=defaultdict(list);errors=[];actual=[];predicted=[]
    for c,p in zip(cases,predictions):
        truth=data.labels.get((c.event_id,c.tier))
        if truth is None:continue
        error=float(p/truth-1)
        groups[c.event_id].append(error);errors.append(error);actual.append(truth);predicted.append(p)
    if not errors:return {'rows':0,'events':0}
    return dict(rows=len(errors),events=len(groups),
                mape=float(np.mean([np.mean(np.abs(v)) for v in groups.values()])),
                bias=float(np.mean([np.mean(v) for v in groups.values()])),
                row_bias=float(np.mean(errors)),
                log_rmse=float(np.sqrt(np.mean(np.log(np.asarray(predicted)/actual)**2))),
                wape=float(np.sum(np.abs(np.asarray(predicted)-actual))/sum(actual)))


def breakdown(cases,predictions,data,key):
    groups=defaultdict(list)
    for i,c in enumerate(cases):groups[key(c)].append(i)
    return {str(k):metrics([cases[i] for i in ii],np.asarray(predictions)[ii],data) for k,ii in groups.items()}


def summary(cases,predictions,data):
    cells=breakdown(cases,predictions,data,lambda c:(c.tier,horizon(c.remaining_hours)))
    values=[v for v in cells.values() if v['rows']]
    return {'overall':metrics(cases,predictions,data),
            'by_horizon':breakdown(cases,predictions,data,lambda c:HORIZONS[horizon(c.remaining_hours)]),
            'by_tier':breakdown(cases,predictions,data,lambda c:c.tier),
            'by_label_source':breakdown(cases,predictions,data,lambda c:data.label_sources.get((c.event_id,c.tier),'missing')),
            'tier_horizon_cells':cells,
            'selection_score':float(np.mean([v['mape']+abs(v['bias']) for v in values])) if values else None}


def paired_bootstrap(cases,baseline,predictions,data,seed):
    groups=defaultdict(list)
    for c,b,p in zip(cases,baseline,predictions):
        truth=data.labels.get((c.event_id,c.tier))
        if truth is not None:groups[c.event_id].append(abs(p/truth-1)-abs(b/truth-1))
    if len(groups)<2:return None
    changes=np.array([np.mean(v) for v in groups.values()])
    means=np.random.default_rng(seed).choice(changes,(2000,len(changes)),replace=True).mean(axis=1)
    return {'mape_change':float(changes.mean()),'interval_95':[float(v) for v in np.quantile(means,[.025,.975])],
            'events':len(changes),'unit':'independent event clusters; negative favors the competitor'}


def interval_radius(cases,predictions,data,alpha):
    scores={};tiers=defaultdict(dict)
    for c,p in zip(cases,predictions):
        truth=data.labels.get((c.event_id,c.tier))
        if truth is not None:
            error=abs(math.log(p/truth))
            scores[c.event_id]=max(scores.get(c.event_id,0),error)
            tiers[c.tier][c.event_id]=max(tiers[c.tier].get(c.event_id,0),error)
    def quantile(values):
        rank=math.ceil((len(values)+1)*(1-alpha))
        radius=sorted(values.values())[rank-1] if 0<rank<=len(values) else None
        return {'log_radius':radius,'events':len(values),'nominal_alpha':alpha}
    result=quantile(scores)
    result.update(by_tier={str(t):quantile(values) for t,values in tiers.items()},
                  scope='Event maxima over observed labels; per-tier bands require enough calibration events for that tier; exchangeability is unverified')
    return result
