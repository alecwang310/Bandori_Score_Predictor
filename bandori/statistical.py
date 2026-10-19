"""An interpretable pace/completion mixture with training-only calibration."""
from collections import defaultdict

import numpy as np

from .analog import horizon, reconcile


def uncalibrated(cases,alpha):
    return np.array([(1-alpha)*c.baseline+alpha*c.pace for c in cases])


def fit_calibration(cases,data,config,alpha,fit_cutoff):
    """Prequential training forecasts alone determine per-tier/horizon factors.

    Factors aim for zero weighted relative bias in future points, then shrink
    toward one according to the number of independent events. No held-out
    label, including tuning or interval-calibration labels, enters this fit.
    """
    groups=defaultdict(list)
    predictions=uncalibrated(cases,alpha)
    for case,prediction in zip(cases,predictions):
        truth=data.labels.get((case.event_id,case.tier))
        if truth is not None and truth>=case.current:
            groups[(case.tier,horizon(case.remaining_hours))].append((case,prediction,truth))
    factors={}
    for (tier,bin_number),rows in groups.items():
        events=defaultdict(list)
        for c,p,y in rows:events[c.event_id].append((c,p,y))
        ages={eid:config.event_weight((fit_cutoff-data.events[eid].end)/86400,data.events[eid].start) for eid in events}
        numerator=denominator=0.
        for eid,rr in events.items():
            for c,p,y in rr:
                quality=config.proxy_label_weight if data.label_sources.get((eid,tier))=='post_close_tracker_proxy' else 1
                weight=ages[eid]*quality/len(rr)
                numerator+=weight*(y-c.current)/y
                denominator+=weight*max(0,p-c.current)/y
        effective=sum(ages.values())**2/sum(a*a for a in ages.values())
        raw=numerator/denominator if denominator>0 else 1.
        shrink=effective/(effective+config.calibration_prior_events)
        factor=float(np.clip(1+shrink*(raw-1),.5,2)) if len(events)>=config.min_reference_events else 1.
        factors[f'{tier}:{bin_number}']={'factor':factor,'events':len(events),'effective_events':effective}
    return factors


def predict(cases,alpha,factors):
    base=uncalibrated(cases,alpha)
    values=[c.current+max(0,p-c.current)*factors.get(f'{c.tier}:{horizon(c.remaining_hours)}',{}).get('factor',1)
            for c,p in zip(cases,base)]
    return reconcile(cases,values)
