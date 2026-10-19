"""Fresh causal physical anchors; no old feature caches or fitted models."""
from dataclasses import dataclass
import math

import numpy as np

from bandori.analog import active_hours
from .data import TYPES,issue_times,restrict


CORE=('intercept','pace_disagreement','rate_6h_vs_24h','same_clock_day_change','progress',
      'closing_fraction','log_duration','history_spread','staleness','clock_sin','clock_cos','cross_tier_pressure',
      'pace_available','rate_contrast_available','same_clock_available','completion_available')
FEATURES=CORE+tuple('type_'+t for t in TYPES)
DEVIATIONS=('intercept','pace_disagreement','closing_fraction','cross_tier_pressure')
DEVIATION_INDICES=np.array([FEATURES.index(f) for f in DEVIATIONS])


def quantile(values,weights,q=.5):
    values=np.asarray(values);weights=np.asarray(weights);i=np.argsort(values,kind='stable')
    return float(np.interp(q*weights.sum(),np.cumsum(weights[i])-.5*weights[i],values[i]))


@dataclass
class Point:
    event_id:int
    tier:int
    cutoff:float
    observed_at:float
    current:float
    anchor:float
    pace:float
    hours_remaining:float
    features:np.ndarray
    references:tuple
    anchor_method:str

    @property
    def remaining_hours(self):return self.hours_remaining


class Anchors:
    def __init__(self,history,policy):
        self.history=history;self.policy=policy;self.cache={};self.publication={};self.config=policy.label_config()

    def sample(self,eid,tier,t,known_at):
        event=self.history.events[eid];key=(eid,tier);curve=self.history.curves[key]
        if key not in self.publication:
            pre=curve.times<event.end;self.publication[key]=float(np.max(curve.available[pre])) if pre.any() else 0.
        publication=self.publication[key]
        cache_key=(eid,tier,None if publication<=known_at else known_at)
        if cache_key not in self.cache:
            c=curve.prefix(known_at,event.start);keep=c.times<event.end
            times=np.r_[event.start,c.times[keep],event.end];scores=np.r_[0.,c.scores[keep],self.history.labels[key]]
            self.cache[cache_key]=(times,scores)
        times,scores=self.cache[cache_key];i=int(np.searchsorted(times,t))
        if i<len(times) and abs(times[i]-t)<1e-3:return float(scores[i])
        if i==0 or i==len(times) or times[i]-times[i-1]>self.policy.reference_gap_hours*3600:return None
        return float(np.interp(t,times[i-1:i+1],scores[i-1:i+1]))

    def rate(self,live,event,curve,t,h,cutoff):
        old=curve.last(t-h*3600,event.start)
        now=curve.last(t,event.start)
        if old is None or now is None or t-h*3600-old[0]>min(3,h/4)*3600:return None
        keep=(curve.times>=old[0])&(curve.times<=t)
        if keep.sum()<2 or np.max(np.diff(curve.times[keep]))>min(12,h/2)*3600 or np.any(np.diff(curve.scores[keep])<0):return None
        exposure=active_hours(live,event,old[0],t,cutoff)
        if exposure<=0 or now[1]<=old[1]:return None
        return (now[1]-old[1])/exposure,(t-old[0])/3600

    def point(self,live,eid,tier,cutoff,max_stale_hours=None):
        p=self.policy;event=live.known_event(live.events[eid],cutoff)[0]
        if not event.start<cutoff<event.end:raise ValueError('Outside known schedule')
        c=live.curves.get((eid,tier));c=None if c is None else c.prefix(cutoff,event.start)
        point=None if c is None else c.last(cutoff,event.start)
        if point is None or point[1]<=0:raise ValueError('No positive published score')
        observed,current=point;age=(cutoff-observed)/3600
        if age>(p.freshness_hours if max_stale_hours is None else max_stale_hours):raise ValueError('Stale published score')
        duration=event.end-event.start;progress=(observed-event.start)/duration;remaining=(event.end-observed)/3600
        rates={h:self.rate(live,event,c,observed,h,cutoff) for h in (6,24)}
        values=[];weights=[];paces=[];pace_weights=[];references=[]
        for rid in sorted(self.history.events):
            ref=self.history.events[rid]
            if rid==eid or self.history.label_available(rid,self.config)>event.start:continue
            source_tier=1000 if tier==1500 and event.start>=p.rule_at and ref.start<p.rule_at else tier
            key=(rid,source_tier)
            if key not in self.history.labels or key not in self.history.curves:continue
            final=self.history.labels[key];rd=ref.end-ref.start
            level=self.sample(rid,source_tier,ref.start+progress*rd,event.start)
            if level is None or not 0<level<=final:continue
            w=float(p.event_weight(ref,event.start))*(2 if event.event_type==ref.event_type else 1)*math.exp(-abs(math.log(duration/rd))/.5)
            if self.history.label_sources.get(key)=='post_close_tracker_proxy':w*=p.proxy_weight
            values.append(current*final/level);weights.append(w);references.append((rid,source_tier))
            end=ref.end-remaining*3600;at=self.sample(rid,source_tier,end,event.start);options=[]
            if at is not None and 0<=at<final:
                for rate in rates.values():
                    if rate is None:continue
                    strength,window=rate;before=self.sample(rid,source_tier,end-window*3600,event.start)
                    exposure=active_hours(self.history,ref,end-window*3600,end,event.start)
                    if before is not None and at>before and exposure>0:options.append(current+strength*(final-at)/((at-before)/exposure))
            if options:paces.append(float(np.median(options)));pace_weights.append(w)
        if len(values)>=p.min_references:
            anchor=max(current,quantile(values,weights));method='historical_completion'
            spread=(quantile(values,weights,.75)-quantile(values,weights,.25))/max(anchor,1)
        else:
            anchor=current+current*remaining/max(1,(observed-event.start)/3600);method='linear_fallback';spread=0.
        pace=max(current,quantile(paces,pace_weights)) if len(paces)>=p.min_references else anchor
        future=max(anchor-current,current*.01,1)
        ratio=lambda a,b:float(np.clip(math.log(max(a,1e-9)/max(b,1e-9)),-3,3))
        recent=rates[6];daily=rates[24]
        yesterday=self.rate(live,event,c,observed-24*3600,6,cutoff) if observed-24*3600>event.start else None
        short_long=ratio(recent[0],daily[0]) if recent is not None and daily is not None else 0.
        same_clock=ratio(recent[0],yesterday[0]) if recent is not None and yesterday is not None else 0.
        clock=(cutoff/3600+8)%24;tag=event.event_type if event.event_type in TYPES else 'unknown'
        features=np.array([1.,np.clip((pace-anchor)/future,-1,3),short_long,same_clock,progress,
            24/(24+remaining),math.log(duration/86400/8),min(3,spread),min(48,age)/24,
            math.sin(2*math.pi*clock/24),math.cos(2*math.pi*clock/24),0.,
            float(len(paces)>=p.min_references),float(recent is not None and daily is not None),
            float(recent is not None and yesterday is not None),float(method=='historical_completion'),
            *[float(tag==t) for t in TYPES]])
        return Point(eid,tier,cutoff,observed,current,anchor,pace,(event.end-cutoff)/3600,features,tuple(references),method)

    def checkpoint(self,live,eid,cutoff,max_stale_hours=None,tiers=None):
        points=[];failures=[]
        for tier in self.policy.tiers if tiers is None else tiers:
            try:points.append(self.point(live,eid,tier,cutoff,max_stale_hours))
            except ValueError as error:failures.append(dict(tier=tier,reason=str(error)))
        # Cross-tier input is a function of the known score lines only.
        pressure=float(np.median([c.features[1] for c in points])) if points else 0.
        for c in points:c.features[FEATURES.index('cross_tier_pressure')]=pressure
        return points,failures


def build_cases(history,live,ids,policy,progress=None):
    builder=Anchors(history,policy);result=[]
    for eid in ids:
        target=restrict(live,[eid],labels=False)
        for t in issue_times(target.events[eid],policy.checkpoint_hours):
            points,_=builder.checkpoint(target,eid,t);result.extend(points)
        if progress:progress(eid,len(result))
    return result
