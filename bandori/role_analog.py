"""Reward-role completion priors without manufacturing training labels."""
from datetime import datetime
import math

from collect_bandori_data import CN, iso
from .advanced_statistics import CachedAnalog
from .analog import Case, FEATURES, active_hours, weighted_quantile
from .full_refit import issue_times
from .rules import era


class RoleAwareAnalog(CachedAnalog):
    def __init__(self,*args):super().__init__(*args);self.role_audit={}

    def build(self,event,tier,cutoff):
        try:return super().build(event,tier,cutoff)
        except ValueError as error:
            if tier!=1500 or not era(event) or not str(error).startswith('Only '):raise
        config=self.config;event,versioned=self.data.known_event(event,cutoff)
        curve=self.data.curves.get((event.event_id,tier));point=curve.last(cutoff,event.start) if curve is not None else None
        if point is None or point[1]<=0:raise ValueError('No positive T1500 score')
        observed,current=point
        if cutoff-observed>config.max_stale_minutes*60:raise ValueError('Latest score is stale')
        duration=event.end-event.start;progress=(observed-event.start)/duration;remaining=(event.end-observed)/3600
        rates,windows=self.recent_rates(event,curve,observed,current,cutoff)
        predictions=[];weights=[];pace=[];pace_weights=[];direct=[];roles=[]
        for eid in sorted(self.training_ids):
            ref=self.data.events[eid]
            if eid==event.event_id or self.data.label_available(eid,config)>event.start:continue
            if ref.start<event.start-730*86400:continue
            source_tier=1500 if era(ref) else 1000
            key=(eid,source_tier)
            if key not in self.data.labels or key not in self.data.curves:continue
            final=self.data.labels[key]
            value=self.sample(eid,source_tier,ref.start+progress*(ref.end-ref.start),event.start)
            if value is None or not 0<value<=final:continue
            weight=config.event_weight((event.start-ref.end)/86400,ref.start)
            weight*=2 if event.event_type==ref.event_type else 1
            weight*=math.exp(-abs(math.log(duration/(ref.end-ref.start)))/.5)
            if self.data.label_sources.get(key)=='post_close_tracker_proxy':weight*=config.proxy_label_weight
            predictions.append(current*final/value);weights.append(weight);roles.append((eid,source_tier))
            if source_tier==1500:direct.append(eid)
            anchor=ref.end-remaining*3600;level=self.sample(eid,source_tier,anchor,event.start);possibilities=[]
            if level is not None and 0<=level<final:
                for hours,rate in rates.items():
                    start=anchor-windows[hours]*3600;before=self.sample(eid,source_tier,start,event.start)
                    exposure=active_hours(self.data,ref,start,anchor,event.start)
                    if before is not None and level>before and exposure>0:possibilities.append(current+rate*(final-level)/((level-before)/exposure))
            if possibilities:pace.append(float(sorted(possibilities)[len(possibilities)//2]));pace_weights.append(weight)
        if len(predictions)<config.min_reference_events:raise ValueError('Insufficient direct or reward-role references')
        baseline=max(current,weighted_quantile(predictions,weights))
        pace_value=max(current,weighted_quantile(pace,pace_weights)) if len(pace)>=config.min_reference_events else baseline
        local=datetime.fromtimestamp(cutoff,CN);effective=sum(weights)**2/sum(w*w for w in weights)
        features=dict(progress=progress,remaining_days=remaining/24,duration_days=duration/86400,
            staleness_hours=(cutoff-observed)/3600,log_current=math.log1p(current)/16,current_fraction=current/baseline,
            future_fraction=(baseline-current)/baseline,history_spread=(weighted_quantile(predictions,weights,.75)-weighted_quantile(predictions,weights,.25))/baseline,
            history_effective_n=math.log1p(effective),clock_sin=math.sin(2*math.pi*local.hour/24),clock_cos=math.cos(2*math.pi*local.hour/24),
            weekend=float(local.weekday()>=5),role_transfer_support=sum(t==1000 for e,t in roles),direct_tier_references=len(direct))
        features.update({f'rate_{h}h':rates.get(h,0)*24/baseline for h in (6,12,24)})
        self.role_audit[(event.event_id,tier,cutoff)]=roles
        # References used by same-rank trajectory and daily modules remain
        # actual T1500 curves only. Cross-rank priors are separately audited.
        return Case(event.event_id,tier,cutoff,current,observed,(event.end-cutoff)/3600,baseline,pace_value,
                    features,tuple(direct),len(pace),versioned)


def build_cases(data,config,ids,references,hours=(8,20,21)):
    predictor=RoleAwareAnalog(data,config,references);cases=[];failures=[]
    for eid in ids:
        event=data.events[eid]
        for cutoff in issue_times(event,hours):
            for tier in config.tiers:
                if (eid,tier) not in data.curves:continue
                try:cases.append(predictor.build(event,tier,cutoff))
                except ValueError as error:failures.append(dict(event_id=eid,tier=tier,checkpoint=iso(cutoff),reason=str(error)))
    roles=[dict(event_id=e,tier=t,cutoff=c,references=[dict(event_id=eid,tier=rank) for eid,rank in refs]) for (e,t,c),refs in predictor.role_audit.items()]
    return cases,failures,roles
