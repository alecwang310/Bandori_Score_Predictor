"""Causal hourly observations and training-reference-only circadian pace.

No final score of the event being observed is read by this module. Historical
curves may be used only after their labels were available before its start.
"""
import math

import numpy as np

from .analog import active_hours, weighted_quantile


CHANNELS=('log_rate', 'relative_rate', 'day_change', 'available',
          'previous_available', 'active_fraction', 'negative', 'log_cycle',
          'clock_sin', 'clock_cos')
SUMMARIES=('coverage', 'previous_coverage', 'active_fraction', 'rate_24h',
           'rate_6h', 'rate_3h', 'rate_std', 'zero_fraction', 'negative_fraction',
           'peak_share', 'night_day_log_ratio', 'harmonic_sin', 'harmonic_cos',
           'harmonic2_sin', 'harmonic2_cos', 'recent_early_log_ratio',
           'day_change', 'day_correlation', 'recent_cycle', 'future_cycle',
           'pace_delta', 'profile_support', 'staleness')


def clock_index(times):
    local=np.asarray(times)/3600+8
    return np.floor(local).astype(int)%24, (np.floor(local/24).astype(int)+3)%7


def sample(curve, points, max_gap_hours=2):
    """Interpolate only within an already publication-filtered prefix."""
    points=np.asarray(points,float);values=np.zeros(points.shape);valid=np.zeros(points.shape,bool)
    if not len(curve.times):return values,valid
    right=np.searchsorted(curve.times,points,side='left')
    exact=right<len(curve.times)
    exact[exact]=np.abs(curve.times[right[exact]]-points[exact])<1e-3
    values[exact]=curve.scores[right[exact]];valid[exact]=True
    inside=(right>0)&(right<len(curve.times))&~exact
    ii=np.flatnonzero(inside);r=right[ii];l=r-1
    good=curve.times[r]-curve.times[l]<=max_gap_hours*3600
    ii,r,l=ii[good],r[good],l[good]
    fraction=(points[ii]-curve.times[l])/(curve.times[r]-curve.times[l])
    values[ii]=curve.scores[l]+fraction*(curve.scores[r]-curve.scores[l]);valid[ii]=True
    return values,valid


def calendar_integral(profile,start,end):
    """Exact integral of a weekday/weekend hourly function, in hours."""
    week=np.concatenate([profile[int(day>=5)] for day in range(7)])
    cumulative=np.r_[0,np.cumsum(week)]
    def primitive(t):
        h=np.asarray(t)/3600+8+72  # Unix epoch was Thursday; Monday is index 0.
        whole=np.floor(h).astype(np.int64);period=np.floor_divide(whole,168);slot=whole%168
        return period*cumulative[-1]+cumulative[slot]+(h-whole)*week[slot]
    return primitive(end)-primitive(start)


def activity_integral(profile,data,event,start,end,known_at):
    total=float(calendar_integral(profile,start,end));right=start
    outages=sorted((max(start,a),min(end,b)) for eid,a,b,k in data.outages
                   if eid==event.event_id and k<=known_at and a<end and b>start)
    for a,b in outages:
        left=max(a,right)
        if b>left:total-=float(calendar_integral(profile,left,b))
        right=max(right,b)
    return max(0,total)


class DailyObserver:
    def __init__(self,data,config,references):
        self.data,self.config=data,config
        self.references=set(references);self.curve_cache={};self.profile_cache={}
        self.availability_cache={};self.shape_cache={}

    def reference_curve(self,eid,tier,known_at):
        event=self.data.known_event(self.data.events[eid],known_at)[0]
        base=(eid,tier,event.start,event.end)
        if base not in self.availability_cache:
            curve=self.data.curves[(eid,tier)]
            relevant=(curve.times>=event.start)&(curve.times<=event.end)
            self.availability_cache[base]=float(np.max(curve.available[relevant])) if relevant.any() else 0.
        key=(*base,float('inf') if known_at>=self.availability_cache[base] else known_at)
        if key not in self.curve_cache:
            self.curve_cache[key]=self.data.curves[(eid,tier)].prefix(min(known_at,event.end),event.start)
        return self.curve_cache[key]

    def profile(self,case):
        event=self.data.known_event(self.data.events[case.event_id],case.cutoff)[0]
        ids=tuple(e for e in case.references if e in self.references and e!=case.event_id
                  and self.data.label_available(e,self.config)<=event.start)
        key=(case.event_id,case.tier,ids,event.start)
        if key in self.profile_cache:return self.profile_cache[key]
        numerator=np.full((2,24),5.,float);denominator=np.full((2,24),5.,float);mass=[]
        for eid in ids:
            ref=self.data.known_event(self.data.events[eid],event.start)[0]
            curve=self.reference_curve(eid,case.tier,event.start)
            shape_key=(id(curve),ref.start,ref.end)
            if shape_key not in self.shape_cache:
                self.shape_cache[shape_key]=self.reference_shape(curve,ref)
            shape=self.shape_cache[shape_key]
            if shape is None:continue
            normalized,observed=shape
            w=self.config.event_weight((event.start-ref.end)/86400,ref.start)
            if event.event_type==ref.event_type:w*=2
            if self.data.label_sources.get((eid,case.tier))=='post_close_tracker_proxy':w*=self.config.proxy_label_weight
            numerator+=w*normalized*observed;denominator+=w*observed;mass.append(w)
        profile=numerator/denominator
        profile/=float(np.average(profile,weights=np.broadcast_to(np.array([5,2])[:,None],profile.shape)))
        support=sum(mass)**2/sum(w*w for w in mass) if mass else 0.
        self.profile_cache[key]=(profile,support)
        return profile,support

    @staticmethod
    def reference_shape(curve,ref):
        if len(curve.times)<3:return None
        dt=np.diff(curve.times)/3600;ds=np.diff(curve.scores);middle=(curve.times[1:]+curve.times[:-1])/2
        good=(dt>0)&(dt<=2)&(ds>=0)&(middle>=ref.start+12*3600)&(middle<ref.end-24*3600)
        if np.count_nonzero(good)<8:return None
        rates=ds[good]/dt[good];positive=rates[rates>0]
        if not len(positive):return None
        scale=float(np.median(positive));hour,day=clock_index(middle[good]);duration=dt[good]
        sums=np.zeros((2,24));exposure=np.zeros((2,24))
        np.add.at(sums,((day>=5).astype(int),hour),np.clip(rates/scale,0,5)*duration)
        np.add.at(exposure,((day>=5).astype(int),hour),duration)
        for _ in range(2):
            sums=(np.roll(sums,1,axis=1)+2*sums+np.roll(sums,-1,axis=1))/4
            exposure=(np.roll(exposure,1,axis=1)+2*exposure+np.roll(exposure,-1,axis=1))/4
        normalized=np.divide(sums,exposure,out=np.ones_like(sums),where=exposure>0)
        observed=exposure>0
        normalized/=max(.1,float(np.average(normalized[observed],weights=exposure[observed])))
        return normalized,observed

    def window(self,case,event,curve,lag=0):
        edges=case.observed_at-(lag+24)*3600+np.arange(25)*3600
        scores,available=sample(curve,edges)
        active=np.array([active_hours(self.data,event,a,b,case.cutoff) for a,b in zip(edges[:-1],edges[1:])])
        differences=np.diff(scores)
        observed=available[:-1]&available[1:]
        negative=observed&(differences<0)
        valid=observed&~negative&(active>.1)
        rates=np.divide(np.maximum(0,differences),active,out=np.zeros(24),where=valid)
        return edges,rates,valid,active,negative

    def observe(self,case):
        event=self.data.known_event(self.data.events[case.event_id],case.cutoff)[0]
        curve=self.data.curves[(case.event_id,case.tier)].prefix(case.cutoff,event.start)
        profile,support=self.profile(case)
        edges,rate,valid,active,negative=self.window(case,event,curve)
        _,previous,prev_valid,_,_=self.window(case,event,curve,24)
        middle=(edges[:-1]+edges[1:])/2;hour,day=clock_index(middle)
        phase=2*np.pi*((middle/3600+8)%24)/24;cycle=profile[(day>=5).astype(int),hour]
        total=float(np.sum(rate[valid]*active[valid]));exposure=float(active[valid].sum())
        mean=total/max(exposure,1e-6)
        norm=rate*24/max(case.baseline,1)
        relative=np.clip(np.log((rate+max(mean,.01)*.05)/(max(mean,.01)*1.05)),-4,4)
        matched=valid&prev_valid
        change=np.zeros(24)
        change[matched]=np.clip(np.log((rate[matched]+max(mean,.01)*.1)/(previous[matched]+max(mean,.01)*.1)),-3,3)
        tokens=np.column_stack((np.log1p(norm),relative,change,valid,prev_valid,active,negative,
                                np.log(cycle),np.sin(phase),np.cos(phase)))
        tokens[~valid,:3]=0
        def average_rate(mask):
            good=valid&mask
            return float(np.sum(rate[good]*active[good])/max(active[good].sum(),1e-6))*24/max(case.baseline,1)
        all_slots=np.ones(24,bool);night=(hour<8);recent=np.arange(24)>=18;early=np.arange(24)<6
        def logratio(a,b):return float(np.clip(np.log((a+.01)/(b+.01)),-3,3))
        def harmonic(values):return float(np.sum(relative[valid]*values[valid])/max(1,valid.sum()))
        correlation=0.
        if matched.sum()>=6 and np.std(rate[matched])>0 and np.std(previous[matched])>0:
            correlation=float(np.corrcoef(rate[matched],previous[matched])[0,1])
        recent_activity=float(np.sum(active[valid]*cycle[valid]))
        future_activity=activity_integral(profile,self.data,event,case.observed_at,event.end,case.cutoff)
        pace=self.pace(case,event,profile,total,recent_activity,float(valid.mean()))
        features=np.array([valid.mean(),prev_valid.mean(),active.mean(),average_rate(all_slots),
            average_rate(recent),average_rate(np.arange(24)>=21),float(np.std(norm[valid])) if valid.any() else 0,
            float(np.mean(rate[valid]==0)) if valid.any() else 0,negative.mean(),
            float(np.max(rate[valid])/max(np.sum(rate[valid]),1)) if valid.any() else 0,
            logratio(average_rate(night),average_rate(~night)),harmonic(np.sin(phase)),harmonic(np.cos(phase)),
            harmonic(np.sin(2*phase)),harmonic(np.cos(2*phase)),logratio(average_rate(recent),average_rate(early)),
            float(np.mean(change[matched])) if matched.any() else 0,correlation,
            recent_activity/max(exposure,1),future_activity/max((event.end-case.observed_at)/3600,1),
            np.clip((pace-case.baseline)/max(1,case.baseline-case.current),-1,2),math.log1p(support),
            (case.cutoff-case.observed_at)/86400],float)
        return dict(sequence=tokens,summaries=features,pace=pace,profile=profile,
                    support=support,coverage=float(valid.mean()))

    def pace(self,case,event,profile,total,recent_activity,coverage):
        if recent_activity<=0 or total<=0 or coverage<1/3:return case.baseline
        target_future=activity_integral(profile,self.data,event,case.observed_at,event.end,case.cutoff)
        remaining=event.end-case.observed_at;values=[];weights=[]
        for eid in case.references:
            if eid not in self.references or eid==case.event_id or self.data.label_available(eid,self.config)>event.start:continue
            ref=self.data.known_event(self.data.events[eid],event.start)[0]
            anchor=ref.end-remaining
            if anchor-86400<ref.start:continue
            curve=self.reference_curve(eid,case.tier,event.start)
            points,valid=sample(curve,np.array([anchor-86400,anchor]))
            # A long interior gap is missing exposure even if both endpoints exist.
            segment=curve.times[(curve.times>=anchor-86400)&(curve.times<=anchor)]
            if not valid.all() or len(segment)<2 or np.max(np.diff(segment))>2*3600:continue
            delta=points[1]-points[0];final=self.data.labels[(eid,case.tier)]
            if delta<=0 or final<points[1]:continue
            past=activity_integral(profile,self.data,ref,anchor-86400,anchor,event.start)
            future=activity_integral(profile,self.data,ref,anchor,ref.end,event.start)
            if past<=0 or future<=0:continue
            estimate=(total/recent_activity)/(delta/past)*(final-points[1])*target_future/future
            weight=self.config.event_weight((event.start-ref.end)/86400,ref.start)
            weight*=2 if event.event_type==ref.event_type else 1
            weight*=math.exp(-abs(math.log((event.end-event.start)/(ref.end-ref.start)))/.5)
            if self.data.label_sources.get((eid,case.tier))=='post_close_tracker_proxy':weight*=self.config.proxy_label_weight
            values.append(estimate);weights.append(weight)
        if len(values)<self.config.min_reference_events:return case.baseline
        effective=sum(weights)**2/sum(w*w for w in weights)
        share=coverage*effective/(effective+5)
        return case.current+share*max(0,weighted_quantile(values,weights))+(1-share)*max(0,case.baseline-case.current)


def observer_arrays(cases,index,data,config,references):
    observer=DailyObserver(data,config,references);n,k=index.shape
    summaries=np.zeros((n,k,len(SUMMARIES)));sequence=np.zeros((n,k,24,len(CHANNELS)))
    pace=np.array([c.baseline for c in cases]);coverage=np.zeros(len(cases));profiles=np.zeros((len(cases),2,24))
    for g,t in zip(*np.where(index>=0)):
        i=index[g,t];result=observer.observe(cases[i])
        summaries[g,t]=result['summaries'];sequence[g,t]=result['sequence']
        pace[i]=result['pace'];coverage[i]=result['coverage'];profiles[i]=result['profile']
    return dict(summaries=summaries,sequence=sequence,pace=pace,coverage=coverage,profiles=profiles)
