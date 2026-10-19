"""Causal peak timing, clock-window slopes and personalized daily patterns."""
import math

import numpy as np

from .daily_observer import activity_integral, clock_index, sample


WINDOWS=tuple(range(0,24,3))
GLOBAL=('peak_sin','peak_cos','peak_concentration','peak_days','peak_strength',
        'peak_prior_delta_sin','peak_prior_delta_cos','peak_width','profile_entropy','profile_correlation',
        'ordinary_coverage','rush_share','opening_share','day_repeatability','day_repeatability_available',
        'prior_peak_sin','prior_peak_cos','prior_peak_strength','weekend_peak_sin','weekend_peak_cos',
        'personal_recent_cycle','personal_future_cycle','clock_pace_delta','clock_adjustment',
        'rate_slope_6h','rate_acceleration_6h','rate_slope_6h_available')
FEATURES=GLOBAL+tuple(f'clock_{h:02d}_{v}' for h in WINDOWS for v in
    ('mean','slope','coverage','slope_available','day_change','day_change_available','future_share'))
COMPACT=('peak_sin','peak_cos','peak_concentration','peak_days','peak_prior_delta_sin','peak_prior_delta_cos',
         'profile_entropy','profile_correlation','ordinary_coverage','clock_pace_delta','rate_slope_6h','rate_acceleration_6h')
COMPACT_INDICES=np.array([FEATURES.index(n) for n in COMPACT])


def slope(x,y):
    if len(x)<2:return 0.
    x=np.asarray(x,float);centered=x-x.mean()
    return float(np.dot(centered,np.asarray(y)-np.mean(y))/max(np.dot(centered,centered),1e-12))


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>=6 and np.std(a)>1e-9 and np.std(b)>1e-9 else 0.


def circular_peak(hours,weights):
    """23:00 and 01:00 average around midnight, never around noon."""
    if not len(hours) or np.sum(weights)<=0:return 0.,1.,0.
    angle=np.asarray(hours)*2*np.pi/24;w=np.asarray(weights,float)
    sine=float(np.average(np.sin(angle),weights=w));cosine=float(np.average(np.cos(angle),weights=w))
    concentration=math.hypot(sine,cosine)
    return (sine/max(concentration,1e-12),cosine/max(concentration,1e-12),concentration)


def observe(case,data,prior,base_pace):
    event=data.known_event(data.events[case.event_id],case.cutoff)[0]
    curve=data.curves[(case.event_id,case.tier)].prefix(case.cutoff,event.start)
    last=math.floor(case.observed_at/3600)*3600;edges=last+np.arange(-72,1)*3600
    values,available=sample(curve,edges);middle=(edges[:-1]+edges[1:])/2
    hour,weekday=clock_index(middle);day=np.floor(middle/86400+8/24).astype(int)
    from .analog import active_hours
    exposure=np.array([active_hours(data,event,a,b,case.cutoff) for a,b in zip(edges[:-1],edges[1:])])
    ds=np.diff(values);valid=available[:-1]&available[1:]&(ds>=0)&(exposure>.1)
    valid&=(edges[:-1]>=event.start)&(edges[1:]<=case.observed_at)
    rate=np.divide(np.maximum(0,ds),exposure,out=np.zeros(72),where=valid)
    ordinary=valid&(middle>=event.start+12*3600)&(middle<event.end-24*3600)
    profile_sum=np.bincount(hour[ordinary],weights=rate[ordinary]*exposure[ordinary],minlength=24)
    profile_mass=np.bincount(hour[ordinary],weights=exposure[ordinary],minlength=24)
    measured=np.divide(profile_sum,profile_mass,out=np.zeros(24),where=profile_mass>0)
    current_day=int(clock_index(np.array([case.observed_at]))[1][0]>=5)
    reference=prior[current_day];prior_peak=int(np.argmax(reference));prior_angle=prior_peak*2*np.pi/24
    prior_s,prior_c=math.sin(prior_angle),math.cos(prior_angle)
    peaks=[];peak_weights=[];peak_strength=[];profiles=[]
    for d in np.unique(day):
        keep=ordinary&(day==d)
        if keep.sum()<16 or np.sum(rate[keep])<=0:continue
        peak=int(np.flatnonzero(keep)[np.argmax(rate[keep])]);peaks.append(float(hour[peak]))
        peak_weights.append(float(exposure[keep].sum())*math.exp(-(case.observed_at-middle[keep].mean())/172800))
        peak_strength.append(float(rate[peak]/max(np.average(rate[keep],weights=exposure[keep]),1e-9)))
        a=np.full(24,np.nan);a[hour[keep]]=rate[keep];profiles.append(a)
    ps,pc,concentration=circular_peak(peaks,peak_weights) if peaks else (prior_s,prior_c,0.)
    delta_s=ps*prior_c-pc*prior_s;delta_c=pc*prior_c+ps*prior_s
    observed_hours=profile_mass>0
    normalized=measured/max(np.average(measured[observed_hours],weights=profile_mass[observed_hours]),1e-9) if observed_hours.any() else np.ones(24)
    # Per-hour evidence is at most three days; shrink it toward the historical
    # weekday/weekend profile with two full day-equivalents of prior strength.
    personal=np.vstack([(2*prior[d]+profile_mass*normalized)/(2+profile_mass) for d in range(2)])
    personal/=max(float(np.average(personal,weights=np.broadcast_to(np.array([5,2])[:,None],personal.shape))),1e-9)
    shares=measured[observed_hours]/max(measured[observed_hours].sum(),1e-9)
    positive=shares>0
    entropy=float(-np.sum(shares[positive]*np.log(shares[positive]))/max(1,math.log(max(2,len(shares)))))
    width=float(np.mean(measured[observed_hours]>=.75*measured[observed_hours].max())) if observed_hours.any() else 0.
    repeats=[]
    for a,b in zip(profiles[:-1],profiles[1:]):
        keep=np.isfinite(a)&np.isfinite(b)
        if keep.sum()>=6:repeats.append(correlation(a[keep],b[keep]))
    corr=correlation(normalized[observed_hours],reference[observed_hours])
    recent_start=max(event.start,case.observed_at-86400)
    old_recent=activity_integral(prior,data,event,recent_start,case.observed_at,case.cutoff)
    new_recent=activity_integral(personal,data,event,recent_start,case.observed_at,case.cutoff)
    old_future=activity_integral(prior,data,event,case.observed_at,event.end,case.cutoff)
    new_future=activity_integral(personal,data,event,case.observed_at,event.end,case.cutoff)
    factor=(new_future/max(old_future,1e-9))/(new_recent/max(old_recent,1e-9)) if old_recent>0 and old_future>0 and new_recent>0 else 1.
    factor=float(np.clip(factor,.75,4/3));pace=case.current+max(0,base_pace-case.current)*factor
    recent=valid&(middle>=case.observed_at-6*3600);scaled=rate*24/max(case.baseline,1)
    trend=slope((middle[recent]-last)/3600,scaled[recent]) if recent.sum()>=3 else 0.
    previous=valid&(middle>=case.observed_at-12*3600)&(middle<case.observed_at-6*3600)
    acceleration=trend-slope((middle[previous]-last)/3600,scaled[previous]) if recent.sum()>=3 and previous.sum()>=3 else 0.
    weekend_angle=int(np.argmax(prior[1]))*2*np.pi/24
    globals_=[ps,pc,concentration,math.log1p(len(peaks)),float(np.average(peak_strength,weights=peak_weights)) if peaks else 1.,
        delta_s,delta_c,width,entropy,corr,float(ordinary.sum()/72),
        float(np.count_nonzero(valid&(middle>=event.end-86400))/max(1,valid.sum())),
        float(np.count_nonzero(valid&(middle<event.start+12*3600))/max(1,valid.sum())),
        float(np.mean(repeats)) if repeats else 0.,float(bool(repeats)),prior_s,prior_c,float(reference.max()/max(reference.mean(),1e-9)),
        math.sin(weekend_angle),math.cos(weekend_angle),new_recent/max(old_recent,1),new_future/max(old_future,1),
        float(np.clip((pace-case.baseline)/max(case.baseline-case.current,1),-1,2)),factor-1,trend,acceleration,float(recent.sum()>=3)]
    windows=[]
    for start in WINDOWS:
        group=(hour>=start)&(hour<start+3);keep=valid&group
        current_day_number=int(day[keep][-1]) if keep.any() else 0
        today=keep&(day==current_day_number);yesterday=keep&(day==current_day_number-1)
        common=set(hour[today])&set(hour[yesterday]);today_rate={int(h):r for h,r in zip(hour[today],rate[today])}
        yesterday_rate={int(h):r for h,r in zip(hour[yesterday],rate[yesterday])}
        ratio=float(np.clip(math.log((np.mean([today_rate[h] for h in common])+.01)/(np.mean([yesterday_rate[h] for h in common])+.01)),-3,3)) if common else 0.
        selected=np.zeros((2,24));selected[:,start:start+3]=personal[:,start:start+3]
        future_share=activity_integral(selected,data,event,case.observed_at,event.end,case.cutoff)/max(new_future,1)
        windows += [float(np.mean(scaled[keep])) if keep.any() else 0.,slope(hour[today]-start,scaled[today]),
                    float(keep.sum()/9),float(today.sum()>=2),ratio,float(bool(common)),float(future_share)]
    features=np.asarray(globals_+windows,float)
    if features.shape!=(len(FEATURES),) or not np.isfinite(features).all():raise ValueError('Invalid clock features')
    return features,float(max(case.current,pace))


def arrays(ctx,data):
    n,k=ctx['index'].shape;features=np.zeros((n,k,len(FEATURES)));pace=np.zeros(len(ctx['cases']))
    for g,t in zip(*np.where(ctx['index']>=0)):
        i=ctx['index'][g,t]
        features[g,t],pace[i]=observe(ctx['cases'][i],data,ctx['daily']['profiles'][i],ctx['daily']['pace'][i])
    return dict(features=features,pace=pace)


class ClockCompression:
    def fit(self,x,weights,seed=1729,components=16):
        w=weights.sum(axis=1);w/=w.sum();self.mean=(x*w[:,None]).sum(axis=0)
        self.scale=np.maximum(.05,np.sqrt(((x-self.mean)**2*w[:,None]).sum(axis=0)))
        a=np.clip((x-self.mean)/self.scale,-6,6)*np.sqrt(w[:,None]);rank=min(components,max(1,np.count_nonzero(w)-1),x.shape[1])
        q,_=np.linalg.qr(a@np.random.default_rng(seed).normal(size=(a.shape[1],min(rank+8,*a.shape))),mode='reduced')
        q,_=np.linalg.qr(a@(a.T@q),mode='reduced');_,s,v=np.linalg.svd(q.T@a,full_matrices=False)
        self.projection=v[:rank].T;self.pc_scale=np.maximum(.5,s[:rank]);return self

    def transform(self,x):return np.clip(np.clip((x-self.mean)/self.scale,-6,6)@self.projection/self.pc_scale,-6,6)

    def save(self,path):np.savez_compressed(path,**{k:getattr(self,k) for k in ('mean','scale','projection','pc_scale')})

    @classmethod
    def load(cls,path):
        result=cls()
        with np.load(path,allow_pickle=False) as a:
            for k in a.files:setattr(result,k,a[k].copy())
        return result
