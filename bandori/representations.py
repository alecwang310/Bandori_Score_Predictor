"""Causal feature ablations and a training-only compressed representation."""
import math

import numpy as np

from .analog import FEATURES
from .joint import pack


MINIMAL=('progress','remaining_days','current_fraction','rate_6h','rate_24h','history_spread')
LAGS=(.5,1,3,6,12,24,48,72)
WINDOWS=(3,12,24,72)
TYPES=('challenge','versus','mission_live','live_try','team_live_festival','medley')
EXTRA=tuple(f'lag_{h:g}h_{s}' for h in LAGS for s in ('growth','available'))+tuple(
    f'window_{h}h_{s}' for h in WINDOWS for s in ('rate_mean','rate_std','rate_median','zero_fraction','negative_fraction','slope','autocorrelation','log_samples','max_gap'))+(
    'progress_squared','log_remaining','log_duration','pace_delta','pace_support','rate_acceleration')+tuple('type_'+t for t in TYPES)
RICH=FEATURES+EXTRA


def extra_features(case,data):
    event=data.known_event(data.events[case.event_id],case.cutoff)[0]
    curve=data.curves[(case.event_id,case.tier)].prefix(case.cutoff,event.start)
    mask=curve.times<=case.observed_at
    times,scores=curve.times[mask],curve.scores[mask]
    result={}
    for h in LAGS:
        t=case.observed_at-h*3600;i=int(np.searchsorted(times,t,side='right'))-1
        valid=i>=0 and t-times[i]<=min(3,h/2)*3600
        result[f'lag_{h:g}h_growth']=(case.current-scores[i])/case.baseline if valid else 0.
        result[f'lag_{h:g}h_available']=float(valid)
    for h in WINDOWS:
        take=times>=case.observed_at-h*3600;t=times[take];s=scores[take]
        values=dict(rate_mean=0.,rate_std=0.,rate_median=0.,zero_fraction=0.,negative_fraction=0.,slope=0.,autocorrelation=0.,log_samples=math.log1p(len(t)),max_gap=h)
        if len(t)>1:
            dt=np.diff(t)/3600;ds=np.diff(s)/case.baseline
            good=dt>0;rate=ds[good]/dt[good]*24
            if len(rate):
                values.update(rate_mean=float(np.mean(rate)),rate_std=float(np.std(rate)),rate_median=float(np.median(rate)),
                              zero_fraction=float(np.mean(ds[good]==0)),negative_fraction=float(np.mean(ds[good]<0)),max_gap=float(max(dt)))
                centered=(t-t.mean())/86400
                values['slope']=float(np.dot(centered,(s-s.mean())/case.baseline)/max(np.dot(centered,centered),1e-8))
                if len(rate)>2 and np.std(rate[:-1])>1e-9 and np.std(rate[1:])>1e-9:
                    values['autocorrelation']=float(np.corrcoef(rate[:-1],rate[1:])[0,1])
        result.update({f'window_{h}h_{k}':v for k,v in values.items()})
    f=case.features
    result.update(progress_squared=f['progress']**2,log_remaining=math.log1p(max(0,case.remaining_hours)),
                  log_duration=math.log1p(f['duration_days']),pace_delta=(case.pace-case.baseline)/case.baseline,
                  pace_support=math.log1p(case.pace_references),rate_acceleration=f['rate_6h']-f['rate_24h'])
    result.update({'type_'+t:float(event.event_type==t) for t in TYPES})
    return result


def matrices(cases,data,tiers):
    _,indices,bins=pack(cases,tiers)
    results={name:np.zeros((len(indices),len(tiers)*(len(schema)+1))) for name,schema in (
        ('minimal',MINIMAL),('standard',FEATURES),('rich',RICH))}
    enriched=[dict(c.features,**extra_features(c,data)) for c in cases]
    for g,t in zip(*np.where(indices>=0)):
        features=enriched[indices[g,t]]
        for name,schema in (('minimal',MINIMAL),('standard',FEATURES),('rich',RICH)):
            width=len(schema)+1;results[name][g,t*width:(t+1)*width]=[features[k] for k in schema]+[1]
    return results,indices,bins


class Compression:
    def fit(self,x,weights,tiers,components=16,seed=1729):
        w=weights.sum(axis=1);w=w/w.sum()
        self.mean=(x*w[:,None]).sum(axis=0)
        self.scale=np.maximum(.05,np.sqrt(((x-self.mean)**2*w[:,None]).sum(axis=0)))
        z=np.clip((x-self.mean)/self.scale,-6,6);a=z*np.sqrt(w[:,None])
        rank=min(components,max(1,np.count_nonzero(w)-1),x.shape[1]);width=min(rank+8,*a.shape)
        q,_=np.linalg.qr(a@np.random.default_rng(seed).normal(size=(a.shape[1],width)),mode='reduced')
        q,_=np.linalg.qr(a@(a.T@q),mode='reduced')
        _,s,v=np.linalg.svd(q.T@a,full_matrices=False)
        self.projection=v[:rank].T;self.pc_scale=np.maximum(s[:rank],.5)
        # Keep a small, physically meaningful path outside the compression.
        self.explicit=np.array([t*(len(RICH)+1)+i for t in range(len(tiers))
            for i in (RICH.index('progress'),RICH.index('remaining_days'),RICH.index('current_fraction'),len(RICH))],int)
        return self

    def transform(self,x):
        z=np.clip((x-self.mean)/self.scale,-6,6)
        return np.column_stack((np.clip(z@self.projection/self.pc_scale,-6,6),z[:,self.explicit]))

    def save(self,path):
        np.savez_compressed(path,**{k:getattr(self,k) for k in ('mean','scale','projection','pc_scale','explicit')})

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path,allow_pickle=False) as a:
            for k in a.files:setattr(model,k,a[k].copy())
        return model
