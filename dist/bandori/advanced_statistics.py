"""Trajectory-matched analogs and partially pooled remaining-score regression."""
import math

import numpy as np

from .analog import AnalogPredictor,weighted_quantile


class CachedAnalog(AnalogPredictor):
    """Share immutable completed curves without dropping publication filters."""
    def __init__(self,*args):
        super().__init__(*args);self.availability={}

    def reference(self,eid,tier,known_at):
        key=(eid,tier)
        if key not in self.availability:
            curve=self.data.curves[key];event=self.data.events[eid]
            relevant=(curve.times>=event.start)&(curve.times<event.end)
            self.availability[key]=float(np.max(curve.available[relevant])) if relevant.any() else 0.
        return super().reference(eid,tier,float('inf') if known_at>=self.availability[key] else known_at)


def trajectory_forecasts(cases,data,config,reference_ids,bandwidths=(.1,.3)):
    predictor=CachedAnalog(data,config,reference_ids)
    result={b:[] for b in bandwidths}
    for c in cases:
        event=data.known_event(data.events[c.event_id],c.cutoff)[0];curve=data.curves[(c.event_id,c.tier)].prefix(c.cutoff,event.start)
        profile={}
        for lag in (6,12,24,48):
            point=curve.last(c.observed_at-lag*3600,event.start)
            if point and c.observed_at-lag*3600-point[0]<=3*3600 and point[1]>0:
                profile[lag]=math.log(point[1]/c.current)
        analogs=[]
        for eid in c.references:
            ref=data.events[eid];final=data.labels[(eid,c.tier)]
            duration=event.end-event.start;ref_duration=ref.end-ref.start
            remaining=(event.end-c.observed_at)
            for t,stretch,axis_weight in ((ref.end-remaining,1.,.6),
                (ref.start+c.features['progress']*ref_duration,ref_duration/duration,.4)):
                level=predictor.sample(eid,c.tier,t,event.start)
                if level is None or not 0<level<=final:continue
                distances=[]
                for lag,target_ratio in profile.items():
                    previous=predictor.sample(eid,c.tier,t-lag*3600*stretch,event.start)
                    if previous is not None and previous>0:distances.append((math.log(previous/level)-target_ratio)**2)
                if not distances:continue
                age=(event.start-ref.end)/86400
                weight=axis_weight*config.event_weight(age,ref.start)
                weight*=2 if event.event_type==ref.event_type else 1
                weight*=math.exp(-abs(math.log(duration/ref_duration))/.5)
                if data.label_sources.get((eid,c.tier))=='post_close_tracker_proxy':weight*=config.proxy_label_weight
                analogs.append((c.current*(final-level)/level,weight,float(np.mean(distances)),eid))
        for bandwidth in bandwidths:
            if len({a[3] for a in analogs})<config.min_reference_events:
                result[bandwidth].append(c.baseline);continue
            future=np.array([a[0] for a in analogs]);w=np.array([a[1]*math.exp(-a[2]/(2*bandwidth**2)) for a in analogs])
            if w.sum()<1e-100:result[bandwidth].append(c.baseline);continue
            # Count each historical event once even when both alignments are used.
            event_weights={}
            for a,weight in zip(analogs,w):event_weights[a[3]]=event_weights.get(a[3],0)+weight
            effective=sum(event_weights.values())**2/sum(v*v for v in event_weights.values())
            low,high=weighted_quantile(future,w,.1),weighted_quantile(future,w,.9)
            estimate=float(np.average(np.clip(future,low,high),weights=w))
            share=effective/(effective+5)
            result[bandwidth].append(c.current+share*estimate+(1-share)*max(0,c.baseline-c.current))
    return {b:np.asarray(v) for b,v in result.items()}


def penalized_fit(design,target,weights,groups,penalty,bias_penalty):
    """Weighted GLS with a squared signed-bias penalty, solved in closed form."""
    a=design.T@(weights[:,None]*design)+np.diag(penalty)
    b=design.T@(weights*target)
    for group in np.unique(groups):
        mask=groups==group;mass=weights[mask].sum()
        if mass<=0:continue
        mean_x=(design[mask]*weights[mask,None]).sum(axis=0)/mass
        mean_y=float(np.dot(weights[mask],target[mask])/mass)
        a+=bias_penalty*mass*np.outer(mean_x,mean_x);b+=bias_penalty*mass*mean_x*mean_y
    return np.linalg.solve(a,b)


class JointRidge:
    def fit(self,x,target,weights,bins,strength=.01,bias_penalty=3):
        row_weight=weights.sum(axis=1)/weights.sum()
        self.mean=(x*row_weight[:,None]).sum(axis=0)
        self.scale=np.maximum(.05,np.sqrt(((x-self.mean)**2*row_weight[:,None]).sum(axis=0)))
        z=np.column_stack((np.clip((x-self.mean)/self.scale,-6,6),np.ones(len(x))))
        self.coefficients=np.zeros((z.shape[1],target.shape[1]));self.trained=weights.sum(axis=0)>0
        for t in np.flatnonzero(self.trained):
            y=np.maximum(target[:,t,2],1);fraction=target[:,t,1]/y
            desired=(y-target[:,t,0]-target[:,t,1])/y
            penalty=np.full(z.shape[1],strength);penalty[-1]=strength*.1
            self.coefficients[:,t]=penalized_fit(z*fraction[:,None],desired,weights[:,t]/weights.sum(),bins,penalty,bias_penalty)
        return self

    def predict(self,x):
        z=np.column_stack((np.clip((x-self.mean)/self.scale,-6,6),np.ones(len(x))))
        result=np.clip(z@self.coefficients,-.5,1.)
        result[:,~self.trained]=0
        return result

    def save(self,path):
        np.savez_compressed(path,mean=self.mean,scale=self.scale,coefficients=self.coefficients,trained=self.trained)

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path,allow_pickle=False) as arrays:
            for k in arrays.files:setattr(model,k,arrays[k].copy())
        return model


def surface_features(cases,shape):
    rows=[]
    for c,k in zip(cases,shape):
        future=max(1,c.baseline-c.current);p=c.features['progress']
        pace=np.clip((c.pace-c.baseline)/future,-1,2);kernel=np.clip((k-c.baseline)/future,-1,2)
        rows.append([1,p,p*p,math.log1p(c.remaining_hours)/6,math.log1p(c.features['duration_days']),
                     c.features['history_spread'],pace,kernel,pace*p,kernel*p])
    return np.asarray(rows,float).reshape(len(cases),10)


class HierarchicalSurface:
    def design(self,x,tiers):
        width=x.shape[1];result=np.zeros((len(x),width*(len(self.tiers)+1)))
        result[:,:width]=x
        for t in range(len(self.tiers)):
            mask=tiers==self.tiers[t];result[mask,(t+1)*width:(t+2)*width]=x[mask]
        return result

    def fit(self,x,cases,data,weights,tiers,global_penalty=.01,tier_penalty=.1,bias_penalty=3):
        self.tiers=np.asarray(tiers);case_tiers=np.array([c.tier for c in cases]);z=self.design(x,case_tiers)
        truth=np.array([data.labels.get((c.event_id,c.tier),1) for c in cases])
        current=np.array([c.current for c in cases]);future=np.array([max(0,c.baseline-c.current) for c in cases])
        fraction=future/truth;desired=(truth-current-future)/truth
        penalty=np.r_[np.full(x.shape[1],global_penalty),np.full(x.shape[1]*len(tiers),tier_penalty)]
        groups=np.array([int(np.flatnonzero(self.tiers==c.tier)[0])*5+min(4,int(c.remaining_hours//24)) for c in cases])
        self.coefficients=penalized_fit(z*fraction[:,None],desired,weights/weights.sum(),groups,penalty,bias_penalty)
        return self

    def predict(self,x,tiers):
        return np.clip(self.design(x,np.asarray(tiers))@self.coefficients,-.5,1.)

    def save(self,path):
        np.savez_compressed(path,tiers=self.tiers,coefficients=self.coefficients)

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path,allow_pickle=False) as a:model.tiers=a['tiers'].copy();model.coefficients=a['coefficients'].copy()
        return model
