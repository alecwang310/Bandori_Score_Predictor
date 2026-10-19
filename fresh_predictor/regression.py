"""One joint partially pooled relative-error model; post-rule supervision only."""
import numpy as np

from .anchors import CORE,FEATURES,DEVIATION_INDICES


def training_weights(cases,data,policy,known_at):
    result=np.zeros(len(cases));groups={}
    for i,c in enumerate(cases):
        event=data.events[c.event_id]
        if event.start<policy.rule_at:raise ValueError('Pre-rule supervised event')
        if data.label_available(c.event_id,policy.label_config())>known_at:raise ValueError('Future training label')
        truth=data.labels.get((c.event_id,c.tier))
        if truth is None or truth<c.current:continue
        groups.setdefault((c.event_id,min(4,int(c.hours_remaining//24))),[]).append(i)
    horizon_mass={}
    for eid,h in groups:horizon_mass[h]=horizon_mass.get(h,0)+policy.event_weight(data.events[eid],known_at)
    for (eid,h),ii in groups.items():
        for i in ii:
            quality=policy.proxy_weight if data.label_sources.get((eid,cases[i].tier))=='post_close_tracker_proxy' else 1.
            result[i]=policy.event_weight(data.events[eid],known_at)/horizon_mass[h]/len(ii)/len(horizon_mass)*quality
    if result.sum()<=0:raise ValueError('No post-rule training targets')
    return result/result.sum()


def reconcile(cases,values):
    values=np.maximum(values,[c.current for c in cases]);groups={}
    for i,c in enumerate(cases):groups.setdefault((c.event_id,c.cutoff),[]).append(i)
    # Pool adjacent violations to the closest ordered forecasts, rather than
    # raising every better rank to an overpredicted worse-rank line.
    for ii in groups.values():
        ii=sorted(ii,key=lambda i:cases[i].tier);blocks=[]
        for i in ii:
            blocks.append(([i],float(values[i]),1.))
            while len(blocks)>1 and blocks[-2][1]<blocks[-1][1]:
                a,va,wa=blocks[-2];b,vb,wb=blocks[-1];blocks[-2:]=[(a+b,(va*wa+vb*wb)/(wa+wb),wa+wb)]
        for indices,v,_ in blocks:
            for i in indices:values[i]=max(v,cases[i].current)
        # Known cumulative cutoffs are ordered; this enforces floors even in
        # the presence of an inconsistent stale snapshot.
        floor=0.
        for i in reversed(ii):values[i]=max(values[i],floor);floor=values[i]
    return values


class JointCorrection:
    def design(self,cases):
        x=np.asarray([c.features for c in cases]);z=np.clip((x-self.mean)/self.scale,-6,6);z[:,0]=1.
        out=np.zeros((len(cases),len(FEATURES)+len(self.tiers)*len(DEVIATION_INDICES)))
        out[:,:len(FEATURES)]=z
        positions={int(t):i for i,t in enumerate(self.tiers)}
        for i,c in enumerate(cases):
            j=positions[c.tier];start=len(FEATURES)+j*len(DEVIATION_INDICES)
            out[i,start:start+len(DEVIATION_INDICES)]=z[i,DEVIATION_INDICES]
        return out

    def fit(self,cases,data,policy,known_at,strength):
        self.tiers=np.asarray(policy.tiers);w=training_weights(cases,data,policy,known_at)
        x=np.asarray([c.features for c in cases]);self.mean=(x*w[:,None]).sum(0)
        self.scale=np.maximum(.1,np.sqrt(((x-self.mean)**2*w[:,None]).sum(0)))
        self.mean[0]=0.;self.scale[0]=1.
        self.trained=np.array([any(c.tier==t and wi>0 for c,wi in zip(cases,w)) for t in self.tiers])
        design=self.design(cases);current=np.array([c.current for c in cases]);future=np.maximum(0,np.array([c.anchor for c in cases])-current)
        truth=np.array([data.labels.get((c.event_id,c.tier),1.) for c in cases]);desired=(truth-current-future)/truth
        a=design*(future/truth)[:,None];penalty=np.full(a.shape[1],strength)
        penalty[0]=strength*.1;penalty[len(CORE):len(FEATURES)]*=4;penalty[len(FEATURES):]*=4
        matrix=a.T@(w[:,None]*a)+np.diag(penalty);rhs=a.T@(w*desired)
        # Explicit relative-bias penalty, shared across each tier/horizon cell.
        groups={}
        for i,c in enumerate(cases):groups.setdefault((c.tier,min(4,int(c.hours_remaining//24))),[]).append(i)
        for ii in groups.values():
            mass=w[ii].sum()
            if mass<=0:continue
            mx=(a[ii]*w[ii,None]).sum(0)/mass;my=float(np.dot(w[ii],desired[ii])/mass)
            matrix+=3*mass*np.outer(mx,mx);rhs+=3*mass*mx*my
        self.coefficients=np.linalg.solve(matrix,rhs);self.strength=float(strength);return self

    def predict(self,cases):
        if not cases:return np.empty(0)
        delta=np.clip(self.design(cases)@self.coefficients,-.9,3.)
        trained={int(t):bool(v) for t,v in zip(self.tiers,self.trained)}
        delta=np.where([trained[c.tier] for c in cases],delta,0.)
        current=np.array([c.current for c in cases]);future=np.maximum(0,np.array([c.anchor for c in cases])-current)
        return reconcile(cases,current+future*(1+delta))

    def save(self,path):
        np.savez_compressed(path,**{k:getattr(self,k) for k in ('tiers','mean','scale','trained','coefficients','strength')})

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path,allow_pickle=False) as a:
            for k in a.files:setattr(model,k,a[k].copy())
        return model
