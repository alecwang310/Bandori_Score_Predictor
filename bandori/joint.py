"""A fresh, deliberately small joint NN with an explicit signed-bias penalty."""
import numpy as np

from .analog import FEATURES, horizon


def pack(cases,tiers):
    groups={}
    for case in cases:groups.setdefault((case.event_id,case.cutoff),len(groups))
    width=len(FEATURES)+1
    x=np.zeros((len(groups),len(tiers)*width))
    index=np.full((len(groups),len(tiers)),-1,int)
    bins=np.zeros(len(groups),int)
    positions={tier:i for i,tier in enumerate(tiers)}
    for i,c in enumerate(cases):
        g,t=groups[(c.event_id,c.cutoff)],positions[c.tier]
        x[g,t*width:(t+1)*width]=[c.features[name] for name in FEATURES]+[1]
        index[g,t]=i;bins[g]=horizon(c.remaining_hours)
    return x,index,bins


def objective(raw,target,weights,bins,bias_penalty):
    """Relative squared error plus squared mean signed error per tier/horizon."""
    delta=.5*np.tanh(raw)
    future=target[...,1]*np.exp(delta)
    truth=np.maximum(target[...,2],1)
    error=(target[...,0]+future-truth)/truth
    w=weights/weights.sum()
    loss=float(np.sum(w*error**2))
    derivative=error.copy()
    for h in range(5):
        mask=bins==h
        mass=w[mask].sum(axis=0)
        mean=np.divide((w[mask]*error[mask]).sum(axis=0),mass,out=np.zeros_like(mass),where=mass>0)
        loss+=bias_penalty*float(np.sum(mass*mean**2))
        derivative[mask]+=bias_penalty*mean
    gradient=2*w*derivative*future/truth*.5*(1-np.tanh(raw)**2)
    return loss,gradient


class JointNet:
    def __init__(self,hidden=4,seed=1729):
        self.hidden,self.seed=hidden,seed

    def gradients(self,z,target,weights,bins,penalty,decay):
        hidden=np.tanh(z@self.a+self.b)
        raw=hidden@self.v+self.d
        loss,grad=objective(raw,target,weights,bins,penalty)
        dh=(grad@self.v.T)*(1-hidden**2)
        loss+=.5*decay*(np.sum(self.a**2)+np.sum(self.v**2))
        return float(loss),[z.T@dh+decay*self.a,dh.sum(axis=0),hidden.T@grad+decay*self.v,grad.sum(axis=0)]

    def fit(self,x,target,weights,bins,config):
        if not weights.sum()>0 or not all(np.isfinite(a).all() for a in (x,target,weights)):
            raise ValueError('Invalid supervised training arrays')
        row_weight=weights.sum(axis=1)/weights.sum()
        self.mean=(x*row_weight[:,None]).sum(axis=0)
        self.scale=np.maximum(.05,np.sqrt(((x-self.mean)**2*row_weight[:,None]).sum(axis=0)))
        z=np.clip((x-self.mean)/self.scale,-6,6)
        rng=np.random.default_rng(self.seed)
        self.a=rng.normal(0,1/np.sqrt(x.shape[1]),(x.shape[1],self.hidden))
        self.b=np.zeros(self.hidden);self.v=np.zeros((self.hidden,target.shape[1]));self.d=np.zeros(target.shape[1])
        self.trained=weights.sum(axis=0)>0
        parameters=[self.a,self.b,self.v,self.d]
        moment=[np.zeros_like(p) for p in parameters];variance=[np.zeros_like(p) for p in parameters]
        for step in range(1,config.epochs+1):
            _,gradients=self.gradients(z,target,weights,bins,config.bias_penalty,config.weight_decay)
            for i,(parameter,gradient) in enumerate(zip(parameters,gradients)):
                moment[i]=.9*moment[i]+.1*gradient;variance[i]=.999*variance[i]+.001*gradient**2
                parameter-=config.learning_rate*(moment[i]/(1-.9**step))/(np.sqrt(variance[i]/(1-.999**step))+1e-8)
        return self

    def predict(self,x):
        z=np.clip((x-self.mean)/self.scale,-6,6)
        result=.5*np.tanh(np.tanh(z@self.a+self.b)@self.v+self.d)
        result[:,~self.trained]=0
        return result

    def save(self,path):
        np.savez_compressed(path,**{name:getattr(self,name) for name in ('mean','scale','a','b','v','d','trained')})

    @classmethod
    def load(cls,path):
        net=cls()
        with np.load(path,allow_pickle=False) as arrays:
            for name in arrays.files:setattr(net,name,arrays[name].copy())
        net.hidden=len(net.b)
        return net
