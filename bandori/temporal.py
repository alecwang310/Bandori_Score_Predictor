"""Small shared causal convolutions followed by one joint all-tier head.

The temporal encoder shares its parameters across ranks. Two width-three
convolutions at dilations one and three see nine-hour local shapes; pooling
over all 24 hours and over its first/last six hours retains daily context.
"""
import numpy as np

from .joint import objective


def convolution(x,kernel,bias,dilation):
    y=np.broadcast_to(bias,(*x.shape[:2],len(bias))).copy()
    for lag in range(len(kernel)):
        shift=lag*dilation
        if shift==0:y+=x@kernel[lag]
        else:y[:,shift:]+=x[:,:-shift]@kernel[lag]
    return y


def convolution_backward(x,kernel,gradient,dilation):
    dx=np.zeros_like(x);dk=np.zeros_like(kernel)
    for lag in range(len(kernel)):
        shift=lag*dilation
        if shift==0:
            dk[lag]=np.einsum('nti,ntj->ij',x,gradient)
            dx+=gradient@kernel[lag].T
        else:
            dk[lag]=np.einsum('nti,ntj->ij',x[:,:-shift],gradient[:,shift:])
            dx[:,:-shift]+=gradient[:,shift:]@kernel[lag].T
    return dx,dk,gradient.sum(axis=(0,1))


class JointTemporalNet:
    PARAMETERS=('k1','b1','k2','b2','a','b','v','d')

    def __init__(self,channels=4,hidden=8,seed=1729):
        self.channels,self.hidden,self.seed=channels,hidden,seed

    def prepare(self,x,sequence):
        z=np.clip((x-self.mean)/self.scale,-6,6)
        mask=sequence[...,3]>0
        active=np.flatnonzero(mask.reshape(-1,24).any(axis=1))
        flat=sequence.reshape(-1,24,sequence.shape[-1])[active]
        present=mask.reshape(-1,24)[active]
        temporal=np.clip((flat-self.seq_mean)/self.seq_scale,-6,6)*present[...,None]
        pooling=np.stack([present,present*(np.arange(24)<6),present*(np.arange(24)>=18)],axis=1).astype(float)
        pooling/=np.maximum(1,pooling.sum(axis=2,keepdims=True))
        return z,temporal,present,active,pooling,sequence.shape[1]

    def forward(self,prepared):
        z,s,mask,active,pool,tiers=prepared
        h1=np.tanh(convolution(s,self.k1,self.b1,1))*mask[...,None]
        h2=np.tanh(convolution(h1,self.k2,self.b2,3))*mask[...,None]
        encoded=np.einsum('npt,nth->nph',pool,h2).reshape(len(active),3*self.channels)
        joint=np.zeros((len(z)*tiers,3*self.channels));joint[active]=encoded
        joined=np.column_stack((z,joint.reshape(len(z),-1)))
        hidden=np.tanh(joined@self.a+self.b);raw=hidden@self.v+self.d
        return raw,(h1,h2,joined,hidden)

    def gradients(self,prepared,target,weights,bins,penalty,decay):
        raw,(h1,h2,joined,hidden)=self.forward(prepared)
        loss,gradient=objective(raw,target,weights,bins,penalty)
        dh=(gradient@self.v.T)*(1-hidden**2)
        joined_gradient=dh@self.a.T
        z,s,mask,active,pool,tiers=prepared
        dg=joined_gradient[:,z.shape[1]:].reshape(len(z)*tiers,3*self.channels)[active].reshape(len(active),3,self.channels)
        dh2=np.einsum('nph,npt->nth',dg,pool)*mask[...,None]*(1-h2**2)
        dh1,dk2,db2=convolution_backward(h1,self.k2,dh2,3)
        ds,dk1,db1=convolution_backward(s,self.k1,dh1*mask[...,None]*(1-h1**2),1)
        gradients=[dk1,db1,dk2,db2,joined.T@dh,dh.sum(axis=0),hidden.T@gradient,gradient.sum(axis=0)]
        for i,name in enumerate(self.PARAMETERS):
            if name in ('k1','k2','a','v'):
                parameter=getattr(self,name);loss+=.5*decay*float(np.sum(parameter**2));gradients[i]+=decay*parameter
        return float(loss),gradients

    def fit(self,x,sequence,target,weights,bins,config):
        if weights.sum()<=0 or not all(np.isfinite(a).all() for a in (x,sequence,target,weights)):
            raise ValueError('Invalid temporal training arrays')
        row_weights=weights.sum(axis=1)/weights.sum()
        self.mean=(x*row_weights[:,None]).sum(axis=0)
        self.scale=np.maximum(.05,np.sqrt(((x-self.mean)**2*row_weights[:,None]).sum(axis=0)))
        temporal_weights=weights[:,:,None]*(sequence[...,3]>0)
        temporal_weights/=max(temporal_weights.sum(),1e-20)
        self.seq_mean=np.sum(sequence*temporal_weights[...,None],axis=(0,1,2))
        self.seq_scale=np.maximum(.05,np.sqrt(np.sum((sequence-self.seq_mean)**2*temporal_weights[...,None],axis=(0,1,2))))
        rng=np.random.default_rng(self.seed);channels=self.channels
        self.k1=rng.normal(0,1/np.sqrt(3*sequence.shape[-1]),(3,sequence.shape[-1],channels));self.b1=np.zeros(channels)
        self.k2=rng.normal(0,1/np.sqrt(3*channels),(3,channels,channels));self.b2=np.zeros(channels)
        width=x.shape[1]+sequence.shape[1]*3*channels
        self.a=rng.normal(0,1/np.sqrt(width),(width,self.hidden));self.b=np.zeros(self.hidden)
        self.v=np.zeros((self.hidden,target.shape[1]));self.d=np.zeros(target.shape[1])
        self.trained=weights.sum(axis=0)>0
        parameters=[getattr(self,k) for k in self.PARAMETERS]
        moments=[np.zeros_like(p) for p in parameters];variances=[np.zeros_like(p) for p in parameters]
        prepared=self.prepare(x,sequence)
        for step in range(1,config.epochs+1):
            _,gradients=self.gradients(prepared,target,weights,bins,config.bias_penalty,config.weight_decay)
            norm=np.sqrt(sum(float(np.sum(g*g)) for g in gradients));clip=min(1,5/max(norm,1e-12))
            for i,(p,g) in enumerate(zip(parameters,gradients)):
                g=g*clip;moments[i]=.9*moments[i]+.1*g;variances[i]=.999*variances[i]+.001*g*g
                p-=config.learning_rate*(moments[i]/(1-.9**step))/(np.sqrt(variances[i]/(1-.999**step))+1e-8)
        return self

    def predict(self,x,sequence):
        raw,_=self.forward(self.prepare(x,sequence));result=.5*np.tanh(raw)
        result[:,~self.trained]=0
        return result

    def save(self,path):
        np.savez_compressed(path,**{k:getattr(self,k) for k in (*self.PARAMETERS,'mean','scale','seq_mean','seq_scale','trained')})

    @classmethod
    def load(cls,path):
        model=cls()
        with np.load(path,allow_pickle=False) as arrays:
            for k in arrays.files:setattr(model,k,arrays[k].copy())
        model.channels=len(model.b1);model.hidden=len(model.b)
        return model
