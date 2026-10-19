"""Small full-history causal temporal transformer, raw inputs + small side path."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self,width,heads,feedforward):
        super().__init__();self.heads=heads;self.width=width
        self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.project=nn.Linear(width,width)
        self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,feedforward),nn.GELU(),nn.Linear(feedforward,width))

    def forward(self,x):
        b,t,d=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True)
        x=x+self.project(attended.transpose(1,2).reshape(b,t,d))
        return x+self.ff(self.norm2(x))


def bucket_batches(bundle,max_batch=8,max_tokens=4096):
    """Unpack only small length buckets; never truncate or mix event histories."""
    starts=np.r_[0,np.cumsum(bundle['batch_sizes'])[:-1]];lengths=bundle['lengths'];groups=[];group=[]
    for i,length in enumerate(lengths):
        if group and (len(group)>=max_batch or int(lengths[group[0]])*(len(group)+1)>max_tokens):groups.append(group);group=[]
        group.append(i)
    if group:groups.append(group)
    result=[]
    for ii in groups:
        n=len(ii);length=int(lengths[ii[0]]);batch={}
        for key in ('x','side','query_side','target','weights','bins'):
            a=bundle[key];out=np.zeros((n,length,*a.shape[1:]),a.dtype)
            if key=='target':out[...,2]=1
            for j,e in enumerate(ii):
                size=int(lengths[e]);out[j,:size]=a[starts[:size]+e]
            batch[key]=torch.from_numpy(out)
        result.append(batch)
    return result


class RawJointTransformer(nn.Module):
    def __init__(self,width=32,heads=4,layers=2,feedforward=64,decoder=16,seed=1729):
        super().__init__();self.settings=dict(width=width,heads=heads,layers=layers,feedforward=feedforward,decoder=decoder,seed=seed)

    def initialize(self,input_width,side_width,tiers):
        s=self.settings;torch.manual_seed(s['seed']);width=s['width']
        if width%s['heads'] or width%2:raise ValueError('Even width must be divisible by attention heads')
        self.input=nn.Linear(input_width,width);self.blocks=nn.ModuleList([CausalBlock(width,s['heads'],s['feedforward']) for _ in range(s['layers'])])
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width+side_width,s['decoder']);self.out=nn.Linear(s['decoder'],tiers)
        # Begin at the transparent linear remaining-points anchor.
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.register_buffer('input_support',torch.ones(input_width));self.register_buffer('side_support',torch.ones(side_width))
        self.register_buffer('trained',torch.ones(tiers,dtype=torch.bool));return self

    def encode(self,x):
        h=self.input(x*self.input_support);length=x.shape[1];width=h.shape[-1]
        position=torch.arange(length,device=h.device,dtype=h.dtype)[:,None]
        frequency=torch.exp(torch.arange(0,width,2,device=h.device,dtype=h.dtype)*(-math.log(10000)/width))
        angles=position*frequency;pe=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        h=h+pe
        for block in self.blocks:h=block(h)
        return self.norm(h)

    def decode(self,h,side):return self.out(torch.tanh(self.head(torch.cat((h,side*self.side_support),dim=-1))))

    def forward(self,x,side,query_side):
        h=self.encode(x)
        return torch.stack((self.decode(h,side),self.decode(h,query_side)),dim=2)

    @staticmethod
    def errors(raw,target):
        delta=3*torch.tanh(raw/3)
        return (target[...,0]+target[...,1]*delta.exp()-target[...,2])/target[...,2].clamp(min=1)

    def fit(self,bundle,epochs=120,learning_rate=.003,bias_penalty=3,decay=.001,progress=None):
        if bundle['weights'].sum()<=0:raise ValueError('No weighted training targets')
        self.initialize(bundle['x'].shape[-1],bundle['side'].shape[-1],bundle['target'].shape[-2])
        self.input_support.copy_(torch.as_tensor(np.ptp(bundle['x'],axis=0)>1e-8))
        self.side_support.copy_(torch.as_tensor(np.ptp(np.concatenate((bundle['side'],bundle['query_side'])),axis=0)>1e-8))
        self.trained.copy_(torch.as_tensor(bundle['weights'].sum(axis=(0,1))>0))
        batches=bucket_batches(bundle);total=float(bundle['weights'].sum());optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate,weight_decay=decay)
        for batch in batches:
            if not all(torch.isfinite(batch[k]).all() for k in ('x','side','query_side','target','weights')):raise ValueError('Nonfinite training input')
            batch['weights']/=total
        self.train()
        for epoch in range(1,epochs+1):
            # Exact full-bank bias gradient, without retaining all event graphs.
            # No dropout: both passes see the identical deterministic function.
            with torch.no_grad():
                mass=torch.zeros((5,len(self.trained)));signed=mass.clone();mse=0.
                for b in batches:
                    error=self.errors(self(b['x'],b['side'],b['query_side']),b['target']);w=b['weights']
                    mse+=float((w*error.square()).sum())
                    for h in range(5):
                        selected=b['bins']==h;mass[h]+=w[selected].sum(0);signed[h]+=(w*error)[selected].sum(0)
                means=signed/mass.clamp(min=1e-30);loss_value=mse+bias_penalty*float((mass*means.square()).sum())
            optimizer.zero_grad()
            for b in batches:
                error=self.errors(self(b['x'],b['side'],b['query_side']),b['target']);w=b['weights']
                loss=(w*error.square()).sum()+2*bias_penalty*(w*error*means[b['bins']]).sum()
                loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(),1.);optimizer.step()
            if progress and (epoch==1 or epoch%20==0 or epoch==epochs):progress(epoch,loss_value)
        self.eval();return self

    @torch.no_grad()
    def forecast(self,x,side,slots):
        self.eval();x=torch.as_tensor(x,dtype=torch.float32)[None];side=torch.as_tensor(side,dtype=torch.float32)
        h=self.encode(x)[0,torch.as_tensor(slots,dtype=torch.long)];raw=self.decode(h,side)
        delta=3*torch.tanh(raw/3);delta[:,~self.trained]=0
        return delta.numpy()

    def save(self,path):
        torch.save(dict(settings=self.settings,input_width=self.input.in_features,side_width=len(self.side_support),tiers=len(self.trained),state_dict=self.state_dict()),path)

    @classmethod
    def load(cls,path):
        p=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(**p['settings']).initialize(p['input_width'],p['side_width'],p['tiers']);model.load_state_dict(p['state_dict']);model.eval();return model
