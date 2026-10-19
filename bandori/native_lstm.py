"""Packed full-history raw LSTM at every native tracker update."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from .lstm import relative_objective


class NativeJointLSTM(nn.Module):
    def __init__(self,hidden=32,decoder=16,seed=1729):
        super().__init__();self.hidden=hidden;self.decoder=decoder;self.seed=seed

    def initialize(self,width,side_width,tiers):
        torch.manual_seed(self.seed);self.cell=nn.LSTM(width,self.hidden)
        self.head=nn.Linear(self.hidden+side_width,self.decoder);self.out=nn.Linear(self.decoder,tiers)
        self.register_buffer('input_support',torch.ones(width));self.register_buffer('side_support',torch.ones(side_width))
        self.register_buffer('trained',torch.ones(tiers,dtype=torch.bool));return self

    def decode(self,h,side):return self.out(torch.tanh(self.head(torch.cat((h,side*self.side_support),dim=-1))))

    def packed_forward(self,x,side,query_side,batch_sizes):
        states,_=self.cell(PackedSequence(x*self.input_support,batch_sizes))
        h=states.data
        return torch.stack((self.decode(h,side),self.decode(h,query_side)),dim=1)

    def fit(self,bundle,epochs=120,learning_rate=.005,bias_penalty=3,decay=.001,progress=None):
        x=torch.from_numpy(bundle['x']);side=torch.from_numpy(bundle['side']);query=torch.from_numpy(bundle['query_side'])
        y=torch.from_numpy(bundle['target']);w=torch.from_numpy(bundle['weights']);bins=torch.as_tensor(bundle['bins'],dtype=torch.long)
        sizes=torch.as_tensor(bundle['batch_sizes'],dtype=torch.long)
        if w.sum()<=0 or not all(torch.isfinite(v).all() for v in (x,side,query,y,w)):raise ValueError('Invalid native LSTM bundle')
        self.initialize(x.shape[-1],side.shape[-1],y.shape[-2])
        self.input_support.copy_(torch.as_tensor(np.ptp(bundle['x'],axis=0)>1e-8))
        both=np.concatenate((bundle['side'],bundle['query_side']))
        self.side_support.copy_(torch.as_tensor(np.ptp(both,axis=0)>1e-8));self.trained.copy_(w.sum(dim=(0,1))>0)
        optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate,weight_decay=decay);k=y.shape[-2]
        for epoch in range(1,epochs+1):
            optimizer.zero_grad();raw=self.packed_forward(x,side,query,sizes)
            loss=relative_objective(raw.reshape(-1,k),y.reshape(-1,k,3),w.reshape(-1,k),bins.reshape(-1),bias_penalty)
            loss.backward();nn.utils.clip_grad_norm_(self.parameters(),1.);optimizer.step()
            if progress and (epoch==1 or epoch%20==0 or epoch==epochs):progress(epoch,float(loss.detach()))
        return self

    @torch.no_grad()
    def forecast(self,x,side,slots):
        # Use the same PackedSequence CPU path as training. No padded history
        # or fractional recurrent writeback is needed for an inference query.
        x=torch.as_tensor(x,dtype=torch.float32);side=torch.as_tensor(side,dtype=torch.float32)
        h,_=self.cell(PackedSequence(x*self.input_support,torch.ones(len(x),dtype=torch.long)))
        raw=self.decode(h.data[torch.as_tensor(slots,dtype=torch.long)],side)
        delta=3*torch.tanh(raw/3);delta[:,~self.trained]=0
        return delta.numpy()

    def save(self,path):
        torch.save(dict(hidden=self.hidden,decoder=self.decoder,seed=self.seed,width=self.cell.input_size,
            side_width=self.head.in_features-self.hidden,tiers=self.out.out_features,state_dict=self.state_dict()),path)

    @classmethod
    def load(cls,path):
        p=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(p['hidden'],p['decoder'],p['seed']).initialize(p['width'],p['side_width'],p['tiers'])
        model.load_state_dict(p['state_dict']);return model
