"""Compatible LSTM improvements without changing the raw reference model."""
import numpy as np
import torch

from .lstm import RawJointLSTM


class BatchedQueryLSTM(RawJointLSTM):
    """Same recurrent cell; batch independent query branches and decoders.

    Main states stay sequential and causal. Branching states never write back.
    This changes execution, not the mathematical architecture or objective.
    """
    def forward_tensors(self,x,side,step_mask,query_x=None,query_side=None,query_mask=None):
        length,n=x.shape[:2]
        h=torch.zeros((n,self.hidden),device=self.device);c=h.clone();states_h=[];states_c=[]
        for step in range(length):
            h,c=self._step(x[step],h,c,step_mask[step]);states_h.append(h);states_c.append(c)
        main_h=torch.stack(states_h);raw=self._decode(main_h.reshape(-1,self.hidden),side.reshape(length*n,-1)).reshape(length,n,-1)
        if query_x is None:return raw
        main_c=torch.stack(states_c)
        qh,qc=self._step(query_x.reshape(length*n,-1),main_h.reshape(-1,self.hidden),
                         main_c.reshape(-1,self.hidden),query_mask.reshape(-1))
        query_raw=self._decode(qh,query_side.reshape(length*n,-1)).reshape(length,n,-1)
        return torch.stack((raw,query_raw),dim=1)


class SupportedInputLSTM(BatchedQueryLSTM):
    """Mask constant/unobserved training inputs before they reach the network.

    An unseen new tier or rule flag cannot activate a randomly initialized,
    never-trained input connection at inference. Masks are fitted on training
    observations only, and are saved with the model.
    """
    def initialize(self,input_width,side_width,tiers):
        super().initialize(input_width,side_width,tiers)
        self.register_buffer('input_support',torch.ones(input_width,device=self.device))
        self.register_buffer('side_support',torch.ones(side_width,device=self.device))
        return self

    def _step(self,x,h,c,mask):
        return super()._step(x*self.input_support,h,c,mask)

    def _decode(self,h,side):return super()._decode(h,side*self.side_support)

    def fit(self,bundle,target,weights,bins,epochs=120,learning_rate=.005,bias_penalty=3,decay=.001,progress=None):
        # The parent's initializer runs inside fit. Compute support now and
        # attach it as soon as initialization completes, before the first step.
        supports=[]
        for main,query in (('x','query_x'),('side','query_side')):
            values=np.concatenate((bundle[main][bundle['step_mask']>0],bundle[query][bundle['query_mask']>0]))
            supports.append((np.ptp(values,axis=0)>1e-8).astype(np.float32))
        self._pending_support=supports
        try:return super().fit(bundle,target,weights,bins,epochs,learning_rate,bias_penalty,decay,progress)
        finally:del self._pending_support

    def to_tensors(self,bundle):
        if hasattr(self,'_pending_support'):
            self.input_support.copy_(torch.as_tensor(self._pending_support[0],device=self.device))
            self.side_support.copy_(torch.as_tensor(self._pending_support[1],device=self.device))
        return super().to_tensors(bundle)
