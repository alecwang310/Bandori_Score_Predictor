"""CPU implementation of the stock LSTM equations with full exact BPTT.

The saved module remains nn.LSTM. This autograd execution avoids thousands
of tiny ATen nodes for a long irregular sequence on CPUs without oneDNN.
"""
import numpy as np
import torch

from .native_lstm import NativeJointLSTM


class PackedLSTMCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,wi,wh,bi,bh,sizes):
        a=x.detach().numpy();w=wi.detach().numpy();v=wh.detach().numpy();batch=sizes.numpy()
        hidden=v.shape[1];n=len(a);gates=a@w.T+(bi.detach().numpy()+bh.detach().numpy())
        h=np.zeros((int(batch[0]),hidden),np.float32);c=h.copy();previous_h=np.zeros((n,hidden),np.float32)
        previous_c=previous_h.copy();states=previous_h.copy();cells=previous_h.copy();offset=0
        for count in batch:
            take=slice(offset,offset+int(count));previous_h[take]=h[:count];previous_c[take]=c[:count]
            z=gates[take]+h[:count]@v.T
            i,f,g,o=np.split(z,4,axis=1)
            i=1/(1+np.exp(-np.clip(i,-60,60)));f=1/(1+np.exp(-np.clip(f,-60,60)))
            g=np.tanh(g);o=1/(1+np.exp(-np.clip(o,-60,60)))
            gates[take]=np.concatenate((i,f,g,o),axis=1)
            c[:count]=f*c[:count]+i*g;h[:count]=o*np.tanh(c[:count])
            states[take]=h[:count];cells[take]=c[:count];offset+=int(count)
        ctx.arrays=(a,w,v,batch,gates,previous_h,previous_c,cells)
        return torch.from_numpy(states)

    @staticmethod
    def backward(ctx,gradient):
        a,w,v,batch,gates,previous_h,previous_c,cells=ctx.arrays;dh=gradient.detach().numpy()
        hidden=v.shape[1];future_h=np.zeros((int(batch[0]),hidden),np.float32);future_c=future_h.copy()
        dg=np.zeros_like(gates);offset=len(a)
        for count in batch[::-1]:
            offset-=int(count);take=slice(offset,offset+int(count));i,f,g,o=np.split(gates[take],4,axis=1)
            tanh=np.tanh(cells[take]);total_h=dh[take]+future_h[:count]
            total_c=total_h*o*(1-tanh*tanh)+future_c[:count]
            di=total_c*g*i*(1-i);df=total_c*previous_c[take]*f*(1-f)
            dgate=total_c*i*(1-g*g);do=total_h*tanh*o*(1-o)
            dz=np.concatenate((di,df,dgate,do),axis=1);dg[take]=dz
            future_h[:count]=dz@v;future_c[:count]=total_c*f
        dx=torch.from_numpy(dg@w) if ctx.needs_input_grad[0] else None
        dw=torch.from_numpy(dg.T@a);dv=torch.from_numpy(dg.T@previous_h);db=torch.from_numpy(dg.sum(axis=0))
        return dx,dw,dv,db,db.clone(),None


class CPUNativeLSTM(NativeJointLSTM):
    def packed_forward(self,x,side,query_side,batch_sizes):
        h=PackedLSTMCPU.apply(x*self.input_support,self.cell.weight_ih_l0,self.cell.weight_hh_l0,
            self.cell.bias_ih_l0,self.cell.bias_hh_l0,batch_sizes)
        return torch.stack((self.decode(h,side),self.decode(h,query_side)),dim=1)
