import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from bandori.lstm import relative_objective
from bandori.native_sequence import pack,stream,forecast_event
from bandori.raw_transformer import RawJointTransformer,bucket_batches
from fixtures import config,dataset


class TransformerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_attention_cannot_see_future_and_padding_does_not_change_prefix(self):
        torch.manual_seed(3);m=RawJointTransformer(width=8,heads=2,layers=2,feedforward=16).initialize(7,5,2)
        x=torch.randn(2,13,7);before=m.encode(x);changed=x.clone();changed[:,7:]+=100
        torch.testing.assert_close(before[:,:7],m.encode(changed)[:,:7],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(before[:,:7],m.encode(x[:,:7]),atol=1e-6,rtol=1e-5)
        loss=m.encode(x)[:,:7].square().sum();x.requires_grad_();m.encode(x)[:,:7].square().sum().backward()
        self.assertEqual(float(x.grad[:,7:].abs().sum()),0)

    def test_two_pass_global_bias_gradient_matches_original_objective(self):
        torch.manual_seed(5);raw=torch.randn(17,3,requires_grad=True);target=torch.rand(17,3,3)*100+1
        weights=torch.rand(17,3);bins=torch.arange(17)%5
        relative_objective(raw,target,weights,bins,3).backward();expected=raw.grad.clone();raw.grad=None
        with torch.no_grad():
            error=RawJointTransformer.errors(raw,target);w=weights/weights.sum();mean=torch.zeros(5,3)
            for h in range(5):mean[h]=(w*error)[bins==h].sum(0)/w[bins==h].sum(0)
        for ii in (slice(0,6),slice(6,17)):
            e=RawJointTransformer.errors(raw[ii],target[ii]);loss=(w[ii]*e.square()).sum()+6*(w[ii]*e*mean[bins[ii]]).sum();loss.backward()
        torch.testing.assert_close(raw.grad,expected,atol=2e-6,rtol=2e-5)

    def test_saved_joint_forecasts_and_causal_native_inputs(self):
        data=dataset(5);cfg=config();events=data.eligible_events(cfg);ids=[e.event_id for e in events[:3]]
        bundle=pack([stream(data,data.events[e],cfg.tiers) for e in ids],data,cfg,events[3].start)
        batches=bucket_batches(bundle,max_batch=2,max_tokens=100)
        self.assertAlmostEqual(sum(float(b['weights'].sum()) for b in batches),float(bundle['weights'].sum()),places=6)
        m=RawJointTransformer(width=8,heads=2,layers=1,feedforward=16,decoder=4).fit(bundle,epochs=2)
        event=events[-1];t=event.start+31.37*3600;before,_=forecast_event(m,data,event,cfg.tiers,[t])
        self.assertEqual(before.shape,(1,len(cfg.tiers)))
        for (eid,tier),c in data.curves.items():
            if eid==event.event_id:c.scores[c.times>t]*=1000;data.labels[(eid,tier)]*=1000
        after,_=forecast_event(m,data,event,cfg.tiers,[t]);np.testing.assert_array_equal(before,after)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.pt';m.save(path);loaded=RawJointTransformer.load(path)
            predicted,_=forecast_event(loaded,data,event,cfg.tiers,[t]);np.testing.assert_array_equal(after,predicted)


if __name__=='__main__':unittest.main()
