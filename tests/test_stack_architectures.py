from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bandori.joint import JointNet
from bandori.stack_architectures import StackNetwork
from bandori.stack_loo_data import FeatureScale, prepare, raw_features, train_cases
from bandori.role_analog import build_cases
from fixtures import config, dataset


class ArchitectureTests(unittest.TestCase):
    def arrays(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(7,50));target=np.zeros((7,2,3))
        target[:,:,0]=rng.uniform(1,3,(7,2));target[:,:,1]=rng.uniform(4,6,(7,2));target[:,:,2]=rng.uniform(6,9,(7,2))
        weights=rng.uniform(.1,1,(7,2));bins=np.arange(7)%5
        return x,target,weights,bins

    def test_new_architecture_gradients(self):
        x,target,w,bins=self.arrays();cfg=config(tiers=(500,1500))
        for settings in [dict(hidden=(3,2)),dict(hidden=(3,),local_skip=True),dict(hidden=(3,),shared=True)]:
            with self.subTest(settings=settings):
                model=StackNetwork(**settings).initialize(x,target,w,cfg.tiers)
                model.v[:]=np.random.default_rng(9).normal(0,.1,model.v.shape)
                if model.local_skip:model.skip[:]=.02
                z=np.clip((x-model.mean)/model.scale,-6,6)
                _,grads=model.gradients(z,target,w,bins,3,.003)
                for p,g in zip(model.parameters(),grads):
                    at=tuple(0 for _ in p.shape);old=p[at];eps=1e-6
                    p[at]=old+eps;high=model.gradients(z,target,w,bins,3,.003)[0]
                    p[at]=old-eps;low=model.gradients(z,target,w,bins,3,.003)[0]
                    p[at]=old
                    self.assertAlmostEqual((high-low)/(2*eps),g[at],places=6)

    def test_vanilla_matches_jointnet_and_every_variant_roundtrips(self):
        x,target,w,bins=self.arrays();cfg=config(tiers=(500,1500),epochs=4)
        original=JointNet(8,1729).fit(x,target,w,bins,cfg)
        new=StackNetwork((8,),1729).fit(x,target,w,bins,cfg)
        np.testing.assert_allclose(original.predict(x),new.predict(x),atol=1e-14,rtol=0)
        for settings in [dict(hidden=(3,2)),dict(hidden=(3,),local_skip=True),dict(hidden=(3,),shared=True)]:
            model=StackNetwork(**settings).fit(x,target,w,bins,cfg)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'network.npz';model.save(path)
                np.testing.assert_array_equal(model.predict(x),StackNetwork.load(path).predict(x))


class FoldTests(unittest.TestCase):
    def test_selected_features_match_the_legacy_stack_path(self):
        from bandori.representations import Compression, matrices
        from bandori.ridge_runtime import ridge_inputs
        from bandori.daily_observer import SUMMARIES
        from bandori.pipeline import training_arrays
        data=dataset(8);cfg=config(min_reference_events=3)
        events=sorted(data.events.values(),key=lambda e:e.start);ids=[e.event_id for e in events]
        cases,_,_=build_cases(data,cfg,ids[-3:],ids)
        raw,index,bins=raw_features(cases,data,cfg)
        targets,weights=training_arrays(cases,index,data,cfg,events[-1].end+86400)
        scale=FeatureScale().fit(raw,weights,cfg.tiers)
        mat,_,_=matrices(cases,data,cfg.tiers)
        comp=Compression().fit(mat['rich'],weights,cfg.tiers)
        k=len(cfg.tiers);pc=comp.projection.shape[1]
        width=pc+4*k+len(SUMMARIES)*k+21*k
        full,idx=ridge_inputs(cases,data,cfg,ids,comp,np.ones(width))
        selected=np.r_[np.arange(pc,pc+4*k),np.arange(width-21*k,width)]
        expected=full[:,selected];expected*=np.ptp(expected[weights.sum(axis=1)>0],axis=0)>1e-8
        np.testing.assert_array_equal(idx,index)
        np.testing.assert_allclose(scale.transform(raw),expected,atol=1e-12,rtol=0)

    def test_held_outcomes_and_curves_cannot_change_training_features(self):
        data=dataset(10);cfg=config(min_reference_events=3)
        events=sorted(data.events.values(),key=lambda e:e.start);refs=[e.event_id for e in events]
        supervised=refs[-5:];held=supervised[1]
        base,_,_=build_cases(data,cfg,supervised,refs)
        cutoff=events[-1].end+86400
        first=prepare(data,cfg,supervised,refs,base,cutoff,held)
        for key,c in data.curves.items():
            if key[0]==held:
                c.scores*=1000;data.labels[key]*=1000
        history,cases,rebuilt=train_cases(data,cfg,supervised,refs,base,held)
        raw,index,bins=raw_features(cases,history,cfg)
        from bandori.pipeline import training_arrays
        targets,weights=training_arrays(cases,index,history,cfg,cutoff)
        scale=FeatureScale().fit(raw,weights,cfg.tiers)
        self.assertTrue(rebuilt)
        self.assertNotIn(held,history.events)
        self.assertFalse(any(e==held for e,t in history.curves))
        np.testing.assert_array_equal(first.x,scale.transform(raw))
        np.testing.assert_array_equal(first.targets,targets)
        np.testing.assert_array_equal(first.weights,weights)


if __name__=='__main__':unittest.main()
