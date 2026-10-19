from pathlib import Path
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest

import numpy as np

from bandori.rules import RULE_START, era, features
from fixtures import config, dataset


class RuleWeightTests(unittest.TestCase):
    def test_exact_three_times_weight_by_date_not_event_id(self):
        from bandori.dataset import Event
        cfg=config(characteristic_time_days=730,post_rule_change_at='2026-05-26T00:00:00+08:00',post_rule_change_multiplier=3)
        self.assertAlmostEqual(cfg.event_weight(730,RULE_START),3/np.e)
        self.assertAlmostEqual(cfg.event_weight(730,RULE_START-1),1/np.e)
        reordered_old=Event(312,RULE_START-180*86400,RULE_START-170*86400,'mission_live')
        reordered_new=Event(310,RULE_START+10*86400,RULE_START+18*86400,'festival')
        self.assertEqual(era(reordered_old),0);self.assertEqual(era(reordered_new),1)
        self.assertEqual(cfg.event_weight(0,reordered_old.start),1)
        self.assertEqual(cfg.event_weight(0,reordered_new.start),3)
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):config(post_rule_change_multiplier=value)
        with self.assertRaises(ValueError):config(post_rule_change_at='2026-05-26')

    def test_old_frozen_configs_keep_original_weights(self):
        cfg=config(characteristic_time_days=730)
        self.assertEqual(cfg.event_weight(730,RULE_START+86400),cfg.recency_weight(730))

    def test_actual_story_and_festival_types_have_distinct_metadata(self):
        from bandori.dataset import Event
        a=Event(1,RULE_START,RULE_START+8*86400,'story');b=Event(2,a.start,a.end,'festival')
        self.assertFalse(np.array_equal(features(a,1500,a.start+86400),features(b,1500,b.start+86400)))

    def test_supervised_and_raw_lstm_weight_odds_are_tripled(self):
        from bandori.experiments import build_cases
        from bandori.joint import pack
        from bandori.pipeline import training_arrays
        from bandori.raw_sequence import training_streams,pack_streams
        from collect_bandori_data import iso
        data=dataset(12);cfg=config(characteristic_time_days=730)
        events=data.eligible_events(cfg);ids=[e.event_id for e in events]
        boundary=events[8].start
        changed=replace(cfg,post_rule_change_at=iso(boundary),post_rule_change_multiplier=3)
        cases,_=build_cases(data,cfg,ids,ids);index=pack(cases,cfg.tiers)[1];cutoff=events[-1].end+86400
        _,a=training_arrays(cases,index,data,cfg,cutoff);_,b=training_arrays(cases,index,data,changed,cutoff)
        groups=[next(c.event_id for c in [cases[i] for i in row if i>=0]) for row in index]
        old=events[7].event_id;new=events[8].event_id
        def mass(array,eid):return array[np.array(groups)==eid].sum()
        self.assertAlmostEqual((mass(b,new)/mass(b,old))/(mass(a,new)/mass(a,old)),3)
        streams=training_streams(data,ids,cfg.tiers)
        _,_,a,_=pack_streams(streams,data,cfg,cutoff);_,_,b,_=pack_streams(streams,data,changed,cutoff)
        self.assertAlmostEqual((b[:,:,8].sum()/b[:,:,7].sum())/(a[:,:,8].sum()/a[:,:,7].sum()),3)

    def test_reward_role_prior_never_creates_surrogate_target_labels(self):
        from bandori.role_analog import RoleAwareAnalog
        from bandori.dataset import Dataset,Curve
        from bandori.full_refit import issue_times
        data=dataset(24);shift=3*365*86400
        events=[replace(e,start=e.start+shift,end=e.end+shift) for e in data.events.values()]
        curves={};labels={}
        for (eid,tier),curve in data.curves.items():
            rank=1000 if tier==1500 else tier
            curves[(eid,rank)]=Curve(curve.times+shift,curve.scores.copy(),curve.available+shift)
            labels[(eid,rank)]=data.labels[(eid,tier)]
        event=max(events,key=lambda e:e.start)
        curves[(event.event_id,1500)]=curves[(event.event_id,1000)]
        labels[(event.event_id,1500)]=labels[(event.event_id,1000)]
        copied=Dataset(events,curves,labels);cfg=config(tiers=(500,1000,1500),characteristic_time_days=730)
        predictor=RoleAwareAnalog(copied,cfg,list(copied.events));cutoff=next(issue_times(event,(8,)))
        before=predictor.build(event,1500,cutoff)
        self.assertGreater(before.features['role_transfer_support'],0)
        self.assertEqual(before.references,())
        self.assertEqual([e for e,t in copied.labels if t==1500],[event.event_id])
        copied.labels[(event.event_id,1500)]*=1000
        curve=copied.curves[(event.event_id,1500)];curve.scores[curve.times>cutoff]*=1000
        after=RoleAwareAnalog(copied,cfg,list(copied.events)).build(event,1500,cutoff)
        self.assertEqual(before,after)


class LSTMVariantTests(unittest.TestCase):
    def test_query_batching_preserves_values_and_gradients(self):
        import torch
        from bandori.lstm import RawJointLSTM
        from bandori.lstm_variants import BatchedQueryLSTM
        torch.set_num_threads(1);rng=np.random.default_rng(2);length,n=5,3
        bundle={name:torch.tensor(rng.normal(size=(length,n,width)),dtype=torch.float32) for name,width in
                (('x',7),('side',4),('query_x',7),('query_side',4))}
        mask=torch.ones((length,n));mask[-2:,1]=0
        base=RawJointLSTM(4,3,device='cpu').initialize(7,4,2)
        fast=BatchedQueryLSTM(4,3,device='cpu').initialize(7,4,2);fast.load_state_dict(base.state_dict())
        arguments=(bundle['x'],bundle['side'],mask,bundle['query_x'],bundle['query_side'],mask)
        a=base.forward_tensors(*arguments);b=fast.forward_tensors(*arguments)
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)
        a.square().sum().backward();b.square().sum().backward()
        for p,q in zip(base.parameters(),fast.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)

    def test_unseen_inputs_are_masked_and_saved(self):
        from bandori.lstm_variants import SupportedInputLSTM
        rng=np.random.default_rng(3);length,n,k=4,2,2
        bundle={name:rng.normal(size=(length,n,width)) for name,width in (('x',5),('side',3),('query_x',5),('query_side',3))}
        for key in ('x','query_x'):bundle[key][...,0]=0
        bundle.update(step_mask=np.ones((length,n)),query_mask=np.ones((length,n)))
        target=np.tile([100.,50.,180.],(length,2,n,k,1));weights=np.ones((length,2,n,k));bins=np.zeros((length,2,n),int)
        model=SupportedInputLSTM(3,3,device='cpu').fit(bundle,target,weights,bins,epochs=1)
        before=model.predict_corrections(bundle)
        for key in ('x','query_x'):bundle[key][...,0]=500
        np.testing.assert_array_equal(before,model.predict_corrections(bundle))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'net.pt';model.save(path)
            np.testing.assert_array_equal(before,SupportedInputLSTM.load(path,'cpu').predict_corrections(bundle))


class RegimeSuiteTests(unittest.TestCase):
    def test_cached_prefix_fit_and_saved_all_family_forecasts(self):
        from bandori.experiments import run_suite
        from bandori.observer_experiment import train as observer_train
        from bandori.regime_suite import (build_cases,make_context,subset,save_context,load_context,fit_models,
                                          finish_context,predictions,load_fit,hyperparameters_only)
        data=dataset(30);cfg=config(epochs=1,hidden_sizes=(2,),characteristic_time_days=730)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);registry=root/'ledger.json'
            with redirect_stdout(io.StringIO()):
                run_suite(data,cfg,root/'base',registry)
                observer_train(data,root/'base',root/'observer',registry,temporal_channels=(2,))
            split=json.loads((root/'base/split.json').read_text());ids=split['train']
            base=hyperparameters_only(json.loads((root/'base/selection.json').read_text()))
            observer=hyperparameters_only(json.loads((root/'observer/selection.json').read_text()))
            cases,_,_=build_cases(data,cfg,ids,ids,hours=(8,))
            ctx=make_context(cases,data,cfg,ids);save_context(root,ctx);restored=load_context(root)
            np.testing.assert_array_equal(ctx['matrices']['rich'],restored['matrices']['rich'])
            own=subset(restored,ids)
            with redirect_stdout(io.StringIO()):
                cores,specs,compression,masks,meta=fit_models(own,data,cfg,ids,split['fit_cutoff'],base,observer,root/'base',root/'fit',lstm_epochs=1)
            cases,_,_=build_cases(data,cfg,split['test'][:1],ids,hours=(8,))
            check=make_context(cases,data,cfg,ids);check,_=finish_context(check,compression,masks)
            before=predictions(specs,cores,check,data,cfg.tiers)
            loaded,ss,cc,mm,metadata=load_fit(root/'fit')
            after=predictions(ss,loaded,check,data,cfg.tiers)
            for name in before:np.testing.assert_array_equal(before[name],after[name])
            self.assertIn('raw_lstm',before);self.assertIn('collection_trimmed',before)
            self.assertTrue(all(np.all(p>=np.array([c.current for c in cases])) for p in before.values()))
            with self.assertRaisesRegex(ValueError,'holdout'):
                subset(restored,ids[-2:])


if __name__=='__main__':unittest.main()
