from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bandori.daily_observer import DailyObserver, calendar_integral, observer_arrays, sample
from bandori.dataset import Curve
from bandori.experiments import build_cases, run_suite
from bandori.joint import pack
from bandori.observer_experiment import predict, train
from bandori.observer_refit import predict as predict_full_observer, refit as refit_observer
from bandori.full_refit import refit as refit_control
from bandori.temporal import JointTemporalNet, convolution, convolution_backward
from fixtures import config, dataset


class DailyObserverTests(unittest.TestCase):
    def cases(self,data,cfg):
        event=data.eligible_events(cfg)[-1]
        cases,_=build_cases(data,cfg,[event.event_id],list(data.events))
        return [c for c in cases if c.cutoff==cases[-1].cutoff]

    def test_only_known_prefix_and_earlier_training_references(self):
        data=dataset(12);cfg=config();cases=self.cases(data,cfg);index=pack(cases,cfg.tiers)[1]
        before=observer_arrays(cases,index,data,cfg,list(data.events))
        for c in cases:
            data.labels[(c.event_id,c.tier)]*=10000
            curve=data.curves[(c.event_id,c.tier)];curve.scores[curve.times>c.cutoff]*=10000
        after=observer_arrays(cases,index,data,cfg,list(data.events))
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        # Held-out historical references may not enter the hourly prior.
        observer=DailyObserver(data,cfg,[])
        np.testing.assert_array_equal(observer.profile(cases[0])[0],1)
        self.assertEqual(observer.profile(cases[0])[1],0)

    def test_late_publication_cannot_fill_hourly_bin(self):
        data=dataset(12);cfg=config();case=self.cases(data,cfg)[0];curve=data.curves[(case.event_id,case.tier)]
        mask=(curve.times>=case.observed_at-5*3600)&(curve.times<=case.observed_at-3600)
        curve.available[mask]=case.cutoff+86400
        first=DailyObserver(data,cfg,list(data.events)).observe(case)
        curve.scores[mask]*=10000
        second=DailyObserver(data,cfg,list(data.events)).observe(case)
        np.testing.assert_array_equal(first['sequence'],second['sequence'])
        self.assertLess(first['coverage'],1)

    def test_gap_is_missing_and_known_outage_is_separate(self):
        data=dataset(12);cfg=config();case=self.cases(data,cfg)[0]
        first=DailyObserver(data,cfg,list(data.events)).observe(case)
        data.outages=[(case.event_id,case.observed_at-3*3600,case.observed_at-3600,case.cutoff+1)]
        future=DailyObserver(data,cfg,list(data.events)).observe(case)
        np.testing.assert_array_equal(first['sequence'],future['sequence'])
        data.outages=[(*data.outages[0][:3],case.cutoff)]
        known=DailyObserver(data,cfg,list(data.events)).observe(case)
        self.assertLess(known['sequence'][:,5].sum(),first['sequence'][:,5].sum())

    def test_calendar_integral_matches_hour_by_hour_and_constant_rate(self):
        from collect_bandori_data import timestamp
        start=timestamp('2026-10-16T08:00:00+08:00');profile=np.vstack((np.arange(1,25),np.arange(1,25)*2))
        # Friday 08:00 -> Saturday 10:00.
        expected=sum(range(9,25))+2*sum(range(1,11))
        self.assertAlmostEqual(calendar_integral(profile,start,start+26*3600),expected)
        self.assertAlmostEqual(calendar_integral(np.ones((2,24)),start+.4*3600,start+173.7*3600),173.3,places=8)

    def test_interpolation_never_bridges_long_gap_or_extrapolates(self):
        curve=Curve(np.array([0.,3600.,5*3600]),np.array([0.,10.,50.]),np.zeros(3))
        values,valid=sample(curve,np.array([-1.,1800.,2*3600,5*3600+1]))
        np.testing.assert_array_equal(valid,[False,True,False,False]);self.assertEqual(values[1],5)

    def test_future_schedule_change_does_not_change_observer(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        data.revisions=[(event.event_id,event.start-1,event.start,event.end),
                        (event.event_id,event.end+1,event.start,event.end+86400)]
        case=self.cases(data,cfg)[0]
        before=DailyObserver(data,cfg,list(data.events)).observe(case)
        data.events[event.event_id]=replace(event,end=event.end+86400)
        after=DailyObserver(data,cfg,list(data.events)).observe(case)
        np.testing.assert_array_equal(before['summaries'],after['summaries'])


class TemporalGradientTests(unittest.TestCase):
    def test_dilated_convolution_gradient(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(2,8,3));k=rng.normal(size=(3,3,2));b=np.zeros(2);g=rng.normal(size=(2,8,2))
        dx,dk,db=convolution_backward(x,k,g,3)
        for array,gradient,index in ((x,dx,(0,2,1)),(k,dk,(2,1,0)),(b,db,(1,))):
            before=array[index];epsilon=1e-6
            array[index]=before+epsilon;high=np.sum(convolution(x,k,b,3)*g)
            array[index]=before-epsilon;low=np.sum(convolution(x,k,b,3)*g)
            array[index]=before
            self.assertAlmostEqual((high-low)/(2*epsilon),gradient[index],places=6)

    def test_joint_temporal_gradients_masks_and_roundtrip(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(6,5));seq=rng.normal(size=(6,2,24,10));seq[...,3]=1
        seq[0,0,:4,3]=0;seq[1,1,:,3]=0
        target=np.tile([100.,50.,180.],(6,2,1));weights=np.ones((6,2));weights[:,1]=0;bins=np.arange(6)%5
        model=JointTemporalNet(2,3).fit(x,seq,target,weights,bins,config(epochs=2))
        model.v[:]=rng.normal(0,.1,model.v.shape)
        prepared=model.prepare(x,seq)
        loss,gradients=model.gradients(prepared,target,weights,bins,3,.003)
        for name,gradient in zip(model.PARAMETERS,gradients):
            array=getattr(model,name);index=tuple(0 for _ in array.shape);before=array[index];epsilon=1e-6
            array[index]=before+epsilon;high=model.gradients(prepared,target,weights,bins,3,.003)[0]
            array[index]=before-epsilon;low=model.gradients(prepared,target,weights,bins,3,.003)[0]
            array[index]=before
            self.assertAlmostEqual((high-low)/(2*epsilon),gradient[index],places=6,msg=name)
        np.testing.assert_array_equal(model.predict(x,seq)[:,1],0)
        empty_day=seq.copy();empty_day[...,3]=0
        self.assertTrue(np.isfinite(model.predict(x,empty_day)).all())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.npz';model.save(path)
            np.testing.assert_array_equal(model.predict(x,seq),JointTemporalNet.load(path).predict(x,seq))
        # Samples with no supervised weight cannot alter learned normalization.
        other_x=x.copy();other_seq=seq.copy();other_weights=weights.copy();other_weights[-1]=0
        a=JointTemporalNet(2,3).fit(x,seq,target,other_weights,bins,config(epochs=1))
        other_x[-1]*=10000;other_seq[-1]*=10000;other_seq[-1,...,3]=1
        b=JointTemporalNet(2,3).fit(other_x,other_seq,target,other_weights,bins,config(epochs=1))
        np.testing.assert_array_equal(a.mean,b.mean);np.testing.assert_array_equal(a.seq_mean,b.seq_mean)


class ObserverIntegrationTests(unittest.TestCase):
    def test_paired_experiment_preserves_ledger_and_saved_inference(self):
        data=dataset();cfg=config(epochs=2,hidden_sizes=(2,),characteristic_time_days=730)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);registry=root/'ledger.json'
            with redirect_stdout(io.StringIO()):run_suite(data,cfg,root/'control',registry)
            ledger=registry.read_bytes()
            with redirect_stdout(io.StringIO()):evaluation=train(data,root/'control',root/'daily',registry,temporal_channels=(2,))
            self.assertEqual(registry.read_bytes(),ledger)
            self.assertEqual(evaluation['test_status'],'previously_inspected_follow_up')
            for family in ('baseline','nn_minimal','nn_standard','nn_rich','nn_pca','bagged','ridge','calibrated_stat','trajectory','hierarchical','tcn24','pace24_flat','collection_mean'):
                self.assertIn(family,evaluation['partitions']['test']['observer'])
            split=json.loads((root/'daily/split.json').read_text())
            cases,_=build_cases(data,cfg,split['test'][:1],split['train']);case=cases[0]
            result=predict(data,root/'daily',case.event_id,case.cutoff,root/'fixed.json')
            self.assertEqual([r['tier'] for r in result['predictions']],[500,1500])
            self.assertTrue(all(r['final']>=r['current'] for r in result['predictions']))
            for tier in cfg.tiers:
                data.labels[(case.event_id,tier)]*=1000
                curve=data.curves[(case.event_id,tier)];curve.scores[curve.times>case.cutoff]*=1000
            self.assertEqual(result,predict(data,root/'daily',case.event_id,case.cutoff))
            with self.assertRaisesRegex(ValueError,'already exists'):predict(data,root/'daily',case.event_id,case.cutoff,root/'fixed.json')

    def test_full_refit_retains_exclusions_and_supports_saved_all_tier_heads(self):
        data=dataset(30);cfg=config(epochs=1,hidden_sizes=(2,),characteristic_time_days=730)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);registry=root/'ledger.json'
            with redirect_stdout(io.StringIO()):
                run_suite(data,cfg,root/'control',registry)
                train(data,root/'control',root/'daily',registry,temporal_channels=(2,))
                excluded=[e.event_id for e in data.eligible_events(cfg)[-2:]]
                refit_control(data,root/'control',root/'full_control',excluded,(8,),registry,True)
            ledger=registry.read_bytes()
            with self.assertRaisesRegex(ValueError,'reuse'):
                refit_observer(data,root/'full_control',root/'daily',root/'disallowed',registry)
            with redirect_stdout(io.StringIO()):
                model=refit_observer(data,root/'full_control',root/'daily',root/'full_daily',registry,True)
            self.assertEqual(registry.read_bytes(),ledger)
            self.assertFalse(set(excluded)&set(model['training_event_ids']))
            cases,_=build_cases(data,cfg,excluded[:1],model['training_event_ids']);case=cases[0]
            before=predict_full_observer(data,root/'full_daily',case.event_id,case.cutoff,all_models=True)
            for tier in cfg.tiers:
                data.labels[(case.event_id,tier)]*=1000
                curve=data.curves[(case.event_id,tier)];curve.scores[curve.times>case.cutoff]*=1000
            self.assertEqual(before,predict_full_observer(data,root/'full_daily',case.event_id,case.cutoff,all_models=True))
            self.assertTrue(all('observer_raw' in row and row['final']>=row['current'] for row in before['predictions']))


if __name__=='__main__':unittest.main()
