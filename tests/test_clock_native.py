from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from bandori.clock_features import FEATURES, circular_peak, observe
from bandori.daily_observer import DailyObserver
from bandori.experiments import build_cases
from bandori.native_lstm import NativeJointLSTM
from bandori.native_sequence import forecast_event, native_times, pack, stream
from fixtures import config, dataset


class ClockFeaturesTests(unittest.TestCase):
    def test_peak_averaging_wraps_midnight(self):
        sine,cosine,concentration=circular_peak([23,1],[1,1])
        self.assertAlmostEqual(sine,0,places=10);self.assertGreater(cosine,.99);self.assertGreater(concentration,.95)

    def test_clock_slopes_and_future_labels_are_causal(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        cases,_=build_cases(data,cfg,[event.event_id],list(data.events));c=cases[-1]
        daily=DailyObserver(data,cfg,list(data.events)).observe(c)
        before,pace=observe(c,data,daily['profile'],daily['pace'])
        self.assertEqual(len(before),len(FEATURES));self.assertTrue(np.isfinite(before).all())
        self.assertGreater(before[FEATURES.index('ordinary_coverage')],0)
        for (eid,tier),curve in data.curves.items():
            if eid==event.event_id:
                curve.scores[curve.times>c.cutoff]*=1000;data.labels[(eid,tier)]*=1000
        after,newpace=observe(c,data,daily['profile'],daily['pace'])
        np.testing.assert_array_equal(before,after);self.assertEqual(pace,newpace)


class NativeSequenceTests(unittest.TestCase):
    def test_resume_accepts_json_roundtrip_but_rejects_training_changes(self):
        import copy,json
        from bandori.native_experiment import LEGACY_RUNNER_SHA,compatible_plan
        original=dict(config=dict(tiers=(500,1500)),epochs=120,
            code={'native_experiment.py':LEGACY_RUNNER_SHA,'native_lstm.py':'frozen'})
        saved=json.loads(json.dumps(original));self.assertTrue(compatible_plan(saved,original))
        resumed=copy.deepcopy(original);resumed['code']['native_experiment.py']='resume-fix'
        self.assertTrue(compatible_plan(saved,resumed))
        resumed['epochs']=121;self.assertFalse(compatible_plan(saved,resumed))
        resumed['epochs']=120;resumed['code']['native_lstm.py']='changed'
        self.assertFalse(compatible_plan(saved,resumed))

    def test_cpu_execution_matches_stock_values_and_full_gradients(self):
        from bandori.native_cpu import CPUNativeLSTM
        rng=np.random.default_rng(7);sizes=torch.tensor([3,3,2,2,1],dtype=torch.long);n=int(sizes.sum())
        x=torch.tensor(rng.normal(size=(n,7)),dtype=torch.float32,requires_grad=True)
        side=torch.tensor(rng.normal(size=(n,5)),dtype=torch.float32)
        base=NativeJointLSTM(4,3).initialize(7,5,2);base.input_support[0]=0
        fast=CPUNativeLSTM(4,3).initialize(7,5,2);fast.load_state_dict(base.state_dict())
        a=base.packed_forward(x,side,side*.3,sizes);b=fast.packed_forward(x,side,side*.3,sizes)
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)
        a.square().sum().backward();gradient=x.grad.clone();x.grad=None;b.square().sum().backward()
        torch.testing.assert_close(gradient,x.grad,atol=2e-6,rtol=2e-5)
        for p,q in zip(base.parameters(),fast.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)

    def test_every_published_timestamp_and_no_future_publication(self):
        data=dataset(3);event=data.eligible_events(config())[0];curve=data.curves[(event.event_id,500)]
        cutoff=event.start+2*3600;curve.available[1]=cutoff+30
        times=native_times(data,event,(500,),cutoff)
        self.assertNotIn(curve.times[1],times)
        later=native_times(data,event,(500,),cutoff+31)
        self.assertIn(cutoff+30,later)
        s=stream(data,event,(500,1500));hourly=stream(data,event,(500,1500),cadence='hourly')
        self.assertGreater(len(s.times),len(hourly.times))
        # No observation exists at the initial state; do not fabricate an age
        # for a tier whose score has never been observed.
        np.testing.assert_array_equal(s.query_side[0,22:24],[0,0])

    def test_packing_keeps_histories_separate_and_saved_queries_causal(self):
        torch.set_num_threads(2);data=dataset(5);cfg=config(characteristic_time_days=730)
        ids=[e.event_id for e in data.eligible_events(cfg)];cutoff=data.events[ids[-1]].end+86400
        streams=[stream(data,data.events[e],cfg.tiers) for e in ids[:3]]
        bundle=pack(streams,data,cfg,cutoff)
        self.assertEqual(bundle['native_steps'],sum(len(s.times) for s in streams))
        self.assertEqual(bundle['batch_sizes'].sum(),bundle['native_steps'])
        model=NativeJointLSTM(4,3).fit(bundle,epochs=2)
        event=data.events[ids[-1]];query=event.start+29.37*3600
        before,_=forecast_event(model,data,event,cfg.tiers,[query])
        for (eid,tier),curve in data.curves.items():
            if eid==event.event_id:
                curve.scores[curve.times>query]*=10000;data.labels[(eid,tier)]*=10000
        after,_=forecast_event(model,data,event,cfg.tiers,[query]);np.testing.assert_array_equal(before,after)
        # Arbitrarily close queries use the same state and a continuous decoder,
        # rather than applying an extra complete LSTM step off the grid.
        near=event.start+29*3600
        a,_=forecast_event(model,data,event,cfg.tiers,[near,near+.001])
        np.testing.assert_allclose(a[0],a[1],rtol=1e-5,atol=1)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'native.pt';model.save(path)
            predicted,_=forecast_event(NativeJointLSTM.load(path),data,event,cfg.tiers,[query])
            np.testing.assert_array_equal(after,predicted)

    def test_three_times_rule_weights_are_unchanged_by_native_density(self):
        from collect_bandori_data import iso
        data=dataset(5);cfg=config(characteristic_time_days=730);events=data.eligible_events(cfg)
        streams=[stream(data,e,cfg.tiers) for e in events]
        modified=replace(cfg,post_rule_change_at=iso(events[3].start),post_rule_change_multiplier=3)
        cutoff=events[-1].end+86400
        a=pack(streams,data,cfg,cutoff);b=pack(streams,data,modified,cutoff)
        # The streams have identical lengths; packed rows alternate all events.
        order_ids=np.array(sorted([s.event_id for s in streams],key=lambda e:e))
        row_ids=np.tile(order_ids,len(streams[0].times))
        old,new=events[2].event_id,events[3].event_id
        mass=lambda bundle,e:bundle['weights'][row_ids==e].sum()
        ratio=(mass(b,new)/mass(b,old))/(mass(a,new)/mass(a,old))
        self.assertAlmostEqual(ratio,3,places=5)

    def test_dense_burst_does_not_dominate_wall_time_weight(self):
        data=dataset(1);cfg=config();event=next(iter(data.events.values()))
        dense=np.arange(event.end-12*3600,event.end,60.)
        for key,c in data.curves.items():
            t=np.unique(np.r_[c.times,dense]);s=np.interp(t,c.times,c.scores)
            data.curves[key]=type(c)(t,s,t.copy())
        s=stream(data,event,cfg.tiers);bundle=pack([s],data,cfg,event.end+86400)
        late=s.times>=event.end-12*3600;early=(s.times>=event.end-24*3600)&~late
        self.assertGreater(late.sum()/early.sum(),10)
        mass=bundle['weights'].sum(axis=(1,2))
        self.assertLess(mass[late].sum()/mass[early].sum(),1.03)


class ClockModelIntegrationTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'old'/'dist_clock_v2/model/deployment.json').exists(),'Build the optional clock package first')
    def test_portable_clock_candidate_matches_full_feature_pipeline(self):
        from bandori.ridge_runtime import RidgePredictor
        from bandori.live_data import load_snapshot
        from collect_bandori_data import timestamp
        root=Path(__file__).resolve().parents[1]/'old'/'dist_clock_v2'
        model=RidgePredictor(root/'model');data=load_snapshot(root/'examples/event_326_20261017_2000.json.gz')
        result=model.predict(data,326,timestamp('2026-10-17T20:00:00+08:00'))
        self.assertEqual(len(model.mask),1472)
        points={r['tier']:r.get('final') for r in result['predictions']}
        self.assertEqual({t:points[t] for t in (500,1000,1500,2000)},
                         {500:3335922,1000:1974652,1500:1974652,2000:1434498})

    def test_train_only_clock_compression_and_saved_all_tier_forecasts(self):
        from contextlib import redirect_stdout
        import io,json
        from bandori.clock_experiment import context,finish,fit_models,load_fit,predictions
        from bandori.joint import JointNet
        from bandori.role_analog import build_cases
        data=dataset(14);cfg=config(epochs=1);events=data.eligible_events(cfg)
        ids=[e.event_id for e in events[:10]];validation=[e.event_id for e in events[10:11]]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';(source/'models').mkdir(parents=True)
            x=np.zeros((2,3));target=np.tile([100.,50.,180.],(2,2,1));weights=np.ones((2,2))
            JointNet(2).fit(x,target,weights,np.zeros(2,int),cfg).save(source/'models/template_standard_h8.npz')
            components={n:dict(kind='nn',representation=rep,cores=['template_standard_h8'],gain=.5) for n,rep in
                (('nn_minimal','minimal'),('nn_standard','standard'),('nn_rich','rich'),('nn_pca','pca'),('bagged','standard'))}
            base=dict(components=dict(calibrated_stat=dict(kind='statistical',alpha=.5),trajectory=dict(kind='trajectory',bandwidth=.3)))
            observer=dict(components={n:dict(kind='old') for n in ('calibrated_stat','trajectory')})
            components.update(ridge=dict(gain=.5),hierarchical=dict(gain=1),
                **{n:dict(base=base,observer=observer) for n in ('calibrated_stat','trajectory')})
            (source/'components.json').write_text(json.dumps(components))
            cases,_,_=build_cases(data,cfg,ids,ids,hours=(8,));ctx=context(cases,data,cfg,ids)
            with redirect_stdout(io.StringIO()):
                cores,specs,compression,clock,masks=fit_models(ctx,data,cfg,ids,events[10].start,source,root/'fit')
            cc,_,_=build_cases(data,cfg,validation,ids,hours=(8,));check=context(cc,data,cfg,ids)
            check,_,_=finish(check,compression,clock,masks);before=predictions(check,specs,cores,cfg.tiers)
            loaded,ss,pc,clock_pc,mm=load_fit(root/'fit');again=context(cc,data,cfg,ids)
            again,_,_=finish(again,pc,clock_pc,mm);after=predictions(again,ss,loaded,cfg.tiers)
            for n in before:np.testing.assert_array_equal(before[n],after[n])
            for c in cc:
                curve=data.curves[(c.event_id,c.tier)];curve.scores[curve.times>max(t.cutoff for t in cc)]*=1000
                data.labels[(c.event_id,c.tier)]*=1000
            changed=context(cc,data,cfg,ids);changed,_,_=finish(changed,pc,clock_pc,mm)
            for n,p in predictions(changed,ss,loaded,cfg.tiers).items():np.testing.assert_array_equal(before[n],p)


if __name__=='__main__':unittest.main()
