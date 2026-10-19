from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from bandori.analog import AnalogPredictor,active_hours,reconcile
from bandori.evaluation import interval_radius,metrics
from bandori.joint import JointNet,objective,pack
from bandori.pipeline import build_cases,predict,split_events,train,training_arrays
from bandori.statistical import fit_calibration,predict as statistical_predict
from collect_bandori_data import checkpoints
from fixtures import config,dataset


class CausalityTests(unittest.TestCase):
    def test_own_final_and_future_tail_do_not_enter_either_system_inputs(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        cutoff=next(checkpoints(event.start,event.end,event.end));refs=list(data.events)
        before=AnalogPredictor(data,cfg,refs).build(event,500,cutoff)
        for tier in cfg.tiers:
            data.labels[(event.event_id,tier)]*=500
            curve=data.curves[(event.event_id,tier)];curve.scores[curve.times>cutoff]*=100
        after=AnalogPredictor(data,cfg,refs).build(event,500,cutoff)
        self.assertEqual(before,after)

    def test_refs_are_training_only_and_available_before_event_started(self):
        data=dataset();cfg=config();split=split_events(data,cfg)
        cases,_=build_cases(data,cfg,split['tune'],split['train'])
        for case in cases:
            self.assertTrue(set(case.references)<=set(split['train']))
            self.assertTrue(all(data.label_available(e,cfg)<=data.events[case.event_id].start for e in case.references))

    def test_unseen_test_and_permanent_holdouts(self):
        data=dataset();cfg=config();initial=split_events(data,cfg)
        protected=initial['train'][5]
        self.assertNotIn(protected,split_events(data,cfg,[protected])['train'])
        with self.assertRaisesRegex(ValueError,'previously inspected'):split_events(data,cfg,initial['test'])

    def test_late_publication_and_freshness_are_respected(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        cutoff=next(checkpoints(event.start,event.end,event.end))
        curve=data.curves[(event.event_id,500)]
        curve.available[curve.times<=cutoff]=cutoff+1
        with self.assertRaisesRegex(ValueError,'No positive score'):
            AnalogPredictor(data,cfg,list(data.events)).build(event,500,cutoff)

    def test_outages_merge_and_future_announcements_are_ignored(self):
        data=dataset(1);event=next(iter(data.events.values()));start=event.start
        data.outages=[(event.event_id,start+3600,start+10800,start+14400),
                     (event.event_id,start+7200,start+14400,start+14400)]
        self.assertEqual(active_hours(data,event,start,start+18000,start+10000),5)
        self.assertEqual(active_hours(data,event,start,start+18000,start+18000),2)

    def test_schedule_revision_is_not_used_before_known(self):
        data=dataset(1);event=next(iter(data.events.values()))
        data.revisions=[(event.event_id,event.start+86400,event.start,event.end+86400)]
        with self.assertRaises(ValueError):data.known_event(event,event.start+3600)

    def test_proxy_labels_require_post_close_observation_and_preserve_archive(self):
        data=dataset(1);event=next(iter(data.events.values()));key=(event.event_id,1500)
        data.labels.pop(key);data.curves[key]=data.curves[key].prefix(event.end-1)
        self.assertEqual(data.add_terminal_labels(config(allow_terminal_labels=True),event.end+2*86400),0)

    def test_known_late_proxy_publication_delays_reference_eligibility(self):
        data=dataset(1);event=next(iter(data.events.values()));key=(event.event_id,1500)
        data.labels.pop(key);data.curves[key].available[-1]=event.end+5*86400
        cfg=config(allow_terminal_labels=True)
        self.assertEqual(data.add_terminal_labels(cfg,event.end+4*86400),0)
        self.assertEqual(data.add_terminal_labels(cfg,event.end+6*86400),1)
        self.assertEqual(data.label_available(event.event_id,cfg),event.end+5*86400)


class NumericalTests(unittest.TestCase):
    def test_signed_error_aggregations_and_event_count_are_explicit(self):
        cases=[SimpleNamespace(event_id=e,tier=500) for e in (1,2,2,2)]
        data=SimpleNamespace(labels={(1,500):100,(2,500):100})
        result=metrics(cases,[110,90,90,90],data)
        self.assertAlmostEqual(result['bias'],0)
        self.assertAlmostEqual(result['row_bias'],-.05)
        self.assertEqual(result['events'],2)
        self.assertIsNone(interval_radius(cases,[110,90,90,90],data,.1)['log_radius'])

    def test_prediction_bands_require_calibration_for_each_tier(self):
        cases=[SimpleNamespace(event_id=e,tier=500) for e in range(10)]
        cases.append(SimpleNamespace(event_id=0,tier=1500))
        data=SimpleNamespace(labels={(c.event_id,c.tier):100 for c in cases})
        result=interval_radius(cases,np.full(len(cases),110),data,.1)
        self.assertIsNotNone(result['by_tier']['500']['log_radius'])
        self.assertIsNone(result['by_tier']['1500']['log_radius'])

    def test_bias_penalized_loss_gradient_and_mask(self):
        raw=np.array([[.2,-.3],[-.1,.1],[.5,.2]])
        target=np.tile([100.,50.,155.],(3,2,1));weights=np.array([[1.,0],[2,1],[1,1]])
        bins=np.array([0,0,1]);loss,gradient=objective(raw,target,weights,bins,3)
        for index in np.ndindex(raw.shape):
            high=raw.copy();low=raw.copy();high[index]+=1e-6;low[index]-=1e-6
            numerical=(objective(high,target,weights,bins,3)[0]-objective(low,target,weights,bins,3)[0])/2e-6
            self.assertAlmostEqual(gradient[index],numerical,places=8)
        target[0,1]=[1000,2000,4000]
        self.assertEqual(loss,objective(raw,target,weights,bins,3)[0])

    def test_all_network_parameter_gradients(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(5,6));target=np.tile([100.,30.,140.],(5,2,1))
        weights=np.ones((5,2));bins=np.array([0,0,1,1,2]);cfg=config(epochs=1)
        net=JointNet(3).fit(x,target,weights,bins,cfg);z=np.clip((x-net.mean)/net.scale,-6,6)
        _,gradients=net.gradients(z,target,weights,bins,3,.003)
        for name,gradient in zip(('a','b','v','d'),gradients):
            parameter=getattr(net,name);index=tuple(0 for _ in parameter.shape);original=parameter[index]
            parameter[index]=original+1e-6;high=net.gradients(z,target,weights,bins,3,.003)[0]
            parameter[index]=original-1e-6;low=net.gradients(z,target,weights,bins,3,.003)[0]
            parameter[index]=original
            self.assertAlmostEqual(gradient[index],(high-low)/2e-6,places=7,msg=name)

    def test_joint_outputs_cold_head_and_serialization(self):
        data=dataset(12);cfg=config();rows,_=build_cases(data,cfg,[data.eligible_events(cfg)[-1].event_id],list(data.events))
        x,index,bins=pack(rows,cfg.tiers)
        self.assertEqual(x.shape[0]*2,len(rows));self.assertEqual(index.shape[1],2)
        targets,weights=training_arrays(rows,index,data,cfg,data.eligible_events(cfg)[-1].start)
        weights[:,1]=0
        model=JointNet(3).fit(x,targets,weights,bins,cfg)
        self.assertTrue((model.predict(x)[:,1]==0).all())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'net.npz';model.save(path)
            np.testing.assert_array_equal(model.predict(x),JointNet.load(path).predict(x))

    def test_statistical_scale_equivariance_and_earned_score_floor(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        cutoff=event.start+72*3600;reference=list(data.events)
        before=AnalogPredictor(data,cfg,reference).build(event,500,cutoff)
        data.curves[(event.event_id,500)].scores*=2
        after=AnalogPredictor(data,cfg,reference).build(event,500,cutoff)
        self.assertAlmostEqual(after.baseline,2*before.baseline)
        self.assertAlmostEqual(after.pace,2*before.pace)
        self.assertGreater(before.pace_references,0)
        np.testing.assert_array_equal(reconcile([before],[0]),[before.current])

    def test_calibration_uses_only_supplied_training_cases(self):
        data=dataset();cfg=config();split=split_events(data,cfg)
        rows,_=build_cases(data,cfg,split['train'],split['train'])
        before=fit_calibration(rows,data,cfg,.5,split['fit_cutoff'])
        for key in data.labels:
            if key[0] in split['tune']+split['test']+split['calibration']:data.labels[key]*=100
        self.assertEqual(before,fit_calibration(rows,data,cfg,.5,split['fit_cutoff']))


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_two_systems_and_frozen_reference_bank(self):
        data=dataset();cfg=config()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with redirect_stdout(io.StringIO()):result=train(data,cfg,root/'run',root/'ledger.json')
            self.assertEqual(set(result['partitions']['test']),{'baseline','baseline_nn','statistical'})
            split=json.loads((root/'run/split.json').read_text());event=data.events[split['test'][0]]
            cutoff=next(checkpoints(event.start,event.end,event.end))
            before=predict(data,root/'run',event.event_id,cutoff,root/'fixed.json')
            self.assertEqual([r['tier'] for r in before['predictions']],[500,1500])
            self.assertTrue(all(r['status']=='ok' for r in before['predictions']))
            for key in data.curves:
                if key[0] in split['train']:
                    data.curves[key].scores*=1000;data.labels[key]*=1000
            self.assertEqual(before,predict(data,root/'run',event.event_id,cutoff))
            with self.assertRaisesRegex(ValueError,'already exists'):predict(data,root/'run',event.event_id,cutoff,root/'fixed.json')
            with self.assertRaisesRegex(ValueError,'08:00'):predict(data,root/'run',event.event_id,cutoff+1)
            with self.assertRaisesRegex(ValueError,'immutable'):train(data,cfg,root/'run',root/'ledger.json')
            bank=json.loads((root/'run/reference_bank.json').read_text())
            self.assertFalse({e['event_id'] for e in bank['events']}&set(split['test']+split['tune']+split['calibration']))
            (root/'run/selection.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):predict(data,root/'run',event.event_id,cutoff)


if __name__=='__main__':unittest.main()
