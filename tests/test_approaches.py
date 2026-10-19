from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bandori.advanced_statistics import CachedAnalog,JointRidge,penalized_fit,trajectory_forecasts
from bandori.analog import AnalogPredictor
from bandori.experiments import build_cases,ensemble_predictions,predict_suite,run_suite
from bandori.representations import Compression,matrices
from fixtures import config,dataset


class RepresentationTests(unittest.TestCase):
    def test_later_schedule_extension_does_not_change_known_shape(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        data.revisions=[(event.event_id,event.start-86400,event.start,event.end-86400),
                        (event.event_id,event.start+2*86400,event.start,event.end)]
        cases,_=build_cases(data,cfg,[event.event_id],list(data.events))
        cases=[c for c in cases if c.cutoff==cases[0].cutoff]
        before=trajectory_forecasts(cases,data,cfg,list(data.events))
        data.events[event.event_id]=replace(event,end=event.end+86400)
        after=trajectory_forecasts(cases,data,cfg,list(data.events))
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_rich_features_and_analogs_exclude_future_tail_and_own_final(self):
        data=dataset(12);cfg=config();event=data.eligible_events(cfg)[-1]
        cases,_=build_cases(data,cfg,[event.event_id],list(data.events))
        cases=[c for c in cases if c.cutoff==cases[0].cutoff]
        before=matrices(cases,data,cfg.tiers)[0]
        analogs=trajectory_forecasts(cases,data,cfg,list(data.events))
        for tier in cfg.tiers:
            data.labels[(event.event_id,tier)]*=100
            curve=data.curves[(event.event_id,tier)];curve.scores[curve.times>cases[0].cutoff]*=500
        after=matrices(cases,data,cfg.tiers)[0]
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        for key,value in trajectory_forecasts(cases,data,cfg,list(data.events)).items():np.testing.assert_array_equal(value,analogs[key])

    def test_shared_cache_preserves_late_publication_filter(self):
        data=dataset(2);cfg=config();event=data.eligible_events(cfg)[0];next_event=data.eligible_events(cfg)[1]
        curve=data.curves[(event.event_id,500)];i=12;when=curve.times[i]
        curve.available[i]=next_event.start+10*86400;curve.scores[i]*=2
        cached=CachedAnalog(data,cfg,list(data.events));ordinary=AnalogPredictor(data,cfg,list(data.events))
        for cutoff in (next_event.start,next_event.start+20*86400):
            self.assertEqual(cached.sample(event.event_id,500,when,cutoff),ordinary.sample(event.event_id,500,when,cutoff))

    def test_compression_ignores_rows_with_no_training_weight(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(15,160));weights=np.ones((15,2));weights[-2:]=0
        before=Compression().fit(x,weights,(500,1500),components=4)
        changed=x.copy();changed[-2:]*=10000
        after=Compression().fit(changed,weights,(500,1500),components=4)
        np.testing.assert_allclose(before.mean,after.mean)
        np.testing.assert_allclose(before.projection,after.projection)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'pca.npz';before.save(path)
            np.testing.assert_array_equal(before.transform(x),Compression.load(path).transform(x))


class StatisticalApproachTests(unittest.TestCase):
    def test_two_year_characteristic_decay_and_legacy_half_life(self):
        cfg=config(characteristic_time_days=730)
        self.assertEqual(cfg.recency_weight(0),1)
        self.assertAlmostEqual(cfg.recency_weight(730),math.exp(-1))
        self.assertAlmostEqual(cfg.recency_weight(1460),math.exp(-2))
        self.assertAlmostEqual(config(half_life_days=180).recency_weight(180),.5)
        for value in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):config(characteristic_time_days=value)

    def test_closed_form_solution_minimizes_bias_penalized_objective(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(20,4));y=rng.normal(size=20)
        weights=rng.uniform(.1,1,20);groups=np.arange(20)%3;penalty=np.full(4,.1)
        coefficient=penalized_fit(x,y,weights,groups,penalty,3)
        def loss(beta):
            error=x@beta-y;value=np.sum(weights*error**2)+np.sum(penalty*beta**2)
            for group in range(3):
                mask=groups==group;mass=weights[mask].sum()
                value+=3*mass*(np.dot(weights[mask],error[mask])/mass)**2
            return value
        for i in range(4):
            high=coefficient.copy();low=coefficient.copy();high[i]+=1e-6;low[i]-=1e-6
            self.assertAlmostEqual((loss(high)-loss(low))/2e-6,0,places=6)

    def test_ridge_ignores_missing_heads_and_roundtrips(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(20,8));target=np.tile([100.,30.,140.],(20,2,1))
        weights=np.ones((20,2));weights[:,1]=0;bins=np.arange(20)%5
        model=JointRidge().fit(x,target,weights,bins)
        changed=target.copy();changed[:,1]*=9000
        other=JointRidge().fit(x,changed,weights,bins)
        np.testing.assert_array_equal(model.predict(x),other.predict(x));np.testing.assert_array_equal(model.predict(x)[:,1],0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ridge.npz';model.save(path)
            np.testing.assert_array_equal(model.predict(x),JointRidge.load(path).predict(x))

    def test_collection_averages_have_fixed_equal_weights(self):
        predictions={str(i):np.array([i,2*i]) for i in range(6)}
        members={name:dict(method=method,members=list(predictions)) for name,method in
                 (('mean','mean'),('median','median'),('trimmed','trimmed_mean'))}
        result=ensemble_predictions(predictions,members)
        for name in members:np.testing.assert_array_equal(result[name],[2.5,5])


class SuiteIntegrationTests(unittest.TestCase):
    def test_full_suite_frozen_membership_and_saved_inference(self):
        data=dataset();cfg=config(epochs=2,hidden_sizes=(2,))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with redirect_stdout(io.StringIO()):result=run_suite(data,cfg,root/'run',root/'registry.json')
            self.assertIn(result['champion'],result['partitions']['test'])
            self.assertIn('collection_mean',result['partitions']['test'])
            split=json.loads((root/'run/split.json').read_text());case=build_cases(data,cfg,split['test'][:1],split['train'])[0][0]
            prediction=predict_suite(data,root/'run',case.event_id,case.cutoff,root/'fixed.json')
            self.assertEqual([r['tier'] for r in prediction['predictions']],[500,1500])
            self.assertTrue(all(r['status']=='ok' and len(r['forecasts'])>=12 for r in prediction['predictions']))
            bank=json.loads((root/'run/reference_bank.json').read_text())
            self.assertFalse({e['event_id'] for e in bank['events']}&set(split['tune']+split['calibration']+split['test']))
            with redirect_stdout(io.StringIO()):
                followup=run_suite(data,replace(cfg,characteristic_time_days=730),root/'followup',root/'registry.json',root/'run')
            self.assertEqual(followup['test_status'],'previously_inspected_follow_up')
            self.assertEqual(json.loads((root/'followup/split.json').read_text()),split)
            with self.assertRaisesRegex(ValueError,'previously inspected'):
                run_suite(data,cfg,root/'disallowed_fresh',root/'registry.json')
            ledger=json.loads((root/'registry.json').read_text())
            ledger['protected_event_ids'].append(split['train'][0])
            (root/'registry.json').write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError,'overlap protected holdouts'):
                run_suite(data,cfg,root/'disallowed_replay',root/'registry.json',root/'run')
            (root/'run/models/compression.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):predict_suite(data,root/'run',case.event_id,case.cutoff)


if __name__=='__main__':unittest.main()
