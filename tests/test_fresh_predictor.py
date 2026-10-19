from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from fresh_predictor.anchors import Anchors,FEATURES,build_cases
from fresh_predictor.data import Policy,restrict,fingerprint
from fresh_predictor.regression import JointCorrection,reconcile,training_weights
from fixtures import dataset


class FreshPredictorTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'old'/'dist_fresh/model/model.json').exists(),'Build standalone fresh package first')
    def test_standalone_numpy_package_matches_saved_production_and_needs_no_torch(self):
        import subprocess,sys
        from bandori.live_data import load_snapshot
        from collect_bandori_data import timestamp
        from fresh_predictor.runtime import Predictor
        root=Path(__file__).resolve().parents[1];package=root/'old'/'dist_fresh'
        snapshot=package/'examples/event_326_20261017_2000.json.gz';data=load_snapshot(snapshot)
        cutoff=timestamp('2026-10-17T20:00:00+08:00')
        packaged=Predictor(package/'model').predict(data,326,cutoff)
        import json
        expected=json.loads((package/'examples/event_326_171h.json').read_text())
        self.assertEqual(packaged['predictions'],expected['predictions'])
        source=root/'runs/fresh_post_rule_v1_20261019/production_model'
        if source.exists():self.assertEqual(packaged['predictions'],Predictor(source).predict(data,326,cutoff)['predictions'])
        script='import sys; sys.modules["torch"]=None; sys.path.insert(0,sys.argv[1]); from fresh_predictor.cli import main; raise SystemExit(main(sys.argv[2:]))'
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-B','-c',script,str(package),'--snapshot',str(snapshot),
                '--event','326','--as-of','2026-10-17T20:00:00+08:00','--no-save'],cwd=tmp,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('T500：3.44(73) * 10^6',result.stdout)

    def test_issue_time_is_a_cutoff_not_an_eight_am_restriction(self):
        from bandori.dataset import Event
        from collect_bandori_data import timestamp
        from fresh_predictor.cli import default_issue_time,parse_issue_time
        event=Event(999,timestamp('2026-10-16T15:00:00+08:00'),timestamp('2026-10-24T23:00:00+08:00'),'normal')
        now=timestamp('2026-10-19T07:00:00+08:00')
        self.assertEqual(default_issue_time(event,now),'2026-10-18 08:00:00')
        self.assertEqual(parse_issue_time('2026-10-18 14:30',event,now),timestamp('2026-10-18T14:30:00+08:00'))
        self.assertEqual(parse_issue_time('2026-10-18T06:30:00Z',event,now),timestamp('2026-10-18T14:30:00+08:00'))
        self.assertEqual(parse_issue_time('now',event,now),now)
        self.assertEqual(default_issue_time(event,event.start+3600),'2026-10-16 16:00:00')
        with self.assertRaisesRegex(ValueError,'future'):parse_issue_time('2026-10-20 08:00',event,now)
        with self.assertRaisesRegex(ValueError,'outside'):parse_issue_time('2026-10-15 08:00',event,now)

    def test_physics_notation_aligns_uncertainty_with_the_last_digit(self):
        from fresh_predictor.runtime import physics_score
        self.assertEqual(physics_score(1230000,450000),'1.23(45) * 10^6')
        self.assertEqual(physics_score(3442551,730946),'3.44(73) * 10^6')
        self.assertEqual(physics_score(72000000,20000000),'7.2(20) * 10^7')
        self.assertEqual(physics_score(1230000),'1.23 * 10^6')
        self.assertEqual(physics_score(9999600,10000),'1.0000(10) * 10^7')

    def make(self,n=12):
        d=dataset(n);events=sorted(d.events.values(),key=lambda e:e.start)
        p=replace(Policy(),tiers=(500,1500),rule_at=events[4].start,min_references=3,checkpoint_hours=(8,20,21))
        return d,p,events

    def test_prefix_and_own_final_cannot_change_inputs(self):
        d,p,events=self.make();event=events[-1];t=event.start+55*3600
        history=restrict(d,[e.event_id for e in events[:-1]]);live=restrict(d,[event.event_id],labels=False)
        a=Anchors(history,p).checkpoint(live,event.event_id,t)[0]
        for key,c in d.curves.items():
            if key[0]==event.event_id:c.scores[c.times>t]*=1000;d.labels[key]*=1000
        b=Anchors(history,p).checkpoint(live,event.event_id,t)[0]
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x.features,y.features);self.assertEqual(x.anchor,y.anchor)
            self.assertTrue(all(rid!=event.event_id and history.label_available(rid,p.label_config())<=event.start for rid,rank in x.references))

    def test_data_firewall_and_chronological_fit_ignore_future_event_labels(self):
        d,p,events=self.make();cutoff=events[9].start;ids=[e.event_id for e in events[:9]]
        a=restrict(d,ids);hash_before=fingerprint(a);supervised=[e.event_id for e in events[4:9]]
        cases=build_cases(a,a,supervised,p);model=JointCorrection().fit(cases,a,p,cutoff,.01)
        held=events[9];live=restrict(d,[held.event_id],labels=False);points=Anchors(a,p).checkpoint(live,held.event_id,held.start+48*3600)[0]
        before=model.predict(points)
        for k in list(d.labels):
            if k[0] not in ids:d.labels[k]*=1000
        b=restrict(d,ids);self.assertEqual(hash_before,fingerprint(b))
        again=build_cases(b,b,supervised,p);retrained=JointCorrection().fit(again,b,p,cutoff,.01)
        np.testing.assert_array_equal(model.coefficients,retrained.coefficients)
        np.testing.assert_array_equal(before,retrained.predict(points))

    def test_post_rule_only_weighting_and_model_roundtrip(self):
        d,p,events=self.make();ids=[e.event_id for e in events[4:10]];cases=build_cases(d,d,ids,p)
        w=training_weights(cases,d,p,events[10].start);self.assertAlmostEqual(w.sum(),1.)
        model=JointCorrection().fit(cases,d,p,events[10].start,.1);self.assertEqual(len(model.coefficients),len(FEATURES)+8)
        old=build_cases(d,d,[events[3].event_id],p)
        with self.assertRaisesRegex(ValueError,'Pre-rule'):training_weights(old,d,p,events[10].start)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';model.save(path)
            np.testing.assert_array_equal(model.predict(cases),JointCorrection.load(path).predict(cases))
        values=model.predict(cases)
        self.assertTrue(np.all(values>=[c.current for c in cases]))

    def test_publication_filter_and_no_label_target_view(self):
        d,p,events=self.make();event=events[-1];t=event.start+48*3600;c=d.curves[(event.event_id,500)]
        i=np.searchsorted(c.times,t);c.available[i]=t+3600
        live=restrict(d,[event.event_id],labels=False);self.assertFalse(live.labels)
        history=restrict(d,[e.event_id for e in events[:-1]])
        point=Anchors(history,p).point(live,event.event_id,500,t)
        self.assertLess(point.observed_at,t)

    def test_reward_role_transfer_changes_prior_source_without_creating_labels(self):
        d,p,events=self.make();event=events[4]
        for old in events[:4]:
            d.curves[(old.event_id,1000)]=d.curves[(old.event_id,1500)]
            d.labels[(old.event_id,1000)]=d.labels[(old.event_id,1500)]
        history=restrict(d,[e.event_id for e in events[:4]]);original=dict(history.labels)
        live=restrict(d,[event.event_id],labels=False)
        point=Anchors(history,p).point(live,event.event_id,1500,event.start+48*3600)
        self.assertEqual(point.anchor_method,'historical_completion')
        self.assertTrue(point.references)
        self.assertTrue(all(rank==1000 for eid,rank in point.references))
        self.assertEqual(history.labels,original)


if __name__=='__main__':unittest.main()
