from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bandori.dataset import Curve, Dataset, Event
from bandori.distribution_cli import default_issue, issue_time, select_event
from bandori.forecast_format import date_label, scientific
from bandori.forecast_uncertainty import add_uncertainty, bucket, cell, estimate
from bandori.live_data import LiveClient, load_snapshot, save_snapshot
from collect_bandori_data import timestamp


ROOT=Path(__file__).resolve().parents[1]


class SigmaTests(unittest.TestCase):
    def test_each_event_has_one_vote_not_each_tracker_row(self):
        rows=[dict(event_id=1,ridge=100,actual=100,event_end=1000)]
        rows += [dict(event_id=2,ridge=100,actual=200,event_end=1000) for _ in range(20)]
        result=cell(rows,1000)
        self.assertAlmostEqual(result['relative_rmse'],np.sqrt(.5))
        self.assertAlmostEqual(result['relative_bias'],-.5)
        self.assertAlmostEqual(result['centered_relative_sd'],.5)
        self.assertEqual(result['events'],2)

    def test_171_hour_band_fallbacks_and_no_implicit_T1500_borrowing(self):
        self.assertEqual(bucket(171),5);self.assertEqual(bucket(192),6)
        c=dict(relative_rmse=.2,relative_bias=.1,centered_relative_sd=.173,events=8,effective_events=7.5,rows=30)
        book=dict(minimum_direct_events=6,minimum_fallback_events=4,top_rank_fallback={'1':20,'10':20},
                  cells={'clean:20:5':c,'clean:500:all':c})
        a=estimate(book,1,171,10_000)
        self.assertEqual(a['value'],2000);self.assertEqual(a['borrowed_from_tier'],20)
        self.assertFalse(a['horizon_fallback']);self.assertFalse(a['coverage_guaranteed'])
        self.assertTrue(estimate(book,500,171,100)['horizon_fallback'])
        self.assertIsNone(estimate(book,1500,171,100))
        book['cells']['gap_6h:20:5']=dict(c,relative_rmse=.1)
        self.assertEqual(estimate(book,1,171,100,6)['relative_rmse'],.2)
        self.assertTrue(estimate(book,500,171,100,24)['stale_calibration_unavailable'])

    def test_calibration_identity_and_availability_are_enforced(self):
        forecast=dict(checkpoint_cn='2026-10-17T20:00:00+08:00',source_model_sha256='a',predictions=[])
        with self.assertRaisesRegex(ValueError,'different'):
            add_uncertainty(forecast,dict(available_at=0,source_model_sha256='b'))
        with self.assertRaisesRegex(ValueError,'unavailable'):
            add_uncertainty(forecast,dict(available_at=timestamp('2026-10-18T20:00:00+08:00'),source_model_sha256='a'))


class BulletinTests(unittest.TestCase):
    def test_uncertainty_digits_align_with_last_decimal(self):
        self.assertEqual(scientific(3_230_000,590_000),'3.23（59）× 10^6')
        self.assertEqual(scientific(72_000_000,20_000_000),'7.2（20）× 10^7')
        self.assertEqual(scientific(13900,4300),'1.39（43）× 10^4')
        self.assertIn('σ未标定',scientific(1e5))
        self.assertEqual(date_label('2026-10-24T22:59:59.999+08:00'),'2026年10月24日23时')
        self.assertEqual(date_label('2026-10-24T22:59:59+08:00',closing=True),'2026年10月24日23时')

    def test_latest_uses_actual_CN_start_and_default_is_8am(self):
        a=Event(312,timestamp('2025-11-10T13:00:00+08:00'),timestamp('2025-11-18T23:00:00+08:00'),'mission_live')
        b=replace(a,event_id=310,start=timestamp('2026-06-05T13:00:00+08:00'),end=timestamp('2026-06-12T23:00:00+08:00'))
        self.assertEqual(select_event([a,b],'latest').event_id,310)
        now=timestamp('2026-06-06T07:00:00+08:00');self.assertEqual(default_issue(b,now),now)
        now=timestamp('2026-06-06T15:00:00+08:00')
        self.assertEqual(default_issue(b,now),timestamp('2026-06-06T08:00:00+08:00'))
        self.assertEqual(issue_time('2026-06-06 08:00',now),timestamp('2026-06-06T08:00:00+08:00'))
        with self.assertRaisesRegex(ValueError,'未来'):issue_time('2026-06-07 08:00',now)

    def test_snapshot_has_only_known_prefix_and_no_labels(self):
        event=Event(326,100,1000,'challenge');curve=Curve(np.array([110,120,130]),np.array([10,20,30]),np.array([110,150,130]))
        data=Dataset([event],{(326,500):curve},{(326,500):12345})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'event.json.gz';save_snapshot(path,data,326,125)
            raw=json.loads(gzip.decompress(path.read_bytes()));self.assertNotIn('labels',raw)
            restored=load_snapshot(path)
            self.assertEqual(restored.labels,{})
            np.testing.assert_array_equal(restored.curves[(326,500)].scores,[10])


class LiveTests(unittest.TestCase):
    def test_CN_schedule_and_fresher_fallback_without_merging_servers(self):
        now=timestamp('2026-10-18T08:00:00+08:00');start=now-86400;cutoff=now
        index={'326':dict(startAt=[None,None,None,start*1000],endAt=[None,None,None,(start+8*86400)*1000],
                          eventType='mission_live',eventName=['JP',None,None,'国服活动'])}
        class FakeCollector:
            failures=[];used={}
            def get(self,kind,event=None,tier=None,sources=(),refresh=False):
                source=sources[0]
                if kind=='events':return dict(payload=index,source=source)
                score=100 if source=='bestdori' else 200
                t=cutoff-7200 if source=='bestdori' else cutoff-60
                return dict(source=source,payload={'result':True,'cutoffs':[{'time':t*1000,'ep':score}]})
        with patch('bandori.live_data.Collector',return_value=FakeCollector()):
            client=LiveClient('/unused');record=client.discover(now)[0]
            data=client.fetch_event(record,(500,),cutoff)
        self.assertEqual(data.events[326].name,'国服活动')
        self.assertEqual(data.curves[(326,500)].last(cutoff)[1],200)
        self.assertEqual(data.provenance['providers_by_tier']['500'],'hhwx')


@unittest.skipUnless((ROOT/'dist/models/ridge/deployment.json').exists(),'Build the deployment first')
class PortablePackageTests(unittest.TestCase):
    def test_frozen_exact_points_and_no_future_tail_or_own_final_dependency(self):
        from bandori.ridge_runtime import RidgePredictor
        model=RidgePredictor(ROOT/'dist/models/ridge')
        data=load_snapshot(ROOT/'dist/examples/event_326_20261017_2000.json.gz');cutoff=timestamp('2026-10-17T20:00:00+08:00')
        before=model.predict(data,326,cutoff)
        points={r['tier']:r.get('final') for r in before['predictions']}
        # Independent canonical full-suite replay on the newer completed download.
        self.assertEqual({t:points[t] for t in (1,10,500,1000,1500,2000)},
                         {1:64399504,10:43666794,500:3405905,1000:2019802,1500:2019802,2000:1456173})
        for key,c in data.curves.items():
            c.times=np.r_[c.times,cutoff+60];c.scores=np.r_[c.scores,1e12];c.available=np.r_[c.available,cutoff+60]
            data.labels[key]=1e12
        after=model.predict(data,326,cutoff)
        self.assertEqual(before,after)

    def test_runs_outside_repository_with_no_torch_or_other_model_files(self):
        with tempfile.TemporaryDirectory() as td:
            env=dict(os.environ);env.pop('PYTHONPATH',None);env['OPENBLAS_NUM_THREADS']='1'
            code='import sys;sys.path.insert(0,sys.argv[1]);import predict;assert "torch" not in sys.modules;raise SystemExit(predict.main(sys.argv[2:]))'
            result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'dist'),'--event','326',
                '--snapshot',str(ROOT/'dist/examples/event_326_20261017_2000.json.gz'),
                '--as-of','2026-10-17T20:00:00+08:00','--model','ridge','--report','full','--no-save'],
                cwd=td,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('T500：3.41(85)',result.stdout)
            self.assertIn('T100000：无',result.stdout)
            self.assertIn('σ借用T20',result.stdout)


if __name__=='__main__':unittest.main()
