"""Uncertainty math and the actual self-contained production CLI contract."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bandori.stack_uncertainty import add_uncertainty, bucket, build_cells, cell, estimate


ROOT = Path(__file__).resolve().parents[1]


class SigmaTests(unittest.TestCase):
    def row(self, event, actual, hours=171, tier=500, end=0):
        return dict(event_id=event, tier=tier, prediction=100., actual=actual,
                    hours_remaining=hours, event_end=end)

    def test_cluster_weighting_and_bias_are_not_removed(self):
        rows = [self.row(1, 100)] + [self.row(2, 200)] * 20
        c = cell(rows, 0)
        self.assertAlmostEqual(c['relative_rmse'], math.sqrt(.5))
        self.assertAlmostEqual(c['relative_bias'], -.5)
        self.assertAlmostEqual(c['centered_relative_sd'], .5)
        # Replicating all observations of one event cannot give it extra weight.
        self.assertEqual(c['relative_rmse'], cell(rows + [self.row(2, 200)] * 20, 0)['relative_rmse'])
        rows = [self.row(1, 100, end=730*86400), self.row(2, 200, end=0)]
        self.assertAlmostEqual(cell(rows, 730*86400)['relative_rmse']**2, 1/(1+math.e))

    def test_sparse_horizons_pool_locally_without_borrowing_tiers(self):
        rows = [self.row(1, 110, 3)] + [self.row(e, 120, 27) for e in (2,3,4)]
        cells = build_cells(rows, (500,1500), 0)
        c = cells['500:0']
        self.assertEqual(c['source_buckets'], [0,1])
        self.assertEqual(c['events'], 4)
        self.assertTrue(c['horizon_fallback'])
        self.assertFalse(any(k.startswith('1500:') for k in cells))
        self.assertEqual(bucket(171), 5)
        self.assertEqual(bucket(192), 6)
        with self.assertRaises(ValueError):
            bucket(float('nan'))

    def test_no_stale_sigma_and_identity_or_time_mismatch_fails(self):
        rows = [self.row(e, 120) for e in (1,2,3,4)]
        cal = dict(cells=build_cells(rows, (500,), 0), max_score_age_hours=1,
                   event_duration_range_hours=[190,240], available_at=100,
                   source_model_sha256='a', formula='rms',calibration_event_ids=[1,2,3,4],limitations=[])
        a = estimate(cal, 500, 171, 1000, .5, 202)
        self.assertAlmostEqual(a['value'], 200)
        self.assertFalse(a['coverage_guaranteed'])
        self.assertIsNone(estimate(cal,500,171,1000,2,202))
        self.assertIsNone(estimate(cal,1500,171,1000,.5,202))
        self.assertTrue(estimate(cal,500,171,1000,.5,373)['outside_duration_support'])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            add_uncertainty(dict(checkpoint_cn='1970-01-01T00:00:01+00:00'), cal)
        with self.assertRaisesRegex(ValueError, 'different'):
            add_uncertainty(dict(checkpoint_cn='2026-10-17T20:00:00+08:00',source_model_sha256='b'), cal)


class ProductionTests(unittest.TestCase):
    def run_cli(self, tempdir, eid=326, model=None, report=None, interactive=False):
        when = '2026-10-17T20:00:00+08:00' if eid == 326 else '2026-10-14T20:00:00+08:00'
        snapshot = 'event_326_20261017_2000.json.gz' if eid == 326 else 'event_325_20261014_2000.json.gz'
        args = ['--snapshot',str(ROOT/'dist/examples'/snapshot),'--as-of',when,
                '--output-prefix',str(Path(tempdir)/'forecast')]
        if not interactive:
            args += ['--event',str(eid)]
        if model:
            args += ['--model', model]
        if report:
            args += ['--report', report, '--max-stale-hours', '1']
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1')
        env.pop('PYTHONPATH',None)
        # Real distribution imports in a fresh interpreter, outside the repo.
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import predict;'
                'rc=predict.main(sys.argv[2:]);assert "torch" not in sys.modules;raise SystemExit(rc)')
        proc = subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'dist'),*args],
            cwd=tempdir, env=env, input=(str(eid)+'\n\n') if interactive else '',
            text=True,capture_output=True,timeout=30)
        self.assertEqual(proc.returncode,0,proc.stderr)
        return json.loads((Path(tempdir)/'forecast.json').read_text()), proc.stdout

    def test_scripted_and_enter_defaults_and_original_finals(self):
        for interactive in (False,True):
            with self.subTest(interactive=interactive), tempfile.TemporaryDirectory() as tmp:
                doc, text = self.run_cli(tmp,interactive=interactive)
                self.assertEqual(doc['model']['name'],'stack')
                self.assertEqual(doc['model']['selection'],'default')
                rows = {r['tier']:r for r in doc['predictions']}
                self.assertEqual(len(rows),24)
                for tier, value in {500:3966815,1000:1945733,1500:1945733,2000:1580956}.items():
                    self.assertEqual(rows[tier]['final'],value)
                    self.assertEqual(rows[tier]['sigma']['method'],'stack_event_loo_rms')
                    self.assertEqual(rows[tier]['sigma']['source_tier'],tier)
                    self.assertGreaterEqual(rows[tier]['sigma']['events'],4)
                self.assertIsNone(rows[1]['sigma']['value'])
                tier_lines = [s.split('：')[0] for s in text.splitlines() if s.startswith('T')]
                self.assertEqual(tier_lines,['T100','T500','T1000','T1500','T2000'])
                self.assertIn('T500：3.97(88) * 10^6',text)

    def test_default_stack_for_long_event_and_explicit_auto_still_routes(self):
        for eid, model, selected in ((325,None,'stack'),(325,'auto','ridge'),
                                     (326,'auto','fresh'),(326,'fresh','fresh')):
            with self.subTest(eid=eid,model=model), tempfile.TemporaryDirectory() as tmp:
                doc,text = self.run_cli(tmp,eid,model)
                self.assertEqual(doc['model']['name'],selected)

    def test_full_report_and_independent_t1500_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc, text = self.run_cli(tmp,report='full')
            self.assertEqual(sum(s.startswith('T') for s in text.splitlines()),24)
        cal = json.loads((ROOT/'dist/models/stack/sigma.json').read_text())
        self.assertEqual(cal['calibration_event_ids'],[311,314,315,316,317,318,319,320])
        self.assertEqual(cal['excluded_event_ids'],[321,322,323,324,325,326])
        a = estimate(cal,1000,171,1000,.1,202)
        cal['cells']['1500:5']['relative_rmse'] *= 2
        b = estimate(cal,1000,171,1000,.1,202)
        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()
