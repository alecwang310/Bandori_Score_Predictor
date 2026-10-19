from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bandori.dataset import Dataset, Event
from bandori.live_data import load_snapshot
from collect_bandori_data import timestamp
from fresh_predictor.dispatch import ForecastRouter, select_predictor
from fresh_predictor.runtime import Predictor, sha


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'old' / 'dist_fresh'


class SelectionTests(unittest.TestCase):
    def test_strict_total_duration_boundary_and_manual_overrides(self):
        event = Event(999, 1000000, 1000000 + 240*3600, 'challenge')
        self.assertEqual(select_predictor(event)['selected'], 'fresh')
        longer = replace(event, end=event.end+1)
        self.assertEqual(select_predictor(longer)['selected'], 'ridge')
        self.assertEqual(select_predictor(longer, '')['selected'], 'ridge')
        self.assertEqual(select_predictor(longer, 'fresh')['selected'], 'fresh')
        self.assertEqual(select_predictor(event, 'ridge')['selected'], 'ridge')
        with self.assertRaises(ValueError):select_predictor(event, 'unknown')

    def test_future_schedule_extension_and_target_labels_do_not_select_a_model(self):
        start = 1000000
        event = Event(999, start, start+300*3600, 'challenge')
        data = Dataset([event], {}, {(999, 500): 1e12}, revisions=[
            (999, start, start, start+200*3600),
            (999, start+50*3600, start, start+300*3600)])
        router = object.__new__(ForecastRouter)
        router._fresh = Mock()
        router._fresh.predict.return_value = dict(predictions=[])
        router._ridge = None
        result = router.predict(data, 999, start+40*3600)
        self.assertEqual(result['selected_predictor'], 'fresh')
        self.assertEqual(result['event_duration_hours'], 200)
        target = router._fresh.predict.call_args.args[0]
        self.assertFalse(target.labels)


@unittest.skipUnless((PACKAGE/'long_event_model/deployment.json').exists(), 'Build combined package first')
class PortableRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = ForecastRouter(PACKAGE/'model', PACKAGE/'long_event_model')
        cls.short = load_snapshot(PACKAGE/'examples/event_326_20261017_2000.json.gz')
        cls.long = load_snapshot(PACKAGE/'examples/event_325_20261014_2000.json.gz')

    def test_long_event_matches_all_eight_saved_rule_aware_ridge_forecasts(self):
        benchmark = json.loads((ROOT/'old'/'benchmarks/event_325_points.json').read_text())
        for issue in benchmark['issues']:
            with self.subTest(hours=issue['hours_remaining']):
                result = self.router.predict(self.long, 325, timestamp(issue['issued_at']), 48)
                self.assertEqual(result['selected_predictor'], 'ridge')
                self.assertGreater(result['event_duration_hours'], 240)
                expected = json.loads((ROOT/"old"/f"predictions/rule_aware_3x_325/event_325_h{issue['hours_remaining']}.json").read_text())
                actual = {r['tier']:r['final'] for r in result['predictions'] if r['status']=='ok'}
                saved = {r['tier']:r['final'] for r in expected['predictions'] if r['status']=='ok'}
                self.assertEqual({t:actual[t] for t in benchmark['tier_order']}, {t:saved[t] for t in benchmark['tier_order']})
                for row in result['predictions']:
                    if row['status']=='ok':self.assertGreaterEqual(row['final'], row['current'])
                self.assertEqual(result['uncertainty']['calibration_event_ids'],
                    json.loads((PACKAGE/'long_event_model/sigma.json').read_text())['calibration_event_ids'])

    def test_short_event_is_unchanged_and_each_override_uses_its_own_sigma(self):
        cutoff = timestamp('2026-10-17T20:00:00+08:00')
        auto = self.router.predict(self.short, 326, cutoff)
        direct = Predictor(PACKAGE/'model').predict(self.short, 326, cutoff)
        self.assertEqual(auto['selected_predictor'], 'fresh')
        self.assertEqual(auto['predictions'], direct['predictions'])
        ridge = self.router.predict(self.short, 326, cutoff, requested='ridge')
        self.assertEqual(ridge['selected_predictor'], 'ridge')
        self.assertTrue(ridge['routing']['manual_override'])
        self.assertIn('uncertainty', ridge)
        self.assertNotEqual(auto['model_sha256'], ridge['model_sha256'])
        long_cutoff = timestamp('2026-10-14T20:00:00+08:00')
        forced = self.router.predict(self.long, 325, long_cutoff, requested='fresh')
        self.assertEqual(forced['predictions'], Predictor(PACKAGE/'model').predict(self.long, 325, long_cutoff)['predictions'])

    def test_ridge_ignores_future_tail_and_own_final(self):
        data = load_snapshot(PACKAGE/'examples/event_325_20261014_2000.json.gz')
        cutoff = timestamp('2026-10-14T20:00:00+08:00')
        before = self.router.predict(data, 325, cutoff)
        for key, curve in data.curves.items():
            curve.times = np.r_[curve.times, cutoff+60]
            curve.scores = np.r_[curve.scores, 1e12]
            curve.available = np.r_[curve.available, cutoff+60]
            data.labels[key] = 1e12
        after = self.router.predict(data, 325, cutoff)
        self.assertEqual(before['routing'], after['routing'])
        self.assertEqual(before['predictions'], after['predictions'])

    def test_cli_selector_enter_means_auto_and_ridge_override_is_available(self):
        from fresh_predictor.cli import main
        args = ['--snapshot', str(PACKAGE/'examples/event_325_20261014_2000.json.gz'), '--no-save']
        for choice, expected in [('', 'ridge'), ('fresh', 'fresh'), ('ridge', 'ridge')]:
            with self.subTest(choice=choice), patch('builtins.input', side_effect=['325','2026-10-14 20:00',choice]) as prompt, \
                 patch('sys.stdout', new_callable=io.StringIO) as output, patch('sys.stderr', new_callable=io.StringIO):
                self.assertEqual(main(args, root=PACKAGE), 0)
                self.assertEqual(prompt.call_count, 3)
                self.assertIn('模型 [回车=auto', prompt.call_args.args[0])
                self.assertIn(('手动指定 ' if choice else '自动切换 ')+expected, output.getvalue())

    def test_package_runs_outside_repository_without_torch_on_both_routes(self):
        script = 'import sys;sys.modules["torch"]=None;sys.path.insert(0,sys.argv[1]);from fresh_predictor.cli import main;raise SystemExit(main(sys.argv[2:]))'
        with tempfile.TemporaryDirectory() as tmp:
            for event, when, selected in [(325,'2026-10-14 20:00','ridge'), (326,'2026-10-17 20:00','fresh')]:
                snapshot = PACKAGE/f'examples/event_{event}_{"20261014" if event==325 else "20261017"}_2000.json.gz'
                prefix = Path(tmp)/str(event)
                result = subprocess.run([sys.executable,'-B','-c',script,str(PACKAGE),'--snapshot',str(snapshot),
                    '--event',str(event),'--as-of',when,'--output-prefix',str(prefix)], cwd=tmp, capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                saved = json.loads(prefix.with_suffix('.json').read_text())
                self.assertEqual(saved['selected_predictor'], selected)
                self.assertIn(' * 10^', result.stdout)


if __name__ == '__main__':unittest.main()
