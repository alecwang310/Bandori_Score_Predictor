import csv
import math
from pathlib import Path
import tempfile
import unittest

from bandori.analog import HORIZONS
from bandori.suite_reporting import interval_diagnostics


class IntervalAuditTests(unittest.TestCase):
    def test_frozen_interval_coverage_groups_and_same_tier_fallback(self):
        bands={'pooled':{'by_tier':{'500':{'log_radius':.3}}},
               'by_horizon':{h:{'by_tier':{}} for h in HORIZONS}}
        bands['by_horizon']['0-24h']['by_tier']['500']={'log_radius':.1}
        rows=[['test',1,500,12,95,100],['test',1,500,6,70,100],
              ['test',1,500,36,80,100],['test',1,1500,12,95,100],
              ['tune',2,500,12,1,100]]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with (root/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['partition','event_id','tier','hours_remaining','actual','model'])
                writer.writerows(rows)
            result=interval_diagnostics(root,{'uncertainty':{'model':bands}})['model']['by_tier']
        self.assertAlmostEqual(result['500']['row_coverage'],2/3)
        self.assertEqual(result['500']['group_coverage'],.5)
        self.assertEqual(result['500']['pooled_fallback_rows'],1)
        self.assertAlmostEqual(result['500']['mean_relative_width'],(4*math.sinh(.1)+2*math.sinh(.3))/3)
        self.assertEqual(result['1500']['unsupported_rows'],1)
        self.assertIsNone(result['1500']['row_coverage'])


if __name__=='__main__':unittest.main()
