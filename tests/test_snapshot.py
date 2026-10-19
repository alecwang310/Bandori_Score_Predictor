import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from collect_bandori_data import Collector
from bandori.dataset import Dataset, top10_curves
from bandori.snapshot import snapshot_cache
from test_collector import Response


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_frozen_and_does_not_write_to_running_collector(self):
        with tempfile.TemporaryDirectory() as td:
            source, target = Path(td)/'source',Path(td)/'snapshot'
            collector=Collector(source,retries=0)
            event={'eventType':'challenge','startAt':[None]*3+[1735700000000],
                   'endAt':[None]*3+[1736000000000]}
            responses=[{'1':event},{'result':True,'cutoffs':[{'time':1735800000000,'ep':12}]}]
            with patch.object(collector,'throttle',return_value=True),patch('collect_bandori_data.urlopen',side_effect=[Response(r) for r in responses]):
                collector.fetch('bestdori/events','https://example.invalid/events')
                collector.fetch('bestdori/tracker/1/tier_500','https://example.invalid/tracker')
            (source/'.collector.lock').write_text('1234')
            before=(source/'state.json').read_bytes()
            log=(source/'requests.jsonl').read_bytes()
            stats=snapshot_cache(source,target)
            self.assertEqual(stats['pairs_with_history'],1)
            self.assertEqual((source/'state.json').read_bytes(),before)
            self.assertEqual((source/'requests.jsonl').read_bytes(),log)
            self.assertEqual((source/'.collector.lock').read_text(),'1234')
            self.assertFalse((target/'.collector.lock').exists())
            d=Dataset.load(target,())
            self.assertEqual(set(d.curves),{(1,500)})

    def test_incomplete_top_ten_does_not_become_a_false_tenth_cutoff(self):
        points=[{'time':1735800000000,'uid':i,'value':100-i} for i in range(10)]
        points += [{'time':1735803600000,'uid':i,'value':200-i} for i in range(9)]
        curves=top10_curves({'points':points})
        self.assertEqual(set(curves),{1,10})
        self.assertEqual(len(curves[10].times),1)
        self.assertEqual(curves[10].scores[0],91)
        self.assertEqual(curves[1].scores[0],100)

    def test_proxy_top_ten_uses_its_own_cutoff_format(self):
        curves=top10_curves({'result':True,'cutoffs':[{'time':1735800000000,'ep':123}]})
        self.assertEqual(set(curves),{10})
        self.assertEqual(curves[10].scores[0],123)


if __name__=='__main__':
    unittest.main()
