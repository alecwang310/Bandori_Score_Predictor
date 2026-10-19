from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'old'))

import wait_and_train as worker
from fixtures import config,dataset


class WorkerTests(unittest.TestCase):
    def test_completed_download_can_run_extended_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();(source/'manifest.json').write_text('{}')
            cfg=root/'config.json';cfg.write_text(json.dumps(config().to_dict()))
            result={'partitions':{'test':{'neural_mean':{'overall':{}},'statistical_mean':{'overall':{}}}}}
            args=['--source',str(source),'--config',str(cfg),'--output',str(root/'suite'),
                  '--snapshot',str(root/'snapshot'),'--status',str(root/'status.json'),'--suite']
            with patch.object(worker,'snapshot_cache'),patch.object(worker.Dataset,'load',return_value=dataset()),patch('bandori.experiments.run_suite',return_value=result) as fit,patch('bandori.suite_reporting.publish'),redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args),0)
            self.assertEqual(fit.call_count,1)
            self.assertTrue(json.loads((root/'suite.plan.json').read_text())['suite'])

    def test_completed_download_trains_both_systems_once(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();(source/'manifest.json').write_text('{}')
            cfg=root/'config.json';cfg.write_text(json.dumps(config().to_dict()))
            output=root/'run';status=root/'status.json'
            result={'partitions':{'test':{'baseline_nn':{'overall':{}},'statistical':{'overall':{}}}}}
            args=['--source',str(source),'--config',str(cfg),'--output',str(output),
                  '--snapshot',str(root/'snapshot'),'--status',str(status)]
            with patch.object(worker,'snapshot_cache'),patch.object(worker.Dataset,'load',return_value=dataset()),patch.object(worker,'train',return_value=result) as fit,patch.object(worker,'write_report'),patch.object(worker,'plot_report'),redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args),0)
            self.assertEqual(fit.call_count,1)
            self.assertEqual(json.loads(status.read_text())['phase'],'complete')
            self.assertEqual(json.loads(output.with_suffix('.plan.json').read_text())['config']['bias_penalty'],3)

    def test_stopped_collector_preserves_lock_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';source.mkdir();lock=source/'.collector.lock';lock.write_text('99999999')
            cfg=root/'config.json';cfg.write_text(json.dumps(config().to_dict()));status=root/'status.json'
            args=['--source',str(source),'--config',str(cfg),'--output',str(root/'run'),
                  '--snapshot',str(root/'snapshot'),'--status',str(status)]
            with patch.object(worker,'alive',return_value=False),redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(args),2)
            self.assertTrue(lock.exists());self.assertEqual(json.loads(status.read_text())['phase'],'collector_stopped')


if __name__=='__main__':unittest.main()
