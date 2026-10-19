from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest

from bandori.experiments import run_suite
from bandori.full_refit import issue_times,predict_point,refit
from collect_bandori_data import CN
from fixtures import config,dataset


class FullRefitTests(unittest.TestCase):
    def test_issue_hours_and_explicit_holdouts_with_causal_point_inference(self):
        data=dataset(28);cfg=config(epochs=1,hidden_sizes=(2,))
        events=sorted(data.events.values(),key=lambda e:e.start);held=[e.event_id for e in events[-2:]]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);registry=root/'registry.json'
            with redirect_stdout(io.StringIO()):run_suite(data,cfg,root/'source',registry)
            ledger=registry.read_bytes()
            with self.assertRaisesRegex(ValueError,'authorization'):
                refit(data,root/'source',root/'disallowed',held,(20,21),registry)
            with redirect_stdout(io.StringIO()):
                meta=refit(data,root/'source',root/'full',held,(8,20,21),registry,True)
            self.assertEqual(len(meta['training_event_ids']),26)
            self.assertFalse(set(held)&set(meta['training_event_ids']))
            self.assertEqual(registry.read_bytes(),ledger)
            event=events[-2];cutoff=next(issue_times(event,(20,21)))
            self.assertEqual(datetime.fromtimestamp(cutoff,CN).hour,20)
            before=predict_point(data,root/'full',event.event_id,cutoff)
            self.assertTrue(all(r['status']=='ok' and isinstance(r['final'],int) for r in before['predictions']))
            self.assertTrue(all('lower' not in r for r in before['predictions']))
            for tier in cfg.tiers:
                curve=data.curves[(event.event_id,tier)]
                curve.scores[curve.times>cutoff]*=100
                data.labels[(event.event_id,tier)]*=100
            self.assertEqual(predict_point(data,root/'full',event.event_id,cutoff),before)
            for tier in cfg.tiers:
                curve=data.curves[(event.event_id,tier)]
                delayed=(curve.times>cutoff-2*3600)&(curve.times<=cutoff)
                curve.available[delayed]=cutoff+86400
            self.assertTrue(all(r['status']=='unavailable' for r in predict_point(data,root/'full',event.event_id,cutoff)['predictions']))
            stale=predict_point(data,root/'full',event.event_id,cutoff,max_stale_hours=4)
            self.assertTrue(all(r['status']=='ok' and r['stale_input'] and r['score_age_hours']>=2 for r in stale['predictions']))
            bank=json.loads((root/'full/reference_bank.json').read_text())
            self.assertFalse(set(held)&{e['event_id'] for e in bank['events']})
            with self.assertRaisesRegex(ValueError,'training event'):
                predict_point(data,root/'full',events[-3].event_id,cutoff)
            with self.assertRaisesRegex(ValueError,'hour'):
                predict_point(data,root/'full',event.event_id,cutoff+3600*2)


if __name__=='__main__':unittest.main()
