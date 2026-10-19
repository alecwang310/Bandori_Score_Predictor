from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'old'))

from bandori.dataset import Dataset,Event
from collect_bandori_data import timestamp
from compare_heldout_points import compare


class MatchedPointTests(unittest.TestCase):
    def run_comparison(self,labelled=True):
        tiers=[500,1000,1500,2000];end=timestamp('2026-10-14T23:00:00+08:00')
        data=Dataset([Event(325,end-20*86400,end,'mission_live')],{},
                     {(325,t):100 for t in tiers} if labelled else {})
        benchmark=dict(event_id=325,tier_order=tiers,target_end='2026-10-14T23:00:00+08:00',issues=[
            dict(issued_at='2026-10-05T20:00:00+08:00',hours_remaining=219,MASHIRO=[120]*4,RUI=[95]*4),
            dict(issued_at='2026-10-14T20:00:00+08:00',hours_remaining=3,MASHIRO=[120]*4,RUI=[95]*4,Nanami=[102]*4)])
        point=dict(predictions=[dict(tier=t,status='ok',final=v) for t,v in zip(tiers,[100,110,90,100])])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);model=root/'model';model.mkdir()
            (model/'model.json').write_text(json.dumps(dict(training_event_ids=[],excluded_event_ids=[325,326])))
            (model/'selection.json').write_text(json.dumps(dict(champion='frozen')))
            (model/'config.json').write_text('{}')
            with patch('compare_heldout_points.predict_point',return_value=point) as predictor,redirect_stdout(io.StringIO()):
                result=compare(data,model,benchmark,root/'report',root/'forecasts')
            self.assertEqual([call.args[-1] for call in predictor.call_args_list],
                             [timestamp(issue['issued_at']) for issue in benchmark['issues']])
        return result

    def test_errors_use_identical_timestamps_and_nanami_subset(self):
        result=self.run_comparison()
        self.assertEqual(result['accuracy']['Ours']['points'],8)
        self.assertAlmostEqual(result['accuracy']['Ours']['mape_percent'],5)
        self.assertAlmostEqual(result['accuracy']['MASHIRO']['mape_percent'],20)
        self.assertEqual(result['accuracy_on_nanami_matched_issues']['Ours']['points'],4)
        self.assertAlmostEqual(result['accuracy_on_nanami_matched_issues']['Nanami']['mape_percent'],2)

    def test_missing_outcomes_never_become_accuracy_scores(self):
        result=self.run_comparison(False)
        self.assertEqual(result['accuracy']['Ours']['points'],0)
        self.assertIsNone(result['accuracy']['Ours']['mape_percent'])


if __name__=='__main__':unittest.main()
