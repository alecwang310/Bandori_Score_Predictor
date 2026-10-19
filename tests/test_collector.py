"""Offline tests; responses are fixtures, never reported as real event data."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError

import collect_bandori_data as c


def ts(s):
    return c.timestamp(s)


def event(event_id=20, start='2025-01-01T15:00:00+08:00', end='2025-01-04T22:59:00+08:00'):
    return {'event_id': event_id, 'cn_event_number': 1, 'start': ts(start), 'end': ts(end),
            'source': 'bestdori', 'meta': {'eventType': 'challenge', 'eventName': [None, None, None, 'fixture']}}


def point(s, score, available=None):
    return {'time': ts(s), 'score': score, 'available_at': ts(available) if available else None}


class Response:
    headers = {}

    def __init__(self, value):
        self.raw = json.dumps(value).encode()

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class CheckpointTests(unittest.TestCase):
    def test_after_8am_observation_never_used(self):
        e = event()
        points = [point('2025-01-02T07:55:00+08:00', 10), point('2025-01-02T08:01:00+08:00', 99999)]
        row = list(c.checkpoint_rows(e, 500, points, ts('2025-01-02T08:00:00+08:00'), 60))[0]
        self.assertEqual(row['score'], 10)
        self.assertEqual(row['staleness_minutes'], 5)
        self.assertEqual(row['elapsed_hours'], 17)

    def test_exact_8am_allowed(self):
        e = event()
        p = point('2025-01-02T08:00:00+08:00', 25)
        row = list(c.checkpoint_rows(e, 500, [p], p['time'], 60))[0]
        self.assertEqual(row['score'], 25)

    def test_late_publication_filtered(self):
        e = event()
        points = [point('2025-01-02T07:00:00+08:00', 10),
                  point('2025-01-02T07:50:00+08:00', 20, '2025-01-02T08:05:00+08:00')]
        row = list(c.checkpoint_rows(e, 500, points, ts('2025-01-02T08:00:00+08:00'), 60))[0]
        self.assertEqual(row['score'], 10)

    def test_missing_and_stale_are_explicit(self):
        e = event()
        cutoff = ts('2025-01-02T08:00:00+08:00')
        missing = list(c.checkpoint_rows(e, 500, [], cutoff, 60))[0]
        self.assertEqual(missing['status'], 'missing')
        self.assertEqual(missing['score'], '')
        points = [point('2025-01-01T20:00:00+08:00', 2)]
        stale = list(c.checkpoint_rows(e, 500, points, cutoff, 60))[0]
        self.assertEqual(stale['status'], 'stale')
        self.assertEqual(stale['score'], 2)

    def test_no_checkpoints_before_start_or_at_end(self):
        start, end = ts('2025-01-01T08:00:00+08:00'), ts('2025-01-03T08:00:00+08:00')
        self.assertEqual(list(c.checkpoints(start, end, end)), [ts('2025-01-02T08:00:00+08:00')])

    def test_duplicate_correction_is_preserved(self):
        payload = {'cutoffs': [{'time': 1735775700000, 'ep': 100}, {'time': 1735775700000, 'ep': 99}]}
        points, bad = c.parse_points(payload)
        self.assertEqual(bad, 0)
        self.assertEqual([p['score'] for p in points], [100, 99])


class SchemaTests(unittest.TestCase):
    def test_download_order_uses_newest_cn_dates_and_priority_tiers(self):
        older=event(900);newer=event(7,'2025-02-01T13:00:00+08:00','2025-02-08T23:00:00+08:00')
        jobs=c.download_jobs([older,newer],[100,500,1000,1500,2000],ts('2026-01-01T00:00:00Z'))
        self.assertEqual(jobs[:4],[('tracker',7,t,False) for t in (500,1000,1500,2000)])
        self.assertEqual(list(dict.fromkeys(j[1] for j in jobs)),[7,900])
        self.assertEqual(list(dict.fromkeys(j[1] for j in c.download_jobs([older,newer],[500],2e9,'oldest'))),[900,7])

    def test_cn_chronology_not_global_id(self):
        mapping = {}
        for eid, date in [(900, '2025-01-01T15:00:00+08:00'), (7, '2025-01-02T15:00:00+08:00')]:
            start = ts(date)
            mapping[str(eid)] = {'startAt': [None]*3+[str(int(start*1000))],
                                 'endAt': [None]*3+[str(int((start+86400)*1000))]}
        mapping['1234'] = {'startAt': [1735700000000, None, None, None], 'endAt': [1735800000000, None, None, None]}
        events, _ = c.select_events([('bestdori', mapping)], ts('2026-01-01T00:00:00Z'))
        self.assertEqual([e['event_id'] for e in events], [900, 7])
        self.assertEqual([e['cn_event_number'] for e in events], [1, 2])

    def test_hhwx_both_event_schemas(self):
        old = {'success': True, 'data': {'events': [{'eventId': 5, 'eventType': 'challenge',
               'timeline': {'cn': {'startAt': 1735700000000, 'endAt': 1735800000000}}}]}}
        new = {'success': True, 'data': {'5': {'startAt': [None]*3+[1735700000000]}}}
        self.assertEqual(c.normalize_events(old, 'hhwx')['5']['startAt'][3], 1735700000000)
        self.assertEqual(c.normalize_events(new, 'hhwx')['5']['startAt'][3], 1735700000000)

    def test_metadata_disagreement_is_reported(self):
        a = {'5': {'startAt': [None]*3+[1735700000000], 'endAt': [None]*3+[1735800000000]}}
        b = {'5': {'startAt': [None]*3+[1735700000000], 'endAt': [None]*3+[1735900000000]}}
        events, conflicts = c.select_events([('bestdori', a), ('hhwx', b)], 2_000_000_000)
        self.assertEqual(events[0]['end'], 1735800000)
        self.assertEqual(len(conflicts), 1)

    def test_invalid_data_rejected_without_smoothing(self):
        p, bad = c.parse_points({'cutoffs': [{'time': 1735700000000, 'ep': 7},
            {'time': 1735700001000, 'ep': 6}, {'time': None, 'ep': 4},
            {'time': 1735700002000, 'ep': -1}, {'time': 1735700002000, 'ep': 3.1},
            {'time': 1735700002000, 'ep': True}]})
        self.assertEqual([x['score'] for x in p], [7, 6])
        self.assertEqual(bad, 4)

    def test_archive_is_exact_tier_and_server(self):
        archive = {'20': {'cutoff': [{'500': 100}, None, None, {'500': 345}]}}
        self.assertEqual(c.archive_score(archive, 20, 500), 345)
        self.assertIsNone(c.archive_score(archive, 20, 1500))

    def test_time_units_and_timezone(self):
        self.assertEqual(c.timestamp('1735700000000'), 1735700000)
        self.assertEqual(c.timestamp(1735700000), 1735700000)
        self.assertIsNone(c.timestamp('2025-01-01T08:00:00'))
        self.assertIsNone(c.timestamp(float('nan')))
        self.assertIsNone(c.timestamp(True))


class CacheTests(unittest.TestCase):
    def test_confirmed_empty_is_cached_across_restarts_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as d:
            first=c.Collector(d,retries=0)
            with patch.object(first,'throttle',return_value=True),patch.object(c,'urlopen',return_value=Response({'result':True,'cutoffs':[]})):
                self.assertIsNone(first.get('tracker',20,1500,sources=('bestdori',)))
            resumed=c.Collector(d,retries=0)
            with patch.object(c,'urlopen',side_effect=AssertionError('Cached empty hit the network')):
                self.assertIsNone(resumed.get('tracker',20,1500,sources=('bestdori',)))
            self.assertEqual(resumed.missing_cache_hits,1)
            retry=c.Collector(d,retries=0,retry_missing=True)
            with patch.object(retry,'throttle',return_value=True),patch.object(c,'urlopen',return_value=Response({'result':True,'cutoffs':[{'time':1735700000000,'ep':10}]})) as network:
                self.assertIsNotNone(retry.get('tracker',20,1500,sources=('bestdori',)))
            self.assertEqual(network.call_count,1)
            self.assertNotIn('bestdori/tracker/20/tier_1500',retry.missing)

    def test_legacy_empty_log_is_migrated_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            first=c.Collector(d,retries=0)
            with patch.object(first,'throttle',return_value=True),patch.object(c,'urlopen',return_value=Response({'result':True,'cutoffs':[]})):
                first.get('tracker',20,1500,sources=('bestdori',))
            (Path(d)/'missing_cache.json').unlink()
            migrated=c.Collector(d,retries=0)
            with patch.object(c,'urlopen',side_effect=AssertionError('Legacy empty retried')):
                self.assertIsNone(migrated.get('tracker',20,1500,sources=('bestdori',)))
            self.assertEqual(migrated.missing_cache_hits,1)

    def test_fallback_cache_skips_primary_even_after_miss_ttl(self):
        with tempfile.TemporaryDirectory() as d:
            first=c.Collector(d,retries=0)
            with patch.object(first,'throttle',return_value=True),patch.object(c,'urlopen',side_effect=[Response({'result':True,'cutoffs':[]}),Response({'result':True,'cutoffs':[{'time':1735700000000,'ep':10}]})]):
                first.get('tracker',20,1500)
            resumed=c.Collector(d,retries=0)
            with patch.object(c.time,'time',return_value=c.time.time()+8*86400),patch.object(c,'urlopen',side_effect=AssertionError('Cached fallback should avoid primary request')):
                self.assertEqual(resumed.get('tracker',20,1500)['source'],'hhwx')

    def test_transient_errors_are_not_negative_cached(self):
        with tempfile.TemporaryDirectory() as d:
            collector=c.Collector(d,retries=0)
            with patch.object(collector,'throttle',return_value=True),patch.object(c,'urlopen',side_effect=URLError('temporary')) as network:
                collector.get('tracker',20,1500,sources=('bestdori',))
                collector.get('tracker',20,1500,sources=('bestdori',))
            self.assertEqual(network.call_count,2);self.assertFalse(collector.missing)

    def test_ongoing_refresh_bypasses_empty_cache(self):
        with tempfile.TemporaryDirectory() as d:
            collector=c.Collector(d,retries=0)
            collector.remember_missing('bestdori/tracker/20/tier_1500','confirmed_empty')
            with patch.object(collector,'throttle',return_value=True),patch.object(c,'urlopen',return_value=Response({'result':True,'cutoffs':[{'time':1735700000000,'ep':11}]})) as network:
                self.assertIsNotNone(collector.get('tracker',20,1500,sources=('bestdori',),refresh=True))
            self.assertEqual(network.call_count,1)

    def test_resume_checksum_and_raw_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            collector = c.Collector(d, retries=0)
            payload = {'cutoffs': [{'time': 1735700000000, 'ep': 10}], 'result': True}
            with patch.object(c, 'urlopen', return_value=Response(payload)) as network:
                self.assertEqual(collector.fetch('bestdori/test', 'https://example.invalid/data'), payload)
                self.assertEqual(network.call_count, 1)
            with patch.object(c, 'urlopen', side_effect=AssertionError('network called on resume')):
                resumed = c.Collector(d, retries=0)
                self.assertEqual(resumed.fetch('bestdori/test', 'https://example.invalid/data'), payload)
            entry = resumed.state['bestdori/test']
            path = Path(d)/entry['path']
            self.assertEqual(hashlib.sha256(c.gzip.decompress(path.read_bytes())).hexdigest(), entry['sha256'])
            self.assertIn('retrieved_at_utc', entry)
            path.write_bytes(b'broken')
            self.assertIsNone(resumed.read_cache('bestdori/test'))

    def test_fallback_on_missing_tier_keeps_both_raw_responses(self):
        with tempfile.TemporaryDirectory() as d:
            collector = c.Collector(d, retries=0)
            with patch.object(collector, 'throttle', return_value=True), patch.object(c, 'urlopen',
                  side_effect=[Response({'result': True, 'cutoffs': []}),
                               Response({'result': True, 'cutoffs': [{'time': 1735700000000, 'ep': 90}]})]):
                result = collector.get('tracker', 20, 1500)
            self.assertEqual(result['source'], 'hhwx')
            self.assertEqual(len(list((Path(d)/'raw').rglob('*.gz'))), 2)
            self.assertEqual(len(collector.failures), 1)

    def test_offline_does_not_touch_network(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(c, 'urlopen', side_effect=AssertionError('offline made network request')):
                collector = c.Collector(d, offline=True)
                self.assertIsNone(collector.fetch('missing', 'https://example.invalid'))

    def test_404_does_not_retry(self):
        with tempfile.TemporaryDirectory() as d:
            collector = c.Collector(d, retries=3)
            error = HTTPError('https://example.invalid', 404, 'missing', {}, None)
            with patch.object(c, 'urlopen', side_effect=error) as network:
                self.assertIsNone(collector.fetch('missing', 'https://example.invalid'))
                self.assertEqual(network.call_count, 1)

    def test_stopped_worker_does_not_request(self):
        with tempfile.TemporaryDirectory() as d:
            collector = c.Collector(d)
            collector.stop.set()
            with patch.object(c, 'urlopen', side_effect=AssertionError('stopped worker requested data')):
                self.assertIsNone(collector.fetch('missing', 'https://example.invalid'))


class ExportTests(unittest.TestCase):
    def test_no_unverified_tail_as_final_and_no_final_in_features(self):
        with tempfile.TemporaryDirectory() as d:
            e = event()
            selected = {(20, 500): {'source': 'bestdori', 'payload': {'cutoffs': [
                {'time': int(ts('2025-01-02T07:55:00+08:00')*1000), 'ep': 100},
                {'time': int(ts('2025-01-04T22:59:00+08:00')*1000), 'ep': 99999}]}}}
            stats = c.summarize(Path(d), [e], [500, 1500], selected, {}, ts('2025-01-06T00:00:00Z'), 60)
            with (Path(d)/'tables/final_labels.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(list(csv.DictReader(f)), [])
            with (Path(d)/'tables/checkpoints_0800_cn.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertNotIn('final_ep', rows[0])
            self.assertEqual(rows[0]['score'], '100')
            self.assertNotIn('99999', [r['score'] for r in rows])
            self.assertEqual(stats['missing_pairs'], 1)

    def test_full_run_bundle_and_offline_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'fixture_bundle'
            e = event()
            index = {'20': {'startAt': [None]*3+[e['start']*1000], 'endAt': [None]*3+[e['end']*1000],
                            'eventType': 'challenge', 'eventName': [None]*3+['fixture']}}

            def respond(request, **kwargs):
                u = request.full_url
                if u.endswith('/events/all.6.json'):
                    return Response(index)
                if u.endswith('/archives/all.5.json'):
                    return Response({'20': {'cutoff': [None]*3+[{'500': 200}]}})
                if u.endswith('/events/20.json'):
                    return Response({'eventType': 'challenge', 'rankingRewards': []})
                if '/tracker/data?' in u:
                    return Response({'result': True, 'cutoffs': [
                        {'time': int(ts('2025-01-02T07:55:00+08:00')*1000), 'ep': 100},
                        {'time': int(e['end']*1000), 'ep': 200}]})
                raise AssertionError('Unexpected URL ' + u)

            argv = ['--output', str(root), '--source', 'bestdori', '--tiers', '500',
                    '--skip-top10', '--skip-supporting-metadata', '--retries', '0']
            with patch.object(c.Collector, 'throttle', return_value=True), patch.object(c, 'urlopen', side_effect=respond), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(argv), 0)
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(len(manifest['resources']), 4)
            archive = next(Path(d).glob('*.zip'))
            with c.zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('fixture_bundle/tables/final_labels.csv', z.namelist())
            self.assertFalse((root/'.collector.lock').exists())
            with patch.object(c, 'urlopen', side_effect=AssertionError('offline used network')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(argv + ['--offline', '--no-zip']), 0)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['cache_hits'], 4)
            self.assertEqual(summary['archived_final_labels'], 1)

    def test_templates_never_overwrite_user_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c.make_templates(root)
            p = root/'annotations/outages.csv'
            p.write_text('user supplied evidence', encoding='utf-8')
            c.make_templates(root)
            self.assertEqual(p.read_text(), 'user supplied evidence')


if __name__ == '__main__':
    unittest.main()
