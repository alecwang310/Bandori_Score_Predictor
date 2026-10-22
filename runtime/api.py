"""Public CN API parsing, requests, caching and provider fallback."""

import gzip

import hashlib

import json

import math

import os

from pathlib import Path

import random

import socket

import threading

import time

from datetime import datetime, timedelta, timezone

from urllib.error import HTTPError, URLError

from urllib.request import Request, urlopen

UTC = timezone.utc

CN = timezone(timedelta(hours=8))

SERVER = 3

VERSION = '1.1.0'

SOURCES = {'bestdori': {'events': 'https://bestdori.com/api/events/all.6.json', 'detail': 'https://bestdori.com/api/events/{event}.json', 'tracker': 'https://bestdori.com/api/tracker/data?server=3&event={event}&tier={tier}', 'top10': 'https://bestdori.com/api/eventtop/data?server=3&event={event}&mid=0&interval=3600000'}, 'hhwx': {'events': 'https://hhwx.org/api/bandori/master/events', 'detail': 'https://hhwx.org/api/bandori/master/events/{event}', 'tracker': 'https://hhwx.org/api/bandori/tracker/data?server=3&event={event}&type=event&tier={tier}', 'top10': 'https://hhwx.org/api/bandori/tracker/data?server=3&event={event}&type=event&tier=10'}}

def now_iso():
    return datetime.now(UTC).isoformat(timespec='seconds')

def timestamp(value):
    """Accept API milliseconds, Unix seconds, or timezone-qualified ISO dates."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
        if not math.isfinite(x) or x <= 0:
            return None
        return x / 1000 if x > 100000000000 else x
    except (ValueError, TypeError, OverflowError):
        try:
            d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            return d.timestamp() if d.tzinfo is not None else None
        except (ValueError, TypeError, OverflowError):
            return None

def iso(value, tz=UTC):
    return datetime.fromtimestamp(value, tz).isoformat(timespec='seconds') if value is not None else ''

def slot(value, index=SERVER):
    if isinstance(value, list):
        return value[index] if len(value) > index else None
    if isinstance(value, dict):
        return value.get(str(index), value.get('cn'))
    return None

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
    os.replace(tmp, path)

def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))

def normalize_events(payload, source):
    if source == 'hhwx':
        if payload.get('success') is not True or not isinstance(payload.get('data'), dict):
            raise ValueError('HHWX event index has an unexpected schema')
        payload = payload['data']
        if isinstance(payload.get('events'), list):
            result = {}
            for e in payload['events']:
                if not isinstance(e, dict) or e.get('eventId') is None:
                    continue
                v = dict(e)
                cn = (e.get('timeline') or {}).get('cn') or {}
                v['startAt'] = [None, None, None, cn.get('startAt')]
                v['endAt'] = [None, None, None, cn.get('endAt')]
                result[str(e['eventId'])] = v
            payload = result
    if not isinstance(payload, dict):
        raise ValueError('Event index must be an object')
    result = {str(k): v for k, v in payload.items() if str(k).isdigit() and isinstance(v, dict)}
    if not result:
        raise ValueError('Event index has no numeric event IDs')
    return result

def select_events(indexes, as_of):
    """CN order is determined by timestamps, never by the global event ID."""
    merged = {}
    conflicts = []
    for source, mapping in indexes:
        for event_id, meta in mapping.items():
            start = timestamp(slot(meta.get('startAt')))
            end = timestamp(slot(meta.get('endAt')))
            if start is None or end is None or end <= start:
                continue
            if event_id in merged:
                old = merged[event_id]
                if old['start'] != start or old['end'] != end:
                    conflicts.append({'event_id': int(event_id), 'preferred_source': old['source'], 'other_source': source, 'preferred_start': iso(old['start']), 'other_start': iso(start), 'preferred_end': iso(old['end']), 'other_end': iso(end)})
                continue
            if start <= as_of:
                merged[event_id] = {'event_id': int(event_id), 'start': start, 'end': end, 'source': source, 'meta': meta}
    events = sorted(merged.values(), key=lambda x: (x['start'], x['event_id']))
    for ordinal, e in enumerate(events, 1):
        e['cn_event_number'] = ordinal
    return (events, conflicts)

def parse_points(payload):
    """Keep valid observations and corrections; never interpolate or smooth."""
    points, invalid = ([], 0)
    for record in payload.get('cutoffs', []):
        if not isinstance(record, dict):
            invalid += 1
            continue
        t = timestamp(record.get('time'))
        try:
            raw_score = record['ep']
            score = float(raw_score)
            if isinstance(raw_score, bool) or not math.isfinite(score) or score < 0 or (not score.is_integer()) or (t is None):
                raise ValueError('Invalid observation')
        except (KeyError, ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        available = timestamp(record.get('availableAt', record.get('available_at')))
        points.append({'time': t, 'score': int(score), 'available_at': available})
    points.sort(key=lambda x: x['time'])
    return (points, invalid)

def archive_score(archives, event_id, tier):
    e = archives.get(str(event_id), {}) if isinstance(archives, dict) else {}
    cn = slot(e.get('cutoff')) if isinstance(e, dict) else None
    if not isinstance(cn, dict):
        return None
    raw = cn.get(str(tier))
    try:
        x = float(raw)
        if isinstance(raw, bool) or not math.isfinite(x) or x <= 0 or (not x.is_integer()):
            return None
        return int(x)
    except (ValueError, TypeError, OverflowError):
        return None

def confirmed_empty(key, payload):
    if not isinstance(payload, dict):
        return False
    if '/tracker/' in key:
        return payload.get('result') is True and payload.get('cutoffs') == []
    if '/top10/' in key:
        return payload.get('result') is True and payload.get('cutoffs') == [] or payload.get('points') == []
    return False

def resource_location(kind, event, tier, source):
    suffix = kind if event is None else f'{kind}/{event}'
    if tier is not None:
        suffix += f'/tier_{tier}'
    return (source + '/' + suffix, SOURCES[source][kind].format(event=event, tier=tier))

def valid_resource(kind, p, source):
    if not isinstance(p, dict):
        return False
    if kind == 'events':
        try:
            normalize_events(p, source)
            return True
        except (ValueError, TypeError, AttributeError):
            return False
    if kind == 'detail':
        body = p.get('data') if source == 'hhwx' and p.get('success') is True else p
        return isinstance(body, dict) and isinstance(body.get('eventType'), str) and bool(body['eventType'])
    if kind == 'tracker':
        return p.get('result') is True and isinstance(p.get('cutoffs'), list) and any((parse_points({'cutoffs': [r]})[0] for r in p['cutoffs']))
    if kind == 'top10':
        return bool(p.get('points')) or (p.get('result') is True and bool(p.get('cutoffs')))
    return True

class Collector:

    def __init__(self, root, rate=2.0, timeout=30.0, retries=3, offline=False, refresh=False, retry_missing=False, missing_ttl_hours=168):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'state.json'
        self.state = json.loads(self.state_path.read_text('utf-8')) if self.state_path.exists() else {}
        self.rate = rate
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        self.refresh = refresh
        self.retry_missing = retry_missing
        self.missing_ttl_seconds = missing_ttl_hours * 3600
        self.missing_path = self.root / 'missing_cache.json'
        self.missing = json.loads(self.missing_path.read_text()).get('records', {}) if self.missing_path.exists() else {}
        self.lock = threading.RLock()
        self.rate_lock = threading.Lock()
        self.next_request = 0.0
        self.failures = []
        self.used = {}
        self.network_successes = 0
        self.cache_hits = 0
        self.missing_cache_hits = 0
        self.stop = threading.Event()
        if not self.missing_path.exists() and (not offline):
            self.seed_missing_cache()

    def seed_missing_cache(self):
        """Recover definitive misses from earlier collector versions' audit log."""
        log_path = self.root / 'requests.jsonl'
        if not log_path.exists():
            return
        payload_kinds = {}
        with log_path.open(encoding='utf-8') as rows:
            for line in rows:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                key = r.get('key')
                at = timestamp(r.get('retrieved_at_utc', r.get('at_utc')))
                if not key or at is None or time.time() - at > self.missing_ttl_seconds:
                    continue
                if r.get('status') == 'ok':
                    self.missing.pop(key, None)
                    continue
                reason = None
                if r.get('status') == 'failed' and r.get('error') in ('HTTP 404', 'HTTP 410'):
                    reason = r['error']
                if r.get('status') == 'invalid_payload' and ('/tracker/' in key or '/top10/' in key):
                    cache_key = (r.get('sha256'), '/tracker/' in key)
                    if cache_key not in payload_kinds:
                        try:
                            raw = gzip.decompress((self.root / r['path']).read_bytes())
                            payload_kinds[cache_key] = hashlib.sha256(raw).hexdigest() == r['sha256'] and confirmed_empty(key, json.loads(raw))
                        except (OSError, ValueError, EOFError, KeyError):
                            payload_kinds[cache_key] = False
                    if payload_kinds[cache_key]:
                        reason = 'confirmed_empty'
                if reason:
                    self.missing[key] = {'checked_at': at, 'reason': reason}
        atomic_json(self.missing_path, {'version': 1, 'records': self.missing})

    def remember_missing(self, key, reason):
        with self.lock:
            self.missing[key] = {'checked_at': time.time(), 'reason': reason}
            atomic_json(self.missing_path, {'version': 1, 'records': self.missing})

    def throttle(self):
        with self.rate_lock:
            delay = max(0, self.next_request - time.monotonic())
            if delay and self.stop.wait(delay):
                return False
            if self.stop.is_set():
                return False
            self.next_request = time.monotonic() + 1 / self.rate
            return True

    def log(self, record):
        with self.lock:
            with (self.root / 'requests.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def read_cache(self, key):
        with self.lock:
            entry = self.state.get(key)
        if not entry:
            return None
        try:
            path = self.root / entry['path']
            raw = gzip.decompress(path.read_bytes())
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                return None
            return (json.loads(raw), entry)
        except (OSError, ValueError, EOFError, KeyError):
            return None

    def remember(self, key, entry, cache_hit):
        with self.lock:
            self.used[key] = {**entry, 'cache_hit': cache_hit}
            if cache_hit:
                self.cache_hits += 1

    def fetch(self, key, url, validator=None, refresh=False):
        cached = self.read_cache(key)
        if cached and (self.offline or not (self.refresh or refresh)):
            payload, entry = cached
            if validator is None or validator(payload):
                self.remember(key, entry, True)
                return payload
        with self.lock:
            missed = self.missing.get(key)
            if missed and (not (self.refresh or refresh or self.retry_missing)) and (time.time() - missed['checked_at'] < self.missing_ttl_seconds):
                self.missing_cache_hits += 1
                return None
        if self.offline:
            self.failures.append({'key': key, 'url': url, 'error': 'Not available in valid offline cache'})
            return None
        last_error = 'unknown failure'
        for attempt in range(self.retries + 1):
            if not self.throttle():
                return None
            retry_after = None
            try:
                request = Request(url, headers={'User-Agent': 'BandoriCNResearchCollector/' + VERSION, 'Accept': 'application/json'})
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    headers = {k: response.headers.get(k) for k in ['ETag', 'Last-Modified', 'Date', 'Content-Type']}
                payload = json.loads(raw)
                sha = hashlib.sha256(raw).hexdigest()
                relative = Path('raw') / key / (sha + '.json.gz')
                entry = {'key': key, 'url': url, 'path': relative.as_posix(), 'sha256': sha, 'retrieved_at_utc': now_iso(), 'bytes_uncompressed': len(raw), 'response_headers': headers}
                valid = validator is None or validator(payload)
                with self.lock:
                    atomic_bytes(self.root / relative, gzip.compress(raw, mtime=0))
                    self.log({**entry, 'status': 'ok' if valid else 'invalid_payload'})
                    if valid:
                        self.state[key] = entry
                        atomic_json(self.state_path, self.state)
                        self.network_successes += 1
                        if key in self.missing:
                            self.missing.pop(key, None)
                            atomic_json(self.missing_path, {'version': 1, 'records': self.missing})
                if not valid:
                    if confirmed_empty(key, payload):
                        self.remember_missing(key, 'confirmed_empty')
                    last_error = 'JSON received, but required data was absent or had an unexpected schema'
                    break
                self.remember(key, entry, False)
                return payload
            except HTTPError as e:
                last_error = 'HTTP ' + str(e.code)
                if e.code in [401, 403, 404, 410]:
                    if e.code in (404, 410):
                        self.remember_missing(key, 'HTTP ' + str(e.code))
                    e.close()
                    break
                if e.code == 429:
                    try:
                        retry_after = min(60, max(1, float(e.headers.get('Retry-After', '5'))))
                    except ValueError:
                        retry_after = 5
                e.close()
            except (URLError, TimeoutError, socket.timeout, OSError, ValueError) as e:
                last_error = str(e)
            if attempt < self.retries:
                if self.stop.wait(retry_after if retry_after is not None else min(20, 2 ** attempt + random.random())):
                    return None
        record = {'key': key, 'url': url, 'error': last_error, 'at_utc': now_iso(), 'status': 'failed'}
        with self.lock:
            self.failures.append(record)
        self.log(record)
        if cached and (validator is None or validator(cached[0])):
            self.remember(key, cached[1], True)
            with self.lock:
                self.used[key]['refresh_failed'] = True
            return cached[0]
        return None

    def get(self, kind, event=None, tier=None, sources=('bestdori', 'hhwx'), refresh=False):
        if self.offline or not (self.refresh or refresh):
            for source in sources:
                key, _ = resource_location(kind, event, tier, source)
                cached = self.read_cache(key)
                if cached and valid_resource(kind, cached[0], source):
                    self.remember(key, cached[1], True)
                    return {'source': source, 'key': key, 'payload': cached[0]}
        for source in sources:
            key, url = resource_location(kind, event, tier, source)
            payload = self.fetch(key, url, lambda p: valid_resource(kind, p, source), refresh=refresh)
            if payload is not None:
                return {'source': source, 'key': key, 'payload': payload}
        return None
