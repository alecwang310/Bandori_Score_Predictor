#!/usr/bin/env python3
"""Collect public Bandori CN forecasting data. Python 3.10+, standard library only.

Run: python3 collect_bandori_data.py
The default collects every started CN event, seven useful cutoff tiers, hourly
T10 history, event details, final-score archives, and supporting game metadata.
Interrupted runs resume. No model fitting or train/validation splitting occurs.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

UTC = timezone.utc
CN = timezone(timedelta(hours=8))
SERVER = 3
VERSION = "1.1.0"
DEFAULT_TIERS = [100, 500, 1000, 1500, 2000, 5000, 10000]
ALL_TIERS = [20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500,
             2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000,
             70000, 100000]
SOURCES = {
    "bestdori": {
        "events": "https://bestdori.com/api/events/all.6.json",
        "detail": "https://bestdori.com/api/events/{event}.json",
        "tracker": "https://bestdori.com/api/tracker/data?server=3&event={event}&tier={tier}",
        "top10": "https://bestdori.com/api/eventtop/data?server=3&event={event}&mid=0&interval=3600000",
    },
    "hhwx": {
        "events": "https://hhwx.org/api/bandori/master/events",
        "detail": "https://hhwx.org/api/bandori/master/events/{event}",
        "tracker": "https://hhwx.org/api/bandori/tracker/data?server=3&event={event}&type=event&tier={tier}",
        "top10": "https://hhwx.org/api/bandori/tracker/data?server=3&event={event}&type=event&tier=10",
    },
}
MASTER_URLS = {
    "archives": "https://bestdori.com/api/archives/all.5.json",
    "cards": "https://bestdori.com/api/cards/all.5.json",
    "gacha": "https://bestdori.com/api/gacha/all.5.json",
    "bands": "https://bestdori.com/api/bands/all.1.json",
    "characters": "https://bestdori.com/api/characters/main.3.json",
    "songs": "https://bestdori.com/api/songs/all.7.json",
    "login_campaigns": "https://bestdori.com/api/loginCampaigns/all.5.json",
}
SOURCE_DOCS = [
    "https://github.com/WindowsSov8forUs/bestdori-api/blob/main/bestdori/eventtracker.py",
    "https://github.com/WindowsSov8forUs/bestdori-api/blob/main/bestdori/eventarchives.py",
    "https://github.com/byydzh/MYCX_1000/blob/main/config.py",
]


def now_iso():
    return datetime.now(UTC).isoformat(timespec="seconds")


def timestamp(value):
    """Accept API milliseconds, Unix seconds, or timezone-qualified ISO dates."""
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
        if not math.isfinite(x) or x <= 0:
            return None
        return x / 1000 if x > 100_000_000_000 else x
    except (ValueError, TypeError, OverflowError):
        try:
            d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return d.timestamp() if d.tzinfo is not None else None
        except (ValueError, TypeError, OverflowError):
            return None


def iso(value, tz=UTC):
    return datetime.fromtimestamp(value, tz).isoformat(timespec="seconds") if value is not None else ""


def slot(value, index=SERVER):
    if isinstance(value, list):
        return value[index] if len(value) > index else None
    if isinstance(value, dict):
        return value.get(str(index), value.get("cn"))
    return None


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def write_csv(path, fields, rows):
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    atomic_bytes(path, s.getvalue().encode("utf-8-sig"))


def normalize_events(payload, source):
    if source == "hhwx":
        if payload.get("success") is not True or not isinstance(payload.get("data"), dict):
            raise ValueError("HHWX event index has an unexpected schema")
        payload = payload["data"]
        if isinstance(payload.get("events"), list):
            result = {}
            for e in payload["events"]:
                if not isinstance(e, dict) or e.get("eventId") is None:
                    continue
                v = dict(e)
                # Normalize only CN; preserve the complete original response on disk.
                cn = (e.get("timeline") or {}).get("cn") or {}
                v["startAt"] = [None, None, None, cn.get("startAt")]
                v["endAt"] = [None, None, None, cn.get("endAt")]
                result[str(e["eventId"])] = v
            payload = result
    if not isinstance(payload, dict):
        raise ValueError("Event index must be an object")
    result = {str(k): v for k, v in payload.items() if str(k).isdigit() and isinstance(v, dict)}
    if not result:
        raise ValueError("Event index has no numeric event IDs")
    return result


def select_events(indexes, as_of):
    """CN order is determined by timestamps, never by the global event ID."""
    merged = {}
    conflicts = []
    for source, mapping in indexes:
        for event_id, meta in mapping.items():
            start = timestamp(slot(meta.get("startAt")))
            end = timestamp(slot(meta.get("endAt")))
            if start is None or end is None or end <= start:
                continue
            if event_id in merged:
                old = merged[event_id]
                if old["start"] != start or old["end"] != end:
                    conflicts.append({"event_id": int(event_id), "preferred_source": old["source"],
                                      "other_source": source, "preferred_start": iso(old["start"]),
                                      "other_start": iso(start), "preferred_end": iso(old["end"]),
                                      "other_end": iso(end)})
                continue
            if start <= as_of:
                merged[event_id] = {"event_id": int(event_id), "start": start, "end": end,
                                    "source": source, "meta": meta}
    events = sorted(merged.values(), key=lambda x: (x["start"], x["event_id"]))
    for ordinal, e in enumerate(events, 1):
        e["cn_event_number"] = ordinal
    return events, conflicts


def parse_points(payload):
    """Keep valid observations and corrections; never interpolate or smooth."""
    points, invalid = [], 0
    for record in payload.get("cutoffs", []):
        if not isinstance(record, dict):
            invalid += 1
            continue
        t = timestamp(record.get("time"))
        try:
            raw_score = record["ep"]
            score = float(raw_score)
            if isinstance(raw_score, bool) or not math.isfinite(score) or score < 0 or not score.is_integer() or t is None:
                raise ValueError("Invalid observation")
        except (KeyError, ValueError, TypeError, OverflowError):
            invalid += 1
            continue
        available = timestamp(record.get("availableAt", record.get("available_at")))
        points.append({"time": t, "score": int(score), "available_at": available})
    points.sort(key=lambda x: x["time"])
    return points, invalid


def checkpoints(start, end, as_of):
    d = datetime.fromtimestamp(start, CN).replace(hour=8, minute=0, second=0, microsecond=0)
    if d.timestamp() <= start:
        d += timedelta(days=1)
    while d.timestamp() < end and d.timestamp() <= as_of:
        yield d.timestamp()
        d += timedelta(days=1)


def checkpoint_rows(event, tier, points, as_of, max_stale_minutes):
    # Stable sorting retains corrections at the same timestamp; the last eligible
    # source record wins. Later observations never enter an earlier checkpoint.
    times = [p["time"] for p in points]
    for day, cutoff in enumerate(checkpoints(event["start"], event["end"], as_of), 1):
        i = bisect.bisect_right(times, cutoff) - 1
        while i >= 0 and points[i]["available_at"] is not None and points[i]["available_at"] > cutoff:
            i -= 1
        p = points[i] if i >= 0 else None
        stale = (cutoff - p["time"]) / 60 if p else None
        status = "missing" if p is None else ("stale" if stale > max_stale_minutes else "ok")
        yield {"event_id": event["event_id"], "cn_event_number": event["cn_event_number"],
               "tier": tier, "checkpoint_number": day, "checkpoint_at_utc": iso(cutoff),
               "checkpoint_at_cn": iso(cutoff, CN), "elapsed_hours": round((cutoff-event["start"])/3600, 6),
               "score": p["score"] if p else "", "observation_at_utc": iso(p["time"]) if p else "",
               "staleness_minutes": round(stale, 3) if stale is not None else "", "status": status,
               "availability_evidence": "upstream_available_at" if p and p["available_at"] is not None else "observation_timestamp_only"}


def archive_score(archives, event_id, tier):
    e = archives.get(str(event_id), {}) if isinstance(archives, dict) else {}
    cn = slot(e.get("cutoff")) if isinstance(e, dict) else None
    if not isinstance(cn, dict):
        return None
    raw = cn.get(str(tier))
    try:
        x = float(raw)
        if isinstance(raw, bool) or not math.isfinite(x) or x <= 0 or not x.is_integer():
            return None
        return int(x)
    except (ValueError, TypeError, OverflowError):
        return None


def confirmed_empty(key,payload):
    if not isinstance(payload,dict):return False
    if '/tracker/' in key:
        return payload.get('result') is True and payload.get('cutoffs')==[]
    if '/top10/' in key:
        return (payload.get('result') is True and payload.get('cutoffs')==[]) or payload.get('points')==[]
    return False


def resource_location(kind,event,tier,source):
    suffix=kind if event is None else f'{kind}/{event}'
    if tier is not None:suffix+=f'/tier_{tier}'
    return source+'/'+suffix,SOURCES[source][kind].format(event=event,tier=tier)


def valid_resource(kind,p,source):
    if not isinstance(p,dict):return False
    if kind=='events':
        try:normalize_events(p,source);return True
        except (ValueError,TypeError,AttributeError):return False
    if kind=='detail':
        body=p.get('data') if source=='hhwx' and p.get('success') is True else p
        return isinstance(body,dict) and isinstance(body.get('eventType'),str) and bool(body['eventType'])
    if kind=='tracker':
        # A single valid record establishes usability; export validates all records.
        return p.get('result') is True and isinstance(p.get('cutoffs'),list) and any(parse_points({'cutoffs':[r]})[0] for r in p['cutoffs'])
    if kind=='top10':return bool(p.get('points')) or (p.get('result') is True and bool(p.get('cutoffs')))
    return True


class Collector:
    def __init__(self, root, rate=2.0, timeout=30.0, retries=3, offline=False, refresh=False,
                 retry_missing=False,missing_ttl_hours=168):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text("utf-8")) if self.state_path.exists() else {}
        self.rate = rate
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        self.refresh = refresh
        self.retry_missing=retry_missing
        self.missing_ttl_seconds=missing_ttl_hours*3600
        self.missing_path=self.root/'missing_cache.json'
        self.missing=json.loads(self.missing_path.read_text()).get('records',{}) if self.missing_path.exists() else {}
        self.lock = threading.RLock()
        self.rate_lock = threading.Lock()
        self.next_request = 0.0
        self.failures = []
        self.used = {}
        self.network_successes = 0
        self.cache_hits = 0
        self.missing_cache_hits=0
        self.stop = threading.Event()
        if not self.missing_path.exists() and not offline:self.seed_missing_cache()

    def seed_missing_cache(self):
        """Recover definitive misses from earlier collector versions' audit log."""
        log_path=self.root/'requests.jsonl'
        if not log_path.exists():return
        payload_kinds={}
        with log_path.open(encoding='utf-8') as rows:
            for line in rows:
                try:r=json.loads(line)
                except ValueError:continue
                key=r.get('key');at=timestamp(r.get('retrieved_at_utc',r.get('at_utc')))
                if not key or at is None or time.time()-at>self.missing_ttl_seconds:continue
                if r.get('status')=='ok':self.missing.pop(key,None);continue
                reason=None
                if r.get('status')=='failed' and r.get('error') in ('HTTP 404','HTTP 410'):reason=r['error']
                if r.get('status')=='invalid_payload' and ('/tracker/' in key or '/top10/' in key):
                    cache_key=(r.get('sha256'),'/tracker/' in key)
                    if cache_key not in payload_kinds:
                        try:
                            raw=gzip.decompress((self.root/r['path']).read_bytes())
                            payload_kinds[cache_key]=hashlib.sha256(raw).hexdigest()==r['sha256'] and confirmed_empty(key,json.loads(raw))
                        except (OSError,ValueError,EOFError,KeyError):payload_kinds[cache_key]=False
                    if payload_kinds[cache_key]:reason='confirmed_empty'
                if reason:self.missing[key]={'checked_at':at,'reason':reason}
        atomic_json(self.missing_path,{'version':1,'records':self.missing})

    def remember_missing(self,key,reason):
        with self.lock:
            self.missing[key]={'checked_at':time.time(),'reason':reason}
            atomic_json(self.missing_path,{'version':1,'records':self.missing})

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
            with (self.root / "requests.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def read_cache(self, key):
        with self.lock:
            entry = self.state.get(key)
        if not entry:
            return None
        try:
            path = self.root / entry["path"]
            raw = gzip.decompress(path.read_bytes())
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                return None
            return json.loads(raw), entry
        except (OSError, ValueError, EOFError, KeyError):
            return None

    def remember(self, key, entry, cache_hit):
        with self.lock:
            self.used[key] = {**entry, "cache_hit": cache_hit}
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
            missed=self.missing.get(key)
            if missed and not (self.refresh or refresh or self.retry_missing) and time.time()-missed['checked_at']<self.missing_ttl_seconds:
                self.missing_cache_hits+=1
                return None
        if self.offline:
            self.failures.append({"key": key, "url": url, "error": "Not available in valid offline cache"})
            return None
        last_error = "unknown failure"
        for attempt in range(self.retries + 1):
            if not self.throttle():
                return None
            retry_after = None
            try:
                request = Request(url, headers={"User-Agent": "BandoriCNResearchCollector/" + VERSION,
                                                "Accept": "application/json"})
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    headers = {k: response.headers.get(k) for k in ["ETag", "Last-Modified", "Date", "Content-Type"]}
                # Reject HTML challenge pages and malformed API responses.
                payload = json.loads(raw)
                sha = hashlib.sha256(raw).hexdigest()
                relative = Path("raw") / key / (sha + ".json.gz")
                entry = {"key": key, "url": url, "path": relative.as_posix(), "sha256": sha,
                         "retrieved_at_utc": now_iso(), "bytes_uncompressed": len(raw), "response_headers": headers}
                # Retain failed JSON responses too, for diagnosis, without treating
                # them as successful resume-cache entries.
                valid = validator is None or validator(payload)
                with self.lock:
                    atomic_bytes(self.root / relative, gzip.compress(raw, mtime=0))
                    self.log({**entry, "status": "ok" if valid else "invalid_payload"})
                    if valid:
                        self.state[key] = entry
                        atomic_json(self.state_path, self.state)
                        self.network_successes += 1
                        if key in self.missing:
                            self.missing.pop(key,None)
                            atomic_json(self.missing_path,{'version':1,'records':self.missing})
                if not valid:
                    if confirmed_empty(key,payload):self.remember_missing(key,'confirmed_empty')
                    last_error = "JSON received, but required data was absent or had an unexpected schema"
                    break
                self.remember(key, entry, False)
                return payload
            except HTTPError as e:
                last_error = "HTTP " + str(e.code)
                if e.code in [401, 403, 404, 410]:
                    if e.code in (404,410):self.remember_missing(key,'HTTP '+str(e.code))
                    e.close()
                    break
                if e.code == 429:
                    try:
                        retry_after = min(60, max(1, float(e.headers.get("Retry-After", "5"))))
                    except ValueError:
                        retry_after = 5
                e.close()
            except (URLError, TimeoutError, socket.timeout, OSError, ValueError) as e:
                last_error = str(e)
            if attempt < self.retries:
                if self.stop.wait(retry_after if retry_after is not None else min(20, 2 ** attempt + random.random())):
                    return None
        record = {"key": key, "url": url, "error": last_error, "at_utc": now_iso(), "status": "failed"}
        with self.lock:
            self.failures.append(record)
        self.log(record)
        if cached and (validator is None or validator(cached[0])):
            self.remember(key,cached[1],True)
            with self.lock:self.used[key]['refresh_failed']=True
            return cached[0]
        return None

    def get(self, kind, event=None, tier=None, sources=("bestdori", "hhwx"), refresh=False):
        # A cached fallback is already usable: do not probe a missing primary first.
        if self.offline or not (self.refresh or refresh):
            for source in sources:
                key,_=resource_location(kind,event,tier,source);cached=self.read_cache(key)
                if cached and valid_resource(kind,cached[0],source):
                    self.remember(key,cached[1],True)
                    return {'source':source,'key':key,'payload':cached[0]}
        for source in sources:
            key,url=resource_location(kind,event,tier,source)
            payload = self.fetch(key, url,lambda p:valid_resource(kind,p,source),refresh=refresh)
            if payload is not None:
                return {"source": source, "key": key, "payload": payload}
        return None


def summarize(root, events, tiers, selected, archives, as_of, max_stale_minutes, collector=None):
    export = root / "tables"
    export.mkdir(parents=True, exist_ok=True)
    observation_path = export / "observations.csv"
    coverage, snapshots, labels, candidates, gaps = [], [], [], [], []
    obs_fields = ["event_id", "tier", "time_utc", "time_cn", "ep", "available_at_utc", "source"]
    tmp = observation_path.with_name(observation_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=obs_fields)
        writer.writeheader()
        for event in events:
            for tier in tiers:
                item = selected.get((event["event_id"], tier))
                if item and "payload" not in item:
                    cached = collector.read_cache(item["key"]) if collector else None
                    if cached is None:
                        raise RuntimeError("A downloaded resource failed its checksum before export: " + item["key"])
                    payload = cached[0]
                else:
                    payload = item["payload"] if item else {}
                points, invalid = parse_points(payload) if item else ([], 0)
                # Do not silently delete duplicate timestamps or score decreases.
                in_event = [p for p in points if event["start"] <= p["time"] <= event["end"]]
                for p in points:
                    writer.writerow({"event_id": event["event_id"], "tier": tier,
                                     "time_utc": iso(p["time"]), "time_cn": iso(p["time"], CN),
                                     "ep": p["score"], "available_at_utc": iso(p["available_at"]),
                                     "source": item["source"]})
                pairs = list(zip(in_event, in_event[1:]))
                gap_count = 0
                for a, b in pairs:
                    hours = (b["time"] - a["time"]) / 3600
                    if hours >= 3:
                        gap_count += 1
                        gaps.append({"event_id": event["event_id"], "tier": tier,
                                     "from_utc": iso(a["time"]), "to_utc": iso(b["time"]),
                                     "hours": round(hours, 3), "kind": "observation_gap_not_confirmed_outage"})
                first, last = (points[0], points[-1]) if points else (None, None)
                archived = archive_score(archives, event["event_id"], tier)
                if archived is not None and event["end"] <= as_of:
                    labels.append({"event_id": event["event_id"], "tier": tier, "final_ep": archived,
                                   "source": "bestdori_archive", "label_status": "archived_final"})
                # Last tracker score is diagnostic only. It is never promoted to
                # a final target merely because the event has ended.
                if last:
                    candidates.append({"event_id": event["event_id"], "tier": tier,
                                       "last_observed_ep": last["score"], "last_observed_at_utc": iso(last["time"]),
                                       "seconds_relative_to_end": round(last["time"]-event["end"], 3),
                                       "archived_final_ep": archived if archived is not None else "",
                                       "label_status": "requires_archive_or_manual_verification"})
                coverage.append({"event_id": event["event_id"], "cn_event_number": event["cn_event_number"],
                                 "tier": tier, "status": "ok" if points else "missing", "source": item["source"] if item else "",
                                 "observations": len(points), "invalid_records": invalid,
                                 "first_at_utc": iso(first["time"]) if first else "",
                                 "last_at_utc": iso(last["time"]) if last else "",
                                 "in_event_observations": len(in_event),
                                 "duplicate_timestamps": sum(a["time"] == b["time"] for a, b in zip(points, points[1:])),
                                 "score_decreases": sum(b["score"] < a["score"] for a, b in zip(points, points[1:])),
                                 "gaps_at_least_3h": gap_count,
                                 "archive_final_available": archived is not None})
                # Filter pre-event records out of features, keep all in raw export.
                snapshots.extend(checkpoint_rows(event, tier, in_event, as_of, max_stale_minutes))
    os.replace(tmp, observation_path)
    write_csv(export / "coverage.csv", ["event_id", "cn_event_number", "tier", "status", "source", "observations",
              "invalid_records", "first_at_utc", "last_at_utc", "in_event_observations", "duplicate_timestamps",
              "score_decreases", "gaps_at_least_3h", "archive_final_available"], coverage)
    write_csv(export / "checkpoints_0800_cn.csv", ["event_id", "cn_event_number", "tier", "checkpoint_number",
              "checkpoint_at_utc", "checkpoint_at_cn", "elapsed_hours", "score", "observation_at_utc",
              "staleness_minutes", "status", "availability_evidence"], snapshots)
    write_csv(export / "final_labels.csv", ["event_id", "tier", "final_ep", "source", "label_status"], labels)
    write_csv(export / "label_candidates.csv", ["event_id", "tier", "last_observed_ep", "last_observed_at_utc",
              "seconds_relative_to_end", "archived_final_ep", "label_status"], candidates)
    write_csv(export / "observation_gaps.csv", ["event_id", "tier", "from_utc", "to_utc", "hours", "kind"], gaps)
    event_rows = []
    for e in events:
        meta = e["meta"]
        event_rows.append({"event_id": e["event_id"], "cn_event_number": e["cn_event_number"],
                           "event_name_cn": slot(meta.get("eventName")) or "", "event_type": meta.get("eventType", ""),
                           "start_at_utc": iso(e["start"]), "end_at_utc": iso(e["end"]),
                           "start_at_cn": iso(e["start"], CN), "end_at_cn": iso(e["end"], CN),
                           "duration_hours": round((e["end"]-e["start"])/3600, 6),
                           "status": "completed" if e["end"] <= as_of else "ongoing", "metadata_source": e["source"]})
    write_csv(export / "events.csv", ["event_id", "cn_event_number", "event_name_cn", "event_type", "start_at_utc",
              "end_at_utc", "start_at_cn", "end_at_cn", "duration_hours", "status", "metadata_source"], event_rows)
    return {"event_count": len(events), "tiers": tiers, "event_tier_pairs": len(coverage),
            "pairs_with_history": sum(r["status"] == "ok" for r in coverage),
            "missing_pairs": sum(r["status"] == "missing" for r in coverage),
            "observation_count": sum(r["observations"] for r in coverage),
            "archived_final_labels": len(labels), "checkpoint_rows": len(snapshots),
            "fresh_checkpoint_rows": sum(r["status"] == "ok" for r in snapshots),
            "stale_checkpoint_rows": sum(r["status"] == "stale" for r in snapshots),
            "missing_checkpoint_rows": sum(r["status"] == "missing" for r in snapshots),
            "gap_count": len(gaps)}


BUNDLE_README = """Bandori CN research data bundle

All files are data collection outputs, not trained models or predictions.
Server is CN (Bestdori server=3). UTC and fixed UTC+08:00 dates are explicit.
Event IDs are the upstream global IDs. cn_event_number is the ordinal derived
from the currently available CN event index, before any --latest/--event-ids
filter. Missing index entries can affect that ordinal; do not use it as identity.

raw/: content-addressed gzip JSON responses, preserved without smoothing.
manifest.json: selected source, URL, retrieval time, SHA-256, and all used inputs.
requests.jsonl: append-only request history, including failed endpoint attempts.
state.json: resume cache. Rerunning retries missing data and reuses valid history.
tables/observations.csv: all valid cutoff observations, including corrections.
tables/checkpoints_0800_cn.csv: last eligible observation at each local 08:00;
  never interpolated from a later observation. Stale and missing rows are flagged.
tables/events.csv: current CN dates, duration, event type, and name.
tables/coverage.csv: missing histories, bad records, duplicates, and decreases.
tables/final_labels.csv: only explicitly archived final scores, when available.
tables/label_candidates.csv: last tracker scores, NOT verified training targets.
tables/observation_gaps.csv: missing observations, NOT confirmed server outages.
metadata_conflicts.json: dates disagreeing across providers, if any.
summary.json: coverage totals and download failures.

Important modeling constraints:
* Historical tracker APIs usually provide observation time, not publication time.
  Retrieval time is preserved separately. Backfilled or revised records cannot
  prove what was actually available at a historical 08:00 checkpoint.
* Current event schedules may include end-time extensions announced later.
  Use announcement records before treating historical remaining time as known.
* Exact archived finals may be absent, especially for newer/intermediate tiers.
  Do not silently use an unfinished tracker tail as the final target.
* A gap or a flat score is not evidence by itself of a game server outage.
  Check official notices and cross-tier behavior before assigning downtime.
* Keep complete events together in chronological train/validation/test splits.
  Never train on validation/test labels or use them as historical references.
  This downloader intentionally performs no model fitting or dataset splitting.

Optional annotations/ CSV templates are supplied for confirmed outages and
schedule revisions. Leave them empty when unknown; never invent those values.
Add official notice text/files/links there if you already have them. No account,
credentials, private game API, community comments, or login is needed.

Read summary.json before modeling. Missing data is normal; a completed download
is not a claim that every tier or event exists at the upstream provider.
"""


def make_templates(root):
    annotation = root / "annotations"
    annotation.mkdir(exist_ok=True)
    templates = {
        "outages.csv": ["event_id", "start_at", "end_at", "known_at", "source_url", "notes"],
        "schedule_revisions.csv": ["event_id", "known_at", "start_at", "end_at", "source_url", "notes"],
        "verified_finals.csv": ["event_id", "tier", "final_ep", "source_url", "notes"],
    }
    for name, fields in templates.items():
        if not (annotation / name).exists():
            write_csv(annotation / name, fields, [])
    atomic_bytes(root / "README.txt", BUNDLE_README.encode("utf-8"))


def make_zip(root):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root.parent / f"{root.name}_{stamp}.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp") and path.name != ".collector.lock":
                archive.write(path, arcname=Path(root.name) / path.relative_to(root))
    return destination


def run(args):
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    make_templates(root)
    # An exclusive output lock prevents two collectors from corrupting the cache.
    # The handle is removed on normal completion and Ctrl+C; after a hard crash,
    # remove .collector.lock only once the earlier process is no longer running.
    lock_path = root / ".collector.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Another run may be using {root}. If it has stopped, remove {lock_path} and rerun.")
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.close(lock_fd)
        return collect(args, root)
    finally:
        lock_path.unlink(missing_ok=True)


def download_jobs(events,tiers,as_of,order='newest',skip_top10=False):
    """Prioritize CN dates and useful score lines; retain chronological exports."""
    jobs=[]
    priority=[t for t in (500,1000,1500,2000) if t in tiers]+[t for t in tiers if t not in (500,1000,1500,2000)]
    for event in sorted(events,key=lambda e:(e['start'],e['event_id']),reverse=order=='newest'):
        ongoing=event['end']>=as_of;eid=event['event_id']
        jobs.extend(('tracker',eid,tier,ongoing) for tier in priority)
        if not skip_top10:jobs.append(('top10',eid,None,ongoing))
        jobs.append(('detail',eid,None,ongoing))
    return jobs


def collect(args, root):
    as_of = time.time()
    sources = ("bestdori", "hhwx") if args.source == "auto" else (args.source,)
    collector = Collector(root, args.rate, args.timeout, args.retries, args.offline, args.refresh,
                          args.retry_missing,args.missing_ttl_hours)
    indexes = []
    print("Fetching CN event indexes...", flush=True)
    # In auto mode keep both indexes, so missing CN dates and disagreements are visible.
    for source in sources:
        item = collector.get("events", sources=(source,), refresh=True)
        if item:
            indexes.append((source, normalize_events(item["payload"], source)))
    if not indexes:
        atomic_json(root / "summary.json", {"status": "failed_no_event_index", "at_utc": now_iso(),
                                           "failures": collector.failures})
        raise RuntimeError("No usable CN event index. Check internet access, proxy settings, and summary.json; rerun to resume.")
    events, conflicts = select_events(indexes, as_of)
    discovered = len(events)
    if args.event_ids:
        wanted = set(args.event_ids)
        found = {e["event_id"] for e in events}
        if wanted - found:
            raise ValueError(f"These global IDs have no started CN schedule: {sorted(wanted-found)}")
        events = [e for e in events if e["event_id"] in wanted]
    if args.latest:
        events = events[-args.latest:]
    if not events:
        raise RuntimeError("No started CN events matched the selection")
    tiers = ALL_TIERS if args.all_tiers else args.tiers
    print(f"Found {discovered} started CN events; collecting {len(events)} events and tiers {tiers}.", flush=True)
    atomic_json(root / "metadata_conflicts.json", conflicts)
    atomic_json(root / "selection.json", {"as_of_utc": iso(as_of), "discovered_cn_events": discovered,
                                        "event_ids": [e["event_id"] for e in events], "tiers": tiers,
                                        "download_order":args.order})
    masters = {}
    if not args.skip_supporting_metadata:
        master_names = list(MASTER_URLS)
    else:
        master_names = ["archives"]
    jobs=download_jobs(events,tiers,as_of,args.order,args.skip_top10)
    selected = {}
    task_status = []
    print(f"Downloading {len(jobs)} event resources (cached successes will be reused)...", flush=True)
    print('Event download order:',args.order,'; first events:',list(dict.fromkeys(j[1] for j in jobs))[:5],flush=True)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(collector.get, kind, event, tier, sources, refresh): (kind, event, tier)
               for kind, event, tier, refresh in jobs}
    try:
        for count, future in enumerate(as_completed(futures), 1):
            kind, event, tier = futures.pop(future)
            item = future.result()
            task_status.append({"kind": kind, "event_id": event, "tier": tier,
                                "status": "ok" if item else "missing", "key": item["key"] if item else None})
            if kind == "tracker" and item:
                # Keep references, not every multi-year time series in memory.
                selected[(event, tier)] = {"key": item["key"], "source": item["source"]}
            if count % 20 == 0 or count == len(jobs):
                print(f"[{count}/{len(jobs)}] histories {len(selected)}, network successes {collector.network_successes}, "
                      f"cache hits {collector.cache_hits}, cached misses {collector.missing_cache_hits}", flush=True)
    except BaseException:
        collector.stop.set()
        for pending in futures:
            pending.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    # Do not delay current-event score histories behind nonessential metadata.
    for name in master_names:
        print("Fetching supporting metadata:", name, flush=True)
        masters[name] = collector.fetch("bestdori/master/" + name, MASTER_URLS[name],
                                       lambda p: isinstance(p, dict) and bool(p), refresh=(args.refresh or name=='archives'))
    print("Exporting chronological observations and 08:00 checkpoints...", flush=True)
    stats = summarize(root, events, tiers, selected, masters.get("archives") or {}, as_of, args.max_stale_minutes, collector)
    manifest = {"collector_version": VERSION, "server": SERVER, "timezone": "UTC+08:00", "collected_at_utc": now_iso(),
                "selection_as_of_utc": iso(as_of), "source_documentation": SOURCE_DOCS,
                "arguments": vars(args), "resources": dict(sorted(collector.used.items())),
                "tasks": sorted(task_status, key=lambda r: (r["event_id"], r["kind"], r["tier"] or 0)),
                "availability_assumption": "Historical upstream publication times unknown unless supplied per observation",
                "statistical_processing": "No interpolation, smoothing, model fitting, or split assignment"}
    atomic_json(root / "manifest.json", manifest)
    missing_tasks = sum(t["status"] != "ok" for t in task_status)
    summary = {"status": "complete_with_missing_data" if missing_tasks or collector.failures else "complete",
               "collected_at_utc": now_iso(), "discovered_cn_events": discovered, **stats,
               "missing_event_resources": missing_tasks, "metadata_conflicts": len(conflicts),
               "network_successes": collector.network_successes, "cache_hits": collector.cache_hits,
               "missing_cache_hits":collector.missing_cache_hits,
               "endpoint_failures": collector.failures,
               "notes": ["Provider failure may have been recovered by the fallback provider.",
                         "Archived finals may be unavailable; tracker tails are not verified targets.",
                         "Observation gaps do not establish server outages.",
                         "Current metadata cannot establish historical schedule announcement times."]}
    atomic_json(root / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["endpoint_failures", "notes"]}, indent=2))
    if not args.no_zip:
        archive = make_zip(root)
        print("\nSend this file back:", archive, flush=True)
    else:
        print("\nCollected data directory:", root, flush=True)
    return 0 if stats["pairs_with_history"] else 2


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return n


def tier_list(value):
    try:
        values = sorted(set(int(x.strip().lstrip("Tt")) for x in value.split(",")))
    except ValueError as e:
        raise argparse.ArgumentTypeError("Use comma-separated tiers, e.g. 500,1000,1500,2000") from e
    if not values or any(x not in ALL_TIERS for x in values):
        raise argparse.ArgumentTypeError(f"Supported cutoff tiers: {ALL_TIERS}; T10 history is collected separately")
    return values


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="bandori_cn_data", help="Data directory (default: bandori_cn_data)")
    p.add_argument("--source", choices=["auto", "bestdori", "hhwx"], default="auto",
                   help="auto prefers Bestdori, with HHWX fallback for missing data")
    p.add_argument("--tiers", type=tier_list, default=DEFAULT_TIERS,
                   help="Comma-separated fixed tiers (default: 100,500,1000,1500,2000,5000,10000)")
    p.add_argument("--all-tiers", action="store_true", help="Request every supported cutoff tier; substantially slower")
    p.add_argument("--event-ids", type=lambda s: [int(x) for x in s.split(",")], help="Optional global event IDs, not CN ordinal numbers")
    p.add_argument("--latest", type=positive_int, help="Only the latest N started CN events (useful for a quick check)")
    p.add_argument('--order',choices=('newest','oldest'),default='newest',help='Download by CN dates, newest first by default')
    p.add_argument('--retry-missing',action='store_true',help='Retry cached empty/404 histories without refreshing successful downloads')
    p.add_argument('--missing-ttl-hours',type=float,default=168,help='Reuse confirmed missing historical resources for this many hours (default: 168)')
    p.add_argument("--workers", type=positive_int, default=2, help="Concurrent requests (default: 2; maximum 4)")
    p.add_argument("--rate", type=float, default=2.0, help="Maximum total requests per second (default: 2; maximum 3)")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="Retries for transient failures (default: 3)")
    p.add_argument("--max-stale-minutes", type=float, default=60.0, help="Flag checkpoint observations older than this; never interpolate")
    p.add_argument("--skip-top10", action="store_true", help="Omit auxiliary T10 histories")
    p.add_argument("--skip-supporting-metadata", action="store_true", help="Omit cards/gacha/bands/songs/login metadata")
    p.add_argument("--offline", action="store_true", help="Rebuild exports and zip exclusively from the local cache")
    p.add_argument("--refresh", action="store_true", help="Refetch cached historical resources; old raw versions are retained")
    p.add_argument("--no-zip", action="store_true", help="Omit the final upload bundle")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if not math.isfinite(args.rate) or not 0 < args.rate <= 3:
        parser.error("--rate must be greater than 0 and at most 3")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 0 <= args.retries <= 10:
        parser.error("--retries must be between 0 and 10")
    if not math.isfinite(args.max_stale_minutes) or args.max_stale_minutes <= 0:
        parser.error("--max-stale-minutes must be positive")
    if not math.isfinite(args.missing_ttl_hours) or args.missing_ttl_hours<=0:
        parser.error('--missing-ttl-hours must be positive')
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nStopped. Successful downloads are cached; run the same command to resume.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as e:
        print("ERROR:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
