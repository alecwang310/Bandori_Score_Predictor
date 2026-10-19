"""Read a completed collector bundle. Labels stay separate from observations."""
from dataclasses import dataclass, replace
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from collect_bandori_data import CN, archive_score, parse_points, timestamp


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def top10_curves(payload):
    """Derive T1/T10 only from complete simultaneous top-ten snapshots."""
    if payload.get("cutoffs"):
        points, _ = parse_points(payload)
        if points:
            a = np.asarray([[p['time'], p['score'], p['available_at'] or p['time']] for p in points])
            return {10: Curve(a[:,0],a[:,1],a[:,2])}
    groups = {}
    for p in payload.get("points", []):
        t = timestamp(p.get("time"))
        try:
            score, uid = float(p["value"]), int(p["uid"])
        except (ValueError,TypeError,KeyError):
            continue
        if t is not None and math.isfinite(score) and score >= 0:
            groups.setdefault(t,{})[uid] = score
    rows = []
    for t, scores in sorted(groups.items()):
        if len(scores) == 10:
            values = sorted(scores.values(),reverse=True)
            rows.append((t,values[0],values[9]))
    if not rows:
        return {}
    a = np.asarray(rows)
    return {1: Curve(a[:,0],a[:,1],a[:,0]), 10: Curve(a[:,0],a[:,2],a[:,0])}


@dataclass(frozen=True)
class Event:
    event_id: int
    start: float
    end: float
    event_type: str
    name: str = ""


@dataclass
class Curve:
    times: np.ndarray
    scores: np.ndarray
    available: np.ndarray

    def prefix(self, cutoff, start=None):
        mask = (self.times <= cutoff) & (self.available <= cutoff)
        if start is not None:
            mask &= self.times >= start
        return Curve(self.times[mask], self.scores[mask], self.available[mask])

    def last(self, cutoff, start=None):
        i = int(np.searchsorted(self.times, cutoff, side="right"))-1
        while i >= 0 and self.available[i] > cutoff:
            i -= 1
        if i < 0 or (start is not None and self.times[i] < start):
            return None
        return float(self.times[i]), float(self.scores[i])


class Dataset:
    def __init__(self, events, curves, labels, outages=(), revisions=(), provenance=None, label_sources=None, final_available=None):
        self.events = {e.event_id: e for e in events}
        self.curves = curves
        self.labels = labels
        self.outages = list(outages)
        self.revisions = list(revisions)
        self.provenance = provenance or {}
        self.label_sources = label_sources or {key: "provided_verified" for key in labels}
        self.final_available = final_available or {}

    @classmethod
    def load(cls, root, tiers):
        root = Path(root)
        if not (root / "manifest.json").exists() or (root / ".collector.lock").exists():
            raise ValueError("Use a completed collector bundle; its download/export is still running or its manifest is missing")
        tables = root / "tables"
        manifest = json.loads((root / "manifest.json").read_text())
        resources = manifest.get("resources", {})

        def raw_payload(key):
            entry = resources.get(key)
            if not entry:
                return None
            content = gzip.decompress((root/entry['path']).read_bytes())
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Raw resource checksum mismatch: '+key)
            return json.loads(content)
        events = []
        for r in read_rows(tables / "events.csv"):
            start, end = timestamp(r["start_at_utc"]), timestamp(r["end_at_utc"])
            if start is not None and end is not None and end > start:
                events.append(Event(int(r["event_id"]), start, end, r["event_type"], r.get("event_name_cn", "")))
        tiers = set(tiers)
        all_tiers = not tiers
        grouped = {}
        # Stream the large CSV and retain only requested ranks.
        with (tables / "observations.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                tier = int(r["tier"])
                if not all_tiers and tier not in tiers:
                    continue
                t = timestamp(r["time_utc"])
                try:
                    s = float(r["ep"])
                except ValueError:
                    continue
                if t is None or not math.isfinite(s) or s < 0:
                    continue
                available = timestamp(r.get("available_at_utc")) or t
                grouped.setdefault((int(r["event_id"]), tier), []).append((t, s, available))
        curves = {}
        for key, records in grouped.items():
            a = np.asarray(sorted(records, key=lambda r: r[0]), dtype=float)
            curves[key] = Curve(a[:, 0], a[:, 1], a[:, 2])
        if all_tiers or tiers.intersection({1,10}):
            # Prefer Bestdori when both providers have a top-ten resource.
            for key in sorted(resources,reverse=True):
                parts = key.split('/')
                if len(parts)==3 and parts[1]=='top10':
                    for tier, curve in top10_curves(raw_payload(key)).items():
                        if all_tiers or tier in tiers:
                            curves[(int(parts[2]),tier)] = curve
        labels, label_sources = {}, {}
        for r in read_rows(tables / "final_labels.csv"):
            if r.get("label_status") == "archived_final" and (all_tiers or int(r["tier"]) in tiers):
                score = float(r["final_ep"])
                if math.isfinite(score) and score > 0:
                    labels[(int(r["event_id"]), int(r["tier"]))] = score
                    label_sources[(int(r["event_id"]), int(r["tier"]))] = "archived_final"
        for r in read_rows(root / "annotations" / "verified_finals.csv"):
            if not r.get("event_id"):
                continue
            if not r.get("source_url", "").strip():
                raise ValueError("Manually verified final scores need a source_url")
            score = float(r["final_ep"])
            if not math.isfinite(score) or score <= 0:
                raise ValueError("Invalid manually verified final score")
            if all_tiers or int(r["tier"]) in tiers:
                labels[(int(r["event_id"]), int(r["tier"]))] = score
                label_sources[(int(r["event_id"]), int(r["tier"]))] = "manually_verified"
        archive = raw_payload('bestdori/master/archives') or {}
        for eid,tier in curves:
            if (eid,tier) not in labels:
                final = archive_score(archive,eid,tier)
                if final is not None:
                    labels[(eid,tier)] = float(final)
                    label_sources[(eid,tier)] = 'archived_final'
        outages, revisions = [], []
        for r in read_rows(root / "annotations" / "outages.csv"):
            if not r.get("event_id"):
                continue
            values = [timestamp(r.get(k)) for k in ("start_at", "end_at", "known_at")]
            if any(v is None for v in values) or values[1] <= values[0] or not r.get("source_url"):
                raise ValueError("Outages require start/end/known_at with timezone and a source_url")
            outages.append((int(r["event_id"]), *values))
        for r in read_rows(root / "annotations" / "schedule_revisions.csv"):
            if not r.get("event_id"):
                continue
            known = timestamp(r.get("known_at"))
            start, end = timestamp(r.get("start_at")), timestamp(r.get("end_at"))
            if known is None or start is None or end is None or end <= start or not r.get("source_url"):
                raise ValueError("Schedule revisions need a complete start/end/known_at and a source_url")
            revisions.append((int(r["event_id"]), known, start, end))
        return cls(events, curves, labels, outages, revisions,
                   {"bundle": str(root.resolve()), "collected_at_utc": manifest.get("collected_at_utc"),
                    "availability_assumption": manifest.get("availability_assumption")}, label_sources)

    def add_terminal_labels(self, config, as_of=None):
        """Opt-in proxy targets require an actual observation at/after closing.

        A pre-close tail never qualifies, regardless of how near the end it is.
        Archived/manually verified finals always take precedence over a proxy.
        """
        if not config.allow_terminal_labels:
            return 0
        as_of = time.time() if as_of is None else as_of
        added = 0
        for key, curve in self.curves.items():
            if key in self.labels or key[0] not in self.events:
                continue
            e = self.events[key[0]]
            if self.label_available(e.event_id, config) > as_of:
                continue
            eligible = ((curve.times >= e.end) & (curve.times <= e.end+config.terminal_window_minutes*60)
                        & (curve.available <= as_of) & (curve.scores > 0))
            if eligible.any():
                index = np.flatnonzero(eligible)[-1]
                self.labels[key] = float(curve.scores[index])
                self.label_sources[key] = "post_close_tracker_proxy"
                self.final_available[e.event_id] = max(self.final_available.get(e.event_id,0),float(curve.available[index]))
                added += 1
        return added

    def known_event(self, event, cutoff):
        versions = sorted([r for r in self.revisions if r[0] == event.event_id], key=lambda r: r[1])
        if not versions:
            return event, False
        known = [r for r in versions if r[1] <= cutoff]
        if not known:
            # Using a later extension before its announcement would leak.
            raise ValueError(f"Event {event.event_id} has no schedule version known by this checkpoint; add its initial schedule")
        _, _, start, end = known[-1]
        return replace(event, start=start, end=end), True

    def downtime(self, event, cutoff):
        intervals = sorted((max(event.start, start), min(end, cutoff))
                           for eid, start, end, known in self.outages
                           if eid == event.event_id and known <= cutoff and start < cutoff)
        total, right = 0.0, event.start
        for a, b in intervals:
            total += max(0.0, b - max(a, right))
            right = max(right, b)
        return total / 3600

    def label_available(self, event_id, config):
        return max(self.events[event_id].end + config.final_availability_delay_hours * 3600,
                   self.final_available.get(event_id,0))

    def eligible_events(self, config):
        ids = {eid for eid, tier in self.labels if tier in config.tiers and (eid, tier) in self.curves}
        return sorted((self.events[i] for i in ids if i in self.events), key=lambda e: (e.start, e.event_id))
