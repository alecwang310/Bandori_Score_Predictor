"""Causal historical completion ratios and reward-role fallback."""
from dataclasses import dataclass
from datetime import datetime
import math
import numpy as np
from .api import CN, iso
from .rules import era

HORIZONS = ('0-24h', '24-48h', '48-72h', '72-96h', '96h+')

FEATURES = ('progress', 'remaining_days', 'duration_days', 'staleness_hours', 'log_current', 'current_fraction', 'future_fraction', 'history_spread', 'history_effective_n', 'rate_6h', 'rate_12h', 'rate_24h', 'clock_sin', 'clock_cos', 'weekend')

class TierUnavailable(ValueError):
    """Expected per-tier unavailability, distinct from programming errors."""

def horizon(hours):
    return min(4, int(max(0, hours) // 24))

def weighted_quantile(values, weights, q=0.5):
    values, weights = (np.asarray(values), np.asarray(weights))
    order = np.argsort(values, kind='stable')
    return float(np.interp(q * np.sum(weights), np.cumsum(weights[order]) - 0.5 * weights[order], values[order]))

def reference_weight(base, same_event_type, duration, reference_duration, proxy_label, proxy_label_weight):
    """Event-type, duration-ratio and proxy-label factors on an age/rule weight.

    ``base`` is the value from ``Config.event_weight`` (age decay times the
    post-rule multiplier); the multiplications below keep their original order.
    """
    weight = base
    weight *= 2 if same_event_type else 1
    weight *= math.exp(-abs(math.log(duration / reference_duration)) / 0.5)
    if proxy_label:
        weight *= proxy_label_weight
    return weight

def active_hours(data, event, start, end, known_at):
    intervals = sorted(((max(start, a), min(end, b)) for eid, a, b, k in data.outages if eid == event.event_id and k <= known_at and (a < end) and (b > start)))
    down, right = (0.0, start)
    for a, b in intervals:
        down += max(0, b - max(a, right))
        right = max(right, b)
    return max(0, (end - start - down) / 3600)

@dataclass
class Case:
    event_id: int
    tier: int
    cutoff: float
    current: float
    observed_at: float
    remaining_hours: float
    baseline: float
    pace: float
    features: dict
    references: tuple
    pace_references: int
    schedule_versioned: bool

class AnalogPredictor:

    def __init__(self, data, config, training_ids):
        self.data, self.config = (data, config)
        self.training_ids = set(training_ids)
        self.cache = {}

    def reference(self, eid, tier, known_at):
        key = (eid, tier, known_at)
        if key not in self.cache:
            event = self.data.events[eid]
            curve = self.data.curves[eid, tier].prefix(known_at, event.start)
            mask = curve.times < event.end
            times, scores = (curve.times[mask], curve.scores[mask])
            if not len(times) or times[0] > event.start:
                times = np.r_[event.start, times]
                scores = np.r_[0.0, scores]
            times = np.r_[times, event.end]
            scores = np.r_[scores, self.data.labels[eid, tier]]
            self.cache[key] = (times, scores)
        return self.cache[key]

    def sample(self, eid, tier, t, known_at):
        times, scores = self.reference(eid, tier, known_at)
        i = int(np.searchsorted(times, t))
        if i < len(times) and abs(times[i] - t) < 0.001:
            return float(scores[i])
        if i == 0 or i == len(times) or times[i] - times[i - 1] > self.config.max_reference_gap_hours * 3600:
            return None
        return float(np.interp(t, times[i - 1:i + 1], scores[i - 1:i + 1]))

    def recent_rates(self, event, curve, observed_at, current, cutoff):
        rates, windows = ({}, {})
        prefix = curve.prefix(observed_at, event.start)
        for hours in (6, 12, 24):
            point = prefix.last(observed_at - hours * 3600, event.start)
            if point is None or observed_at - hours * 3600 - point[0] > min(3, hours / 4) * 3600:
                continue
            segment = prefix.times[(prefix.times >= point[0]) & (prefix.times <= observed_at)]
            if len(segment) < 2 or np.max(np.diff(segment)) > min(12, hours / 2) * 3600:
                continue
            exposure = active_hours(self.data, event, point[0], observed_at, cutoff)
            if exposure > 0 and current > point[1]:
                rates[hours] = (current - point[1]) / exposure
                windows[hours] = (observed_at - point[0]) / 3600
        return (rates, windows)

    def build(self, event, tier, cutoff):
        cfg = self.config
        event, versioned = self.data.known_event(event, cutoff)
        if not event.start < cutoff < event.end:
            raise TierUnavailable('Checkpoint outside the known event schedule')
        curve = self.data.curves.get((event.event_id, tier))
        point = curve.last(cutoff, event.start) if curve is not None else None
        if point is None or point[1] <= 0:
            raise TierUnavailable('No positive score known at this checkpoint')
        observed, current = point
        if cutoff - observed > cfg.max_stale_minutes * 60:
            raise TierUnavailable('Latest score is stale')
        duration = event.end - event.start
        progress = (observed - event.start) / duration
        remaining = (event.end - observed) / 3600
        rates, windows = self.recent_rates(event, curve, observed, current, cutoff)
        forecasts, weights, ids, pace_values, pace_weights = ([], [], [], [], [])
        for eid in sorted(self.training_ids):
            key = (eid, tier)
            if eid == event.event_id or key not in self.data.labels or key not in self.data.curves:
                continue
            reference = self.data.events[eid]
            if self.data.label_available(eid, cfg) > event.start:
                continue
            final = self.data.labels[key]
            matched = reference.start + progress * (reference.end - reference.start)
            score = self.sample(eid, tier, matched, event.start)
            if score is None or not 0 < score <= final:
                continue
            age = (event.start - reference.end) / 86400
            weight = reference_weight(cfg.event_weight(age, reference.start), event.event_type == reference.event_type, duration, reference.end - reference.start, self.data.label_sources.get(key) == 'post_close_tracker_proxy', cfg.proxy_label_weight)
            forecasts.append(current * final / score)
            weights.append(weight)
            ids.append(eid)
            end = reference.end - remaining * 3600
            level = self.sample(eid, tier, end, event.start)
            possibilities = []
            if level is not None and 0 <= level < final:
                for hours, rate in rates.items():
                    start = end - windows[hours] * 3600
                    before = self.sample(eid, tier, start, event.start)
                    exposure = active_hours(self.data, reference, start, end, event.start)
                    if before is not None and level > before and (exposure > 0):
                        historical_rate = (level - before) / exposure
                        possibilities.append(current + rate * (final - level) / historical_rate)
            if possibilities:
                pace_values.append(float(np.median(possibilities)))
                pace_weights.append(weight)
        if len(ids) < cfg.min_reference_events:
            raise TierUnavailable(f'Only {len(ids)} usable earlier training references')
        baseline = max(current, weighted_quantile(forecasts, weights))
        pace = max(current, weighted_quantile(pace_values, pace_weights)) if len(pace_values) >= cfg.min_reference_events else baseline
        local = datetime.fromtimestamp(cutoff, CN)
        effective = sum(weights) ** 2 / sum((w * w for w in weights))
        spread = (weighted_quantile(forecasts, weights, 0.75) - weighted_quantile(forecasts, weights, 0.25)) / baseline
        features = dict(progress=progress, remaining_days=remaining / 24, duration_days=duration / 86400, staleness_hours=(cutoff - observed) / 3600, log_current=math.log1p(current) / 16, current_fraction=current / baseline, future_fraction=(baseline - current) / baseline, history_spread=spread, history_effective_n=math.log1p(effective), clock_sin=math.sin(2 * math.pi * local.hour / 24), clock_cos=math.cos(2 * math.pi * local.hour / 24), weekend=float(local.weekday() >= 5))
        features.update({f'rate_{h}h': rates.get(h, 0) * 24 / baseline for h in (6, 12, 24)})
        return Case(event.event_id, tier, cutoff, current, observed, (event.end - cutoff) / 3600, baseline, pace, features, tuple(ids), len(pace_values), versioned)

def reconcile(cases, predictions):
    values = np.maximum(np.asarray(predictions, float), [c.current for c in cases])
    groups = {}
    for i, c in enumerate(cases):
        groups.setdefault((c.event_id, c.cutoff), []).append(i)
    for indices in groups.values():
        floor = 0.0
        for i in sorted(indices, key=lambda i: cases[i].tier, reverse=True):
            values[i] = max(values[i], floor)
            floor = values[i]
    return values

class CachedAnalog(AnalogPredictor):
    """Share immutable completed curves without dropping publication filters."""

    def __init__(self, *args):
        super().__init__(*args)
        self.availability = {}

    def reference(self, eid, tier, known_at):
        key = (eid, tier)
        if key not in self.availability:
            curve = self.data.curves[key]
            event = self.data.events[eid]
            relevant = (curve.times >= event.start) & (curve.times < event.end)
            self.availability[key] = float(np.max(curve.available[relevant])) if relevant.any() else 0.0
        return super().reference(eid, tier, float('inf') if known_at >= self.availability[key] else known_at)

class RoleAwareAnalog(CachedAnalog):

    def __init__(self, *args):
        super().__init__(*args)
        self.role_audit = {}

    def build(self, event, tier, cutoff):
        try:
            return super().build(event, tier, cutoff)
        except TierUnavailable as error:
            if tier != 1500 or not era(event) or (not str(error).startswith('Only ')):
                raise
        config = self.config
        event, versioned = self.data.known_event(event, cutoff)
        curve = self.data.curves.get((event.event_id, tier))
        point = curve.last(cutoff, event.start) if curve is not None else None
        if point is None or point[1] <= 0:
            raise TierUnavailable('No positive T1500 score')
        observed, current = point
        if cutoff - observed > config.max_stale_minutes * 60:
            raise TierUnavailable('Latest score is stale')
        duration = event.end - event.start
        progress = (observed - event.start) / duration
        remaining = (event.end - observed) / 3600
        rates, windows = self.recent_rates(event, curve, observed, current, cutoff)
        predictions = []
        weights = []
        pace = []
        pace_weights = []
        direct = []
        roles = []
        for eid in sorted(self.training_ids):
            ref = self.data.events[eid]
            if eid == event.event_id or self.data.label_available(eid, config) > event.start:
                continue
            if ref.start < event.start - 730 * 86400:
                continue
            source_tier = 1500 if era(ref) else 1000
            key = (eid, source_tier)
            if key not in self.data.labels or key not in self.data.curves:
                continue
            final = self.data.labels[key]
            value = self.sample(eid, source_tier, ref.start + progress * (ref.end - ref.start), event.start)
            if value is None or not 0 < value <= final:
                continue
            weight = reference_weight(config.event_weight((event.start - ref.end) / 86400, ref.start), event.event_type == ref.event_type, duration, ref.end - ref.start, self.data.label_sources.get(key) == 'post_close_tracker_proxy', config.proxy_label_weight)
            predictions.append(current * final / value)
            weights.append(weight)
            roles.append((eid, source_tier))
            if source_tier == 1500:
                direct.append(eid)
            anchor = ref.end - remaining * 3600
            level = self.sample(eid, source_tier, anchor, event.start)
            possibilities = []
            if level is not None and 0 <= level < final:
                for hours, rate in rates.items():
                    start = anchor - windows[hours] * 3600
                    before = self.sample(eid, source_tier, start, event.start)
                    exposure = active_hours(self.data, ref, start, anchor, event.start)
                    if before is not None and level > before and (exposure > 0):
                        possibilities.append(current + rate * (final - level) / ((level - before) / exposure))
            if possibilities:
                pace.append(float(sorted(possibilities)[len(possibilities) // 2]))
                pace_weights.append(weight)
        if len(predictions) < config.min_reference_events:
            raise TierUnavailable('Insufficient direct or reward-role references')
        baseline = max(current, weighted_quantile(predictions, weights))
        pace_value = max(current, weighted_quantile(pace, pace_weights)) if len(pace) >= config.min_reference_events else baseline
        local = datetime.fromtimestamp(cutoff, CN)
        effective = sum(weights) ** 2 / sum((w * w for w in weights))
        features = dict(progress=progress, remaining_days=remaining / 24, duration_days=duration / 86400, staleness_hours=(cutoff - observed) / 3600, log_current=math.log1p(current) / 16, current_fraction=current / baseline, future_fraction=(baseline - current) / baseline, history_spread=(weighted_quantile(predictions, weights, 0.75) - weighted_quantile(predictions, weights, 0.25)) / baseline, history_effective_n=math.log1p(effective), clock_sin=math.sin(2 * math.pi * local.hour / 24), clock_cos=math.cos(2 * math.pi * local.hour / 24), weekend=float(local.weekday() >= 5), role_transfer_support=sum((t == 1000 for e, t in roles)), direct_tier_references=len(direct))
        features.update({f'rate_{h}h': rates.get(h, 0) * 24 / baseline for h in (6, 12, 24)})
        self.role_audit[event.event_id, tier, cutoff] = roles
        return Case(event.event_id, tier, cutoff, current, observed, (event.end - cutoff) / 3600, baseline, pace_value, features, tuple(direct), len(pace), versioned)
