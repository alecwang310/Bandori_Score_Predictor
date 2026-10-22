"""Frozen input transforms and forecast reconciliation; no fitted transforms."""
import math
import numpy as np
from .analog import FEATURES, horizon, reconcile
from .rules import era, packed_metadata

def pack(cases, tiers):
    groups = {}
    for case in cases:
        groups.setdefault((case.event_id, case.cutoff), len(groups))
    width = len(FEATURES) + 1
    x = np.zeros((len(groups), len(tiers) * width))
    index = np.full((len(groups), len(tiers)), -1, int)
    bins = np.zeros(len(groups), int)
    positions = {tier: i for i, tier in enumerate(tiers)}
    for i, c in enumerate(cases):
        g, t = (groups[c.event_id, c.cutoff], positions[c.tier])
        x[g, t * width:(t + 1) * width] = [c.features[name] for name in FEATURES] + [1]
        index[g, t] = i
        bins[g] = horizon(c.remaining_hours)
    return (x, index, bins)

def raw_features(cases, data, config):
    """The selected explicit and metadata columns, before training-only scaling."""
    _, index, bins = pack(cases, config.tiers)
    n, k = index.shape
    explicit = np.zeros((n, k, 4))
    metadata = packed_metadata(cases, index, data)
    support = np.zeros((n, k, 3))
    for g, t in zip(*np.where(index >= 0)):
        c = cases[index[g, t]]
        explicit[g, t] = [c.features['progress'], c.features['remaining_days'], c.features['current_fraction'], 1]
        support[g, t, :2] = [math.log1p(c.features.get('role_transfer_support', 0)), math.log1p(c.features.get('direct_tier_references', len(c.references)))]
        support[g, t, 2] = sum((era(data.events[e]) == era(data.events[c.event_id]) for e in c.references)) / max(1, len(c.references))
    rules = np.concatenate((metadata, support), axis=2)
    return (np.ascontiguousarray(np.column_stack((explicit.reshape(n, -1), rules.reshape(n, -1)))), index, bins)

class FeatureScale:

    def _transform(self, raw):
        return np.column_stack((np.clip((raw[:, :self.explicit_width] - self.mean) / self.scale, -6, 6), raw[:, self.explicit_width:]))

    def transform(self, raw):
        return np.ascontiguousarray(self._transform(raw) * self.mask)

def prediction(cases, index, delta):
    values = np.array([c.baseline for c in cases])
    for g, t in zip(*np.where(index >= 0)):
        i = index[g, t]
        c = cases[i]
        values[i] = c.current + max(0, c.baseline - c.current) * np.exp(delta[g, t])
    return reconcile(cases, values)

def correction_forecast(cases, index, corrections, gain, kind='nn'):
    result = np.array([c.baseline for c in cases])
    for g, t in zip(*np.where(index >= 0)):
        i = index[g, t]
        c = cases[i]
        delta = gain * corrections[g, t]
        multiplier = math.exp(delta) if kind == 'nn' else 1 + delta
        result[i] = c.current + max(0, c.baseline - c.current) * multiplier
    return reconcile(cases, result)
