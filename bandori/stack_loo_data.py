"""Outcome-isolated LOO folds for the stack's exact 600-column feature path."""
from dataclasses import asdict
import gzip
import json
import math
from types import SimpleNamespace

import numpy as np

from fresh_predictor.data import restrict
from .analog import horizon, reconcile
from .dataset import Dataset
from .joint import pack
from .pipeline import training_arrays
from .representations import RICH
from .role_analog import build_cases
from .rules import era, packed_metadata


def combine(history, target):
    return Dataset([*history.events.values(), *target.events.values()],
        {**history.curves, **target.curves}, dict(history.labels),
        history.outages+target.outages, history.revisions+target.revisions,
        dict(target.provenance), dict(history.label_sources), dict(history.final_available))


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
        support[g, t, :2] = [math.log1p(c.features.get('role_transfer_support', 0)),
                            math.log1p(c.features.get('direct_tier_references', len(c.references)))]
        support[g, t, 2] = sum(era(data.events[e]) == era(data.events[c.event_id]) for e in c.references)/max(1, len(c.references))
    rules = np.concatenate((metadata, support), axis=2)
    return np.ascontiguousarray(np.column_stack((explicit.reshape(n, -1), rules.reshape(n, -1)))), index, bins


class FeatureScale:
    def fit(self, raw, weights, tiers):
        k = len(tiers)
        # Preserve the legacy wide C-order summation layout. Unselected PCs
        # and daily columns have no effect on the chosen explicit columns.
        selected = np.array([t*(len(RICH)+1)+i for t in range(k)
            for i in (RICH.index('progress'), RICH.index('remaining_days'), RICH.index('current_fraction'), len(RICH))])
        wide = np.zeros((len(raw), k*(len(RICH)+1)))
        wide[:, selected] = raw[:, :4*k]
        w = weights.sum(axis=1); w = w/w.sum()
        means = (wide*w[:, None]).sum(axis=0)
        scales = np.maximum(.05, np.sqrt(((wide-means)**2*w[:, None]).sum(axis=0)))
        self.mean, self.scale, self.explicit_width = means[selected], scales[selected], 4*k
        values = self._transform(raw)
        self.mask = (np.ptp(values[weights.sum(axis=1)>0], axis=0)>1e-8).astype(float)
        return self

    def _transform(self, raw):
        return np.column_stack((np.clip((raw[:, :self.explicit_width]-self.mean)/self.scale, -6, 6),
                                raw[:, self.explicit_width:]))

    def transform(self, raw):
        return np.ascontiguousarray(self._transform(raw)*self.mask)


def check_references(cases, data, config, forbidden=()):
    forbidden = set(forbidden)
    for c in cases:
        if c.event_id in forbidden:
            raise ValueError('Held event in training cases')
        for e in c.references:
            if e in forbidden or e == c.event_id or data.label_available(e, config)>data.events[c.event_id].start:
                raise ValueError('Invalid training reference')


def train_cases(data, config, supervised, references, base_cases, held=None):
    """Reuse only cases strictly before held outcomes could become available.

    Later cases are rebuilt with held curves, metadata and labels physically
    absent. This also rebuilds cross-rank reward priors not listed in references.
    """
    forbidden = [] if held is None else [held]
    history = restrict(data, [e for e in references if e not in forbidden])
    supervised = [e for e in supervised if e not in forbidden]
    if held is None:
        cases = [c for c in base_cases if c.event_id in supervised]
        rebuilt = []
    else:
        available = data.label_available(held, config)
        rebuilt = [e for e in supervised if data.events[e].start >= available]
        safe = set(supervised)-set(rebuilt)
        cases = [c for c in base_cases if c.event_id in safe]
        changed, _, _ = build_cases(history, config, rebuilt, list(history.events))
        cases += changed
    cases.sort(key=lambda c:(history.events[c.event_id].start, c.event_id, c.cutoff, c.tier))
    check_references(cases, history, config, forbidden)
    return history, cases, rebuilt


def prepare(data, config, supervised, references, base_cases, fit_cutoff, held=None):
    history, cases, rebuilt = train_cases(data, config, supervised, references, base_cases, held)
    raw, index, bins = raw_features(cases, history, config)
    targets, weights = training_arrays(cases, index, history, config, fit_cutoff)
    scale = FeatureScale().fit(raw, weights, config.tiers)
    result = SimpleNamespace(cases=cases, x=scale.transform(raw), index=index, bins=bins,
        targets=targets, weights=weights, scale=scale, rebuilt_event_ids=rebuilt,
        reference_event_ids=list(history.events))
    if held is not None:
        target = restrict(data, [held], labels=False)
        valid_data = combine(history, target)
        valid, failed, _ = build_cases(valid_data, config, [held], list(history.events))
        raw, idx, bins = raw_features(valid, valid_data, config)
        result.valid_cases, result.valid_x, result.valid_index = valid, scale.transform(raw), idx
        result.valid_bins, result.valid_failures = bins, failed
    return result


def prediction(cases, index, delta):
    values = np.array([c.baseline for c in cases])
    for g, t in zip(*np.where(index >= 0)):
        i = index[g, t]; c = cases[i]
        values[i] = c.current+max(0, c.baseline-c.current)*np.exp(delta[g, t])
    return reconcile(cases, values)


def save_cases(path, cases):
    path.write_bytes(gzip.compress(json.dumps([asdict(c) for c in cases], separators=(',', ':'), default=float).encode(), mtime=0))


def load_cases(path):
    from .analog import Case
    return [Case(**dict(c, references=tuple(c['references']))) for c in json.loads(gzip.decompress(path.read_bytes()))]
