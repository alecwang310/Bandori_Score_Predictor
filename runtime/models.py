"""Frozen production model adapters."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json
import math
import numpy as np
from .api import CN, iso
from .data import restrict, load_history, combine, attach_bank, digest, asset_path
from .config import Config
from .analog import TierUnavailable, RoleAwareAnalog
from .core import JointRidge, MediumStack
from .features import pack, FeatureScale, raw_features, prediction, correction_forecast
from .rules import DECOR_START, EVENT_TYPES, RULE_START

class GlobalRidgePredictor:
    """Per-tier linear correction on 12 global features; no curve/MF inputs."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'deployment.json').read_text())
        if self.manifest.get('kind') != 'bandori_cn_global_ridge_v1':
            raise ValueError('Unknown global Ridge package')
        for name, expected in self.manifest['hashes'].items():
            if digest(asset_path(self.directory, name)) != expected:
                raise ValueError('Model checksum mismatch: ' + name)
        self.config = Config.load(asset_path(self.directory, 'config.json'))
        self.core = JointRidge.load(self.directory / 'ridge.npz')
        self.ids = self.manifest['training_event_ids']
        self.gain = float(self.manifest.get('gain', 1.0))

    @staticmethod
    def _design(cases, index, data):
        out = np.zeros((len(index), 12))
        for g in range(len(index)):
            valid = index[g][index[g] >= 0]
            if len(valid) == 0:
                continue
            c = cases[valid[0]]
            event = data.known_event(data.events[c.event_id], c.cutoff)[0]
            out[g] = [*[float(event.event_type == t) for t in EVENT_TYPES], float(event.start >= RULE_START), float(event.start >= DECOR_START), c.features['progress'], c.features['progress'] ** 2]
        return out

    def predict(self, live, event_id, cutoff, max_stale_hours=None):
        if event_id in self.ids or cutoff < self.manifest['fit_cutoff']:
            raise ValueError('Prediction overlaps the frozen training history')
        cfg = self.config
        if max_stale_hours is not None:
            if not math.isfinite(max_stale_hours) or max_stale_hours <= 0:
                raise ValueError('Invalid maximum score age')
            cfg = replace(cfg, max_stale_minutes=60 * max_stale_hours)
        data = attach_bank(self.directory, live)
        if event_id not in data.events:
            raise ValueError('Requested event has no CN schedule')
        event = data.known_event(data.events[event_id], cutoff)[0]
        if not event.start < cutoff < event.end:
            raise ValueError('Forecast time must be inside the event')
        analog = RoleAwareAnalog(data, cfg, self.ids)
        cases, failed = ([], [])
        for tier in cfg.tiers:
            try:
                cases.append(analog.build(event, tier, cutoff))
            except TierUnavailable as error:
                failed.append(dict(tier=tier, status='unavailable', reason=str(error)))
        if not cases:
            raise ValueError('本时刻没有可用分数线；请检查数据新鲜度或起报时间')
        _, index, _ = pack(cases, cfg.tiers)
        design = self._design(cases, index, data)
        delta = self.core.predict(design)
        values = correction_forecast(cases, index, delta, self.gain, 'ridge')
        details = {}
        for g, t in zip(*np.where(index >= 0)):
            i = index[g, t]
            c = cases[i]
            details[i] = dict(baseline=c.baseline, remaining_multiplier=1 + self.gain * float(delta[g, t]), raw_correction=float(delta[g, t]))
        rows = [dict(tier=c.tier, status='ok', final=math.ceil(values[i]), current=math.ceil(c.current), observed_at=iso(c.observed_at), score_age_hours=(cutoff - c.observed_at) / 3600, supervised_training_targets=self.manifest['supervised_targets_by_tier'][str(c.tier)], sigma=None, **details[i]) for i, c in enumerate(cases)]
        duration = (event.end - event.start) / 3600
        return dict(event_id=event_id, event_name=event.name, event_type=event.event_type, server='CN', server_id=3, checkpoint_cn=datetime.fromtimestamp(cutoff, CN).isoformat(), closes_cn=datetime.fromtimestamp(event.end, CN).isoformat(), hours_remaining=(event.end - cutoff) / 3600, event_duration_hours=duration, duration_exceeds_training_range=duration > max(self.manifest['training_durations_hours']), model='仅规则线性修正模型', feature_version=self.manifest.get('feature_version', 'global_v1'), source_model_sha256=self.manifest['source_model_sha256'], age_characteristic_days=cfg.characteristic_time_days, post_rule_weight=cfg.post_rule_change_multiplier, training_events=len(self.ids), supervised_events=len(self.manifest['supervised_event_ids']), predictions=sorted(rows + failed, key=lambda r: r['tier']))

class MFStackPredictor:

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'deployment.json').read_text())
        if self.manifest['kind'] != 'bandori_cn_mf_stack_v1':
            raise ValueError('Unknown MF Stack package')
        for name, expected in self.manifest['hashes'].items():
            if digest(asset_path(self.directory, name)) != expected:
                raise ValueError('Model checksum mismatch: ' + name)
        self.config = Config.load(asset_path(self.directory, 'config.json'))
        self.representation = self.manifest.get('representation', 'combined')
        sigma_path = self.directory / 'sigma.json'
        self.calibration = json.loads(sigma_path.read_text()) if sigma_path.exists() else None
        self.history = load_history(self.directory)
        self.ids = self.manifest['reference_event_ids']
        if set(self.history.events) != set(self.ids):
            raise ValueError('Reference bank differs from frozen deployment')
        self.models = [MediumStack.load(self.directory / f'seed_{s}.npz') for s in self.manifest['seeds']]
        self.base_scale = FeatureScale()
        with np.load(self.directory / 'input_scaling.npz', allow_pickle=False) as a:
            for key, attr in [('explicit_mean', 'mean'), ('explicit_scale', 'scale'), ('feature_mask', 'mask')]:
                setattr(self.base_scale, attr, a[key].copy())
        self.base_scale.explicit_width = 4 * len(self.config.tiers)
        if self.representation != 'zero_mf':
            raise ValueError('Unsupported deployed neural representation')

    def predict_cases(self, cases, live):
        """All case prefixes are filtered by publication time in the extractors."""
        if not cases:
            raise ValueError('No available prediction cases')
        if any((c.event_id in self.ids or c.cutoff < self.manifest['fit_cutoff'] for c in cases)):
            raise ValueError('Prediction overlaps the frozen training history')
        target = restrict(live, {c.event_id for c in cases}, labels=False)
        data = combine(self.history, target)
        raw, index, _ = raw_features(cases, data, self.config)
        base = self.base_scale.transform(raw)
        extra = np.zeros((len(base), 48 * len(self.config.tiers)))
        x = np.column_stack((base, extra))
        seeds = [prediction(cases, index, m.predict(x)) for m in self.models]
        return (np.mean(seeds, axis=0), seeds, x, index)

    def predict(self, live, event_id, cutoff, max_stale_hours=None):
        if event_id in self.ids or cutoff < self.manifest['fit_cutoff']:
            raise ValueError('Prediction overlaps the frozen training history')
        cfg = self.config
        if max_stale_hours is not None:
            if not math.isfinite(max_stale_hours) or max_stale_hours <= 0:
                raise ValueError('Maximum score age must be finite and positive')
            cfg = replace(cfg, max_stale_minutes=60 * max_stale_hours)
        target = restrict(live, [event_id], labels=False)
        data = combine(self.history, target)
        event = data.known_event(data.events[event_id], cutoff)[0]
        if not event.start < cutoff < event.end:
            raise ValueError('Forecast time must be inside the event')
        analog = RoleAwareAnalog(data, cfg, self.ids)
        cases, failed = ([], [])
        for tier in cfg.tiers:
            try:
                cases.append(analog.build(event, tier, cutoff))
            except TierUnavailable as error:
                failed.append(dict(tier=tier, status='unavailable', reason=str(error)))
        values, _, _, _ = self.predict_cases(cases, target)
        rows = [dict(tier=c.tier, status='ok', final=math.ceil(v), current=math.ceil(c.current), observed_at=iso(c.observed_at), score_age_hours=(cutoff - c.observed_at) / 3600, baseline=c.baseline, supervised_training_targets=self.manifest['supervised_targets_by_tier'][str(c.tier)], sigma=None) for c, v in zip(cases, values)]
        duration = (event.end - event.start) / 3600
        result = dict(event_id=event_id, event_name=event.name, event_type=event.event_type, server='CN', server_id=3, checkpoint_cn=datetime.fromtimestamp(cutoff, CN).isoformat(), closes_cn=datetime.fromtimestamp(event.end, CN).isoformat(), hours_remaining=(event.end - cutoff) / 3600, event_duration_hours=duration, duration_exceeds_training_range=duration > max(self.manifest['training_durations_hours']), model=self.manifest.get('kind', 'MLP8 + combined MF (two-seed ensemble)'), source_model_sha256=self.manifest.get('source_model_sha256', digest(self.directory / 'deployment.json')), age_characteristic_days=cfg.characteristic_time_days, post_rule_weight=cfg.post_rule_change_multiplier, training_events=len(self.ids), supervised_events=len(self.manifest['supervised_event_ids']), predictions=sorted(rows + failed, key=lambda r: r['tier']))
        return result
