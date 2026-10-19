"""NumPy-only deployment of the tuned joint NN correction (nn_pca family).

Mirrors ridge_runtime.RidgePredictor: identical feature layout (ridge_inputs
with clock=None), identical case building and row schema; only the core model
(JointNet) and the multiplier rule differ — the NN maps a bounded correction
through exp(gain*delta) instead of the ridge's linear 1+gain*delta.
"""
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN, iso
from .analog import reconcile
from .config import Config
from .joint import JointNet
from .pipeline import attach_bank, digest
from .representations import Compression
from .ridge_runtime import ridge_inputs
from .role_analog import RoleAwareAnalog


class JointPredictor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'deployment.json').read_text())
        if self.manifest.get('kind') != 'bandori_cn_joint_deployment_v1':
            raise ValueError('Unknown model package')
        for name, expected in self.manifest['hashes'].items():
            if digest(self.directory / name) != expected:
                raise ValueError('Model checksum mismatch: ' + name)
        self.config = Config.load(self.directory / 'config.json')
        self.core = JointNet.load(self.directory / 'joint_model.npz')
        self.compression = Compression.load(self.directory / 'compression.npz')
        with np.load(self.directory / 'input_masks.npz', allow_pickle=False) as a:
            self.mask = a['pca'].copy()

    def predict(self, live, event_id, cutoff, max_stale_hours=None):
        ids = self.manifest['training_event_ids']  # reference/analog pool, full 315
        config = self.config
        if event_id in ids:
            raise ValueError('This event was used in training; use an event after the frozen fit')
        if cutoff < self.manifest['fit_cutoff']:
            raise ValueError('Forecast precedes the frozen model fit')
        if max_stale_hours is not None:
            if not math.isfinite(max_stale_hours) or max_stale_hours <= 0:
                raise ValueError('Invalid maximum score age')
            config = config.replace(max_stale_minutes=max_stale_hours * 60)
        data = attach_bank(self.directory, live)
        if event_id not in data.events:
            raise ValueError('Requested event has no CN schedule')
        event = data.known_event(data.events[event_id], cutoff)[0]
        if not event.start < cutoff < event.end:
            raise ValueError('起报时间必须在活动开始与截活之间')
        analog = RoleAwareAnalog(data, config, ids)
        cases = []
        failed = []
        for tier in config.tiers:
            try:
                cases.append(analog.build(event, tier, cutoff))
            except ValueError as error:
                failed.append(dict(tier=tier, status='unavailable', reason=str(error)))
        if not cases:
            raise ValueError('本时刻没有可用分数线；请检查数据新鲜度或起报时间')
        x, index = ridge_inputs(cases, data, config, ids, self.compression, self.mask)
        delta = self.core.predict(x)
        values = np.array([c.baseline for c in cases])
        details = {}
        gain = self.manifest['gain']
        for g, t in zip(*np.where(index >= 0)):
            i = index[g, t]
            c = cases[i]
            multiplier = float(np.exp(gain * delta[g, t]))
            values[i] = c.current + max(0, c.baseline - c.current) * multiplier
            details[i] = dict(baseline=c.baseline, remaining_multiplier=multiplier,
                              raw_correction=float(delta[g, t]))
        values = reconcile(cases, values)
        rows = [dict(tier=c.tier, status='ok', final=math.ceil(values[i]), current=math.ceil(c.current),
                     observed_at=iso(c.observed_at), score_age_hours=(cutoff - c.observed_at) / 3600,
                     stale_input=(cutoff - c.observed_at) > self.config.max_stale_minutes * 60,
                     supervised_training_targets=self.manifest['supervised_targets_by_tier'][str(c.tier)],
                     **details[i])
                for i, c in enumerate(cases)]
        durations = self.manifest['training_durations_hours']
        hours = (event.end - event.start) / 3600
        return dict(event_id=event_id, event_name=event.name, event_type=event.event_type,
                    server='CN', server_id=3,
                    checkpoint_cn=datetime_fromtimestamp(cutoff), event_end_cn=datetime_fromtimestamp(event.end),
                    hours_remaining=(event.end - cutoff) / 3600, event_duration_hours=hours,
                    duration_exceeds_training_range=hours > max(durations),
                    near_duration_reference_events=sum(abs(d / hours - 1) <= .2 for d in durations),
                    model='JointNet pca (tuned) + 24h observer',
                    feature_version=self.manifest.get('feature_version', 'daily_v1'),
                    source_model_sha256=self.manifest['source_model_sha256'],
                    age_characteristic_days=config.characteristic_time_days,
                    post_rule_weight=config.post_rule_change_multiplier,
                    training_events=len(ids),
                    supervised_events=len(self.manifest['supervised_event_ids']),
                    predictions=sorted(rows + failed, key=lambda r: r['tier']))


def datetime_fromtimestamp(ts):
    from datetime import datetime
    return datetime.fromtimestamp(ts, CN).isoformat()
