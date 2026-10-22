"""Fixed rule-era and event-type inputs used by the shipped models."""
import math
import numpy as np
from .api import timestamp

RULE_DATE = '2026-05-26T00:00:00+08:00'

RULE_START = timestamp(RULE_DATE)

DECOR_START = timestamp('2026-04-07T00:00:00+08:00')

EVENT_TYPES = ('story', 'challenge', 'versus', 'mission_live', 'live_try', 'festival', 'medley', 'unknown')

FEATURES = ('new_reward_era', 'T2000_decoration_era', 'log_voiced_boundary', 'voiced_boundary', 'animated_boundary', 'flashing_boundary', 'decoration_boundary', 'log_rank', 'new_era_progress', 'voiced_boundary_progress') + tuple(('type_' + t for t in EVENT_TYPES))

def era(event):
    return int(event.start >= RULE_START)

def features(event, tier, cutoff):
    post = era(event)
    voice = 1500 if post else 1000
    decoration = float(event.start >= DECOR_START)
    p = np.clip((cutoff - event.start) / max(1, event.end - event.start), 0, 1)
    tag = event.event_type if event.event_type in EVENT_TYPES else 'unknown'
    return np.array([post, decoration, math.log(voice / 1000), float(tier == voice), float(post and tier == 500), float(post and tier == 1000), float(decoration and tier == 2000), math.log(tier) / 12, post * p, float(tier == voice) * p, *[float(tag == name) for name in EVENT_TYPES]], float)

def packed_metadata(cases, index, data):
    n, k = index.shape
    result = np.zeros((n, k, len(FEATURES)))
    for g, t in zip(*np.where(index >= 0)):
        c = cases[index[g, t]]
        event = data.known_event(data.events[c.event_id], c.cutoff)[0]
        result[g, t] = features(event, c.tier, c.cutoff)
    return result
