"""Approximate Stack error scales from event-isolated residuals, no point refit.

Cells average squared errors within each event before applying 730-day decay.
Sparse horizon cells widen to adjacent bins; tiers are never borrowed.
"""
import math

from collect_bandori_data import timestamp


EDGES = (0, 24, 48, 72, 96, 144, 192, math.inf)
METHOD = 'stack_event_loo_rms'


def bucket(hours):
    if not math.isfinite(hours) or hours < 0:
        raise ValueError('Invalid forecast horizon')
    return next(i for i, end in enumerate(EDGES[1:]) if hours < end)


def cell(records, fit_cutoff, tau=730):
    groups = {}
    for row in records:
        p, y = row['prediction'], row['actual']
        if not (math.isfinite(p) and math.isfinite(y) and p > 0 and y > 0):
            raise ValueError('Invalid calibration prediction or outcome')
        groups.setdefault(row['event_id'], []).append(row)
    if not groups:
        return None
    first, second, weights = [], [], []
    for rows in groups.values():
        errors = [(r['prediction'] - r['actual']) / r['prediction'] for r in rows]
        first.append(sum(errors) / len(errors))
        second.append(sum(e * e for e in errors) / len(errors))
        weights.append(math.exp(-max(0, fit_cutoff - rows[0]['event_end']) / 86400 / tau))
    mass = sum(weights)
    mean = sum(w * e for w, e in zip(weights, first)) / mass
    mse = sum(w * e for w, e in zip(weights, second)) / mass
    return dict(relative_rmse=math.sqrt(mse), relative_bias=mean,
                centered_relative_sd=math.sqrt(max(0, mse - mean * mean)),
                events=len(groups), rows=len(records), event_ids=sorted(groups),
                effective_events=mass * mass / sum(w * w for w in weights))


def build_cells(records, tiers, fit_cutoff, minimum_events=4):
    cells = {}
    for tier in tiers:
        rows = [r for r in records if r['tier'] == tier]
        for h in range(len(EDGES) - 1):
            for radius in range(len(EDGES) - 1):
                lo, hi = max(0, h - radius), min(len(EDGES) - 2, h + radius)
                chosen = [r for r in rows if lo <= bucket(r['hours_remaining']) <= hi]
                c = cell(chosen, fit_cutoff)
                if c and c['events'] >= minimum_events:
                    c.update(source_buckets=list(range(lo, hi + 1)),
                             horizon_fallback=radius > 0,
                             observed_hours_range=[min(r['hours_remaining'] for r in chosen),
                                                   max(r['hours_remaining'] for r in chosen)])
                    cells[f'{tier}:{h}'] = c
                    break
    return cells


def estimate(calibration, tier, hours, point, score_age_hours=0, duration_hours=None):
    if not math.isfinite(point) or point <= 0:
        raise ValueError('Invalid point forecast')
    h = bucket(hours)
    if score_age_hours is None or score_age_hours > calibration['max_score_age_hours']:
        return None  # No outage/stale-input residuals were generated for this calibration.
    c = calibration['cells'].get(f'{tier}:{h}')
    if c is None:
        return None
    relative = c['relative_rmse']
    lo, hi = c['observed_hours_range']
    durations = calibration['event_duration_range_hours']
    outside_duration = (duration_hours is not None and
                        not durations[0] <= duration_hours <= durations[1])
    return dict(value=point * relative, relative=relative, method=METHOD,
                relative_bias=c['relative_bias'], centered_relative_sd=c['centered_relative_sd'],
                events=c['events'], effective_events=c['effective_events'], rows=c['rows'],
                source_tier=tier, horizon_bucket=h, source_buckets=c['source_buckets'],
                horizon_fallback=c['horizon_fallback'], observed_hours_range=[lo, hi],
                outside_horizon_support=not lo <= hours <= hi,
                outside_duration_support=outside_duration,
                approximate=True, coverage_guaranteed=False)


def add_uncertainty(forecast, calibration):
    if timestamp(forecast['checkpoint_cn']) < calibration['available_at']:
        raise ValueError('Stack calibration was unavailable at this issue time')
    if forecast['source_model_sha256'] != calibration['source_model_sha256']:
        raise ValueError('Stack uncertainty belongs to a different frozen model')
    for row in forecast['predictions']:
        if row['status'] == 'ok':
            row['sigma'] = estimate(calibration, row['tier'], forecast['hours_remaining'],
                                    row['final'], row.get('score_age_hours'),
                                    forecast.get('event_duration_hours'))
    forecast['uncertainty'] = dict(method=METHOD, formula=calibration['formula'],
        calibration_event_ids=calibration['calibration_event_ids'],
        limitations=calibration['limitations'])
    return forecast
