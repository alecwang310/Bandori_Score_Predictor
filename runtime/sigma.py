"""Apply frozen per-tier uncertainty scales; no calibration fitting."""
import math

METHOD = 'loglinear_two_clock_rms'

def _number(value, name):
    if value is None or not math.isfinite(value):
        raise ValueError('Invalid ' + name)
    return float(value)

def estimate(calibration, tier, hours_remaining, point, duration_hours=None, score_age_hours=0):
    """Sigma for one tier/timestamp.  Returns None when duration is unknown."""
    tiers = calibration.get('sigma_tiers')
    borrowed = None
    if tiers is not None and tier not in tiers:
        source = (calibration.get('top_rank_fallback') or {}).get(str(tier))
        if source is None:
            return None
        borrowed = int(source)
    point = _number(point, 'point forecast')
    if point <= 0:
        raise ValueError('Invalid point forecast')
    hours_remaining = _number(hours_remaining, 'remaining time')
    if hours_remaining < 0:
        raise ValueError('Invalid remaining time')
    if duration_hours is None:
        return None
    duration_hours = _number(duration_hours, 'event duration')
    elapsed = max(0.0, duration_hours - hours_remaining)
    coef = calibration['coefficients']
    log_mse = float(coef['intercept']) + float(coef['log1p_elapsed']) * math.log1p(elapsed) + float(coef['log1p_remaining']) * math.log1p(hours_remaining)
    relative = math.sqrt(math.exp(log_mse))
    relative = min(max(relative, float(calibration.get('min_relative', 0.01))), float(calibration.get('max_relative', 1.5)))
    result = dict(value=point * relative, relative_rmse=relative, relative=relative, method=calibration.get('method', METHOD), formula=calibration.get('formula'), coefficients=dict(coef), elapsed_hours=elapsed, remaining_hours=hours_remaining, events=calibration.get('calibration_events'), rows=calibration.get('calibration_rows'), horizon_fallback=False, approximate=True, coverage_guaranteed=False, source_tier=borrowed if borrowed is not None else tier, borrowed_from_tier=borrowed, outside_duration_support=False, outside_horizon_support=False, outside_elapsed_support=False)
    duration_range = calibration.get('event_duration_range_hours')
    if duration_range and (not duration_range[0] <= duration_hours <= duration_range[1]):
        result['outside_duration_support'] = True
    remaining_range = calibration.get('remaining_range_hours')
    if remaining_range and (not remaining_range[0] <= hours_remaining <= remaining_range[1]):
        result['outside_horizon_support'] = True
    elapsed_range = calibration.get('elapsed_range_hours')
    if elapsed_range and (not elapsed_range[0] <= elapsed <= elapsed_range[1]):
        result['outside_elapsed_support'] = True
    return result
