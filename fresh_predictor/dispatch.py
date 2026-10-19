"""Duration-based selection of the two frozen NumPy predictors."""
from datetime import datetime, timezone
import json
from pathlib import Path

from collect_bandori_data import CN
from .data import Policy, restrict
from .runtime import Predictor, sha


THRESHOLD_HOURS = 240.0


def select_predictor(event, requested='auto'):
    """Use total known scheduled duration, never remaining forecast hours."""
    requested = requested.strip().lower() or 'auto'
    if requested not in ('auto', 'fresh', 'ridge'):
        raise ValueError('模型应为 auto、fresh 或 ridge')
    hours = (event.end - event.start) / 3600
    selected = ('ridge' if hours > THRESHOLD_HOURS else 'fresh') if requested == 'auto' else requested
    return dict(requested=requested, selected=selected, event_duration_hours=hours,
        threshold_hours=THRESHOLD_HOURS, rule='ridge if total scheduled duration > 240 hours; otherwise fresh',
        schedule_basis='Schedule known at the data cutoff; not hours remaining',
        manual_override=requested != 'auto')


class ForecastRouter:
    def __init__(self, fresh_directory, ridge_directory):
        self.fresh_directory, self.ridge_directory = Path(fresh_directory), Path(ridge_directory)
        values = json.loads((self.fresh_directory / 'policy.json').read_text())
        values['tiers'], values['checkpoint_hours'] = tuple(values['tiers']), tuple(values['checkpoint_hours'])
        self.policy = Policy(**values)
        self._fresh = self._ridge = None

    def predict(self, live, event_id, cutoff, max_stale_hours=None, requested='auto'):
        # Remove target labels and unrelated live histories for either implementation.
        target = restrict(live, [event_id], labels=False)
        event = target.known_event(target.events[event_id], cutoff)[0]
        routing = select_predictor(event, requested)
        if routing['selected'] == 'fresh':
            if self._fresh is None:
                self._fresh = Predictor(self.fresh_directory)
            result = self._fresh.predict(target, event_id, cutoff, max_stale_hours)
        else:
            from bandori.forecast_uncertainty import add_uncertainty
            from bandori.ridge_runtime import RidgePredictor
            if self._ridge is None:
                self._ridge = RidgePredictor(self.ridge_directory)
            result = self._ridge.predict(target, event_id, cutoff, max_stale_hours)
            calibration = json.loads((self.ridge_directory / 'sigma.json').read_text())
            add_uncertainty(result, calibration)
            result.update(closes_cn=result['event_end_cn'],
                generated_at_utc=datetime.now(timezone.utc).isoformat(),
                model_sha256=sha(self.ridge_directory / 'deployment.json'),
                supervised_events=result['training_events'], reference_events=result['training_events'])
            for row in result['predictions']:
                if row['status'] == 'ok':
                    row['trained'] = row['supervised_training_targets'] > 0
        result.update(selected_predictor=routing['selected'], routing=routing,
            event_duration_hours=routing['event_duration_hours'],
            known_start_cn=datetime.fromtimestamp(event.start, CN).isoformat())
        return result
