"""Select the two existing production configurations and their components."""
import json
from pathlib import Path
from .api import CN
from .config import Config
from .models import GlobalRidgePredictor
from .data import restrict, asset_path
from .sigma import estimate
from .formatting import API_VERSION, FOCUS_TIERS, LABELS

class Distribution:

    def __init__(self, root):
        self.root = Path(root)
        self.registry = json.loads((self.root / 'models/registry.json').read_text())
        self.policy = Config.load(asset_path(self.package_dir('ridge'), 'config.json'))
        self._models = {}

    def package_dir(self, name):
        if name not in self.registry['models']:
            raise ValueError('Unknown production configuration: ' + str(name))
        return self.root / 'models' / self.registry['models'][name]['package']

    def manifest(self, name):
        return json.loads((self.package_dir(name) / 'manifest.json').read_text())

    def select(self, event, requested=None):
        name = requested or self.registry['default']
        if name == 'all':
            name = 'both'
        if name != 'both':
            self.package_dir(name)
        return dict(routing=dict(rule='Two separately reported configurations; no duration switch', threshold=None, selected=name), model=dict(selection='default' if not requested else 'manual', reason='Fixed production configuration'), duration_hours=(event.end - event.start) / 3600)

    def _predictor(self, name):
        if name not in self._models:
            if name in ('ridge', 'previous_ridge'):
                self._models[name] = GlobalRidgePredictor(self.package_dir(name))
            elif name == 'stack_mf':
                from .models import MFStackPredictor
                self._models[name] = MFStackPredictor(self.package_dir(name))
                self._models[name].calibration = None
            else:
                raise ValueError('Configuration has no single predictor')
        return self._models[name]

    def _calibrate(self, name, result):
        manifest = self.manifest(name)
        sigma = json.loads((self.package_dir(name) / 'sigma.json').read_text())
        if sigma['source_model_sha256'] != manifest['point_source_sha256']:
            raise ValueError('Sigma belongs to another point model')
        for row in result['predictions']:
            if row['status'] != 'ok':
                continue
            cal = sigma['by_tier'].get(str(row['tier']))
            valid = cal and row['score_age_hours'] <= 1 and (row['detail'].get('component_count', 2) == 2)
            row['sigma'] = estimate(cal, row['tier'], result['hours_remaining'], row['final'], result['total_duration_hours'], row['score_age_hours']) if valid else None
            if row['sigma']:
                for key in ('outside_duration_support', 'outside_horizon_support', 'outside_elapsed_support'):
                    row['detail']['sigma_' + key] = row['sigma'][key]
        result['source_model_sha256'] = manifest['point_source_sha256']
        result['uncertainty'] = dict(method=sigma['method'], calibration_event_ids=sigma['calibration_event_ids'], formula=sigma['formula'], same_tier_only=True)
        return result

    def predict(self, name, live, event_id, cutoff, max_stale_hours=None, selection=None):
        if name in ('both', 'all'):
            return self.predict_all(live, event_id, cutoff, max_stale_hours)
        self.package_dir(name)
        target = restrict(live, [event_id], labels=False)
        event = target.known_event(target.events[event_id], cutoff)[0]
        if selection is None:
            selection = self.select(event, name)
        if name == 'average':
            components = [self.predict(n, target, event_id, cutoff, max_stale_hours) for n in ('previous_ridge', 'stack_mf')]
            by = [{r['tier']: r for r in model['predictions']} for model in components]
            rows = []
            for tier in sorted(set(by[0]) | set(by[1])):
                pair = [d[tier] for d in by if tier in d and d[tier]['status'] == 'ok']
                if not pair:
                    rows.append(dict(tier=tier, status='unavailable', final=None, current=None, observed_at=None, score_age_hours=None, sigma=None, detail=dict(reason='No usable component score')))
                    continue
                row = dict(pair[0])
                row['final'] = sum((r['final'] for r in pair)) / len(pair)
                row['detail'] = dict(component_count=len(pair), trained=True, components={n: d[tier]['final'] for n, d in zip(('previous_ridge', 'stack_mf'), by) if tier in d and d[tier]['status'] == 'ok'})
                rows.append(row)
            base = components[0]
            result = {k: base[k] for k in ('api_version', 'event_id', 'event_name', 'event_type', 'server', 'server_id', 'checkpoint_cn', 'closes_cn', 'hours_remaining', 'total_duration_hours')}
            result.update(model=dict(name=name, kind=LABELS[name], **selection['model']), routing=selection['routing'], predictions=rows, limitations=self.manifest(name)['limitations'])
        else:
            native = self._predictor(name).predict(target, event_id, cutoff, max_stale_hours)
            rows = []
            for r in native['predictions']:
                if r['status'] != 'ok':
                    rows.append(dict(tier=r['tier'], status='unavailable', final=None, current=None, observed_at=None, score_age_hours=None, sigma=None, detail=dict(reason=r.get('reason', 'unavailable'))))
                    continue
                detail = {k: r[k] for k in ('baseline', 'remaining_multiplier', 'raw_correction', 'supervised_training_targets') if k in r}
                detail['trained'] = r.get('supervised_training_targets', 0) > 0
                rows.append(dict(tier=r['tier'], status='ok', final=r['final'], current=r['current'], observed_at=r['observed_at'], score_age_hours=r['score_age_hours'], sigma=None, detail=detail))
            result = dict(api_version=API_VERSION, model=dict(name=name, kind=LABELS[name], **selection['model']), event_id=event_id, event_name=native['event_name'], event_type=native['event_type'], server='CN', server_id=3, checkpoint_cn=native['checkpoint_cn'], closes_cn=native['closes_cn'], hours_remaining=native['hours_remaining'], total_duration_hours=native['event_duration_hours'], routing=selection['routing'], predictions=rows, supervised_events=native['supervised_events'], reference_events=native['training_events'], limitations=self.manifest(name)['limitations'])
        return self._calibrate(name, result)

    def predict_all(self, live, event_id, cutoff, max_stale_hours=None):
        names = self.registry['default_configurations']
        models = [self.predict(n, live, event_id, cutoff, max_stale_hours) for n in names]
        base = models[0]
        result = {k: base[k] for k in ('api_version', 'event_id', 'event_name', 'event_type', 'server', 'server_id', 'checkpoint_cn', 'closes_cn', 'hours_remaining', 'total_duration_hours')}
        result.update(mode='two_configurations', model=dict(name='both', kind=LABELS['both'], selection='default', reason='Separate forecasts'), model_order=names, models=models, routing=dict(rule='Separate old average and recent Ridge', threshold=None, selected='both'), comparison=[dict(tier=t, models={n: next((r for r in m['predictions'] if r['tier'] == t)) for n, m in zip(names, models)}) for t in sorted({r['tier'] for m in models for r in m['predictions']})], limitations=[])
        return result
