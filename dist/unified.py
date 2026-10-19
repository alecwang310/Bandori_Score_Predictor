"""Registry-driven model selection, one unified output schema, TXT bulletin.

Every model (fresh / ridge / stack) consumes the same input — a snapshot or a
live data directory plus event id and cutoff — and emits the same JSON shape
(PLAN.md):

    {api_version, model:{name,kind,selection,reason}, event_id, event_name,
     event_type, checkpoint_cn, closes_cn, hours_remaining, total_duration_hours,
     routing:{rule,threshold,selected},
     predictions:[{tier,status,final,current,observed_at,score_age_hours,
                   sigma:{value,relative,method}|null, detail:{...}}],
     limitations:[...]}

`detail` carries model-specific fields (anchor/pace for fresh, multiplier and
correction for ridge/stack, per-tier supervised counts). Sigma: fresh and
ridge keep their calibrated values; stack adds approximate event-isolated RMS
scales where same-tier historical residuals are available.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from collect_bandori_data import CN
from fresh_predictor.data import Policy, restrict
from fresh_predictor.runtime import physics_score

API_VERSION = 1
FOCUS_TIERS = (100, 500, 1000, 1500, 2000)

MODEL_LABELS = {
    'fresh': 'Fresh（规则变更后联合统计模型）',
    'ridge': 'Rule-aware Ridge（规则感知Ridge + 24h观测器）',
    'stack': 'Stack（规则+显式600列联合神经网络校正，exp增益1.0）',
}

STACK_SIGMA_NOTE = ('No supported same-tier Stack sigma for this forecast: '
                    'unlabeled tier, stale input, or missing calibration package')


def _load_manifest(package_dir, name):
    path = Path(package_dir) / 'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    return dict(name=name, kind='unknown', api_version=API_VERSION, fit_cutoff=None,
                supervised_events=None, assets={}, training_provenance={},
                limitations=[])


class Distribution:
    """Loads the registry and the frozen model packages; predicts uniformly."""

    def __init__(self, root, registry_path=None, overrides=None):
        self.root = Path(root)
        registry_path = Path(registry_path) if registry_path else self.root / 'models' / 'registry.json'
        self.registry = json.loads(registry_path.read_text())
        values = json.loads((self.root / 'models' / 'fresh' / 'policy.json').read_text())
        values['tiers'] = tuple(values['tiers'])
        values['checkpoint_hours'] = tuple(values['checkpoint_hours'])
        self.policy = Policy(**values)          # input tiers/checkpoints (as in the original package)
        self.overrides = {k: v for k, v in (overrides or {}).items() if v}
        self._predictors = {}
        self._manifests = {}

    # ------------------------------------------------------------ registry
    def package_dir(self, name):
        if name not in self.registry['models']:
            raise ValueError('Unknown model: ' + name + '; available: ' +
                             ', '.join(sorted(self.registry['models'])) + ' (or auto)')
        if name in self.overrides:
            return Path(self.overrides[name])
        return self.root / 'models' / self.registry['models'][name]['package']

    def manifest(self, name):
        if name not in self._manifests:
            self._manifests[name] = _load_manifest(self.package_dir(name), name)
        return self._manifests[name]

    def select(self, event, requested=None):
        """Apply the registry default, explicit auto rule, or manual override."""
        is_default = requested is None or not requested.strip()
        requested = (requested or '').strip().lower() or self.registry.get('default', 'auto')
        auto = self.registry['auto']
        threshold = float(auto['threshold_hours'])
        above, below = auto['above_threshold'], auto['at_or_below_threshold']
        hours = (event.end - event.start) / 3600
        # Derived from the registry fields so a one-line swap stays self-describing
        # (identical wording to the stored `rule` for the default configuration).
        rule = (f'total scheduled duration > {threshold:g} hours -> {above}; '
                f'otherwise {below}')
        if requested != 'auto':
            self.package_dir(requested)   # validates the name
            selected, selection = requested, 'default' if is_default else 'manual'
            prefix = 'registry default' if is_default else 'manual override --model'
            reason = f'{prefix} {requested}; auto rule would select ' \
                     f'{above if hours > threshold else below}'
        else:
            selection = 'auto'
            if hours > threshold:
                selected = above
                reason = f'total scheduled duration {hours:g}h > {threshold:g}h -> {selected} (registry auto rule)'
            else:
                selected = below
                reason = f'total scheduled duration {hours:g}h <= {threshold:g}h -> {selected} (registry auto rule)'
        self.package_dir(selected)
        return dict(routing=dict(rule=rule, threshold=threshold, selected=selected),
                    model=dict(selection=selection, reason=reason),
                    duration_hours=hours)

    # ------------------------------------------------------------- predict
    def _predictor(self, name):
        if name not in self._predictors:
            directory = self.package_dir(name)
            if name == 'fresh':
                from fresh_predictor.runtime import Predictor
                self._predictors[name] = Predictor(directory)
            elif name == 'ridge':
                from bandori.ridge_runtime import RidgePredictor
                self._predictors[name] = RidgePredictor(directory)
            else:
                from bandori.stack_runtime import StackPredictor
                self._predictors[name] = StackPredictor(directory)
        return self._predictors[name]

    def predict(self, name, live, event_id, cutoff, max_stale_hours=None, selection=None):
        """Run one frozen model and normalize it into the unified schema."""
        manifest = self.manifest(name)
        target = restrict(live, [event_id], labels=False)
        if name == 'fresh':
            native = self._predictor(name).predict(target, event_id, cutoff, max_stale_hours)
        elif name == 'ridge':
            from bandori.forecast_uncertainty import add_uncertainty
            native = self._predictor(name).predict(target, event_id, cutoff, max_stale_hours)
            calibration = json.loads((self.package_dir('ridge') / 'sigma.json').read_text())
            add_uncertainty(native, calibration)
            native['closes_cn'] = native.get('event_end_cn') or native.get('closes_cn')
            for row in native['predictions']:
                if row['status'] == 'ok':
                    row['trained'] = row['supervised_training_targets'] > 0
        else:
            native = self._predictor(name).predict(target, event_id, cutoff, max_stale_hours)
        if selection is None:
            selection = self.select(target.known_event(target.events[event_id], cutoff)[0], name)
        return normalize(name, manifest, selection, native, self.registry)


def _row(name, native_row):
    if native_row.get('status') != 'ok':
        return dict(tier=native_row['tier'], status='unavailable', final=None, current=None,
                    observed_at=None, score_age_hours=None, sigma=None,
                    detail=dict(reason=native_row.get('reason', 'unavailable')))
    row = dict(tier=native_row['tier'], status='ok', final=native_row['final'],
               current=native_row.get('current'), observed_at=native_row.get('observed_at'),
               score_age_hours=native_row.get('score_age_hours'))
    if name == 'fresh':
        sigma = native_row.get('sigma')
        row['sigma'] = None if not sigma else dict(
            value=sigma.get('value'), relative=sigma.get('relative_rms'),
            method='fresh_chronological_rms')
        detail = {k: native_row.get(k) for k in
                  ('anchor', 'pace', 'anchor_method', 'trained', 'reference_events',
                   'role_transfer_references')}
        if sigma:
            detail['sigma_scope'] = sigma.get('scope')
            detail['sigma_events'] = sigma.get('events')
    elif name == 'ridge':
        sigma = native_row.get('sigma')
        row['sigma'] = None if not sigma else dict(
            value=sigma.get('value'), relative=sigma.get('relative_rmse'),
            method='ridge_out_of_fold_rms')
        detail = {k: native_row.get(k) for k in
                  ('baseline', 'remaining_multiplier', 'raw_correction', 'trained',
                   'supervised_training_targets')}
        if sigma:
            if sigma.get('borrowed_from_tier'):
                detail['sigma_borrowed_from_tier'] = sigma['borrowed_from_tier']
            if sigma.get('stale_calibration_unavailable'):
                detail['stale_calibration_unavailable'] = True
    else:
        detail = {k: native_row.get(k) for k in
                  ('baseline', 'remaining_multiplier', 'raw_correction',
                   'supervised_training_targets')}
        detail['trained'] = native_row.get('supervised_training_targets', 0) > 0
        sigma = native_row.get('sigma')
        row['sigma'] = dict(sigma) if sigma else dict(
            value=None, relative=None, method='uncalibrated', note=STACK_SIGMA_NOTE)
        if sigma:
            detail.update(sigma_events=sigma['events'], sigma_horizon_fallback=sigma['horizon_fallback'],
                          sigma_outside_duration_support=sigma['outside_duration_support'],
                          sigma_outside_horizon_support=sigma['outside_horizon_support'])
    row['detail'] = detail
    return row


def normalize(name, manifest, selection, native, registry):
    """Map one model's native result onto the unified schema."""
    routing = dict(selection['routing'])
    hours = selection['duration_hours']
    limitations = list(manifest.get('limitations', []))
    limitations.append('Routing rule provenance: ' +
                       registry['auto'].get('selection_basis', 'user-specified'))
    if native.get('duration_exceeds_training_range'):
        limitations.append('Event duration exceeds the longest training event of the selected model')
    if name == 'stack':
        for note in native.get('uncertainty', {}).get('limitations', [STACK_SIGMA_NOTE]):
            if note not in limitations:
                limitations.append(note)
    return dict(
        api_version=API_VERSION,
        model=dict(name=name, kind=manifest.get('kind', 'unknown'),
                   selection=selection['model']['selection'],
                   reason=selection['model']['reason']),
        event_id=native['event_id'], event_name=native['event_name'],
        event_type=native.get('event_type', ''),
        checkpoint_cn=native['checkpoint_cn'],
        closes_cn=native.get('closes_cn') or native['event_end_cn'],
        hours_remaining=native['hours_remaining'],
        total_duration_hours=hours,
        routing=routing,
        predictions=[_row(name, r) for r in native['predictions']],
        limitations=limitations)


def bulletin(result, tiers=None):
    """Chinese TXT bulletin; focus tiers by default, all tiers on --report full."""
    name = result['model']['name']

    def local_time(value):
        return datetime.fromisoformat(value).astimezone(CN).strftime('%Y-%m-%d %H:%M:%S')

    route = result['routing']
    mode = {'manual': '手动指定', 'default': '默认模型', 'auto': '自动切换'}[result['model']['selection']]
    lines = [f"{result['event_id']}-{result['event_name']}-国服",
             f"起报时间（数据截止）：{local_time(result['checkpoint_cn'])} 中国时间",
             f"截活时间：{local_time(result['closes_cn'])} 中国时间",
             f"时效（距截活）：{result['hours_remaining']:g}小时",
             f"模型选择：{mode} {name}；活动总时长{result['total_duration_hours']:g}小时；"
             f"阈值{route['threshold']:g}小时",
             '====点预报====',
             MODEL_LABELS.get(name, name) + '：']
    for r in result['predictions']:
        if tiers is not None and r['tier'] not in tiers:
            continue
        if r['status'] != 'ok':
            lines.append(f"T{r['tier']}：无（{r['detail'].get('reason', '')}）")
            continue
        s = r['sigma']
        detail = r['detail']
        value = s.get('value') if s else None
        flags = []
        if value is None:
            flags.append('σ未标定')
        if detail.get('sigma_borrowed_from_tier'):
            flags.append(f"σ借用T{detail['sigma_borrowed_from_tier']}")
        if detail.get('stale_calibration_unavailable'):
            flags.append('缺测情景σ未独立标定')
        if detail.get('sigma_horizon_fallback'):
            flags.append('σ合并相邻时效')
        if detail.get('sigma_outside_duration_support') or detail.get('sigma_outside_horizon_support'):
            flags.append('σ超出标定范围')
        if r['score_age_hours'] is not None and r['score_age_hours'] > 1:
            flags.append(f"旧观测{r['score_age_hours']:.1f}h")
        if not detail.get('trained'):
            flags.append('无本档监督，回退先验')
        lines.append(f"T{r['tier']}：" + physics_score(r['final'], value) +
                     (' [' + '；'.join(flags) + ']' if flags else ''))
    lines += ['', '括号为近似1σ，对齐末位：1.23(45) * 10^6 表示 (1.23 ± 0.45) * 10^6。JSON保留未舍入的数值。',
              '起报时间是数据截止时刻；只使用该时刻及之前已知的数据，预测截活时的最终分数。']
    if name == 'stack':
        lines.append('Stack σ：311、314–320共8场逐场留一残差，按本档/剩余小时分组、730天衰减；'
                     'RMS含偏差，不保证正态或68%覆盖。无本档残差或观测超1h不报σ。')
    else:
        lines.append('σ为所选模型历史残差的RMS近似，样本很少且含系统偏差；不保证正态或68%覆盖。全量重训不自动缩小σ。')
    if name == 'fresh':
        lines.append('Fresh的σ来自317–320的逐场时间预报；无本档样本不借档。规则变更前的活动仅作历史曲线参考。')
    elif name == 'ridge':
        lines += ['Ridge已冻结拟合319场；730天特征时间衰减，规则变更后活动3倍权重。σ使用Ridge自身历史残差。',
                  'σ来自此前检验过的12场按时间留出预报，多为收尾代理标签；借档时单独标注，未专门标定超长活动。']
    if any('longest training event' in l for l in result['limitations']):
        lines.append('本场时长超过该模型训练活动的最长时长。')
    if tiers is not None:
        lines.append('仅显示 ' + '/'.join('T' + str(t) for t in tiers) + '；--report full 查看全部分数线。')
    return '\n'.join(lines) + '\n'
