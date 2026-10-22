"""Chinese bulletins with aligned physics-style uncertainty notation."""
from datetime import datetime, timedelta
import math
from .api import CN

API_VERSION = 1

FOCUS_TIERS = (100, 500, 1000, 1500, 2000)

LABELS = {'average': '旧版双模型平均', 'ridge': '近期规则线性修正', 'previous_ridge': '旧版规则线性修正', 'stack_mf': '旧版非线性修正', 'both': '两套配置'}

def physics_score(value, sigma=None):
    """Two significant uncertainty digits, aligned with the score's last place."""
    exponent = math.floor(math.log10(value)) if value else 0
    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        mantissa = value / 10.0 ** exponent
        if round(mantissa, 2) >= 10:
            mantissa /= 10
            exponent += 1
        return f'{mantissa:.2f} * 10^{exponent}'
    unit = math.floor(math.log10(sigma)) - 1
    uncertainty = round(sigma / 10.0 ** unit)
    if uncertainty >= 100:
        unit += 1
        uncertainty = round(sigma / 10.0 ** unit)
    rounded = round(value / 10.0 ** unit) * 10.0 ** unit
    if rounded > 0:
        exponent = math.floor(math.log10(rounded))
    decimals = max(0, exponent - unit)
    if unit > exponent:
        exponent = unit
    return f'{rounded / 10.0 ** exponent:.{decimals}f}({uncertainty}) * 10^{exponent}'

def cn_time(value, closing=False):
    dt = datetime.fromisoformat(value).astimezone(CN)
    if closing and dt.minute == 59 and (dt.second >= 59):
        dt += timedelta(seconds=1)
    base = dt.strftime('%Y年%m月%d日%H时')
    if dt.minute:
        base += f'{dt.minute:02d}分'
    return base

def bulletin(result, tiers=None):
    h = result['hours_remaining']
    hours = str(round(h)) if abs(h - round(h)) <= 1 / 3600 + 1e-08 else f'{h:.2f}'
    lines = [f"{result['event_id']}-{result['event_name']}-国服", f"起报时间：{cn_time(result['checkpoint_cn'])}", f"截活时间：{cn_time(result['closes_cn'], True)}", f'时效：{hours}小时', '====点预报====']
    models = result['models'] if result.get('mode') == 'two_configurations' else [result]
    for model in models:
        lines += ['', LABELS[model['model']['name']] + '：']
        for row in model['predictions']:
            if tiers is not None and row['tier'] not in tiers:
                continue
            if row['status'] != 'ok':
                lines.append(f"T{row['tier']}：无")
                continue
            sigma = row['sigma']
            flags = []
            if sigma is None:
                flags.append('σ未标定')
            elif any((sigma[key] for key in ('outside_duration_support', 'outside_horizon_support', 'outside_elapsed_support'))):
                flags.append('σ超出标定范围')
            if row['score_age_hours'] > 1:
                flags.append(f"观测已过{row['score_age_hours']:.1f}小时")
            if row['detail'].get('component_count', 2) != 2:
                flags.append('部分分量缺失')
            lines.append(f"T{row['tier']}：{physics_score(row['final'], sigma['value'] if sigma else None)}" + (' [' + '；'.join(flags) + ']' if flags else ''))
    lines += ['', '括号为近似1σ，对齐末位：1.23(45) * 10^6 表示 (1.23 ± 0.45) * 10^6。', 'σ按已过时间与剩余时间的对数拟合，各档独立标定，包含系统偏差，不保证正态分布或68%覆盖。', '仅使用起报时已知的数据；最终分数标签为截活后的追踪记录。']
    return '\n'.join(lines) + '\n'
