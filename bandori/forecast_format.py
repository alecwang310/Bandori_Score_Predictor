"""Chinese point-forecast bulletin with aligned two-digit uncertainty."""
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import math

from collect_bandori_data import CN, timestamp


def scientific(point,sigma=None):
    if not math.isfinite(point) or point<=0:raise ValueError('Positive finite point forecast required')
    exponent=int(math.floor(math.log10(point)))
    if sigma is None:return f'{point/10**exponent:.3g} × 10^{exponent}（σ未标定）'
    if not math.isfinite(sigma) or sigma<0:raise ValueError('Invalid sigma')
    unit_exponent=int(math.floor(math.log10(sigma)))-1 if sigma else exponent-2
    if unit_exponent>exponent:
        return f'({point/10**exponent:.3g} ± {sigma/10**exponent:.2g}) × 10^{exponent}'
    unit=Decimal(10)**unit_exponent
    center=(Decimal(str(point))/unit).quantize(Decimal(1),rounding=ROUND_HALF_UP)*unit
    uncertainty=(Decimal(str(sigma))/unit).quantize(Decimal(1),rounding=ROUND_HALF_UP)
    decimals=exponent-unit_exponent
    mantissa=center/(Decimal(10)**exponent)
    return f'{mantissa:.{decimals}f}（{uncertainty:.0f}）× 10^{exponent}'


def date_label(value,closing=False):
    dt=datetime.fromtimestamp(round(timestamp(value)),CN)
    # APIs store the last scoring second (e.g. 22:59:59); bulletins name the
    # following closing minute. Exact API timing remains in the JSON.
    if closing and dt.second==59:
        from datetime import timedelta
        dt+=timedelta(seconds=1)
    result=dt.strftime('%Y年%m月%d日%H时')
    if dt.minute or dt.second:result+=dt.strftime('%M分')
    if dt.second:result+=dt.strftime('%S秒')
    return result


def bulletin(forecast):
    lines=[f"{forecast['event_id']}-{forecast['event_name']}-国服",
        '起报时间：'+date_label(forecast['checkpoint_cn']),
        '截活时间：'+date_label(forecast['event_end_cn'],closing=True),
        f"时效：{forecast['hours_remaining']:g}小时",'', '====报文====','']
    borrowed=[];broad=[];stale=[];uncalibrated=[];bias=[];stale_unscaled=[]
    for row in forecast['predictions']:
        tier=row['tier']
        if row['status']!='ok':lines.append(f'T{tier}：无');continue
        sigma=row.get('sigma');lines.append(f"T{tier}："+scientific(row['final'],sigma['value'] if sigma else None))
        if sigma:
            if sigma.get('borrowed_from_tier'):borrowed.append(f"T{tier}借T{sigma['source_tier']}")
            if sigma['horizon_fallback']:broad.append(f'T{tier}')
            if abs(sigma['relative_bias'])>=.05 and (tier in (500,1000,1500,2000) or abs(sigma['relative_bias'])>=.20):
                bias.append(f"T{tier} {100*sigma['relative_bias']:+.1f}%")
            if sigma.get('stale_calibration_unavailable'):stale_unscaled.append(f'T{tier}')
        else:uncalibrated.append(f'T{tier}')
        if row.get('stale_input'):stale.append(f"T{tier} {row['score_age_hours']:.2f}h")
    lines+=['','Rule-aware Ridge + 24h观测器（全量重训）点预报。'
        f"训练{forecast['training_events']}场，时间衰减特征时间730天，2026-05-26起活动额外3倍权重。",
        'σ为历史误差的均方根等效尺度，含系统偏差；以(预报−终值)/预报的分档事件加权RMS乘当前点预报。'
        '括号与末位对齐，例如3.23（59）×10^6表示(3.23±0.59)×10^6。',
        '误差样本来自311–324中12场活动的按时间留出预报（按实际日期分组，不含325/326），'
        '每场先汇总，按730天衰减加权。此历史数据此前已检验且用于模型选择；σ未经独立覆盖率验证，不保证正态或68%覆盖。'
        '终值为收尾代理标签；全量重训并不自动缩小σ。']
    if forecast.get('feature_version')=='clock_v2':lines.append('本报使用附加83项日型特征的实验性版本；历史平均误差仅小幅改善，尚未建立独立验证收益。')
    if borrowed:lines.append('借档σ（缺少本档独立标定）：'+'、'.join(borrowed)+'；仅作粗略参考。')
    if broad:lines.append('本时效样本不足，σ使用本档全时效样本：'+'、'.join(broad)+'。')
    if uncalibrated:lines.append('σ无可用标定：'+'、'.join(uncalibrated)+'。')
    if bias:lines.append('对应σ样本的平均相对偏差（正值为偏高）：'+'；'.join(bias)+'；已含入σ，点预报未据此事后调偏。')
    if stale:lines.append('使用较旧的末次观测：'+'；'.join(stale)+'。无未来数据补点；跟踪器缺测不等于服务器停机。')
    if stale_unscaled:lines.append('缺测情景σ样本不足，仍用普通σ：'+'、'.join(stale_unscaled)+'。')
    if forecast.get('duration_exceeds_training_range'):lines.append('本场时长超出训练样本范围；σ未覆盖这种时长外推风险。')
    if forecast.get('input_source'):lines.append('数据：'+forecast['input_source']+'；无正分、缺测过久或参考不足的档位不发报。')
    return '\n'.join(lines)+'\n'
