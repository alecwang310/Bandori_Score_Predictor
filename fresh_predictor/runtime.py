"""NumPy-only live forecasts from the new post-rule model."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bandori.dataset import Curve,Dataset,Event
from collect_bandori_data import CN,iso
from .anchors import Anchors
from .data import Policy,restrict
from .regression import JointCorrection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_history(directory):
    directory=Path(directory);b=json.loads((directory/'reference_bank.json').read_text());curves={}
    with np.load(directory/'reference_bank.npz',allow_pickle=False) as arrays:
        for name in arrays.files:
            e,t=(int(s[1:]) for s in name.split('_'));a=arrays[name]
            curves[(e,t)]=Curve(a[:,0].copy(),a[:,1].copy(),a[:,2].copy())
    return Dataset([Event(**e) for e in b['events']],curves,{(e,t):y for e,t,y,s in b['labels']},
        b['outages'],b['revisions'],{}, {(e,t):s for e,t,y,s in b['labels']},
        {int(e):v for e,v in b.get('final_available',{}).items()})


class Predictor:
    def __init__(self,directory):
        self.directory=Path(directory);self.meta=json.loads((self.directory/'model.json').read_text())
        for name,h in self.meta['hashes'].items():
            if sha(self.directory/name)!=h:raise ValueError('Model asset checksum mismatch: '+name)
        values=json.loads((self.directory/'policy.json').read_text());values['tiers']=tuple(values['tiers']);values['checkpoint_hours']=tuple(values['checkpoint_hours'])
        self.policy=Policy(**values);self.model=JointCorrection.load(self.directory/'correction.npz');self.history=load_history(self.directory)
        with (self.directory/'calibration_forecasts.csv').open(encoding='utf-8-sig') as f:self.calibration=list(csv.DictReader(f))

    def sigma(self,tier,hours,prediction):
        rows=[r for r in self.calibration if int(r['tier'])==tier and r['actual'] and float(r['prediction'])>0]
        h=min(4,int(max(0,hours)//24));direct=[r for r in rows if min(4,int(float(r['hours_remaining'])//24))==h]
        if len({r['event_id'] for r in direct})>=3:rows=direct;scope='same_tier_horizon'
        else:scope='same_tier_all_horizons'
        groups={}
        for r in rows:
            error=1-float(r['actual'])/float(r['prediction']);groups.setdefault(int(r['event_id']),[]).append(error**2)
        if len(groups)<3:return None
        ids=sorted(groups);w=np.array([self.policy.event_weight(self.history.events[e],self.meta['fit_cutoff']) for e in ids])
        rms=float(np.sqrt(np.average([np.mean(groups[e]) for e in ids],weights=w)))
        return dict(value=prediction*rms,relative_rms=rms,events=len(ids),event_ids=ids,scope=scope,
            source='Adaptive chronological forecasts from earlier tuning-bank events; own outcome never selects its forecast',
            limitations='Very small proxy-label sample; contains bias; no independent coverage or normality guarantee')

    def predict(self,live,event_id,cutoff,max_stale_hours=None):
        if event_id in self.meta['reference_event_ids'] or cutoff<self.meta['fit_cutoff']:raise ValueError('Training event or forecast before fit cutoff')
        target=restrict(live,[event_id],labels=False);event=target.known_event(target.events[event_id],cutoff)[0]
        if event.start<self.policy.rule_at:raise ValueError('This predictor is for post-rule events only')
        cases,failed=Anchors(self.history,self.policy).checkpoint(target,event_id,cutoff,max_stale_hours)
        if not cases:raise ValueError('No fresh positive score lines; refresh data or explicitly allow older inputs')
        values=self.model.predict(cases);rows=[]
        for c,p in zip(cases,values):
            rows.append(dict(tier=c.tier,status='ok',final=math.ceil(p),current=math.ceil(c.current),observed_at=iso(c.observed_at),
                score_age_hours=(cutoff-c.observed_at)/3600,anchor=math.ceil(c.anchor),pace=math.ceil(c.pace),anchor_method=c.anchor_method,
                reference_events=len(c.references),role_transfer_references=sum(t!=c.tier for e,t in c.references),
                trained=self.meta['trained_tiers'][str(c.tier)],sigma=self.sigma(c.tier,c.hours_remaining,float(p))))
        for r in failed:rows.append(dict(tier=r['tier'],status='unavailable',reason=r['reason']))
        return dict(event_id=event_id,event_name=event.name,event_type=event.event_type,
            checkpoint_cn=datetime.fromtimestamp(cutoff,CN).isoformat(),generated_at_utc=datetime.now(timezone.utc).isoformat(),
            closes_cn=datetime.fromtimestamp(event.end,CN).isoformat(),hours_remaining=(event.end-cutoff)/3600,
            model='Fresh post-rule joint remaining-score model',model_sha256=sha(self.directory/'model.json'),
            coefficient_count=self.meta['coefficient_count'],strength=self.meta['strength'],
            supervised_events=len(self.meta['supervised_event_ids']),reference_events=len(self.meta['reference_event_ids']),
            predictions=sorted(rows,key=lambda r:r['tier']))


def physics_score(value,sigma=None):
    """Two significant uncertainty digits, aligned with the score's last place."""
    exponent=math.floor(math.log10(value)) if value else 0
    if sigma is None or not math.isfinite(sigma) or sigma<=0:
        mantissa=value/10.**exponent
        if round(mantissa,2)>=10:mantissa/=10;exponent+=1
        return f'{mantissa:.2f} * 10^{exponent}'
    unit=math.floor(math.log10(sigma))-1
    uncertainty=round(sigma/10.**unit)
    if uncertainty>=100:unit+=1;uncertainty=round(sigma/10.**unit)
    rounded=round(value/10.**unit)*10.**unit
    if rounded>0:exponent=math.floor(math.log10(rounded))
    decimals=max(0,exponent-unit)
    # Extremely large uncertainty can exceed the point estimate's exponent.
    if unit>exponent:exponent=unit
    return f'{rounded/10.**exponent:.{decimals}f}({uncertainty}) * 10^{exponent}'


FOCUS_TIERS=(100,500,1000,1500,2000)


def bulletin(result,tiers=None):
    def local_time(value):return datetime.fromisoformat(value).astimezone(CN).strftime('%Y-%m-%d %H:%M:%S')
    ridge=result.get('selected_predictor')=='ridge'
    lines=[f"{result['event_id']}-{result['event_name']}-国服",f"起报时间（数据截止）：{local_time(result['checkpoint_cn'])} 中国时间",
        f"截活时间：{local_time(result['closes_cn'])} 中国时间",f"时效（距截活）：{result['hours_remaining']:g}小时"]
    if result.get('routing'):
        route=result['routing'];mode='手动指定' if route['manual_override'] else '自动切换'
        lines.append(f"模型选择：{mode} {route['selected']}；活动总时长{route['event_duration_hours']:g}小时；阈值{route['threshold_hours']:g}小时")
    lines+=['====点预报====', 'Rule-aware Ridge（规则感知Ridge + 24h观测器）：' if ridge else 'Fresh（规则变更后联合统计模型）：']
    for r in result['predictions']:
        if tiers is not None and r['tier'] not in tiers:continue
        if r['status']!='ok':lines.append(f"T{r['tier']}：无（{r['reason']}）");continue
        s=r['sigma']
        flags=[]
        if s is None:flags.append('σ未标定')
        elif s.get('borrowed_from_tier'):flags.append(f"σ借用T{s['borrowed_from_tier']}")
        if s and s.get('stale_calibration_unavailable'):flags.append('缺测情景σ未独立标定')
        if r['score_age_hours']>1:flags.append(f"旧观测{r['score_age_hours']:.1f}h")
        if not r['trained']:flags.append('无本档监督，回退先验')
        lines.append(f"T{r['tier']}："+physics_score(r['final'],None if s is None else s['value'])+(' ['+'；'.join(flags)+']' if flags else ''))
    lines+=['','括号为近似1σ，对齐末位：1.23(45) * 10^6 表示 (1.23 ± 0.45) * 10^6。JSON保留未舍入的数值。',
        '起报时间是数据截止时刻；只使用该时刻及之前已知的数据，预测截活时的最终分数。',
        'σ为所选模型历史残差的RMS近似，样本很少且含系统偏差；不保证正态或68%覆盖。全量重训不自动缩小σ。']
    if tiers is not None:
        lines.append('仅显示 '+'/'.join('T'+str(t) for t in tiers)+'；--report full 查看全部分数线。')
    if ridge:
        lines += [f"Ridge已冻结拟合{result['training_events']}场；730天特征时间衰减，规则变更后活动3倍权重。σ使用Ridge自身历史残差。",
            'σ来自此前检验过的12场按时间留出预报，多为收尾代理标签；借档时单独标注，未专门标定超长活动。']
        if result.get('duration_exceeds_training_range'):lines.append('本场时长超过Ridge训练活动的最长时长。')
    else:
        lines += ['Fresh的σ来自317–320的逐场时间预报；无本档样本不借档。规则变更前的活动仅作历史曲线参考。']
    return '\n'.join(lines)+'\n'
