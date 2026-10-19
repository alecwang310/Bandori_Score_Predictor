"""Interactive latest-CN-event forecasts from the frozen new predictor."""
import argparse
from datetime import datetime,timedelta
import sys
import time
from pathlib import Path

from bandori.dataset import Dataset,Event
from bandori.live_data import LiveClient,load_snapshot,save_snapshot
from collect_bandori_data import CN,atomic_bytes,atomic_json,timestamp
from .runtime import FOCUS_TIERS, bulletin
from .dispatch import ForecastRouter,select_predictor


def default_issue_time(event,now):
    """Latest 08:00 within the event, falling back to the latest valid instant."""
    end=min(now,event.end-1)
    default=datetime.fromtimestamp(end,CN).replace(hour=8,minute=0,second=0,microsecond=0)
    if default.timestamp()>end:default-=timedelta(days=1)
    if default.timestamp()<=event.start:default=datetime.fromtimestamp(end,CN)
    return default.strftime('%Y-%m-%d %H:%M:%S')


def parse_issue_time(value,event,now):
    value=value.strip()
    cutoff=now if value.lower()=='now' else timestamp(value)
    if cutoff is None:
        try:cutoff=datetime.fromisoformat(value).replace(tzinfo=CN).timestamp()
        except (ValueError,OverflowError):
            raise ValueError('时间格式：2026-10-19 08:00（中国时间），或 now（当前时间）') from None
    if cutoff>now+1:raise ValueError('Cannot forecast using a future issue time')
    if not event.start<cutoff<event.end:raise ValueError('Issue time is outside event schedule')
    return cutoff


def main(argv=None,root=None):
    root=Path(root or Path(__file__).resolve().parents[1]);p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(root/'model'),help='Fresh模型目录')
    p.add_argument('--ridge-model',default=str(root/'long_event_model'),help='规则感知Ridge模型目录')
    p.add_argument('--predictor',choices=('auto','fresh','ridge'),help='auto：总时长>240小时用Ridge，否则Fresh；fresh/ridge：强制指定。不填时交互选择，回车=auto。')
    p.add_argument('--event')
    p.add_argument('--as-of',metavar='TIME',help='数据截止时间：中国时间 YYYY-MM-DD HH:MM，或 now；默认活动内最近一次08:00。可指定任意时刻。')
    source=p.add_mutually_exclusive_group();source.add_argument('--data');source.add_argument('--snapshot')
    p.add_argument('--cache',default=str(root/'cache'));p.add_argument('--timeout',type=float,default=10)
    p.add_argument('--max-stale-hours',type=float);p.add_argument('--output-prefix');p.add_argument('--no-save',action='store_true');p.add_argument('--save-snapshot')
    p.add_argument('--report',choices=('focus','full'),default='focus',
        help='focus=只报 T100/T500/T1000/T1500/T2000（默认）；full=报告全部分数线。JSON始终保留全部档位。')
    a=p.parse_args(argv)
    try:
        model=ForecastRouter(a.model,a.ridge_model);now=time.time();client=None;records=None
        if a.snapshot:live=load_snapshot(a.snapshot);events=list(live.events.values())
        elif a.data:live=Dataset.load(a.data,model.policy.tiers);events=[e for e in live.events.values() if e.start<=now]
        else:
            client=LiveClient(a.cache,timeout=a.timeout);records=client.discover(now)
            events=[Event(r['event_id'],r['start'],r['end'],r['meta'].get('eventType','')) for r in records if r['start']<=now];live=None
        if not events:raise ValueError('No started CN event available')
        latest=max(events,key=lambda e:(e.start,e.event_id))
        if a.event is None:
            try:choice=input(f'活动编号 [回车=最新 {latest.event_id}]：').strip() or 'latest'
            except EOFError:choice='latest'
        else:choice=a.event
        eid=latest.event_id if choice=='latest' else int(choice)
        event=next((e for e in events if e.event_id==eid),None)
        if event is None:raise ValueError('Event absent from CN input schedule')
        default=default_issue_time(event,now)
        if a.as_of is None and a.event is None:
            print('预测的是活动结束时的最终分数。下面选择使用截至何时已知的数据，所有时间均为中国时间。\n'
                '回车：使用下列默认时刻；now：使用当前数据；也可输入任意时刻，如 2026-10-19 14:30。\n'
                '08:00只是默认值；历史检验覆盖08:00、20:00、21:00。',file=sys.stderr)
            try:when=input(f'数据截止时间 [回车={default}；now=当前]：').strip() or default
            except EOFError:when=default
        else:when=a.as_of or default
        cutoff=parse_issue_time(when,event,now)
        known_event=live.known_event(event,cutoff)[0] if live is not None else event
        automatic=select_predictor(known_event)
        if a.predictor is None and a.event is None:
            print(f"活动总时长：{automatic['event_duration_hours']:g}小时；自动选择：{automatic['selected']}。",file=sys.stderr)
            try:requested=input('模型 [回车=auto自动；fresh=强制Fresh；ridge=强制规则感知Ridge]：').strip().lower() or 'auto'
            except EOFError:requested='auto'
        else:requested=a.predictor or 'auto'
        select_predictor(known_event,requested)
        if live is None:
            record=next(r for r in records if r['event_id']==eid);live=client.fetch_event(record,model.policy.tiers,cutoff)
        result=model.predict(live,eid,cutoff,a.max_stale_hours,requested);result['input_provenance']=live.provenance
        text=bulletin(result,tiers=None if a.report=='full' else FOCUS_TIERS)
        if not a.no_save:
            stamp=datetime.now(CN).strftime('%Y%m%d_%H%M%S_%f');prefix=Path(a.output_prefix) if a.output_prefix else root/'output'/f'event_{eid}_{stamp}'
            paths=[prefix.with_suffix('.json'),prefix.with_suffix('.txt')]
            if any(path.exists() for path in paths):raise ValueError('Fixed forecast already exists')
            atomic_json(paths[0],result);atomic_bytes(paths[1],text.encode())
            print('已保存：'+str(paths[1]),file=sys.stderr)
        if a.save_snapshot:save_snapshot(a.save_snapshot,live,eid,cutoff)
        print(text,end='');return 0
    except (ValueError,OSError,KeyError,StopIteration) as error:
        print('无法生成预报：'+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
