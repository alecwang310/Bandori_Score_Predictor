"""Interactive latest-event CLI for a frozen, portable ridge deployment."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import time

from collect_bandori_data import CN, atomic_bytes, atomic_json, timestamp
from .dataset import Dataset
from .forecast_format import bulletin
from .forecast_uncertainty import add_uncertainty
from .live_data import LiveClient, load_snapshot, save_snapshot
from .ridge_runtime import RidgePredictor


def issue_time(value,now):
    if value.lower()=='now':return now
    parsed=timestamp(value)
    if parsed is None:
        try:parsed=datetime.fromisoformat(value).replace(tzinfo=CN).timestamp()
        except ValueError:raise ValueError('时间格式应为 2026-10-17T20:00:00+08:00 或 now') from None
    if parsed>now+1:raise ValueError('不能使用未来的起报时间')
    return parsed


def default_issue(event,now):
    end=min(now,event.end-1)
    dt=datetime.fromtimestamp(end,CN).replace(hour=8,minute=0,second=0,microsecond=0)
    if dt.timestamp()>end:dt-=timedelta(days=1)
    # Before the first daily issue, use the current time, preserving causality.
    return dt.timestamp() if dt.timestamp()>event.start else min(now,event.end-1)


def select_event(events,choice):
    if choice.lower()=='latest':return max(events,key=lambda e:(e.start,e.event_id))
    try:eid=int(choice)
    except ValueError:raise ValueError('活动编号应为数字或 latest') from None
    for event in events:
        if event.event_id==eid:return event
    raise ValueError('所选活动无国服日程或尚未开始')


def main(argv=None,root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    parser=argparse.ArgumentParser(description='国服 Bandori 最新活动分数线预报（冻结 Ridge + 24h 观测器）')
    parser.add_argument('--event',help='活动编号或 latest；不填则提示输入')
    parser.add_argument('--as-of',help='起报时间（默认最近08:00中国时间）；支持 now')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--snapshot',help='离线单活动 JSON 或 JSON.GZ')
    source.add_argument('--data',help='已完成的全量下载目录；不会修改其中数据')
    parser.add_argument('--model',default=str(root/'model'))
    parser.add_argument('--cache',default=str(root/'cache'))
    parser.add_argument('--output-prefix',help='输出 TXT/JSON 的文件名前缀；拒绝覆盖')
    parser.add_argument('--no-save',action='store_true',help='仅输出报文，不保存预报')
    parser.add_argument('--max-stale-hours',type=float,help='允许并标记较旧分数；默认最多1小时')
    parser.add_argument('--save-snapshot',help='另存本次所用活动历史（不含终值标签）')
    parser.add_argument('--timeout',type=float,default=10)
    args=parser.parse_args(argv)
    try:
        now=time.time();model=RidgePredictor(args.model);client=None;records=None
        if args.snapshot:
            live=load_snapshot(args.snapshot);events=list(live.events.values());input_source='离线单场快照'
        elif args.data:
            live=Dataset.load(args.data,model.config.tiers);events=[e for e in live.events.values() if e.start<=now];input_source='本地已完成下载'
        else:
            print('获取国服活动日程…',file=sys.stderr,flush=True)
            client=LiveClient(args.cache,timeout=args.timeout);records=client.discover(now)
            from .dataset import Event
            events=[Event(r['event_id'],r['start'],r['end'],r['meta'].get('eventType','')) for r in records]
            live=None;input_source='Bestdori/HHWX 国服 server=3 实时刷新'
        latest=select_event(events,'latest')
        if args.event is None:
            try:choice=input(f'活动编号 [回车=最新 {latest.event_id}]：').strip() or 'latest'
            except EOFError:choice='latest'
        else:choice=args.event
        event=select_event(events,choice)
        default=datetime.fromtimestamp(default_issue(event,now),CN).isoformat(timespec='seconds')
        if args.as_of is None and args.event is None:
            try:when=input(f'起报时间 [回车={default}；输入 now 为当前时刻]：').strip() or default
            except EOFError:when=default
        else:when=args.as_of or default
        cutoff=issue_time(when,now)
        if not event.start<cutoff<event.end:raise ValueError('起报时间必须在所选活动的开始与截活之间')
        if live is None:
            record=next(r for r in records if r['event_id']==event.event_id)
            print(f'仅刷新活动 {event.event_id} 的分数线历史…',file=sys.stderr,flush=True)
            def progress(label,completed,total):
                print(f'\r已获取 {completed}/{total} 档（{label}）       ',end='',file=sys.stderr,flush=True)
            live=client.fetch_event(record,model.config.tiers,cutoff,progress)
            print(file=sys.stderr)
        forecast=model.predict(live,event.event_id,cutoff,args.max_stale_hours)
        forecast['input_source']=input_source
        forecast['input_provenance']=live.provenance
        calibration=json.loads((Path(args.model)/'sigma.json').read_text())
        add_uncertainty(forecast,calibration);text=bulletin(forecast)
        if not args.no_save:
            generated=datetime.now(CN).strftime('%Y%m%d_%H%M%S_%f')
            prefix=Path(args.output_prefix) if args.output_prefix else root/'output'/f'event_{event.event_id}_{datetime.fromtimestamp(cutoff,CN):%Y%m%d_%H%M%S}_{generated}'
            paths=[prefix.with_suffix('.txt'),prefix.with_suffix('.json')]
            if any(p.exists() for p in paths):raise ValueError('输出预报已存在；请使用新文件名前缀')
            atomic_json(paths[1],forecast);atomic_bytes(paths[0],text.encode('utf-8'))
            print('已保存：'+str(paths[0]),file=sys.stderr)
        if args.save_snapshot:save_snapshot(args.save_snapshot,live,event.event_id,cutoff)
        print(text,end='')
        return 0
    except (ValueError,OSError,KeyError) as error:
        print('无法生成预报：'+str(error),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
