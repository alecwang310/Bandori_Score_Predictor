#!/usr/bin/env python3
"""国服活动最终分数预测。默认分别输出旧版双模型平均与近期规则线性修正。

时间为中国时间；起报时间是输入数据截止时刻，预测目标始终为截活分数。
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from runtime.data import Dataset, Event
from runtime.live import LiveClient, load_snapshot, save_snapshot
from runtime.api import CN, atomic_bytes, atomic_json, timestamp
from runtime.engine import Distribution
from runtime.formatting import FOCUS_TIERS, bulletin


def default_issue_time(event, now):
    """Latest 08:00 within the event, falling back to the latest valid instant."""
    end = min(now, event.end - 1)
    default = datetime.fromtimestamp(end, CN).replace(hour=8, minute=0, second=0, microsecond=0)
    if default.timestamp() > end:
        default -= timedelta(days=1)
    if default.timestamp() <= event.start:
        default = datetime.fromtimestamp(end, CN)
    return default.strftime('%Y-%m-%d %H:%M:%S')


def parse_issue_time(value, event, now):
    value = value.strip()
    cutoff = now if value.lower() == 'now' else timestamp(value)
    if cutoff is None:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=CN)
            cutoff = parsed.timestamp()
        except (ValueError, OverflowError):
            raise ValueError('请输入中国时间YYYY-MM-DD HH:MM、带时区时间，或now') from None
    if cutoff > now + 1:
        raise ValueError('起报时间不能晚于当前时间')
    if not event.start < cutoff < event.end:
        raise ValueError('起报时间必须在活动期间')
    return cutoff


def main(argv=None, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',choices=('both','average','ridge','previous_ridge','stack_mf','all'),help='默认分别输出两套配置；average=旧版平均；ridge=近期修正；previous_ridge/stack_mf=旧版分量；all等同both')
    p.add_argument('--event',help='活动编号或latest；留空进入提示')
    p.add_argument('--as-of',metavar='TIME',help='数据截止时间：中国时间YYYY-MM-DD HH:MM或now；默认最近08:00')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--data',help='已下载的国服数据目录')
    source.add_argument('--snapshot',help='离线快照.json.gz')
    p.add_argument('--report',choices=('focus','full'),default='focus',help='focus显示五个常用档位，full显示全部档位；保存数据始终包含全部档位')
    p.add_argument('--output-prefix',help='保存文件的路径前缀；不覆盖已有预报')
    p.add_argument('--no-save',action='store_true',help='只显示，不保存预报')
    p.add_argument('--max-stale-hours',type=float,help='调整一小时观测时效限制；旧观测不提供σ')
    p.add_argument('--cache', default=str(root / 'cache'))
    p.add_argument('--timeout', type=float, default=10)
    p.add_argument('--save-snapshot',help='保存本次使用的输入快照')
    a = p.parse_args(argv)
    try:
        dist = Distribution(root)
        now = time.time()
        client = None
        records = None
        if a.snapshot:
            live = load_snapshot(a.snapshot)
            events = list(live.events.values())
        elif a.data:
            live = Dataset.load(a.data, dist.policy.tiers)
            events = [e for e in live.events.values() if e.start <= now]
        else:
            client = LiveClient(a.cache, timeout=a.timeout)
            records = client.discover(now)
            events = [Event(r['event_id'], r['start'], r['end'], r['meta'].get('eventType', ''))
                      for r in records if r['start'] <= now]
            live = None
        if not events:
            raise ValueError('没有已开始的国服活动')
        latest = max(events, key=lambda e: (e.start, e.event_id))
        if a.event is None:
            try:
                choice = input(f'活动编号 [回车=最新{latest.event_id}]：').strip() or 'latest'
            except EOFError:
                choice = 'latest'
        else:
            choice = a.event
        eid = latest.event_id if choice == 'latest' else int(choice)
        event = next((e for e in events if e.event_id == eid), None)
        if event is None:
            raise ValueError('输入数据中没有该国服活动')
        default = default_issue_time(event, now)
        if a.as_of is None and a.event is None:
            print('请选择输入数据的截止时间。预测目标始终是截活时的最终分数。未注明时区的时间均为中国时间。',file=sys.stderr)
            try:
                when = input(f'数据截止时间 [回车={default}；now=当前]：').strip() or default
            except EOFError:
                when = default
        else:
            when = a.as_of or default
        cutoff = parse_issue_time(when, event, now)
        known_event = live.known_event(event, cutoff)[0] if live is not None else event
        selection = dist.select(known_event, a.model)
        if live is None:
            record = next(r for r in records if r['event_id'] == eid)
            live = client.fetch_event(record, dist.policy.tiers, cutoff)
        result = dist.predict(selection['routing']['selected'],live,eid,cutoff,a.max_stale_hours,selection)
        text = bulletin(result,tiers=None if a.report=='full' else FOCUS_TIERS)
        if not a.no_save:
            stamp = datetime.now(CN).strftime('%Y%m%d_%H%M%S_%f')
            prefix = Path(a.output_prefix) if a.output_prefix else root / 'output' / f'event_{eid}_{stamp}'
            paths = [prefix.with_suffix('.json'), prefix.with_suffix('.txt')]
            if any(path.exists() for path in paths):
                raise ValueError('预报文件已存在，请选择其他路径')
            atomic_json(paths[0], result)
            atomic_bytes(paths[1], text.encode())
            print('已保存：' + str(paths[1]), file=sys.stderr)
        if a.save_snapshot:
            save_snapshot(a.save_snapshot, live, eid, cutoff)
        print(text, end='')
        return 0
    except (ValueError, OSError, KeyError, StopIteration) as error:
        print('无法生成预报：' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
