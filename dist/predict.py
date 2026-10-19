#!/usr/bin/env python3
"""Unified fresh/ridge/stack CN final-score forecasts from frozen packages.

    python3 predict.py [--model auto|fresh|ridge|stack] [--event N|latest]
      [--as-of TIME|now] [--snapshot path.json.gz | --data /path/to/bandori_cn_data]
      [--report focus|full] [--output-prefix /path/name] [--no-save]
      [--max-stale-hours H]

Identical input and output for every model: one unified JSON schema plus the
matching TXT bulletin (see README.md). Model choice and the 240h auto-routing
rule come from models/registry.json — swapping the auto model is one line.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from bandori.dataset import Dataset, Event
from bandori.live_data import LiveClient, load_snapshot, save_snapshot
from collect_bandori_data import CN, atomic_bytes, atomic_json, timestamp
from unified import Distribution, FOCUS_TIERS, bulletin


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
            cutoff = datetime.fromisoformat(value).replace(tzinfo=CN).timestamp()
        except (ValueError, OverflowError):
            raise ValueError('时间格式：2026-10-19 08:00（中国时间），或 now（当前时间）') from None
    if cutoff > now + 1:
        raise ValueError('Cannot forecast using a future issue time')
    if not event.start < cutoff < event.end:
        raise ValueError('Issue time is outside event schedule')
    return cutoff


def main(argv=None, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=('auto', 'fresh', 'ridge', 'stack'),
                   help='模型选择：auto=按注册表规则（总时长>240小时用ridge，否则fresh）；'
                        'fresh/ridge/stack=强制指定。不填或交互回车用注册表默认模型（stack）。')
    p.add_argument('--event', help='活动编号，或 latest（按国服开始时间）')
    p.add_argument('--as-of', metavar='TIME',
                   help='数据截止时间：中国时间 YYYY-MM-DD HH:MM，或 now；默认活动内最近一次08:00。')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--data', help='本地已下载的数据目录（如 bandori_cn_data）')
    source.add_argument('--snapshot', help='离线快照 json.gz')
    p.add_argument('--report', choices=('focus', 'full'), default='focus',
                   help='focus=只报 T100/T500/T1000/T1500/T2000（默认）；full=报告全部分数线。'
                        'JSON始终保留全部档位。')
    p.add_argument('--output-prefix', help='保存的 TXT/JSON 路径前缀（默认 output/event_N_时间戳）')
    p.add_argument('--no-save', action='store_true', help='只打印，不写文件')
    p.add_argument('--max-stale-hours', type=float,
                   help='允许的最大分数观测年龄（默认1小时；如 48 会明确标记）')
    p.add_argument('--cache', default=str(root / 'cache'))
    p.add_argument('--timeout', type=float, default=10)
    p.add_argument('--save-snapshot', help='把实际使用的输入另存为快照')
    p.add_argument('--registry', default=str(root / 'models' / 'registry.json'),
                   help='模型注册表（含auto路由），默认 models/registry.json')
    p.add_argument('--fresh-model', help='Fresh模型目录覆盖（默认 models/fresh）')
    p.add_argument('--ridge-model', help='Ridge模型目录覆盖（默认 models/ridge）')
    p.add_argument('--stack-model', help='Stack模型目录覆盖（默认 models/stack）')
    a = p.parse_args(argv)
    try:
        dist = Distribution(root, a.registry,
                            overrides=dict(fresh=a.fresh_model, ridge=a.ridge_model,
                                           stack=a.stack_model))
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
            raise ValueError('No started CN event available')
        latest = max(events, key=lambda e: (e.start, e.event_id))
        if a.event is None:
            try:
                choice = input(f'活动编号 [回车=最新 {latest.event_id}]：').strip() or 'latest'
            except EOFError:
                choice = 'latest'
        else:
            choice = a.event
        eid = latest.event_id if choice == 'latest' else int(choice)
        event = next((e for e in events if e.event_id == eid), None)
        if event is None:
            raise ValueError('Event absent from CN input schedule')
        default = default_issue_time(event, now)
        if a.as_of is None and a.event is None:
            print('预测的是活动结束时的最终分数。下面选择使用截至何时已知的数据，所有时间均为中国时间。\n'
                  '回车：使用下列默认时刻；now：使用当前数据；也可输入任意时刻，如 2026-10-19 14:30。\n'
                  '08:00只是默认值；历史检验覆盖08:00、20:00、21:00。', file=sys.stderr)
            try:
                when = input(f'数据截止时间 [回车={default}；now=当前]：').strip() or default
            except EOFError:
                when = default
        else:
            when = a.as_of or default
        cutoff = parse_issue_time(when, event, now)
        known_event = live.known_event(event, cutoff)[0] if live is not None else event
        automatic = dist.select(known_event, 'auto')
        default_model = dist.registry.get('default', 'auto')
        if a.model is None and a.event is None:
            print(f"活动总时长：{automatic['duration_hours']:g}小时；"
                  f"默认模型：{default_model}；显式auto会选择：{automatic['routing']['selected']}。", file=sys.stderr)
            try:
                requested = input(f'模型 [回车={default_model}；auto/fresh/ridge/stack=指定]：').strip().lower() or None
            except EOFError:
                requested = None
        else:
            requested = a.model
        selection = dist.select(known_event, requested)
        if live is None:
            record = next(r for r in records if r['event_id'] == eid)
            live = client.fetch_event(record, dist.policy.tiers, cutoff)
        result = dist.predict(selection['routing']['selected'], live, eid, cutoff,
                              a.max_stale_hours, selection)
        text = bulletin(result, tiers=None if a.report == 'full' else FOCUS_TIERS)
        if not a.no_save:
            stamp = datetime.now(CN).strftime('%Y%m%d_%H%M%S_%f')
            prefix = Path(a.output_prefix) if a.output_prefix else root / 'output' / f'event_{eid}_{stamp}'
            paths = [prefix.with_suffix('.json'), prefix.with_suffix('.txt')]
            if any(path.exists() for path in paths):
                raise ValueError('Fixed forecast already exists')
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
