"""Fetch only one CN event, with cached raw responses and provider fallback."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np

from .api import (Collector, atomic_bytes, normalize_events, parse_points,
                                 select_events, slot)
from .data import Curve, Dataset, Event, top10_curves


def save_snapshot(path,data,event_id,cutoff):
    event=data.known_event(data.events[event_id],cutoff)[0]
    curves={str(t):np.column_stack((c.times,c.scores,c.available)).tolist()
            for (e,t),curve in data.curves.items() if e==event_id
            for c in (curve.prefix(cutoff,event.start),)}
    payload=dict(version=1,event=asdict(event),curves=curves,
        outages=[r for r in data.outages if r[0]==event_id and r[3]<=cutoff],
        revisions=[r for r in data.revisions if r[0]==event_id and r[1]<=cutoff],
        provenance=dict(snapshot_cutoff=cutoff,source='causal exported tracker history; no final labels'))
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    atomic_bytes(path,gzip.compress(raw,mtime=0) if str(path).endswith('.gz') else raw)


def load_snapshot(path):
    raw=Path(path).read_bytes();payload=json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)
    if payload.get('version')!=1:raise ValueError('Unknown snapshot version')
    event=Event(**payload['event']);curves={}
    for tier,rows in payload['curves'].items():
        a=np.asarray(rows,float)
        if not len(a):continue
        if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or (a[:,1]<0).any():raise ValueError('Invalid snapshot scores')
        if (np.diff(a[:,0])<0).any():raise ValueError('Snapshot times are not ordered')
        curves[(event.event_id,int(tier))]=Curve(a[:,0],a[:,1],a[:,2])
    return Dataset([event],curves,{},payload.get('outages',()),payload.get('revisions',()),payload.get('provenance',{}))


class LiveClient:
    def __init__(self,cache,timeout=10,retries=1,rate=4,workers=4):
        self.collector=Collector(cache,timeout=timeout,retries=retries,rate=rate)
        self.workers=workers

    def discover(self,now):
        for source in ('bestdori','hhwx'):
            response=self.collector.get('events',sources=(source,),refresh=True)
            if response:
                events,_=select_events([(source,normalize_events(response['payload'],source))],now)
                if events:return events
        errors='; '.join(r['error'] for r in self.collector.failures[-2:])
        raise ValueError('无法获取国服活动列表；可用 --snapshot 或 --data 离线输入。'+errors)

    def fetch_event(self,record,tiers,cutoff,progress=None):
        eid=record['event_id'];collector=self.collector;curves={};sources={}
        meta=record['meta'];event_type=meta.get('eventType')
        if not event_type:
            detail=collector.get('detail',eid,refresh=True)
            if detail:
                body=detail['payload'];body=body.get('data',body)
                event_type=body.get('eventType')
        if not event_type:raise ValueError('活动类型未提供；请重试或使用已下载的数据')
        event=Event(eid,record['start'],record['end'],event_type,slot(meta.get('eventName')) or f'Event {eid}')
        jobs=[('tracker',t) for t in tiers if t not in (1,10)]
        if set(tiers)&{1,10}:jobs.append(('top10',None))

        def get(kind,tier):
            best=None;quality=(-1,-1.)
            for source in ('bestdori','hhwx'):
                item=collector.get(kind,eid,tier,sources=(source,),refresh=True)
                if not item:continue
                if kind=='top10':found=top10_curves(item['payload'])
                else:
                    points,_=parse_points(item['payload'])
                    a=np.asarray([[p['time'],p['score'],p['available_at'] or p['time']] for p in points])
                    found={tier:Curve(a[:,0],a[:,1],a[:,2])} if len(a) else {}
                points=[c.last(cutoff,event.start) for c in found.values()]
                positive=[p for p in points if p and p[1]>0]
                q=(len(positive),max((p[0] for p in positive),default=0.))
                if q>quality:best=(found,item['source']);quality=q
                expected=2 if kind=='top10' and set(tiers)>={1,10} else 1
                if q[0]>=expected and all(cutoff-p[0]<=3600 for p in positive):break
            return best

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures={pool.submit(get,*job):job for job in jobs}
            for f in as_completed(futures):
                kind,tier=futures[f];result=f.result()
                if result:
                    found,source=result
                    for t,c in found.items():
                        if t in tiers:curves[(eid,t)]=c;sources[str(t)]=source
                if progress:progress(f'T{tier}' if tier else 'T1/T10',len(curves),len(tiers))
        provenance=dict(server=3,providers_by_tier=sources,
            availability_assumption='Missing publication timestamps are assumed equal to observation timestamps',
            failed_requests=collector.failures,
            cached_after_refresh_failure=[key for key,entry in collector.used.items() if entry.get('refresh_failed')])
        return Dataset([event],curves,{},provenance=provenance)
