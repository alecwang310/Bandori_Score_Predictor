from dataclasses import asdict,dataclass
from datetime import datetime
import hashlib
import json

import numpy as np

from bandori.config import Config
from bandori.dataset import Dataset
from collect_bandori_data import CN,timestamp


TIERS=(1,10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000,40000,50000,70000,100000)
FOCAL=(500,1000,1500,2000)
TYPES=('story','challenge','versus','mission_live','live_try','festival','medley','unknown')


@dataclass(frozen=True)
class Policy:
    tiers:tuple=TIERS
    rule_at:float=timestamp('2026-05-26T00:00:00+08:00')
    age_days:float=730.
    post_multiplier:float=3.
    proxy_weight:float=.75
    label_delay_hours:float=24.
    reference_gap_hours:float=12.
    freshness_hours:float=1.
    min_references:int=5
    checkpoint_hours:tuple=(8,20,21)

    def label_config(self):
        return Config(tiers=self.tiers,allow_terminal_labels=True,final_availability_delay_hours=self.label_delay_hours,
                      terminal_window_minutes=5,characteristic_time_days=self.age_days)

    def event_weight(self,event,known_at):
        return np.exp(-max(0,known_at-event.end)/86400/self.age_days)*(self.post_multiplier if event.start>=self.rule_at else 1)

    def to_dict(self):return json.loads(json.dumps(asdict(self)))


def restrict(data,ids,labels=True):
    """A new data view: excluded outcomes, curves and metadata are unreachable."""
    allowed=set(ids)
    return Dataset([e for i,e in data.events.items() if i in allowed],
        {k:v for k,v in data.curves.items() if k[0] in allowed},
        {k:v for k,v in data.labels.items() if k[0] in allowed} if labels else {},
        [r for r in data.outages if r[0] in allowed],[r for r in data.revisions if r[0] in allowed],
        dict(data.provenance),{k:v for k,v in data.label_sources.items() if k[0] in allowed} if labels else {},
        {k:v for k,v in data.final_available.items() if k in allowed} if labels else {})


def fingerprint(data):
    h=hashlib.sha256()
    h.update(json.dumps([asdict(data.events[i]) for i in sorted(data.events)],sort_keys=True).encode())
    for key in sorted(data.curves):
        h.update(str(key).encode());c=data.curves[key]
        for a in (c.times,c.scores,c.available):h.update(np.asarray(a,dtype='<f8').tobytes())
    for key in sorted(data.labels):h.update(repr((key,data.labels[key],data.label_sources.get(key))).encode())
    h.update(repr((data.outages,data.revisions,sorted(data.final_available.items()))).encode())
    return h.hexdigest()


def issue_times(event,hours):
    day=datetime.fromtimestamp(event.start,CN).replace(hour=0,minute=0,second=0,microsecond=0)
    while day.timestamp()<event.end:
        for hour in sorted(hours):
            t=day.replace(hour=hour).timestamp()
            if event.start<t<event.end:yield t
        day=day.fromtimestamp(day.timestamp()+86400,CN)
