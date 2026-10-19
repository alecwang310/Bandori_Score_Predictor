from dataclasses import replace

import numpy as np

from bandori.config import Config
from bandori.dataset import Curve,Dataset,Event
from collect_bandori_data import timestamp


def dataset(n=48):
    start=timestamp('2023-01-01T13:00:00+08:00');events=[];curves={};labels={}
    for i in range(n):
        event=Event(1000-i,start+i*8*86400,start+(i*8+5)*86400,
                    'challenge' if i%2 else 'mission_live')
        events.append(event);times=np.linspace(event.start,event.end,481)
        for tier in (500,1500):
            final=2e6*(500/tier)**.4*(1+.004*i)
            scores=final*((times-event.start)/(event.end-event.start))**(1.05+.03*(i%3))
            curves[(event.event_id,tier)]=Curve(times.copy(),scores,times.copy())
            labels[(event.event_id,tier)]=final
    return Dataset(events,curves,labels)


def config(**changes):
    return replace(Config(tiers=(500,1500),min_train_events=8,min_reference_events=3,
                          tuning_events=4,calibration_events=4,test_events=4,
                          epochs=5,hidden_sizes=(3,)),**changes)
