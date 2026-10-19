"""Every publication-causal tracker timestamp; no regular-grid downsampling."""
from dataclasses import dataclass
import math

import numpy as np

from .raw_sequence import snapshot
from .rules import stream_metadata


@dataclass
class NativeStream:
    event_id:int
    times:np.ndarray
    x:np.ndarray
    side:np.ndarray
    query_times:np.ndarray
    query_side:np.ndarray
    current:np.ndarray
    future:np.ndarray
    observed:np.ndarray
    remaining:np.ndarray
    present:np.ndarray
    query_current:np.ndarray
    query_future:np.ndarray
    query_remaining:np.ndarray
    query_present:np.ndarray


def native_times(data,event,tiers,cutoff):
    times=[event.start]
    for tier in tiers:
        c=data.curves.get((event.event_id,tier))
        if c is None:continue
        arrivals=np.maximum(c.times,c.available)
        valid=(c.times>=event.start)&(arrivals<=cutoff)&(arrivals<event.end)
        times.extend(arrivals[valid])
    times.extend(r[1] for r in data.revisions if r[0]==event.event_id and event.start<=r[1]<=cutoff and r[1]<event.end)
    times.extend(r[3] for r in data.outages if r[0]==event.event_id and event.start<=r[3]<=cutoff and r[3]<event.end)
    return np.unique(np.asarray(times,float))


def decoder_side(row,cutoff,state_time):
    clock=(cutoff/3600+8)%24
    age=np.where(row['present'],np.minimum(72,np.maximum(0,cutoff-row['observed'])/3600)/24,0.)
    return np.r_[row['side'],math.sin(2*math.pi*clock/24),math.cos(2*math.pi*clock/24),
                 min(72,max(0,(cutoff-state_time)/3600))/24,age]


def stream(data,event,tiers,seed=1729,cadence='native'):
    times=native_times(data,event,tiers,event.end-1e-6) if cadence=='native' else event.start+np.arange(int(math.ceil((event.end-event.start)/3600)))*3600
    rows=[];query=[];x=[];side=[];qside=[];qt=[]
    rng=np.random.default_rng(seed+event.event_id*17);previous=None;previous_time=event.start
    for i,t in enumerate(times):
        row=snapshot(data,event,tiers,float(t),previous,previous_time);rows.append(row)
        metadata=stream_metadata(data.known_event(event,float(t))[0],[t],tiers)[0]
        x.append(np.r_[row['x'],metadata]);side.append(np.r_[decoder_side(row,t,t),metadata])
        next_time=times[i+1] if i+1<len(times) else event.end
        query_time=t+rng.uniform(.05,.95)*min(24*3600,next_time-t)
        q=snapshot(data,event,tiers,float(query_time),row['current'],float(t));query.append(q);qt.append(query_time)
        qmetadata=stream_metadata(data.known_event(event,float(query_time))[0],[query_time],tiers)[0]
        qside.append(np.r_[decoder_side(q,query_time,t),qmetadata]);previous=row['current'];previous_time=float(t)
    def values(rr,key):return np.asarray([r[key] for r in rr])
    return NativeStream(event.event_id,times,np.asarray(x),np.asarray(side),np.asarray(qt),np.asarray(qside),
        *[values(rows,key) for key in ('current','future','observed','remaining','present')],
        *[values(query,key) for key in ('current','future','remaining','present')])


def pack(streams,data,config,fit_cutoff):
    streams=sorted(streams,key=lambda s:(-len(s.times),s.event_id));lengths=np.array([len(s.times) for s in streams])
    offsets=np.r_[0,np.cumsum(lengths)];batch_sizes=np.array([np.count_nonzero(lengths>i) for i in range(lengths.max())])
    order=np.concatenate([offsets[:n]+i for i,n in enumerate(batch_sizes)])
    k=len(config.tiers);total=int(lengths.sum());target=np.zeros((total,2,k,3));target[...,2]=1
    weights=np.zeros((total,2,k));bins=np.zeros((total,2),int);cells={};ages={}
    for e,s in enumerate(streams):
        if data.label_available(s.event_id,config)>fit_cutoff:raise ValueError('Unavailable training outcome')
        a,b=offsets[e:e+2];ages[e]=config.event_weight((fit_cutoff-data.events[s.event_id].end)/86400,data.events[s.event_id].start)
        # Integrate error over wall time rather than giving every packet one
        # vote: 12-second bursts must not outweigh quiet 30-minute periods.
        duration=np.minimum(24,np.maximum(1e-9,np.diff(np.r_[s.times,data.events[s.event_id].end])/3600))/2
        exposure=np.broadcast_to(duration[:,None,None],(b-a,2,k))
        for branch,(current,future,remaining,present) in enumerate(((s.current,s.future,s.remaining,s.present),
                (s.query_current,s.query_future,s.query_remaining,s.query_present))):
            bins[a:b,branch]=np.minimum(4,np.floor(remaining/24).astype(int))
            target[a:b,branch,:,0]=current;target[a:b,branch,:,1]=future
            for t,tier in enumerate(config.tiers):
                truth=data.labels.get((s.event_id,tier))
                if truth is None:continue
                quality=config.proxy_label_weight if data.label_sources.get((s.event_id,tier))=='post_close_tracker_proxy' else 1.
                target[a:b,branch,t,2]=truth;weights[a:b,branch,t]=present[:,t]*(truth>=current[:,t])*quality*duration
        for h in range(5):
            mass=float(np.sum(exposure*(weights[a:b]>0)*(bins[a:b]==h)[:,:,None]))
            if mass:cells[(e,h)]=mass
    masses={h:sum(ages[e] for e,g in cells if g==h) for h in range(5)};horizons=sum(m>0 for m in masses.values())
    for (e,h),count in cells.items():
        a,b=offsets[e:e+2];mask=bins[a:b]==h;weights[a:b][mask]*=ages[e]/masses[h]/count/horizons
    return dict(x=np.concatenate([s.x for s in streams])[order].astype('float32'),
        side=np.concatenate([s.side for s in streams])[order].astype('float32'),
        query_side=np.concatenate([s.query_side for s in streams])[order].astype('float32'),
        target=target[order].astype('float32'),weights=weights[order].astype('float32'),bins=bins[order],batch_sizes=batch_sizes,
        native_steps=total,event_ids=[s.event_id for s in streams],lengths=lengths)


def forecast_event(model,data,event,tiers,cutoffs,cadence='native'):
    cutoffs=np.asarray(sorted(set(map(float,cutoffs))))
    if not len(cutoffs):return np.empty((0,len(tiers))),None
    if cutoffs[0]<event.start or cutoffs[-1]>=event.end:raise ValueError('Query outside event')
    times=native_times(data,event,tiers,float(cutoffs[-1])) if cadence=='native' else event.start+np.arange(int(np.floor((cutoffs[-1]-event.start)/3600))+1)*3600
    rows=[];x=[];side=[];previous=None;last=event.start
    for t in times:
        row=snapshot(data,event,tiers,float(t),previous,last);rows.append(row)
        metadata=stream_metadata(data.known_event(event,float(t))[0],[t],tiers)[0]
        x.append(np.r_[row['x'],metadata]);side.append(np.r_[decoder_side(row,t,t),metadata])
        previous=row['current'];last=float(t)
    slots=np.searchsorted(times,cutoffs,side='right')-1;qrows=[];qside=[]
    for t,i in zip(cutoffs,slots):
        row=snapshot(data,event,tiers,float(t),rows[i]['current'],float(times[i]));qrows.append(row)
        metadata=stream_metadata(data.known_event(event,float(t))[0],[t],tiers)[0]
        qside.append(np.r_[decoder_side(row,t,times[i]),metadata])
    correction=model.forecast(np.asarray(x),np.asarray(qside),slots)
    current=np.asarray([r['current'] for r in qrows]);future=np.asarray([r['future'] for r in qrows]);values=current+future*np.exp(correction)
    for i in range(len(values)):
        floor=0.
        for j in np.argsort(tiers)[::-1]:
            if qrows[i]['present'][j]:values[i,j]=max(current[i,j],values[i,j],floor);floor=values[i,j]
    return values,dict(current=current,observed=np.asarray([r['observed'] for r in qrows]),
        present=np.asarray([r['present'] for r in qrows]),remaining=np.asarray([r['remaining'] for r in qrows]),native_steps=len(times))
