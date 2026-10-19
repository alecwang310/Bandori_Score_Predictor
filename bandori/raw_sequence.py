"""Publication-causal raw event streams with arbitrary-time query branches."""
from dataclasses import dataclass
import math

import numpy as np

from collect_bandori_data import CN
from .analog import active_hours


EVENT_TYPES=('challenge','versus','mission_live','live_try','medley','story','festival','unknown')
RAW_FIELDS=('log_score','increment_fraction','fresh','score_age')
SIDE_FIELDS=('growth_6h','growth_24h','growth_6h_available','growth_24h_available')


@dataclass
class Stream:
    event_id:int
    times:np.ndarray
    x:np.ndarray
    side:np.ndarray
    current:np.ndarray
    future:np.ndarray
    observed:np.ndarray
    remaining:np.ndarray
    present:np.ndarray


def snapshot(data,event,tiers,cutoff,previous=None,previous_time=None):
    event=data.known_event(event,cutoff)[0];k=len(tiers)
    current=np.zeros(k);observed=np.full(k,event.start);raw=np.zeros((k,4));side=np.zeros((k,4));present=np.zeros(k,bool)
    for index,tier in enumerate(tiers):
        curve=data.curves.get((event.event_id,tier))
        point=curve.last(cutoff,event.start) if curve is not None else None
        if point is None or point[1]<=0:continue
        t,score=point;current[index]=score;observed[index]=t;present[index]=True
        prior=0 if previous is None else previous[index]
        raw[index]=[math.log1p(score)/16,np.clip((score-prior)/max(score,1),-1,1),
                    float(cutoff-t<=3600),min(72,(cutoff-t)/3600)/24]
        for j,h in enumerate((6,24)):
            old=curve.last(t-h*3600,event.start)
            if old and t-h*3600-old[0]<=min(3,h/4)*3600:
                side[index,j]=np.clip((score-old[1])/max(score,1),-1,1)
                side[index,j+2]=1
    duration=max(3600,event.end-event.start);elapsed=max(0,cutoff-event.start);remaining=max(0,(event.end-cutoff)/3600)
    clock=(cutoff/3600+8)%24
    dt=max(0,(cutoff-(event.start if previous_time is None else previous_time))/3600)
    exposure=active_hours(data,event,event.start if previous_time is None else previous_time,cutoff,cutoff)
    tag=np.zeros(len(EVENT_TYPES));tag[EVENT_TYPES.index(event.event_type) if event.event_type in EVENT_TYPES else -1]=1
    meta=np.r_[elapsed/duration,remaining*3600/duration,math.log1p(duration/3600)/6,tag]
    x=np.r_[raw.ravel(),math.sin(2*math.pi*clock/24),math.cos(2*math.pi*clock/24),min(dt,4)/4,exposure/max(dt,1e-6),meta]
    decoder=np.r_[side.ravel(),meta]
    # The linear anchor uses the actual observation age, rather than pretending
    # an old score was measured at the current query time.
    future=current*np.maximum(0,(event.end-observed)/3600)/np.maximum(1,(observed-event.start)/3600)
    return dict(x=x,side=decoder,current=current,future=future,observed=observed,remaining=remaining,present=present)


def grid_times(event,cutoff=None):
    end=event.end if cutoff is None else cutoff
    return event.start+np.arange(max(1,int(math.ceil((end-event.start)/3600))))*3600


def stream(data,event,tiers,times,previous_stream=None):
    rows=[];previous=None;previous_time=None
    for i,t in enumerate(times):
        if previous_stream is not None:
            previous=previous_stream.current[i];previous_time=previous_stream.times[i]
        row=snapshot(data,event,tiers,float(t),previous,previous_time);rows.append(row)
        if previous_stream is None:previous=row['current'];previous_time=t
    return Stream(event.event_id,np.asarray(times),**{name:np.asarray([r[name] for r in rows]) for name in
        ('x','side','current','future','observed','remaining','present')})


def training_streams(data,ids,tiers,seed=1729):
    result=[]
    for eid in ids:
        event=data.events[eid];times=grid_times(event);main=stream(data,event,tiers,times)
        rng=np.random.default_rng(seed+eid*17)
        # Queries branch from the same hourly state used at inference. Their
        # outcomes do not alter the recurrent hourly history.
        query_times=np.minimum(times+rng.uniform(.05,.95,len(times))*3600,event.end-1)
        query=stream(data,event,tiers,query_times,main);result.append((main,query))
    return result


def pack_streams(streams,data,config,fit_cutoff):
    length=max(len(main.times) for main,_ in streams);n=len(streams);k=len(config.tiers)
    example=streams[0][0];width=example.x.shape[-1];side_width=example.side.shape[-1]
    bundle={name:np.zeros((length,n,size)) for name,size in (('x',width),('side',side_width),('query_x',width),('query_side',side_width))}
    bundle.update(step_mask=np.zeros((length,n)),query_mask=np.zeros((length,n)))
    target=np.zeros((length,2,n,k,3));target[...,2]=1
    bins=np.zeros((length,2,n),int);weights=np.zeros((length,2,n,k))
    cell_counts={};ages={}
    for e,(main,query) in enumerate(streams):
        eid=main.event_id
        if data.label_available(eid,config)>fit_cutoff:raise ValueError('Training label was unavailable at fit cutoff')
        ages[e]=config.event_weight((fit_cutoff-data.events[eid].end)/86400,data.events[eid].start)
        for branch,s in enumerate((main,query)):
            size=len(s.times);prefix='query_' if branch else ''
            bundle[prefix+'x'][:size,e]=s.x;bundle[prefix+'side'][:size,e]=s.side;bundle['query_mask' if branch else 'step_mask'][:size,e]=1
            bins[:size,branch,e]=np.minimum(4,np.floor(s.remaining/24).astype(int))
            target[:size,branch,e,:,0]=s.current;target[:size,branch,e,:,1]=s.future
            for t,tier in enumerate(config.tiers):
                truth=data.labels.get((eid,tier))
                if truth is None:continue
                valid=s.present[:,t]&(truth>=s.current[:,t])
                target[:size,branch,e,t,2]=truth
                quality=config.proxy_label_weight if data.label_sources.get((eid,tier))=='post_close_tracker_proxy' else 1
                weights[:size,branch,e,t]=valid*quality
        for h in range(5):
            mask=bins[:,:,e]==h;count=int(np.count_nonzero(weights[:,:,e][mask]))
            if count:cell_counts[(e,h)]=count
    totals={h:sum(ages[e] for e,bin_ in cell_counts if bin_==h) for h in range(5)}
    available_horizons=sum(v>0 for v in totals.values())
    for (e,h),count in cell_counts.items():
        mask=bins[:,:,e]==h
        weights[:,:,e][mask]*=ages[e]/totals[h]/count/available_horizons
    return bundle,target,weights,bins


def main_bundle(s):return dict(x=s.x[:,None],side=s.side[:,None],step_mask=np.ones((len(s.times),1)))


def forecast_event(model,data,event,tiers,cutoffs):
    """Evaluate any cutoffs using one causal hourly pass and batched branches."""
    cutoffs=np.asarray(sorted(set(map(float,cutoffs))))
    if not len(cutoffs):return np.empty((0,len(tiers))),None
    if cutoffs[0]<event.start or cutoffs[-1]>=event.end:raise ValueError('Query outside event schedule')
    # Include the most recent exact grid state even when the last query is on it.
    grid=event.start+np.arange(int(math.floor((cutoffs[-1]-event.start)/3600))+1)*3600
    main=stream(data,event,tiers,grid)
    slots=np.floor((cutoffs-event.start)/3600+1e-10).astype(int)
    rows=[snapshot(data,event,tiers,t,main.current[i],grid[i]) for t,i in zip(cutoffs,slots)]
    x=np.asarray([r['x'] for r in rows]);side=np.asarray([r['side'] for r in rows]);is_off_grid=(cutoffs-grid[slots])>1e-3
    delta=model.forecast(main.x,main.side,x,side,slots,is_off_grid)
    current=np.asarray([r['current'] for r in rows]);future=np.asarray([r['future'] for r in rows])
    values=current+future*np.exp(delta)
    # Reconcile all outputs simultaneously to retain rank order and observed floors.
    for i in range(len(values)):
        floor=0.
        for t in np.argsort(tiers)[::-1]:
            if rows[i]['present'][t]:values[i,t]=max(values[i,t],current[i,t],floor);floor=values[i,t]
    details=dict(times=cutoffs,current=current,future=future,observed=np.asarray([r['observed'] for r in rows]),
                 present=np.asarray([r['present'] for r in rows]),remaining=np.asarray([r['remaining'] for r in rows]))
    return values,details
