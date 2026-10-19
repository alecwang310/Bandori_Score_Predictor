"""Raw LSTM streams plus dated, outcome-independent reward metadata."""
import numpy as np

from .raw_sequence import snapshot, stream, training_streams as raw_training_streams
from .rules import stream_metadata


def append_metadata(s,data,tiers):
    rows=np.concatenate([stream_metadata(data.known_event(data.events[s.event_id],float(t))[0],[t],tiers) for t in s.times])
    s.x=np.column_stack((s.x,rows));s.side=np.column_stack((s.side,rows))
    return s


def training_streams(data,ids,tiers,seed=1729):
    pairs=raw_training_streams(data,ids,tiers,seed)
    return [(append_metadata(main,data,tiers),append_metadata(query,data,tiers)) for main,query in pairs]


def forecast_event(model,data,event,tiers,cutoffs):
    cutoffs=np.asarray(sorted(set(map(float,cutoffs))))
    if not len(cutoffs):return np.empty((0,len(tiers))),None
    if cutoffs[0]<event.start or cutoffs[-1]>=event.end:raise ValueError('Query outside event schedule')
    grid=event.start+np.arange(int(np.floor((cutoffs[-1]-event.start)/3600))+1)*3600
    main=append_metadata(stream(data,event,tiers,grid),data,tiers)
    slots=np.floor((cutoffs-event.start)/3600+1e-10).astype(int)
    rows=[snapshot(data,event,tiers,t,main.current[i],grid[i]) for t,i in zip(cutoffs,slots)]
    extra=np.concatenate([stream_metadata(data.known_event(event,float(t))[0],[t],tiers) for t in cutoffs])
    x=np.column_stack((np.asarray([r['x'] for r in rows]),extra))
    side=np.column_stack((np.asarray([r['side'] for r in rows]),extra))
    delta=model.forecast(main.x,main.side,x,side,slots,(cutoffs-grid[slots])>1e-3)
    current=np.asarray([r['current'] for r in rows]);future=np.asarray([r['future'] for r in rows]);values=current+future*np.exp(delta)
    for i in range(len(values)):
        floor=0.
        for j in np.argsort(tiers)[::-1]:
            if rows[i]['present'][j]:values[i,j]=max(current[i,j],values[i,j],floor);floor=values[i,j]
    details=dict(times=cutoffs,current=current,future=future,observed=np.asarray([r['observed'] for r in rows]),
                 present=np.asarray([r['present'] for r in rows]),remaining=np.asarray([r['remaining'] for r in rows]))
    return values,details
