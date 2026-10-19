"""NumPy-only deployment of the frozen joint ridge correction.

The feature layout and forecast formula match regime_suite, but inference
does not load the other model families, their weights, or PyTorch.
"""
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN, iso
from .advanced_statistics import JointRidge
from .analog import reconcile
from .config import Config
from .daily_observer import observer_arrays
from .pipeline import attach_bank, digest
from .representations import Compression, matrices
from .role_analog import RoleAwareAnalog
from .rules import era, packed_metadata


def ridge_inputs(cases, data, config, training_ids, compression, input_mask,clock_compression=None,base_mask=None):
    arrays,index,_=matrices(cases,data,config.tiers)
    daily=observer_arrays(cases,index,data,config,training_ids)
    metadata=packed_metadata(cases,index,data);n,k=index.shape
    support=np.zeros((n,k,3))
    for g,t in zip(*np.where(index>=0)):
        c=cases[index[g,t]]
        support[g,t,:2]=[math.log1p(c.features.get('role_transfer_support',0)),
                         math.log1p(c.features.get('direct_tier_references',len(c.references)))]
        support[g,t,2]=sum(era(data.events[e])==era(data.events[c.event_id]) for e in c.references)/max(1,len(c.references))
    rule_metadata=np.concatenate((metadata,support),axis=2)
    x=np.column_stack((compression.transform(arrays['rich']),
                       daily['summaries'].reshape(n,-1),rule_metadata.reshape(n,-1)))
    if clock_compression is not None:
        from .clock_features import COMPACT_INDICES, arrays as clock_arrays
        clock=clock_arrays(dict(cases=cases,index=index,daily=daily),data)
        extra=np.column_stack((clock['features'][:,:,COMPACT_INDICES].reshape(n,-1),
                               clock_compression.transform(clock['features'].reshape(n,-1))))
        x=np.column_stack((x*base_mask,extra))
    if x.shape[-1]!=len(input_mask):raise ValueError('Incompatible ridge feature layout')
    return x*input_mask,index


class RidgePredictor:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'deployment.json').read_text())
        if self.manifest.get('kind')!='bandori_cn_ridge_deployment_v1':raise ValueError('Unknown model package')
        for name,expected in self.manifest['hashes'].items():
            if digest(self.directory/name)!=expected:raise ValueError('Model checksum mismatch: '+name)
        self.config=Config.load(self.directory/'config.json')
        self.core=JointRidge.load(self.directory/'ridge.npz')
        self.compression=Compression.load(self.directory/'compression.npz')
        self.clock_compression=None;self.base_mask=None
        with np.load(self.directory/'input_masks.npz',allow_pickle=False) as a:
            if self.manifest.get('feature_version')=='clock_v2':
                from .clock_features import ClockCompression
                self.clock_compression=ClockCompression.load(self.directory/'clock_compression.npz')
                self.mask=a['new_pca'].copy();self.base_mask=a['base_pca'].copy()
            else:self.mask=a['pca'].copy()

    def predict(self,live,event_id,cutoff,max_stale_hours=None):
        ids=self.manifest['training_event_ids'];config=self.config
        if event_id in ids:raise ValueError('This event was used in training; use an event after the frozen fit')
        if cutoff<self.manifest['fit_cutoff']:raise ValueError('Forecast precedes the frozen model fit')
        if max_stale_hours is not None:
            if not math.isfinite(max_stale_hours) or max_stale_hours<=0:raise ValueError('Invalid maximum score age')
            config=replace(config,max_stale_minutes=max_stale_hours*60)
        data=attach_bank(self.directory,live)
        if event_id not in data.events:raise ValueError('Requested event has no CN schedule')
        event=data.known_event(data.events[event_id],cutoff)[0]
        if not event.start<cutoff<event.end:raise ValueError('起报时间必须在活动开始与截活之间')
        analog=RoleAwareAnalog(data,config,ids);cases=[];failed=[]
        for tier in config.tiers:
            try:cases.append(analog.build(event,tier,cutoff))
            except ValueError as error:failed.append(dict(tier=tier,status='unavailable',reason=str(error)))
        if not cases:raise ValueError('本时刻没有可用分数线；请检查数据新鲜度或起报时间')
        x,index=ridge_inputs(cases,data,config,ids,self.compression,self.mask,self.clock_compression,self.base_mask)
        delta=self.core.predict(x);values=np.array([c.baseline for c in cases]);details={}
        gain=self.manifest['gain']
        for g,t in zip(*np.where(index>=0)):
            i=index[g,t];c=cases[i];multiplier=1+gain*delta[g,t]
            values[i]=c.current+max(0,c.baseline-c.current)*multiplier
            details[i]=dict(baseline=c.baseline,remaining_multiplier=float(multiplier),raw_correction=float(delta[g,t]))
        values=reconcile(cases,values)
        rows=[dict(tier=c.tier,status='ok',final=math.ceil(values[i]),current=math.ceil(c.current),
            observed_at=iso(c.observed_at),score_age_hours=(cutoff-c.observed_at)/3600,
            stale_input=(cutoff-c.observed_at)>self.config.max_stale_minutes*60,
            supervised_training_targets=self.manifest['training_targets_by_tier'][str(c.tier)],**details[i]) for i,c in enumerate(cases)]
        durations=self.manifest['training_durations_hours'];hours=(event.end-event.start)/3600
        return dict(event_id=event_id,event_name=event.name,event_type=event.event_type,server='CN',server_id=3,
            checkpoint_cn=datetime.fromtimestamp(cutoff,CN).isoformat(),event_end_cn=datetime.fromtimestamp(event.end,CN).isoformat(),
            hours_remaining=(event.end-cutoff)/3600,event_duration_hours=hours,
            duration_exceeds_training_range=hours>max(durations),
            near_duration_reference_events=sum(abs(d/hours-1)<=.2 for d in durations),
            model='Rule-aware Ridge + clock features' if self.clock_compression else 'Rule-aware Ridge + 24h observer',
            feature_version=self.manifest.get('feature_version','daily_v1'),source_model_sha256=self.manifest['source_model_sha256'],
            age_characteristic_days=config.characteristic_time_days,post_rule_weight=config.post_rule_change_multiplier,
            training_events=len(ids),predictions=sorted(rows+failed,key=lambda r:r['tier']))
