"""Evaluate the new raw LSTM at actual tracker timestamps and hourly grids."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from collect_bandori_data import atomic_json
from .config import Config
from .dataset import Dataset
from .evaluation import summary
from .lstm_variants import SupportedInputLSTM
from .regime_sequence import forecast_event


def evaluate(data,directory):
    directory=Path(directory);plan=json.loads((directory/'plan.json').read_text());config=Config(**{
        **plan['config'],'tiers':tuple(plan['config']['tiers']),'hidden_sizes':tuple(plan['config']['hidden_sizes']),
        'neural_gains':tuple(plan['config']['neural_gains']),'statistical_blends':tuple(plan['config']['statistical_blends'])})
    data.add_terminal_labels(config);tiers=config.tiers;focal=(500,1000,1500,2000);cases=[];values=[];linear=[];rows=[]
    for fold in plan['folds']:
        metadata=json.loads((directory/fold['name']/'model.json').read_text())
        if set(fold['validation'])&set(metadata['training_event_ids']):raise ValueError('Validation event in training')
        model=SupportedInputLSTM.load(directory/fold['name']/'models/raw_lstm.pt','cpu')
        for eid in fold['validation']:
            event=data.events[eid]
            hourly=event.start+np.arange(1,int(np.ceil((event.end-event.start)/3600)))*3600
            observed=[]
            for tier in focal:
                curve=data.curves.get((eid,tier))
                if curve is not None:observed.extend(curve.times[(curve.times>event.start)&(curve.times<event.end)])
            cutoffs=np.unique(np.r_[hourly,observed])
            print('All-time raw LSTM:',fold['name'],eid,len(cutoffs),'query times',flush=True)
            predicted,details=forecast_event(model,data,event,tiers,cutoffs)
            for j,tier in enumerate(tiers):
                if tier not in focal or (eid,tier) not in data.labels:continue
                truth=data.labels[(eid,tier)];valid=details['present'][:,j]&(details['current'][:,j]<=truth)
                for i in np.flatnonzero(valid):
                    cases.append(SimpleNamespace(event_id=eid,tier=tier,remaining_hours=float(details['remaining'][i])))
                    values.append(float(predicted[i,j]));linear.append(float(details['current'][i,j]+details['future'][i,j]))
                    rows.append((eid,tier,cutoffs[i],details['remaining'][i],details['current'][i,j],predicted[i,j],truth,
                                 (cutoffs[i]-details['observed'][i,j])/3600,
                                 metadata['lstm_targets_by_tier'][str(tier)]))
    values=np.asarray(values);linear=np.asarray(linear);a=np.asarray(rows)
    supported=a[:,8]>0
    grid=np.array([abs((row[2]-data.events[int(row[0])].start)/3600-round((row[2]-data.events[int(row[0])].start)/3600))<1e-8 for row in a])
    def sliced(mask,predicted=values):
        return summary([c for c,keep in zip(cases,mask) if keep],predicted[mask],data)
    result=dict(scope='All hourly and original focal-tier tracker timestamps; T500/T1000/T1500/T2000',
        lstm=summary(cases,values,data),linear_anchor=summary(cases,linear,data),
        fresh_lstm=sliced(a[:,7]<=1),supported_heads_lstm=sliced(supported),supported_heads_linear=sliced(supported,linear),
        hourly_grid_lstm=sliced(grid),off_grid_lstm=sliced(~grid),
        initially_unsupported_head_rows=int(np.count_nonzero(a[:,8]==0)),
        limitations=['Only twelve historical event clusters; dense observations are not independent samples',
            'Actual-time density varies by event and tier; primary matched-clock comparison is reported separately',
            'An untrained T1500 head uses its linear anchor before joint rank reconciliation, explicitly counted as unsupported',
            'Previously inspected historical replay, not fresh generalization'])
    np.savez_compressed(directory/'all_timestamp_predictions.npz',rows=a,columns=np.array([
        'event_id','tier','cutoff','remaining_hours','current','prediction','actual','score_age_hours','supervised_training_targets']))
    atomic_json(directory/'all_timestamp_evaluation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',required=True);parser.add_argument('--model',required=True)
    args=parser.parse_args();config=Config.load(Path(args.model)/'full_refit/config.json');data=Dataset.load(args.data,config.tiers)
    result=evaluate(data,args.model);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
