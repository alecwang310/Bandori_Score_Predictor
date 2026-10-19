"""Near-bare raw-sequence LSTM experiment on a frozen chronological split.

One stock PyTorch LSTM cell reads publication-causal raw hourly score
streams and emits joint tier corrections through a small shared head.
Fractional-hour query branches share the cell without writing back, so
saved inference accepts any timestamp, not only 08:00. There is no model
selection: the architecture is fixed here and compared pairwise against
the frozen source champion on identical cases. Training runs on Apple
Metal (MPS) when available.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from collect_bandori_data import CN, atomic_json, iso, timestamp, write_csv
from .config import Config
from .dataset import Dataset
from .evaluation import paired_bootstrap, summary
from .experiments import build_cases, context
from .lstm import RawJointLSTM, device_of
from .observer_experiment import CLASSES, controls_at, validate_source
from .pipeline import digest, save_bank
from .raw_sequence import (RAW_FIELDS, SIDE_FIELDS, forecast_event, pack_streams,
                           training_streams)
from .representations import Compression


def lstm_case_predictions(model, data, cases, tiers):
    predictions = np.zeros(len(cases))
    tier_index = {t: j for j, t in enumerate(tiers)}
    by_event = {}
    for i, c in enumerate(cases):
        by_event.setdefault(c.event_id, []).append(i)
    for eid, indices in by_event.items():
        cutoffs = sorted({cases[i].cutoff for i in indices})
        values, _ = forecast_event(model, data, data.events[eid], tiers, cutoffs)
        slot = {t: j for j, t in enumerate(cutoffs)}
        for i in indices:
            predictions[i] = values[slot[cases[i].cutoff], tier_index[cases[i].tier]]
    return predictions


def train(data, source, output, registry='evaluation_registry.json', device=None,
          hidden=32, decoder=16, epochs=120, learning_rate=.005):
    source, output = Path(source), Path(output)
    config, split, selection, metadata = validate_source(source, data, registry)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new immutable experiment directory')
    data.add_terminal_labels(config)
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    ledger_before = digest(registry) if Path(registry).exists() else None
    device = device_of(device)
    settings = dict(framework='pytorch', device=str(device), hidden=hidden, decoder=decoder,
                    epochs=epochs, learning_rate=learning_rate, decay=.001, optimizer='Adam',
                    gradient_clip=1.0, objective='relative squared error with per-horizon signed-bias penalty',
                    correction='bounded log pace delta 3*tanh(raw/3) on a linear close anchor',
                    raw_fields=RAW_FIELDS, side_fields=SIDE_FIELDS,
                    query_branch='fractional-hour snapshots branch from the hourly grid state and never write back')
    atomic_json(output / 'plan.json', dict(
        source=str(source.resolve()), split=split, config=config.to_dict(), settings=settings,
        implementation={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
        paired_design='Fixed near-bare architecture without selection; compared against the frozen source champion on identical cases',
        test_policy='Previously inspected chronological test; follow-up evidence, not fresh validation',
        inference='Any timestamp inside the known event schedule; evaluation cases are 08:00 checkpoints'))
    atomic_json(output / 'config.json', config.to_dict())
    atomic_json(output / 'split.json', split)
    compression = Compression.load(source / 'models/compression.npz')
    cores = {key: CLASSES[kind].load(source / 'models' / (key + '.npz'))
             for key, kind in metadata['core_kinds'].items()}
    print('Build raw training streams:', len(split['train']), 'events on', device, flush=True)
    streams = training_streams(data, split['train'], config.tiers)
    bundle, target, weights, bins = pack_streams(streams, data, config, split['fit_cutoff'])
    print('Packed', bundle['x'].shape[0], 'hourly steps;', int(np.count_nonzero(weights)),
          'supervised rows', flush=True)
    model = RawJointLSTM(hidden, decoder, config.seed, device)
    model.fit(bundle, target, weights, bins, epochs, learning_rate, config.bias_penalty, .001,
              progress=lambda step, loss: print('LSTM epoch', step, 'loss', round(loss, 6), flush=True))
    evaluation = dict(test_status='previously_inspected_follow_up', champion=selection['champion'],
                      split_event_counts={k: len(split[k]) for k in ('train', 'tune', 'calibration', 'test')},
                      settings=settings,
                      training_targets_by_tier={str(t): int(np.count_nonzero(weights[..., i]))
                                                for i, t in enumerate(config.tiers)},
                      partitions={}, paired_bootstrap={}, snapshot=data.provenance)
    excluded, audit = {}, []
    for partition in ('tune', 'calibration', 'test'):
        print('Evaluate LSTM and frozen controls:', partition, flush=True)
        cases, excluded[partition] = build_cases(data, config, split[partition], split['train'])
        controls = controls_at(selection, context(cases, data, config, split['train'], compression),
                               cores, config.tiers)
        lstm = lstm_case_predictions(model, data, cases, config.tiers)
        evaluation['partitions'][partition] = dict(
            lstm=summary(cases, lstm, data), control={n: summary(cases, p, data) for n, p in controls.items()})
        if partition == 'test':
            evaluation['paired_bootstrap'] = paired_bootstrap(
                cases, controls[selection['champion']], lstm, data, config.seed)
        for i, c in enumerate(cases):
            audit.append(dict(partition=partition, event_id=c.event_id, tier=c.tier, checkpoint=iso(c.cutoff),
                              hours_remaining=c.remaining_hours, current=c.current,
                              actual=data.labels.get((c.event_id, c.tier), ''),
                              label_source=data.label_sources.get((c.event_id, c.tier), 'missing'),
                              lstm=float(lstm[i]),
                              **{'control_' + n: float(p[i]) for n, p in controls.items()}))
    write_csv(output / 'predictions.csv', list(audit[0]), audit)
    model.save(output / 'models/lstm.pt')
    save_bank(output, data, split['train'], config.tiers)
    atomic_json(output / 'evaluation.json', evaluation)
    atomic_json(output / 'exclusions.json', excluded)
    files = ['plan.json', 'config.json', 'split.json', 'models/lstm.pt', 'reference_bank.json', 'reference_bank.npz']
    atomic_json(output / 'model.json', dict(
        version=1, kind='raw_lstm', fit_cutoff=split['fit_cutoff'], training_event_ids=split['train'],
        settings=settings, training_targets_by_tier=evaluation['training_targets_by_tier'],
        hashes={f: digest(output / f) for f in files}, limitations=[
            'Previously inspected test block: descriptive follow-up, not a fresh test',
            'Closing tracker proxies are not independently verified official finals',
            'Historical publication and schedule revision timing may be incomplete',
            'Single point-forecast model; no probability intervals are calibrated',
            'Tiers without supervised training targets emit their linear close anchor']))
    if ledger_before is not None and digest(registry) != ledger_before:
        raise ValueError('Holdout ledger changed unexpectedly')
    return evaluation


def predict(live, model, event_id, cutoff, output=None, device=None):
    model = Path(model)
    metadata = json.loads((model / 'model.json').read_text())
    for path, expected in metadata['hashes'].items():
        if digest(model / path) != expected:
            raise ValueError('Model checksum mismatch: ' + path)
    if event_id in metadata['training_event_ids'] or cutoff < metadata['fit_cutoff']:
        raise ValueError('Training event or pre-fit forecast')
    config = Config.load(model / 'config.json')
    if event_id not in live.events:
        raise ValueError('Event absent from input data')
    event = live.known_event(live.events[event_id], cutoff)[0]
    if not event.start <= cutoff < event.end:
        raise ValueError('Checkpoint outside the known event schedule')
    net = RawJointLSTM.load(model / 'models/lstm.pt', device)
    values, details = forecast_event(net, live, event, config.tiers, np.array([float(cutoff)]))
    rows = []
    for j, tier in enumerate(config.tiers):
        if details['present'][0, j]:
            rows.append(dict(tier=tier, status='ok', current=math.ceil(details['current'][0, j]),
                             final=math.ceil(values[0, j]), observed_at=iso(details['observed'][0, j]),
                             score_age_hours=(cutoff - details['observed'][0, j]) / 3600,
                             hours_remaining=float(details['remaining'][0])))
        else:
            rows.append(dict(tier=tier, status='unavailable', reason='No positive score known at this checkpoint'))
    result = dict(event_id=event_id, checkpoint_cn=datetime.fromtimestamp(cutoff, CN).isoformat(),
                  model='raw_lstm', device=str(net.device),
                  predictions=rows, limitations=metadata['limitations'])
    if output:
        if Path(output).exists():
            raise ValueError('Fixed prediction already exists')
        atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--data', required=True)
    fit.add_argument('--source-model', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--registry', default='evaluation_registry.json')
    fit.add_argument('--device', default='auto')
    fit.add_argument('--hidden', type=int, default=32)
    fit.add_argument('--decoder', type=int, default=16)
    fit.add_argument('--epochs', type=int, default=120)
    fit.add_argument('--learning-rate', type=float, default=.005)
    pred = commands.add_parser('predict')
    pred.add_argument('--data', required=True)
    pred.add_argument('--model', required=True)
    pred.add_argument('--event', required=True, type=int)
    pred.add_argument('--as-of', required=True)
    pred.add_argument('--output', required=True)
    pred.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.command == 'train':
        config = Config.load(Path(args.source_model) / 'config.json')
        data = Dataset.load(args.data, config.tiers)
        train(data, args.source_model, args.output, args.registry, args.device,
              args.hidden, args.decoder, args.epochs, args.learning_rate)
    else:
        config = Config.load(Path(args.model) / 'config.json')
        data = Dataset.load(args.data, config.tiers)
        cutoff = timestamp(args.as_of)
        if cutoff is None:
            raise ValueError('Timezone-qualified --as-of required')
        print(json.dumps(predict(data, args.model, args.event, cutoff, args.output, args.device), indent=2))


if __name__ == '__main__':
    main()
